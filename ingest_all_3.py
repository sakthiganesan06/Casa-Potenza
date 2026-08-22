"""
ingest_all_3.py — Stream MSMARCO-XI and ingest English + Tamil + Hindi.
Uses streaming=True to avoid loading the full 4.57GB dataset into memory.
Each language streams and stops after LIMIT rows are collected.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main():
    from datasets import load_dataset
    from loguru import logger
    import config
    from data.vector_store import get_vector_store
    from retrieval.embedder import get_embedder
    from data.chunkers import semantic_parent_child as spc
    from data.chunkers import sliding_window as sw

    LANG_MAP = {
        "en": "eng_Latn",
        "ta": "tam_Taml",
        "hi": "hin_Deva",
    }
    LIMIT_PER_LANG = 3000
    STRATEGIES = ["semantic", "sliding"]
    EMBED_BATCH = 200   # rows per embed call

    embedder = await get_embedder()
    store    = await get_vector_store()

    for lang_code, target_lang in LANG_MAP.items():
        logger.info(f"\n=== [{lang_code}] Streaming target_lang='{target_lang}' (limit={LIMIT_PER_LANG}) ===")

        # ---------------------------------------------------------------
        # Stream the dataset — stops after LIMIT matching rows collected
        # ---------------------------------------------------------------
        t_stream = time.perf_counter()
        logger.info(f"[{lang_code}] Opening streaming dataset...")
        streamed = load_dataset(
            "ai4bharat/MSMARCO-XI",
            "default",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )

        rows = []
        scanned = 0
        for row in streamed:
            scanned += 1
            if row.get("target_lang") == target_lang:
                rows.append(row)
                if len(rows) >= LIMIT_PER_LANG:
                    break
            # Progress log every 5000 rows scanned
            if scanned % 5000 == 0:
                logger.info(f"[{lang_code}] Scanned {scanned:,} rows, collected {len(rows)}/{LIMIT_PER_LANG}")

        t_collected = time.perf_counter() - t_stream
        logger.info(f"[{lang_code}] Collected {len(rows)} rows in {t_collected:.1f}s (scanned {scanned:,})")

        if not rows:
            logger.warning(f"[{lang_code}] No rows found for target_lang='{target_lang}', skipping")
            continue

        # ---------------------------------------------------------------
        # Chunk + embed + upsert for each strategy
        # ---------------------------------------------------------------
        for strategy in STRATEGIES:
            chunker_fn = spc.chunk_dataset_record if strategy == "semantic" else sw.chunk_dataset_record
            table_name = config.LANCEDB_TABLE_SEMANTIC if strategy == "semantic" else config.LANCEDB_TABLE_SLIDING

            logger.info(f"[{lang_code}/{strategy}] Chunking {len(rows)} rows...")
            t_s = time.perf_counter()
            total_chunks = 0

            for batch_start in range(0, len(rows), EMBED_BATCH):
                batch = rows[batch_start: batch_start + EMBED_BATCH]

                # Chunk all rows in this batch
                chunks = []
                for row in batch:
                    chunks.extend(chunker_fn(row))

                if not chunks:
                    continue

                # Embed
                chunks = await embedder.embed_chunks(chunks)

                # Upsert to LanceDB
                upserted = await store.upsert_chunks(table_name, chunks)
                total_chunks += upserted

                done = min(batch_start + EMBED_BATCH, len(rows))
                logger.info(f"  [{lang_code}/{strategy}] {done}/{len(rows)} rows → {total_chunks} chunks so far")

            elapsed = time.perf_counter() - t_s
            logger.info(f"[{lang_code}/{strategy}] ✅ Done: {total_chunks} chunks in {elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Build ANN indexes
    # -------------------------------------------------------------------
    for table in [config.LANCEDB_TABLE_SEMANTIC, config.LANCEDB_TABLE_SLIDING]:
        logger.info(f"Building IVF-PQ index on '{table}'...")
        await store.build_index(table)

    stats = store.get_table_stats()
    logger.info("\n=== LanceDB Table Stats ===")
    for t, c in stats.items():
        logger.info(f"  {t}: {c:,} rows")
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    asyncio.run(main())
