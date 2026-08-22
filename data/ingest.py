"""
data/ingest.py — Dataset ingestion, chunking, and LanceDB indexing pipeline.

Workflow:
1. Download AI4Bharat/MSMARCO-XI for configured language subsets
2. Apply both chunking strategies (semantic + sliding) in parallel
3. Embed all chunks with BGE-m3
4. Upsert into LanceDB tables
5. Build IVF-PQ ANN indexes

Usage:
    python data/ingest.py                         # ingest all configured languages
    python data/ingest.py --lang hi --limit 5000  # quick smoke test
    python data/ingest.py --strategy semantic      # only build semantic table
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.chunkers import semantic_parent_child, sliding_window
from data.vector_store import get_vector_store
from retrieval.embedder import get_embedder


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------

async def ingest_language(
    lang: str,
    limit: int | None = None,
    strategies: list[str] | None = None,
) -> dict:
    """
    Ingest one language subset of MSMARCO-XI into LanceDB.

    Args:
        lang:       Language code (e.g. "hi", "ta", "bn").
        limit:      Max records to process (None = all).
        strategies: List of chunking strategies. Defaults to both.

    Returns:
        Stats dict with chunk counts per strategy.
    """
    strategies = strategies or ["semantic", "sliding"]
    logger.info(f"[Ingest] Starting ingestion for language: '{lang}'")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    # MSMARCO-XI has only a 'default' config; filter rows by target_lang field.
    # target_lang values: hin_Deva, tam_Taml, ben_Beng, etc.
    LANG_TO_TARGET: dict[str, str] = {
        "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "bn": "ben_Beng",
        "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva",
        "te": "tel_Telu", "gu": "guj_Gujr", "ur": "urd_Arab",
    }
    target_lang_val = LANG_TO_TARGET.get(lang)
    if not target_lang_val:
        logger.error(f"[Ingest] No target_lang mapping for lang='{lang}'")
        return {"lang": lang, "error": "no target_lang mapping"}

    logger.info(f"[Ingest] Loading {config.DATASET_NAME} (default config, filtering target_lang={target_lang_val})...")
    t_load = time.perf_counter()
    try:
        full_dataset = load_dataset(
            config.DATASET_NAME,
            "default",
            split="train",
            trust_remote_code=False,
        )
        dataset = full_dataset.filter(lambda row: row.get("target_lang") == target_lang_val)
    except Exception as exc:
        logger.error(f"[Ingest] Failed to load dataset for lang='{lang}': {exc}")
        return {"lang": lang, "error": str(exc)}

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    n_records = len(dataset)
    elapsed_load = (time.perf_counter() - t_load) * 1000
    logger.info(f"[Ingest] Loaded {n_records} records in {elapsed_load:.0f}ms")

    # ------------------------------------------------------------------
    # 2. Get embedder and vector store
    # ------------------------------------------------------------------
    embedder = await get_embedder()
    store    = await get_vector_store()

    stats = {"lang": lang, "n_records": n_records}

    # ------------------------------------------------------------------
    # 3. Chunk and index for each strategy
    # ------------------------------------------------------------------
    for strategy in strategies:
        if strategy == "semantic":
            chunker_fn = semantic_parent_child.chunk_dataset_record
            table_name = config.QDRANT_COLLECTION_SEMANTIC
        else:
            chunker_fn = sliding_window.chunk_dataset_record
            table_name = config.QDRANT_COLLECTION_SLIDING

        logger.info(f"[Ingest:{strategy}] Processing {n_records} records...")
        t_strategy = time.perf_counter()

        BATCH_SIZE = 500  # records per embedding batch
        total_chunks = 0

        for batch_start in tqdm(
            range(0, n_records, BATCH_SIZE),
            desc=f"  {lang}/{strategy}",
            unit="batch",
        ):
            batch_end = min(batch_start + BATCH_SIZE, n_records)
            batch_records = dataset[batch_start:batch_end]

            # Convert HuggingFace batch dict to list of row dicts
            rows = _batch_to_rows(batch_records, batch_end - batch_start)

            # Chunk all records in this batch
            all_chunks: list[dict] = []
            for row in rows:
                all_chunks.extend(chunker_fn(row))

            if not all_chunks:
                continue

            # Embed all chunks (batched within embed_chunks)
            all_chunks = await embedder.embed_chunks(all_chunks)

            # Upsert to LanceDB
            n_upserted = await store.upsert_chunks(table_name, all_chunks)
            total_chunks += n_upserted

        elapsed_strategy = (time.perf_counter() - t_strategy)
        logger.info(
            f"[Ingest:{strategy}] Done — {total_chunks} chunks in {elapsed_strategy:.1f}s "
            f"({total_chunks / elapsed_strategy:.0f} chunks/sec)"
        )
        stats[f"{strategy}_chunks"] = total_chunks

    return stats


def _batch_to_rows(batch_dict: dict, n: int) -> list[dict]:
    """Convert HuggingFace columnar batch dict to list of row dicts."""
    rows = []
    for i in range(n):
        row = {}
        for key, values in batch_dict.items():
            if isinstance(values, list) and len(values) > i:
                row[key] = values[i]
            else:
                row[key] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Full ingestion (all configured languages)
# ---------------------------------------------------------------------------

async def ingest_all(
    languages: list[str] | None = None,
    limit: int | None = None,
    strategies: list[str] | None = None,
) -> None:
    """
    Ingest all specified language subsets and build ANN indexes.

    Languages are ingested SEQUENTIALLY to avoid OOM from parallel BGE-m3 calls.
    """
    languages = languages or config.INGEST_LANGUAGES
    strategies = strategies or ["semantic", "sliding"]

    logger.info(f"=== Starting full ingestion: {languages} ===")
    t_total = time.perf_counter()

    all_stats = []
    for lang in languages:
        stats = await ingest_language(lang, limit=limit, strategies=strategies)
        all_stats.append(stats)

    # Build ANN indexes after all data is loaded
    logger.info("[Ingest] Building ANN indexes on all collections...")
    store = await get_vector_store()
    if "semantic" in strategies:
        await store.build_index(config.QDRANT_COLLECTION_SEMANTIC)
    if "sliding" in strategies:
        await store.build_index(config.QDRANT_COLLECTION_SLIDING)

    total_elapsed = time.perf_counter() - t_total
    logger.info(f"=== Ingestion complete in {total_elapsed:.1f}s ===")

    # Print summary
    logger.info("\n=== Ingestion Summary ===")
    for s in all_stats:
        lang = s.get("lang", "?")
        logger.info(
            f"  {lang}: {s.get('n_records', 0)} records | "
            f"semantic={s.get('semantic_chunks', 0)} chunks | "
            f"sliding={s.get('sliding_chunks', 0)} chunks"
        )

    # Print vector store stats
    store_stats = store.get_table_stats()
    logger.info(f"\n=== Qdrant Collection Stats ===")
    for table, count in store_stats.items():
        logger.info(f"  {table}: {count:,} points")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def _main():
    parser = argparse.ArgumentParser(description="MSMARCO-XI dataset ingestion pipeline")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Specific language to ingest (e.g. 'hi'). Default: all configured languages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records per language (for testing). Default: all.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["semantic", "sliding", "both"],
        default="both",
        help="Chunking strategy to build. Default: both.",
    )
    args = parser.parse_args()

    languages = [args.lang] if args.lang else None
    strategies = ["semantic", "sliding"] if args.strategy == "both" else [args.strategy]

    await ingest_all(languages=languages, limit=args.limit, strategies=strategies)


if __name__ == "__main__":
    asyncio.run(_main())
