"""
download_and_ingest.py — Ingest English (SQuAD) + Hindi/Tamil/Telugu (MSMARCO-XI)
into LanceDB using both chunking strategies.

Parquets are pre-downloaded to ./data_cache/ via hf_hub_download.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Parquet file paths (pre-downloaded)
# ---------------------------------------------------------------------------
LANG_PARQUETS = {
    "en": Path("./data_cache/plain_text/train-00000-of-00001.parquet"),  # SQuAD
    "hi": Path("./data_cache/validation/hinval.parquet"),                 # MSMARCO-XI
    "ta": Path("./data_cache/validation/tamval.parquet"),                 # MSMARCO-XI
    "te": Path("./data_cache/validation/telval.parquet"),                 # MSMARCO-XI
}

LIMIT_PER_LANG = 150  # 150 records per lang -> ~1,500 chunks per lang -> fast & optimal for CPU


# ---------------------------------------------------------------------------
# Schema normalizers -> unified MSMARCO-XI format for chunkers
# ---------------------------------------------------------------------------

def normalize_msmarco_xi(row: dict, lang_code: str) -> dict:
    """MSMARCO-XI rows already match chunker expectations."""
    return row


def normalize_squad(row: dict, lang_code: str) -> dict:
    """
    Convert SQuAD row to MSMARCO-XI-compatible format.
    SQuAD: {id, title, context, question, answers}
    Target: {query_id, query, passages, Answer, target_lang, source_lang}
    """
    answers = row.get("answers", {})
    answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
    raw_id = row.get("id", "0")
    try:
        numeric_id = int(raw_id, 16) % (2**31)
    except (ValueError, TypeError):
        numeric_id = abs(hash(str(raw_id))) % (2**31)

    return {
        "query_id":    numeric_id,
        "query":       row.get("question", ""),
        "target_lang": "en",
        "source_lang": "en",
        "Answer":      answer_texts[0] if answer_texts else "",
        "Eng_Answer":  answer_texts[0] if answer_texts else "",
        "Eng_Query":   row.get("question", ""),
        "passages": {
            "Translated_passages": [row.get("context", "")],
            "is_selected": [1],
            "url": [f"squad://{row.get('title', 'unknown')}"]
        },
        "meta": {"title": row.get("title", "")},
        "query_type": "description",
    }


NORMALIZERS = {
    "en": normalize_squad,
    "hi": normalize_msmarco_xi,
    "ta": normalize_msmarco_xi,
    "te": normalize_msmarco_xi,
}


def load_rows(ppath: Path, limit: int) -> list[dict]:
    """Read parquet with pyarrow — returns raw row dicts."""
    import pyarrow.parquet as pq
    print(f"  Reading {ppath.name} (limit={limit})...")
    t0 = time.perf_counter()
    table = pq.read_table(str(ppath))
    n = min(limit, len(table))
    sliced = table.slice(0, n)
    print(f"  -> {n}/{len(table)} rows read in {time.perf_counter()-t0:.1f}s")
    return [{col: sliced.column(col)[i].as_py() for col in sliced.schema.names} for i in range(n)]


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

async def main():
    from loguru import logger
    import config
    from data.vector_store import get_vector_store
    from retrieval.embedder import get_embedder
    from data.chunkers import semantic_parent_child as spc
    from data.chunkers import sliding_window as sw

    logger.info("=== Voice RAG — Parquet Ingestion ===")
    logger.info(f"Languages: {list(LANG_PARQUETS.keys())} | Limit: {LIMIT_PER_LANG}/lang")

    # Verify all files exist
    logger.info("\nStep 1: Verifying files...")
    for lang, ppath in LANG_PARQUETS.items():
        if not ppath.exists():
            logger.error(f"  [{lang}] MISSING: {ppath}")
            return
        logger.info(f"  [{lang}] {ppath.name} ({ppath.stat().st_size//1024//1024}MB) OK")

    # Load model + store
    logger.info("\nStep 2: Loading BGE-m3 + LanceDB...")
    embedder = await get_embedder()
    store    = await get_vector_store()

    EMBED_BATCH = 50

    # Per-language ingest
    for lang_code, ppath in LANG_PARQUETS.items():
        logger.info(f"\n=== [{lang_code.upper()}] Ingesting {LIMIT_PER_LANG} rows ===")
        raw_rows = load_rows(ppath, LIMIT_PER_LANG)
        normalizer = NORMALIZERS[lang_code]
        rows = [normalizer(r, lang_code) for r in raw_rows]
        logger.info(f"  Normalized {len(rows)} rows")

        for strategy in ["semantic", "sliding"]:
            chunker_fn = spc.chunk_dataset_record if strategy == "semantic" else sw.chunk_dataset_record
            table_name = (config.LANCEDB_TABLE_SEMANTIC if strategy == "semantic"
                          else config.LANCEDB_TABLE_SLIDING)
            total_chunks = 0
            t_s = time.perf_counter()

            for i in range(0, len(rows), EMBED_BATCH):
                batch = rows[i: i + EMBED_BATCH]
                chunks = []
                for row in batch:
                    try:
                        chunks.extend(chunker_fn(row))
                    except Exception as e:
                        pass  # skip bad rows silently
                if not chunks:
                    continue
                chunks = await embedder.embed_chunks(chunks)
                upserted = await store.upsert_chunks(table_name, chunks)
                total_chunks += upserted
                logger.info(f"  [{lang_code}/{strategy}] {min(i+EMBED_BATCH, len(rows))}/{len(rows)} rows "
                             f"-> {total_chunks} chunks")

            logger.info(f"  [{lang_code}/{strategy}] DONE: {total_chunks} chunks in "
                        f"{time.perf_counter()-t_s:.1f}s")

    # Build indexes
    logger.info("\nStep 3: Building IVF-PQ indexes...")
    for table in [config.LANCEDB_TABLE_SEMANTIC, config.LANCEDB_TABLE_SLIDING]:
        logger.info(f"  Indexing '{table}'...")
        await store.build_index(table)

    stats = store.get_table_stats()
    logger.info("\n=== Final LanceDB Stats ===")
    for t, c in stats.items():
        logger.info(f"  {t}: {c:,} rows")
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    asyncio.run(main())
