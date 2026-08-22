"""
eval/query_generator.py — Load evaluation queries directly from local parquet files.

Stratified across English, Hindi, Tamil, Telugu.
Queries are loaded as text (no audio) — the eval harness uses run_text_pipeline().
Returns EvalQuery objects with the gold doc_ids for recall@k computation.
"""
import asyncio
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import pyarrow.parquet as pq

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# EvalQuery dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with metadata for recall@5 computation."""
    query_id:       str
    text:           str                # Query in target language
    eng_text:       str                # Original English query
    lang_code:      str                # e.g. "en", "hi", "ta", "te"
    query_type:     str                # DESCRIPTION, ENTITY, etc.
    gold_doc_ids:   set[str]           # doc_ids of is_selected=1 passages
    n_passages:     int                # total passages in this query
    target_lang:    str


# ---------------------------------------------------------------------------
# Local Parquet Mapping
# ---------------------------------------------------------------------------

LANG_PARQUETS = {
    "en": Path("./data_cache/plain_text/train-00000-of-00001.parquet"),
    "hi": Path("./data_cache/validation/hinval.parquet"),
    "ta": Path("./data_cache/validation/tamval.parquet"),
    "te": Path("./data_cache/validation/telval.parquet"),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

async def load_eval_queries(
    n_total: int = config.EVAL_NUM_QUERIES,
    languages: list[str] | None = None,
    split: str = config.EVAL_SPLIT,
    seed: int = 42,
) -> list[EvalQuery]:
    """
    Load n_total evaluation queries from local parquet files.
    Stratified equally across configured languages (en, hi, ta, te).
    """
    languages = languages or config.EVAL_LANGUAGES
    n_per_lang = max(1, n_total // len(languages))
    remainder  = n_total - n_per_lang * len(languages)

    rng = random.Random(seed)
    all_queries: list[EvalQuery] = []

    for i, lang in enumerate(languages):
        quota = n_per_lang + (remainder if i == 0 else 0)
        queries = await _load_language_queries(lang, quota, rng)
        all_queries.extend(queries)
        logger.info(f"[EvalGen] Loaded {len(queries)} queries for lang='{lang}'")

    rng.shuffle(all_queries)
    logger.info(f"[EvalGen] Total eval queries: {len(all_queries)}")
    return all_queries


async def _load_language_queries(
    lang: str,
    quota: int,
    rng: random.Random,
) -> list[EvalQuery]:
    """Load up to `quota` queries for a single language."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _load_language_queries_sync, lang, quota, rng
    )


def _load_language_queries_sync(
    lang: str,
    quota: int,
    rng: random.Random,
) -> list[EvalQuery]:
    """Synchronous parquet reader for fast query loading."""
    ppath = LANG_PARQUETS.get(lang)
    if not ppath or not ppath.exists():
        logger.error(f"[EvalGen] Parquet for lang='{lang}' not found at {ppath}")
        return []

    try:
        table = pq.read_table(str(ppath))
        n_avail = min(500, len(table))  # pick from ingested subset
        sliced = table.slice(0, n_avail)
        indices = list(range(n_avail))
        rng.shuffle(indices)
        selected_idx = indices[:min(quota, n_avail)]

        queries: list[EvalQuery] = []
        for idx in selected_idx:
            row = {col: sliced.column(col)[idx].as_py() for col in sliced.schema.names}
            queries.append(_row_to_eval_query(row, lang))
        return queries
    except Exception as exc:
        logger.error(f"[EvalGen] Failed to read {ppath}: {exc}")
        return []


def _row_to_eval_query(row: dict, lang: str) -> EvalQuery:
    """Convert a row to an EvalQuery object."""
    if lang == "en":
        raw_id = row.get("id", "0")
        try:
            numeric_id = int(raw_id, 16) % (2**31)
        except (ValueError, TypeError):
            numeric_id = abs(hash(str(raw_id))) % (2**31)

        q_text = row.get("question", "")
        return EvalQuery(
            query_id=str(numeric_id),
            text=q_text,
            eng_text=q_text,
            lang_code="en",
            query_type="description",
            gold_doc_ids={f"q{numeric_id}_p0"},
            n_passages=1,
            target_lang="en",
        )

    # MSMARCO-XI Indic format
    query_id    = str(row.get("query_id", -1))
    query_text  = row.get("query", "") or ""
    eng_text    = row.get("Eng_Query", "") or ""
    query_type  = row.get("query_type", "UNKNOWN") or "UNKNOWN"
    target_lang = row.get("target_lang", lang) or ""

    passages       = row.get("passages", {}) or {}
    is_selected_l  = passages.get("is_selected", []) or []
    n_passages     = len(is_selected_l)

    gold_doc_ids: set[str] = set()
    for p_idx, sel in enumerate(is_selected_l):
        if int(sel) == 1:
            gold_doc_ids.add(f"q{query_id}_p{p_idx}")

    return EvalQuery(
        query_id=query_id,
        text=query_text,
        eng_text=eng_text,
        lang_code=lang,
        query_type=query_type,
        gold_doc_ids=gold_doc_ids,
        n_passages=n_passages,
        target_lang=target_lang,
    )
