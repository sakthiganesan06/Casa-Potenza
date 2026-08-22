"""
retrieval/bm25_reranker.py — BM25 sparse re-ranker with reciprocal rank fusion.

Architecture:
- One BM25 corpus per (table_name, lang_code) built at startup
- FAISS returns top-20 candidates → BM25 re-scores → RRF fusion → top-3 returned
- Pure Python (rank_bm25), no GPU needed
- Target: <5ms per re-rank on <=20 candidates

Reciprocal Rank Fusion:
  rrf(d) = Σ 1 / (k + rank_i(d))
where rank_i is the rank from FAISS and BM25 separately.
"""
import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Works for Latin, Devanagari, Tamil, Telugu scripts.
    """
    text = text.lower()
    # Split on whitespace and common punctuation
    tokens = re.split(r"[\s\.,;:!?()\[\]{}\"\'\-/\\।॥]+", text)
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25Reranker
# ---------------------------------------------------------------------------

class BM25Reranker:
    """
    BM25 corpus builder and re-ranker.

    At startup: builds a BM25 index per (table_name, lang_code).
    At query time: re-ranks FAISS candidates using BM25 and fuses with RRF.
    """

    def __init__(self):
        # { (table_name, lang_code): BM25Okapi }
        self._corpora: dict[tuple[str, str], Any] = {}
        # { (table_name, lang_code): list[dict] }  — parallel to BM25 corpus
        self._corpus_meta: dict[tuple[str, str], list[dict]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bm25")
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    async def build_from_lancedb(self, vector_store) -> None:
        """Alias for build_from_vector_store to preserve backwards compatibility."""
        await self.build_from_vector_store(vector_store)

    async def build_from_vector_store(self, vector_store) -> None:
        """Build BM25 corpus from vector store collections. Called once at startup."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._build_sync, vector_store)
        self._built = True
        logger.info("[BM25] All corpora built and ready")

    def _build_sync(self, vector_store) -> None:
        """Synchronous BM25 corpus build — runs in thread pool."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("[BM25] rank_bm25 not installed — BM25 disabled. pip install rank_bm25")
            return

        collections = [config.QDRANT_COLLECTION_SEMANTIC, config.QDRANT_COLLECTION_SLIDING]
        t0 = time.perf_counter()

        for table_name in collections:
            try:
                records = vector_store.get_all_records_sync(table_name)
            except Exception as e:
                logger.warning(f"[BM25] Could not load collection '{table_name}': {e}")
                continue

            if not records:
                continue

            from collections import defaultdict
            lang_groups = defaultdict(list)
            for r in records:
                lang_groups[r["lang_code"]].append(r)

            for lang_code, group_records in lang_groups.items():
                texts = [r["text"] for r in group_records]
                tokenized = [_tokenize(t) for t in texts]

                bm25 = BM25Okapi(tokenized)
                meta = group_records

                key = (table_name, str(lang_code))
                self._corpora[key] = bm25
                self._corpus_meta[key] = meta

                logger.info(f"[BM25] Built corpus ({table_name}, {lang_code}): {len(meta)} docs")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[BM25] All corpora built in {elapsed:.0f}ms")


    # ------------------------------------------------------------------
    # Re-rank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        faiss_results: list[dict],
        table_name: str,
        lang_code: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Re-rank FAISS candidates using BM25 + Reciprocal Rank Fusion.

        Args:
            query:         Query text.
            faiss_results: FAISS top-N candidates (already filtered by lang).
            table_name:    Which LanceDB table the results came from.
            lang_code:     Language code for BM25 corpus lookup.
            top_k:         Final number of results to return.
            rrf_k:         RRF k-constant (default 60 per literature).

        Returns:
            Re-ranked list of chunk dicts, top_k items.
        """
        if not self._built or not faiss_results:
            return faiss_results[:top_k]

        key = (table_name, lang_code)
        if key not in self._corpora:
            # No BM25 corpus for this lang — return FAISS results directly
            return faiss_results[:top_k]

        t0 = time.perf_counter()
        bm25 = self._corpora[key]
        query_tokens = _tokenize(query)

        if not query_tokens:
            return faiss_results[:top_k]

        # Get BM25 scores for ALL docs in the corpus
        bm25_scores = bm25.get_scores(query_tokens)

        # --- Reciprocal Rank Fusion ---
        # Map chunk id -> FAISS rank
        faiss_rank_map: dict[str, int] = {
            r.get("id", f"__{i}"): i for i, r in enumerate(faiss_results)
        }

        # For each FAISS candidate, get BM25 rank among corpus
        corpus_meta = self._corpus_meta[key]
        # Build doc_id -> BM25 score mapping
        id_to_bm25: dict[str, float] = {}
        for meta, score in zip(corpus_meta, bm25_scores):
            id_to_bm25[meta["id"]] = float(score)

        # Sort corpus by BM25 to get BM25 ranks
        bm25_sorted_ids = sorted(id_to_bm25.keys(), key=lambda x: id_to_bm25[x], reverse=True)
        bm25_rank_map: dict[str, int] = {cid: rank for rank, cid in enumerate(bm25_sorted_ids)}

        # Compute RRF score for each FAISS candidate
        fused: list[tuple[float, dict]] = []
        for result in faiss_results:
            chunk_id = result.get("id", "")
            faiss_rank = faiss_rank_map.get(chunk_id, len(faiss_results))
            bm25_rank  = bm25_rank_map.get(chunk_id, len(corpus_meta))

            rrf_score = (1 / (rrf_k + faiss_rank)) + config.BM25_WEIGHT * (1 / (rrf_k + bm25_rank))
            fused.append((rrf_score, result))

        fused.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in fused[:top_k]]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[BM25] Re-ranked {len(faiss_results)} → {len(reranked)} in {elapsed:.2f}ms")
        return reranked

    @property
    def is_ready(self) -> bool:
        return self._built


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bm25_reranker: BM25Reranker | None = None


def get_bm25_reranker() -> BM25Reranker:
    """Return the module-level BM25Reranker singleton."""
    global _bm25_reranker
    if _bm25_reranker is None:
        _bm25_reranker = BM25Reranker()
    return _bm25_reranker
