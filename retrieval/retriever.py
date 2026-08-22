"""
retrieval/retriever.py — Hybrid retrieval: FAISS ANN + BM25 re-rank.

Pipeline per query:
  1. Embed query (BGE-m3, ~200ms cold / <1ms cached)
  2. FAISS ANN search top-20 (1–5ms in-process)
  3. BM25 re-rank top-20 → top-3 via Reciprocal Rank Fusion (2–5ms)
  4. Expand children to parents for semantic strategy (parallel async fetch)

Falls back to LanceDB ANN if FAISS index is not built yet.

Target latency: <10ms (excluding query embedding).
"""
import asyncio
import time
from typing import Literal
import numpy as np

from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config
from retrieval.embedder import BGEEmbedder, get_embedder
from data.vector_store import VectorStore, get_vector_store


# ---------------------------------------------------------------------------
# Adaptive strategy selector
# ---------------------------------------------------------------------------

# Interrogative words that suggest a descriptive/explanatory query → semantic
_SEMANTIC_KEYWORDS = {
    # English
    "what", "how", "why", "explain", "describe", "tell", "definition",
    "history", "impact", "effect", "cause", "relationship",
    # Hindi
    "क्या", "कैसे", "क्यों", "बताओ", "समझाओ", "परिभाषा",
    # Tamil
    "என்ன", "எப்படி", "ஏன்", "விளக்கு",
    # Telugu
    "ఏమి", "ఎలా", "ఎందుకు", "వివరించు",
}


def pick_strategy(query: str) -> str:
    """
    Adaptively pick chunking strategy based on query characteristics.

    Rules:
    - Short queries (≤4 words) → 'sliding' (narrow, fast)
    - Queries with interrogative/explanatory words → 'semantic' (richer context)
    - Default: config.CHUNKING_STRATEGY
    """
    tokens = query.strip().lower().split()
    if len(tokens) <= 4:
        return "sliding"
    for tok in tokens[:6]:  # only check first 6 words
        if tok in _SEMANTIC_KEYWORDS:
            return "semantic"
    return config.CHUNKING_STRATEGY


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Hybrid retrieval: Qdrant dense vector ANN + BM25 Reciprocal Rank Fusion.
    """

    def __init__(self):
        self._embedder: BGEEmbedder | None = None
        self._store: VectorStore | None = None
        self.last_query_embedding = None

    async def initialize(self) -> None:
        """Load embedder, connect to vector store, and build BM25 indexes from Qdrant."""
        self._embedder = await get_embedder()
        self._store = await get_vector_store()

        if config.BM25_ENABLED:
            from retrieval.bm25_reranker import get_bm25_reranker
            bm25 = get_bm25_reranker()
            if not bm25.is_ready:
                logger.info("[Retriever] Building BM25 corpus...")
                await bm25.build_from_vector_store(self._store)

        logger.info("[Retriever] Initialized: Qdrant + BM25 + RRF")

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        lang_code: str,
        strategy: Literal["semantic", "sliding"] | None = None,
        top_k: int = config.TOP_K_RETRIEVAL,
        query_vector: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Strategy is auto-selected via pick_strategy() if not explicitly given.
        Uses Qdrant dense vector search -> BM25 RRF re-ranking.

        Args:
            query:        Query text.
            lang_code:    Short language code (e.g. "hi", "ta").
            strategy:     "semantic" or "sliding". Auto-selected if None.
            top_k:        Final number of results.
            query_vector: Pre-computed embedding (skips embed step).

        Returns:
            List of chunk dicts with text, score, doc_id.
        """
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = pick_strategy(query)
            logger.debug(f"[Retriever] Auto-selected strategy: '{strategy}' for query: '{query[:40]}'")

        collection_name = (
            config.QDRANT_COLLECTION_SEMANTIC
            if strategy == "semantic"
            else config.QDRANT_COLLECTION_SLIDING
        )

        # ------------------------------------------------------------------
        # Step 1: Embed the query (or reuse pre-computed embedding)
        # ------------------------------------------------------------------
        if query_vector is not None:
            query_vec = query_vector
        else:
            t_embed_start = time.perf_counter()
            query_vec = await self._embedder.embed_one(query)
            embed_ms = (time.perf_counter() - t_embed_start) * 1000
            logger.debug(f"[Retriever] Query embedded in {embed_ms:.1f}ms")

        self.last_query_embedding = query_vec  # for guardrail access

        # ------------------------------------------------------------------
        # Step 2: Qdrant dense ANN search
        # ------------------------------------------------------------------
        t_search_start = time.perf_counter()

        dense_results = await self._store.search(
            collection_name=collection_name,
            query_vector=query_vec.tolist(),
            lang_code=lang_code,
            top_k=config.TOP_K_DENSE_CANDIDATES,
        )

        # ------------------------------------------------------------------
        # Step 3: BM25 re-rank (RRF)
        # ------------------------------------------------------------------
        if dense_results and config.BM25_ENABLED:
            from retrieval.bm25_reranker import get_bm25_reranker
            bm25 = get_bm25_reranker()
            results = bm25.rerank(
                query=query,
                faiss_results=dense_results,
                table_name=collection_name,
                lang_code=lang_code,
                top_k=top_k,
            )
        else:
            results = dense_results[:top_k]

        t_search_done = time.perf_counter()
        search_ms = (t_search_done - t_search_start) * 1000
        logger.debug(
            f"[Retriever] Qdrant+BM25 ({strategy}) returned {len(results)} results "
            f"in {search_ms:.2f}ms"
        )

        # ------------------------------------------------------------------
        # Step 4 (Semantic only): Expand children → parents
        # ------------------------------------------------------------------
        if strategy == "semantic" and results:
            results = await self._expand_to_parents(results, collection_name)

        return results

    async def _expand_to_parents(
        self, child_results: list[dict], collection_name: str
    ) -> list[dict]:
        """
        For child chunks, replace text with parent passage text.
        Fetches parents in parallel. Returns deduped list.
        """
        seen_parent_ids: set[str] = set()
        expanded: list[dict] = []
        parent_fetch_tasks = []
        non_child_results = []

        for result in child_results:
            chunk_type = result.get("chunk_type", "")
            parent_id  = result.get("parent_id", "")

            if chunk_type == "child" and parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_fetch_tasks.append(
                    self._store.fetch_parent(collection_name, parent_id)
                )
                non_child_results.append(result)
            elif chunk_type != "child":
                expanded.append(result)

        if parent_fetch_tasks:
            parents = await asyncio.gather(*parent_fetch_tasks, return_exceptions=True)
            for child_result, parent in zip(non_child_results, parents):
                if isinstance(parent, Exception) or parent is None:
                    expanded.append(child_result)
                else:
                    expanded.append({
                        **child_result,
                        "text":   parent["text"],
                        "id":     parent["id"],
                        "doc_id": parent.get("doc_id", child_result["doc_id"]),
                    })

        # Sort by score, deduplicate by doc_id
        seen_doc_ids: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(expanded, key=lambda x: x.get("score", 0), reverse=True):
            doc_id = r.get("doc_id", "")
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                deduped.append(r)

        return deduped[:config.TOP_K_RETRIEVAL]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_retriever: Retriever | None = None


async def get_retriever() -> Retriever:
    """Return the module-level singleton Retriever, initializing if needed."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
        await _retriever.initialize()
    return _retriever
