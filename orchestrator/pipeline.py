"""
orchestrator/pipeline.py — Main async pipeline stitching all modules.

Pipeline stages per query:
  1. STT: Sarvam WebSocket → final transcript (t0: Audio_End, t1: Text_Ready)
  2. Guardrail + Embed: Parallel — safety check + query embedding (<5ms)
  3. Retrieve: LanceDB ANN search with lang_code filter (t2: Context_Retrieved)
  4. LLM: Groq streaming JSON generation (t3: First_LLM_Token)
  5. Log: LatencyTracker records all 4 milestones

Concurrency:
- asyncio.Semaphore caps parallel pipeline runs
- Guardrail check and embedding run concurrently
- LanceDB search runs in thread pool (non-blocking)
"""
import asyncio
import time
import uuid
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config
from retrieval.embedder import get_embedder
from retrieval.retriever import get_retriever, pick_strategy
from llm.guardrails import get_router
from llm.groq_client import get_groq_client
from llm.prompts import build_guardrail_refusal, build_context_empty_refusal
from orchestrator.latency_tracker import get_tracker, LatencyRecord
import numpy as np

# ---------------------------------------------------------------------------
# 2-Tier In-Memory High Speed Response Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Ultra-Fast 2-Tier In-Memory Cache for sub-5ms voice responses.
    Tier 1: Normalized Exact String Hash Lookup (<0.01ms)
    Tier 2: High-Cosine Semantic Similarity Match (<15ms)
    """
    def __init__(self, max_items: int = 5000, semantic_threshold: float = 0.94):
        self._exact_cache: dict[str, dict] = {}
        self._semantic_keys: list[str] = []
        self._semantic_vectors: list[np.ndarray] = []
        self._semantic_responses: list[dict] = []
        self.max_items = max_items
        self.threshold = semantic_threshold

    def _clean_lang(self, lang: str) -> str:
        return (lang or "auto").split("-")[0].split("_")[0].lower().strip()

    def _normalize(self, query: str, lang: str) -> str:
        return f"{self._clean_lang(lang)}:{query.strip().lower()}"

    def get_exact(self, query: str, lang: str) -> dict | None:
        clean_lang = self._clean_lang(lang)
        keys_to_try = [
            f"{clean_lang}:{query.strip().lower()}",
            f"auto:{query.strip().lower()}",
            f"ta:{query.strip().lower()}",
            f"hi:{query.strip().lower()}",
            f"en:{query.strip().lower()}",
            f"te:{query.strip().lower()}",
        ]
        for key in keys_to_try:
            if key in self._exact_cache:
                res = self._exact_cache[key].copy()
                res["cached"] = True
                return res
        return None

    def get_semantic(self, query_vec: np.ndarray, lang: str) -> dict | None:
        if not self._semantic_vectors:
            return None

        # Normalized cosine similarity against all cached vectors
        matrix = np.stack(self._semantic_vectors)  # shape (N, 1024)
        norm_q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        norm_m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        sims = np.dot(norm_m, norm_q)

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.threshold:
            cached_lang = self._clean_lang(self._semantic_keys[best_idx].split(":")[0])
            req_lang = self._clean_lang(lang)
            if req_lang == "auto" or cached_lang == req_lang:
                logger.info(f"[Cache] Semantic HIT (score={best_score:.3f}) for cached query: '{self._semantic_keys[best_idx]}'")
                res = self._semantic_responses[best_idx].copy()
                res["cached"] = True
                res["cache_similarity"] = round(best_score, 3)
                return res
        return None


    def put(self, query: str, lang: str, query_vec: np.ndarray | None, response: dict) -> None:
        if not response or response.get("refused") or not response.get("answer"):
            return

        key = self._normalize(query, lang)
        self._exact_cache[key] = response

        if query_vec is not None:
            if len(self._semantic_vectors) >= self.max_items:
                self._semantic_vectors.pop(0)
                self._semantic_keys.pop(0)
                self._semantic_responses.pop(0)

            self._semantic_keys.append(key)
            self._semantic_vectors.append(query_vec)
            self._semantic_responses.append(response)


_response_cache = ResponseCache()

def get_response_cache() -> ResponseCache:
    return _response_cache

# ---------------------------------------------------------------------------
# Concurrency limiter
# ---------------------------------------------------------------------------

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_PIPELINE)
    return _semaphore


# ---------------------------------------------------------------------------
# Pipeline initialization (call once at startup)
# ---------------------------------------------------------------------------

_initialized = False


async def initialize_pipeline() -> None:
    """
    warm up all singleton components.

    Call this once before processing any queries:
    - Loads BGE-m3 model into memory
    - Connects to Qdrant
    - Pre-computes semantic router centroid embeddings
    - Warms up Groq client connection pool

    After this, query processing begins with sub-millisecond overhead.
    """
    global _initialized
    if _initialized:
        return

    logger.info("=== Initializing Voice RAG Pipeline ===")
    t0 = time.perf_counter()

    # 1. Load embedder (heaviest — BGE-m3 ~2.3GB)
    embedder = await get_embedder()
    logger.info("✅ Embedder loaded")

    # 2. Connect vector store
    retriever = await get_retriever()
    logger.info("✅ Vector store + Qdrant + BM25 indexes built")

    # 3. Initialize semantic router with embedder
    router = await get_router(embedder=embedder)
    logger.info("✅ Semantic router initialized")

    # 4. Initialize Groq client (just instantiation, no network call yet)
    get_groq_client()
    logger.info("✅ Groq client ready")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"=== Pipeline ready in {elapsed:.0f}ms ===")
    _initialized = True


# ---------------------------------------------------------------------------
# Core pipeline run (one query)
# ---------------------------------------------------------------------------

async def run_pipeline(
    transcript: str,
    lang_code: str,
    t0_audio_end: float,
    t1_text_ready: float,
    query_id: str | None = None,
    chunking_strategy: str | None = None,
    is_selected_doc_ids: set[str] | None = None,
) -> dict:
    """
    Execute the full retrieval-generation pipeline for a transcribed query.

    This function represents the "hot path" — it must complete in <150ms
    (the remaining budget after STT has consumed ~50ms).

    Args:
        transcript:          The transcribed query text.
        lang_code:           Short language code (e.g. "hi", "ta", "bn").
        t0_audio_end:        perf_counter timestamp when audio ended.
        t1_text_ready:       perf_counter timestamp when final transcript arrived.
        query_id:            Optional identifier for logging (auto-generated if None).
        chunking_strategy:   "semantic" or "sliding" (defaults to config value).
        is_selected_doc_ids: Set of gold doc_ids for recall@5 computation in eval.

    Returns:
        dict with keys: answer, sources, confidence, language, refused,
                        refusal_reason, latency (sub-dict), query_id
    """
    if not _initialized:
        await initialize_pipeline()

    qid = query_id or str(uuid.uuid4())[:8]

    async with _get_semaphore():
        return await _run_pipeline_inner(
            transcript=transcript,
            lang_code=lang_code,
            t0_audio_end=t0_audio_end,
            t1_text_ready=t1_text_ready,
            query_id=qid,
            # Auto-select strategy if not explicitly given
            chunking_strategy=chunking_strategy or pick_strategy(transcript),
            is_selected_doc_ids=is_selected_doc_ids or set(),
        )


async def _run_pipeline_inner(
    transcript: str,
    lang_code: str,
    t0_audio_end: float,
    t1_text_ready: float,
    query_id: str,
    chunking_strategy: str,
    is_selected_doc_ids: set[str],
) -> dict:
    """Internal (unsynchronized) pipeline implementation."""

    embedder  = await get_embedder()
    retriever = await get_retriever()
    router    = await get_router()
    groq      = get_groq_client()
    tracker   = get_tracker()

    short_lang = lang_code.split("-")[0].split("_")[0].lower().strip() if lang_code else "en"
    bcp47_lang = config.LANG_TO_BCP47.get(short_lang, config.LANG_TO_BCP47.get(lang_code, "en-IN"))

    # ------------------------------------------------------------------
    # Stage 1.5: Tier 1 Exact Response Cache Lookup (<0.01ms)
    # ------------------------------------------------------------------
    exact_hit = _response_cache.get_exact(transcript, short_lang)
    if exact_hit:
        t_now = time.perf_counter()
        logger.info(f"[Pipeline:{query_id}] Exact Cache HIT for '{transcript[:40]}' (0.01ms)")
        record = tracker.build_record(
            query_id=query_id, lang_code=lang_code, query_text=transcript,
            t0_audio_end=t0_audio_end, t1_text_ready=t1_text_ready,
            t2_context_ready=t_now, t3_first_token=t_now,
            guardrail_rejected=False, llm_refused=False,
        )
        await tracker.log(record)
        return {**exact_hit, "query_id": query_id, "latency": _latency_dict(record)}

    # ------------------------------------------------------------------
    # Stage 2: Guardrail + Embedding (parallel)
    # Embed the query and check safety simultaneously
    # ------------------------------------------------------------------
    async def embed_query():
        return await embedder.embed_one(transcript)

    async def check_guardrail(query_vec):
        return await router.check(query_vec)

    # First embed, then check guardrail with the embedding
    query_vec = await embed_query()

    # ------------------------------------------------------------------
    # Stage 2.5: Tier 2 Semantic Response Cache Lookup (<15ms)
    # ------------------------------------------------------------------
    semantic_hit = _response_cache.get_semantic(query_vec, short_lang)
    if semantic_hit:
        t_now = time.perf_counter()
        logger.info(f"[Pipeline:{query_id}] Semantic Cache HIT for '{transcript[:40]}' (sim={semantic_hit.get('cache_similarity')})")
        record = tracker.build_record(
            query_id=query_id, lang_code=lang_code, query_text=transcript,
            t0_audio_end=t0_audio_end, t1_text_ready=t1_text_ready,
            t2_context_ready=t_now, t3_first_token=t_now,
            guardrail_rejected=False, llm_refused=False,
        )
        await tracker.log(record)
        return {**semantic_hit, "query_id": query_id, "latency": _latency_dict(record)}

    is_safe, topic_label, guard_score = await check_guardrail(query_vec)

    if not is_safe and topic_label in getattr(config, "GUARDRAIL_REJECT_LABELS", {"unsafe_harmful"}):
        t_now = time.perf_counter()
        logger.warning(
            f"[Pipeline:{query_id}] Guardrail REJECTED — "
            f"topic={topic_label}, score={guard_score:.3f}"
        )
        refusal = build_guardrail_refusal(
            reason="unsafe",
            lang=bcp47_lang,
        )
        # Record latency (guardrail rejection path)
        record = tracker.build_record(
            query_id=query_id, lang_code=lang_code,
            query_text=transcript,
            t0_audio_end=t0_audio_end,  t1_text_ready=t1_text_ready,
            t2_context_ready=t_now, t3_first_token=t_now,
            guardrail_rejected=True, llm_refused=False,
        )
        await tracker.log(record)
        return {**refusal, "query_id": query_id, "latency": _latency_dict(record)}

    # ------------------------------------------------------------------
    # Stage 3: Hybrid Retrieval (FAISS+BM25 or LanceDB fallback)
    # ------------------------------------------------------------------
    t_retrieve_start = time.perf_counter()
    chunks = await retriever.retrieve(
        query=transcript,
        lang_code=short_lang,
        strategy=chunking_strategy,
        top_k=config.TOP_K_RETRIEVAL,
        query_vector=query_vec,
    )
    t2_context_ready = time.perf_counter()

    # Compute retrieval recall@k (for eval harness)
    recall = 0.0
    if is_selected_doc_ids and chunks:
        retrieved_ids = {c.get("doc_id", "") for c in chunks}
        recall = 1.0 if retrieved_ids & is_selected_doc_ids else 0.0

    logger.debug(
        f"[Pipeline:{query_id}] Retrieved {len(chunks)} chunks in "
        f"{(t2_context_ready - t_retrieve_start)*1000:.2f}ms | recall={recall}"
    )

    # ------------------------------------------------------------------
    # Stage 3.5: Grounding Score Check
    # If best chunk cosine < threshold — skip LLM, return safe refusal
    # Saves ~800ms Groq call when retrieval confidence is low
    # ------------------------------------------------------------------
    if chunks:
        best_score = max(c.get("score", 0.0) for c in chunks)
    else:
        best_score = 0.0

    if best_score < config.GROUNDING_SCORE_THRESHOLD:
        t_now = time.perf_counter()
        logger.info(
            f"[Pipeline:{query_id}] Grounding score too low ({best_score:.3f} < "
            f"{config.GROUNDING_SCORE_THRESHOLD}) — safe refusal without LLM call"
        )
        refusal = build_context_empty_refusal(bcp47_lang)
        record = tracker.build_record(
            query_id=query_id, lang_code=lang_code, query_text=transcript,
            t0_audio_end=t0_audio_end, t1_text_ready=t1_text_ready,
            t2_context_ready=t_now, t3_first_token=t_now,
            guardrail_rejected=False, llm_refused=True,
        )
        await tracker.log(record)
        return {**refusal, "query_id": query_id, "latency": _latency_dict(record)}

    # ------------------------------------------------------------------
    # Stage 4: LLM Generation
    # ------------------------------------------------------------------
    response_dict, t3_first_token = await groq.generate(
        query=transcript,
        context_chunks=chunks,
        lang_code=bcp47_lang,
    )

    # Save to 2-tier response cache for future instant responses
    if not response_dict.get("refused") and response_dict.get("answer"):
        _response_cache.put(transcript, short_lang, query_vec, response_dict)

    # ------------------------------------------------------------------
    # Stage 5: Log latency milestones
    # ------------------------------------------------------------------
    record = tracker.build_record(
        query_id=query_id,
        lang_code=lang_code,
        query_text=transcript,
        t0_audio_end=t0_audio_end,
        t1_text_ready=t1_text_ready,
        t2_context_ready=t2_context_ready,
        t3_first_token=t3_first_token,
        guardrail_rejected=False,
        llm_refused=response_dict.get("refused", False),
        retrieval_recall=recall,
        chunking_strategy=chunking_strategy,
    )
    await tracker.log(record)

    return {
        **response_dict,
        "query_id": query_id,
        "latency": _latency_dict(record),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latency_dict(record: LatencyRecord) -> dict:
    return {
        "stt_ms":        round(record.stt_latency_ms, 2),
        "retrieval_ms":  round(record.retrieval_latency_ms, 2),
        "llm_ttft_ms":   round(record.llm_ttft_ms, 2),
        "total_ms":      round(record.total_latency_ms, 2),
        "within_budget": record.within_budget,
    }


# ---------------------------------------------------------------------------
# High-level pipeline for text-only input (bypasses STT)
# Used by eval harness
# ---------------------------------------------------------------------------

async def run_text_pipeline(
    transcript: str,
    lang_code: str,
    query_id: str | None = None,
    chunking_strategy: str | None = None,
    is_selected_doc_ids: set[str] | None = None,
) -> dict:
    """
    Text-only pipeline entry point (no STT stage).
    Used for the 300-query eval harness where queries are loaded from dataset.
    t0 and t1 are set to the same timestamp (simulating instantaneous transcription).
    """
    now = time.perf_counter()
    return await run_pipeline(
        transcript=transcript,
        lang_code=lang_code,
        t0_audio_end=now,
        t1_text_ready=now,
        query_id=query_id,
        chunking_strategy=chunking_strategy,
        is_selected_doc_ids=is_selected_doc_ids,
    )
