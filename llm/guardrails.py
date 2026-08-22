"""
llm/guardrails.py — Fast semantic routing for query safety and relevance checks.

Architecture:
- 5 topic embedding centroids pre-computed at startup from representative phrases
- Cosine similarity of incoming query embedding vs. each centroid (pure numpy, <5ms)
- Queries mapped to closest topic label
- Rejection if best label is in GUARDRAIL_REJECT_LABELS or score < threshold

This runs BEFORE the expensive LLM call, enabling <5ms rejection of off-topic/unsafe queries.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Representative phrases per topic (used to compute centroid embeddings)
# ---------------------------------------------------------------------------

TOPIC_SEED_PHRASES: dict[str, list[str]] = {
    "information_retrieval": [
        "what is the information about",
        "tell me about",
        "explain this topic",
        "describe",
        "what does this mean",
        "इसके बारे में बताओ",
        "இதைப் பற்றி சொல்",
        "దీని గురించి చెప్పు",
    ],
    "question_answering": [
        "what is",
        "who is",
        "when did",
        "where is",
        "how does",
        "why is",
        "how many",
        "how much",
        "क्या है", "कौन है", "कब हुआ", "कहाँ है", "कितने", "kaun", "kya", "kab", "kaise",
        "என்ன", "யார்", "எப்போது", "எங்கு", "எத்தனை", "எப்படி", "எந்த", "உள்ளது",
        "enna", "yaar", "yar", "eppadi", "ethana", "ethanai", "ullathu", "kandam",
        "ఏమిటి", "ఎవరు", "ఎప్పుడు", "ఎక్కడ", "ఎన్ని", "evaru", "enti", "eppudu",
    ],
    "factual_lookup": [
        "fact about",
        "definition of",
        "history of",
        "statistics",
        "data about",
        "capital of",
        "national animal",
        "national anthem",
        "national bird",
        "तथ्य", "इतिहास", "राजधानी",
        "உண்மை", "வரலாறு", "தலைநகரம்", "தேசிய",
        "వాస్తవం", "చరిత్ర", "రాజధాని", "జాతీయ",
    ],
    "off_topic": [
        "play music",
        "book a flight",
        "tell me a joke",
        "what is the weather",
        "write me a poem",
        "stock price",
        "recipe for cooking",
        "sports score live",
    ],
    "unsafe_harmful": [
        "how to manufacture an explosive weapon bomb",
        "terrorist attack instructions",
        "how to assassinate kill suicide murder poison",
        "illegal narcotics drug manufacturing",
        "how to build weapons for violence mass harm",
    ],
}



class SemanticRouter:
    """
    Fast topic classifier using pre-computed centroid embeddings.

    Initialization:
        router = SemanticRouter()
        await router.initialize(embedder)

    Checking queries:
        is_safe, label, score = await router.check(query_embedding)
    """

    def __init__(self):
        self._centroids: dict[str, np.ndarray] = {}
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="guardrail")

    async def initialize(self, embedder: "Any") -> None:  # noqa: F821
        """
        Pre-compute centroid embeddings for all topic seed phrases.

        Called once at startup. Takes ~2-5 seconds depending on GPU/CPU.
        After initialization, all checks are <5ms (pure numpy cosine similarity).

        Args:
            embedder: An instance of retrieval.embedder.BGEEmbedder
        """
        logger.info("Guardrail: Pre-computing topic centroid embeddings...")
        t0 = time.perf_counter()

        for topic, phrases in TOPIC_SEED_PHRASES.items():
            # Embed all seed phrases for this topic
            embeddings = await embedder.embed_batch(phrases)
            # Centroid = mean of all phrase embeddings
            centroid = np.mean(embeddings, axis=0)
            # L2-normalize for cosine similarity via dot product
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._centroids[topic] = centroid

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Guardrail: {len(self._centroids)} topic centroids computed in {elapsed:.0f}ms"
        )
        self._initialized = True

    def _cosine_sim_sync(self, query_vec: np.ndarray) -> Tuple[str, float]:
        """
        Compute cosine similarity against all centroids. Returns (best_label, score).
        Runs synchronously — called via run_in_executor for async safety.
        """
        best_label = "off_topic"
        best_score = -1.0

        # L2-normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        for label, centroid in self._centroids.items():
            # Cosine similarity = dot product of normalized vectors
            score = float(np.dot(query_vec, centroid))
            if score > best_score:
                best_score = score
                best_label = label

        return best_label, best_score

    async def check(
        self, query_embedding: np.ndarray
    ) -> Tuple[bool, str, float]:
        """
        Check if a query is safe and on-topic.

        Args:
            query_embedding: numpy float32 array of shape (EMBEDDING_DIM,)

        Returns:
            (is_safe, topic_label, confidence_score)
            - is_safe: True if query should proceed to retrieval+LLM
            - topic_label: closest topic category
            - confidence_score: cosine similarity score [−1, 1]
        """
        if not self._initialized:
            # If not initialized, fail open (allow query through)
            logger.warning("Guardrail not initialized — allowing query through")
            return True, "unknown", 1.0

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        label, score = await loop.run_in_executor(
            self._executor, self._cosine_sim_sync, query_embedding
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Guardrail check: label={label}, score={score:.3f}, time={elapsed_ms:.2f}ms")

        # Rejection conditions: only reject if matched unsafe/off_topic with high similarity
        threshold = getattr(config, "GUARDRAIL_SIMILARITY_THRESHOLD", 0.70)
        is_rejected = (label in config.GUARDRAIL_REJECT_LABELS and score >= threshold)

        is_safe = not is_rejected
        return is_safe, label, score


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: SemanticRouter | None = None


async def get_router(embedder=None) -> SemanticRouter:
    """Return the module-level singleton router, initializing if needed."""
    global _router
    if _router is None:
        _router = SemanticRouter()
        if embedder is not None:
            await _router.initialize(embedder)
    return _router
