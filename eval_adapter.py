"""
eval_adapter.py — Adapter wiring Casa Potenza / Voice RAG into rag-local-eval-loop.

Provides the standard interface expected by eval/target.py:
- embed(texts: list[str]) -> np.ndarray (shape: len(texts), dim)
- embed_one(text: str) -> np.ndarray (shape: dim,)
- get_model() -> str / Any
- generate_answer(query: str, results: list) -> AnswerResult (text, grounded, generation_ms, model)
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import retrieval.embedder as emb_module
from groq import Groq

# ---------------------------------------------------------------------------
# Embedder Interface
# ---------------------------------------------------------------------------

_cached_embedder: emb_module.BGEEmbedder | None = None


def get_model() -> str:
    """Ensure embedding model is initialized, warmed up, and loaded."""
    global _cached_embedder
    if emb_module._model_instance is None:
        emb_module._model_instance = emb_module._load_model_sync()
    if _cached_embedder is None:
        try:
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
        except Exception:
            pass
        _cached_embedder = emb_module.BGEEmbedder()
        # Warm up embedder to avoid first-query cold start
        try:
            _cached_embedder._embed_sync(["query: warmup"], max_length=32)
        except Exception:
            pass
    return config.EMBEDDING_MODEL_NAME


def embed(texts: list[str]) -> np.ndarray:
    """Generates vector embeddings for a batch of text strings (passages)."""
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

    get_model()
    # Multilingual E5 expects "passage: " prefix for indexing / retrieval passages
    prefixed = [
        t if t.startswith(("passage: ", "query: ")) else f"passage: {t}"
        for t in texts
    ]
    return _cached_embedder._embed_sync(prefixed, max_length=config.EMBEDDING_MAX_LENGTH)


def embed_one(text: str) -> np.ndarray:
    """Generates a vector embedding for a single text string (query or probe)."""
    get_model()
    # Multilingual E5 expects "query: " prefix for search queries
    prefixed = text if text.startswith(("query: ", "passage: ")) else f"query: {text}"
    # Queries are short sentences (5-30 tokens) — max_length=48 is optimal for CPU speed
    vecs = _cached_embedder._embed_sync([prefixed], max_length=48)
    return vecs[0]


# ---------------------------------------------------------------------------
# Generator Interface
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


@dataclass
class AnswerResult:
    text: str
    grounded: bool
    generation_ms: float
    model: str = config.GROQ_MODEL


def _is_refusal_or_ungrounded(text: str) -> bool:
    """Check if the response text indicates context refusal or inability to answer."""
    lower = text.lower().strip()
    refusal_phrases = [
        "cannot answer",
        "can't answer",
        "cannot be answered",
        "can't be answered",
        "unable to answer",
        "not enough information",
        "insufficient information",
        "not mentioned in the context",
        "not mentioned in the provided",
        "not provided in the context",
        "not found in the context",
        "not found in the provided",
        "context does not provide",
        "context does not contain",
        "context does not mention",
        "context does not state",
        "not contain enough information",
        "no information provided",
        "no information is provided",
        "based on the provided context, there is no",
        "cannot be determined from the provided",
        "i do not have enough information",
        "i don't have enough information",
        "does not mention",
        "does not provide",
        "does not state",
        "no direct answer",
    ]
    return any(p in lower for p in refusal_phrases)


def generate_answer(query: str, results: list) -> AnswerResult:
    """
    Generates an answer given a query and retrieved context results.

    Args:
        query: User question string.
        results: List of context objects, each duck-typed with .text, .source, .score.

    Returns:
        AnswerResult with .text, .grounded, .generation_ms, .model.
    """
    t0 = time.perf_counter()
    model_name = config.GROQ_MODEL or "openai/gpt-oss-20b"

    # Filter and format context passages
    context_lines = []
    for i, res in enumerate(results, start=1):
        txt = getattr(res, "text", str(res)).strip()
        if txt:
            context_lines.append(f"[Passage {i}]\n{txt}")

    if not context_lines:
        dt = (time.perf_counter() - t0) * 1000
        return AnswerResult(
            text="I cannot answer this question based on the provided context.",
            grounded=False,
            generation_ms=dt,
            model=model_name,
        )

    context_str = "\n\n".join(context_lines)
    prompt = f"""You are a helpful and precise RAG assistant.
Answer the user's question using the factual information in the provided context passages.

CRITICAL RULES:
1. If the provided context contains facts that answer the question, give a clear, direct, and concise answer (1-2 sentences).
2. If the context passages do NOT contain information to answer the question, or are completely unrelated, state: "I cannot answer this question based on the provided context."
3. Do NOT hallucinate facts not present in the context.

Context Passages:
{context_str}

User Question: {query}

Answer:"""

    client = _get_groq_client()
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            timeout=8.0,
        )
        ans_text = (res.choices[0].message.content or "").strip()
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return AnswerResult(
            text=f"Error generating answer: {e}",
            grounded=False,
            generation_ms=dt,
            model=model_name,
        )

    dt = (time.perf_counter() - t0) * 1000
    is_refused = _is_refusal_or_ungrounded(ans_text) or not ans_text
    grounded = not is_refused

    return AnswerResult(
        text=ans_text,
        grounded=grounded,
        generation_ms=dt,
        model=model_name,
    )