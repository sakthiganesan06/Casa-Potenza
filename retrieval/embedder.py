"""
retrieval/embedder.py — Multilingual E5 local embedding model wrapper with ONNX Runtime acceleration.

Features:
- Fast ONNX Runtime CPU inference for multilingual-e5-small (~15ms on CPU)
- Automatic fallback to standard PyTorch CPU model if ONNX fails to load
- 384-dimensional dense vectors
- Thread-safe asynchronous executions
- Dynamic input prefixing ("query: " / "passage: ") for E5 model compatibility
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Model Loader (lazy, singleton)
# ---------------------------------------------------------------------------

_model_instance: Any = None
_model_lock = asyncio.Lock()

# Fast in-memory LRU cache for single-query embeddings
_query_cache: dict[str, np.ndarray] = {}
MAX_QUERY_CACHE_SIZE = 2000

class _FallbackEmbedder:
    """Lightweight deterministic 384-d semantic hash embedder for serverless environments."""
    @staticmethod
    def embed_texts(texts: list[str]) -> np.ndarray:
        import hashlib
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % 384
                sign = 1.0 if (h % 2 == 0) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def _load_model_sync() -> tuple[Any, Any, bool]:
    """Load AutoTokenizer and either ONNX Session or AutoModel synchronously."""
    try:
        from transformers import AutoTokenizer
        
        logger.info("Initializing embedding tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)

        # Try ONNX Runtime first for optimal CPU execution speed
        try:
            import os
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            
            logger.info("Attempting to load E5-small in ONNX Runtime format...")
            t0 = time.perf_counter()
            
            onnx_path = hf_hub_download(
                repo_id="Xenova/multilingual-e5-small", 
                filename="onnx/model.onnx",
                local_files_only=True,
            )
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            session = ort.InferenceSession(
                onnx_path, 
                sess_options=sess_options, 
                providers=["CPUExecutionProvider"]
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"✅ ONNX model loaded successfully in {elapsed:.0f}ms")
            return tokenizer, session, True
            
        except Exception as exc:
            logger.warning(f"ONNX initialization skipped ({exc}). Trying PyTorch...")
            import torch
            from transformers import AutoModel
            model = AutoModel.from_pretrained(config.EMBEDDING_MODEL_NAME)
            return tokenizer, model, False

    except Exception as exc:
        logger.warning(f"Local embedding models not found in environment ({exc}). Using zero-dependency serverless embedder.")
        return None, None, False



class BGEEmbedder:
    """
    Thread-safe asynchronous wrapper around embedding models (ONNX Runtime / PyTorch).
    """

    def __init__(self):
        self._model = None
        self._executor = ThreadPoolExecutor(
            max_workers=config.EMBEDDING_THREAD_POOL_SIZE,
            thread_name_prefix="embedder",
        )

    async def initialize(self) -> None:
        """Load the model into memory. Idempotent."""
        global _model_instance
        if self._model is not None:
            return

        async with _model_lock:
            if _model_instance is None:
                loop = asyncio.get_event_loop()
                _model_instance = await loop.run_in_executor(
                    self._executor, _load_model_sync
                )
            self._model = _model_instance

    async def load(self) -> None:
        """Alias for initialize."""
        await self.initialize()

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    def _embed_sync(self, texts: list[str], max_length: int = config.EMBEDDING_MAX_LENGTH) -> np.ndarray:
        """
        Synchronous embedding call. Returns normalized float32 array of shape (N, dim).
        """
        tokenizer, model_or_session, is_onnx = _model_instance
        
        if tokenizer is None:
            return _FallbackEmbedder.embed_texts(texts)
        elif is_onnx:
            # --------------------------------------------------
            # ONNX Runtime Inference
            # --------------------------------------------------

            encoded_input = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='np'
            )
            
            # Map input names to numpy arrays
            inputs = {k: v.astype(np.int64) for k, v in encoded_input.items()}
            # Add dummy token_type_ids if required by the ONNX model but omitted by XLM-R tokenizer
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)
            
            # Run session
            outputs = model_or_session.run(None, inputs)
            token_embeddings = outputs[0]

            
            # Mean pooling in NumPy
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask
            
            # L2 Normalization in NumPy
            norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
            normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            return normalized.astype(np.float32)
            
        else:
            # --------------------------------------------------
            # PyTorch Inference (Fallback)
            # --------------------------------------------------
            import torch
            import torch.nn.functional as F

            encoded_input = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            )
            
            with torch.no_grad():
                model_output = model_or_session(**encoded_input)
                
            token_embeddings = model_output[0]
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            normalized = F.normalize(embeddings, p=2, dim=1)
            return normalized.cpu().numpy().astype(np.float32)

    async def embed_batch(self, texts: list[str], max_length: int = config.EMBEDDING_MAX_LENGTH) -> np.ndarray:
        """Embed a batch of texts asynchronously."""
        if not texts:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(self._executor, self._embed_sync, texts, max_length)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Embedded {len(texts)} texts in {elapsed_ms:.1f}ms")
        return result

    async def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single text string with cache lookup and short sequence optimization.
        Prefixes query with "query: " for E5 model compatibility.
        """
        clean_text = text.strip()
        cache_key  = clean_text.lower()

        if cache_key in _query_cache:
            logger.debug(f"[Embedder] Cache HIT for: '{cache_key[:40]}'")
            return _query_cache[cache_key]

        query_prefixed = "query: " + clean_text
        result = await self.embed_batch([query_prefixed], max_length=48)
        vec = result[0]

        if len(_query_cache) >= MAX_QUERY_CACHE_SIZE:
            _query_cache.pop(next(iter(_query_cache)))
        _query_cache[cache_key] = vec
        return vec

    async def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed the 'text' field of each chunk and attach the vector in-place.
        Prefixes passages with "passage: " for E5 model compatibility.
        """
        texts = ["passage: " + c["text"] for c in chunks]
        all_vecs = await self.embed_batch(texts)

        for i, chunk in enumerate(chunks):
            chunk["vector"] = all_vecs[i].tolist()

        return chunks


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedder: BGEEmbedder | None = None


async def get_embedder() -> BGEEmbedder:
    """Return the module-level singleton embedder, loading model if needed."""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
        await _embedder.load()
    return _embedder
