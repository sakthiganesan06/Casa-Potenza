"""
data/vector_store.py — Qdrant vector store with numpy-accelerated ANN search.

Provides:
- Qdrant local storage for persistence and ingestion
- Numpy matrix-based ANN search for sub-2ms cosine similarity queries
- At startup, vectors are loaded into dense numpy arrays per collection per language
- Batch upsert of chunk records with embeddings and payloads
- Payload indexing for rapid keyword matching
"""
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


def string_to_uuid(s: str) -> str:
    """Generate a deterministic UUID string from any string ID for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


class _NumpyIndex:
    """
    In-memory numpy matrix index for ultra-fast cosine similarity search.
    Vectors are stored pre-normalized so cosine similarity = dot product.
    Organized by collection and language for O(1) pre-filtering.
    """

    def __init__(self):
        # {collection_name: {"all": {"vectors": np.ndarray, "payloads": list[dict]}}}
        # {collection_name: {lang_code: {"vectors": np.ndarray, "payloads": list[dict]}}}
        self._indices: dict[str, dict[str, dict]] = {}
        # {collection_name: {point_uuid: dict}} for parent lookups
        self._point_map: dict[str, dict[str, dict]] = {}

    def build_from_scroll(self, collection_name: str, records: list) -> None:
        """Build numpy matrix index from scrolled Qdrant records."""
        if not records:
            self._indices[collection_name] = {}
            self._point_map[collection_name] = {}
            return

        # Group by language
        lang_groups: dict[str, list] = {"all": []}
        point_map: dict[str, dict] = {}

        for r in records:
            payload = r.payload or {}
            vec = np.array(r.vector, dtype=np.float32)
            lang = payload.get("lang_code", "")

            entry = {"vector": vec, "payload": payload, "id": r.id}
            lang_groups["all"].append(entry)

            if lang:
                if lang not in lang_groups:
                    lang_groups[lang] = []
                lang_groups[lang].append(entry)

            # Store in point_map for parent lookups
            point_map[str(r.id)] = payload

        # Build dense matrices per group
        indices = {}
        for lang_key, entries in lang_groups.items():
            vecs = np.stack([e["vector"] for e in entries], axis=0)
            # Pre-normalize for dot product = cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-9, None)
            vecs = vecs / norms
            indices[lang_key] = {
                "vectors": vecs,
                "payloads": [e["payload"] for e in entries],
            }

        self._indices[collection_name] = indices
        self._point_map[collection_name] = point_map
        logger.info(
            f"NumpyIndex: built {collection_name} — "
            f"{len(lang_groups['all'])} vectors, {len(lang_groups) - 1} lang groups"
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        lang_code: str,
        top_k: int,
    ) -> list[dict]:
        """
        Fast cosine similarity search using np.dot.
        Returns list of dicts sorted by descending score.
        """
        col_idx = self._indices.get(collection_name)
        if not col_idx:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            q = q / q_norm

        # Try language-filtered search first
        lang_key = lang_code if lang_code and lang_code in col_idx else None
        if lang_key:
            idx = col_idx[lang_key]
            scores = idx["vectors"] @ q  # BLAS-optimized dot product
            if len(scores) >= top_k:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for i in top_indices:
                p = idx["payloads"][i]
                results.append({
                    "id":          p.get("original_id", p.get("id", "")),
                    "text":        p.get("text", ""),
                    "score":       float(scores[i]),
                    "lang_code":   p.get("lang_code", ""),
                    "doc_id":      p.get("doc_id", ""),
                    "query_id":    p.get("query_id", -1),
                    "is_selected": p.get("is_selected", 0),
                    "chunk_type":  p.get("chunk_type", ""),
                    "parent_id":   p.get("parent_id", ""),
                    "chunk_index": p.get("chunk_index", 0),
                })
            if results:
                return results

        # Fallback to global (all languages) search
        idx = col_idx.get("all")
        if not idx or len(idx["vectors"]) == 0:
            return []

        scores = idx["vectors"] @ q
        if len(scores) >= top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            p = idx["payloads"][i]
            results.append({
                "id":          p.get("original_id", p.get("id", "")),
                "text":        p.get("text", ""),
                "score":       float(scores[i]),
                "lang_code":   p.get("lang_code", ""),
                "doc_id":      p.get("doc_id", ""),
                "query_id":    p.get("query_id", -1),
                "is_selected": p.get("is_selected", 0),
                "chunk_type":  p.get("chunk_type", ""),
                "parent_id":   p.get("parent_id", ""),
                "chunk_index": p.get("chunk_index", 0),
            })
        return results

    def get_point(self, collection_name: str, point_id: str) -> dict | None:
        """Fetch a point's payload by its UUID string."""
        col_map = self._point_map.get(collection_name, {})
        return col_map.get(point_id)

    def add_points(self, collection_name: str, records: list[dict]) -> None:
        """Add new points to an existing numpy index (for runtime upserts)."""
        for r in records:
            orig_id = r["id"]
            point_id = string_to_uuid(orig_id)
            vector = np.array(r["vector"], dtype=np.float32)
            payload = {k: v for k, v in r.items() if k != "vector"}
            payload["original_id"] = orig_id
            lang = payload.get("lang_code", "")

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 1e-9:
                vector = vector / norm

            # Store in point_map
            if collection_name not in self._point_map:
                self._point_map[collection_name] = {}
            self._point_map[collection_name][point_id] = payload

            # Add to indices
            if collection_name not in self._indices:
                self._indices[collection_name] = {}
            col_idx = self._indices[collection_name]

            for key in ["all"] + ([lang] if lang else []):
                if key not in col_idx:
                    col_idx[key] = {"vectors": vector.reshape(1, -1), "payloads": [payload]}
                else:
                    col_idx[key]["vectors"] = np.vstack([col_idx[key]["vectors"], vector.reshape(1, -1)])
                    col_idx[key]["payloads"].append(payload)


class VectorStore:
    """
    Manages two Qdrant collections — one per chunking strategy.
    Uses Qdrant for persistence and a numpy matrix index for fast search.
    """

    def __init__(self):
        self._client: QdrantClient | None = None
        self._np_index: _NumpyIndex = _NumpyIndex()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qdrant")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _connect_and_index_sync(self) -> QdrantClient:
        """Initialize Qdrant client and build numpy indices synchronously (runs in executor)."""
        if config.QDRANT_PATH.startswith("http://") or config.QDRANT_PATH.startswith("https://"):
            logger.info(f"Connecting to remote Qdrant service at: {config.QDRANT_PATH}")
            client = QdrantClient(url=config.QDRANT_PATH)
        else:
            logger.info(f"Initializing local in-process Qdrant at: {config.QDRANT_PATH}")
            client = QdrantClient(path=config.QDRANT_PATH)

        logger.info("Qdrant client connected successfully")

        # Build numpy matrix indices from all collections
        t0 = time.perf_counter()
        collections = [config.QDRANT_COLLECTION_SEMANTIC, config.QDRANT_COLLECTION_SLIDING]

        for col in collections:
            try:
                exists = client.collection_exists(col)
            except Exception:
                try:
                    client.get_collection(col)
                    exists = True
                except Exception:
                    exists = False

            if not exists:
                self._np_index.build_from_scroll(col, [])
                continue

            info = client.get_collection(col)
            logger.info(f"NumpyIndex: scrolling {info.points_count} points for '{col}'...")

            all_records = []
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=col,
                    limit=10000,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                all_records.extend(records)
                if offset is None:
                    break

            self._np_index.build_from_scroll(col, all_records)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"NumpyIndex: all collections indexed in {elapsed:.0f}ms")
        return client

    async def connect(self) -> None:
        """Async connect — runs synchronous open + indexing in executor."""
        loop = asyncio.get_event_loop()
        self._client = await loop.run_in_executor(self._executor, self._connect_and_index_sync)


    def _init_collection_sync(self, collection_name: str) -> None:
        """Create collection if it doesn't exist, and index metadata field."""
        assert self._client is not None, "Call connect() first"
        try:
            exists = self._client.collection_exists(collection_name)
        except Exception:
            try:
                self._client.get_collection(collection_name)
                exists = True
            except Exception:
                exists = False

        if exists:
            info = self._client.get_collection(collection_name)
            logger.info(f"Opened existing Qdrant collection: '{collection_name}' ({info.points_count} points)")
        else:
            logger.info(f"Creating new Qdrant collection: '{collection_name}'")
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="lang_code",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created collection and indexed 'lang_code' field for '{collection_name}'")

    async def init_tables(self) -> None:
        """Initialize both chunking strategy collections."""
        loop = asyncio.get_event_loop()
        for name in [config.QDRANT_COLLECTION_SEMANTIC, config.QDRANT_COLLECTION_SLIDING]:
            await loop.run_in_executor(self._executor, self._init_collection_sync, name)


    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _upsert_sync(self, collection_name: str, records: list[dict]) -> int:
        """Synchronously upsert records into Qdrant and update numpy index."""
        points = []
        for r in records:
            orig_id = r["id"]
            point_id = string_to_uuid(orig_id)
            vector = r["vector"]
            payload = {k: v for k, v in r.items() if k != "vector"}
            payload["original_id"] = orig_id

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self._client.upsert(collection_name=collection_name, points=points)
        # Also update numpy index for new points
        self._np_index.add_points(collection_name, records)
        return len(records)

    async def upsert_chunks(self, collection_name: str, records: list[dict]) -> int:
        """
        Async batch upsert of chunk records.
        Each record must have a unique 'id', 'vector', and metadata fields.
        """
        if not records:
            return 0
        loop = asyncio.get_event_loop()
        n = await loop.run_in_executor(
            self._executor, self._upsert_sync, collection_name, records
        )
        return n


    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _normalize_lang(self, lang_code: str | None) -> str:
        """Normalize language code to short 2-letter key stored in DB (e.g. en-IN -> en)."""
        if not lang_code:
            return ""
        code = lang_code.split("-")[0].split("_")[0].lower().strip()
        if code in ("auto", "all", "none", "*"):
            return ""
        return code

    def _search_sync(
        self,
        collection_name: str,
        query_vector: list[float],
        lang_code: str,
        top_k: int,
    ) -> list[dict]:
        """
        Fast numpy-based ANN search with language pre-filtering.
        Target: <2ms wall time for 20K vectors.
        """
        norm_lang = self._normalize_lang(lang_code)
        return self._np_index.search(collection_name, query_vector, norm_lang, top_k)

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        lang_code: str,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> list[dict]:
        """
        Async ANN search. Target: <5ms wall time.
        Returns list of dicts sorted by descending cosine similarity score.
        """
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        results = await loop.run_in_executor(
            self._executor, self._search_sync, collection_name, query_vector, lang_code, top_k
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Qdrant ANN search completed in {elapsed_ms:.2f}ms — {len(results)} results")
        return results

    def _fetch_parent_sync(self, collection_name: str, parent_id: str) -> dict | None:
        """Fetch a parent chunk by its original string id using numpy point map."""
        cache_key = f"{collection_name}:{parent_id}"
        if not hasattr(self, "_parent_cache"):
            self._parent_cache = {}
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]

        point_id = string_to_uuid(parent_id)
        p = self._np_index.get_point(collection_name, point_id)
        if p:
            res = {
                "id": p.get("original_id", parent_id),
                "text": p.get("text", ""),
                "doc_id": p.get("doc_id", "")
            }
            self._parent_cache[cache_key] = res
            return res

        # Fallback to Qdrant disk if not in numpy index
        try:
            records = self._client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )
            if records:
                r = records[0]
                p = r.payload or {}
                res = {
                    "id": p.get("original_id", parent_id),
                    "text": p.get("text", ""),
                    "doc_id": p.get("doc_id", "")
                }
                self._parent_cache[cache_key] = res
                return res
        except Exception as exc:
            logger.error(f"[VectorStore] Failed to fetch parent {parent_id}: {exc}")
        return None

    async def fetch_parent(self, collection_name: str, parent_id: str) -> dict | None:
        """Async fetch of a parent chunk for Semantic Parent-Child retrieval."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._fetch_parent_sync, collection_name, parent_id
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_table_stats(self) -> dict[str, int]:
        """Return point count per collection."""
        stats = {}
        if self._client is None:
            return stats
        for name in [config.QDRANT_COLLECTION_SEMANTIC, config.QDRANT_COLLECTION_SLIDING]:
            try:
                info = self._client.get_collection(name)
                stats[name] = info.points_count
            except Exception:
                stats[name] = -1
        return stats

    def get_all_records_sync(self, collection_name: str) -> list[dict]:
        """Utility function to scan/scroll all records in a collection (used for BM25 corpus builds)."""
        assert self._client is not None, "Call connect() first"
        records = []
        offset = None
        while True:
            scroll_res = self._client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            points, next_offset = scroll_res
            for p in points:
                payload = p.payload or {}
                records.append({
                    "id":          payload.get("original_id", payload.get("id", str(p.id))),
                    "text":        payload.get("text", ""),
                    "lang_code":   payload.get("lang_code", ""),
                    "doc_id":      payload.get("doc_id", ""),
                    "chunk_type":  payload.get("chunk_type", ""),
                    "parent_id":   payload.get("parent_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                })
            if next_offset is None:
                break
            offset = next_offset
        return records


# ---------------------------------------------------------------------------
# Module-level singleton (imported by retriever.py and ingest.py)
# ---------------------------------------------------------------------------
_store: VectorStore | None = None


async def get_vector_store() -> VectorStore:
    """Return the module-level singleton VectorStore, initializing if needed."""
    global _store
    if _store is None:
        _store = VectorStore()
        await _store.connect()
        await _store.init_tables()
    return _store
