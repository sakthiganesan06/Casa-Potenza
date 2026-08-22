"""
data/chunkers/sliding_window.py — Strategy B: Sliding Window with 15% Overlap.

Algorithm:
- Tokenize passage text into whitespace-delimited tokens.
- Slide a window of SLIDING_WINDOW_TOKENS tokens across the passage.
- Each step advances by (1 - 0.15) * window_size = 85% of window size.
- Result: ~15% overlap between consecutive chunks ensures cross-boundary context.

Each chunk stores:
  chunk_index, start_token, end_token, lang_code, doc_id, query_id, etc.

This strategy is compared against Semantic Parent-Child for retrieval accuracy
in the eval harness.
"""
import hashlib
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import config


# ---------------------------------------------------------------------------
# Token-level sliding window
# ---------------------------------------------------------------------------

def _make_id(text: str, suffix: str = "") -> str:
    """Generate a deterministic short ID from text content."""
    raw = (text + suffix).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def chunk_passage(
    passage_text: str,
    lang_code: str,
    doc_id: str,
    query_id: int,
    query_type: str,
    is_selected: int,
    source_lang: str,
    target_lang: str,
    window_tokens: int = config.SLIDING_WINDOW_TOKENS,
    overlap_pct: float = config.SLIDING_WINDOW_OVERLAP_PCT,
) -> list[dict]:
    """
    Produce sliding-window chunk records for a single passage.

    Args:
        passage_text:  The translated passage string.
        lang_code:     Short lang code (e.g. "hi", "ta").
        doc_id:        Passage-level unique identifier.
        query_id:      MSMARCO query_id (int).
        query_type:    Query category.
        is_selected:   1 if gold passage, 0 otherwise.
        source_lang:   e.g. "eng_Latn"
        target_lang:   e.g. "hin_Deva"
        window_tokens: Number of tokens per chunk (default 256).
        overlap_pct:   Fraction of overlap between windows (default 0.15 = 15%).

    Returns:
        List of chunk dicts with all metadata fields populated (sans 'vector').
    """
    passage_text = passage_text.strip()
    if not passage_text:
        return []

    # Simple whitespace tokenization (Unicode-safe; works for all Indic scripts)
    tokens = passage_text.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return []

    # If passage is short enough, return as a single chunk (no windowing needed)
    if n_tokens <= window_tokens:
        chunk_text = passage_text
        chunk_id = _make_id(chunk_text, f"slide:{doc_id}:0")
        return [{
            "id":          chunk_id,
            "vector":      [],
            "text":        chunk_text,
            "lang_code":   lang_code,
            "doc_id":      doc_id,
            "query_id":    query_id,
            "query_type":  query_type,
            "is_selected": is_selected,
            "chunk_type":  "window",
            "parent_id":   "",
            "chunk_index": 0,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }]

    # Calculate step size (how many tokens to advance per window)
    overlap_tokens = max(1, int(window_tokens * overlap_pct))
    step = window_tokens - overlap_tokens  # 256 - 38 = 218 tokens per step

    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < n_tokens:
        end = min(start + window_tokens, n_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)

        chunk_id = _make_id(chunk_text, f"slide:{doc_id}:{chunk_index}")
        chunks.append({
            "id":          chunk_id,
            "vector":      [],
            "text":        chunk_text,
            "lang_code":   lang_code,
            "doc_id":      doc_id,
            "query_id":    query_id,
            "query_type":  query_type,
            "is_selected": is_selected,
            "chunk_type":  "window",
            "parent_id":   "",         # sliding windows have no parent hierarchy
            "chunk_index": chunk_index,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        # Break after processing the final window
        if end == n_tokens:
            break

        start += step
        chunk_index += 1

    return chunks


def chunk_dataset_record(record: dict) -> list[dict]:
    """
    Process a single MSMARCO-XI dataset record into sliding-window chunks.

    Handles the passages struct: English_passages, Translated_passages, is_selected.
    Only processes Translated_passages.

    Args:
        record: A HuggingFace dataset row dict.

    Returns:
        Flat list of all chunk dicts across all passages in this record.
    """
    lang_code   = record.get("target_lang", "unk").split("_")[0][:2].lower()
    source_lang = record.get("source_lang", "eng_Latn")
    target_lang = record.get("target_lang", "unk")
    query_id    = int(record.get("query_id", -1))
    query_type  = record.get("query_type", "UNKNOWN")

    passages       = record.get("passages", {})
    trans_passages = passages.get("Translated_passages", []) or []
    is_selected_l  = passages.get("is_selected", []) or []

    all_chunks: list[dict] = []
    for p_idx, passage_text in enumerate(trans_passages):
        if not isinstance(passage_text, str) or not passage_text.strip():
            continue
        is_sel = int(is_selected_l[p_idx]) if p_idx < len(is_selected_l) else 0
        doc_id = f"q{query_id}_p{p_idx}"

        chunks = chunk_passage(
            passage_text=passage_text,
            lang_code=lang_code,
            doc_id=doc_id,
            query_id=query_id,
            query_type=query_type,
            is_selected=is_sel,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        all_chunks.extend(chunks)

    return all_chunks
