"""
data/chunkers/semantic_parent_child.py — Strategy A: Semantic Parent-Child Splitting.

Algorithm:
1. Each full translated passage becomes a "parent" chunk (preserves full context).
2. The passage is split into sentence-level "child" chunks.
3. Each child stores a parent_id FK pointing to its parent.

At retrieval time:
- ANN search is run on CHILD embeddings (dense, sentence-level)
- Top-k children are retrieved
- Their parent passages are fetched for full-context LLM generation

This improves retrieval precision (searching dense child embeddings) while
maintaining full answer context (returning the parent text).

Metadata preserved per chunk:
  lang_code, doc_id, query_id, query_type, is_selected, source_lang, target_lang
"""
import hashlib
import re
import unicodedata
from typing import Generator

from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import config

# ---------------------------------------------------------------------------
# Sentence splitter (language-agnostic, handles Indic scripts)
# ---------------------------------------------------------------------------

# Unicode sentence-ending punctuation covering Indic scripts
_SENTENCE_END_PATTERN = re.compile(
    r'(?<=[।॥\.\!\?।؟])\s+|(?<=\n)\s*'
)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using Unicode-aware punctuation detection.

    Handles:
    - Devanagari danda (।) and double danda (॥)
    - Standard ASCII . ! ?
    - Newline breaks as sentence boundaries
    Falls back to whitespace splitting if text is too short for sentence detection.
    """
    text = text.strip()
    if not text:
        return []

    # Try Unicode-aware split
    sentences = _SENTENCE_END_PATTERN.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Fallback: if still just 1 chunk and it's long, split by approx 100 chars at word boundary
    if len(sentences) == 1 and len(text) > 200:
        words = text.split()
        chunk, chunks = [], []
        for word in words:
            chunk.append(word)
            if len(" ".join(chunk)) >= 150:
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    return sentences if sentences else [text]


def _make_id(text: str, suffix: str = "") -> str:
    """Generate a deterministic short ID from text content."""
    raw = (text + suffix).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_passage(
    passage_text: str,
    lang_code: str,
    doc_id: str,
    query_id: int,
    query_type: str,
    is_selected: int,
    source_lang: str,
    target_lang: str,
) -> list[dict]:
    """
    Produce parent + child chunk records for a single passage.

    Returns a flat list of chunk dicts. Parent comes first, then its children.
    All dicts are ready to be embedded and upserted into LanceDB (sans 'vector').

    Args:
        passage_text:  The translated passage string.
        lang_code:     Short lang code (e.g. "hi", "ta").
        doc_id:        Passage-level unique identifier.
        query_id:      MSMARCO query_id (int).
        query_type:    Query category ("DESCRIPTION", "ENTITY", etc.).
        is_selected:   1 if this is a gold passage, 0 otherwise.
        source_lang:   e.g. "eng_Latn"
        target_lang:   e.g. "hin_Deva"

    Returns:
        List of chunk dicts with all metadata fields populated.
    """
    passage_text = passage_text.strip()
    if not passage_text:
        return []

    chunks: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Parent chunk (full passage)
    # ------------------------------------------------------------------
    parent_id = _make_id(passage_text, f"parent:{doc_id}")
    parent_chunk = {
        "id":          parent_id,
        "vector":      [],              # filled by embedder
        "text":        passage_text,
        "lang_code":   lang_code,
        "doc_id":      doc_id,
        "query_id":    query_id,
        "query_type":  query_type,
        "is_selected": is_selected,
        "chunk_type":  "parent",
        "parent_id":   "",             # parents have no parent
        "chunk_index": 0,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    chunks.append(parent_chunk)

    # ------------------------------------------------------------------
    # 2. Child chunks (sentence-level splits)
    # ------------------------------------------------------------------
    sentences = _split_sentences(passage_text)

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 10:
            # Skip trivially short fragments (punctuation, single chars)
            continue

        # Merge very short children with the next sentence for context
        if (len(sentence.split()) < 5 and idx + 1 < len(sentences)):
            sentences[idx + 1] = sentence + " " + sentences[idx + 1]
            continue

        child_id = _make_id(sentence, f"child:{doc_id}:{idx}")
        child_chunk = {
            "id":          child_id,
            "vector":      [],              # filled by embedder
            "text":        sentence,
            "lang_code":   lang_code,
            "doc_id":      doc_id,
            "query_id":    query_id,
            "query_type":  query_type,
            "is_selected": is_selected,
            "chunk_type":  "child",
            "parent_id":   parent_id,      # FK to parent
            "chunk_index": idx,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        chunks.append(child_chunk)

    return chunks


def chunk_dataset_record(record: dict) -> list[dict]:
    """
    Process a single MSMARCO-XI dataset record into parent+child chunks.

    Handles the passages struct: English_passages, Translated_passages, is_selected.
    Only processes Translated_passages (the target language corpus).

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
