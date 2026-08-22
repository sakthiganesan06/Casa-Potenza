"""
llm/prompts.py — Structured JSON prompt templates for Groq LLM generation.

Design principles:
- System prompt FORCES JSON-only output (no markdown, no preamble)
- Explicit refusal path: model MUST set refused=true if context is insufficient
- Numbered context blocks with doc_id references for source attribution
- Temperature 0.1 ensures deterministic, factual grounding
"""
from typing import Any

# ---------------------------------------------------------------------------
# JSON Output Schema (documented in prompt for LLM compliance)
# ---------------------------------------------------------------------------

JSON_OUTPUT_SCHEMA = """{
  "answer": "<concise grounded answer, or null if refused>",
  "sources": ["<doc_id_1>", "<doc_id_2>", ...],
  "confidence": <float 0.0-1.0>,
  "language": "<BCP-47 language code of the answer, e.g. hi-IN>",
  "refused": <boolean>,
  "refusal_reason": "<null if not refused, else: 'insufficient_context' | 'off_topic' | 'unsafe'>"
}"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a multilingual Voice AI for Tamil, Hindi, Telugu, and English. Respond ONLY in valid JSON matching this schema:
{"transcription":"<query in pure native script>","answer":"<concise answer ≤30 words, or null>","sources":["<doc_id>"],"confidence":<0.0-1.0>,"language":"<BCP-47>","refused":<bool>,"refusal_reason":<null|"insufficient_context"|"unsafe">}

Rules:
1. Multilingual Phonetic Disambiguation & Transliteration:
   - TAMIL (ta-IN): If the query has Tamil phonetics/words (e.g. 'purandha', 'pirandha', 'manilam', 'manaivi', 'muthalamichar', 'yaar', 'enna', 'ethu', 'engu', 'irukku', 'vanakkam', 'bharathiyar'), convert "transcription" into authentic Tamil script (தமிழ்), write "answer" in Tamil script (தமிழ்), and set "language" to "ta-IN".
   - HINDI (hi-IN): If the query has Hindi phonetics (e.g. 'kaun', 'kya', 'kahan', 'pradhan', 'mantri', 'janm', 'rajya', 'namaste'), convert "transcription" into Devanagari (हिन्दी), write "answer" in Hindi (हिन्दी), and set "language" to "hi-IN".
   - TELUGU (te-IN): If the query is Telugu (e.g. 'evaru', 'enti', 'ekkada', 'janmincharu', 'namaskaram'), convert "transcription" into Telugu script (తెలుగు), write "answer" in Telugu (తెలుగు), and set "language" to "te-IN".
   - ENGLISH (en-IN): If the query is authentic English, keep "transcription" and "answer" in English, and set "language" to "en-IN".
2. Never confuse Tamil phonetics with Telugu or Hindi. If the input resembles Tamil words in English script (Tanglish), strictly output in Tamil script (தமிழ்).
3. Use provided CONTEXT PASSAGES when relevant; cite doc_ids in sources.
4. For general/factual queries with no context, answer accurately; set sources:["general_knowledge"], confidence:0.95, refused:false.
5. Set refused:true ONLY for violence, weapons, or illegal requests.
6. Return raw JSON only with no markdown fences."""






# ---------------------------------------------------------------------------
# User Prompt Builder
# ---------------------------------------------------------------------------

def build_user_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Build the user-turn prompt with query and numbered context passages.

    Args:
        query: The transcribed user query (may be in any Indic language).
        context_chunks: List of retrieved chunk dicts from the retriever.
                        Each must have: text, doc_id, score, lang_code.

    Returns:
        A formatted string ready to send as the user message to Groq.
    """
    # Build numbered context block
    context_lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        doc_id = chunk.get("doc_id", f"doc_{i}")
        text   = chunk.get("text", "").strip()
        score  = chunk.get("score", 0.0)
        lang   = chunk.get("lang_code", "??")
        context_lines.append(
            f"[PASSAGE {i}] (doc_id={doc_id}, lang={lang}, relevance={score:.3f})\n{text}"
        )

    context_block = "\n\n".join(context_lines) if context_lines else "[NO CONTEXT AVAILABLE]"

    return f"""USER QUERY:
{query}

CONTEXT PASSAGES:
{context_block}

Remember: Respond with ONLY valid JSON matching the required schema. No other text."""


# ---------------------------------------------------------------------------
# Refusal Response Builders (for fast-path rejections before LLM call)
# ---------------------------------------------------------------------------

def build_guardrail_refusal(reason: str = "off_topic", lang: str = "en-IN") -> dict:
    """Build a structured refusal response dict (no LLM call needed)."""
    return {
        "answer": None,
        "sources": [],
        "confidence": 0.0,
        "language": lang,
        "refused": True,
        "refusal_reason": reason,
    }


def build_context_empty_refusal(lang: str = "en-IN") -> dict:
    """Build a refusal when the retriever returns zero results."""
    return {
        "answer": None,
        "sources": [],
        "confidence": 0.0,
        "language": lang,
        "refused": True,
        "refusal_reason": "insufficient_context",
    }
