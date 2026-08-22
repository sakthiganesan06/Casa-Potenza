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

SYSTEM_PROMPT = """You are an ultra-fast, direct multilingual Voice AI assistant for Tamil, Hindi, Telugu, and English.
You must always output a valid, parseable JSON object with these exact keys:
{
  "transcription": "user query in pure native script",
  "answer": "direct factual answer in 1-2 concise sentences",
  "sources": ["general_knowledge"],
  "confidence": 0.95,
  "language": "en-IN",
  "refused": false,
  "refusal_reason": null
}

CRITICAL INSTRUCTIONS:
1. Provide ONLY the direct factual answer to the question asked.
2. NEVER echo schema placeholders (do not output '<query>' or '<answer>').
3. NEVER output thinking process, reasoning steps, or analysis.
4. Multilingual Language Rules:
   - TAMIL (ta-IN): For Tamil/Tanglish queries (e.g. 'muthalamichar', 'purandha', 'manaivi', 'yaar', 'enna'), output "transcription" and "answer" in authentic Tamil script (தமிழ்), setting "language": "ta-IN".
   - HINDI (hi-IN): For Hindi/Hinglish queries (e.g. 'pradhan mantri', 'kaun', 'kya', 'kahan', 'namaste'), output "transcription" and "answer" in Devanagari (हिन्दी), setting "language": "hi-IN".
   - TELUGU (te-IN): For Telugu queries (e.g. 'evaru', 'enti', 'ekkada'), output "transcription" and "answer" in Telugu script (తెలుగు), setting "language": "te-IN".
   - ENGLISH (en-IN): For English queries, output "transcription" and "answer" in English, setting "language": "en-IN".
5. For factual questions (Chief Ministers, Prime Ministers, dates, history), answer accurately.
6. Return ONLY the raw JSON object without markdown fences."""








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
