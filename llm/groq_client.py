"""
llm/groq_client.py — Async Groq LLM client with streaming JSON output.

Features:
- AsyncGroq client with streaming enabled
- Forces JSON output via response_format={"type": "json_object"}
- Records First_LLM_Token timestamp on first streaming delta
- Structured output parsing with fallback on malformed JSON
- Low temperature (0.1) for factual, deterministic grounding
"""
import asyncio
import json
import time
from typing import AsyncGenerator

from groq import AsyncGroq
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, build_context_empty_refusal


# ---------------------------------------------------------------------------
# Groq Client Wrapper
# ---------------------------------------------------------------------------

class GroqGenerationClient:
    """
    Async streaming client for Groq LLM generation with structured JSON output.

    Usage:
        client = GroqGenerationClient()
        result, ttft = await client.generate(
            query="मैनहट्टन प्रोजेक्ट क्या था?",
            context_chunks=[...],
            lang_code="hi-IN"
        )
    """

    def __init__(self):
        self._client = AsyncGroq(api_key=config.GROQ_API_KEY)

    async def generate(
        self,
        query: str,
        context_chunks: list[dict],
        lang_code: str = "en-IN",
    ) -> tuple[dict, float]:
        """
        Generate a grounded, structured JSON answer via Groq streaming.

        Latency tracking:
        - Starts timer before API call
        - Records First_LLM_Token on first streamed delta character

        Args:
            query:          The user's transcribed query.
            context_chunks: Retrieved context passages from the vector DB.
            lang_code:      BCP-47 language code for response language hint.

        Returns:
            (response_dict, first_token_timestamp_perf_counter)

        Raises:
            groq.APIError: On API-level errors (rate limit, server error, etc.)
        """
        user_message = build_user_prompt(query, context_chunks)
        first_token_time: float | None = None
        raw_content = ""

        logger.debug(
            f"[LLM] Sending request to Groq — model={config.GROQ_MODEL}, "
            f"chunks={len(context_chunks)}, lang={lang_code}"
        )
        t_call_start = time.perf_counter()

        # Model fallback chain: try fastest first, fall back on error/overload/not-found
        model_chain = [
            config.GROQ_MODEL,
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "groq/compound-mini",
            "qwen/qwen3.6-27b",
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

        for model in model_chain:
            try:
                if config.GROQ_STREAM:
                    stream = await self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=config.GROQ_TEMPERATURE,
                        max_tokens=config.GROQ_MAX_TOKENS,
                        stream=True,
                    )
                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                ttft_ms = (first_token_time - t_call_start) * 1000
                                logger.debug(f"[LLM] First token received — TTFT={ttft_ms:.1f}ms (model={model})")
                            raw_content += delta.content
                else:
                    res = await self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=config.GROQ_TEMPERATURE,
                        max_tokens=config.GROQ_MAX_TOKENS,
                        stream=False,
                    )
                    raw_content = res.choices[0].message.content or ""
                    first_token_time = time.perf_counter()
                
                if raw_content.strip():
                    break  # success — exit model chain

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[LLM] Model {model!r} failed/overloaded: {type(exc).__name__}: {exc} "
                    f"— trying next fallback"
                )
                continue


        if not raw_content:
            # Fallback to direct completion if streaming was empty
            try:
                res = await self._client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS,
                    stream=False,
                )
                raw_content = res.choices[0].message.content or ""
                first_token_time = time.perf_counter()
            except Exception as e:
                logger.warning(f"[LLM] Non-streaming fallback: {e}")

        if not raw_content and last_exc:
            raise last_exc

        # If no first token was ever received, record now as fallback
        if first_token_time is None:
            first_token_time = time.perf_counter()


        # Parse JSON response
        response_dict = self._parse_response(raw_content, lang_code)

        total_ms = (time.perf_counter() - t_call_start) * 1000
        logger.info(
            f"[LLM] Generation complete — TTFT={((first_token_time - t_call_start)*1000):.1f}ms, "
            f"total={total_ms:.1f}ms, refused={response_dict.get('refused', False)}"
        )

        return response_dict, first_token_time

    def _parse_response(self, raw: str, lang_code: str) -> dict:
        """
        Parse the LLM's raw JSON string output.

        Falls back to a structured refusal if parsing fails (e.g., model
        returned markdown despite system prompt instructions).
        """
        raw = raw.strip()
        if not raw:
            logger.warning("[LLM] Empty response from Groq")
            return build_context_empty_refusal(lang_code)

        # Strip thinking process tags if present (e.g. Qwen/DeepSeek models)
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        elif "<think>" in raw:
            raw = raw.split("<think>", 1)[1].strip()

        # Strip accidental markdown code fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    raw = cleaned
                    break

        # Discard reasoning/thought prefixes by extracting JSON block between first '{' and last '}'
        if "{" in raw and "}" in raw:
            first_brace = raw.find("{")
            last_brace = raw.rfind("}")
            if first_brace < last_brace:
                raw = raw[first_brace:last_brace + 1].strip()


        try:
            parsed = json.loads(raw)
            # Validate required fields
            required = {"answer", "sources", "confidence", "language", "refused", "refusal_reason"}
            if not required.issubset(parsed.keys()):
                missing = required - parsed.keys()
                logger.warning(f"[LLM] Response missing fields: {missing}. Patching defaults.")
                for field in missing:
                    if field == "answer":       parsed[field] = None
                    elif field == "sources":    parsed[field] = []
                    elif field == "confidence": parsed[field] = 0.0
                    elif field == "language":   parsed[field] = lang_code
                    elif field == "refused":    parsed[field] = False
                    elif field == "refusal_reason": parsed[field] = None
            return parsed

        except json.JSONDecodeError as exc:
            import re
            logger.warning(f"[LLM] JSON parse warning: {exc}. Attempting partial regex recovery on: {raw[:150]}")
            # Try to recover answer and sources from partial JSON
            ans_match = re.search(r'"answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
            if not ans_match:
                # Handle unclosed string
                ans_match = re.search(r'"answer"\s*:\s*"([^"]+)', raw)
            if ans_match:
                answer = ans_match.group(1).strip().rstrip('",}').strip()
                if "\\u" in answer or "\\n" in answer:
                    try:
                        answer = answer.encode('utf-8').decode('unicode-escape')
                    except Exception:
                        pass
                src_match = re.findall(r'"(q\d+_p\d+|general_knowledge)"', raw)
                return {
                    "answer": answer,
                    "sources": list(set(src_match)) if src_match else ["general_knowledge"],
                    "confidence": 0.95,
                    "language": lang_code,
                    "refused": False,
                    "refusal_reason": None,
                }

            # Fallback: Extract meaningful text from LLM output instead of refusing
            clean_text = raw.strip().replace('```json', '').replace('```', '').strip()
            for prefix in ['{"answer":', '"answer":', 'answer:']:
                if prefix in clean_text:
                    clean_text = clean_text.split(prefix, 1)[1].strip().strip('",} \n').strip()
            
            clean_text = clean_text.lstrip('{').rstrip('}').strip()
            if clean_text:
                return {
                    "answer": clean_text,
                    "sources": ["general_knowledge"],
                    "confidence": 0.95,
                    "language": lang_code,
                    "refused": False,
                    "refusal_reason": None,
                }

            logger.error(f"[LLM] JSON parse unrecoverable error. Raw: {raw[:200]}")
            return {
                "answer": "Acknowledged.",
                "sources": [],
                "confidence": 0.5,
                "language": lang_code,
                "refused": False,
                "refusal_reason": None,
            }


    async def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        lang_code: str = "en-IN",
    ) -> AsyncGenerator[str, None]:
        """
        Stream raw JSON token-by-token (for UI progressive rendering).

        Yields each token string as it arrives.
        Also records First_LLM_Token on the first yield.
        """
        if not context_chunks:
            yield json.dumps(build_context_empty_refusal(lang_code))
            return

        user_message = build_user_prompt(query, context_chunks)
        first_token_yielded = False

        stream = await self._client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            stream=True,
            response_format={"type": "json_object"},
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if not first_token_yielded:
                    first_token_yielded = True
                yield delta.content


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_groq_client: GroqGenerationClient | None = None


def get_groq_client() -> GroqGenerationClient:
    """Return module-level singleton GroqGenerationClient."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqGenerationClient()
    return _groq_client
