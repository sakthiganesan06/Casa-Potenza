"""
verify_imports.py — Dry-run import and logic verification for all modules
that don't require heavy ML packages (no BGE-m3, no Groq call, no LanceDB I/O).
Run with: python verify_imports.py
"""
import sys
import time
import traceback

sys.path.insert(0, ".")
errors = []

def check(label, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        errors.append((label, str(e)))

print("\n=== Voice RAG — Module Verification ===\n")

# -------------------------------------------------------
# 1. config
# -------------------------------------------------------
def test_config():
    import config
    assert config.INGEST_LANGUAGES == ["hi", "ta", "bn"], f"Got {config.INGEST_LANGUAGES}"
    assert config.SARVAM_WS_URL.startswith("wss://")
    assert config.EMBEDDING_DIM == 1024
    assert config.SLIDING_WINDOW_OVERLAP_PCT == 0.15
    assert config.LATENCY_BUDGET_MS == 200

check("config: constants and paths", test_config)

# -------------------------------------------------------
# 2. Chunker — Semantic Parent-Child
# -------------------------------------------------------
def test_spc_chunker():
    from data.chunkers import semantic_parent_child as spc

    record = {
        "target_lang": "hin_Deva",
        "source_lang": "eng_Latn",
        "query_id": 1001,
        "query_type": "DESCRIPTION",
        "passages": {
            "is_selected": [1, 0],
            "Translated_passages": [
                "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव यह था कि अमेरिका के पास परमाणु हथियार थे। "
                "यह दुनिया के लिए एक बड़ा बदलाव था। वैज्ञानिकों ने मिलकर काम किया।",
                "यह एक ऐतिहासिक घटना थी।",
            ],
            "English_passages": ["The manhattan project...", "It was historic."],
        },
    }

    chunks = spc.chunk_dataset_record(record)
    assert len(chunks) > 0, "No chunks produced"

    parents  = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]
    assert len(parents) >= 1, "No parent chunks"
    assert len(children) >= 1, "No child chunks"

    # Verify parent-child FK linkage
    parent_ids = {p["id"] for p in parents}
    for child in children:
        assert child["parent_id"] in parent_ids, f"Orphan child: {child['id']}"

    # Verify metadata preservation
    assert parents[0]["lang_code"] == "hi"
    assert parents[0]["query_id"] == 1001
    assert parents[0]["is_selected"] == 1
    assert parents[0]["query_type"] == "DESCRIPTION"

check("chunker/semantic_parent_child: structure + FK linkage", test_spc_chunker)

# -------------------------------------------------------
# 3. Chunker — Sliding Window
# -------------------------------------------------------
def test_sw_chunker():
    from data.chunkers import sliding_window as sw

    long_text = " ".join([f"word{i}" for i in range(400)])  # 400 tokens
    record = {
        "target_lang": "tam_Taml",
        "source_lang": "eng_Latn",
        "query_id": 2002,
        "query_type": "ENTITY",
        "passages": {
            "is_selected": [0],
            "Translated_passages": [long_text],
            "English_passages": [long_text],
        },
    }

    chunks = sw.chunk_dataset_record(record)
    assert len(chunks) > 1, f"Expected multiple windows, got {len(chunks)}"

    # Verify overlap: last token of chunk[n] should appear in chunk[n+1]
    for i in range(len(chunks) - 1):
        tokens_i   = set(chunks[i]["text"].split())
        tokens_next = set(chunks[i + 1]["text"].split())
        overlap = tokens_i & tokens_next
        assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    # Verify metadata
    assert all(c["lang_code"] == "ta" for c in chunks)
    assert all(c["chunk_type"] == "window" for c in chunks)

check("chunker/sliding_window: window overlap + metadata", test_sw_chunker)

# -------------------------------------------------------
# 4. Prompts
# -------------------------------------------------------
def test_prompts():
    import json
    from llm.prompts import (
        SYSTEM_PROMPT, build_user_prompt,
        build_guardrail_refusal, build_context_empty_refusal
    )

    assert "refused" in SYSTEM_PROMPT
    assert "json" in SYSTEM_PROMPT.lower()

    chunks = [
        {"text": "परमाणु प्रयोग था।", "doc_id": "q1001_p0", "score": 0.92, "lang_code": "hi"},
        {"text": "वैज्ञानिकों ने काम किया।", "doc_id": "q1001_p1", "score": 0.81, "lang_code": "hi"},
    ]
    user_msg = build_user_prompt("मैनहट्टन प्रोजेक्ट क्या था?", chunks)
    assert "q1001_p0" in user_msg
    assert "PASSAGE 1" in user_msg

    refusal = build_guardrail_refusal("off_topic", "hi-IN")
    assert refusal["refused"] is True
    assert refusal["answer"] is None
    assert refusal["refusal_reason"] == "off_topic"

    empty = build_context_empty_refusal("ta-IN")
    assert empty["refused"] is True

check("llm/prompts: schema + user message construction", test_prompts)

# -------------------------------------------------------
# 5. Latency Tracker
# -------------------------------------------------------
def test_latency_tracker():
    from orchestrator.latency_tracker import LatencyRecord

    t = time.perf_counter()
    rec = LatencyRecord(
        query_id="q_test",
        lang_code="hi",
        query_text="test query",
        t0_audio_end=t,
        t1_text_ready=t + 0.048,    # 48ms STT
        t2_context_ready=t + 0.056, # 8ms retrieval
        t3_first_token=t + 0.175,   # 119ms LLM TTFT
    )
    rec.compute_latencies()

    assert abs(rec.stt_latency_ms - 48.0) < 1.0,       f"STT={rec.stt_latency_ms}"
    assert abs(rec.retrieval_latency_ms - 8.0) < 1.0,  f"Ret={rec.retrieval_latency_ms}"
    assert abs(rec.llm_ttft_ms - 119.0) < 1.0,         f"TTFT={rec.llm_ttft_ms}"
    assert abs(rec.total_latency_ms - 175.0) < 1.0,    f"Total={rec.total_latency_ms}"
    assert rec.within_budget is True, "175ms should be within 200ms budget"

    # Over-budget record
    rec2 = LatencyRecord(
        query_id="q_slow",
        lang_code="ta",
        query_text="slow query",
        t0_audio_end=t,
        t1_text_ready=t + 0.09,
        t2_context_ready=t + 0.1,
        t3_first_token=t + 0.25,
    )
    rec2.compute_latencies()
    assert rec2.within_budget is False, "250ms should exceed 200ms budget"

check("orchestrator/latency_tracker: milestone math + SLA flag", test_latency_tracker)

# -------------------------------------------------------
# 6. Backoff delays
# -------------------------------------------------------
def test_backoff():
    from stt.backoff import _compute_delays
    import config

    delays = _compute_delays()
    assert len(delays) == config.BACKOFF_MAX_RETRIES - 1
    for d in delays:
        assert config.BACKOFF_MIN_WAIT <= d <= config.BACKOFF_MAX_WAIT + 0.5, f"Delay {d} out of bounds"
    # Verify monotonically increasing (base, ignoring jitter)
    assert delays[-1] > delays[0], "Last delay should be larger than first"

check("stt/backoff: delay computation + bounds", test_backoff)

# -------------------------------------------------------
# 7. VAD state machine (no torch needed)
# -------------------------------------------------------
def test_vad_state():
    from stt.vad import VADState, VADEvent
    assert VADState.SILENT.name == "SILENT"
    assert VADState.SPEAKING.name == "SPEAKING"
    assert VADState.TRAILING.name == "TRAILING"
    assert VADEvent.SPEECH_START.name == "SPEECH_START"
    assert VADEvent.SPEECH_END.name == "SPEECH_END"

check("stt/vad: state machine enum values", test_vad_state)

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print()
if errors:
    print(f"=== FAILED: {len(errors)} test(s) ===")
    for label, err in errors:
        print(f"  - {label}: {err}")
    sys.exit(1)
else:
    print("=== ALL CHECKS PASSED ===")
    sys.exit(0)
