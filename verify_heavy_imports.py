"""
verify_heavy_imports.py — Test imports of all heavy ML/API modules.
Run: python verify_heavy_imports.py
"""
import sys
import traceback

sys.path.insert(0, ".")
errors = []

def check(label, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        errors.append(label)

print("\n=== Heavy Module Import Verification ===\n")

# 1. groq SDK
def test_groq():
    from groq import AsyncGroq
    from llm.groq_client import GroqGenerationClient
    c = GroqGenerationClient()
    assert c is not None
check("groq SDK + GroqGenerationClient", test_groq)

# 2. lancedb
def test_lancedb():
    import lancedb
    from data.vector_store import VectorStore, CHUNK_SCHEMA
    assert len(CHUNK_SCHEMA) == 13  # 13 fields in schema
    s = VectorStore()
    assert s is not None
check("lancedb + VectorStore schema (13 fields)", test_lancedb)

# 3. datasets
def test_datasets():
    from datasets import load_dataset
    assert callable(load_dataset)
check("HuggingFace datasets", test_datasets)

# 4. FlagEmbedding (import only, no model download)
def test_flag():
    from FlagEmbedding import BGEM3FlagModel
    from retrieval.embedder import BGEEmbedder
    e = BGEEmbedder()
    assert e is not None
check("FlagEmbedding + BGEEmbedder", test_flag)

# 5. websockets
def test_ws():
    import websockets
    from stt.client import SarvamSTTClient, _build_ws_url
    url = _build_ws_url("hi-IN")
    assert "saaras:v3-realtime" in url
    assert "stream_type=fast" in url
    assert "mode=codemix" in url
check("websockets + SarvamSTTClient URL builder", test_ws)

# 6. guardrails (import only, no embedding needed)
def test_guardrails():
    from llm.guardrails import SemanticRouter, TOPIC_SEED_PHRASES
    assert len(TOPIC_SEED_PHRASES) == 5
    r = SemanticRouter()
    assert r is not None
check("guardrails SemanticRouter (5 topics)", test_guardrails)

# 7. retriever
def test_retriever():
    from retrieval.retriever import Retriever
    r = Retriever()
    assert r is not None
check("Retriever import", test_retriever)

# 8. orchestrator pipeline
def test_pipeline():
    from orchestrator.pipeline import run_pipeline, run_text_pipeline, initialize_pipeline
    assert callable(run_pipeline)
    assert callable(run_text_pipeline)
    assert callable(initialize_pipeline)
check("orchestrator/pipeline functions", test_pipeline)

# 9. eval harness
def test_eval():
    from eval.run_eval import run_evaluation
    from eval.query_generator import load_eval_queries, EvalQuery
    from eval.analytics import compute_stats, print_summary
    assert callable(run_evaluation)
check("eval harness (run_eval + query_generator + analytics)", test_eval)

# 10. main entry
def test_main():
    from main import run_file_demo, run_microphone_demo
    assert callable(run_file_demo)
check("main.py entry points", test_main)

print()
if errors:
    print(f"=== FAILED: {len(errors)} test(s) ===")
    sys.exit(1)
else:
    print("=== ALL 10 HEAVY IMPORTS PASSED ===")
    sys.exit(0)
