"""
config.py — Centralized configuration for the Voice-Enabled RAG System.

All constants, API endpoints, model names, and tuning parameters live here.
Modules import from this file — never hardcode values elsewhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("LAMBDA_TASK_ROOT"))

if IS_VERCEL:
    LOG_DIR = "/tmp/logs"
    EVAL_OUTPUT_DIR = "/tmp/eval"
    QDRANT_PATH = os.getenv("QDRANT_PATH", ":memory:")
else:
    LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
    EVAL_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", str(BASE_DIR / "eval" / "results"))
    QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_db"))

# Ensure directories exist safely
try:
    for _dir in [LOG_DIR, EVAL_OUTPUT_DIR]:
        Path(_dir).mkdir(parents=True, exist_ok=True)
    if not QDRANT_PATH.startswith("http") and not QDRANT_PATH.startswith("grpc") and QDRANT_PATH != ":memory:":
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
except Exception:
    pass



# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not SARVAM_API_KEY:
    import warnings
    warnings.warn("SARVAM_API_KEY is not set. STT client will fail at runtime.", stacklevel=2)

if not GROQ_API_KEY:
    import warnings
    warnings.warn("GROQ_API_KEY is not set. LLM client will fail at runtime.", stacklevel=2)

# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "ai4bharat/MSMARCO-XI"

# Language subsets to ingest. Full list: hi, ta, bn, kn, ml, mr, te, gu, ur, or, pa, as, ne, sa
_raw_langs = os.getenv("INGEST_LANGUAGES", "hi,ta,bn")
INGEST_LANGUAGES: list[str] = [lang.strip() for lang in _raw_langs.split(",")]

# BCP-47 language code mapping for Sarvam API (from dataset target_lang codes)
LANG_TO_BCP47: dict[str, str] = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "te": "te-IN",
    "gu": "gu-IN",
    "ur": "ur-IN",
    "or": "or-IN",
    "pa": "pa-IN",
    "as": "as-IN",
    "ne": "ne-NP",
    "sa": "sa-IN",
    "en": "en-IN",
}

# Max passages to ingest per language (set to None to ingest all)
MAX_PASSAGES_PER_LANG: int | None = 50_000

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "semantic")  # "semantic" | "sliding"

# Sliding Window params
SLIDING_WINDOW_TOKENS: int = 256
SLIDING_WINDOW_OVERLAP_PCT: float = 0.15  # 15% overlap = ~38 tokens

# Semantic Parent-Child params
SEMANTIC_MAX_CHILD_TOKENS: int = 128  # max tokens per child chunk

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
EMBEDDING_DIM: int = 384
EMBEDDING_BATCH_SIZE: int = 64
EMBEDDING_MAX_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Qdrant / Vector Store Configuration
# ---------------------------------------------------------------------------
# Collection names for each chunking strategy
QDRANT_COLLECTION_SEMANTIC: str = "chunks_semantic"
QDRANT_COLLECTION_SLIDING: str = "chunks_sliding"

# ANN search parameters
TOP_K_RETRIEVAL: int = 2          # Fast retrieval: fewer chunks = faster prompt

TOP_K_DENSE_CANDIDATES: int = 20  # Qdrant retrieves 20 dense candidates, BM25 re-ranks to TOP_K_RETRIEVAL

# BM25 re-ranking
BM25_ENABLED: bool = True         # Reciprocal rank fusion after Qdrant
BM25_WEIGHT: float = 0.3          # Weight for BM25 score in fusion (0=pure Qdrant, 1=pure BM25)


# Grounding score threshold — below this, skip LLM and return safe refusal
# NOTE: FAISS IndexFlatIP cosine scores for good matches are ~0.45–0.80;
GROUNDING_SCORE_THRESHOLD: float = 0.0  # Allow LLM to answer general knowledge queries while using context when available


# ---------------------------------------------------------------------------
# Sarvam STT Configuration
# ---------------------------------------------------------------------------
SARVAM_WS_URL: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
SARVAM_MODEL: str = "saaras:v3-realtime"
SARVAM_STREAM_TYPE: str = "fast"          # fast | balanced | simulated
SARVAM_MODE: str = "codemix"              # transcribe | translate | verbatim | translit | codemix
SARVAM_DEFAULT_LANG: str = "hi-IN"       # default if language detection fails

# Audio format requirements
AUDIO_SAMPLE_RATE: int = 16_000          # 16kHz required by Sarvam
AUDIO_CHANNELS: int = 1                  # mono
AUDIO_CHUNK_MS: int = 100               # ms per chunk (100ms = 1600 samples at 16kHz)
AUDIO_CHUNK_SAMPLES: int = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)

# VAD parameters
VAD_SILENCE_THRESHOLD_MS: int = 200      # ms of silence to trigger Audio_End (was 300)
VAD_SPEECH_PAD_MS: int = 50             # ms of padding around speech segments

# ---------------------------------------------------------------------------
# Exponential Backoff (Sarvam WebSocket reconnect)
# ---------------------------------------------------------------------------
BACKOFF_MULTIPLIER: float = 0.5
BACKOFF_MIN_WAIT: float = 0.5   # seconds
BACKOFF_MAX_WAIT: float = 8.0   # seconds
BACKOFF_MAX_RETRIES: int = 5

# ---------------------------------------------------------------------------
# Groq LLM Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

GROQ_MAX_TOKENS: int = 180
GROQ_TEMPERATURE: float = 0.0
GROQ_STREAM: bool = False


# ---------------------------------------------------------------------------
# Guardrail / Semantic Router
# ---------------------------------------------------------------------------
# Cosine similarity threshold below which queries are rejected as off-topic
GUARDRAIL_SIMILARITY_THRESHOLD: float = 0.75

# Topic labels used by the semantic router
GUARDRAIL_TOPIC_LABELS: list[str] = [
    "information_retrieval",
    "question_answering",
    "factual_lookup",
    "off_topic",
    "unsafe_harmful",
]

# Labels that trigger immediate rejection (only strictly harmful/dangerous content)
GUARDRAIL_REJECT_LABELS: set[str] = {"unsafe_harmful"}

# ---------------------------------------------------------------------------
# Latency Targets (ms) — used in analytics to flag SLA violations
# ---------------------------------------------------------------------------
LATENCY_BUDGET_MS: int = 200         # total end-to-end budget
LATENCY_STT_BUDGET_MS: int = 55      # Text_Ready - Audio_End
LATENCY_RETRIEVAL_BUDGET_MS: int = 10  # Context_Retrieved - Text_Ready
LATENCY_LLM_TTFT_BUDGET_MS: int = 130  # First_LLM_Token - Context_Retrieved

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_NUM_QUERIES: int = 300
EVAL_SPLIT: str = "validation"
EVAL_LANGUAGES: list[str] = INGEST_LANGUAGES  # same as ingest languages

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MAX_CONCURRENT_PIPELINE: int = 3   # asyncio.Semaphore cap on parallel pipeline runs
EMBEDDING_THREAD_POOL_SIZE: int = 4  # threads for run_in_executor embedding calls
