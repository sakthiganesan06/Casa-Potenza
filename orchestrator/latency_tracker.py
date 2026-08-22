"""
orchestrator/latency_tracker.py — Milestone-based latency logger.

Records 4 key pipeline milestones per query:
  t0: Audio_End        — VAD detects end of speech
  t1: Text_Ready       — Sarvam returns is_final=true transcript
  t2: Context_Retrieved — LanceDB ANN search + parent expansion complete
  t3: First_LLM_Token  — First streaming delta from Groq

Output: JSONL file at LOG_DIR/latency_log.jsonl
"""
import asyncio
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Milestone dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """Stores timing data for a single pipeline invocation."""
    query_id:           str   = ""
    lang_code:          str   = ""
    chunking_strategy:  str   = config.CHUNKING_STRATEGY
    query_text:         str   = ""

    # Raw perf_counter timestamps (seconds)
    t0_audio_end:       float = 0.0
    t1_text_ready:      float = 0.0
    t2_context_ready:   float = 0.0
    t3_first_token:     float = 0.0

    # Derived latencies (milliseconds)
    stt_latency_ms:       float = field(init=False, default=0.0)
    retrieval_latency_ms: float = field(init=False, default=0.0)
    llm_ttft_ms:          float = field(init=False, default=0.0)
    total_latency_ms:     float = field(init=False, default=0.0)

    # Quality flags
    guardrail_rejected: bool  = False
    llm_refused:        bool  = False
    retrieval_recall:   float = 0.0  # 1.0 if is_selected=1 passage in top-5

    # SLA compliance
    within_budget:      bool  = field(init=False, default=False)

    def __post_init__(self):
        self.compute_latencies()

    def compute_latencies(self):
        """Recompute derived latency fields from raw timestamps."""
        self.stt_latency_ms       = (self.t1_text_ready  - self.t0_audio_end)  * 1000
        self.retrieval_latency_ms = (self.t2_context_ready - self.t1_text_ready) * 1000
        self.llm_ttft_ms          = (self.t3_first_token - self.t2_context_ready) * 1000
        self.total_latency_ms     = (self.t3_first_token - self.t0_audio_end)    * 1000
        self.within_budget        = self.total_latency_ms <= config.LATENCY_BUDGET_MS


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """
    Async latency tracker that logs milestone timestamps per query.

    Thread-safe log writing via asyncio.Lock.
    """

    def __init__(self):
        self._log_path = Path(config.LOG_DIR) / "latency_log.jsonl"
        self._records: list[LatencyRecord] = []
        self._write_lock = asyncio.Lock()
        # Ensure log file exists safely
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.touch(exist_ok=True)
        except Exception:
            pass
        logger.info(f"LatencyTracker: logging to {self._log_path}")


    def build_record(
        self,
        query_id: str,
        lang_code: str,
        query_text: str,
        t0_audio_end: float,
        t1_text_ready: float,
        t2_context_ready: float,
        t3_first_token: float,
        guardrail_rejected: bool = False,
        llm_refused: bool = False,
        retrieval_recall: float = 0.0,
        chunking_strategy: str | None = None,
    ) -> LatencyRecord:
        """Create and store a LatencyRecord."""
        record = LatencyRecord(
            query_id=query_id,
            lang_code=lang_code,
            chunking_strategy=chunking_strategy or config.CHUNKING_STRATEGY,
            query_text=query_text[:120],  # truncate for log
            t0_audio_end=t0_audio_end,
            t1_text_ready=t1_text_ready,
            t2_context_ready=t2_context_ready,
            t3_first_token=t3_first_token,
            guardrail_rejected=guardrail_rejected,
            llm_refused=llm_refused,
            retrieval_recall=retrieval_recall,
        )
        record.compute_latencies()
        self._records.append(record)
        return record

    async def log(self, record: LatencyRecord) -> None:
        """Async append record to JSONL log file."""
        line = json.dumps(asdict(record)) + "\n"
        async with self._write_lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self._log_path.open("a", encoding="utf-8").write, line
            )

        # Console milestone summary
        status = "✅" if record.within_budget else "🔴"
        logger.info(
            f"{status} [{record.query_id}] "
            f"STT={record.stt_latency_ms:.1f}ms | "
            f"Retrieval={record.retrieval_latency_ms:.2f}ms | "
            f"LLM_TTFT={record.llm_ttft_ms:.1f}ms | "
            f"Total={record.total_latency_ms:.1f}ms"
        )

    async def log_and_return(self, record: LatencyRecord) -> LatencyRecord:
        """Log and return the record (convenience method for pipeline)."""
        await self.log(record)
        return record

    def get_summary(self) -> dict:
        """Return an in-memory summary of all recorded latencies."""
        if not self._records:
            return {}
        import numpy as np
        totals = [r.total_latency_ms for r in self._records]
        stts   = [r.stt_latency_ms for r in self._records]
        rets   = [r.retrieval_latency_ms for r in self._records]
        ttfts  = [r.llm_ttft_ms for r in self._records]
        return {
            "n_queries": len(self._records),
            "total": {
                "p50": float(np.percentile(totals, 50)),
                "p70": float(np.percentile(totals, 70)),
                "p90": float(np.percentile(totals, 90)),
                "p100": float(np.max(totals)),
            },
            "stt": {
                "p50": float(np.percentile(stts, 50)),
                "p70": float(np.percentile(stts, 70)),
                "p100": float(np.max(stts)),
            },
            "retrieval": {
                "p50": float(np.percentile(rets, 50)),
                "p70": float(np.percentile(rets, 70)),
                "p100": float(np.max(rets)),
            },
            "llm_ttft": {
                "p50": float(np.percentile(ttfts, 50)),
                "p70": float(np.percentile(ttfts, 70)),
                "p100": float(np.max(ttfts)),
            },
            "within_budget_pct": sum(r.within_budget for r in self._records) / len(self._records) * 100,
            "guardrail_rejected": sum(r.guardrail_rejected for r in self._records),
            "llm_refused": sum(r.llm_refused for r in self._records),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: LatencyTracker | None = None


def get_tracker() -> LatencyTracker:
    """Return the module-level singleton LatencyTracker."""
    global _tracker
    if _tracker is None:
        _tracker = LatencyTracker()
    return _tracker
