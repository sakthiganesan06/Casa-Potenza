"""
eval/run_eval.py — 300-query evaluation harness.

Runs all queries through the full RAG pipeline (text-only, bypassing STT)
using asyncio.gather() for high-concurrency parallel execution.

Outputs:
  - eval/results/eval_300.jsonl — per-query results with latency + accuracy
  - eval/results/summary.json  — aggregate stats (auto-triggers analytics)

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --n 50 --strategy sliding
    python eval/run_eval.py --n 300 --strategy both  # runs both strategies
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from orchestrator.pipeline import initialize_pipeline, run_text_pipeline
from orchestrator.latency_tracker import get_tracker
from eval.query_generator import load_eval_queries, EvalQuery
import eval.analytics as analytics


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

async def run_evaluation(
    n_queries: int = config.EVAL_NUM_QUERIES,
    strategies: list[str] | None = None,
) -> dict:
    """
    Run the full evaluation harness.

    Args:
        n_queries:  Number of queries to evaluate (default 300).
        strategies: List of chunking strategies to test. None = config default.

    Returns:
        Summary dict with P50/P70/P100 metrics per strategy.
    """
    strategies = strategies or [config.CHUNKING_STRATEGY]
    output_dir = Path(config.EVAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Initialize pipeline components
    # ------------------------------------------------------------------
    logger.info("=== Voice RAG Evaluation Harness ===")
    logger.info(f"N queries: {n_queries} | Strategies: {strategies}")
    await initialize_pipeline()

    # ------------------------------------------------------------------
    # 2. Load evaluation queries
    # ------------------------------------------------------------------
    logger.info(f"Loading {n_queries} eval queries from MSMARCO-XI validation split...")
    eval_queries = await load_eval_queries(n_total=n_queries)
    logger.info(f"Loaded {len(eval_queries)} queries")

    all_results = {}

    # ------------------------------------------------------------------
    # 3. Run evaluation for each strategy
    # ------------------------------------------------------------------
    for strategy in strategies:
        logger.info(f"\n=== Evaluating strategy: '{strategy}' ===")
        results = await _eval_strategy(eval_queries, strategy)
        all_results[strategy] = results

        # Write per-strategy JSONL
        output_path = output_dir / f"eval_{strategy}_{n_queries}q.jsonl"
        _write_jsonl(results, output_path)
        logger.info(f"Results written to: {output_path}")

    # ------------------------------------------------------------------
    # 4. Run analytics and print summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("=== EVALUATION COMPLETE — Running Analytics ===")
    logger.info("=" * 60 + "\n")

    summary = {}
    for strategy, results in all_results.items():
        strategy_summary = analytics.compute_stats(results, strategy)
        summary[strategy] = strategy_summary
        analytics.print_summary(strategy_summary, strategy)

    # Compare strategies if multiple were run
    if len(strategies) > 1:
        analytics.print_comparison(summary)

    # Write final summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary written to: {summary_path}")

    return summary


async def _eval_strategy(
    queries: list[EvalQuery],
    strategy: str,
) -> list[dict]:
    """
    Run all queries through the pipeline for a given chunking strategy.
    Uses asyncio.gather with a semaphore to cap concurrent requests.
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_PIPELINE)

    async def run_one(query: EvalQuery, idx: int) -> dict:
        async with semaphore:
            try:
                result = await run_text_pipeline(
                    transcript=query.text,
                    lang_code=query.lang_code,
                    query_id=f"eval_{strategy}_{idx:04d}",
                    chunking_strategy=strategy,
                    is_selected_doc_ids=query.gold_doc_ids,
                )
                result["eval_meta"] = {
                    "query_id":   query.query_id,
                    "lang_code":  query.lang_code,
                    "query_type": query.query_type,
                    "query_text": query.text[:100],
                    "gold_count": len(query.gold_doc_ids),
                }
                return result
            except Exception as exc:
                logger.error(f"[Eval:{strategy}] Query {idx} failed: {type(exc).__name__}: {exc}")
                return {
                    "error":      str(exc),
                    "query_id":   f"eval_{strategy}_{idx:04d}",
                    "latency":    {"total_ms": -1, "within_budget": False},
                    "eval_meta":  {"lang_code": query.lang_code, "query_type": query.query_type},
                }

    tasks = [run_one(q, i) for i, q in enumerate(queries)]

    logger.info(f"Running {len(tasks)} queries with concurrency={config.MAX_CONCURRENT_PIPELINE}...")
    t0 = time.perf_counter()

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"  Eval [{strategy}]",
        unit="query",
    )

    elapsed = time.perf_counter() - t0
    success_count = sum(1 for r in results if "error" not in r)
    logger.info(
        f"[Eval:{strategy}] Done — {success_count}/{len(results)} successful "
        f"in {elapsed:.1f}s ({len(results)/elapsed:.1f} q/s)"
    )

    return list(results)


def _write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def _main():
    parser = argparse.ArgumentParser(description="300-query evaluation harness for Voice RAG")
    parser.add_argument(
        "--n",
        type=int,
        default=config.EVAL_NUM_QUERIES,
        help=f"Number of queries to evaluate (default: {config.EVAL_NUM_QUERIES})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["semantic", "sliding", "both"],
        default="both",
        help="Chunking strategy to evaluate (default: both)",
    )
    args = parser.parse_args()

    strategies = ["semantic", "sliding"] if args.strategy == "both" else [args.strategy]
    await run_evaluation(n_queries=args.n, strategies=strategies)


if __name__ == "__main__":
    asyncio.run(_main())
