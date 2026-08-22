"""
eval/analytics.py — Statistical analytics engine for the Voice RAG system.

Computes:
  - Latency percentiles: P50, P70, P90, P100 (total and per-stage)
  - Accuracy: Recall@5 (proportion of queries where at least one gold passage was retrieved)
  - Guardrail rejection count & LLM refusal count
  - Per-language latency breakdown (en, hi, ta, te)
  - Strategy comparison table: Semantic Parent-Child vs. Sliding Window
"""
import json
import sys
from pathlib import Path
import numpy as np
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

console = Console(force_terminal=True, legacy_windows=False)


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute P50, P70, P90, P100 (max), P0 (min), and mean for a list of numbers."""
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p100": 0.0, "p0": 0.0, "mean": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "p0":   float(np.min(arr)),
        "p50":  float(np.percentile(arr, 50)),
        "p70":  float(np.percentile(arr, 70)),
        "p90":  float(np.percentile(arr, 90)),
        "p100": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _get_lat(r: dict, key: str) -> float:
    """Helper to extract latency metric from either top-level or latency dict."""
    if key in r:
        return float(r[key])
    lat = r.get("latency", {})
    if key in lat:
        return float(lat[key])
    if key == "total_latency_ms" and "total_ms" in lat:
        return float(lat["total_ms"])
    return 0.0


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(results: list[dict], k: int = 5) -> float:
    """Compute Recall@k across all successful evaluation results."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return 0.0

    hits = 0
    for r in valid:
        recall = r.get("recall_at_5", r.get("retrieval_recall", r.get("recall", 0.0)))
        if recall > 0.0 or r.get("is_hit", False):
            hits += 1

    return hits / len(valid)


# ---------------------------------------------------------------------------
# Full Stats Computation
# ---------------------------------------------------------------------------

def compute_stats(results: list[dict], strategy: str = "unknown") -> dict:
    """Compute complete statistical summary for a set of query results."""
    if not results:
        return {"error": "No results to analyze", "strategy": strategy}

    valid_results = [r for r in results if not r.get("error")]
    n_total   = len(results)
    n_valid   = len(valid_results)
    n_errors  = n_total - n_valid

    # Stage latencies
    stt_times       = [_get_lat(r, "stt_ms") for r in valid_results if _get_lat(r, "stt_ms") > 0]
    retrieval_times = [_get_lat(r, "retrieval_ms") for r in valid_results if _get_lat(r, "retrieval_ms") > 0]
    llm_times       = [_get_lat(r, "llm_ttft_ms") for r in valid_results if _get_lat(r, "llm_ttft_ms") > 0]
    total_times     = [_get_lat(r, "total_latency_ms") for r in valid_results if _get_lat(r, "total_latency_ms") > 0]

    # Status counts
    rejected_count = sum(1 for r in results if r.get("status") == "GUARDRAIL_REJECTED" or r.get("guardrail_rejected"))
    refused_count  = sum(1 for r in results if r.get("status") == "LLM_REFUSED" or r.get("refused"))
    ok_count       = sum(1 for r in results if not r.get("error") and not r.get("refused") and not r.get("guardrail_rejected"))

    # SLA compliance
    within_budget_count = sum(
        1 for r in valid_results
        if _get_lat(r, "total_latency_ms") <= config.LATENCY_BUDGET_MS
    )
    within_budget_pct = (within_budget_count / n_valid * 100) if n_valid > 0 else 0.0

    # Recall@5
    recall_5 = compute_recall_at_k(valid_results, k=5)

    # Per-language breakdown
    by_lang: dict[str, list[float]] = {}
    for r in valid_results:
        lang = r.get("lang") or r.get("lang_code") or r.get("eval_meta", {}).get("lang_code", "unknown")
        by_lang.setdefault(lang, []).append(_get_lat(r, "total_latency_ms"))

    lang_stats = {
        lang: {
            "n":    len(times),
            "p50":  float(np.percentile(times, 50)) if times else 0.0,
            "p100": float(np.max(times)) if times else 0.0,
            "mean": float(np.mean(times)) if times else 0.0,
        }
        for lang, times in by_lang.items()
    }

    return {
        "strategy": strategy,
        "n_total":  n_total,
        "n_valid":  n_valid,
        "n_errors": n_errors,
        "status_counts": {
            "OK":                  ok_count,
            "GUARDRAIL_REJECTED":  rejected_count,
            "LLM_REFUSED":         refused_count,
            "ERROR":               n_errors,
        },
        "rejected_count":    rejected_count,
        "refused_count":     refused_count,
        "within_budget_pct": within_budget_pct,
        "recall_at_5":       recall_5,
        "total_ms":          compute_percentiles(total_times),
        "stt_ms":            compute_percentiles(stt_times),
        "retrieval_ms":      compute_percentiles(retrieval_times),
        "llm_ttft_ms":       compute_percentiles(llm_times),
        "by_language":       lang_stats,
    }


# ---------------------------------------------------------------------------
# Rich Console Output
# ---------------------------------------------------------------------------

def print_summary(stats: dict, strategy: str) -> None:
    """Print a formatted summary table to console."""
    if "error" in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return

    console.rule(f"[bold blue]--- Strategy: {strategy.upper()} ---[/bold blue]")

    table = Table(
        title=f"Latency Percentiles — {strategy} ({stats['n_valid']} queries)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Stage",       style="white", min_width=20)
    table.add_column("P50 (ms)",    style="green",  justify="right")
    table.add_column("P70 (ms)",    style="yellow", justify="right")
    table.add_column("P90 (ms)",    style="orange3", justify="right")
    table.add_column("P100 (ms)",   style="red",    justify="right")
    table.add_column("Mean (ms)",   style="dim",    justify="right")

    def _fmt(v): return f"{v:.1f}"

    budget_color = "green" if stats["total_ms"]["p70"] < config.LATENCY_BUDGET_MS else "red"

    table.add_row(
        "[STT] Text_Ready",
        _fmt(stats["stt_ms"]["p50"]),
        _fmt(stats["stt_ms"]["p70"]),
        "—",
        _fmt(stats["stt_ms"]["p100"]),
        _fmt(stats["stt_ms"]["mean"]),
    )
    table.add_row(
        "[Retrieval] Context_Ready",
        _fmt(stats["retrieval_ms"]["p50"]),
        _fmt(stats["retrieval_ms"]["p70"]),
        "—",
        _fmt(stats["retrieval_ms"]["p100"]),
        _fmt(stats["retrieval_ms"]["mean"]),
    )
    table.add_row(
        "[LLM TTFT] First_Token",
        _fmt(stats["llm_ttft_ms"]["p50"]),
        _fmt(stats["llm_ttft_ms"]["p70"]),
        "—",
        _fmt(stats["llm_ttft_ms"]["p100"]),
        _fmt(stats["llm_ttft_ms"]["mean"]),
    )
    table.add_row(
        f"[{budget_color}][TOTAL] End-to-End[/{budget_color}]",
        f"[{budget_color}]{_fmt(stats['total_ms']['p50'])}[/{budget_color}]",
        f"[{budget_color}]{_fmt(stats['total_ms']['p70'])}[/{budget_color}]",
        f"[{budget_color}]{_fmt(stats['total_ms']['p90'])}[/{budget_color}]",
        f"[{budget_color}]{_fmt(stats['total_ms']['p100'])}[/{budget_color}]",
        f"[{budget_color}]{_fmt(stats['total_ms']['mean'])}[/{budget_color}]",
    )
    console.print(table)

    console.print(
        f"\n  Recall@5:           [bold]{stats['recall_at_5']*100:.1f}%[/bold]\n"
        f"  Within 200ms SLA:   [bold]{stats['within_budget_pct']:.1f}%[/bold]\n"
        f"  Guardrail rejected: {stats['rejected_count']}\n"
        f"  LLM refused:        {stats['refused_count']}\n"
        f"  Errors:             {stats['n_errors']}\n"
    )

    if stats.get("by_language"):
        lang_table = Table(
            title="Per-Language Breakdown",
            box=box.SIMPLE,
            header_style="bold magenta",
        )
        lang_table.add_column("Language", style="white")
        lang_table.add_column("N Queries", justify="right")
        lang_table.add_column("P50 (ms)",  justify="right", style="green")
        lang_table.add_column("P100 (ms)", justify="right", style="red")

        for lang, ls in sorted(stats["by_language"].items()):
            lang_table.add_row(
                lang, str(ls["n"]),
                f"{ls['p50']:.1f}", f"{ls['p100']:.1f}"
            )
        console.print(lang_table)


def print_comparison(all_stats: dict[str, dict]) -> None:
    """Print side-by-side comparison of strategies."""
    console.rule("[bold magenta]--- Strategy Comparison ---[/bold magenta]")

    headers = ["Metric", *[s.upper() for s in all_stats.keys()]]
    rows = []

    metrics = [
        ("Total P50 (ms)",       lambda s: f"{s['total_ms']['p50']:.1f}"),
        ("Total P70 (ms)",       lambda s: f"{s['total_ms']['p70']:.1f}"),
        ("Total P100 (ms)",      lambda s: f"{s['total_ms']['p100']:.1f}"),
        ("STT P50 (ms)",         lambda s: f"{s['stt_ms']['p50']:.1f}"),
        ("Retrieval P50 (ms)",   lambda s: f"{s['retrieval_ms']['p50']:.1f}"),
        ("LLM TTFT P50 (ms)",    lambda s: f"{s['llm_ttft_ms']['p50']:.1f}"),
        ("Recall@5",             lambda s: f"{s['recall_at_5']*100:.1f}%"),
        ("Within SLA %",         lambda s: f"{s['within_budget_pct']:.1f}%"),
    ]

    for label, extractor in metrics:
        row = [label]
        for strategy_stats in all_stats.values():
            try:
                row.append(extractor(strategy_stats))
            except Exception:
                row.append("—")
        rows.append(row)

    console.print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
