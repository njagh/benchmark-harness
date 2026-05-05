"""Markdown report generator for benchmark results."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any


def generate_report(
    runs: list[dict[str, Any]],
    suite_id: str,
    models_config: dict[str, Any],
    out_path: str,
) -> str:
    """Generate a Markdown report from run results.

    Includes summary, timing summary, per-task results, per-task timing,
    and slowest tasks sections.

    Args:
        runs: List of run result dicts from SQLiteStore.get_runs().
        suite_id: Suite identifier.
        models_config: Model config dict (from load_model_config).
        out_path: Output file path for the report.

    Returns:
        The generated markdown string.
    """
    lines: list[str] = []

    # Header
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = platform.node()

    lines.append(f"# Benchmark Report: {suite_id}")
    lines.append(f"")
    lines.append(f"Date: {now}")
    lines.append(f"Host: {host}")
    lines.append(f"")

    # Models table
    lines.append(f"## Models")
    lines.append(f"")
    lines.append(f"| Alias | Backend | Quantization | Notes |")
    lines.append(f"|---|---|---|---|")

    model_info = models_config.get("models", {})
    for alias, mconfig in model_info.items():
        backend = mconfig.get("backend", "unknown")
        quant = mconfig.get("quantization", "unknown")
        notes = mconfig.get("notes", "")
        lines.append(f"| {alias} | {backend} | {quant} | {notes} |")

    lines.append(f"")

    # Summary table
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Model | Tasks Run | Passed | Failed | Avg Score | Avg TTFT (ms) |")
    lines.append(f"|---|---|---|---|---|---|")

    # Group by model
    model_stats: dict[str, list[dict]] = {}
    for run in runs:
        alias = run.get("model_alias", "unknown")
        if alias not in model_stats:
            model_stats[alias] = []
        model_stats[alias].append(run)

    for alias, model_runs in sorted(model_stats.items()):
        total = len(model_runs)
        passed = sum(1 for r in model_runs if r.get("exit_status") == "success")
        failed = total - passed
        avg_score = sum(r.get("score_primary", 0) or 0 for r in model_runs) / max(total, 1)
        avg_ttft = sum(r.get("ttft_ms", 0) for r in model_runs) / max(total, 1)
        tps_vals = []
        for r in model_runs:
            wall = r.get("total_wall_ms", 0)
            comp = r.get("completion_tokens", 0)
            if wall > 0:
                tps_vals.append(comp / (wall / 1000.0))
        avg_tps = sum(tps_vals) / max(len(tps_vals), 1) if tps_vals else 0
        lines.append(
            f"| {alias} | {total} | {passed} | {failed} | {avg_score:.3f} | {avg_ttft:.0f} | {avg_tps:.1f} |"
        )

    lines.append(f"")

    # Timing Summary
    lines.append(f"## Timing Summary")
    lines.append(f"")
    lines.append(f"| Model | Avg TTFT (ms) | Avg Decode (ms) | Avg Tok/s | Avg Total Wall (ms) |")
    lines.append(f"|---|---|---|---|---|")

    for alias, model_runs in sorted(model_stats.items()):
        avg_ttft = sum(r.get("ttft_ms", 0) for r in model_runs) / max(len(model_runs), 1)
        avg_decode = sum(r.get("decode_ms", 0) for r in model_runs) / max(len(model_runs), 1)
        avg_tps_vals = []
        for r in model_runs:
            if r.get("tokens_per_second", 0) > 0:
                avg_tps_vals.append(r["tokens_per_second"])
        avg_tps = sum(avg_tps_vals) / max(len(avg_tps_vals), 1) if avg_tps_vals else 0
        avg_wall = sum(r.get("total_wall_ms", 0) for r in model_runs) / max(len(model_runs), 1)
        lines.append(
            f"| {alias} | {avg_ttft:.0f} | {avg_decode:.0f} | {avg_tps:.1f} | {avg_wall:.0f} |"
        )

    lines.append(f"")

    # Per-Task Results
    lines.append(f"## Per-Task Results")
    lines.append(f"")
    lines.append(f"| Task | Model | Status | TTFT (ms) | Tokens |")
    lines.append(f"|---|---|---|---|---|")

    for run in runs:
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")
        status = run.get("exit_status", "unknown")
        ttft = run.get("ttft_ms", 0)
        tokens = run.get("total_tokens", 0)
        lines.append(f"| {task_id} | {alias} | {status} | {ttft:.0f} | {tokens} |")

    lines.append(f"")

    # Per-Task Timing
    lines.append(f"## Per-Task Timing")
    lines.append(f"")
    lines.append(f"| Task | Model | TTFT (ms) | Decode (ms) | Tok/s | Prompt Tok | Completion Tok |")
    lines.append(f"|---|---|---|---|---|---|---|")

    for run in runs:
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")
        ttft = run.get("ttft_ms", 0)
        decode = run.get("decode_ms", 0)
        tps = run.get("tokens_per_second", 0)
        pt = run.get("prompt_tokens", 0)
        ct = run.get("completion_tokens", 0)
        tps_str = f"{tps:.1f}" if tps > 0 else "N/A"
        lines.append(
            f"| {task_id} | {alias} | {ttft:.0f} | {decode:.0f} | {tps_str} | {pt} | {ct} |"
        )

    lines.append(f"")

    # Slowest Tasks
    lines.append(f"## Slowest Tasks (Top 5)")
    lines.append(f"")
    lines.append(f"| Rank | Task | Model | Wall Time (ms) | Tokens |")
    lines.append(f"|---|---|---|---|---|")

    sorted_runs = sorted(runs, key=lambda r: r.get("total_wall_ms", 0), reverse=True)
    for rank, run in enumerate(sorted_runs[:5], 1):
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")
        wall = run.get("total_wall_ms", 0)
        tokens = run.get("total_tokens", 0)
        lines.append(f"| {rank} | {task_id} | {alias} | {wall:.0f} | {tokens} |")

    lines.append(f"")

    # Scoring Summary
    scored_runs = [r for r in runs if r.get("score_primary") is not None]
    if scored_runs:
        lines.append(f"## Scoring Summary")
        lines.append(f"")
        lines.append(f"| Model | Avg Primary Score | Tasks Scored | Format Failures |")
        lines.append(f"|---|---|---|---|")

        scored_by_model: dict[str, list[dict]] = {}
        for r in scored_runs:
            alias = r.get("model_alias", "unknown")
            if alias not in scored_by_model:
                scored_by_model[alias] = []
            scored_by_model[alias].append(r)

        for alias, model_runs in sorted(scored_by_model.items()):
            avg_score = sum(r.get("score_primary", 0) or 0 for r in model_runs) / max(len(model_runs), 1)
            lines.append(f"| {alias} | {avg_score:.3f} | {len(model_runs)} | - |")

        lines.append(f"")

        # Per-task scoring
        lines.append(f"## Per-Task Scores")
        lines.append(f"")
        lines.append(f"| Task | Model | Primary Score | Secondary Scores | Explanation |")
        lines.append(f"|---|---|---|---|---|")

        for r in scored_runs:
            task_id = r.get("task_id", "unknown")
            alias = r.get("model_alias", "unknown")
            primary = r.get("score_primary", 0)
            sec_scores = r.get("score_secondary", {})
            sec_str = ", ".join(
                f"{k}={v.get('score', '?')}"
                for k, v in (sec_scores or {}).items()
            ) or "-"
            explanation = (r.get("score_explanation") or "").replace("\n", " ")
            lines.append(
                f"| {task_id} | {alias} | {primary:.3f} | {sec_str} | {explanation[:80]} |"
            )

        lines.append(f"")

    # Raw output excerpts for failed tasks
    failures = [r for r in runs if r.get("exit_status") == "error"]
    if failures:
        lines.append(f"## Failures")
        lines.append(f"")
        for run in failures:
            lines.append(f"### {run.get('task_id')} — {run.get('model_alias')}")
            lines.append(f"")
            lines.append(f"**Error:** {run.get('error_message', 'unknown')}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    # Write to file
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    report_str = "\n".join(lines)
    return report_str


def print_summary(runs: list[dict[str, Any]], suite_id: str) -> None:
    """Print a brief summary to stdout."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Summary: {suite_id}")
    table.add_column("Model", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg Tok/s", justify="right")

    model_stats: dict[str, list[dict]] = {}
    for run in runs:
        alias = run.get("model_alias", "unknown")
        if alias not in model_stats:
            model_stats[alias] = []
        model_stats[alias].append(run)

    for alias, model_runs in sorted(model_stats.items()):
        total = len(model_runs)
        passed = sum(1 for r in model_runs if r.get("exit_status") == "success")
        failed = total - passed
        avg_ttft = sum(r.get("ttft_ms", 0) for r in model_runs) / max(total, 1)
        tps_vals = []
        for r in model_runs:
            wall = r.get("total_wall_ms", 0)
            comp = r.get("completion_tokens", 0)
            if wall > 0:
                tps_vals.append(comp / (wall / 1000.0))
        avg_tps = sum(tps_vals) / max(len(tps_vals), 1) if tps_vals else 0

        table.add_row(
            alias, str(total), str(passed), str(failed), f"{avg_ttft:.0f}", f"{avg_tps:.1f}"
        )

    console.print(table)
