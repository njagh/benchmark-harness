"""Markdown report generator for benchmark results."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any


def _generate_legacy_report(
    runs: list[dict[str, Any]],
    suite_id: str,
    models_config: dict[str, Any],
    out_path: str,
) -> str:
    """Generate a Markdown report from run results (legacy v1).

    Includes summary, timing summary, per-task results, per-task timing,
    and slowest tasks sections.
    """
    lines: list[str] = []
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

    # Coding Agent Ranking (when code_type is present in runs)
    _append_coding_agent_ranking(lines, runs)

    # M7 Judge sections
    _append_judge_sections(lines, runs, model_stats)

    # M8 Style Comparison sections
    _append_style_comparison(lines, runs)

    # M9 Context Length Analysis sections
    _append_context_analysis(lines, runs)

    # M10 Quantization Comparison sections
    _append_quantization_comparison(lines, runs)

    # Write to file
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    report_str = "\n".join(lines)
    return report_str


def generate_report(
    runs: list[dict[str, Any]],
    suite_id: str,
    models_config: dict[str, Any],
    out_path: str,
    v2: bool = False,
    sections: list[str] | None = None,
    prior_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a Markdown report from run results.

    Args:
        runs: List of run result dicts from SQLiteStore.get_runs().
        suite_id: Suite identifier.
        models_config: Model config dict (from load_model_config).
        out_path: Output file path for the report.
        v2: If True, use the modular v2 report generator.
        sections: Which sections to include (default: all). Used when v2=True.
        prior_runs: Prior run results for regression detection. Used when v2=True.

    Returns:
        The generated markdown string.
    """
    if v2:
        from bench_harness.reports.v2 import generate_report_v2
        return generate_report_v2(
            runs, suite_id, models_config, out_path,
            sections=sections, prior_runs=prior_runs,
        )
    else:
        return _generate_legacy_report(runs, suite_id, models_config, out_path)


def _group_by_family(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group runs by task family.

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping family name to list of runs for that family.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        task_id = run.get("task_id", "unknown")
        # Extract family from task_id: "local.docker_compose.fix_yaml_001" -> "docker_compose"
        family = _extract_family_from_task_id(task_id)
        if family is None:
            family = "other"
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(run)
    return by_family


def _extract_family_from_task_id(task_id: str) -> str | None:
    """Extract the sub-family from a task ID.

    For task IDs like "local.docker_compose.fix_yaml_001", returns "docker_compose".
    For task IDs without a clear family component, returns None.

    Args:
        task_id: The task identifier string.

    Returns:
        The family sub-component or None.
    """
    # Handle dot-separated task IDs
    parts = task_id.split(".")
    if len(parts) >= 3:
        # local.docker_compose.fix_yaml_001 -> docker_compose
        # local.litellm_routing.routing_001 -> litellm_routing
        # local.qwen3_debug.debug_001 -> qwen3_debug
        candidate = parts[1]
        skip = {"smoke", "local", "public", "synthetic"}
        if candidate.lower() not in skip:
            return candidate
    # smoke.factual_001: "smoke" is prefix, factual_001 is name (not a family)
    if len(parts) >= 2:
        candidate = parts[1]
        skip = {"factual", "json", "python", "debug", "instruction"}
        if candidate.lower() not in skip:
            # Only return if it doesn't look like a task name with underscore
            if "_" not in candidate or len(parts) < 3:
                return candidate
    # Handle underscore-based task IDs for public benchmarks
    parts = task_id.split("_")
    if len(parts) >= 2:
        candidate = parts[1]
        skip = {"factual", "json", "python", "debug", "code", "instruction"}
        if candidate.lower() not in skip:
            return candidate
    return None


def _append_coding_agent_ranking(lines: list[str], runs: list[dict[str, Any]]) -> None:
    """Append Coding Agent Ranking and Family Breakdown sections if code tasks exist.

    Args:
        lines: List of markdown line strings to append to.
        runs: List of run result dicts.
    """
    # Check if any run has a code_type (indicating code tasks were run)
    code_runs = [r for r in runs if r.get("code_type") is not None]
    if not code_runs:
        return

    scored_runs = [r for r in code_runs if r.get("score_primary") is not None]
    if not scored_runs:
        return

    # Group by model
    model_stats: dict[str, list[dict]] = {}
    for run in scored_runs:
        alias = run.get("model_alias", "unknown")
        if alias not in model_stats:
            model_stats[alias] = []
        model_stats[alias].append(run)

    # Build ranking
    rankings = []
    for alias, model_runs in sorted(model_stats.items()):
        total_tasks = len(model_runs)
        avg_score = sum(
            r.get("score_primary", 0) or 0 for r in model_runs
        ) / max(total_tasks, 1)
        # Count successfully executable outputs (tests_passed > 0 or code_status is success)
        executable_success = sum(
            1
            for r in model_runs
            if r.get("tests_passed", 0) and r.get("tests_passed", 0) > 0
            or r.get("code_status") == "success"
        )

        # Best and worst family
        family_scores: dict[str, list[float]] = {}
        for r in model_runs:
            family = _extract_family_from_task_id(r.get("task_id", ""))
            if family is None:
                family = "other"
            score = r.get("score_primary", 0) or 0
            if family not in family_scores:
                family_scores[family] = []
            family_scores[family].append(score)

        best_family = ""
        worst_family = ""
        best_avg = -1
        worst_avg = 2
        for fam, scores in sorted(family_scores.items()):
            fam_avg = sum(scores) / max(len(scores), 1)
            if fam_avg > best_avg:
                best_avg = fam_avg
                best_family = fam
            if fam_avg < worst_avg:
                worst_avg = fam_avg
                worst_family = fam

        rankings.append(
            {
                "alias": alias,
                "total_tasks": total_tasks,
                "avg_score": avg_score,
                "executable_success": executable_success,
                "best_family": best_family,
                "worst_family": worst_family,
                "family_scores": family_scores,
            }
        )

    # Sort by avg score descending
    rankings.sort(key=lambda x: x["avg_score"], reverse=True)

    # Coding Agent Ranking table
    lines.append(f"## Coding Agent Ranking")
    lines.append(f"")
    lines.append(f"| Rank | Model | Tasks | Avg Score | Best Family | Worst Family |")
    lines.append(f"|---|---|---|---|---|---|")

    for rank, info in enumerate(rankings, 1):
        lines.append(
            f"| {rank} | {info['alias']} | {info['total_tasks']} "
            f"| {info['avg_score']:.3f} | {info['best_family']} | {info['worst_family']} |"
        )

    lines.append(f"")

    # Per-Family Breakdown
    lines.append(f"## Family Breakdown")
    lines.append(f"")
    lines.append(f"| Family | Tasks | Avg Score | Passed | Failed |")
    lines.append(f"|---|---|---|---|---|")

    family_data = _group_by_family(scored_runs)
    for family, family_runs in sorted(family_data.items()):
        total = len(family_runs)
        passed = sum(
            1 for r in family_runs
            if r.get("tests_passed", 0) and r.get("tests_passed", 0) > 0
            or r.get("exit_status") == "success"
            or (r.get("score_primary", 0) or 0) >= 0.5
        )
        failed = total - passed
        avg_score = sum(r.get("score_primary", 0) or 0 for r in family_runs) / max(total, 1)
        lines.append(
            f"| {family} | {total} | {avg_score:.3f} | {passed} | {failed} |"
        )

    lines.append(f"")


def _append_judge_sections(
    lines: list[str], runs: list[dict[str, Any]], model_stats: dict[str, list[dict]],
) -> None:
    """Append judge-scored task, dimension breakdown, and pairwise comparison sections.

    Args:
        lines: List of markdown line strings to append to.
        runs: List of run result dicts.
        model_stats: Pre-computed dict mapping model alias to list of runs.
    """
    # 1. Judge-Scored Tasks
    judge_runs = [r for r in runs if r.get("judge_score") is not None]
    if judge_runs:
        lines.append(f"## Judge-Scored Tasks")
        lines.append(f"")
        lines.append(f"| Task | Model | Judge Score | Judge Model | Dimensions |")
        lines.append(f"|---|---|---|---|---|")

        for r in judge_runs:
            task_id = r.get("task_id", "unknown")
            alias = r.get("model_alias", "unknown")
            score = f"{r['judge_score']:.3f}" if r.get("judge_score") is not None else "N/A"
            judge_model = r.get("judge_model") or "N/A"
            dims = r.get("judge_dimensions") or {}
            if isinstance(dims, str):
                try:
                    import json
                    dims = json.loads(dims)
                except (json.JSONDecodeError, ValueError):
                    dims = {}
            dim_str = ", ".join(f"{k}={v}" for k, v in dims.items()) if dims else "-"
            lines.append(f"| {task_id} | {alias} | {score} | {judge_model} | {dim_str} |")

        lines.append(f"")

    # 2. Judge Dimension Breakdown
    if judge_runs:
        lines.append(f"## Judge Dimension Breakdown")
        lines.append(f"")
        lines.append(f"| Model | Dimension | Avg Score | Std Dev |")
        lines.append(f"|---|---|---|---|")

        # Gather dimension scores per model
        dim_scores: dict[str, dict[str, list[float]]] = {}
        for r in judge_runs:
            alias = r.get("model_alias", "unknown")
            dims = r.get("judge_dimensions") or {}
            if isinstance(dims, str):
                try:
                    import json
                    dims = json.loads(dims)
                except (json.JSONDecodeError, ValueError):
                    dims = {}
            for dim_name, dim_value in dims.items():
                if dim_name not in dim_scores:
                    dim_scores[dim_name] = {}
                if alias not in dim_scores[dim_name]:
                    dim_scores[dim_name][alias] = []
                try:
                    dim_scores[dim_name][alias].append(float(dim_value))
                except (ValueError, TypeError):
                    pass

        def _stddev(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5

        for dim_name in sorted(dim_scores.keys()):
            for alias in sorted(dim_scores[dim_name].keys()):
                values = dim_scores[dim_name][alias]
                avg = sum(values) / max(len(values), 1)
                sd = _stddev(values)
                lines.append(
                    f"| {alias} | {dim_name} | {avg:.3f} | {sd:.3f} |"
                )

        lines.append(f"")

    # 3. Pairwise Comparisons
    pairwise_runs = [r for r in runs if r.get("pairwise_winner") is not None]
    if pairwise_runs:
        lines.append(f"## Pairwise Comparisons")
        lines.append(f"")
        lines.append(f"| Task | Model A | Model B | Winner | Margin | Confidence |")
        lines.append(f"|---|---|---|---|---|---|")

        for r in pairwise_runs:
            task_id = r.get("task_id", "unknown")
            model_a = r.get("pairwise_model_a", "N/A")
            model_b = r.get("pairwise_model_b", "N/A")
            winner = r.get("pairwise_winner", "N/A")
            margin = r.get("pairwise_margin") or "N/A"
            confidence = r.get("pairwise_confidence")
            if confidence is not None:
                conf_str = f"{confidence:.3f}"
            else:
                conf_str = "N/A"
            lines.append(
                f"| {task_id} | {model_a} | {model_b} | {winner} | {margin} | {conf_str} |"
            )

        lines.append(f"")


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


def _append_style_comparison(lines: list[str], runs: list[dict[str, Any]]) -> None:
    """Append Style Comparison report sections if runs have prompt_style metadata.

    Generates 6 sub-sections when prompt_style is detected in run data:
    1. Style Comparison Summary
    2. Per-Task Style Breakdown
    3. Best Style Per Family
    4. Verbosity Analysis
    5. Latency Comparison
    6. Recommended Style

    Args:
        lines: List of markdown line strings to append to.
        runs: List of run result dicts.
    """
    style_runs = [r for r in runs if r.get("prompt_style") is not None]
    if not style_runs:
        return

    # Delegate to the style comparison module for the full report
    from bench_harness.reports.style_comparison import generate_style_report

    report = generate_style_report(runs)
    if report:
        lines.append("")
        lines.append(report)
        lines.append("")


def _append_context_analysis(
    lines: list[str], runs: list[dict[str, Any]],
) -> None:
    """Append Context Length Analysis sections if context data exists.

    Generates 3 sub-sections when context_tokens is detected in run data:
    1. Context Length Analysis — grouped by context size bucket
    2. Context Quality vs Length — score vs estimated_tokens table
    3. Context Breakpoint Detection — identify quality drop points

    Args:
        lines: List of markdown line strings to append to.
        runs: List of run result dicts.
    """
    context_runs = [r for r in runs if r.get("context_tokens") is not None]
    if not context_runs:
        return

    # Group by context size
    by_size: dict[str, list[dict]] = {}
    for r in context_runs:
        size = r["context_tokens"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)

    # 1. Context Length Analysis — avg score, avg wall, avg tok/s per bucket
    lines.append("## Context Length Analysis")
    lines.append("")
    lines.append("| Context Size | Tasks Run | Avg Score | Avg Wall (ms) | Avg Tok/s | Avg Est. Prompt Tokens |")
    lines.append("|---|---|---|---|---|---|")

    size_order = ["small", "medium", "large", "xlarge"]
    for size in size_order:
        size_runs = by_size.get(size, [])
        if not size_runs:
            continue
        total = len(size_runs)
        avg_score = sum(r.get("score_primary", 0) or 0 for r in size_runs) / max(total, 1)
        avg_wall = sum(r.get("total_wall_ms", 0) for r in size_runs) / max(total, 1)
        tps_vals = [r.get("tokens_per_second", 0) for r in size_runs if r.get("tokens_per_second", 0) > 0]
        avg_tps = sum(tps_vals) / max(len(tps_vals), 1) if tps_vals else 0
        est_tokens_vals = [r.get("estimated_prompt_tokens", 0) or 0 for r in size_runs]
        avg_est = sum(est_tokens_vals) / max(total, 1)
        lines.append(
            f"| {size} | {total} | {avg_score:.3f} | {avg_wall:.0f} | {avg_tps:.1f} | {avg_est:.0f} |"
        )

    lines.append("")

    # 2. Context Quality vs Length — per-task detail table
    lines.append("## Context Quality vs Length")
    lines.append("")
    lines.append("| Task | Model | Context Size | Est. Tokens | Score | Wall (ms) |")
    lines.append("|---|---|---|---|---|---|")

    for r in context_runs:
        task_id = r.get("task_id", "unknown")
        alias = r.get("model_alias", "unknown")
        size = r.get("context_tokens", "unknown")
        est = r.get("estimated_prompt_tokens") or 0
        score = r.get("score_primary")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        wall = r.get("total_wall_ms", 0)
        lines.append(f"| {task_id} | {alias} | {size} | {est} | {score_str} | {wall:.0f} |")

    lines.append("")

    # 3. Context Breakpoint Detection — identify where quality drops > 10%
    lines.append("## Context Breakpoint Detection")
    lines.append("")

    # Group by model and task, ordered by size
    breakpoints_found = False
    for alias in sorted(by_size.keys()):
        model_runs = by_size[alias]
        # Group by task_id and compute avg score per size
        task_scores: dict[str, dict[str, list[float]]] = {}
        for r in model_runs:
            tid = r.get("task_id", "unknown")
            size = r.get("context_tokens", "unknown")
            score = r.get("score_primary")
            if tid not in task_scores:
                task_scores[tid] = {}
            if score is not None:
                if size not in task_scores[tid]:
                    task_scores[tid][size] = []
                task_scores[tid][size].append(score)

        for tid in sorted(task_scores.keys()):
            size_scores = task_scores[tid]
            # Get ordered scores
            ordered_scores: list[tuple[str, float]] = []
            for size in size_order:
                if size in size_scores:
                    avg = sum(size_scores[size]) / max(len(size_scores[size]), 1)
                    ordered_scores.append((size, avg))

            # Detect breakpoints: consecutive size pairs where score drops > 10%
            for i in range(len(ordered_scores) - 1):
                curr_size, curr_score = ordered_scores[i]
                next_size, next_score = ordered_scores[i + 1]
                if curr_score > 0:
                    drop_pct = (curr_score - next_score) / curr_score
                    if drop_pct > 0.10:
                        breakpoints_found = True
                        lines.append(
                            f"- **{tid}** ({alias}): Quality drops {drop_pct:.0%} "
                            f"from {curr_size} ({curr_score:.3f}) to {next_size} ({next_score:.3f})"
                        )

    if not breakpoints_found:
        lines.append("No significant context breakpoints detected (>10% quality drop between consecutive sizes).")

    lines.append("")


def _append_quantization_comparison(lines: list[str], runs: list[dict[str, Any]]) -> None:
    """Append Quantization Comparison report sections if runs have quantization metadata.

    Generates full quantization comparison report when quantization field
    is detected in run data.

    Args:
        lines: List of markdown line strings to append to.
        runs: List of run result dicts.
    """
    quant_runs = [r for r in runs if r.get("quantization") is not None]
    if not quant_runs:
        return

    from bench_harness.reports.quantization_comparison import generate_quantization_report

    report = generate_quantization_report(runs)
    if report:
        lines.append("")
        lines.append(report)
        lines.append("")
