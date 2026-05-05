"""Style comparison report generator.

Generates markdown reports comparing prompt styles (repl, terse, patch_only,
architect, json_schema, step_by_step, plain) across the same tasks.
"""

from __future__ import annotations

import re
from typing import Any


def generate_style_report(
    runs: list[dict[str, Any]],
    suite_id: str = "",
    styles: list[str] | None = None,
) -> str:
    """Generate a markdown report comparing prompt styles.

    Groups runs by prompt_style field and compares across six sections:
    1. Style Comparison Summary — per style: avg score, avg tokens, avg wall time, task count
    2. Per-Task Style Breakdown — for each task, compare all styles side by side
    3. Best Style Per Family — which style performs best per task family
    4. Verbosity Analysis — token usage per style (response length vs quality)
    5. Latency Comparison — TTFT and decode time per style
    6. Recommended Style — overall recommendation based on score/token ratio

    Args:
        runs: List of run result dicts, each with a 'prompt_style' field.
        suite_id: Suite identifier for the report header.
        styles: Optional list of style names to include (defaults to all found).

    Returns:
        Markdown report string.
    """
    style_runs: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        style = run.get("prompt_style")
        if style is None:
            continue
        if style not in style_runs:
            style_runs[style] = []
        style_runs[style].append(run)

    if not style_runs:
        return ""

    # Filter to requested styles if specified
    if styles:
        style_runs = {s: v for s, v in style_runs.items() if s in styles}
        if not style_runs:
            return ""

    lines: list[str] = []
    lines.append("## Style Comparison Report")
    lines.append("")

    # Section 1: Style Comparison Summary
    _append_style_summary(lines, style_runs)

    # Section 2: Per-Task Style Breakdown
    _append_per_task_breakdown(lines, runs, style_runs)

    # Section 3: Best Style Per Family
    _append_best_style_family(lines, runs, style_runs)

    # Section 4: Verbosity Analysis
    _append_verbosity_analysis(lines, style_runs)

    # Section 5: Latency Comparison
    _append_latency_comparison(lines, style_runs)

    # Section 6: Recommended Style
    _append_recommended_style(lines, style_runs)

    return "\n".join(lines)


def _avg(values: list[float]) -> float:
    """Compute average of a list of values, returning 0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _extract_family_from_task_id(task_id: str) -> str | None:
    """Extract the sub-family from a task ID.

    For task IDs like 'local.docker_compose.fix_yaml_001', returns 'docker_compose'.
    """
    parts = task_id.split(".")
    if len(parts) >= 3:
        candidate = parts[1]
        skip = {"smoke", "local", "public", "synthetic"}
        if candidate.lower() not in skip:
            return candidate
    if len(parts) >= 2:
        candidate = parts[1]
        skip = {"factual", "json", "python", "debug", "instruction"}
        if candidate.lower() not in skip:
            if "_" not in candidate or len(parts) < 3:
                # Only return if candidate doesn't look like a task name
                # (task names typically have format: name_number, e.g. "fix_001")
                if not re.match(r"^[a-zA-Z]+_\d+$", candidate):
                    return candidate
    parts = task_id.split("_")
    if len(parts) >= 2:
        candidate = parts[1]
        skip = {"factual", "json", "python", "debug", "code", "instruction"}
        if candidate.lower() not in skip:
            return candidate
    return None


def _score_values(runs: list[dict[str, Any]]) -> list[float]:
    """Extract score_primary values from runs."""
    return [
        r.get("score_primary", 0) or 0
        for r in runs
        if r.get("score_primary") is not None
    ]


def _append_style_summary(
    lines: list[str],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Style Comparison Summary section."""
    lines.append("### Style Comparison Summary")
    lines.append("")
    lines.append(
        "| Style | Tasks Run | Avg Score | Avg Completion Toks | "
        "Avg Wall Time (ms) | Avg TTFT (ms) | Avg TPS |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for style in sorted(style_runs.keys()):
        sruns = style_runs[style]
        total = len(sruns)
        scored = [r for r in sruns if r.get("score_primary") is not None]
        avg_score = (
            sum(r.get("score_primary", 0) or 0 for r in scored) / max(len(scored), 1)
        )
        comp_tokens = [
            r.get("completion_tokens", 0) for r in sruns if r.get("completion_tokens")
        ]
        avg_comp = _avg(comp_tokens)
        wall_times = [r.get("total_wall_ms", 0) for r in sruns]
        avg_wall = _avg(wall_times)
        ttfts = [r.get("ttft_ms", 0) for r in sruns]
        avg_ttft = _avg(ttfts)
        tps_vals = [r.get("tokens_per_second", 0) for r in sruns if r.get("tokens_per_second", 0) > 0]
        avg_tps = _avg(tps_vals)
        lines.append(
            f"| {style} | {total} | {avg_score:.3f} | {avg_comp:.0f} "
            f"| {avg_wall:.0f} | {avg_ttft:.0f} | {avg_tps:.1f} |"
        )

    lines.append("")


def _append_per_task_breakdown(
    lines: list[str],
    runs: list[dict[str, Any]],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Per-Task Style Breakdown section."""
    lines.append("### Per-Task Style Breakdown")
    lines.append("")

    # Group runs by task_id
    task_runs: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        tid = run.get("task_id", "unknown")
        if tid not in task_runs:
            task_runs[tid] = []
        task_runs[tid].append(run)

    # Only show tasks where multiple styles were used
    multi_style_tasks = {
        tid: sruns
        for tid, sruns in task_runs.items()
        if len({r.get("prompt_style") for r in sruns if r.get("prompt_style")}) > 1
    }

    if not multi_style_tasks:
        lines.append("No tasks were run with multiple styles. This section is empty.")
        lines.append("")
        return

    for task_id in sorted(multi_style_tasks.keys()):
        sruns = multi_style_tasks[task_id]
        lines.append(f"#### {task_id}")
        lines.append("")
        lines.append("| Style | Score | Status | Completion Toks | Wall (ms) |")
        lines.append("|---|---|---|---|---|")

        for style in sorted({r.get("prompt_style") for r in sruns if r.get("prompt_style")}):
            style_task_runs = [r for r in sruns if r.get("prompt_style") == style]
            if not style_task_runs:
                continue
            avg_score = (
                sum(r.get("score_primary", 0) or 0 for r in style_task_runs)
                / max(len(style_task_runs), 1)
            )
            status = style_task_runs[0].get("exit_status", "unknown")
            comp_tok = style_task_runs[0].get("completion_tokens", 0)
            wall = style_task_runs[0].get("total_wall_ms", 0)
            lines.append(
                f"| {style} | {avg_score:.3f} | {status} | {comp_tok} | {wall:.0f} |"
            )

        lines.append("")


def _append_best_style_family(
    lines: list[str],
    runs: list[dict[str, Any]],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Best Style Per Family section."""
    lines.append("### Best Style Per Task Family")
    lines.append("")

    # Group by family
    family_runs: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        tid = run.get("task_id", "")
        family = _extract_family_from_task_id(tid)
        if family is None:
            continue
        if family not in family_runs:
            family_runs[family] = []
        family_runs[family].append(run)

    if not family_runs:
        lines.append("No family data available.")
        lines.append("")
        return

    lines.append("| Task Family | Best Style | Avg Score | Tasks |")
    lines.append("|---|---|---|---|")

    for family in sorted(family_runs.keys()):
        f_runs = family_runs[family]
        # Group by style within family
        by_style: dict[str, list[dict[str, Any]]] = {}
        for r in f_runs:
            s = r.get("prompt_style")
            if s is None:
                continue
            if s not in by_style:
                by_style[s] = []
            by_style[s].append(r)

        best_style = ""
        best_score = -1
        best_count = 0
        for style, sruns in sorted(by_style.items()):
            scored = [r for r in sruns if r.get("score_primary") is not None]
            if not scored:
                continue
            avg_s = sum(r.get("score_primary", 0) or 0 for r in scored) / len(scored)
            if avg_s > best_score:
                best_score = avg_s
                best_style = style
                best_count = len(scored)

        if best_style:
            lines.append(
                f"| {family} | {best_style} | {best_score:.3f} | {best_count} |"
            )
        else:
            lines.append(f"| {family} | - | N/A | 0 |")

    lines.append("")


def _append_verbosity_analysis(
    lines: list[str],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Verbosity Analysis section."""
    lines.append("### Verbosity Analysis")
    lines.append("")
    lines.append("Comparing token usage vs quality across styles:")
    lines.append("")
    lines.append("| Style | Avg Prompt Toks | Avg Completion Toks | Score/Tok Ratio | Quality/1k Toks |")
    lines.append("|---|---|---|---|---|")

    for style in sorted(style_runs.keys()):
        sruns = style_runs[style]
        prompt_toks = [r.get("prompt_tokens", 0) for r in sruns]
        comp_toks = [r.get("completion_tokens", 0) for r in sruns]
        scored = [r for r in sruns if r.get("score_primary") is not None]

        avg_prompt = _avg(prompt_toks)
        avg_comp = _avg(comp_toks)
        total_tok = avg_prompt + avg_comp

        if scored:
            avg_score = _score_values(scored)
            score_avg = _avg(avg_score)
            ratio = score_avg / max(total_tok, 1)
            per_k = score_avg / max(total_tok / 1000.0, 0.001)
        else:
            ratio = 0
            per_k = 0

        lines.append(
            f"| {style} | {avg_prompt:.0f} | {avg_comp:.0f} | "
            f"{ratio:.5f} | {per_k:.3f} |"
        )

    lines.append("")
    lines.append("**Note:** Higher quality-per-1k-tokens is better. "
                 "This metric favors concise, high-quality responses over verbose ones.")
    lines.append("")


def _append_latency_comparison(
    lines: list[str],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Latency Comparison section."""
    lines.append("### Latency Comparison")
    lines.append("")
    lines.append(
        "| Style | Avg TTFT (ms) | Avg Decode (ms) | "
        "Avg Wall Time (ms) | P95 Wall Time (ms) |"
    )
    lines.append("|---|---|---|---|---|")

    for style in sorted(style_runs.keys()):
        sruns = style_runs[style]
        ttfts = [r.get("ttft_ms", 0) for r in sruns]
        decodes = [r.get("decode_ms", 0) for r in sruns]
        walls = sorted([r.get("total_wall_ms", 0) for r in sruns])

        avg_ttft = _avg(ttfts)
        avg_decode = _avg(decodes)
        avg_wall = _avg(walls)

        # P95
        p95_idx = int(len(walls) * 0.95)
        p95_wall = walls[min(p95_idx, len(walls) - 1)] if walls else 0

        lines.append(
            f"| {style} | {avg_ttft:.0f} | {avg_decode:.0f} | "
            f"{avg_wall:.0f} | {p95_wall:.0f} |"
        )

    lines.append("")
    lines.append("**TTFT:** Time to first token. Lower is better for interactivity.")
    lines.append("**Decode:** Generation time. Lower is better for throughput.")
    lines.append("")


def _append_recommended_style(
    lines: list[str],
    style_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Recommended Style section based on score/token ratio."""
    lines.append("### Recommended Style")
    lines.append("")
    lines.append("Based on score-per-token ratio (quality efficiency):")
    lines.append("")

    # Calculate score/token ratio for each style
    style_rankings = []
    for style, sruns in style_runs.items():
        scored = [r for r in sruns if r.get("score_primary") is not None]
        if not scored:
            continue

        avg_score = _avg(_score_values(scored))
        total_tokens = sum(
            (r.get("prompt_tokens", 0) or 0) + (r.get("completion_tokens", 0) or 0)
            for r in sruns
        )
        avg_total = total_tokens / max(len(sruns), 1)
        ratio = avg_score / max(avg_total, 1)

        style_rankings.append({
            "style": style,
            "avg_score": avg_score,
            "avg_total_tokens": avg_total,
            "ratio": ratio,
        })

    if not style_rankings:
        lines.append("No scored runs available to make a recommendation.")
        lines.append("")
        return

    style_rankings.sort(key=lambda x: x["ratio"], reverse=True)

    lines.append("| Rank | Style | Avg Score | Score/Token Ratio |")
    lines.append("|---|---|---|---|")

    for rank, info in enumerate(style_rankings, 1):
        lines.append(
            f"| {rank} | {info['style']} | {info['avg_score']:.3f} | "
            f"{info['ratio']:.5f} |"
        )

    lines.append("")

    # Recommendation text
    top = style_rankings[0]
    lines.append(
        f"**Recommended:** `{top['style']}` offers the best score-per-token ratio "
        f"({top['ratio']:.5f}), balancing quality and efficiency. "
    )

    # Compare top style vs plain
    plain_runs = style_runs.get("plain", [])
    if plain_runs:
        plain_scored = [r for r in plain_runs if r.get("score_primary") is not None]
        if plain_scored:
            plain_score = _avg(_score_values(plain_scored))
            if top["style"] != "plain":
                diff = top["avg_score"] - plain_score
                direction = "better" if diff > 0 else "worse"
                lines.append(
                    f"Compared to plain style ({plain_score:.3f} avg), "
                    f"`{top['style']}` is {direction} by {abs(diff):.3f} points.\n"
                )
            else:
                lines.append(
                    f"Plain style is the recommended baseline.\n"
                )

    lines.append("")
