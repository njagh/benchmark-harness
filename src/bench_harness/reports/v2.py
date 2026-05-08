"""Comprehensive v2 report generator for benchmark results.

Provides modular, configurable report generation with sections for:
- Executive Summary
- Model Comparison
- Best by Family
- Speed/Quality Frontier
- Context Analysis
- Quantization Comparison
- Style Analysis
- Judge Analysis
- Failure Analysis
- Regression Detection
- Discriminating Tasks
"""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any

from bench_harness.reports.helpers import (
    _avg,
    _stddev,
    _extract_family_from_task_id,
    group_by_model,
    group_by_family,
    group_by_quantization,
    group_by_context_size,
    compute_score_variance_by_task,
    find_pareto_frontier,
    cluster_failures,
    detect_regressions,
)


def _get_score_for_model(
    model_runs: list[dict[str, Any]],
) -> float:
    """Compute average primary score for a model's runs."""
    scored = [r for r in model_runs if r.get("score_primary") is not None]
    if not scored:
        return 0.0
    return sum(r["score_primary"] for r in scored) / len(scored)


def _get_speed_for_model(
    model_runs: list[dict[str, Any]],
) -> float:
    """Compute average tokens/sec for a model's runs."""
    tps_vals = [r.get("tokens_per_second", 0) for r in model_runs if r.get("tokens_per_second", 0) > 0]
    if not tps_vals:
        return 0.0
    return _avg(tps_vals)


def _get_context_for_model(
    model_runs: list[dict[str, Any]],
) -> int:
    """Get the max prompt_tokens seen for a model (proxy for context size)."""
    ctx_vals = [r.get("prompt_tokens", 0) for r in model_runs]
    return max(ctx_vals) if ctx_vals else 0


def _get_quantization_for_model(
    model_runs: list[dict[str, Any]],
) -> str:
    """Get the most common quantization for a model's runs."""
    quant_counts: dict[str, int] = {}
    for r in model_runs:
        q = r.get("quantization", "") or ""
        if q:
            quant_counts[q] = quant_counts.get(q, 0) + 1
    if quant_counts:
        return max(quant_counts, key=quant_counts.get)
    return ""


def _format_delta(delta: float) -> str:
    """Format a score delta with sign."""
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.3f}"


def _format_safe_pct(safe_count: int, total: int) -> str:
    """Format a safety percentage."""
    if total == 0:
        return "N/A"
    pct = (safe_count / total) * 100
    return f"{pct:.0f}%"


def _append_executive_summary(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append the Executive Summary section.

    Answers: which model is best overall, fastest, most efficient, safest,
    best long-context, and best quantized.
    """
    lines.append("## Executive Summary")
    lines.append("")

    best_score_model = ""
    best_score_val = -1.0
    fastest_model = ""
    fastest_tps = 0.0
    most_efficient_model = ""
    most_efficient_tps = 0.0
    safest_model = ""
    safest_pct = -1.0
    best_context_model = ""
    best_context_size = 0
    best_quantized_model = ""
    best_quantized_loss = 100.0

    # Find reference (highest score) model
    for alias, runs in model_stats.items():
        score = _get_score_for_model(runs)
        if score > best_score_val:
            best_score_val = score
            best_score_model = alias

    # Find fastest and most efficient
    for alias, runs in model_stats.items():
        tps = _get_speed_for_model(runs)
        if tps > fastest_tps:
            fastest_tps = tps
            fastest_model = alias
        if tps > most_efficient_tps:
            most_efficient_tps = tps
            most_efficient_model = alias

    # Find safest (lowest error rate)
    for alias, runs in model_stats.items():
        total = len(runs)
        safe = sum(1 for r in runs if r.get("exit_status") == "success")
        if total > 0:
            pct = (safe / total) * 100
            if pct > safest_pct:
                safest_pct = pct
                safest_model = alias

    # Find best context size
    for alias, runs in model_stats.items():
        ctx = _get_context_for_model(runs)
        if ctx > best_context_size:
            best_context_size = ctx
            best_context_model = alias

    # Find best quantized (smallest quality loss from reference)
    quant_groups = group_by_quantization(
        [r for runs in model_stats.values() for r in runs]
    )
    if len(quant_groups) >= 2:
        # Find reference quantization (highest avg score)
        ref_q = ""
        ref_score = -1.0
        for q, qruns in quant_groups.items():
            qscore = _get_score_for_model(qruns)
            if qscore > ref_score:
                ref_score = qscore
                ref_q = q

        # Find quantized with smallest loss
        for q, qruns in quant_groups.items():
            if q == ref_q:
                continue
            qscore = _get_score_for_model(qruns)
            loss = ref_score - qscore
            if loss < best_quantized_loss:
                best_quantized_loss = loss
                best_quantized_model = q

    # Build summary table
    lines.append("| Metric | Winner |")
    lines.append("|---|---|")
    lines.append(f"| Best Overall Score | **{best_score_model}** ({best_score_val:.3f}) |")

    if fastest_tps > 0:
        lines.append(f"| Fastest | {fastest_model} ({fastest_tps:.1f} tok/s) |")
    if most_efficient_tps > 0:
        lines.append(f"| Most Efficient | {most_efficient_model} ({most_efficient_tps:.1f} tok/s) |")
    if safest_pct > 0:
        lines.append(f"| Safest | {safest_model} ({_format_safe_pct(0, 0).replace('N/A', str(int(safest_pct)) + '%')}) |")

    if best_context_model:
        ctx_label = f"{best_context_size:,} tokens" if best_context_size >= 1000 else str(best_context_size)
        lines.append(f"| Best Long-Context | {best_context_model} ({ctx_label} prompt) |")

    if best_quantized_model and best_quantized_loss < 100:
        lines.append(
            f"| Best Quantized | {best_quantized_model} "
            f"({best_quantized_loss:.1%} quality loss) |"
        )

    lines.append("")


def _append_model_comparison(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append the Model Comparison cross-ranking table."""
    lines.append("## Model Comparison")
    lines.append("")

    entries = []
    for alias, runs in model_stats.items():
        score = _get_score_for_model(runs)
        tps = _get_speed_for_model(runs)
        ctx = _get_context_for_model(runs)
        quant = _get_quantization_for_model(runs)
        avg_wall = sum(r.get("total_wall_ms", 0) for r in runs) / max(len(runs), 1)

        entries.append({
            "alias": alias,
            "score": score,
            "wall_ms": avg_wall,
            "tps": tps,
            "context": ctx,
            "quantization": quant,
            "tasks_run": len(runs),
        })

    entries.sort(key=lambda x: x["score"], reverse=True)

    lines.append(
        "| Rank | Model | Score | Wall (ms) | Tok/s | Context | Quantization |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for rank, e in enumerate(entries, 1):
        ctx_label = ""
        if e["context"] >= 1000:
            ctx_label = f"{e['context'] // 1000}k"
        elif e["context"] > 0:
            ctx_label = str(e["context"])
        else:
            ctx_label = "—"

        quant_label = e["quantization"] if e["quantization"] else "—"

        lines.append(
            f"| {rank} | {e['alias']} | {e['score']:.3f} "
            f"| {e['wall_ms']:.0f} | {e['tps']:.1f} "
            f"| {ctx_label} | {quant_label} |"
        )

    lines.append("")


def _append_best_by_family(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append the Best Model by Task Family section."""
    lines.append("## Best Model by Task Family")
    lines.append("")

    # Flatten all runs
    all_runs = [r for runs in model_stats.values() for r in runs]
    by_family = group_by_family(all_runs)

    if not by_family:
        lines.append("No family data available.")
        lines.append("")
        return

    lines.append("| Family | Best Model | Score | Margin |")
    lines.append("|---|---|---|---|")

    for family in sorted(by_family.keys()):
        family_runs = by_family[family]
        by_model_in_family: dict[str, list[dict[str, Any]]] = {}
        for r in family_runs:
            alias = r.get("model_alias", "unknown")
            if alias not in by_model_in_family:
                by_model_in_family[alias] = []
            by_model_in_family[alias].append(r)

        if len(by_model_in_family) < 2:
            continue

        best_alias = ""
        best_score = -1.0
        second_alias = ""
        second_score = -1.0

        for alias, runs in by_model_in_family.items():
            score = _get_score_for_model(runs)
            if score > best_score:
                second_alias = best_alias
                second_score = best_score
                best_score = score
                best_alias = alias
            elif score > second_score:
                second_score = score
                second_alias = alias

        margin = best_score - second_score
        lines.append(
            f"| {family} | {best_alias} | {best_score:.3f} | {_format_delta(margin)} |"
        )

    lines.append("")


def _append_speed_quality_frontier(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Speed/Quality Frontier section with Pareto-optimal models."""
    lines.append("## Speed/Quality Frontier")
    lines.append("")

    all_runs = [r for runs in model_stats.values() for r in runs]
    frontier = find_pareto_frontier(all_runs)

    if not frontier:
        lines.append("No scored data available for frontier analysis.")
        lines.append("")
        return

    lines.append(
        "Models on the Pareto frontier offer the best trade-off between "
        "quality (score) and speed (tokens/sec). No other model has both "
        "higher quality and higher speed."
    )
    lines.append("")
    lines.append("| Model | Score | Tok/s | Pareto Optimal |")
    lines.append("|---|---|---|---|")

    # Sort all models by score
    all_models = []
    for alias, runs in model_stats.items():
        score = _get_score_for_model(runs)
        tps = _get_speed_for_model(runs)
        is_pareto = any(f["model"] == alias for f in frontier)
        all_models.append({
            "alias": alias,
            "score": score,
            "tps": tps,
            "pareto": is_pareto,
        })

    all_models.sort(key=lambda x: x["score"], reverse=True)

    for m in all_models:
        pareto_label = "Yes *" if m["pareto"] else "No"
        lines.append(
            f"| {m['alias']} | {m['score']:.3f} | {m['tps']:.1f} | {pareto_label} |"
        )

    lines.append("")
    lines.append("* Pareto-optimal: no other model has both higher score and higher tok/s.")
    lines.append("")


def _append_context_analysis(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Context Analysis — quality vs context size."""
    lines.append("## Context Analysis")
    lines.append("")

    all_runs = [r for runs in model_stats.values() for r in runs]
    by_context = group_by_context_size(all_runs)

    if len(by_context) < 2:
        lines.append(
            "Not enough context size variation for analysis. "
            "Need runs with different prompt sizes."
        )
        lines.append("")
        return

    lines.append("| Context Size (tokens) | Score | Tok/s | Tasks | Std Dev |")
    lines.append("|---|---|---|---|---|")

    for ctx_size in sorted(by_context.keys()):
        ctx_runs = by_context[ctx_size]
        scored = [r for r in ctx_runs if r.get("score_primary") is not None]
        if not scored:
            continue
        avg_score = _get_score_for_model(scored)
        tps = _avg([r.get("tokens_per_second", 0) for r in ctx_runs if r.get("tokens_per_second", 0) > 0])
        scores = [r["score_primary"] for r in scored]
        std = _stddev(scores)
        lines.append(
            f"| {ctx_size:,} | {avg_score:.3f} | {tps:.1f} | {len(scored)} | {std:.4f} |"
        )

    lines.append("")


def _append_quantization_comparison(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Quantization Comparison section."""
    all_runs = [r for runs in model_stats.values() for r in runs]
    quant_groups = group_by_quantization(all_runs)

    if len(quant_groups) < 2:
        lines.append(
            "## Quantization Analysis"
        )
        lines.append("")
        lines.append(
            "Only one quantization level detected. Comparison requires "
            "at least two (e.g., FP16 vs INT4 vs FP8)."
        )
        lines.append("")
        return

    lines.append("## Quantization Comparison")
    lines.append("")

    # Find reference
    ref_q = ""
    ref_score = -1.0
    for q, qruns in quant_groups.items():
        qs = _get_score_for_model(qruns)
        if qs > ref_score:
            ref_score = qs
            ref_q = q

    lines.append(f"Reference: `{ref_q}` (avg score: {ref_score:.3f})")
    lines.append("")
    lines.append("| Quantization | Avg Score | Delta (pts) | Delta (%) | Status |")
    lines.append("|---|---|---|---|---|")

    for q in sorted(quant_groups.keys()):
        if q == ref_q:
            continue
        qruns = quant_groups[q]
        qs = _get_score_for_model(qruns)
        delta = qs - ref_score
        delta_pct = (delta / max(ref_score, 0.001)) * 100 if ref_score != 0 else 0.0

        if delta >= 0:
            status = "Equal or better"
        elif abs(delta) <= 0.05:
            status = "Negligible"
        elif abs(delta) <= 0.15:
            status = "Minor"
        else:
            status = "Significant"

        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {q} | {qs:.3f} | {sign}{delta:.3f} | {sign}{delta_pct:.1f}% | {status} |"
        )

    lines.append("")
    lines.append("**Status legend:** Negligible ≤ 0.05pts, Minor ≤ 0.15pts, Significant > 0.15pts")
    lines.append("")


def _append_style_analysis(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Style Analysis section for runs with prompt_style."""
    all_runs = [r for runs in model_stats.values() for r in runs]
    style_runs = [r for r in all_runs if r.get("prompt_style") is not None]

    if not style_runs:
        return

    from bench_harness.reports.style_comparison import generate_style_report
    report = generate_style_report(style_runs)
    if report:
        lines.append("## Style Analysis")
        lines.append("")
        lines.append(report)
        lines.append("")


def _append_prompt_optimization(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Prompt Optimization section if optimization data exists."""
    # Check for optimization results in the run data
    all_runs = [r for runs in model_stats.values() for r in runs]

    # Look for optimization suite IDs (contain "-optimization-")
    opt_runs = [r for r in all_runs if "-optimization-" in (r.get("suite_id", "") or "")]

    if not opt_runs:
        return

    # Group by suite_id to find optimization runs
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for r in opt_runs:
        sid = r.get("suite_id", "")
        if sid not in by_suite:
            by_suite[sid] = []
        by_suite[sid].append(r)

    for suite_id, suite_runs in sorted(by_suite.items()):
        # Compute candidate summaries
        by_style: dict[str, list[dict[str, Any]]] = {}
        for r in suite_runs:
            style = r.get("prompt_style", "")
            if style not in by_style:
                by_style[style] = []
            by_style[style].append(r)

        # Find baselines and candidates
        candidate_names = set()
        baseline_names: dict[str, float] = {}
        for style, sruns in by_style.items():
            scored = [r for r in sruns if r.get("score_primary") is not None]
            if scored:
                avg_score = sum(r["score_primary"] for r in scored) / len(scored)
                if style.endswith("-optimization-"):
                    continue  # Skip the suite ID itself
                # Simple heuristic: if the style matches a known candidate pattern
                # (anything not in the predefined list is a candidate)
                known_styles = {"plain", "repl", "terse", "patch_only", "architect", "json_schema", "step_by_step"}
                if style not in known_styles:
                    candidate_names.add(style)
                else:
                    baseline_names[style] = avg_score

        if not candidate_names:
            continue

        lines.append("## Prompt Optimization")
        lines.append("")
        lines.append(f"Suite: `{suite_id}`")
        lines.append("")

        # Identify baseline (default: plain)
        plain_score = baseline_names.get("plain", 0)

        lines.append("| Candidate | Avg Score | vs Plain |")
        lines.append("|---|---|---|")
        for candidate in sorted(candidate_names):
            c_runs = by_style.get(candidate, [])
            scored = [r for r in c_runs if r.get("score_primary") is not None]
            if scored:
                c_score = sum(r["score_primary"] for r in scored) / len(scored)
                delta = c_score - plain_score
                lines.append(f"| {candidate} | {c_score:.3f} | {delta:+.3f} |")

        lines.append("")

        # Recommendations
        recommended = []
        for candidate in candidate_names:
            c_runs = by_style.get(candidate, [])
            scored = [r for r in c_runs if r.get("score_primary") is not None]
            if scored:
                c_score = sum(r["score_primary"] for r in scored) / len(scored)
                if c_score - plain_score > 0.05:
                    recommended.append(candidate)

        if recommended:
            lines.append(f"**Recommended:** {', '.join(f'`{r}`' for r in recommended)} "
                         f"score above plain baseline ({plain_score:.3f}).")
        else:
            lines.append("**Note:** No candidate exceeded the 0.05 improvement threshold over plain.")
        lines.append("")


def _append_judge_analysis(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append LLM Judge Analysis section."""
    all_runs = [r for runs in model_stats.values() for r in runs]
    judge_runs = [r for r in all_runs if r.get("judge_score") is not None]

    if not judge_runs:
        return

    lines.append("## Judge Analysis")
    lines.append("")
    lines.append("| Model | Judge Score | Tasks Judged | Dimensions |")
    lines.append("|---|---|---|---|")

    for alias, runs in sorted(model_stats.items()):
        model_judge = [r for r in runs if r.get("judge_score") is not None]
        if not model_judge:
            continue
        avg_judge = sum(r["judge_score"] for r in model_judge) / len(model_judge)

        # Collect dimensions
        dims: dict[str, list[float]] = {}
        for r in model_judge:
            dim_data = r.get("judge_dimensions") or {}
            if isinstance(dim_data, str):
                try:
                    import json
                    dim_data = json.loads(dim_data)
                except (json.JSONDecodeError, ValueError):
                    dim_data = {}
            for k, v in dim_data.items():
                if k not in dims:
                    dims[k] = []
                try:
                    dims[k].append(float(v))
                except (ValueError, TypeError):
                    pass

        dim_str = ", ".join(f"{k}={_avg(v):.2f}" for k, v in sorted(dims.items())) or "-"
        lines.append(
            f"| {alias} | {avg_judge:.3f} | {len(model_judge)} | {dim_str} |"
        )

    lines.append("")


def _append_failure_analysis(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Failure Clustering section with safety flag detection."""
    all_runs = [r for runs in model_stats.values() for r in runs]
    clusters = cluster_failures(all_runs)

    if not clusters:
        return

    lines.append("## Failure Analysis")
    lines.append("")
    lines.append("Failed runs grouped by error pattern:")
    lines.append("")

    # Detect tasks with safety flags
    safety_flagged_runs = [
        r for r in all_runs
        if r.get("safety_score") is not None and r["safety_score"] < 1.0
    ]

    if safety_flagged_runs:
        lines.append("### Safety Flags")
        lines.append("")
        lines.append(
            f"{len(safety_flagged_runs)} run(s) flagged for unsafe commands "
            f"(safety_score < 1.0):"
        )
        lines.append("")
        for sr in safety_flagged_runs[:20]:
            alias = sr.get("model_alias", "unknown")
            task_id = sr.get("task_id", "unknown")
            score = sr.get("safety_score", 0.0)
            lines.append(
                f"- `{task_id}` on `{alias}` — safety_score: **{score:.4f}**"
            )
        lines.append("")

    for cluster_name, cluster_runs in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        # Truncate long cluster names
        display_name = cluster_name[:60]
        if len(cluster_name) > 60:
            display_name += "..."
        lines.append(f"- **{display_name}** ({len(cluster_runs)} run(s))")
        for cr in cluster_runs:
            alias = cr.get("model_alias", "unknown")
            task_id = cr.get("task_id", "unknown")
            safety = cr.get("safety_score")
            if safety is not None and safety < 1.0:
                lines.append(
                    f"  - `{task_id}` on `{alias}` [safety_score={safety:.4f}]"
                )
            else:
                lines.append(f"  - `{task_id}` on `{alias}`")

    lines.append("")


def _append_regression_detection(
    lines: list[str],
    new_runs: list[dict[str, Any]],
    old_runs: list[dict[str, Any]],
) -> None:
    """Append Regression Detection section comparing new vs prior runs."""
    if not old_runs:
        return

    regressions = detect_regressions(new_runs, old_runs)

    if not regressions:
        lines.append("## Regression Detection")
        lines.append("")
        lines.append("No significant regressions or improvements detected.")
        lines.append("")
        return

    lines.append("## Regression Detection")
    lines.append("")
    lines.append(
        f"Comparing current runs against prior run. "
        f"Threshold for detection: 0.05 points."
    )
    lines.append("")
    lines.append(
        "| Task | Model | Old Score | New Score | Delta | Status |"
    )
    lines.append("|---|---|---|---|---|---|")

    for r in regressions:
        status_marker = "REGRESSION" if r["delta"] < 0 else "IMPROVED"
        lines.append(
            f"| {r['task_id']} | {r['model_alias']} "
            f"| {r['old_score']:.3f} | {r['new_score']:.3f} "
            f"| {_format_delta(r['delta'])} | **{status_marker}** |"
        )

    lines.append("")


def _append_discriminating_tasks(
    lines: list[str],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Most Discriminating Tasks section."""
    all_runs = [r for runs in model_stats.values() for r in runs]
    variance_tasks = compute_score_variance_by_task(all_runs)

    if not variance_tasks:
        return

    lines.append("## Most Discriminating Tasks")
    lines.append("")
    lines.append(
        "Tasks where models differ most. These tasks are best at "
        "distinguishing between model capabilities."
    )
    lines.append("")
    lines.append("| Task | Score Range | Best Model | Best Score | Worst Model | Worst Score | Gap |")
    lines.append("|---|---|---|---|---|---|---|")

    for task in variance_tasks[:15]:
        lines.append(
            f"| {task['task_id']} "
            f"| {task['score_range']:.3f} "
            f"| {task['best_model']} "
            f"| {task['best_score']:.3f} "
            f"| {task['worst_model']} "
            f"| {task['worst_score']:.3f} "
            f"| {task['score_range']:.3f} |"
        )

    lines.append("")


def _append_public_baseline_scores(
    lines: list[str],
    runs: list[dict[str, Any]],
    model_stats: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Public Baseline Scores section for lm_eval results.

    Shows standard benchmark scores (MMLU, GPQA, BBH, MATH) alongside
    local task scores, with a clear disclaimer.
    """
    # Filter for public benchmark runs (task_id starts with "public.")
    public_runs = [r for r in runs if r.get("task_id", "").startswith("public.")]

    if not public_runs:
        lines.append("## Public Baseline Scores")
        lines.append("")
        lines.append("_No public benchmark results available. Run with `bench-harness run-lm-eval` to populate._")
        lines.append("")
        lines.append(
            "> These are public benchmark scores (may not reflect real workflow performance). "
            "They measure general capability on standardized tasks, not specific workflow quality."
        )
        lines.append("")
        return

    lines.append("## Public Baseline Scores")
    lines.append("")
    lines.append(
        "> These are public benchmark scores (may not reflect real workflow performance). "
        "They measure general capability on standardized tasks, not specific workflow quality."
    )
    lines.append("")

    # Group public runs by model
    public_by_model: dict[str, list[dict[str, Any]]] = {}
    for r in public_runs:
        alias = r.get("model_alias", "unknown")
        if alias not in public_by_model:
            public_by_model[alias] = []
        public_by_model[alias].append(r)

    lines.append("| Task | Model | Accuracy | Samples |")
    lines.append("|---|---|---|---|")

    for alias in sorted(public_by_model.keys()):
        model_runs = public_by_model[alias]
        for r in sorted(model_runs, key=lambda x: x.get("task_id", "")):
            task_id = r.get("task_id", "unknown")
            # Remove "public." prefix for display
            display_task = task_id.replace("public.", "", 1)
            score = r.get("score_primary")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            samples = r.get("score_secondary")
            if isinstance(samples, str):
                try:
                    import json
                    samples = json.loads(samples)
                except (json.JSONDecodeError, TypeError):
                    samples = {}
            if isinstance(samples, dict):
                samples_str = samples.get("samples_run", "?")
            else:
                samples_str = "?"
            lines.append(
                f"| {display_task} | {alias} | {score_str} | {samples_str} |"
            )

    lines.append("")


def _append_identity_stamp(lines: list[str], runs: list[dict[str, Any]]) -> None:
    """Append model identity verification section.

    Shows alias vs actual served model, revealing when a model alias maps
    to a different model on the server (e.g., 'qwen-dense served as agent-code').
    """
    # Collect runs with identity stamp data
    identity_runs = [r for r in runs if r.get("openai_models_id") or r.get("vllm_served_model_name")]
    if not identity_runs:
        lines.append("## Model Identity Stamp")
        lines.append("")
        lines.append("_No identity stamp data available (model not queried via /v1/models)._")
        lines.append("")
        return

    lines.append("## Model Identity Stamp")
    lines.append("")
    lines.append(
        "Before each task, the harness calls `/v1/models` to verify which model "
        "the backend actually serves. This reveals when an alias (e.g., `qwen-dense`) "
        "maps to a different model on the server."
    )
    lines.append("")

    # Group by (model_alias, openai_models_id) to show alias vs actual mapping
    from collections import defaultdict
    alias_vs_actual: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in identity_runs:
        alias = r.get("model_alias", "unknown")
        actual_id = r.get("openai_models_id") or r.get("vllm_served_model_name") or "(none)"
        alias_vs_actual[alias][actual_id] += 1

    lines.append("| Alias | Actual Model ID | Served Model Name | Container | HF Model ID | Backend URL | Runs |")
    lines.append("|---|---|---|---|---|---|---|")
    for alias in sorted(alias_vs_actual.keys()):
        for actual_id, count in sorted(alias_vs_actual[alias].items()):
            # Get other fields from a matching run
            sample_run = next(r for r in identity_runs if r.get("model_alias") == alias and r.get("openai_models_id") == actual_id)
            served = sample_run.get("vllm_served_model_name") or ""
            container = sample_run.get("vllm_container_name") or ""
            hf = sample_run.get("hf_model_id") or ""
            url = sample_run.get("backend_url") or ""
            lines.append(
                f"| {alias} "
                f"| {actual_id} "
                f"| {served} "
                f"| {container} "
                f"| {hf} "
                f"| {url} "
                f"| {count} |"
            )
    lines.append("")

    # Highlight mismatches
    mismatches = []
    for alias, actuals in alias_vs_actual.items():
        for actual_id in actuals:
            if actual_id != alias:
                mismatches.append((alias, actual_id))

    if mismatches:
        lines.append("### ⚠️ Alias Mismatches Detected")
        lines.append("")
        lines.append("The following aliases do not match the actual model served:")
        lines.append("")
        for alias, actual in mismatches:
            lines.append(f"- **{alias}** → served as **{actual}**")
        lines.append("")


def generate_report_v2(
    runs: list[dict[str, Any]],
    suite_id: str,
    models_config: dict[str, Any],
    out_path: str,
    sections: list[str] | None = None,
    prior_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a comprehensive v2 report with configurable sections.

    Args:
        runs: List of run result dicts.
        suite_id: Suite identifier.
        models_config: Model config dict.
        out_path: Output file path.
        sections: Which sections to include (default: all). Options:
            - "executive_summary" — top-line: which model is best overall
            - "model_comparison" — cross-model ranking table
            - "best_by_family" — best model per task family
            - "fastest" — speed/quality frontier
            - "context_analysis" — quality vs context size
            - "quantization_comparison" — quantization impact
            - "style_analysis" — prompt style comparison
            - "judge_analysis" — LLM judge results
            - "failure_analysis" — failure clustering
            - "regression_detection" — compare against prior run
            - "discriminating_tasks" — tasks with most score variance across models
        prior_runs: Optional list of run dicts from a previous run for regression detection.

    Returns:
        The markdown string.
    """
    lines: list[str] = []

    # Header
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = platform.node()

    lines.append(f"# Benchmark Report v2: {suite_id}")
    lines.append("")
    lines.append(f"Date: {now}")
    lines.append(f"Host: {host}")
    lines.append("")

    # Model info
    model_info = models_config.get("models", {})
    if model_info:
        lines.append("## Models")
        lines.append("")
        lines.append("| Alias | Backend | Quantization | Notes |")
        lines.append("|---|---|---|---|")
        for alias, mconfig in sorted(model_info.items()):
            backend = mconfig.get("backend", "unknown")
            quant = mconfig.get("quantization", "unknown")
            notes = mconfig.get("notes", "")
            lines.append(f"| {alias} | {backend} | {quant} | {notes} |")
        lines.append("")

    model_stats = group_by_model(runs)

    default_sections = [
        "executive_summary",
        "model_comparison",
        "best_by_family",
        "fastest",
        "context_analysis",
        "quantization_comparison",
        "style_analysis",
        "prompt_optimization",
        "judge_analysis",
        "failure_analysis",
        "regression_detection",
        "discriminating_tasks",
        "public_baseline",
        "identity_stamp",
    ]

    active_sections = sections if sections is not None else default_sections

    section_map: dict[str, Any] = {
        "executive_summary": lambda: _append_executive_summary(lines, model_stats),
        "model_comparison": lambda: _append_model_comparison(lines, model_stats),
        "best_by_family": lambda: _append_best_by_family(lines, model_stats),
        "fastest": lambda: _append_speed_quality_frontier(lines, model_stats),
        "context_analysis": lambda: _append_context_analysis(lines, model_stats),
        "quantization_comparison": lambda: _append_quantization_comparison(lines, model_stats),
        "style_analysis": lambda: _append_style_analysis(lines, model_stats),
        "prompt_optimization": lambda: _append_prompt_optimization(lines, model_stats),
        "judge_analysis": lambda: _append_judge_analysis(lines, model_stats),
        "failure_analysis": lambda: _append_failure_analysis(lines, model_stats),
        "regression_detection": lambda: _append_regression_detection(
            lines, runs, prior_runs or []
        ),
        "discriminating_tasks": lambda: _append_discriminating_tasks(lines, model_stats),
        "public_baseline": lambda: _append_public_baseline_scores(lines, runs, model_stats),
        "identity_stamp": lambda: _append_identity_stamp(lines, runs),
    }

    for section_name in active_sections:
        if section_name in section_map:
            section_map[section_name]()

    # Write to file
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    report_str = "\n".join(lines)
    return report_str

