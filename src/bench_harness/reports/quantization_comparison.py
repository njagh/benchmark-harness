"""Quantization comparison report generator.

Generates markdown reports comparing quantized vs non-quantized (or between
different quantization) models across the same tasks. Identifies which task
families are most sensitive to quantization.
"""

from __future__ import annotations

import math
import re
from typing import Any


def generate_quantization_report(
    runs: list[dict[str, Any]],
    suite_id: str = "",
) -> str:
    """Generate a quantization comparison report.

    Sections:
    1. Quantization Summary — per quantization: avg score, avg wall, avg tok/s, tasks
    2. Quality Delta — score difference between quantized and FP16/FP8 variants
    3. Best Quantized Model by Family — which quantization performs best per task family
    4. Speed/Quality Frontier — pareto front of quality vs throughput per quantization
    5. Sensitivity Analysis — which task families are most/least sensitive to quantization

    Args:
        runs: List of run result dicts, each with a 'quantization' field.
        suite_id: Suite identifier for the report header.

    Returns:
        Markdown report string.
    """
    quant_runs: dict[str | None, list[dict[str, Any]]] = {}
    for run in runs:
        q = run.get("quantization")
        if q is None:
            continue
        if q not in quant_runs:
            quant_runs[q] = []
        quant_runs[q].append(run)

    if not quant_runs:
        return ""

    # Filter: require at least 2 quantization levels for comparison
    if len(quant_runs) < 2:
        # Still generate a summary with a single quantization level
        lines: list[str] = []
        lines.append("## Quantization Analysis")
        lines.append("")
        lines.append(
            f"**Note:** Only one quantization level detected ({', '.join(quant_runs.keys())}). "
            f"Comparison requires at least two quantization levels (e.g., FP16 vs INT4 vs FP8)."
        )
        lines.append("")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("## Quantization Comparison Report")
    lines.append("")

    # Section 1: Quantization Summary
    _append_quant_summary(lines, quant_runs)

    # Section 2: Quality Delta
    _append_quality_delta(lines, quant_runs)

    # Section 3: Best Quantized Model by Family
    _append_best_quant_family(lines, runs, quant_runs)

    # Section 4: Speed/Quality Frontier
    _append_speed_quality_frontier(lines, quant_runs)

    # Section 5: Sensitivity Analysis
    _append_sensitivity_analysis(lines, runs, quant_runs)

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


def _p95(values: list[float]) -> float:
    """Compute P95 of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * 0.95)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _extract_family_from_task_id(task_id: str) -> str | None:
    """Extract the sub-family from a task ID."""
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


def _is_reference_quantization(q: str | None) -> bool:
    """Check if a quantization label represents a reference (high-quality) baseline.

    FP16, BF16, and None (missing quantization metadata) are treated as
    reference configurations — the highest quality variants to compare against.
    """
    if q is None:
        return True
    upper = q.upper().strip()
    return upper in ("FP16", "BF16", "FP32", "NATIVE", "NONE", "FLOAT16", "FLOAT32", "NO_QUANTIZATION")


def _append_quant_summary(
    lines: list[str],
    quant_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Quantization Summary section."""
    lines.append("### Quantization Summary")
    lines.append("")
    lines.append(
        "| Quantization | Tasks Run | Avg Score | "
        "Avg Wall (ms) | Avg Tok/s | P95 Wall (ms) | Std Dev Score |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for q in sorted(quant_runs.keys()):
        qruns = quant_runs[q]
        total = len(qruns)
        scored = [r for r in qruns if r.get("score_primary") is not None]
        avg_score = (
            sum(r.get("score_primary", 0) or 0 for r in scored) / max(len(scored), 1)
        )
        walls = [r.get("total_wall_ms", 0) for r in qruns]
        avg_wall = _avg(walls)
        p95_wall = _p95(walls)

        tps_vals = [r.get("tokens_per_second", 0) for r in qruns if r.get("tokens_per_second", 0) > 0]
        avg_tps = _avg(tps_vals)

        scores = _score_values(qruns)
        score_std = _stddev(scores) if scored else 0.0

        is_ref = _is_reference_quantization(q)
        label = f"**{q}**" if is_ref else q
        lines.append(
            f"| {label} | {total} | {avg_score:.3f} "
            f"| {avg_wall:.0f} | {avg_tps:.1f} "
            f"| {p95_wall:.0f} | {score_std:.4f} |"
        )

    lines.append("")


def _append_quality_delta(
    lines: list[str],
    quant_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Quality Delta section — score differences between quantizations."""
    lines.append("### Quality Delta (Score Difference from Reference)")
    lines.append("")

    # Find reference quantization (FP16/BF16/None)
    reference_q = None
    for q in quant_runs:
        if _is_reference_quantization(q):
            reference_q = q
            break

    # If no reference found, use the highest-scoring quantization as baseline
    if reference_q is None:
        scores_by_q: dict[str, float] = {}
        for q, qruns in quant_runs.items():
            scored = [r for r in qruns if r.get("score_primary") is not None]
            if scored:
                scores_by_q[q] = _avg(_score_values(scored))
            else:
                scores_by_q[q] = 0.0
        reference_q = max(scores_by_q, key=scores_by_q.get)

    ref_runs = quant_runs.get(reference_q, [])
    ref_avg = _avg(_score_values(ref_runs)) if ref_runs else 0.0

    lines.append(
        f"Baseline: `{reference_q}` (avg score: {ref_avg:.3f})"
    )
    lines.append("")
    lines.append(
        "| Quantization | Avg Score | Delta (pts) | Delta (%) | Status |"
    )
    lines.append("|---|---|---|---|---|")

    for q in sorted(quant_runs.keys()):
        if q == reference_q:
            continue
        qruns = quant_runs[q]
        q_avg = _avg(_score_values(qruns)) if qruns else 0.0
        delta = q_avg - ref_avg
        delta_pct = (delta / max(ref_avg, 0.001)) * 100 if ref_avg != 0 else 0.0

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
            f"| {q} | {q_avg:.3f} | {sign}{delta:.3f} "
            f"| {sign}{delta_pct:.1f}% | {status} |"
        )

    lines.append("")
    lines.append(
        "**Status legend:** Negligible ≤ 0.05pts, Minor ≤ 0.15pts, "
        "Significant > 0.15pts"
    )
    lines.append("")


def _append_best_quant_family(
    lines: list[str],
    runs: list[dict[str, Any]],
    quant_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Best Quantized Model by Family section."""
    lines.append("### Best Quantization by Task Family")
    lines.append("")

    # Group runs by family
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

    lines.append("| Task Family | Best Quantization | Avg Score | All Quantizations |")
    lines.append("|---|---|---|---|")

    for family in sorted(family_runs.keys()):
        f_runs = family_runs[family]

        # Group by quantization within family
        by_quant: dict[str | None, list[dict[str, Any]]] = {}
        for r in f_runs:
            q = r.get("quantization")
            if q is None:
                continue
            if q not in by_quant:
                by_quant[q] = []
            by_quant[q].append(r)

        best_q = ""
        best_score = -1
        all_q_info: dict[str, float] = {}

        for q, qruns in sorted(by_quant.items()):
            scored = [r for r in qruns if r.get("score_primary") is not None]
            if not scored:
                continue
            avg_s = _avg(_score_values(qruns))
            all_q_info[q] = avg_s
            if avg_s > best_score:
                best_score = avg_s
                best_q = q

        if best_q:
            q_list = ", ".join(f"{k}({v:.2f})" for k, v in sorted(all_q_info.items()))
            lines.append(
                f"| {family} | {best_q} | {best_score:.3f} | {q_list} |"
            )

    lines.append("")


def _append_speed_quality_frontier(
    lines: list[str],
    quant_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Speed/Quality Frontier — Pareto front of quality vs throughput."""
    lines.append("### Speed/Quality Frontier")
    lines.append("")
    lines.append(
        "Plotting quality (avg score) vs throughput (tokens/sec) per quantization. "
        "Points on the Pareto front offer the best trade-off."
    )
    lines.append("")
    lines.append("| Quantization | Avg Score | Avg Tok/s | Pareto Optimal |")
    lines.append("|---|---|---|---|")

    # Compute metrics per quantization
    q_metrics: list[dict[str, Any]] = []
    for q, qruns in quant_runs.items():
        scored = [r for r in qruns if r.get("score_primary") is not None]
        if not scored:
            continue

        avg_score = _avg(_score_values(qruns))
        tps_vals = [r.get("tokens_per_second", 0) for r in qruns if r.get("tokens_per_second", 0) > 0]
        avg_tps = _avg(tps_vals)

        q_metrics.append({
            "q": q,
            "score": avg_score,
            "tps": avg_tps,
        })

    if not q_metrics:
        lines.append("No scored data available for frontier analysis.")
        lines.append("")
        return

    # Identify Pareto front (maximize both score and tps)
    max_score = max(m["score"] for m in q_metrics) if q_metrics else 0
    max_tps = max(m["tps"] for m in q_metrics) if q_metrics else 0

    pareto_points = set()
    for i, m in enumerate(q_metrics):
        is_pareto = True
        for j, other in enumerate(q_metrics):
            if i == j:
                continue
            if other["score"] >= m["score"] and other["tps"] >= m["tps"]:
                if other["score"] > m["score"] or other["tps"] > m["tps"]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.add(i)

    q_metrics.sort(key=lambda x: x["score"], reverse=True)

    for idx, m in enumerate(q_metrics):
        is_pareto = any(
            idx == i for i, mi in enumerate(q_metrics) if i in pareto_points
        )
        # Re-check using the sorted list
        is_pareto = any(
            m["q"] == other["q"] for i, other in enumerate(q_metrics)
            if i in pareto_points
        )
        label = "Yes" if is_pareto else "No"
        lines.append(
            f"| {m['q']} | {m['score']:.3f} | {m['tps']:.1f} | {label} |"
        )

    lines.append("")
    lines.append(
        "**Pareto optimal:** A quantization is on the frontier if no other "
        "quantization simultaneously achieves both higher score and higher throughput."
    )
    lines.append("")


def _append_sensitivity_analysis(
    lines: list[str],
    runs: list[dict[str, Any]],
    quant_runs: dict[str, list[dict[str, Any]]],
) -> None:
    """Append Sensitivity Analysis — which task families are most/least sensitive."""
    lines.append("### Sensitivity Analysis by Task Family")
    lines.append("")
    lines.append(
        "Measures how much each task family's score drops when moving from "
        "reference quantization to quantized variants. Larger drops = more sensitive."
    )
    lines.append("")

    # Find reference quantization
    reference_q = None
    for q in quant_runs:
        if _is_reference_quantization(q):
            reference_q = q
            break

    if reference_q is None:
        lines.append("No reference (FP16/BF16) quantization found for sensitivity comparison.")
        lines.append("")
        return

    ref_runs = quant_runs.get(reference_q, [])
    if not ref_runs:
        lines.append(f"No runs found for reference quantization `{reference_q}`.")
        lines.append("")
        return

    # Group reference runs by family
    ref_by_family: dict[str, list[float]] = {}
    for r in ref_runs:
        tid = r.get("task_id", "")
        family = _extract_family_from_task_id(tid)
        if family is None:
            continue
        score = r.get("score_primary")
        if score is not None:
            if family not in ref_by_family:
                ref_by_family[family] = []
            ref_by_family[family].append(score)

    # For each quantization, compute score drop per family
    lines.append("| Task Family | Reference Score | "
                 "Quantized Score | Score Drop | Sensitivity |")
    lines.append("|---|---|---|---|---|")

    sensitivity_data: list[dict[str, Any]] = []

    for family in sorted(ref_by_family.keys()):
        ref_scores = ref_by_family[family]
        ref_avg = _avg(ref_scores)

        # Find the largest score drop across all quantizations
        max_drop = 0.0
        worst_q = ""
        worst_q_score = ref_avg

        for q, qruns in quant_runs.items():
            if q == reference_q:
                continue
            family_q = [r for r in qruns if _extract_family_from_task_id(r.get("task_id", "")) == family]
            if not family_q:
                continue
            q_avg = _avg(_score_values(family_q))
            drop = ref_avg - q_avg
            if drop > max_drop:
                max_drop = drop
                worst_q = q
                worst_q_score = q_avg

        if max_drop > 0:
            if max_drop <= 0.05:
                sens = "Low"
            elif max_drop <= 0.15:
                sens = "Medium"
            elif max_drop <= 0.30:
                sens = "High"
            else:
                sens = "Critical"
        else:
            sens = "None"

        sensitivity_data.append({
            "family": family,
            "ref_avg": ref_avg,
            "drop": max_drop,
            "sensitivity": sens,
        })

        lines.append(
            f"| {family} | {ref_avg:.3f} | {worst_q_score:.3f} "
            f"| {max_drop:.3f} | {sens} |"
        )

    lines.append("")

    # Rank families by sensitivity
    if sensitivity_data:
        lines.append("#### Sensitivity Rankings")
        lines.append("")
        lines.append("| Rank | Task Family | Score Drop | Sensitivity |")
        lines.append("|---|---|---|---|")

        sorted_sensitivity = sorted(
            sensitivity_data, key=lambda x: x["drop"], reverse=True
        )

        for rank, sd in enumerate(sorted_sensitivity, 1):
            lines.append(
                f"| {rank} | {sd['family']} | {sd['drop']:.3f} | {sd['sensitivity']} |"
            )

        lines.append("")

        # Top sensitive family
        top = sorted_sensitivity[0]
        lines.append(
            f"**Most sensitive:** `{top['family']}` loses {top['drop']:.3f} points on average. "
        )

        # Least sensitive family
        bottom = sorted_sensitivity[-1]
        lines.append(
            f"**Least sensitive:** `{bottom['family']}` loses only {bottom['drop']:.3f} points."
        )
        lines.append("")
