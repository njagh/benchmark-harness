"""Prompt optimization report generator.

Generates markdown reports for the prompt optimization section of benchmark
results, including analysis summary, best styles per family, score variance,
candidate results, and recommendations.
"""

from __future__ import annotations

from typing import Any

from bench_harness.prompt_optimization.analysis import PromptAnalysis


def generate_optimization_report(
    analysis: PromptAnalysis,
    candidate_results: list[dict[str, Any]],
    suite_id: str = "",
) -> str:
    """Generate a full markdown optimization report.

    Args:
        analysis: PromptAnalysis from existing data.
        candidate_results: Results from candidate evaluation.
        suite_id: Suite identifier for the report header.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("## Prompt Optimization Report")
    lines.append("")

    if suite_id:
        lines.append(f"**Suite:** `{suite_id}`")
        lines.append("")

    # Section 1: Analysis Summary
    lines.append("### Analysis Summary")
    lines.append("")
    lines.append(f"- **Total runs analyzed:** {analysis.total_style_runs}")
    lines.append(f"- **Styles found:** {', '.join(analysis.all_styles) if analysis.all_styles else 'none'}")
    best = analysis.best_style_overall or "N/A"
    lines.append(f"- **Best style overall:** {best}")
    lines.append("")

    if analysis.insufficient_data:
        lines.append(
            f"> :warning: **Insufficient data** (< 3 runs per style) for: "
            f"{', '.join(analysis.insufficient_data)}"
        )
        lines.append("")

    # Section 2: Best Styles Per Family
    if analysis.family_rankings:
        lines.append("### Best Styles Per Task Family")
        lines.append("")
        lines.append("| Task Family | Best Style | Avg Score | Margin |")
        lines.append("|---|---|---|---|")
        for family, rankings in sorted(analysis.family_rankings.items()):
            if rankings:
                best_style, avg_score, margin = rankings[0]
                lines.append(
                    f"| {family} | {best_style} | {avg_score:.3f} | {margin:+.3f} |"
                )
        lines.append("")

    # Section 3: Style Score Variance
    if analysis.style_variances:
        lines.append("### Score Variance by Style")
        lines.append("")
        lines.append(
            "Higher variance means the style's performance is more task-dependent. "
            "Lower variance indicates consistent performance across tasks."
        )
        lines.append("")
        lines.append("| Style | Variance |")
        lines.append("|---|---|")
        for style, var in sorted(analysis.style_variances.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {style} | {var:.4f} |")
        lines.append("")

    # Section 4: Candidate Results
    if candidate_results:
        lines.append("### Candidate Template Results")
        lines.append("")

        complete = [r for r in candidate_results if r.get("status") == "complete"]
        analyzed = [r for r in candidate_results if r.get("status") == "analyzed"]
        no_data = [r for r in candidate_results if r.get("status") == "no_data"]
        baselines = [r for r in candidate_results if r.get("status") == "baseline"]

        if complete:
            lines.append("#### Live Run Results")
            lines.append("")
            lines.append("| Candidate | Score | Baseline | Delta | Runs |")
            lines.append("|---|---|---|---|---|")
            for r in sorted(complete, key=lambda x: x.get("score_delta", 0), reverse=True):
                delta_str = f"{r['score_delta']:+.3f}" if r.get("score_delta") is not None else "N/A"
                baseline_str = f"{r.get('baseline_score', ''):.3f}" if r.get('baseline_score') is not None else "-"
                lines.append(
                    f"| {r['name']} | {r.get('score', 0):.3f} | {baseline_str} "
                    f"({r.get('baseline', '-')}) | {delta_str} | {r.get('run_count', 0)} |"
                )
            lines.append("")

            # Recommendations
            recommended = [r for r in complete if r.get("score_delta", 0) > 0.05]
            if recommended:
                lines.append("**Recommendations:**")
                lines.append("")
                for r in recommended:
                    lines.append(
                        f"- `{r['name']}` scores **{r['score_delta']:+.3f}** vs "
                        f"baseline `{r['baseline']}` — {r.get('instructions', '')}"
                    )
                lines.append("")
            else:
                lines.append(
                    "**Note:** No candidates exceeded the 0.05 improvement threshold. "
                    "Current baselines are performing well."
                )
                lines.append("")

        if analyzed:
            lines.append("#### Analyzed from Existing Data")
            lines.append("")
            lines.append("| Candidate | Score | Baseline | Delta | Runs |")
            lines.append("|---|---|---|---|---|")
            for r in sorted(analyzed, key=lambda x: x.get("score_delta", 0), reverse=True):
                delta_str = f"{r['score_delta']:+.3f}" if r.get("score_delta") is not None else "N/A"
                baseline_str = f"{r.get('baseline_score', 0):.3f}" if r.get('baseline_score') is not None else "N/A"
                lines.append(
                    f"| {r['name']} | {r['score']:.3f} | {baseline_str} "
                    f"({r.get('baseline', '-')}) | {delta_str} | {r.get('run_count', 0)} |"
                )
            lines.append("")

        if no_data:
            lines.append("#### Candidates with No Data")
            lines.append("")
            for r in no_data:
                lines.append(f"- `{r['name']}` — no runs found for this style")
            lines.append("")

        if baselines:
            lines.append("#### Baseline Styles")
            lines.append("")
            lines.append("| Style | Avg Score | Runs |")
            lines.append("|---|---|---|")
            for r in sorted(baselines, key=lambda x: x.get("score", 0), reverse=True):
                lines.append(
                    f"| {r['name']} | {r.get('score', 0):.3f} | {r.get('run_count', 0)} |"
                )
            lines.append("")

    # Section 5: Best Style by Family Recommendations
    if analysis.best_style_by_family:
        lines.append("### Recommended Style by Family")
        lines.append("")
        lines.append("Based on analysis of existing benchmark data:")
        lines.append("")
        for family, style in sorted(analysis.best_style_by_family.items()):
            lines.append(f"- **{family}:** `{style}`")
        lines.append("")

    return "\n".join(lines)


def generate_optimization_summary(
    analysis: PromptAnalysis,
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a structured JSON-serializable summary of optimization results.

    Args:
        analysis: PromptAnalysis from existing data.
        candidate_results: Results from candidate evaluation.

    Returns:
        Dict with summary fields suitable for JSON output.
    """
    summary: dict[str, Any] = {
        "total_runs": analysis.total_style_runs,
        "styles_found": analysis.all_styles,
        "best_style_overall": analysis.best_style_overall,
        "best_style_by_family": analysis.best_style_by_family,
        "style_variances": analysis.style_variances,
        "insufficient_data": analysis.insufficient_data,
        "family_rankings": {
            family: [
                {"style": s, "avg_score": avg, "margin": m}
                for s, avg, m in rankings
            ]
            for family, rankings in analysis.family_rankings.items()
        },
        "candidates": [],
    }

    for r in candidate_results:
        candidate_entry: dict[str, Any] = {
            "name": r.get("name", ""),
            "baseline": r.get("baseline"),
            "score": r.get("score"),
            "score_delta": r.get("score_delta"),
            "run_count": r.get("run_count", 0),
            "status": r.get("status", ""),
        }
        if r.get("baseline_score") is not None:
            candidate_entry["baseline_score"] = r["baseline_score"]
        if r.get("instructions"):
            candidate_entry["instructions"] = r["instructions"]
        summary["candidates"].append(candidate_entry)

    # Determine overall recommendation
    recommended = [r for r in candidate_results if r.get("score_delta") is not None and r["score_delta"] > 0.05]
    if recommended:
        summary["recommendation"] = {
            "action": "adopt",
            "candidates": [r["name"] for r in recommended],
        }
    elif candidate_results:
        summary["recommendation"] = {
            "action": "no_change",
            "reason": "No candidates exceeded the 0.05 improvement threshold",
        }
    else:
        summary["recommendation"] = {
            "action": "insufficient_data",
            "reason": "No candidate results available for evaluation",
        }

    return summary
