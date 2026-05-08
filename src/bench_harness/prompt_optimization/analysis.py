"""Style analysis — rank prompt styles per task family from benchmark results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis results from existing benchmark style comparison data."""

    family_rankings: dict[str, list[tuple[str, float, float]]] = field(
        default_factory=dict,
    )
    """Per-family rankings: family -> [(style, avg_score, margin_over_second)]"""

    best_style_overall: str = ""
    """Best style across all families (by avg score)."""

    best_style_by_family: dict[str, str] = field(default_factory=dict)
    """Mapping: task_family -> best_style_name."""

    style_variances: dict[str, float] = field(default_factory=dict)
    """Per-style score variance: style -> variance across all tasks."""

    insufficient_data: list[str] = field(default_factory=list)
    """Task families flagged as having too few style comparisons."""

    all_styles: list[str] = field(default_factory=list)
    """All unique styles found in the data."""

    total_style_runs: int = 0
    """Total number of style-tagged runs analyzed."""


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
    """Extract the sub-family from a task ID."""
    import re
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


def best_style_family(
    family_runs: list[dict[str, Any]],
    min_runs_per_style: int = 1,
) -> tuple[str | None, float, float]:
    """Determine the best prompt style for a set of runs (one task family).

    Args:
        family_runs: Run result dicts filtered to one task family.
        min_runs_per_style: Minimum runs per style to consider it valid.

    Returns:
        Tuple of (best_style_name, best_avg_score, margin_over_second).
        Returns (None, 0, 0) if no style has enough runs.
    """
    by_style: dict[str, list[float]] = {}
    for run in family_runs:
        style = run.get("prompt_style")
        score = run.get("score_primary")
        if style is None or score is None:
            continue
        if style not in by_style:
            by_style[style] = []
        by_style[style].append(score)

    # Filter to styles with enough runs
    scored_styles: dict[str, float] = {}
    for style, scores in by_style.items():
        if len(scores) >= min_runs_per_style:
            scored_styles[style] = _avg(scores)

    if not scored_styles:
        return None, 0, 0

    sorted_styles = sorted(scored_styles.items(), key=lambda x: x[1], reverse=True)
    best_style, best_score = sorted_styles[0]
    second_score = sorted_styles[1][1] if len(sorted_styles) > 1 else best_score
    margin = best_score - second_score

    return best_style, best_score, margin


def detect_insufficient_data(
    runs: list[dict[str, Any]],
    min_runs_per_style: int = 3,
) -> list[str]:
    """Detect task families with insufficient style comparison data.

    A family has insufficient data if any of the styles used on it
    has fewer than min_runs_per_style runs.

    Args:
        runs: Run result dicts (should all have prompt_style set).
        min_runs_per_style: Minimum runs per style to count as sufficient.

    Returns:
        List of family names with insufficient data.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        tid = run.get("task_id", "")
        family = _extract_family_from_task_id(tid)
        if family is None:
            continue
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(run)

    insufficient: list[str] = []
    for family, family_runs in by_family.items():
        by_style: dict[str, int] = {}
        for run in family_runs:
            style = run.get("prompt_style")
            if style is None:
                continue
            by_style[style] = by_style.get(style, 0) + 1

        has_insufficient = any(count < min_runs_per_style for count in by_style.values())
        if has_insufficient:
            insufficient.append(family)

    return insufficient


def analyze_style_data(
    runs: list[dict[str, Any]],
    suite_id: str = "",
    min_runs_per_style: int = 1,
) -> PromptAnalysis:
    """Analyze existing benchmark results to rank prompt styles per task family.

    Reads run data (from SQLite or any source), groups by family and style,
    computes averages, margins, and variances.

    Args:
        runs: List of run result dicts, each with prompt_style field.
        suite_id: If provided, filter to only this suite.
        min_runs_per_style: Minimum runs per style to consider it valid.

    Returns:
        PromptAnalysis with rankings, best styles, and variance data.
    """
    # Filter by suite if specified
    if suite_id:
        runs = [r for r in runs if r.get("suite_id") == suite_id]

    analysis = PromptAnalysis()
    analysis.total_style_runs = len(runs)

    if not runs:
        return analysis

    # Collect all styles
    style_set: set[str] = set()
    for run in runs:
        s = run.get("prompt_style")
        if s is not None:
            style_set.add(s)
    analysis.all_styles = sorted(style_set)

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

    # Compute per-family rankings
    for family, f_runs in sorted(family_runs.items()):
        best_style, best_score, margin = best_style_family(
            f_runs, min_runs_per_style=min_runs_per_style
        )
        if best_style:
            analysis.best_style_by_family[family] = best_style

        # Build full ranking for this family
        rankings: list[tuple[str, float, float]] = []
        by_style: dict[str, list[float]] = {}
        for run in f_runs:
            style = run.get("prompt_style")
            score = run.get("score_primary")
            if style is None or score is None:
                continue
            if style not in by_style:
                by_style[style] = []
            by_style[style].append(score)

        style_avgs = []
        for style, scores in by_style.items():
            avg_s = _avg(scores)
            style_avgs.append((style, avg_s, len(scores)))

        style_avgs.sort(key=lambda x: x[1], reverse=True)
        for i, (style, avg_s, count) in enumerate(style_avgs):
            second_score = style_avgs[i + 1][1] if i + 1 < len(style_avgs) else avg_s
            rankings.append((style, avg_s, avg_s - second_score))

        if rankings:
            analysis.family_rankings[family] = rankings

    # Compute overall best style
    all_by_style: dict[str, list[float]] = {}
    for run in runs:
        style = run.get("prompt_style")
        score = run.get("score_primary")
        if style is None or score is None:
            continue
        if style not in all_by_style:
            all_by_style[style] = []
        all_by_style[style].append(score)

    if all_by_style:
        style_avgs = [
            (style, _avg(scores)) for style, scores in all_by_style.items()
        ]
        style_avgs.sort(key=lambda x: x[1], reverse=True)
        analysis.best_style_overall = style_avgs[0][0]

    # Compute per-style variance across all tasks
    for style in analysis.all_styles:
        scores = all_by_style.get(style, [])
        analysis.style_variances[style] = _stddev(scores)

    # Detect insufficient data
    analysis.insufficient_data = detect_insufficient_data(runs, min_runs_per_style)

    return analysis
