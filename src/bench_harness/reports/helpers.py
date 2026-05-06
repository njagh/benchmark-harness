"""Utility functions for benchmark report generation."""

from __future__ import annotations

import math
from typing import Any


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
                return candidate
    parts = task_id.split("_")
    if len(parts) >= 2:
        candidate = parts[1]
        skip = {"factual", "json", "python", "debug", "code", "instruction"}
        if candidate.lower() not in skip:
            return candidate
    return None


def group_by_model(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group runs by model alias.

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping model alias to list of runs for that model.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        alias = run.get("model_alias", "unknown")
        if alias not in by_model:
            by_model[alias] = []
        by_model[alias].append(run)
    return by_model


def group_by_family(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group runs by task family.

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping family name to list of runs for that family.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        task_id = run.get("task_id", "unknown")
        family = _extract_family_from_task_id(task_id)
        if family is None:
            family = "other"
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(run)
    return by_family


def group_by_quantization(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group runs by quantization type.

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping quantization type to list of runs for that quantization.
    """
    by_quant: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        quant = run.get("quantization", "unquantized")
        if not quant or quant == "None":
            quant = "unquantized"
        if quant not in by_quant:
            by_quant[quant] = []
        by_quant[quant].append(run)
    return by_quant


def group_by_context_size(runs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group runs by context size (prompt_tokens).

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping context size bucket to list of runs.
    """
    by_context: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        ctx_size = run.get("prompt_tokens", 0)
        if ctx_size not in by_context:
            by_context[ctx_size] = []
        by_context[ctx_size].append(run)
    return by_context


def compute_score_variance_by_task(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute score variance for each task across models.

    Returns tasks sorted by variance descending — tasks with highest variance
    are the most discriminating (best for distinguishing model quality).

    Args:
        runs: List of run result dicts.

    Returns:
        List of dicts with task_id, score variance info, sorted by variance desc.
    """
    # Group by task
    by_task: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        task_id = run.get("task_id", "unknown")
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(run)

    results = []
    for task_id, task_runs in by_task.items():
        scored = [r for r in task_runs if r.get("score_primary") is not None]
        if len(scored) < 2:
            continue

        scores = [r["score_primary"] for r in scored]
        aliases = [r.get("model_alias", "unknown") for r in scored]

        # Find best and worst model for this task
        best_run = max(scored, key=lambda r: r["score_primary"])
        worst_run = min(scored, key=lambda r: r["score_primary"])

        variance = _stddev(scores)
        score_range = max(scores) - min(scores)

        results.append({
            "task_id": task_id,
            "variance": variance,
            "score_range": score_range,
            "best_model": best_run.get("model_alias", "unknown"),
            "best_score": best_run["score_primary"],
            "worst_model": worst_run.get("model_alias", "unknown"),
            "worst_score": worst_run["score_primary"],
            "num_models": len(aliases),
            "scores": scores,
        })

    results.sort(key=lambda x: x["variance"], reverse=True)
    return results


def find_pareto_frontier(
    runs: list[dict[str, Any]],
    score_key: str = "score_primary",
    speed_key: str = "tokens_per_second",
) -> list[dict[str, Any]]:
    """Find Pareto-optimal models based on score vs speed.

    A model is Pareto-optimal if no other model has both higher score
    and higher speed (tokens/sec).

    Args:
        runs: List of run result dicts.
        score_key: Key to use for quality score.
        speed_key: Key to use for speed metric.

    Returns:
        List of model info dicts on the Pareto frontier, sorted by score desc.
    """
    # Aggregate by model
    by_model: dict[str, list[dict[str, Any]]] = group_by_model(runs)

    candidates = []
    for alias, model_runs in by_model.items():
        scored = [r for r in model_runs if r.get(score_key) is not None]
        if not scored:
            continue

        avg_score = sum(r.get(score_key, 0) or 0 for r in scored) / len(scored)
        tps_vals = [r.get(speed_key, 0) for r in model_runs if r.get(speed_key, 0) > 0]
        avg_speed = _avg(tps_vals) if tps_vals else 0

        candidates.append({
            "model": alias,
            score_key: avg_score,
            speed_key: avg_speed,
            "tasks_run": len(scored),
        })

    if not candidates:
        return []

    # Pareto frontier: no other candidate dominates
    frontier = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            # b dominates a if b has both higher score and higher speed
            if (b[score_key] >= a[score_key] and b[speed_key] >= a[speed_key]
                    and (b[score_key] > a[score_key] or b[speed_key] > a[speed_key])):
                dominated = True
                break
        if not dominated:
            frontier.append(a)

    frontier.sort(key=lambda x: x[score_key], reverse=True)
    return frontier


def cluster_failures(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Cluster failed runs by error pattern.

    Groups runs with exit_status='error' into clusters based on
    error message patterns. Runs with no error message go into 'unknown'.

    Args:
        runs: List of run result dicts.

    Returns:
        Dict mapping cluster name to list of run dicts.
    """
    errors = [r for r in runs if r.get("exit_status") == "error"]
    if not errors:
        return {}

    clusters: dict[str, list[dict[str, Any]]] = {}

    for run in errors:
        error_msg = run.get("error_message", "") or ""
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")

        if not error_msg:
            cluster_name = "unknown errors"
        else:
            # Create cluster key from first 50 chars of error message
            # This groups similar errors together
            short_msg = error_msg.strip()[:80].replace("\n", " ").replace("\r", "")
            cluster_name = short_msg

        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append({
            "task_id": task_id,
            "model_alias": alias,
            "error_message": error_msg,
            "exit_status": run.get("exit_status"),
        })

    return clusters


def detect_regressions(
    new_runs: list[dict[str, Any]],
    old_runs: list[dict[str, Any]],
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect regressions by comparing new runs against prior runs.

    Groups by (task_id, model_alias) and compares average scores.

    Args:
        new_runs: Current run results.
        old_runs: Prior run results for comparison.
        tolerance: Minimum absolute change to flag (default 0.05).

    Returns:
        List of dicts with task_id, model, old_score, new_score, delta, status.
    """
    # Aggregate old runs by (task_id, model_alias)
    old_agg: dict[tuple[str, str], float] = {}
    for run in old_runs:
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")
        score = run.get("score_primary")
        if score is None:
            continue
        key = (task_id, alias)
        if key not in old_agg:
            old_agg[key] = []
        old_agg[key].append(score)

    old_avgs: dict[tuple[str, str], float] = {}
    for key, scores in old_agg.items():
        old_avgs[key] = sum(scores) / len(scores)

    # Aggregate new runs by (task_id, model_alias)
    new_agg: dict[tuple[str, str], float] = {}
    for run in new_runs:
        task_id = run.get("task_id", "unknown")
        alias = run.get("model_alias", "unknown")
        score = run.get("score_primary")
        if score is None:
            continue
        key = (task_id, alias)
        if key not in new_agg:
            new_agg[key] = []
        new_agg[key].append(score)

    new_avgs: dict[tuple[str, str], float] = {}
    for key, scores in new_agg.items():
        new_avgs[key] = sum(scores) / len(scores)

    # Compare
    regressions = []
    for key in new_avgs:
        if key not in old_avgs:
            continue
        old_score = old_avgs[key]
        new_score = new_avgs[key]
        delta = new_score - old_score

        if abs(delta) < tolerance:
            continue

        task_id, alias = key
        if delta < 0:
            status = "REGRESSION"
        else:
            status = "IMPROVED"

        regressions.append({
            "task_id": task_id,
            "model_alias": alias,
            "old_score": old_score,
            "new_score": new_score,
            "delta": delta,
            "status": status,
        })

    regressions.sort(key=lambda x: x["delta"])
    return regressions
