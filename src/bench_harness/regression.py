"""Quick regression suite generator for benchmark runs.

Analyzes run history from a benchmark database to automatically select
the most discriminating or failing tasks, producing a compact YAML suite
for fast regression testing before model/backend changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from bench_harness.reports.helpers import (
    compute_score_variance_by_task,
    detect_regressions,
)
from bench_harness.export.base import get_runs_by_suite, get_task_by_id

logger = logging.getLogger(__name__)


def _get_failed_tasks(
    runs: list[dict[str, Any]],
) -> set[str]:
    """Return set of task_ids that have at least one failure.

    A task is considered failed if any run has exit_status == 'error'
    or score_primary == 0 (or 0.0).

    Args:
        runs: List of run dicts from the database.

    Returns:
        Set of task_ids with failures.
    """
    failed: set[str] = set()
    for run in runs:
        exit_status = run.get("exit_status", "")
        score_primary = run.get("score_primary")

        if exit_status == "error":
            failed.add(run.get("task_id", ""))
        elif score_primary is not None and float(score_primary) == 0.0:
            failed.add(run.get("task_id", ""))

    return failed


def _get_high_variance_tasks(
    runs: list[dict[str, Any]],
    max_tasks: int = 10,
) -> list[str]:
    """Select tasks with highest score variance across models.

    Args:
        runs: List of run dicts.
        max_tasks: Maximum number of task IDs to return.

    Returns:
        List of task_id strings, sorted by variance descending.
    """
    variance_results = compute_score_variance_by_task(runs)
    return [r["task_id"] for r in variance_results[:max_tasks]]


def _get_regressed_tasks(
    runs: list[dict[str, Any]],
    max_tasks: int = 10,
) -> list[str]:
    """Select tasks that show regression (no multi-model comparison available).

    Uses detect_regressions as a fallback when there's only a single model's
    data, selecting tasks with the lowest scores.

    Args:
        runs: List of run dicts.
        max_tasks: Maximum number of task IDs to return.

    Returns:
        List of task_id strings that show regressions or low scores.
    """
    regressions = detect_regressions(runs, [], tolerance=0.0)

    if regressions:
        return [r["task_id"] for r in regressions[:max_tasks]]

    # Fallback: pick tasks with lowest average score
    by_task: dict[str, list[float]] = {}
    for run in runs:
        task_id = run.get("task_id", "")
        score = run.get("score_primary")
        if score is not None:
            by_task.setdefault(task_id, []).append(float(score))

    task_avgs = []
    for task_id, scores in by_task.items():
        avg = sum(scores) / len(scores)
        task_avgs.append((task_id, avg))

    task_avgs.sort(key=lambda x: x[1])
    return [t[0] for t in task_avgs[:max_tasks]]


def generate_regression_suite(
    db_path: str,
    out_dir: str,
    max_tasks: int = 10,
) -> str:
    """Generate a YAML regression suite from benchmark run history.

    Selects tasks using two criteria:
    1. Tasks with the highest score variance across models (most discriminating).
    2. Tasks that have failed (exit_status == 'error' or score_primary == 0).

    The selected tasks are written to ``_regression_suite.yaml`` in the
    specified output directory, in the same YAML format as task files
    in the ``tasks/`` directory.

    Args:
        db_path: Path to the benchmark SQLite database.
        out_dir: Directory to write the regression suite YAML file.
        max_tasks: Maximum number of variance-based tasks to include
            (in addition to all failed tasks).

    Returns:
        Path to the generated regression suite YAML file.
    """
    from bench_harness.storage.sqlite import SQLiteStore

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_file = out_path / "_regression_suite.yaml"

    # Load all runs (no suite filter)
    store = SQLiteStore(db_path)
    runs = store.get_runs()

    if not runs:
        logger.warning("No runs found in %s", db_path)
        with open(output_file, "w") as f:
            yaml.dump([], f, default_flow_style=False, sort_keys=False)
        return str(output_file)

    # Identify failed tasks (always included)
    failed_tasks = _get_failed_tasks(runs)

    # Identify high-variance tasks
    high_var_tasks = _get_high_variance_tasks(runs, max_tasks)

    # If no multi-model variance data available, try regression detection
    if not high_var_tasks:
        high_var_tasks = _get_regressed_tasks(runs, max_tasks)

    # Union of failed and high-variance tasks
    selected_task_ids: set[str] = set(high_var_tasks) | failed_tasks

    # Build suite entries
    suite: list[dict[str, Any]] = []

    for task_id in selected_task_ids:
        task = get_task_by_id(db_path, task_id)

        # Gather failure info for this task
        task_runs = [r for r in runs if r.get("task_id") == task_id]
        failures = []
        for run in task_runs:
            exit_status = run.get("exit_status", "")
            score_primary = run.get("score_primary")

            if exit_status == "error" or (
                score_primary is not None and float(score_primary) == 0.0
            ):
                failure_entry: dict[str, Any] = {
                    "model": run.get("model_alias", "unknown"),
                    "exit_status": exit_status,
                }
                error_message = run.get("error_message")
                if error_message:
                    failure_entry["error"] = error_message
                if score_primary is not None:
                    failure_entry["score"] = score_primary
                failures.append(failure_entry)

        entry: dict[str, Any] = {
            "id": task_id,
            "family": task.get("family", "unknown") if task else "unknown",
        }

        if task:
            entry["category"] = task.get("category")
            entry["scoring"] = task.get("scoring", {})
            entry["expected"] = task.get("expected", {})

        # Include failure info if this task has any failures
        if failures:
            entry["failures"] = failures

        suite.append(entry)

    # Sort: failed tasks first, then by variance ranking
    failed_set = set()
    for task_id in failed_tasks:
        failed_set.add(task_id)

    def sort_key(entry: dict[str, Any]) -> tuple[bool, str]:
        tid = entry["id"]
        is_failed = tid in failed_set
        # Failed tasks sort first (False < True when negated)
        return (not is_failed, tid)

    suite.sort(key=sort_key)

    with open(output_file, "w") as f:
        yaml.dump(
            suite,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info(
        "Generated regression suite with %d tasks -> %s",
        len(suite),
        output_file,
    )
    return str(output_file)
