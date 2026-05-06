"""Regression export — YAML format for failed runs."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from bench_harness.export.base import (
    get_runs_by_suite,
    get_task_by_id,
)

logger = logging.getLogger(__name__)


def _resolve_output_path(out_path: str | None) -> Path:
    """Resolve output path, defaulting to exports/regression_tasks.yaml."""
    if out_path is None:
        return Path("exports") / "regression_tasks.yaml"
    p = Path(out_path)
    if p.is_dir():
        return p / "regression_tasks.yaml"
    return p


def export_regression(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
) -> str:
    """Export failed runs as a regression test suite in YAML.

    Groups failed runs by task_id. Excludes runs where raw_response is empty
    and error_message is empty (API errors that should not become regression tests).

    Output format:
    [
        {
            "task_id": "task_001",
            "family": "docker_compose",
            "prompt": "...",
            "expected": {...},
            "task_definition": {...},
            "failures": [
                {"model": "model-a", "error": "...", "exit_status": "error"},
                {"model": "model-b", "raw_response": "...", "score": 0.2, "exit_status": "success"}
            ]
        }
    ]
    """
    runs = get_runs_by_suite(db_path, suite_id)

    # Group failed runs by task_id
    task_failures: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        exit_status = run.get("exit_status", "")
        score_primary = run.get("score_primary")
        is_failure = False

        if exit_status == "error":
            is_failure = True
        elif score_primary is not None and float(score_primary) == 0.0:
            is_failure = True

        if not is_failure:
            continue

        task_id = run.get("task_id", "")
        task_failures.setdefault(task_id, []).append(run)

    regression_tasks: list[dict[str, Any]] = []
    for task_id, failures in sorted(task_failures.items()):
        task = get_task_by_id(db_path, task_id) or {}

        failure_list: list[dict[str, Any]] = []
        for fail_run in failures:
            raw_response = fail_run.get("raw_response", "") or ""
            error_message = fail_run.get("error_message") or ""

            # Exclude runs with empty raw_response and no error_message (pure API errors)
            if not raw_response and not error_message:
                continue

            failure_entry: dict[str, Any] = {
                "model": fail_run.get("model_alias", ""),
                "exit_status": fail_run.get("exit_status", "error"),
            }

            if error_message:
                failure_entry["error"] = error_message

            if raw_response:
                failure_entry["raw_response"] = raw_response

            score = fail_run.get("score_primary")
            if score is not None:
                failure_entry["score"] = score

            failure_list.append(failure_entry)

        if not failure_list:
            continue

        task_entry: dict[str, Any] = {
            "task_id": task_id,
            "family": task.get("family", "unknown"),
            "prompt": task.get("prompt", ""),
            "expected": task.get("expected", {}),
            "failures": failure_list,
        }

        # Include full task definition
        if task:
            task_entry["task_definition"] = task

        regression_tasks.append(task_entry)

    output_file = _resolve_output_path(out_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(
            regression_tasks,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info("Exported %d regression tasks to %s", len(regression_tasks), output_file)
    return str(output_file)
