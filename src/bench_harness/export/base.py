"""Base exporter functionality shared by all export formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from bench_harness.tasks.loaders import load_tasks

logger = logging.getLogger(__name__)

TASK_DIRS = (
    "tasks/smoke",
    "tasks/coding_smoke",
    "tasks/local_coding_agent_v1",
    "tasks/quantization_test",
)


def get_tasks_from_dir(task_dir: str) -> dict[str, dict]:
    """Load all tasks from a directory, keyed by task ID."""
    tasks = load_tasks(task_dir)
    return {t.get("id", ""): t for t in tasks if t.get("id")}


def _load_all_tasks() -> dict[str, dict]:
    """Load all tasks from known task directories, keyed by task ID."""
    all_tasks: dict[str, dict] = {}
    root = Path(__file__).resolve().parent.parent.parent.parent

    for td in TASK_DIRS:
        td_path = root / td
        if td_path.exists() and td_path.is_dir():
            try:
                tasks = get_tasks_from_dir(str(td_path))
                all_tasks.update(tasks)
            except Exception as e:
                logger.warning("Failed to load tasks from %s: %s", td_path, e)

    logger.info("Loaded %d tasks from all directories", len(all_tasks))
    return all_tasks


_task_cache: dict[str, dict] | None = None


def _get_task_cache() -> dict[str, dict]:
    """Lazy-load and cache all tasks across directories."""
    global _task_cache
    if _task_cache is None:
        _task_cache = _load_all_tasks()
    return _task_cache


def get_runs_by_suite(db_path: str, suite_id: str) -> list[dict]:
    """Get all run results for a suite."""
    from bench_harness.storage.sqlite import SQLiteStore

    store = SQLiteStore(db_path)
    return store.get_runs(suite_id=suite_id)


def get_task_by_id(db_path: str, task_id: str) -> dict | None:
    """Look up task definition by ID from loaded task YAML files."""
    cache = _get_task_cache()
    return cache.get(task_id)


def get_tasks_from_task_dir(db_path: str, suite_id: str | None = None) -> dict[str, dict]:
    """Return the full task cache (all tasks across directories)."""
    return _get_task_cache()


def get_judge_evaluations(db_path: str, suite_id: str) -> list[dict]:
    """Get judge evaluation records for a suite."""
    from bench_harness.storage.sqlite import SQLiteStore

    store = SQLiteStore(db_path)
    return store.get_judge_evaluations(suite_id)


def get_pairwise_comparisons(db_path: str, suite_id: str) -> list[dict]:
    """Get pairwise comparison records for a suite."""
    from bench_harness.storage.sqlite import SQLiteStore

    store = SQLiteStore(db_path)
    return store.get_pairwise_comparisons(suite_id)
