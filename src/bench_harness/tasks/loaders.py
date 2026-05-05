"""Task loading utilities with schema validation.

Supports both the new Task schema (with versioning, families, categories)
and backward-compatible legacy YAML format from M1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)

REQUIRED_TASK_KEYS = {"id", "prompt", "scoring"}


def _is_legacy_task(data: dict[str, Any]) -> bool:
    """Heuristic: old-format tasks have 'scoring' but not 'expected'."""
    return "scoring" in data and "expected" not in data and "version" not in data


def _migrate_legacy_task(data: dict[str, Any], task_path: str) -> dict[str, Any]:
    """Convert a legacy M1-format task dict to the new Task schema format.

    Migration:
    - 'prompt' -> kept as 'prompt'
    - 'scoring.primary' -> kept as 'scoring.primary'
    - 'expected.type' -> becomes 'expected.type'
    - Adds version="1.0", family="unknown", source="local"
    """
    task_id = data.get("id", Path(task_path).stem)

    # Derive family from ID prefix if available
    family = "unknown"
    if "." in task_id:
        parts = task_id.split(".")
        if len(parts) >= 2:
            family = parts[1] if len(parts) > 2 else "unknown"

    migrated: dict[str, Any] = {
        "id": task_id,
        "version": "1.0",
        "family": family,
        "source": "local",
        "prompt": data.get("prompt", ""),
        "expected": data.get("expected", {"type": "unknown"}),
        "scoring": data.get("scoring", {"primary": "unknown"}),
        "risk_level": "low",
        "context_tokens": "small",
    }

    logger.info("Migrated legacy task %s from %s", task_id, task_path)
    return migrated


def _to_dict(task: Task | dict[str, Any]) -> dict[str, Any]:
    """Normalize a task to a dict (handles both Task objects and dicts)."""
    if isinstance(task, Task):
        return task.to_dict()
    return task


def load_task(task_path: str) -> dict[str, Any]:
    """Load a single task from a YAML file with schema validation.

    Supports both the new Task schema and legacy M1 format (auto-migrated).

    Args:
        task_path: Path to the YAML task file.

    Returns:
        Task dict with all required fields normalized.

    Raises:
        FileNotFoundError: If the task file doesn't exist.
        ValueError: If required keys are missing after loading/migration.
    """
    filepath = Path(task_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Task file not found: {filepath}")

    with open(filepath, "r") as f:
        task = yaml.safe_load(f)

    if task is None:
        raise ValueError(f"Task file is empty: {filepath}")

    # Handle legacy format — auto-migrate
    if _is_legacy_task(task):
        task = _migrate_legacy_task(task, task_path)

    # Validate required fields for new format
    try:
        task_obj = Task.model_validate(task)
        return task_obj.to_dict()
    except Exception as e:
        raise ValueError(f"Task {task.get('id', '<unknown>')!r} validation failed: {e}") from e


def load_task_object(task_path: str) -> Task:
    """Load a single task and return a validated Task object.

    Args:
        task_path: Path to the YAML task file.

    Returns:
        Validated Task object.
    """
    filepath = Path(task_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Task file not found: {filepath}")

    with open(filepath, "r") as f:
        task = yaml.safe_load(f)

    if task is None:
        raise ValueError(f"Task file is empty: {filepath}")

    # Handle legacy format
    if _is_legacy_task(task):
        task = _migrate_legacy_task(task, task_path)

    return Task.model_validate(task)


def load_tasks(task_dir: str) -> list[dict[str, Any]]:
    """Load all task YAML files from a directory.

    Args:
        task_dir: Path to directory containing task YAML files.

    Returns:
        List of task dicts. Skips files that fail to load with a warning.
    """
    return _load_tasks_impl(task_dir, as_objects=False)


def load_tasks_as_objects(task_dir: str) -> list[Task]:
    """Load all task YAML files from a directory as Task objects.

    Args:
        task_dir: Path to directory containing task YAML files.

    Returns:
        List of validated Task objects. Skips files that fail with a warning.
    """
    return _load_tasks_impl(task_dir, as_objects=True)


def _load_tasks_impl(task_dir: str, as_objects: bool) -> list[Task] | list[dict[str, Any]]:
    """Internal implementation shared by load_tasks and load_tasks_as_objects."""
    task_path = Path(task_dir)
    if not task_path.exists():
        raise FileNotFoundError(f"Task directory not found: {task_path}")

    if not task_path.is_dir():
        raise ValueError(f"Task path is not a directory: {task_path}")

    results: list[Task] | list[dict[str, Any]] = []
    yaml_files = sorted(task_path.glob("*.yaml")) + sorted(task_path.glob("*.yml"))

    if not yaml_files:
        logger.warning("No .yaml/.yml files found in %s", task_path)
        return results

    load_fn = load_task_object if as_objects else load_task

    for yf in yaml_files:
        try:
            task = load_fn(str(yf))
            results.append(task)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Skipping task file %s: %s", yf.name, e)
        except Exception as e:
            logger.error("Skipping task file %s (unexpected error): %s", yf.name, e)

    logger.info("Loaded %d tasks from %s", len(results), task_path)
    return results


def filter_tasks(
    tasks: list[dict[str, Any]] | list[Task],
    family: str | None,
) -> list[dict[str, Any]] | list[Task]:
    """Optionally filter tasks by family.

    Args:
        tasks: List of task dicts or Task objects.
        family: If provided, only return tasks matching this family.

    Returns:
        Filtered list, preserving the input type.
    """
    if family is None:
        return tasks

    def _get_family(t):
        if isinstance(t, Task):
            return t.family
        return t.get("family", "")

    filtered = [t for t in tasks if _get_family(t) == family]
    logger.info("Filtered to %d tasks in family '%s'", len(filtered), family)
    return filtered


def filter_tasks_by_source(
    tasks: list[dict[str, Any]] | list[Task],
    source: str,
) -> list[dict[str, Any]] | list[Task]:
    """Optionally filter tasks by source."""

    def _get_source(t):
        if isinstance(t, Task):
            return t.source
        return t.get("source", "")

    filtered = [t for t in tasks if _get_source(t) == source]
    return filtered
