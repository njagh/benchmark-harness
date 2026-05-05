"""Task registry — centralized task storage and lookup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bench_harness.tasks.loaders import (
    load_tasks_as_objects,
    filter_tasks,
    filter_tasks_by_source,
)
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Central registry for benchmark tasks.

    Supports loading from directories, lookups by ID/family/source,
    and versioned task storage.
    """

    def __init__(self):
        # tasks keyed by {id}@{version}
        self._tasks: dict[str, Task] = {}

    def register(self, task: Task) -> None:
        """Add a task to the registry.

        If a task with the same ID and version exists, it is overwritten.
        Tasks with the same ID but different versions are kept separately.

        Args:
            task: The Task object to register.
        """
        key = f"{task.id}@{task.version}"
        self._tasks[key] = task
        logger.debug("Registered task: %s (version %s)", task.id, task.version)

    def load_from_directory(self, dir_path: str) -> int:
        """Load all tasks from a directory and register them.

        Args:
            dir_path: Path to directory containing task YAML files.

        Returns:
            Number of tasks successfully loaded.
        """
        tasks = load_tasks_as_objects(dir_path)
        for task in tasks:
            self.register(task)
        return len(tasks)

    def get(self, task_id: str) -> Task | None:
        """Look up the latest version of a task by ID.

        Args:
            task_id: The stable task ID (e.g. "smoke.factual_001").

        Returns:
            The Task object, or None if not found.
        """
        # Exact match with @version first
        for key, task in self._tasks.items():
            if task.id == task_id:
                return task
        return None

    def get_versioned(self, key: str) -> Task | None:
        """Look up a specific version of a task by {id}@{version}.

        Args:
            key: Versioned key (e.g. "smoke.factual_001@1.0").

        Returns:
            The Task object, or None if not found.
        """
        return self._tasks.get(key)

    def list_by_family(self, family: str) -> list[Task]:
        """List all tasks in a family.

        Args:
            family: Task family identifier.

        Returns:
            List of Task objects in the family.
        """
        return [t for t in self._tasks.values() if t.family == family]

    def list_by_source(self, source: str) -> list[Task]:
        """List all tasks from a given source.

        Args:
            source: Source identifier (local, public, synthetic).

        Returns:
            List of Task objects from the source.
        """
        return [t for t in self._tasks.values() if t.source == source]

    def list_all(self) -> list[Task]:
        """Return all registered tasks.

        Returns:
            List of all Task objects.
        """
        return list(self._tasks.values())

    def count(self) -> int:
        """Return total number of registered tasks."""
        return len(self._tasks)

    def summary(self) -> dict[str, dict[str, int]]:
        """Return a summary of tasks grouped by family and source.

        Returns:
            Dict with 'family' and 'source' keys, each mapping to
            a dict of {name: count}.
        """
        by_family: dict[str, int] = {}
        by_source: dict[str, int] = {}

        for task in self._tasks.values():
            by_family[task.family] = by_family.get(task.family, 0) + 1
            by_source[task.source] = by_source.get(task.source, 0) + 1

        return {"family": by_family, "source": by_source}

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all tasks to plain dicts."""
        return [t.to_dict() for t in self._tasks.values()]

    def __len__(self) -> int:
        return len(self._tasks)

    def __bool__(self) -> bool:
        return bool(self._tasks)
