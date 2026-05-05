"""Base scorer interface and scorer registry."""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from bench_harness.tasks.task_schema import Task


def _normalize_task(task) -> Task:
    """Convert a dict to Task if needed."""
    if isinstance(task, dict):
        return Task.model_validate(task)
    return task

logger = logging.getLogger(__name__)

# Global registry: name -> scorer class
_SCORERS: dict[str, type["BaseScorer"]] = {}


def register_scorer(cls: type["BaseScorer"]) -> type["BaseScorer"]:
    """Decorator to register a scorer class by its `name` attribute."""
    _SCORERS[cls.name] = cls
    logger.debug("Registered scorer: %s", cls.name)
    return cls


def get_scorer(name: str, **kwargs: Any) -> "BaseScorer":
    """Look up and instantiate a scorer by name.

    Args:
        name: Scorer name (e.g. "exact_match").
        **kwargs: Arguments passed to the scorer constructor.

    Returns:
        An instance of the registered scorer class.

    Raises:
        KeyError: If no scorer with that name is registered.
    """
    if name not in _SCORERS:
        raise KeyError(
            f"Unknown scorer '{name}'. Registered scorers: {list(_SCORERS.keys())}"
        )
    return _SCORERS[name](**kwargs)


def list_scorers() -> list[str]:
    """Return all registered scorer names."""
    return list(_SCORERS.keys())


def score_all(
    task: Task, raw_response: str, scorer_names: list[str]
) -> dict[str, "ScoreResult"]:
    """Run multiple scorers on a single task/response pair.

    Args:
        task: The task definition.
        raw_response: The model's raw response.
        scorer_names: List of scorer names to run.

    Returns:
        Dict mapping scorer name to ScoreResult.
    """
    results: dict[str, ScoreResult] = {}
    for name in scorer_names:
        try:
            scorer = get_scorer(name)
            if scorer.validate_task(task):
                results[name] = scorer.score(task, raw_response)
            else:
                logger.warning(
                    "Scorer '%s' not applicable for task '%s' (type '%s')",
                    name,
                    task.id if hasattr(task, "id") else task.get("id", "unknown"),
                    getattr(task.expected, "type", "unknown") if hasattr(task, "expected") else "unknown",
                )
        except Exception as e:
            task_id = task.id if hasattr(task, "id") else task.get("id", "unknown")
            logger.error("Scorer '%s' failed for task '%s': %s", name, task_id, e)
            results[name] = ScoreResult(
                scorer_name=name,
                scorer_version="error",
                score=-1.0,
                passed=False,
                details={"error": str(e)},
            )
    return results


@dataclass
class ScoreResult:
    """Result of scoring a single response.

    Attributes:
        scorer_name: Name of the scorer that produced this result.
        scorer_version: Version of the scorer.
        score: Score from 0.0 (fail) to 1.0 (perfect pass).
        passed: Boolean pass/fail decision.
        details: Scorer-specific breakdown data.
        explanation: Human-readable reason for the score.
        duration_ms: Time taken to score (if available).
    """

    scorer_name: str
    scorer_version: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON storage)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreResult":
        """Deserialize from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaseScorer(abc.ABC):
    """Abstract base class for all scorers.

    All scorers inherit from this class and implement `score()`.
    They are registered with `@register_scorer` decorator so they
    can be looked up by name.
    """

    name: str = "base"
    version: str = "1.0"

    def __init__(self, **kwargs: Any):
        """Initialize scorer with optional config kwargs.

        Args:
            **kwargs: Scorer-specific configuration options.
        """
        self._config = kwargs

    @abc.abstractmethod
    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Score a single response against a task's expected output.

        Args:
            task: The task definition containing expected output and scoring config.
            raw_response: The raw text response from the model.

        Returns:
            ScoreResult with score, pass/fail, and details.
        """

    def validate_task(self, task: Task) -> bool:
        """Check if this scorer is appropriate for the task's expected output type.

        Override in subclasses to enforce type requirements.

        Args:
            task: The task definition.

        Returns:
            True if this scorer can handle the task.
        """
        return True

    def _compute_duration(self, start: float, result: ScoreResult) -> ScoreResult:
        """Attach duration_ms to a ScoreResult."""
        result.duration_ms = (time.perf_counter() - start) * 1000.0
        return result
