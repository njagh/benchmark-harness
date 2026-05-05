"""Exact match scorer — compares response to expected answer."""

from __future__ import annotations

import logging
import time
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


@register_scorer
class ExactMatchScorer(BaseScorer):
    """Scores responses by exact string match against the expected answer.

    Supports case-insensitive comparison and whitespace normalization.
    """

    name = "exact_match"
    version = "1.0"

    def __init__(self, case_insensitive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.case_insensitive = case_insensitive

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        expected = task.expected.answer
        if expected is None:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "No expected answer in task"},
                explanation="Task has no expected answer to match against.",
            )

        resp = raw_response.strip()
        exp = str(expected).strip()

        if self.case_insensitive:
            resp = resp.lower()
            exp = exp.lower()

        passed = resp == exp
        score = 1.0 if passed else 0.0

        explanation = (
            "Exact match" if passed
            else f"Expected: {exp!r}; Got: {resp!r}"
        )

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=score,
            passed=passed,
            details={
                "expected": exp,
                "response": resp,
                "case_insensitive": self.case_insensitive,
            },
            explanation=explanation,
        )
