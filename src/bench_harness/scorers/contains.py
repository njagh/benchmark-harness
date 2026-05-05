"""Contains scorer — checks for required and forbidden substrings."""

from __future__ import annotations

import logging
import re
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


@register_scorer
class ContainsScorer(BaseScorer):
    """Scores responses by checking for presence/absence of specific strings.

    Required patterns must appear; absent patterns must NOT appear.
    Score is a composite: required patterns found / required total,
    with penalties for absent patterns found.
    """

    name = "contains"
    version = "1.0"

    def __init__(self, default_case_insensitive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.default_case_insensitive = default_case_insensitive

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        required = task.expected.patterns or []
        absent = task.expected.absent_patterns or []
        case_insensitive = task.expected.case_insensitive if hasattr(task.expected, 'case_insensitive') else self.default_case_insensitive

        flags = re.IGNORECASE if case_insensitive else 0
        resp = raw_response

        required_found = []
        required_missed = []
        for p in required:
            if re.search(p, resp, flags=flags) or p in resp:
                required_found.append(p)
            else:
                required_missed.append(p)

        absent_found = []
        for p in absent:
            if re.search(p, resp, flags=flags) or p in resp:
                absent_found.append(p)

        # Score: fraction of required patterns found
        total_required = len(required)
        if total_required > 0:
            score = len(required_found) / total_required
        else:
            score = 0.0

        # Penalty for absent patterns found
        if absent_found:
            penalty = len(absent_found) * 0.25
            score = max(0.0, score - penalty)

        passed = score >= 0.9 and len(absent_found) == 0

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=round(score, 4),
            passed=passed,
            details={
                "required_patterns": required,
                "required_found": required_found,
                "required_missed": required_missed,
                "absent_patterns": absent,
                "absent_found": absent_found,
                "case_insensitive": case_insensitive,
            },
            explanation=(
                f"Required: {len(required_found)}/{len(required)} found. "
                f"Absent violations: {absent_found or 'none'}"
            ),
        )
