"""Regex scorer — matches response against regex patterns."""

from __future__ import annotations

import logging
import re
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = {"all", "any", "none"}


@register_scorer
class RegexScorer(BaseScorer):
    """Scores responses by matching regex patterns.

    Supports three modes:
    - "all" (default): all patterns must match for a full score
    - "any": at least one pattern must match
    - "none": none of the patterns should match (anti-patterns)

    Score is the fraction of patterns matched.
    """

    name = "regex"
    version = "1.0"

    def __init__(self, default_mode: str = "all", case_insensitive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.default_mode = default_mode
        self.case_insensitive = case_insensitive

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        patterns = task.expected.patterns
        if not patterns:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "No patterns defined in task"},
            )

        mode = task.expected.type if task.expected.type in _SUPPORTED_MODES else self.default_mode

        flags = re.IGNORECASE if self.case_insensitive else 0
        results: list[bool] = []
        matched: list[str] = []

        for pattern in patterns:
            try:
                match = re.search(pattern, raw_response, flags=flags)
                results.append(match is not None)
                if match:
                    matched.append(pattern)
            except re.error as e:
                logger.warning("Invalid regex pattern '%s': %s", pattern, e)
                results.append(False)

        total = len(results)
        if mode == "none":
            # For "none" mode, score is high when patterns DON'T match
            match_count = sum(1 for r in results if not r)
            score = match_count / total
            passed = score == 1.0
        elif mode == "any":
            match_count = sum(1 for r in results if r)
            score = match_count / total
            passed = match_count > 0
        else:  # "all"
            match_count = sum(1 for r in results if r)
            score = match_count / total
            passed = score == 1.0

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=score,
            passed=passed,
            details={
                "mode": mode,
                "total_patterns": total,
                "matched_patterns": match_count,
                "matched": matched,
                "pattern_results": {p: r for p, r in zip(patterns, results)},
                "case_insensitive": self.case_insensitive,
            },
            explanation=(
                f"Mode '{mode}': {match_count}/{total} patterns matched"
            ),
        )
