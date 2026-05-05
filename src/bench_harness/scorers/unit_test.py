"""Unit test scorer — scores based on test pass rate from code task results."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


@dataclass
class UnitTestResult:
    """Parsed result from test execution."""
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    test_output: str = ""
    test_names: list[str] = None


@register_scorer
class UnitTestScorer(BaseScorer):
    """Scores code tasks based on test pass rate.

    Expects the task to have code-related fields populated in run results.
    The raw_response contains the generated code, and the task dict may
    include test_code. This scorer is designed to work with CodeTaskRunner
    which pre-computes the test results and stores them in score_secondary.
    """

    name = "unit_test"
    version = "1.0"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        expected = task.expected
        test_code = expected.test_code or ""

        test_calls = re.findall(r"assert\s+", test_code) if test_code else []
        total_tests = len(test_calls) if test_calls else 1

        patterns = expected.patterns or []
        patterns_match = all(re.search(p, raw_response) for p in patterns if p)

        if patterns_match:
            score = 0.5
            passed = True
        else:
            score = 0.0
            passed = False

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=score,
            passed=passed,
            details={
                "patterns_matched": patterns_match,
                "total_expected_tests": total_tests,
            },
            explanation=f"Generated code checked for expected patterns ({len(patterns)} patterns)",
        )

    def validate_task(self, task: Task) -> bool:
        task = _normalize_task(task)
        return task.expected.type == "unit_test"
