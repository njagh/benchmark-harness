"""Format compliance scorer — checks response follows output format."""

from __future__ import annotations

import logging
import re
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)

# Conversational filler phrases to detect
_FILLER_PHRASES = [
    r"^sure\b",
    r"^of\s+course\b",
    r"^sure\s+thing\b",
    r"^here\s+you\s+go\b",
    r"^here's\b",
    r"^ok\b",
    r"^okay\b",
    r"^no\s+problem\b",
    r"^certainly\b",
    r"^absolutely\b",
    r"^of\s+course\b",
    r"^i'd\s+be\s+happy\b",
]

_FILLER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _FILLER_PHRASES]


@register_scorer
class FormatComplianceScorer(BaseScorer):
    """Scores responses by checking output format compliance.

    Supports various format checks defined in task.expected.format_checks:
    - starts_with_numbered_list
    - starts_with_bullet_list
    - is_markdown_table
    - is_code_block
    - line_count_min: N
    - line_count_max: N
    - no_conversational_filler
    - ends_with_keyword: X
    - item_count: N
    """

    name = "format_compliance"
    version = "1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        checks = task.expected.format_checks or []
        if not checks:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "No format_checks defined in task"},
            )

        results: dict[str, bool] = {}
        check_names = self._parse_format_checks(checks)

        for check_name, check_args in check_names.items():
            checker = self._get_checker(check_name)
            results[check_name] = checker(raw_response, check_args)

        total = len(results)
        passed_count = sum(1 for v in results.values() if v)
        score = passed_count / total if total > 0 else 0.0
        passed = score >= 0.9

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=round(score, 4),
            passed=passed,
            details={
                "format_checks": checks,
                "check_results": results,
                "total_checks": total,
                "passed_checks": passed_count,
            },
            explanation=(
                f"{passed_count}/{total} format checks passed"
            ),
        )

    def _parse_format_checks(self, checks: list) -> dict[str, dict]:
        """Parse format_checks list into a dict of {name: args}."""
        parsed: dict[str, dict] = {}
        for check in checks:
            if isinstance(check, str):
                parsed[check] = {}
            elif isinstance(check, dict):
                for k, v in check.items():
                    parsed[k] = v if isinstance(v, dict) else {v: v} if not isinstance(v, bool) else {"enabled": v}
        return parsed

    def _get_checker(self, check_name: str):
        """Get the checker function for a format check name."""
        checkers = {
            "starts_with_numbered_list": self._check_numbered_list,
            "starts_with_bullet_list": self._check_bullet_list,
            "is_markdown_table": self._check_markdown_table,
            "is_code_block": self._check_code_block,
            "line_count_max": self._check_line_count_max,
            "line_count_min": self._check_line_count_min,
            "no_conversational_filler": self._check_no_filler,
            "ends_with_keyword": self._check_ends_with_keyword,
            "item_count": self._check_item_count,
        }
        return checkers.get(check_name, self._check_default)

    def _check_default(self, response: str, args: dict) -> bool:
        """Default checker — always passes."""
        return True

    @staticmethod
    def _check_numbered_list(response: str, args: dict) -> bool:
        """Check if response starts with a numbered list."""
        stripped = response.strip()
        return bool(re.match(r'^\d+[\.\)]\s', stripped))

    @staticmethod
    def _check_bullet_list(response: str, args: dict) -> bool:
        """Check if response starts with a bullet list."""
        stripped = response.strip()
        return bool(re.match(r'^[\-\*\+]\s', stripped))

    @staticmethod
    def _check_markdown_table(response: str, args: dict) -> bool:
        """Check if response contains a valid markdown table."""
        return bool(re.search(r'^\|.+`\|', response, re.MULTILINE))

    @staticmethod
    def _check_code_block(response: str, args: dict) -> bool:
        """Check if response contains a fenced code block."""
        return bool(re.search(r'```', response))

    @staticmethod
    def _check_line_count_max(response: str, args: dict) -> bool:
        """Check that response has at most N lines."""
        max_lines = args.get("max", args.get(10, 10))
        lines = response.strip().split('\n')
        return len(lines) <= max_lines

    @staticmethod
    def _check_line_count_min(response: str, args: dict) -> bool:
        """Check that response has at least N lines."""
        min_lines = args.get("min", args.get(1, 1))
        lines = response.strip().split('\n')
        return len(lines) >= min_lines

    @staticmethod
    def _check_no_filler(response: str, args: dict) -> bool:
        """Check that response does not start with conversational filler."""
        stripped = response.strip()
        for pattern in _FILLER_PATTERNS:
            if pattern.match(stripped):
                return False
        return True

    @staticmethod
    def _check_ends_with_keyword(response: str, args: dict) -> bool:
        """Check that response ends with a specific keyword."""
        keyword = args.get("keyword", args.get(1, ""))
        if not keyword:
            return True
        return response.strip().lower().endswith(keyword.strip().lower())

    @staticmethod
    def _check_item_count(response: str, args: dict) -> bool:
        """Check that a numbered/bullet list has exactly N items."""
        count = args.get("count", args.get(3, 0))
        if count == 0:
            return True
        # Count numbered list items
        items = re.findall(r'^\d+[\.\)]\s', response, re.MULTILINE)
        return len(items) == count
