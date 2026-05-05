"""Multiple-choice scorer — extracts answer from model response."""

from __future__ import annotations

import re
import logging
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)

# Patterns to extract a choice letter from a model response
# Order matters: more specific patterns first
_ANSWER_PATTERNS = [
    r"\(([A-D])\)",                                      # "(A)" — parenthesized letter
    r"([A-D])\.+\s+\w",                                 # "A. ..."
    r"(?:^|\b)(?:answer\s*[:\.]?\s*)?([A-D])\b",        # "A", "Answer: A"
    r"\b([A-D])\s*[:\.]\s",                             # "A: ..." or "A. ..."
    r"(?:the\s+)?(?:answer\s+is|selected|chose)\s+[a-z]*(?:\s+is)?\s*([A-D])",  # "the answer is B"
]


@register_scorer
class MultipleChoiceScorer(BaseScorer):
    """Scores multiple-choice answers by extracting the model's chosen letter."""

    name = "multiple_choice"
    version = "1.0"

    def __init__(self, partial_credit: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.partial_credit = partial_credit

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        expected_answer = task.expected.answer
        choices = task.expected.choices

        if not expected_answer or not choices:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "Missing expected answer or choices"},
            )

        # Extract the model's answer letter
        model_answer = self._extract_answer(raw_response)

        if model_answer is None:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={
                    "expected": expected_answer,
                    "extracted_answer": None,
                    "reason": "Could not extract answer letter from response",
                },
                explanation="No answer letter could be extracted from the response.",
            )

        passed = model_answer == expected_answer
        score = 1.0 if passed else 0.0

        # Check if choice text also matches
        choice_matches = 0
        if not passed:
            # Check if the response contains the correct choice text
            correct_text = choices.get(expected_answer, "")
            if correct_text and correct_text.lower() in raw_response.lower():
                score = 0.5 if self.partial_credit else 0.0
                passed = False
                choice_matches = 1

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=score,
            passed=passed,
            details={
                "expected": expected_answer,
                "extracted_answer": model_answer,
                "choices": choices,
                "choice_text_match": choice_matches > 0,
            },
            explanation=(
                f"Extracted '{model_answer}', expected '{expected_answer}'"
                if not passed
                else f"Extracted '{model_answer}', matches expected '{expected_answer}'"
            ),
        )

    def _extract_answer(self, response: str) -> str | None:
        """Extract a choice letter from the model response.

        Tries multiple regex patterns and returns the first match.
        """
        for pattern in _ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
