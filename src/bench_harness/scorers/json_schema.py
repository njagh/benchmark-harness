"""JSON schema scorer — validates response JSON against a schema."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import jsonschema

from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


@register_scorer
class JsonSchemaScorer(BaseScorer):
    """Scores responses by validating parsed JSON against a schema.

    Handles JSON extraction from markdown code fences, bare JSON,
    and partial/broken JSON with repair attempts.
    """

    name = "json_schema"
    version = "1.0"

    def __init__(self, strict: bool = False, repair: bool = True,
                 repair_penalty: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strict = strict
        self.repair = repair
        self.repair_penalty = repair_penalty

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        schema = task.expected.json_schema or task.expected.schema
        if not schema:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "No JSON schema defined in task"},
            )

        # Extract JSON from response
        extracted, repair_needed = self._extract_json(raw_response)

        if extracted is None:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "Could not extract JSON from response"},
                explanation="No valid JSON found in the response.",
            )

        # Validate against schema
        try:
            validator = jsonschema.Draft202012Validator(schema)
            if self.strict:
                validator = jsonschema.Draft202012Validator(
                    schema, format_checker=jsonschema.FormatChecker()
                )
            validator.validate(extracted)
            passed = True
            score = 1.0 if not repair_needed else (1.0 - self.repair_penalty)
        except jsonschema.ValidationError as e:
            passed = False
            score = 0.0
            # If repair is enabled, try to fix common issues
            if self.repair and not self._is_severe_error(e):
                repaired, repaired_valid = self._attempt_repair(extracted, schema)
                if repaired_valid:
                    score = 1.0 - self.repair_penalty
                    passed = True

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=round(score, 4),
            passed=passed,
            details={
                "schema": schema,
                "repair_needed": repair_needed,
                "strict": self.strict,
                "validation_errors": self._error_to_list(extracted, schema),
            },
            explanation=(
                "Valid JSON matching schema" if passed
                else "JSON validation failed" if repair_needed
                else "Could not parse JSON from response"
            ),
        )

    def _extract_json(self, response: str) -> tuple[Any | None, bool]:
        """Extract JSON from a response string.

        Returns (parsed_json, was_repaired).

        Strategy:
        1. Look for ```json ... ``` blocks
        2. Look for ``` ... ``` blocks
        3. Try parsing the whole string as JSON
        4. Try to find balanced braces/brackets
        """
        # 1. JSON code fence
        match = re.search(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1)), False
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Try whole response
        try:
            return json.loads(response.strip()), False
        except (json.JSONDecodeError, ValueError):
            pass

        # 3. Find first top-level { or [ and try balanced extraction
        result = self._extract_balanced(response)
        if result is not None:
            try:
                return json.loads(result), False
            except (json.JSONDecodeError, ValueError):
                pass

        # 4. If repair is enabled, try to fix common issues
        if self.repair:
            return self._try_repair_response(response)

        return None, False

    def _extract_balanced(self, text: str) -> str | None:
        """Extract a balanced JSON object or array from text."""
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                open_bracket = ch
                close_bracket = '}' if ch == '{' else ']'
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == open_bracket:
                        depth += 1
                    elif text[j] == close_bracket:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            # Make sure no nested quotes break it
                            return candidate
                break
        return None

    def _try_repair_response(self, response: str) -> tuple[Any | None, bool]:
        """Attempt to repair common JSON issues."""
        # Remove trailing commas before } or ]
        repaired = re.sub(r',(\s*[}\]])', r'\1', response)
        try:
            return json.loads(repaired), True
        except (json.JSONDecodeError, ValueError):
            pass

        # Remove comments (// style)
        repaired = re.sub(r'//.*', '', response)
        try:
            return json.loads(repaired), True
        except (json.JSONDecodeError, ValueError):
            pass

        return None, False

    def _attempt_repair(
        self, data: Any, schema: dict
    ) -> tuple[Any, bool]:
        """Attempt to fix validation errors by repairing the data."""
        # Simple repair: add missing required fields with null
        if isinstance(data, dict):
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            repaired = dict(data)
            for field_name in required:
                if field_name not in repaired:
                    prop_schema = properties.get(field_name, {})
                    repaired[field_name] = prop_schema.get(
                        "default", None
                    )
            try:
                jsonschema.Draft202012Validator(schema).validate(repaired)
                return repaired, True
            except jsonschema.ValidationError:
                pass
        return data, False

    def _is_severe_error(self, error: jsonschema.ValidationError) -> bool:
        """Check if a validation error is likely unrecoverable."""
        # Type errors on fundamental values are hard to repair
        return error.validator in ("type", "enum")

    def _error_to_list(self, data: Any, schema: dict) -> list[str]:
        """Convert validation errors to a list of strings."""
        try:
            validator = jsonschema.Draft202012Validator(schema)
            errors = list(validator.iter_errors(data))
            return [f"{e.json_path}: {e.message}" for e in errors]
        except Exception:
            return []
