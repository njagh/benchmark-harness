"""Config valid scorer — validates generated config files for correctness."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


@register_scorer
class ConfigValidScorer(BaseScorer):
    """Scores responses by validating config file syntax and structure.

    Validates YAML, JSON, systemd unit files, and other config formats
    by parsing them and checking for required patterns/fields.

    Expected task config:
        expected.config_type: "yaml" | "json" | "systemd" | "ini" | "docker-compose"
        expected.config_patterns: list of dicts with pattern/required fields
        expected.required_fields: list of required keys (YAML/JSON only)
        expected.validation_command: shell command to run for validation
    """

    name = "config_valid"
    version = "1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_task(self, task: Task) -> bool:
        """Only apply to tasks with a config_type in expected."""
        task = _normalize_task(task)
        config_type = getattr(task.expected, "config_type", None)
        if config_type is None:
            # Check if task has a dict-style expected with config_type
            expected = task.expected
            if isinstance(expected, dict) and "config_type" in expected:
                return True
            return False
        return config_type in ("yaml", "json", "systemd", "ini", "docker-compose")

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        expected = task.expected

        # Determine config type
        if hasattr(expected, "config_type"):
            config_type = expected.config_type
        elif isinstance(expected, dict):
            config_type = expected.get("config_type", "")
        else:
            config_type = ""

        if not config_type:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": "No config_type specified in task"},
                explanation="No config_type specified",
            )

        # Validate based on config type
        validation_result = self._validate_config(config_type, raw_response, task)

        if validation_result is None:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"reason": f"Unsupported config_type: {config_type}"},
                explanation=f"Unsupported config_type: {config_type}",
            )

        valid = validation_result.get("valid", False)
        error = validation_result.get("error")
        patterns = validation_result.get("patterns_checked", {})

        # Score components:
        # 1. Format validity (50% weight)
        # 2. Required field/field presence (30% weight)
        # 3. Pattern matching (20% weight)

        format_score = 1.0 if valid else 0.0
        fields = validation_result.get("fields_checked", {})
        fields_score = sum(1 for v in fields.values() if v) / max(len(fields), 1)

        pattern_score = sum(1 for v in patterns.values() if v) / max(len(patterns), 1) if patterns else 1.0

        total_score = (format_score * 0.5) + (fields_score * 0.3) + (pattern_score * 0.2)
        total_score = round(total_score, 4)

        passed = valid and total_score >= 0.8

        details = {
            "config_type": config_type,
            "valid": valid,
            "format_score": format_score,
            "fields_score": round(fields_score, 4),
            "pattern_score": round(pattern_score, 4),
            "total_score": total_score,
            "fields_checked": fields,
            "patterns_checked": patterns,
        }

        if error:
            details["error"] = error

        explanation_parts = [f"Config type: {config_type}"]
        if valid:
            explanation_parts.append("format valid")
        else:
            explanation_parts.append(f"format invalid: {error}")

        if fields:
            fields_pass = sum(1 for v in fields.values() if v)
            explanation_parts.append(f"{fields_pass}/{len(fields)} fields present")

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=total_score,
            passed=passed,
            details=details,
            explanation="; ".join(explanation_parts),
        )

    def _validate_config(
        self,
        config_type: str,
        content: str,
        task: Task,
    ) -> dict[str, Any] | None:
        """Validate config content based on type."""
        expected = task.expected

        try:
            if config_type in ("yaml", "docker-compose"):
                return self._validate_yaml_config(content, expected)
            elif config_type == "json":
                return self._validate_json_config(content, expected)
            elif config_type == "systemd":
                return self._validate_systemd_config(content, expected)
            elif config_type == "ini":
                return self._validate_ini_config(content, expected)
        except Exception as e:
            logger.error("Config validation error for type %s: %s", config_type, e)
            return {"valid": False, "error": str(e), "fields_checked": {}, "patterns_checked": {}}

        return None

    def _get_expected_fields(self, expected) -> list[str] | None:
        """Extract required fields from task expected."""
        if hasattr(expected, "required_fields"):
            fields = expected.required_fields
            if isinstance(fields, list):
                return fields
            elif isinstance(fields, dict):
                return list(fields.keys())
        elif isinstance(expected, dict) and "required_fields" in expected:
            fields = expected["required_fields"]
            if isinstance(fields, list):
                return fields
            elif isinstance(fields, dict):
                return list(fields.keys())
        return None

    def _get_pattern_specs(self, expected) -> list[dict] | None:
        """Extract pattern specs from task expected."""
        if hasattr(expected, "patterns"):
            patterns = expected.patterns
            if isinstance(patterns, list):
                return patterns
        elif isinstance(expected, dict) and "patterns" in expected:
            patterns = expected["patterns"]
            if isinstance(patterns, list):
                return patterns
        return None

    def _validate_yaml_config(self, content: str, expected) -> dict[str, Any]:
        """Validate YAML config content."""
        from bench_harness.runners.config_validator import ConfigValidator

        result = ConfigValidator.validate_yaml(content)
        valid = result.get("valid", False)
        fields_checked: dict[str, bool] = {}
        patterns_checked: dict[str, bool] = {}

        if valid and result.get("parsed") is not None:
            parsed = result["parsed"]
            required_fields = self._get_expected_fields(expected)
            if required_fields:
                if isinstance(parsed, dict):
                    for field in required_fields:
                        fields_checked[field] = field in parsed and parsed[field] is not None
                else:
                    fields_checked["parsed_as_dict"] = isinstance(parsed, dict)

            pattern_specs = self._get_pattern_specs(expected)
            import re
            if pattern_specs:
                for p in pattern_specs:
                    if isinstance(p, dict):
                        pattern_str = p.get("pattern", "")
                    else:
                        pattern_str = str(p)
                    if pattern_str:
                        try:
                            matches = bool(re.search(pattern_str, content))
                        except re.error:
                            matches = pattern_str in content
                        patterns_checked[pattern_str] = matches

        return {
            "valid": valid,
            "error": result.get("error"),
            "fields_checked": fields_checked,
            "patterns_checked": patterns_checked,
        }

    def _validate_json_config(self, content: str, expected) -> dict[str, Any]:
        """Validate JSON config content."""
        from bench_harness.runners.config_validator import ConfigValidator

        result = ConfigValidator.validate_json(content)
        valid = result.get("valid", False)
        fields_checked: dict[str, bool] = {}
        patterns_checked: dict[str, bool] = {}

        if valid and result.get("parsed") is not None:
            parsed = result["parsed"]
            required_fields = self._get_expected_fields(expected)
            if required_fields:
                if isinstance(parsed, dict):
                    for field in required_fields:
                        fields_checked[field] = field in parsed and parsed[field] is not None
                else:
                    fields_checked["parsed_as_dict"] = isinstance(parsed, dict)

            pattern_specs = self._get_pattern_specs(expected)
            import re
            if pattern_specs:
                for p in pattern_specs:
                    if isinstance(p, dict):
                        pattern_str = p.get("pattern", "")
                    else:
                        pattern_str = str(p)
                    if pattern_str:
                        try:
                            matches = bool(re.search(pattern_str, content))
                        except re.error:
                            matches = pattern_str in content
                        patterns_checked[pattern_str] = matches

        return {
            "valid": valid,
            "error": result.get("error"),
            "fields_checked": fields_checked,
            "patterns_checked": patterns_checked,
        }

    def _validate_systemd_config(self, content: str, expected) -> dict[str, Any]:
        """Validate systemd unit file content."""
        from bench_harness.runners.config_validator import ConfigValidator

        format_result = ConfigValidator.validate_systemd_unit(content)
        valid = format_result.get("valid", False)
        fields_checked: dict[str, bool] = {}
        patterns_checked: dict[str, bool] = {}

        # Map format checks to fields_checked
        checks = format_result.get("checks", {})
        fields_checked["has_unit_section"] = checks.get("has_unit_section", False)
        fields_checked["has_service_section"] = checks.get("has_service_section", False)
        fields_checked["has_install_section"] = checks.get("has_install_section", False)
        fields_checked["has_exec_start"] = checks.get("has_exec_start", False)

        pattern_specs = self._get_pattern_specs(expected)
        import re
        if pattern_specs:
            for p in pattern_specs:
                if isinstance(p, dict):
                    pattern_str = p.get("pattern", "")
                else:
                    pattern_str = str(p)
                if pattern_str:
                    try:
                        matches = bool(re.search(pattern_str, content))
                    except re.error:
                        matches = pattern_str in content
                    patterns_checked[pattern_str] = matches

        # Try systemd-analyze if available
        with tempfile.TemporaryDirectory() as tmpdir:
            analyze_result = ConfigValidator.validate_systemd_unit_via_analyze(content, tmpdir)
            if analyze_result.get("valid"):
                valid = True

        return {
            "valid": valid,
            "error": format_result.get("error"),
            "fields_checked": fields_checked,
            "patterns_checked": patterns_checked,
        }

    def _validate_ini_config(self, content: str, expected) -> dict[str, Any]:
        """Validate INI config content."""
        import configparser

        fields_checked: dict[str, bool] = {}
        patterns_checked: dict[str, bool] = {}

        parser = configparser.ConfigParser()
        try:
            parser.read_string(content)
            valid = True
            sections = parser.sections()
            fields_checked["has_sections"] = len(sections) > 0

            required_fields = self._get_expected_fields(expected)
            if required_fields:
                for field in required_fields:
                    found = False
                    for section in sections:
                        if parser.has_option(section, field):
                            found = True
                            break
                    fields_checked[field] = found
        except configparser.Error as e:
            valid = False
            fields_checked["has_sections"] = False

        pattern_specs = self._get_pattern_specs(expected)
        import re
        if pattern_specs:
            for p in pattern_specs:
                if isinstance(p, dict):
                    pattern_str = p.get("pattern", "")
                else:
                    pattern_str = str(p)
                if pattern_str:
                    try:
                        matches = bool(re.search(pattern_str, content))
                    except re.error:
                        matches = pattern_str in content
                    patterns_checked[pattern_str] = matches

        return {
            "valid": valid,
            "error": str(e) if not valid else None,
            "fields_checked": fields_checked,
            "patterns_checked": patterns_checked,
        }


def shutil_which(cmd: str) -> bool:
    """Check if a command exists on PATH (lightweight, no import shutil needed)."""
    try:
        subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
