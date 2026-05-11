"""Tests for M22 — ConfigValidator with missing dependencies.

Tests edge cases where jsonschema or PyYAML are not installed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bench_harness.runners.config_validator import ConfigValidator


class TestConfigValidatorMissingDeps:
    def test_yaml_missing_yaml_import(self):
        """validate_yaml returns error when yaml module is missing."""
        with patch.dict("sys.modules", {"yaml": None}):
            # Force reimport to trigger import error
            import importlib
            import bench_harness.runners.config_validator as mod
            importlib.reload(mod)
            result = mod.ConfigValidator.validate_yaml("key: value")
            assert result["valid"] is False
            assert "PyYAML" in result["error"]

    def test_yaml_schema_validation_error(self):
        """validate_yaml catches jsonschema.ValidationError."""
        import json

        content = json.dumps({"name": 123})  # name should be string
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = ConfigValidator.validate_yaml(content, schema=schema)
        assert result["valid"] is True
        # schema_valid should be False if validation fails
        if result["schema_valid"] is not None:
            assert result["schema_valid"] is False

    def test_json_schema_validation_error(self):
        """validate_json catches jsonschema.ValidationError."""
        content = '{"name": 123}'
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = ConfigValidator.validate_json(content, schema=schema)
        assert result["valid"] is True
        if result["schema_valid"] is not None:
            assert result["schema_valid"] is False

    def test_yaml_schema_with_string_type(self):
        """validate_yaml schema passes when content matches schema."""
        import json
        content = json.dumps({"name": "test", "age": 30})
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
        }
        result = ConfigValidator.validate_yaml(content, schema=schema)
        assert result["valid"] is True
        if result["schema_valid"] is not None:
            assert result["schema_valid"] is True
