"""Tests for M22 — ConfigValidator (zero-coverage module)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bench_harness.runners.config_validator import ConfigValidator


class TestValidateYaml:
    def test_valid_yaml(self):
        """Valid YAML string parses correctly."""
        result = ConfigValidator.validate_yaml("key: value\nlist:\n  - 1\n  - 2")
        assert result["valid"] is True
        assert result["error"] is None
        assert result["parsed"] == {"key": "value", "list": [1, 2]}

    def test_invalid_yaml(self):
        """Invalid YAML returns error."""
        result = ConfigValidator.validate_yaml("key: [unclosed")
        assert result["valid"] is False
        assert result["error"] is not None
        assert result["parsed"] is None

    def test_empty_yaml(self):
        """Empty YAML returns valid=False because parsed is None."""
        result = ConfigValidator.validate_yaml("")
        assert result["valid"] is False
        assert result["parsed"] is None

    def test_yaml_with_schema_valid(self):
        """YAML passes schema validation."""
        yaml_content = "name: test\nversion: 1"
        schema = {"type": "object", "required": ["name"]}
        result = ConfigValidator.validate_yaml(yaml_content, schema=schema)
        assert result["valid"] is True
        assert result["schema_valid"] is True

    def test_yaml_with_schema_invalid(self):
        """YAML fails schema validation."""
        yaml_content = "version: 1"
        schema = {"type": "object", "required": ["name"]}
        result = ConfigValidator.validate_yaml(yaml_content, schema=schema)
        assert result["valid"] is True
        assert result["schema_valid"] is False


class TestValidateJson:
    def test_valid_json(self):
        """Valid JSON string parses correctly."""
        result = ConfigValidator.validate_json('{"key": "value", "num": 42}')
        assert result["valid"] is True
        assert result["error"] is None
        assert result["parsed"] == {"key": "value", "num": 42}

    def test_invalid_json(self):
        """Invalid JSON returns error."""
        result = ConfigValidator.validate_json("{key: value")
        assert result["valid"] is False
        assert result["error"] is not None
        assert result["parsed"] is None

    def test_json_array(self):
        """JSON array parses correctly."""
        result = ConfigValidator.validate_json('[1, 2, 3]')
        assert result["valid"] is True
        assert result["parsed"] == [1, 2, 3]

    def test_json_with_schema_valid(self):
        """JSON passes schema validation."""
        json_content = '{"name": "test"}'
        schema = {"type": "object", "required": ["name"]}
        result = ConfigValidator.validate_json(json_content, schema=schema)
        assert result["schema_valid"] is True

    def test_json_with_schema_invalid(self):
        """JSON fails schema validation."""
        json_content = '{"version": 1}'
        schema = {"type": "object", "required": ["name"]}
        result = ConfigValidator.validate_json(json_content, schema=schema)
        assert result["schema_valid"] is False


class TestValidateSystemdUnit:
    def test_valid_unit_file(self):
        """Complete systemd unit file passes validation."""
        content = """[Unit]
Description=Benchmark service

[Service]
ExecStart=/usr/bin/benchmark

[Install]
WantedBy=multi-user.target
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["valid"] is True
        assert result["error"] is None
        assert result["checks"]["has_exec_start"] is True

    def test_missing_unit_section(self):
        """Unit file without [Unit] fails."""
        content = """[Service]
ExecStart=/usr/bin/benchmark
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["valid"] is False
        assert "Missing required section: [Unit]" in result["error"]

    def test_missing_service_section(self):
        """Unit file without [Service] fails."""
        content = """[Unit]
Description=Test
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["valid"] is False
        assert "Missing required section: [Service]" in result["error"]

    def test_missing_exec_start(self):
        """Unit file without ExecStart fails."""
        content = """[Unit]
Description=Test

[Service]

[Install]
WantedBy=multi-user.target
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["valid"] is False
        assert "Missing required directive ExecStart" in result["error"]

    def test_empty_content(self):
        """Empty content returns validation with issues."""
        result = ConfigValidator.validate_systemd_unit("")
        assert result["valid"] is False

    def test_comments_ignored(self):
        """Comment lines are ignored."""
        content = """# This is a comment
[Unit]
; This is also a comment
Description=Test

[Service]
ExecStart=/usr/bin/benchmark

[Install]
WantedBy=multi-user.target
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["valid"] is True

    def test_checks_details(self):
        """Return dict includes detailed checks."""
        content = """[Unit]
Description=Test

[Service]
ExecStart=/usr/bin/benchmark

[Install]
WantedBy=multi-user.target
"""
        result = ConfigValidator.validate_systemd_unit(content)
        assert result["checks"]["has_unit_section"] is True
        assert result["checks"]["has_service_section"] is True
        assert result["checks"]["has_install_section"] is True
        assert result["details"]["sections_found"] == ["Install", "Service", "Unit"]


class TestValidateSystemdUnitViaAnalyze:
    def test_format_validation_fallback(self, tmp_path):
        """When systemd-analyze not available, falls back to format validation."""
        content = """[Unit]
Description=Test

[Service]
ExecStart=/usr/bin/benchmark

[Install]
WantedBy=multi-user.target
"""
        result = ConfigValidator.validate_systemd_unit_via_analyze(content, str(tmp_path))
        # Either systemd-analyze works or format validation is used
        assert "valid" in result
        assert "method" in result

    def test_invalid_via_analyze_fallback(self, tmp_path):
        """Invalid unit file returns invalid even via fallback."""
        content = """[Unit]
Description=Test
"""
        result = ConfigValidator.validate_systemd_unit_via_analyze(content, str(tmp_path))
        assert result["valid"] is False
