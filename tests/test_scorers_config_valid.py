"""Comprehensive tests for ConfigValidScorer — validates generated config file correctness."""

from __future__ import annotations

from bench_harness.scorers.config_valid import ConfigValidScorer
from bench_harness.tasks.task_schema import Task


# ── Helpers ────────────────────────────────────────────────────────────────

def make_task(expected_attrs=None, **overrides):
    """Build a minimal Task and optionally attach extra attributes on expected."""
    d = {
        "id": "test.config.task_001",
        "family": "general",
        "expected": {"type": "exact", "answer": "test"},
        "scoring": {"primary": "config_valid"},
    }
    d.update(overrides)
    task = Task.model_validate(d)
    if expected_attrs is not None:
        cls = type("FakeExpected", (), expected_attrs)
        task.expected = cls()
    return task


VALID_YAML = "name: test\nversion: 1\n"

VALID_JSON = '{"name": "test", "version": 1}'

VALID_SYSTEMD = """[Unit]
Description=Benchmark service

[Service]
ExecStart=/usr/bin/benchmark

[Install]
WantedBy=multi-user.target
"""

VALID_INI = """[general]
name = test
version = 1
"""

INVALID_YAML = "key: [unclosed"

INVALID_JSON_STR = '{key: value'

INVALID_SYSTEMD = """[Unit]
Description=Test
"""

MALFORMED_INI = "[section\nnot valid"


# ── TestConfigValidScorerInit ───────────────────────────────────────────────

class TestConfigValidScorerInit:
    def test_init_default_kwargs(self):
        """No args → default init with empty _config."""
        scorer = ConfigValidScorer()
        assert scorer._config == {}
        assert scorer.name == "config_valid"
        assert scorer.version == "1.0"

    def test_init_with_custom_kwargs(self):
        """Pass kwargs → stored in _config."""
        scorer = ConfigValidScorer(strict=True, timeout=30)
        assert scorer._config["strict"] is True
        assert scorer._config["timeout"] == 30


# ── TestConfigValidScorerValidateTask ──────────────────────────────────────

class TestConfigValidScorerValidateTask:
    def test_validate_true_yaml(self):
        """Task with expected.config_type = 'yaml' → True."""
        task = make_task(expected_attrs={"config_type": "yaml"})
        scorer = ConfigValidScorer()
        assert scorer.validate_task(task) is True

    def test_validate_true_json(self):
        """config_type = 'json' → True."""
        task = make_task(expected_attrs={"config_type": "json"})
        scorer = ConfigValidScorer()
        assert scorer.validate_task(task) is True

    def test_validate_false_no_config_type(self):
        """No config_type → False."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        scorer = ConfigValidScorer()
        assert scorer.validate_task(task) is False

    def test_validate_false_dict_style(self):
        """Plain dict expected with config_type → True via dict path."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        # Replace expected with a dict containing config_type
        task.expected = {"config_type": "yaml", "type": "exact", "answer": "a"}
        scorer = ConfigValidScorer()
        assert scorer.validate_task(task) is True


# ── TestConfigValidScorerScore ─────────────────────────────────────────────

class TestConfigValidScorerScore:
    def test_score_no_config_type(self):
        """Task without config_type → score=0.0, 'No config_type specified'."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        scorer = ConfigValidScorer()
        result = scorer.score(task, "some content")
        assert result.score == 0.0
        assert result.passed is False
        assert "No config_type specified" in result.explanation

    def test_score_unsupported_type(self):
        """config_type = 'toml' → None result, score=0.0."""
        task = make_task(expected_attrs={"config_type": "toml"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, "[tool]")
        assert result.score == 0.0
        assert result.passed is False
        assert "Unsupported config_type" in result.explanation

    def test_score_yaml_valid(self):
        """Valid YAML content → passes."""
        task = make_task(expected_attrs={"config_type": "yaml"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_YAML)
        assert result.details["valid"] is True
        assert result.details["config_type"] == "yaml"

    def test_score_yaml_invalid(self):
        """Invalid YAML → valid=False, lower score."""
        task = make_task(expected_attrs={"config_type": "yaml"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, INVALID_YAML)
        assert result.details["valid"] is False
        assert "error" in result.details
        assert result.score < 0.5

    def test_score_yaml_with_required_fields(self):
        """YAML with all required fields → high score."""
        task = make_task(
            expected_attrs={
                "config_type": "yaml",
                "required_fields": ["name", "version"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_YAML)
        assert result.details["valid"] is True
        fields = result.details["fields_checked"]
        assert fields["name"] is True
        assert fields["version"] is True
        assert result.score >= 0.8
        assert result.passed is True

    def test_score_yaml_with_patterns(self):
        """YAML matching pattern specs → higher score."""
        task = make_task(
            expected_attrs={
                "config_type": "yaml",
                "patterns": [{"pattern": "name:"}],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_YAML)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("name:") is True

    def test_score_yaml_list_parsed(self):
        """YAML parsed as list (non-dict) → fields_checked has parsed_as_dict=False."""
        yaml_list = "- name: item1\n- name: item2\n"
        task = make_task(
            expected_attrs={
                "config_type": "yaml",
                "required_fields": ["name"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, yaml_list)
        assert result.details["valid"] is True
        assert result.details["fields_checked"]["parsed_as_dict"] is False

    def test_score_yaml_string_pattern(self):
        """YAML with string (non-dict) pattern specs → string fallback path."""
        task = make_task(
            expected_attrs={
                "config_type": "yaml",
                "patterns": ["name:"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_YAML)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("name:") is True

    def test_score_yaml_re_error_fallback(self):
        """YAML with invalid regex pattern → falls back to substring match."""
        task = make_task(
            expected_attrs={
                "config_type": "yaml",
                "patterns": ["[unclosed"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_YAML)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("[unclosed") is False

    def test_score_json_valid(self):
        """Valid JSON → passes."""
        task = make_task(expected_attrs={"config_type": "json"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_JSON)
        assert result.details["valid"] is True
        assert result.details["config_type"] == "json"

    def test_score_json_invalid(self):
        """Invalid JSON → fails."""
        task = make_task(expected_attrs={"config_type": "json"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, INVALID_JSON_STR)
        assert result.details["valid"] is False
        assert "error" in result.details

    def test_score_json_list_parsed(self):
        """JSON parsed as list (non-dict) → fields_checked has parsed_as_dict=False."""
        json_list = '[1, 2, 3]'
        task = make_task(
            expected_attrs={
                "config_type": "json",
                "required_fields": ["name"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, json_list)
        assert result.details["valid"] is True
        assert result.details["fields_checked"]["parsed_as_dict"] is False

    def test_score_json_string_pattern(self):
        """JSON with string (non-dict) pattern specs → string fallback path."""
        task = make_task(
            expected_attrs={
                "config_type": "json",
                "patterns": ["name"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_JSON)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("name") is True

    def test_score_json_with_required_fields(self):
        """JSON with all required fields present → high score with fields_checked populated."""
        task = make_task(
            expected_attrs={
                "config_type": "json",
                "required_fields": ["name", "version"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_JSON)
        assert result.details["valid"] is True
        fields = result.details["fields_checked"]
        assert fields["name"] is True
        assert fields["version"] is True

    def test_score_systemd_valid(self):
        """Valid systemd unit file → passes."""
        task = make_task(expected_attrs={"config_type": "systemd"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_SYSTEMD)
        assert result.details["valid"] is True
        fields = result.details["fields_checked"]
        assert fields["has_unit_section"] is True
        assert fields["has_service_section"] is True
        assert fields["has_exec_start"] is True

    def test_score_systemd_missing_exec_start(self):
        """Missing ExecStart → fails format."""
        task = make_task(expected_attrs={"config_type": "systemd"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, INVALID_SYSTEMD)
        assert result.details["valid"] is False
        assert result.details["fields_checked"]["has_exec_start"] is False

    def test_score_systemd_with_patterns(self):
        """Systemd with pattern specs → patterns_checked populated."""
        task = make_task(
            expected_attrs={
                "config_type": "systemd",
                "patterns": [{"pattern": "ExecStart"}],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_SYSTEMD)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("ExecStart") is True

    def test_score_systemd_string_pattern(self):
        """Systemd with string (non-dict) pattern specs → string fallback."""
        task = make_task(
            expected_attrs={
                "config_type": "systemd",
                "patterns": ["ExecStart"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_SYSTEMD)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("ExecStart") is True

    def test_score_systemd_re_error_fallback(self):
        """Systemd with invalid regex → falls back to substring match."""
        task = make_task(
            expected_attrs={
                "config_type": "systemd",
                "patterns": ["[bad"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_SYSTEMD)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("[bad") is False

    def test_score_ini_valid(self):
        """Valid INI → passes."""
        task = make_task(expected_attrs={"config_type": "ini"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_INI)
        assert result.details["valid"] is True
        assert result.details["fields_checked"]["has_sections"] is True

    def test_score_ini_with_required_fields(self):
        """INI with required fields → fields_checked populated."""
        task = make_task(
            expected_attrs={
                "config_type": "ini",
                "required_fields": ["name", "version"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_INI)
        assert result.details["valid"] is True
        fields = result.details["fields_checked"]
        assert fields["name"] is True
        assert fields["version"] is True

    def test_score_ini_with_patterns(self):
        """INI with pattern specs → patterns_checked populated."""
        task = make_task(
            expected_attrs={
                "config_type": "ini",
                "patterns": [{"pattern": "name"}],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_INI)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("name") is True

    def test_score_ini_string_pattern(self):
        """INI with string (non-dict) pattern specs → string fallback."""
        task = make_task(
            expected_attrs={
                "config_type": "ini",
                "patterns": ["name"],
            },
        )
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_INI)
        assert result.details["valid"] is True
        patterns = result.details["patterns_checked"]
        assert patterns.get("name") is True

    def test_score_ini_parse_error(self):
        """Malformed INI → parse error."""
        task = make_task(expected_attrs={"config_type": "ini"})
        scorer = ConfigValidScorer()
        result = scorer.score(task, MALFORMED_INI)
        assert result.details["valid"] is False

    def test_score_dict_style_config_type(self):
        """Dict-style expected with config_type → routed correctly."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        task.expected = {"config_type": "json", "type": "exact", "answer": "a"}
        scorer = ConfigValidScorer()
        result = scorer.score(task, VALID_JSON)
        assert result.details["valid"] is True
        assert result.details["config_type"] == "json"


# ── TestConfigValidScorerHelperMethods ──────────────────────────────────────

class TestConfigValidScorerHelperMethods:
    def test_get_expected_fields_list(self):
        """expected.required_fields is a list → returns the list."""
        fake = type("FakeExpected", (), {"required_fields": ["name", "version"]})()
        scorer = ConfigValidScorer()
        result = scorer._get_expected_fields(fake)
        assert result == ["name", "version"]

    def test_get_expected_fields_dict(self):
        """expected.required_fields is a dict → extracts keys."""
        fake = type("FakeExpected", (), {"required_fields": {"name": str, "version": int}})()
        scorer = ConfigValidScorer()
        result = scorer._get_expected_fields(fake)
        assert sorted(result) == ["name", "version"]

    def test_get_expected_fields_none(self):
        """No required_fields → None."""
        fake = type("FakeExpected", (), {})()
        scorer = ConfigValidScorer()
        result = scorer._get_expected_fields(fake)
        assert result is None

    def test_get_expected_fields_dict(self):
        """Dict-style expected with required_fields as dict → extracts keys."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        task.expected = {"required_fields": {"name": str, "version": int}}
        scorer = ConfigValidScorer()
        result = scorer._get_expected_fields(task.expected)
        assert sorted(result) == ["name", "version"]

    def test_get_pattern_specs(self):
        """patterns as list → returns list."""
        fake = type("FakeExpected", (), {"patterns": [{"pattern": "name:"}]})()
        scorer = ConfigValidScorer()
        result = scorer._get_pattern_specs(fake)
        assert result == [{"pattern": "name:"}]

    def test_get_pattern_specs_none(self):
        """No patterns → None."""
        fake = type("FakeExpected", (), {})()
        scorer = ConfigValidScorer()
        result = scorer._get_pattern_specs(fake)
        assert result is None

    def test_get_pattern_specs_dict_style(self):
        """Dict-style expected with patterns as list → returns list."""
        task = Task.model_validate({
            "id": "t", "family": "f",
            "expected": {"type": "exact", "answer": "a"},
            "scoring": {"primary": "config_valid"},
        })
        task.expected = {"patterns": [{"pattern": "version:"}]}
        scorer = ConfigValidScorer()
        result = scorer._get_pattern_specs(task.expected)
        assert result == [{"pattern": "version:"}]

    def test_get_pattern_specs_string_pattern(self):
        """Pattern specs list with non-dict entries → string pattern fallback."""
        fake = type("FakeExpected", (), {"patterns": ["some_pattern"]})()
        scorer = ConfigValidScorer()
        result = scorer._get_pattern_specs(fake)
        assert result == ["some_pattern"]


# ── TestConfigValidScorerValidation ─────────────────────────────────────────

class TestConfigValidScorerValidation:
    def test_validate_yaml_config_direct(self):
        """Direct test of _validate_yaml_config method with valid YAML."""
        task = make_task(expected_attrs={"config_type": "yaml"})
        scorer = ConfigValidScorer()
        result = scorer._validate_yaml_config(VALID_YAML, task.expected)
        assert result["valid"] is True
        assert result["error"] is None

    def test_validate_systemd_config_direct(self):
        """Direct test of _validate_systemd_config method with valid systemd."""
        task = make_task(expected_attrs={"config_type": "systemd"})
        scorer = ConfigValidScorer()
        result = scorer._validate_systemd_config(VALID_SYSTEMD, task.expected)
        assert result["valid"] is True
        fields = result["fields_checked"]
        assert fields["has_unit_section"] is True
        assert fields["has_service_section"] is True
        assert fields["has_install_section"] is True
        assert fields["has_exec_start"] is True
