"""Tests for M22 — Config loader functions.

Covers load_yaml, load_model_config, get_model, get_suite, get_quantization,
load_suite_config, get_context_budget, etc.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.config import (
    get_context_budget,
    get_model,
    get_quantization,
    get_suite,
    load_model_config,
    load_suite_config,
    CONTEXT_BUDGETS,
)


def _write_config(tmp_path: Path, filename: str, data: dict) -> Path:
    """Write a YAML config file and return its path."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / filename
    config_path.write_text(yaml.dump(data, default_flow_style=False))
    return config_path


class TestLoadModelConfig:
    def test_load_model_config(self, tmp_path):
        """Load a valid models.yaml."""
        config_path = _write_config(tmp_path, "models.yaml", {
            "models": {
                "agent-code": {"base_url": "http://localhost:8000/v1", "model": "test-model"},
            }
        })
        # Temporarily change cwd so _find_config_dir finds our test configs
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            data = load_model_config()
        finally:
            os.chdir(orig_cwd)
        assert "models" in data
        assert "agent-code" in data["models"]

class TestGetModel:
    def test_get_model_found(self):
        """Lookup existing model returns its config."""
        config = {"models": {"m1": {"base_url": "http://x"}, "m2": {"base_url": "http://y"}}}
        assert get_model(config, "m1") == {"base_url": "http://x"}

    def test_get_model_not_found(self):
        """Lookup non-existent model returns None."""
        config = {"models": {"m1": {"base_url": "http://x"}}}
        assert get_model(config, "nonexistent") is None

    def test_get_model_no_models_key(self):
        """Config without 'models' key returns None."""
        assert get_model({}, "m1") is None


class TestGetQuantization:
    def test_get_quantization_from_config(self):
        """Quantization read from model config."""
        config = {"models": {"m1": {"quantization": "FP8"}}}
        assert get_quantization(config, "m1") == "FP8"

    def test_get_quantization_missing(self):
        """Model without quantization returns None."""
        config = {"models": {"m1": {}}}
        assert get_quantization(config, "m1") is None

    def test_get_quantization_none_model(self):
        """None model config returns None."""
        assert get_quantization(None, "m1") is None


class TestLoadSuiteConfig:
    def test_load_suite_config(self, tmp_path):
        """Load a valid suites.yaml."""
        config_path = _write_config(tmp_path, "suites.yaml", {
            "suites": {
                "smoke": {"task_dir": "tasks/smoke"},
            }
        })
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            data = load_suite_config()
        finally:
            os.chdir(orig_cwd)
        assert "suites" in data


class TestGetSuite:
    def test_get_suite_found(self):
        """Lookup existing suite."""
        config = {"suites": {"smoke": {"task_dir": "tasks/smoke"}}}
        assert get_suite(config, "smoke") == {"task_dir": "tasks/smoke"}

    def test_get_suite_normalized_hyphen(self):
        """Suite lookup normalizes hyphens."""
        config = {"suites": {"coding_benchmark": {"task_dir": "tasks/coding"}}}
        assert get_suite(config, "coding-benchmark") == {"task_dir": "tasks/coding"}

    def test_get_suite_not_found(self):
        """Non-existent suite returns None."""
        config = {"suites": {"smoke": {}}}
        assert get_suite(config, "nonexistent") is None

    def test_get_suite_no_suites_key(self):
        """Config without 'suites' key returns None."""
        assert get_suite({}, "smoke") is None


class TestGetContextBudget:
    def test_known_bucket_small(self):
        assert get_context_budget("small") == 1024

    def test_known_bucket_medium(self):
        assert get_context_budget("medium") == 4096

    def test_known_bucket_large(self):
        assert get_context_budget("large") == 16384

    def test_known_bucket_xlarge(self):
        assert get_context_budget("xlarge") == 65536

    def test_unknown_bucket_defaults_to_max(self):
        assert get_context_budget("gigantic") == 65536

    def test_custom_max_budget(self):
        assert get_context_budget("xlarge", max_budget=32768) == 32768

    def test_budget_capped_by_max(self):
        assert get_context_budget("xlarge", max_budget=30000) == 30000

    def test_context_budgets_constant(self):
        assert set(CONTEXT_BUDGETS.keys()) == {"small", "medium", "large", "xlarge"}
