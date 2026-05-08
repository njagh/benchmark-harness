"""Tests for the runner factory/registry.

Covers:
  - get_runner for all registered backends
  - Unknown backend raises ValueError
  - RUNNER_REGISTRY completeness check
  - Each runner kind produces the correct class
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bench_harness.runners import RUNNER_REGISTRY, get_runner
from bench_harness.runners.base import RuntimeRunner
from bench_harness.runners.openai_compatible import OpenAICompatibleRunner
from bench_harness.runners.vllm import VLLMRunner
from bench_harness.runners.trtllm import TRTLLMRunner
from bench_harness.runners.llamacpp import LlamaCPPRunner
from bench_harness.runners.external import ExternalRunner


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    """Create a minimal mock StorageConfig."""
    config = MagicMock()
    config.registry_root = MagicMock()
    config.registry_root.mkdir = MagicMock()
    config.registry_root.joinpath = MagicMock()
    return config


# ── RUNNER_REGISTRY completeness ─────────────────────────────────────


class TestRunnerRegistry:
    """Test that RUNNER_REGISTRY has all expected backends."""

    def test_registry_is_dict(self):
        assert isinstance(RUNNER_REGISTRY, dict)

    def test_registry_known_keys(self):
        expected_keys = {"openai_compatible", "vllm", "trtllm", "llamacpp", "external"}
        assert set(RUNNER_REGISTRY.keys()) == expected_keys

    def test_registry_values_are_classes(self):
        for kind, cls in RUNNER_REGISTRY.items():
            assert isinstance(cls, type), f"{kind} maps to non-class: {cls}"

    def test_registry_subclasses_runtime_runner(self):
        for kind, cls in RUNNER_REGISTRY.items():
            assert issubclass(cls, RuntimeRunner), (
                f"{kind} ({cls.__name__}) is not a subclass of RuntimeRunner"
            )

    def test_registry_no_duplicates(self):
        classes = list(RUNNER_REGISTRY.values())
        assert len(classes) == len(set(classes)), "Duplicate classes in registry"

    def test_registry_all_kinds_have_property(self):
        for kind, cls in RUNNER_REGISTRY.items():
            # Each class must define a `kind` property
            assert hasattr(cls, "kind"), f"{cls.__name__} missing 'kind' property"


# ── get_runner — backend-specific ─────────────────────────────────────


class TestGetRunner:
    """Test get_runner for all backend kinds."""

    def test_openai_compatible(self, mock_config):
        runner = get_runner("openai_compatible", mock_config)
        assert isinstance(runner, OpenAICompatibleRunner)
        assert runner.kind == "openai_compatible"

    def test_vllm(self, mock_config):
        runner = get_runner("vllm", mock_config)
        assert isinstance(runner, VLLMRunner)
        assert runner.kind == "vllm"

    def test_trtllm(self, mock_config):
        runner = get_runner("trtllm", mock_config)
        assert isinstance(runner, TRTLLMRunner)
        assert runner.kind == "trtllm"

    def test_llamacpp(self, mock_config):
        runner = get_runner("llamacpp", mock_config)
        assert isinstance(runner, LlamaCPPRunner)
        assert runner.kind == "llamacpp"

    def test_external(self, mock_config):
        runner = get_runner("external", mock_config)
        assert isinstance(runner, ExternalRunner)
        assert runner.kind == "external"

    def test_external_is_openai_subclass(self, mock_config):
        runner = get_runner("external", mock_config)
        assert isinstance(runner, OpenAICompatibleRunner), (
            "ExternalRunner should be a subclass of OpenAICompatibleRunner"
        )

    def test_get_runner_passes_config(self, mock_config):
        runner = get_runner("vllm", mock_config)
        assert runner.config is mock_config


# ── Unknown backend ──────────────────────────────────────────────────


class TestUnknownBackend:
    """Test that unknown backend kinds raise appropriate errors."""

    def test_unknown_backend_raises_value_error(self, mock_config):
        with pytest.raises(ValueError) as exc_info:
            get_runner("unknown_backend", mock_config)
        assert "Unknown runner kind" in str(exc_info.value)
        assert "unknown_backend" in str(exc_info.value)

    def test_unknown_backend_mentions_available(self, mock_config):
        with pytest.raises(ValueError) as exc_info:
            get_runner("fake_kind", mock_config)
        available = str(exc_info.value)
        for key in RUNNER_REGISTRY:
            assert key in available, f"Registry key '{key}' not mentioned in error message"

    def test_unknown_backend_case_sensitive(self, mock_config):
        with pytest.raises(ValueError):
            get_runner("OpenAI_Compatible", mock_config)


# ── Runner kind property consistency ──────────────────────────────────


class TestRunnerKindConsistency:
    """Ensure each runner's kind property matches its registry key."""

    def test_registry_keys_match_kind_properties(self, mock_config):
        for key, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert runner.kind == key, (
                f"Registry key '{key}' does not match kind property of {cls.__name__}"
            )


# ── Runner initialization ────────────────────────────────────────────


class TestRunnerInitialization:
    """Test that all runners initialize correctly with a config."""

    def test_all_runners_instantiate(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert runner is not None
            assert runner.config is mock_config
            assert hasattr(runner, "kind")
            assert runner.kind == kind

    def test_runner_prepare_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "prepare")
            assert callable(runner.prepare)

    def test_runner_launch_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "launch")
            assert callable(runner.launch)

    def test_runner_wait_until_ready_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "wait_until_ready")
            assert callable(runner.wait_until_ready)

    def test_runner_run_workload_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "run_workload")
            assert callable(runner.run_workload)

    def test_runner_collect_logs_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "collect_logs")
            assert callable(runner.collect_logs)

    def test_runner_shutdown_exists(self, mock_config):
        for kind, cls in RUNNER_REGISTRY.items():
            runner = cls(mock_config)
            assert hasattr(runner, "shutdown")
            assert callable(runner.shutdown)


# ── get_runner idempotency ───────────────────────────────────────────


class TestGetRunnerIdempotency:
    """Test that get_runner returns fresh instances each time."""

    def test_get_runner_returns_new_instance(self, mock_config):
        r1 = get_runner("vllm", mock_config)
        r2 = get_runner("vllm", mock_config)
        assert r1 is not r2, "get_runner should return a new instance each time"

    def test_get_runner_same_config(self, mock_config):
        r1 = get_runner("vllm", mock_config)
        r2 = get_runner("vllm", mock_config)
        assert r1.config is r2.config
