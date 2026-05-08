"""Integration tests for runner lifecycle.

Covers:
  - RuntimeRunner creation and initialization
  - Runner factory — different backends (vLLM, TRT-LLM, llama.cpp, external)
  - Runner cleanup (shutdown)
  - Runner error handling — unavailable backends raise appropriate errors
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from subprocess import Popen
from unittest.mock import MagicMock, patch

import pytest

from bench_harness.runners.base import RuntimeRunner, ProcessHandle
from bench_harness.runners.openai_compatible import OpenAICompatibleRunner
from bench_harness.runners.vllm import VLLMRunner
from bench_harness.runners.trtllm import TRTLLMRunner
from bench_harness.runners.llamacpp import LlamaCPPRunner
from bench_harness.runners.external import ExternalRunner
from bench_harness.runners import RUNNER_REGISTRY, get_runner
from bench_harness.schemas.run_spec import (
    RunSpec, ArtifactSpec, ArtifactKind, RuntimeSpec, RuntimeKind,
    WorkloadSpec, LaunchMode,
)
from bench_harness.storage.config import StorageConfig


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_run_spec(**overrides: dict) -> RunSpec:
    """Build a minimal RunSpec for testing runner lifecycle."""
    kwargs = {
        "name": "test-run",
        "project": "test_project",
        "artifact": ArtifactSpec(
            kind=ArtifactKind.openai_endpoint,
            path="http://localhost:8000/v1",
            model_id="test-model",
        ),
        "runtime": RuntimeSpec(
            kind=RuntimeKind.openai_compatible,
            launch=LaunchMode.existing,
        ),
        "workload": WorkloadSpec(
            prompt_suite="smoke",
            max_tokens=64,
            temperature=0.0,
            num_runs=1,
            concurrency=1,
        ),
    }
    kwargs.update(overrides)
    return RunSpec(**kwargs)


def _make_storage_config(tmp_path: Path) -> StorageConfig:
    """Create a StorageConfig backed by a temporary directory."""
    root = tmp_path / "llm-bench-storage"
    root.mkdir()
    return StorageConfig(root=root, allow_unsafe=True)


# ── RuntimeRunner creation and initialization ─────────────────────────

class TestRuntimeRunnerCreation:
    def test_openai_compatible_runner_creation(self, tmp_path: Path):
        """OpenAICompatibleRunner is created with config."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        assert isinstance(runner, RuntimeRunner)
        assert runner.config is config
        assert runner.kind == "openai_compatible"

    def test_vllm_runner_creation(self, tmp_path: Path):
        """VLLMRunner is created with config."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        assert isinstance(runner, RuntimeRunner)
        assert runner.config is config
        assert runner.kind == "vllm"

    def test_external_runner_creation(self, tmp_path: Path):
        """ExternalRunner is created with config."""
        config = _make_storage_config(tmp_path)
        runner = ExternalRunner(config)
        assert isinstance(runner, RuntimeRunner)
        assert runner.config is config
        assert runner.kind == "external"

    def test_runner_has_required_methods(self, tmp_path: Path):
        """RuntimeRunner subclass has all required abstract methods."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        for method in ("prepare", "launch", "wait_until_ready",
                       "run_workload", "collect_logs", "shutdown"):
            assert hasattr(runner, method), f"Missing method: {method}"
            assert callable(getattr(runner, method))


# ── Runner factory ────────────────────────────────────────────────────

class TestRunnerFactory:
    def test_get_runner_openai_compatible(self, tmp_path: Path):
        """get_runner returns OpenAICompatibleRunner for kind 'openai_compatible'."""
        config = _make_storage_config(tmp_path)
        runner = get_runner("openai_compatible", config)
        assert isinstance(runner, OpenAICompatibleRunner)

    def test_get_runner_vllm(self, tmp_path: Path):
        """get_runner returns VLLMRunner for kind 'vllm'."""
        config = _make_storage_config(tmp_path)
        runner = get_runner("vllm", config)
        assert isinstance(runner, VLLMRunner)

    def test_get_runner_external(self, tmp_path: Path):
        """get_runner returns ExternalRunner for kind 'external'."""
        config = _make_storage_config(tmp_path)
        runner = get_runner("external", config)
        assert isinstance(runner, ExternalRunner)

    def test_get_runner_unknown_raises(self, tmp_path: Path):
        """get_runner raises ValueError for unknown kind."""
        config = _make_storage_config(tmp_path)
        with pytest.raises(ValueError, match="Unknown runner kind"):
            get_runner("nonexistent_backend", config)

    def test_runner_registry_contains_all_backends(self):
        """RUNNER_REGISTRY contains expected backend kinds."""
        expected = {"openai_compatible", "vllm", "trtllm", "llamacpp", "external"}
        assert set(RUNNER_REGISTRY.keys()) == expected

    def test_each_registry_entry_is_callable(self):
        """Every entry in RUNNER_REGISTRY is a class."""
        for kind, cls in RUNNER_REGISTRY.items():
            assert isinstance(cls, type), f"{kind} is not a class"

    def test_trtllm_runner_creation(self, tmp_path: Path):
        """TRTLLMRunner is created (prepare raises when used)."""
        config = _make_storage_config(tmp_path)
        runner = TRTLLMRunner(config)
        assert isinstance(runner, RuntimeRunner)
        assert runner.kind == "trtllm"

    def test_llamacpp_runner_creation(self, tmp_path: Path):
        """LlamaCPPRunner is created (prepare raises when used)."""
        config = _make_storage_config(tmp_path)
        runner = LlamaCPPRunner(config)
        assert isinstance(runner, RuntimeRunner)
        assert runner.kind == "llamacpp"


# ── Runner lifecycle ──────────────────────────────────────────────────

class TestRunnerLifecycle:
    def test_openai_compatible_lifecycle(self, tmp_path: Path):
        """Full lifecycle with OpenAICompatibleRunner."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.openai_compatible,
                launch=LaunchMode.existing,
                host="localhost",
                port=8000,
            )
        )

        prep = runner.prepare(spec)
        assert isinstance(prep, dict)
        assert "endpoint" in prep
        assert "localhost" in prep["endpoint"]

        handle = runner.launch(spec, prep)
        assert handle is None

        ready = runner.wait_until_ready(spec, prep, timeout=1.0)
        assert ready is False

        runner.shutdown(spec, prep, None)

    def test_vllm_managed_lifecycle(self, tmp_path: Path):
        """VLLMRunner with managed process generates command."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.vllm,
                launch=LaunchMode.managed_process,
                host="127.0.0.1",
                port=8000,
                model_name="test-vllm-model",
                args={"max_model_len": 4096},
            )
        )

        prep = runner.prepare(spec)
        assert "vllm_command" in prep
        assert "serve" in prep["vllm_command"]
        assert prep["vllm_command"][0] == "vllm"
        assert "endpoint" in prep

    def test_vllm_launch_returns_process_handle(self, tmp_path: Path):
        """VLLMRunner.launch with managed_process returns ProcessHandle."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.vllm,
                launch=LaunchMode.managed_process,
                host="127.0.0.1",
                port=8001,
                model_name="test-vllm",
            )
        )
        prep = runner.prepare(spec)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock(spec=Popen)
            mock_popen.return_value = mock_proc
            handle = runner.launch(spec, prep)
            assert handle is not None
            assert isinstance(handle, ProcessHandle)
            assert handle.proc == mock_proc
            assert handle.port == 8001

    def test_external_runner_prepare_adds_command(self, tmp_path: Path):
        """ExternalRunner.prepare adds producing_command to prep dict."""
        config = _make_storage_config(tmp_path)
        runner = ExternalRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.external,
                launch=LaunchMode.existing,
                args={"producing_command": "docker run my-server"},
            )
        )
        prep = runner.prepare(spec)
        assert "producing_command" in prep
        assert prep["producing_command"] == "docker run my-server"


# ── Runner cleanup (shutdown) ─────────────────────────────────────────

class TestRunnerCleanup:
    def test_openai_compatible_shutdown_noop(self, tmp_path: Path):
        """OpenAICompatibleRunner.shutdown does nothing."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        spec = _make_run_spec()
        prep = runner.prepare(spec)
        runner.shutdown(spec, prep, None)
        runner.shutdown(spec, prep, MagicMock())

    def test_vllm_shutdown_terminates_process(self, tmp_path: Path):
        """VLLMRunner.shutdown terminates the process handle."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.vllm,
                launch=LaunchMode.managed_process,
                host="127.0.0.1",
                port=8000,
            )
        )
        prep = runner.prepare(spec)

        mock_proc = MagicMock(spec=Popen)
        mock_proc.poll.return_value = None  # still running
        handle = ProcessHandle(
            proc=mock_proc, host="127.0.0.1", port=8000,
            ready_url="http://127.0.0.1:8000/v1",
        )

        runner.shutdown(spec, prep, handle)
        mock_proc.terminate.assert_called_once()

    def test_vllm_shutdown_with_dead_process(self, tmp_path: Path):
        """VLLMRunner.shutdown does nothing if process already exited."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        spec = _make_run_spec()
        prep = {}
        mock_proc = MagicMock(spec=Popen)
        mock_proc.poll.return_value = 1  # already exited
        handle = ProcessHandle(
            proc=mock_proc, host="127.0.0.1", port=8000,
            ready_url="http://127.0.0.1:8000/v1",
        )

        runner.shutdown(spec, prep, handle)
        mock_proc.terminate.assert_not_called()


# ── Runner error handling ─────────────────────────────────────────────

class TestRunnerErrorHandling:
    def test_trtllm_prepare_raises(self, tmp_path: Path):
        """TRTLLMRunner.prepare raises RuntimeError."""
        config = _make_storage_config(tmp_path)
        runner = TRTLLMRunner(config)
        spec = _make_run_spec()
        with pytest.raises(RuntimeError, match="tensorrt_llm"):
            runner.prepare(spec)

    def test_trtllm_run_workload_raises(self, tmp_path: Path):
        """TRTLLMRunner.run_workload raises RuntimeError."""
        config = _make_storage_config(tmp_path)
        runner = TRTLLMRunner(config)
        spec = _make_run_spec()
        with pytest.raises(RuntimeError, match="not available"):
            runner.run_workload(spec, {}, Path(tmp_path))

    def test_llamacpp_prepare_raises(self, tmp_path: Path):
        """LlamaCPPRunner.prepare raises RuntimeError."""
        config = _make_storage_config(tmp_path)
        runner = LlamaCPPRunner(config)
        spec = _make_run_spec()
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            runner.prepare(spec)

    def test_llamacpp_run_workload_raises(self, tmp_path: Path):
        """LlamaCPPRunner.run_workload raises RuntimeError."""
        config = _make_storage_config(tmp_path)
        runner = LlamaCPPRunner(config)
        spec = _make_run_spec()
        with pytest.raises(RuntimeError, match="not available"):
            runner.run_workload(spec, {}, Path(tmp_path))

    def test_vllm_launch_non_managed_returns_none(self, tmp_path: Path):
        """VLLMRunner.launch with existing mode returns None."""
        config = _make_storage_config(tmp_path)
        runner = VLLMRunner(config)
        spec = _make_run_spec(
            runtime=RuntimeSpec(
                kind=RuntimeKind.vllm,
                launch=LaunchMode.existing,
            )
        )
        # launch() checks launch mode before calling prepare,
        # so it returns None directly for non-managed mode.
        handle = runner.launch(spec, {})
        assert handle is None

    def test_trtllm_wait_until_ready_fails(self, tmp_path: Path):
        """TRTLLMRunner.wait_until_ready returns False."""
        config = _make_storage_config(tmp_path)
        runner = TRTLLMRunner(config)
        spec = _make_run_spec()
        ready = runner.wait_until_ready(spec, {}, timeout=1.0)
        assert ready is False

    def test_llamacpp_wait_until_ready_fails(self, tmp_path: Path):
        """LlamaCPPRunner.wait_until_ready returns False."""
        config = _make_storage_config(tmp_path)
        runner = LlamaCPPRunner(config)
        spec = _make_run_spec()
        ready = runner.wait_until_ready(spec, {}, timeout=1.0)
        assert ready is False

    def test_trtllm_collect_logs_returns_empty(self, tmp_path: Path):
        """TRTLLMRunner.collect_logs returns empty dict."""
        config = _make_storage_config(tmp_path)
        runner = TRTLLMRunner(config)
        spec = _make_run_spec()
        logs = runner.collect_logs(spec, {}, Path(tmp_path))
        assert logs == {}

    def test_llamacpp_collect_logs_returns_empty(self, tmp_path: Path):
        """LlamaCPPRunner.collect_logs returns empty dict."""
        config = _make_storage_config(tmp_path)
        runner = LlamaCPPRunner(config)
        spec = _make_run_spec()
        logs = runner.collect_logs(spec, {}, Path(tmp_path))
        assert logs == {}


# ── StorageConfig integration with runners ────────────────────────────

class TestRunnerStorageConfig:
    def test_runner_receives_valid_storage_config(self, tmp_path: Path):
        """Runner stores the provided StorageConfig."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        assert runner.config.root == config.root

    def test_runner_config_has_namespace_properties(self, tmp_path: Path):
        """StorageConfig used by runner exposes all namespace properties."""
        config = _make_storage_config(tmp_path)
        runner = OpenAICompatibleRunner(config)
        assert runner.config.artifacts_root.parent == config.root
        assert runner.config.results_root.parent == config.root
        assert runner.config.registry_root.parent == config.root
        assert runner.config.logs_root.parent == config.root
        assert runner.config.cache_root.parent == config.root
        assert runner.config.tmp_root.parent == config.root

    def test_runner_factory_with_different_configs(self, tmp_path: Path):
        """Different StorageConfigs produce different runner instances."""
        # Use mkdir(parents=True) so nested subdirs are created
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        config_a = _make_storage_config(tmp_path / "a")
        config_b = _make_storage_config(tmp_path / "b")
        runner_a = get_runner("openai_compatible", config_a)
        runner_b = get_runner("openai_compatible", config_b)
        assert runner_a.config.root != runner_b.config.root
