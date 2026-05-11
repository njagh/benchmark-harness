"""Tests for the OpenAI-compatible runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bench_harness.runners.openai_compatible import OpenAICompatibleRunner
from bench_harness.runners.base import ProcessHandle
from bench_harness.schemas.run_spec import RunSpec, ArtifactSpec, ArtifactKind, RuntimeSpec, RuntimeKind, LaunchMode, WorkloadSpec
from bench_harness.storage.config import StorageConfig


@pytest.fixture
def storage_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(root=tmp_path / "storage")


@pytest.fixture
def run_spec_existing(tmp_path: Path) -> RunSpec:
    return RunSpec(
        name="test-openai-runner",
        project="test_project",
        artifact=ArtifactSpec(
            kind=ArtifactKind.openai_endpoint,
            path="http://localhost:8000/v1",
            model_id="test-model",
        ),
        runtime=RuntimeSpec(
            kind=RuntimeKind.openai_compatible,
            launch=LaunchMode.existing,
            host="localhost",
            port=8000,
            model_name="test-model",
        ),
        workload=WorkloadSpec(
            prompt_suite="smoke",
            max_tokens=64,
            temperature=0.0,
            num_runs=1,
        ),
    )


@pytest.fixture
def run_spec_managed(tmp_path: Path) -> RunSpec:
    return RunSpec(
        name="test-openai-managed",
        project="test_project",
        artifact=ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            path="/models/test-model",
            model_id="test-model",
        ),
        runtime=RuntimeSpec(
            kind=RuntimeKind.openai_compatible,
            launch=LaunchMode.managed_process,
            host="localhost",
            port=8000,
            model_name="test-model",
        ),
        workload=WorkloadSpec(
            prompt_suite="smoke",
            max_tokens=64,
            temperature=0.0,
            num_runs=1,
        ),
    )


class TestOpenAICompatibleRunnerInit:
    """Test OpenAICompatibleRunner initialization in both launch modes."""

    def test_init_existing_launch(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        assert runner.kind == "openai_compatible"

    def test_init_managed_launch(self, storage_config: StorageConfig, run_spec_managed: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        assert runner.kind == "openai_compatible"

    def test_kind_property(self, storage_config: StorageConfig):
        runner = OpenAICompatibleRunner(config=storage_config)
        assert runner.kind == "openai_compatible"


class TestPrepare:
    """Test the prepare() method."""

    def test_prepare_with_host_and_port(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "test-model"}]}
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = runner.prepare(run_spec_existing)

            assert prep["endpoint"] == "http://localhost:8000/v1"
            assert prep["model_name"] == "test-model"
            assert prep["v1_models"] == {"data": [{"id": "test-model"}]}

    def test_prepare_without_host_and_port(self, storage_config: StorageConfig):
        spec = RunSpec(
            name="test-no-host",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.openai_endpoint,
                path="http://default:8000/v1",
                model_id="default-model",
            ),
            runtime=RuntimeSpec(
                kind=RuntimeKind.openai_compatible,
                launch=LaunchMode.existing,
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = runner.prepare(spec)

            assert prep["endpoint"] == "http://default:8000/v1"
            assert prep["model_name"] == "default-model"

    def test_prepare_fails_gracefully_on_http_error(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(side_effect=Exception("connection refused"))
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = runner.prepare(run_spec_existing)

            assert prep["v1_models"] is None


class TestLaunch:
    """Test the launch() method."""

    def test_launch_returns_none_for_existing(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
        handle = runner.launch(run_spec_existing, prep)
        assert handle is None

    def test_launch_returns_none_for_managed(self, storage_config: StorageConfig, run_spec_managed: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
        handle = runner.launch(run_spec_managed, prep)
        assert handle is None


class TestWaitUntilReady:
    """Test the wait_until_ready() method."""

    def test_ready_on_first_call(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.wait_until_ready(run_spec_existing, prep, timeout=5.0)
            assert result is True

    def test_ready_on_400_response(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.wait_until_ready(run_spec_existing, prep, timeout=5.0)
            assert result is True

    def test_ready_after_retries(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls, \
             patch("bench_harness.runners.openai_compatible.time.sleep"):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client = MagicMock()

            def post_side_effect(*args, **kwargs):
                if post_side_effect.call_count <= 2:
                    post_side_effect.call_count += 1
                    raise httpx.ConnectError("connection refused")
                post_side_effect.call_count += 1
                return mock_response

            post_side_effect.call_count = 0
            mock_client.post.side_effect = post_side_effect
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.wait_until_ready(run_spec_existing, prep, timeout=30.0)
            assert result is True

    def test_timeout_returns_false(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        with patch("bench_harness.runners.openai_compatible.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("connection refused")
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.wait_until_ready(run_spec_existing, prep, timeout=0.1)
            assert result is False


class TestCollectLogs:
    """Test the collect_logs() method."""

    def test_collect_logs_with_v1_models(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        runner = OpenAICompatibleRunner(config=storage_config)
        prep = {
            "endpoint": "http://localhost:8000/v1",
            "model_name": "test-model",
            "v1_models": {"data": [{"id": "model-1", "object": "model", "owned_by": "test"}]},
        }
        logs = runner.collect_logs(run_spec_existing, prep, tmp_path)
        assert "v1_models.json" in logs
        parsed = json.loads(logs["v1_models.json"])
        assert parsed == {"data": [{"id": "model-1", "object": "model", "owned_by": "test"}]}

    def test_collect_logs_without_v1_models(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        runner = OpenAICompatibleRunner(config=storage_config)
        prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
        logs = runner.collect_logs(run_spec_existing, prep, tmp_path)
        assert logs == {}

    def test_collect_logs_with_null_v1_models(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        runner = OpenAICompatibleRunner(config=storage_config)
        prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model", "v1_models": None}
        logs = runner.collect_logs(run_spec_existing, prep, tmp_path)
        assert logs == {}


class TestShutdown:
    """Test the shutdown() method (inherited from base class)."""

    def test_shutdown_noop_existing(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        runner.shutdown(run_spec_existing, {}, None)

    def test_shutdown_noop_with_handle(self, storage_config: StorageConfig, run_spec_existing: RunSpec):
        runner = OpenAICompatibleRunner(config=storage_config)
        mock_handle = MagicMock()
        runner.shutdown(run_spec_existing, {}, mock_handle)
        # Base class shutdown is a no-op for endpoint runners


class TestRunWorkload:
    """Test the run_workload() method with mocked dependencies."""

    def test_run_workload_success(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        mock_task_result = MagicMock()
        mock_task_result.prompt_tokens = 10
        mock_task_result.completion_tokens = 20
        mock_task_result.ttft_ms = 50.0
        mock_task_result.decode_ms = 150.0
        mock_task_result.total_wall_ms = 200.0
        mock_task_result.tokens_per_second = 133.33
        mock_task_result.tokens_per_second_total = 150.0
        mock_task_result.score_primary = 0.9

        mock_task = {"id": "task-001", "prompt": "test prompt"}

        with patch("bench_harness.runners.openai_compatible.OpenAICompatClient") as mock_client_cls, \
             patch("bench_harness.runners.openai_compatible.CompletionRunner") as mock_runner_cls, \
             patch("bench_harness.runners.openai_compatible.load_tasks") as mock_load_tasks, \
             patch("bench_harness.runners.openai_compatible.asyncio.run") as mock_asyncio_run:

            mock_client_cls.return_value = MagicMock()
            mock_runner_instance = MagicMock()
            mock_runner_instance.run = MagicMock(return_value=mock_task_result)
            mock_runner_cls.return_value = mock_runner_instance
            mock_load_tasks.return_value = [mock_task]
            mock_asyncio_run.return_value = mock_task_result

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {
                "endpoint": "http://localhost:8000/v1",
                "model_name": "test-model",
            }
            result = runner.run_workload(run_spec_existing, prep, tmp_path)

            assert result.run_id == run_spec_existing.name
            assert len(result.per_request) == 1
            req = result.per_request[0]
            assert req.request_id == "task-001-0"
            assert req.prompt_id == "task-001"
            assert req.prompt_tokens == 10
            assert req.generated_tokens == 20
            assert req.finish_reason == "stop"
            assert req.error is None
            assert req.quality_score == 0.9
            assert result.summary is not None

    def test_run_workload_no_tasks(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        with patch("bench_harness.runners.openai_compatible.OpenAICompatClient") as mock_client_cls, \
             patch("bench_harness.runners.openai_compatible.CompletionRunner") as mock_runner_cls, \
             patch("bench_harness.runners.openai_compatible.load_tasks") as mock_load_tasks, \
             patch("bench_harness.runners.openai_compatible.asyncio.run") as mock_asyncio_run:

            mock_client_cls.return_value = MagicMock()
            mock_runner_cls.return_value = MagicMock()
            mock_load_tasks.return_value = []
            mock_asyncio_run.return_value = MagicMock()

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.run_workload(run_spec_existing, prep, tmp_path)

            assert len(result.per_request) == 0
            assert result.summary is not None

    def test_run_workload_with_task_dir(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        mock_task_result = MagicMock()
        mock_task_result.prompt_tokens = 5
        mock_task_result.completion_tokens = 10
        mock_task_result.ttft_ms = 20.0
        mock_task_result.decode_ms = 80.0
        mock_task_result.total_wall_ms = 100.0
        mock_task_result.tokens_per_second = 125.0
        mock_task_result.tokens_per_second_total = 150.0
        mock_task_result.score_primary = None

        mock_task = {"id": "task-dir-1"}
        task_dir = tmp_path / "tasks" / "smoke"
        task_dir.mkdir(parents=True)

        workload_with_task_dir = RunSpec(
            name="test-task-dir",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.openai_endpoint,
                path="http://localhost:8000/v1",
            ),
            runtime=RuntimeSpec(
                kind=RuntimeKind.openai_compatible,
                launch=LaunchMode.existing,
            ),
            workload=WorkloadSpec(
                prompt_suite="smoke",
                task_dir=str(task_dir),
                max_tokens=32,
                temperature=0.0,
                num_runs=1,
            ),
        )

        with patch("bench_harness.runners.openai_compatible.OpenAICompatClient") as mock_client_cls, \
             patch("bench_harness.runners.openai_compatible.CompletionRunner") as mock_runner_cls, \
             patch("bench_harness.runners.openai_compatible.load_tasks") as mock_load_tasks, \
             patch("bench_harness.runners.openai_compatible.asyncio.run") as mock_asyncio_run:

            mock_client_cls.return_value = MagicMock()
            mock_runner_instance = MagicMock()
            mock_runner_instance.run = MagicMock(return_value=mock_task_result)
            mock_runner_cls.return_value = mock_runner_instance
            mock_load_tasks.return_value = [mock_task]
            mock_asyncio_run.return_value = mock_task_result

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.run_workload(workload_with_task_dir, prep, tmp_path)

            assert len(result.per_request) == 1
            assert result.per_request[0].prompt_id == "task-dir-1"

    def test_run_workload_multiple_runs(self, storage_config: StorageConfig, run_spec_existing: RunSpec, tmp_path: Path):
        mock_task_result = MagicMock()
        mock_task_result.prompt_tokens = 10
        mock_task_result.completion_tokens = 20
        mock_task_result.ttft_ms = 50.0
        mock_task_result.decode_ms = 150.0
        mock_task_result.total_wall_ms = 200.0
        mock_task_result.tokens_per_second = 133.33
        mock_task_result.tokens_per_second_total = 150.0
        mock_task_result.score_primary = 0.85

        mock_task = {"id": "multi-task"}

        spec_multi = RunSpec(
            name="test-multi-run",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.openai_endpoint,
                path="http://localhost:8000/v1",
                model_id="test-model",
            ),
            runtime=RuntimeSpec(
                kind=RuntimeKind.openai_compatible,
                launch=LaunchMode.existing,
                model_name="test-model",
            ),
            workload=WorkloadSpec(
                prompt_suite="smoke",
                max_tokens=64,
                temperature=0.0,
                num_runs=3,
            ),
        )

        with patch("bench_harness.runners.openai_compatible.OpenAICompatClient") as mock_client_cls, \
             patch("bench_harness.runners.openai_compatible.CompletionRunner") as mock_runner_cls, \
             patch("bench_harness.runners.openai_compatible.load_tasks") as mock_load_tasks, \
             patch("bench_harness.runners.openai_compatible.asyncio.run") as mock_asyncio_run:

            mock_client_cls.return_value = MagicMock()
            mock_runner_instance = MagicMock()
            mock_runner_instance.run = MagicMock(return_value=mock_task_result)
            mock_runner_cls.return_value = mock_runner_instance
            mock_load_tasks.return_value = [mock_task]
            mock_asyncio_run.return_value = mock_task_result

            runner = OpenAICompatibleRunner(config=storage_config)
            prep = {"endpoint": "http://localhost:8000/v1", "model_name": "test-model"}
            result = runner.run_workload(spec_multi, prep, tmp_path)

            assert len(result.per_request) == 3
            assert result.per_request[0].request_id == "multi-task-0"
            assert result.per_request[1].request_id == "multi-task-1"
            assert result.per_request[2].request_id == "multi-task-2"
