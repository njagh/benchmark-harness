"""Integration tests for mock benchmark execution.

Covers:
  - Run a mock benchmark with mocked LiteLLM / OpenAI responses
  - Verify results are stored correctly in SQLite
  - Verify timing metrics are captured
  - Test different mock response types (success, error, timeout)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.runners.completion_runner import CompletionRunner, RunResult
from bench_harness.storage.sqlite import SQLiteStore


# ── Fixtures ──────────────────────────────────────────────────────────

def _sample_task():
    """Return a minimal task dict for mock testing."""
    return {
        "id": "mock.test_001",
        "prompt": "What is 2+2?",
        "scoring": {"primary": "exact_match"},
        "expected": {"type": "exact", "answer": "4"},
    }


def _mock_success_response():
    """Return a mock API response dict for successful completion."""
    return {
        "content": "4",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "total_tokens": 9,
        },
        "finish_reason": "stop",
        "model": "mock-model",
    }


def _mock_error_response():
    """Return a mock API response dict for an error."""
    return {
        "content": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "finish_reason": "error",
        "error": "Connection refused",
    }


def _make_mock_client(return_value=None, side_effect=None):
    """Create a mocked OpenAICompatClient with an AsyncMock for chat_complete."""
    client = AsyncMock(spec=OpenAICompatClient)
    client.chat_complete = AsyncMock(
        return_value=return_value,
        side_effect=side_effect,
    )
    return client


# ── Successful mock benchmark run ─────────────────────────────────────

class TestMockSuccessRun:
    def test_run_with_mocked_success(self):
        """CompletionRunner.run with mocked API returns successful RunResult."""
        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert isinstance(result, RunResult)
        assert result.exit_status == "success"
        assert result.error_message is None
        assert result.raw_response == "4"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 1
        assert result.total_tokens == 9
        assert result.ttft_ms > 0
        assert result.decode_ms > 0
        assert result.total_wall_ms > 0
        assert result.tokens_per_second > 0
        assert result.tokens_per_second_total > 0

    def test_run_success_is_stored_in_sqlite(self, tmp_path: Path):
        """Results from a mock run are saved to SQLite correctly."""
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)
        store.init()

        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
            "prompt_style": "plain",
            "context_tokens": "small",
            "quantization": "FP8",
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))
        store.save_run(result)

        runs = store.get_runs(suite_id="mock_suite")
        assert len(runs) == 1
        run = runs[0]
        assert run["task_id"] == "mock.test_001"
        assert run["model_alias"] == "test-model"
        assert run["exit_status"] == "success"
        assert run["prompt_tokens"] == 8
        assert run["completion_tokens"] == 1

    def test_timing_metrics_are_reasonable(self):
        """Timing values are positive and internally consistent."""
        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.ttft_ms >= 0
        assert result.decode_ms >= 0
        assert result.prefill_ms >= 0
        assert result.total_wall_ms >= 0
        assert result.total_wall_ms >= result.ttft_ms

        # TTFT is estimated as 15% of wall
        ttft_ratio = result.ttft_ms / max(result.total_wall_ms, 1e-9)
        decode_ratio = result.decode_ms / max(result.total_wall_ms, 1e-9)
        assert abs(ttft_ratio - 0.15) < 0.001
        assert abs(decode_ratio - 0.85) < 0.001


# ── Error mock responses ──────────────────────────────────────────────

class TestMockErrorRun:
    def test_run_with_mocked_api_error(self):
        """CompletionRunner.run with error response returns error RunResult.

        The completion_runner treats response dicts with 'error' keys as
        exceptions, so the runner catches them and returns exit_status='error'.
        """
        client = _make_mock_client(side_effect=ValueError("Connection refused"))
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert isinstance(result, RunResult)
        assert result.exit_status == "error"
        assert result.error_message is not None
        assert "Connection refused" in result.error_message
        assert result.raw_response == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_run_with_timeout_exception(self):
        """CompletionRunner.run with an exception returns error RunResult."""
        client = _make_mock_client(
            side_effect=asyncio.TimeoutError("Request timed out")
        )
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert isinstance(result, RunResult)
        assert result.exit_status == "error"
        assert "timed out" in result.error_message.lower()
        assert result.total_wall_ms > 0

    def test_error_result_stored_in_sqlite(self, tmp_path: Path):
        """Error results are still saved to SQLite with error flags."""
        db_path = str(tmp_path / "error_test.db")
        store = SQLiteStore(db_path)
        store.init()

        client = _make_mock_client(
            side_effect=asyncio.TimeoutError("timeout")
        )
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))
        store.save_run(result)

        runs = store.get_runs(suite_id="mock_suite")
        assert len(runs) == 1
        assert runs[0]["exit_status"] == "error"
        assert runs[0]["task_id"] == "mock.test_001"

    def test_run_with_generic_exception(self):
        """CompletionRunner.run with a generic exception returns error RunResult."""
        client = _make_mock_client(
            side_effect=ValueError("Bad response format")
        )
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert isinstance(result, RunResult)
        assert result.exit_status == "error"
        assert "Bad response format" in result.error_message


# ── Timing metrics capture ────────────────────────────────────────────

class TestTimingMetrics:
    def test_ttft_and_decode_estimates(self):
        """TTFT and decode are estimated as fractions of wall time."""
        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.total_wall_ms > 0
        ttft_ratio = result.ttft_ms / max(result.total_wall_ms, 1e-9)
        decode_ratio = result.decode_ms / max(result.total_wall_ms, 1e-9)
        assert abs(ttft_ratio - 0.15) < 0.001
        assert abs(decode_ratio - 0.85) < 0.001

    def test_tokens_per_second_computed(self):
        """Tokens per second is derived from completion_tokens / decode_ms."""
        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.tokens_per_second > 0
        assert result.tokens_per_second_total > 0

    def test_timing_with_zero_tokens(self):
        """Zero tokens still produces valid timing metrics."""
        client = _make_mock_client(return_value={
            "content": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "finish_reason": "stop",
            "model": "mock",
        })
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.exit_status == "success"
        assert result.raw_response == ""
        assert result.completion_tokens == 0
        assert result.total_tokens == 0


# ── Multiple runs ─────────────────────────────────────────────────────

class TestMultipleRuns:
    def test_multiple_mock_runs_stored(self, tmp_path: Path):
        """Multiple mock benchmark runs are all stored correctly."""
        db_path = str(tmp_path / "multi.db")
        store = SQLiteStore(db_path)
        store.init()

        client = _make_mock_client(return_value=_mock_success_response())
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        for _ in range(3):
            result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))
            store.save_run(result)
            assert result.exit_status == "success"

        runs = store.get_runs(suite_id="mock_suite")
        assert len(runs) == 3
        for run in runs:
            assert run["task_id"] == "mock.test_001"
            assert run["model_alias"] == "test-model"

    def test_mixed_success_and_error_runs(self, tmp_path: Path):
        """A mix of successful and errored runs are all stored."""
        db_path = str(tmp_path / "mixed.db")
        store = SQLiteStore(db_path)
        store.init()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return _mock_success_response()
            raise asyncio.TimeoutError("timeout")

        client = _make_mock_client(side_effect=side_effect)
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        for _ in range(4):
            result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))
            store.save_run(result)

        runs = store.get_runs(suite_id="mock_suite")
        assert len(runs) == 4
        successes = sum(1 for r in runs if r["exit_status"] == "success")
        errors = sum(1 for r in runs if r["exit_status"] == "error")
        assert successes == 2
        assert errors == 2


# ── Different mock response types ─────────────────────────────────────

class TestMockResponseTypes:
    def test_mock_with_custom_usage_tokens(self):
        """Mock response with custom token counts is reflected in RunResult."""
        custom_usage = {
            "content": "custom answer",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "finish_reason": "stop",
            "model": "custom-model",
        }

        client = _make_mock_client(return_value=custom_usage)
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.raw_response == "custom answer"

    def test_mock_with_null_content(self):
        """Mock response with null/missing content is handled gracefully."""
        client = _make_mock_client(return_value={
            "content": None,
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            "finish_reason": "stop",
            "model": "test-model",
        })
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.exit_status == "success"
        assert result.raw_response == ""
        assert result.completion_tokens == 0

    def test_mock_with_early_stop(self):
        """Mock response with finish_reason='length' is handled."""
        client = _make_mock_client(return_value={
            "content": "partial answer that was truncated",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 64,
                "total_tokens": 74,
            },
            "finish_reason": "length",
            "model": "test-model",
        })
        runner_instance = CompletionRunner(client)
        task = _sample_task()
        params = {
            "model_alias": "test-model",
            "model_backend": "vllm",
            "temperature": 0.0,
            "max_tokens": 64,
        }

        result = asyncio.run(runner_instance.run(task, params, suite_id="mock_suite"))

        assert result.exit_status == "success"
        assert "partial answer" in result.raw_response
        assert result.completion_tokens == 64
