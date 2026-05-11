"""Tests for M22 — ContextStressRunner (zero-coverage module)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bench_harness.runners.context_stress_runner import (
    ContextStressResult,
    ContextStressRunner,
    create_run_spec_for_stress,
    stress_results_to_run_result,
)


def _mock_success_response():
    """Create a mock httpx Response for a successful API call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
        },
    }
    return resp


def _mock_context_error_response():
    """Create a mock httpx Response for a context length error."""
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "Error: context length exceeded"
    return resp


class TestContextStressResult:
    def test_default_finish_reason(self):
        """finish_reason defaults to 'stop'."""
        r = ContextStressResult(
            task_id="t1", context_tokens=1000, prompt_tokens=500,
            completion_tokens=100, total_tokens=600,
            ttft_ms=10.0, decode_ms=50.0, wall_ms=60.0,
            decode_tps=2.0, quality_score=1.0, error=None,
        )
        assert r.finish_reason == "stop"

    def test_error_result(self):
        """Result can have error and non-stop finish_reason."""
        r = ContextStressResult(
            task_id="t1", context_tokens=1000, prompt_tokens=0,
            completion_tokens=0, total_tokens=0,
            ttft_ms=0, decode_ms=0, wall_ms=0,
            decode_tps=0, quality_score=None, error="timeout",
            finish_reason="error",
        )
        assert r.error == "timeout"
        assert r.finish_reason == "error"


class TestContextStressRunner:
    def test_run_success(self):
        """Successful API call produces valid result."""
        runner = ContextStressRunner(
            base_url="http://localhost:4000/v1",
            model_name="test-model",
        )
        spec = {"name": "stress-test", "project": "test"}
        stress_levels = [
            {
                "id": "stress.ctx_10k",
                "context_tokens": 10000,
                "prompt": "x" * 100,
                "scoring": {},
                "expected": {"type": "exact"},
                "estimated_prompt_tokens": 10000,
            }
        ]

        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_success_response()
            results = runner.run(spec, stress_levels, num_runs=2)

        assert len(results) == 2
        for r in results:
            assert r.task_id == "stress.ctx_10k"
            assert r.context_tokens == 10000
            assert r.error is None
            assert r.quality_score == 1.0

    def test_run_context_length_error(self):
        """400 context error returns ContextStressResult with error."""
        runner = ContextStressRunner(
            base_url="http://localhost:4000/v1",
            model_name="test-model",
        )
        spec = {"name": "stress-test", "project": "test"}
        stress_levels = [
            {
                "id": "stress.ctx_500k",
                "context_tokens": 500000,
                "prompt": "x" * 100,
                "scoring": {},
                "expected": {},
                "estimated_prompt_tokens": 500000,
            }
        ]

        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_context_error_response()
            results = runner.run(spec, stress_levels, num_runs=1)

        assert len(results) == 1
        r = results[0]
        assert r.error is not None
        assert "400" in r.error
        assert r.finish_reason == "context_error"
        assert r.quality_score is None

    def test_run_connect_timeout(self):
        """ConnectTimeout produces error result."""
        runner = ContextStressRunner(
            base_url="http://localhost:4000/v1",
            model_name="test-model",
        )
        spec = {"name": "stress-test", "project": "test"}
        stress_levels = [
            {
                "id": "stress.ctx_10k",
                "context_tokens": 10000,
                "prompt": "x" * 100,
                "scoring": {},
                "expected": {},
                "estimated_prompt_tokens": 10000,
            }
        ]

        import httpx

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectTimeout("Connection refused")
            results = runner.run(spec, stress_levels, num_runs=1)

        assert len(results) == 1
        assert results[0].error == "ConnectTimeout"

    def test_run_empty_content_score_zero(self):
        """Empty content produces quality_score of 0."""
        runner = ContextStressRunner(
            base_url="http://localhost:4000/v1",
            model_name="test-model",
        )
        spec = {"name": "stress-test", "project": "test"}
        stress_levels = [
            {
                "id": "stress.ctx_10k",
                "context_tokens": 10000,
                "prompt": "x" * 100,
                "scoring": {},
                "expected": {},
                "estimated_prompt_tokens": 10000,
            }
        ]

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100},
        }

        with patch("httpx.post") as mock_post:
            mock_post.return_value = resp
            results = runner.run(spec, stress_levels, num_runs=1)

        assert results[0].quality_score == 0.0


class TestStressResultsToRunResult:
    def test_basic_conversion(self):
        """Results convert to RunResult dict correctly."""
        results = [
            ContextStressResult(
                task_id="stress.ctx_10k", context_tokens=10000,
                prompt_tokens=500, completion_tokens=100, total_tokens=600,
                ttft_ms=10.0, decode_ms=50.0, wall_ms=60.0,
                decode_tps=2.0, quality_score=1.0, error=None,
            ),
        ]
        spec = {"name": "stress-test", "project": "test_proj"}
        run_result = stress_results_to_run_result(results, spec)

        assert run_result["run_id"] == "stress-stress-test"
        assert run_result["project"] == "test_proj"
        assert len(run_result["per_request"]) == 1
        req = run_result["per_request"][0]
        assert req["prompt_tokens"] == 500
        assert req["generated_tokens"] == 100
        assert req["quality_score"] == 1.0

    def test_error_result_conversion(self):
        """Error results convert with error field."""
        results = [
            ContextStressResult(
                task_id="stress.ctx_10k", context_tokens=10000,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                ttft_ms=0, decode_ms=0, wall_ms=60.0,
                decode_tps=0, quality_score=None, error="context length exceeded",
                finish_reason="context_error",
            ),
        ]
        spec = {"name": "stress-test", "project": "test"}
        run_result = stress_results_to_run_result(results, spec)

        req = run_result["per_request"][0]
        assert req["error"] == "context length exceeded"
        assert req["finish_reason"] == "context_error"


class TestCreateRunSpecForStress:
    def test_default_spec(self):
        """Default spec creates correct structure."""
        spec = create_run_spec_for_stress()
        assert spec["name"] == "context-stress-qwen36-fast"
        assert spec["project"] == "benchmark-harness"
        assert spec["runtime"]["model_name"] == "qwen36-fast"
        assert spec["workload"]["num_runs"] == 3
        assert "10K" in spec["tags"]

    def test_custom_spec(self):
        """Custom parameters are reflected in spec."""
        spec = create_run_spec_for_stress(
            base_url="http://custom:8000/v1",
            model_name="my-model",
            num_runs=5,
            stress_levels=["25K", "50K"],
        )
        assert spec["name"] == "context-stress-my-model"
        assert spec["artifact"]["path"] == "http://custom:8000/v1"
        assert spec["workload"]["num_runs"] == 5
        assert "25K" in spec["tags"]
        assert "250K" not in spec["tags"]  # Not in custom levels
