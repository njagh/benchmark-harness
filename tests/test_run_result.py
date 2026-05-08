"""Tests for run_result.py — RequestResult, ResultSummary, RunResult
schema, finalization, and edge cases."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bench_harness.schemas.run_result import (
    RunResult,
    RequestResult,
    ResultSummary,
)


# ── RequestResult ────────────────────────────────────────────────────


class TestRequestResult:
    def test_all_fields_set(self):
        """RequestResult with all fields populated."""
        result = RequestResult(
            request_id="req-1",
            prompt_id="prompt-1",
            prompt_tokens=10,
            generated_tokens=20,
            ttft_ms=50.0,
            decode_ms=200.0,
            total_wall_ms=250.0,
            tokens_per_second_decode=100.0,
            tokens_per_second_wall=80.0,
            finish_reason="stop",
            error=None,
            peak_gpu_memory_mb=16384.0,
            quality_score=0.9,
            quality_explanation="Good response",
        )
        assert result.request_id == "req-1"
        assert result.peak_gpu_memory_mb == 16384.0
        assert result.quality_score == 0.9
        assert result.quality_explanation == "Good response"

    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        result = RequestResult(
            request_id="req-1",
            prompt_id="prompt-1",
            prompt_tokens=10,
            generated_tokens=20,
            ttft_ms=50.0,
            decode_ms=200.0,
            total_wall_ms=250.0,
            tokens_per_second_decode=100.0,
            tokens_per_second_wall=80.0,
            finish_reason="stop",
        )
        assert result.error is None
        assert result.peak_gpu_memory_mb is None
        assert result.quality_score is None
        assert result.quality_explanation is None

    def test_with_error(self):
        """RequestResult with error field set."""
        result = RequestResult(
            request_id="req-1",
            prompt_id="prompt-1",
            prompt_tokens=10,
            generated_tokens=20,
            ttft_ms=50.0,
            decode_ms=200.0,
            total_wall_ms=250.0,
            tokens_per_second_decode=100.0,
            tokens_per_second_wall=80.0,
            finish_reason="error",
            error="Connection timeout",
        )
        assert result.error == "Connection timeout"

    def test_serialization(self):
        """RequestResult can be serialized to JSON."""
        result = RequestResult(
            request_id="req-1",
            prompt_id="prompt-1",
            prompt_tokens=10,
            generated_tokens=20,
            ttft_ms=50.0,
            decode_ms=200.0,
            total_wall_ms=250.0,
            tokens_per_second_decode=100.0,
            tokens_per_second_wall=80.0,
            finish_reason="stop",
        )
        data = json.loads(result.model_dump_json())
        assert data["request_id"] == "req-1"
        assert data["ttft_ms"] == 50.0


# ── ResultSummary ────────────────────────────────────────────────────


class TestResultSummaryFromRequests:
    def test_empty_requests(self):
        """ResultSummary.from_requests with empty list returns zeros."""
        summary = ResultSummary.from_requests([])
        assert summary.mean_ttft_ms == 0
        assert summary.median_ttft_ms == 0
        assert summary.p95_ttft_ms == 0
        assert summary.mean_decode_tps == 0
        assert summary.success_rate == 0.0
        assert summary.oom_count == 0
        assert summary.timeout_count == 0
        assert summary.quality_stddev == 0.0

    def test_single_successful_request(self):
        """Single successful request produces correct summary."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            )
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.success_rate == 1.0
        assert summary.mean_ttft_ms == 50.0
        assert summary.median_ttft_ms == 50.0
        assert summary.oom_count == 0
        assert summary.timeout_count == 0

    def test_multiple_requests(self):
        """Multiple requests produce aggregated statistics."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=100.0,
                decode_ms=300.0,
                total_wall_ms=400.0,
                tokens_per_second_decode=66.67,
                tokens_per_second_wall=50.0,
                finish_reason="stop",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.success_rate == 1.0
        assert summary.mean_ttft_ms == 75.0  # (50 + 100) / 2
        assert summary.oom_count == 0
        assert summary.timeout_count == 0

    def test_with_error_requests(self):
        """Errors are counted in oom_count and timeout_count."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="OOM: out of memory",
                peak_gpu_memory_mb=16384.0,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="timeout exceeded",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.success_rate == 0.0
        assert summary.oom_count == 1
        assert summary.timeout_count == 1

    def test_oom_detection_case_insensitive(self):
        """OOM detection is case-insensitive."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="OOM error",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.oom_count == 1

    def test_oom_detection_lowercase_oom(self):
        """OOM detection matches lowercase 'oom'."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="oom error",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.oom_count == 1

    def test_timeout_detection_case_insensitive(self):
        """Timeout detection is case-insensitive."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="TIMEOUT",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.timeout_count == 1

    def test_peak_vram(self):
        """peak_vram_mb is the maximum of all peak_gpu_memory_mb values."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                peak_gpu_memory_mb=8192.0,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                peak_gpu_memory_mb=16384.0,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.peak_vram_mb == 16384.0

    def test_average_vram(self):
        """average_vram_mb is the mean of all peak_gpu_memory_mb values."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                peak_gpu_memory_mb=8192.0,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                peak_gpu_memory_mb=16384.0,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.average_vram_mb == 12288.0

    def test_peak_vram_none_when_no_values(self):
        """peak_vram_mb is None when no peak_gpu_memory_mb values exist."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.peak_vram_mb is None

    def test_qualitative_score(self):
        """qualitative_score is the mean of quality_score values."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=0.8,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=1.0,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.qualitative_score == 0.9

    def test_qualitative_score_none_when_no_scores(self):
        """qualitative_score is None when no quality_score values exist."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.qualitative_score is None

    def test_quality_stddev(self):
        """quality_stddev is computed from quality scores."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=0.5,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=1.0,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.quality_stddev > 0

    def test_quality_stddev_zero_single_score(self):
        """quality_stddev is 0.0 when only one quality score exists."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=0.9,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.quality_stddev == 0.0

    def test_quality_stddev_zero_no_scores(self):
        """quality_stddev is 0.0 when no quality scores exist."""
        summary = ResultSummary.from_requests([])
        assert summary.quality_stddev == 0.0

    def test_mixed_success_and_failure(self):
        """success_rate is computed correctly with mixed results."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                error="Some error",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.success_rate == 0.5


# ── RunResult ────────────────────────────────────────────────────────


class TestRunResult:
    def test_schema_version_default(self):
        """RunResult schema_version defaults to v1."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        assert run.schema_version == "llm_bench.run_result.v1"

    def test_empty_per_request(self):
        """per_request defaults to empty list."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        assert run.per_request == []

    def test_empty_artifact_fingerprint(self):
        """artifact_fingerprint defaults to empty dict."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        assert run.artifact_fingerprint == {}

    def test_summary_none_by_default(self):
        """summary is None before finalize()."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        assert run.summary is None

    def test_finalize_empty_requests(self):
        """finalize() with empty per_request computes zero summary."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        run.finalize()
        assert run.summary is not None
        assert run.summary.mean_ttft_ms == 0
        assert run.summary.success_rate == 0.0

    def test_finalize_with_requests(self):
        """finalize() computes summary from per_request data."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
            per_request=[
                RequestResult(
                    request_id="req-1",
                    prompt_id="prompt-1",
                    prompt_tokens=10,
                    generated_tokens=20,
                    ttft_ms=50.0,
                    decode_ms=200.0,
                    total_wall_ms=250.0,
                    tokens_per_second_decode=100.0,
                    tokens_per_second_wall=80.0,
                    finish_reason="stop",
                )
            ],
        )
        run.finalize()
        assert run.summary is not None
        assert run.summary.success_rate == 1.0
        assert run.summary.mean_ttft_ms == 50.0

    def test_finalize_returns_self(self):
        """finalize() returns the RunResult instance."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
        )
        result = run.finalize()
        assert result is run

    def test_write_to_directory_files_created(self, tmp_path):
        """write_to_directory creates all expected files."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
            per_request=[
                RequestResult(
                    request_id="req-1",
                    prompt_id="prompt-1",
                    prompt_tokens=10,
                    generated_tokens=20,
                    ttft_ms=50.0,
                    decode_ms=200.0,
                    total_wall_ms=250.0,
                    tokens_per_second_decode=100.0,
                    tokens_per_second_wall=80.0,
                    finish_reason="stop",
                )
            ],
        )
        run.finalize()
        run.write_to_directory(tmp_path)
        assert (tmp_path / "metrics.jsonl").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "run_result.json").exists()

    def test_write_to_directory_no_summary(self, tmp_path):
        """write_to_directory without summary only creates metrics and run_result."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
            per_request=[
                RequestResult(
                    request_id="req-1",
                    prompt_id="prompt-1",
                    prompt_tokens=10,
                    generated_tokens=20,
                    ttft_ms=50.0,
                    decode_ms=200.0,
                    total_wall_ms=250.0,
                    tokens_per_second_decode=100.0,
                    tokens_per_second_wall=80.0,
                    finish_reason="stop",
                )
            ],
        )
        # Don't call finalize() — summary is None
        run.write_to_directory(tmp_path)
        assert (tmp_path / "metrics.jsonl").exists()
        assert not (tmp_path / "summary.json").exists()
        assert (tmp_path / "run_result.json").exists()

    def test_write_metrics_jsonl_content(self, tmp_path):
        """metrics.jsonl contains one JSON line per request."""
        run = RunResult(
            run_id="run-1",
            run_spec_ref="spec.yaml",
            project="test_project",
            per_request=[
                RequestResult(
                    request_id="req-1",
                    prompt_id="prompt-1",
                    prompt_tokens=10,
                    generated_tokens=20,
                    ttft_ms=50.0,
                    decode_ms=200.0,
                    total_wall_ms=250.0,
                    tokens_per_second_decode=100.0,
                    tokens_per_second_wall=80.0,
                    finish_reason="stop",
                ),
                RequestResult(
                    request_id="req-2",
                    prompt_id="prompt-2",
                    prompt_tokens=10,
                    generated_tokens=30,
                    ttft_ms=60.0,
                    decode_ms=210.0,
                    total_wall_ms=260.0,
                    tokens_per_second_decode=110.0,
                    tokens_per_second_wall=85.0,
                    finish_reason="stop",
                ),
            ],
        )
        run.finalize()
        run.write_to_directory(tmp_path)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["request_id"] == "req-1"

    def test_partial_data_summary(self):
        """ResultSummary handles partial data correctly."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
                quality_score=0.9,
                peak_gpu_memory_mb=16384.0,
            ),
            RequestResult(
                request_id="req-2",
                prompt_id="prompt-2",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=0,  # zero tps
                tokens_per_second_wall=0,  # zero tps
                finish_reason="stop",
                # no quality_score
                # no peak_gpu_memory_mb
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        # TPS values from zero tps requests should be filtered out
        assert summary.mean_decode_tps == 100.0  # only req-1 has decode tps
        assert summary.mean_wall_tps == 80.0

    def test_none_values_in_summary(self):
        """ResultSummary handles None values gracefully."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.peak_vram_mb is None
        assert summary.average_vram_mb is None
        assert summary.qualitative_score is None


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_tokens_per_second_filtered(self):
        """Zero TPS values are excluded from statistics."""
        requests = [
            RequestResult(
                request_id="req-1",
                prompt_id="prompt-1",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=50.0,
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=0.0,
                tokens_per_second_wall=0.0,
                finish_reason="stop",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.mean_decode_tps == 0.0
        assert summary.mean_wall_tps == 0.0

    def test_many_requests_p95(self):
        """p95 is computed correctly with many requests."""
        requests = [
            RequestResult(
                request_id=f"req-{i}",
                prompt_id=f"prompt-{i}",
                prompt_tokens=10,
                generated_tokens=20,
                ttft_ms=float(i * 10),
                decode_ms=200.0,
                total_wall_ms=250.0,
                tokens_per_second_decode=100.0,
                tokens_per_second_wall=80.0,
                finish_reason="stop",
            )
            for i in range(20)
        ]
        summary = ResultSummary.from_requests(requests)
        # p95 should be near the higher end
        assert summary.p95_ttft_ms >= 160.0  # 95th percentile of 0..190

    def test_model_dump_json_roundtrip(self):
        """RequestResult JSON roundtrip preserves values."""
        result = RequestResult(
            request_id="req-1",
            prompt_id="prompt-1",
            prompt_tokens=10,
            generated_tokens=20,
            ttft_ms=50.0,
            decode_ms=200.0,
            total_wall_ms=250.0,
            tokens_per_second_decode=100.0,
            tokens_per_second_wall=80.0,
            finish_reason="stop",
            error="OOM",
            peak_gpu_memory_mb=16384.0,
            quality_score=0.9,
            quality_explanation="good",
        )
        data = json.loads(result.model_dump_json())
        assert data["request_id"] == "req-1"
        assert data["error"] == "OOM"
        assert data["peak_gpu_memory_mb"] == 16384.0
