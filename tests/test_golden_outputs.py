"""Golden output tests for benchmark harness run results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench_harness.schemas.run_result import RunResult, RequestResult, ResultSummary

GOLDEN_DIR = Path(__file__).parent / "golden"


# ── Golden Run Result Schema Tests ───────────────────────────────────


class TestGoldenRunResultSchema:
    def test_golden_file_exists(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        assert golden_file.exists(), f"Golden file not found at {golden_file}"

    def test_golden_file_valid_json(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        content = golden_file.read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_golden_schema_version(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["schema_version"] == "llm_bench.run_result.v1"

    def test_golden_required_top_level_fields(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        required_fields = ["schema_version", "run_id", "run_spec_ref", "project"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_golden_run_id_present(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) > 0

    def test_golden_project_present(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["project"], str)
        assert len(data["project"]) > 0

    def test_golden_artifact_fingerprint(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["artifact_fingerprint"], dict)
        assert "model_id" in data["artifact_fingerprint"]

    def test_golden_per_request_is_list(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["per_request"], list)
        assert len(data["per_request"]) > 0

    def test_golden_per_request_count(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert len(data["per_request"]) == 3

    def test_golden_per_request_item_schema(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        required_request_fields = [
            "request_id", "prompt_id", "prompt_tokens",
            "generated_tokens", "ttft_ms", "decode_ms",
            "total_wall_ms", "tokens_per_second_decode",
            "tokens_per_second_wall", "finish_reason",
        ]
        for item in data["per_request"]:
            for field in required_request_fields:
                assert field in item, f"Missing field {field} in request"

    def test_golden_request_ids_unique(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        ids = [r["request_id"] for r in data["per_request"]]
        assert len(ids) == len(set(ids)), "Duplicate request IDs found"

    def test_golden_request_tokens_positive(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert req["prompt_tokens"] > 0
            assert req["generated_tokens"] > 0

    def test_golden_request_times_positive(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert req["ttft_ms"] > 0
            assert req["decode_ms"] > 0
            assert req["total_wall_ms"] > 0

    def test_golden_request_tokensecs_positive(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert req["tokens_per_second_decode"] > 0
            assert req["tokens_per_second_wall"] > 0

    def test_golden_request_finish_reason(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert isinstance(req["finish_reason"], str)
            assert len(req["finish_reason"]) > 0

    def test_golden_request_errors_null(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert req["error"] is None

    def test_golden_request_quality_scores(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert req["quality_score"] is not None
            assert 0 <= req["quality_score"] <= 1.0

    def test_golden_request_quality_explanations(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        for req in data["per_request"]:
            assert isinstance(req["quality_explanation"], str)
            assert len(req["quality_explanation"]) > 0

    def test_golden_summary_exists(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert "summary" in data
        assert data["summary"] is not None

    def test_golden_summary_required_fields(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        required_summary_fields = [
            "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms",
            "mean_decode_tps", "median_decode_tps", "p95_decode_tps",
            "mean_wall_tps", "median_wall_tps", "p95_wall_tps",
            "success_rate", "oom_count", "timeout_count",
            "peak_vram_mb", "average_vram_mb",
            "qualitative_score", "quality_stddev",
        ]
        for field in required_summary_fields:
            assert field in data["summary"], f"Missing summary field: {field}"

    def test_golden_summary_success_rate(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["success_rate"] == 1.0

    def test_golden_summary_oom_count(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["oom_count"] == 0

    def test_golden_summary_timeout_count(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["timeout_count"] == 0

    def test_golden_summary_qualitative_score(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["qualitative_score"] is not None
        assert 0 <= data["summary"]["qualitative_score"] <= 1.0

    def test_golden_summary_quality_stddev_positive(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["quality_stddev"] > 0

    def test_golden_summary_vram_values(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert data["summary"]["peak_vram_mb"] > 0
        assert data["summary"]["average_vram_mb"] is not None
        assert data["summary"]["average_vram_mb"] > 0

    def test_golden_from_pydantic_roundtrip(self):
        requests = [
            RequestResult(
                request_id="test-rt-001",
                prompt_id="prompt-1",
                prompt_tokens=100,
                generated_tokens=200,
                ttft_ms=40.0,
                decode_ms=250.0,
                total_wall_ms=290.0,
                tokens_per_second_decode=800.0,
                tokens_per_second_wall=690.0,
                finish_reason="stop",
                quality_score=0.88,
            ),
        ]
        run = RunResult(
            run_id="test-rt",
            run_spec_ref="test.yaml",
            project="test_project",
            per_request=requests,
        )
        run.finalize()
        data = json.loads(run.model_dump_json())
        assert data["run_id"] == "test-rt"
        assert data["schema_version"] == "llm_bench.run_result.v1"
        assert len(data["per_request"]) == 1
        assert data["summary"] is not None
        assert data["summary"]["success_rate"] == 1.0

    def test_golden_from_pydantic_empty_requests(self):
        run = RunResult(
            run_id="test-rt-empty",
            run_spec_ref="test.yaml",
            project="test_project",
            per_request=[],
        )
        run.finalize()
        data = json.loads(run.model_dump_json())
        assert data["summary"]["success_rate"] == 0.0

    def test_golden_from_pydantic_with_error(self):
        requests = [
            RequestResult(
                request_id="test-err-001",
                prompt_id="prompt-1",
                prompt_tokens=100,
                generated_tokens=0,
                ttft_ms=50.0,
                decode_ms=0.0,
                total_wall_ms=50.0,
                tokens_per_second_decode=0.0,
                tokens_per_second_wall=0.0,
                finish_reason="error",
                error="OOM: out of memory",
                peak_gpu_memory_mb=16384.0,
            ),
        ]
        run = RunResult(
            run_id="test-rt-error",
            run_spec_ref="test.yaml",
            project="test_project",
            per_request=requests,
        )
        run.finalize()
        data = json.loads(run.model_dump_json())
        assert data["summary"]["oom_count"] == 1
        assert data["summary"]["success_rate"] == 0.0

    def test_golden_summary_stats_consistency(self):
        """Verify summary statistics are consistent with per_request data."""
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        requests = data["per_request"]
        summary = data["summary"]
        assert summary["success_rate"] == 1.0
        assert summary["mean_ttft_ms"] > 0
        assert summary["median_ttft_ms"] > 0

    def test_golden_summary_p95_gte_median(self):
        """P95 should be >= median for all metrics."""
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        summary = data["summary"]
        assert summary["p95_ttft_ms"] >= summary["median_ttft_ms"]
        assert summary["p95_decode_tps"] >= summary["median_decode_tps"]
        assert summary["p95_wall_tps"] >= summary["median_wall_tps"]

    def test_golden_summary_mean_gte_0(self):
        """All mean metrics should be >= 0."""
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        data = json.loads(golden_file.read_text())
        summary = data["summary"]
        assert summary["mean_ttft_ms"] >= 0
        assert summary["mean_decode_tps"] >= 0
        assert summary["mean_wall_tps"] >= 0


# ── Golden Optimization Result Tests ─────────────────────────────────


class TestGoldenOptimizationResult:
    def test_optimization_golden_file_exists(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        assert golden_file.exists()

    def test_optimization_golden_valid_json(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        content = golden_file.read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_optimization_golden_required_fields(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        for field in ["name", "total_runs_analyzed", "styles_found", "best_style_overall"]:
            assert field in data, f"Missing field: {field}"

    def test_optimization_golden_styles_found_list(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["styles_found"], list)
        assert len(data["styles_found"]) > 0

    def test_optimization_golden_best_style_present(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["best_style_overall"], str)
        assert len(data["best_style_overall"]) > 0

    def test_optimization_golden_family_rankings(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["family_rankings"], dict)
        for family, rankings in data["family_rankings"].items():
            assert isinstance(rankings, list)
            for rank_entry in rankings:
                assert len(rank_entry) == 3

    def test_optimization_golden_style_variances(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["style_variances"], dict)
        for style, var in data["style_variances"].items():
            assert var >= 0

    def test_optimization_golden_candidate_results(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        assert isinstance(data["candidate_results"], list)
        assert len(data["candidate_results"]) > 0

    def test_optimization_golden_candidate_schema(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        required_candidate_fields = [
            "name", "score", "baseline", "score_delta",
            "run_count", "status",
        ]
        for candidate in data["candidate_results"]:
            for field in required_candidate_fields:
                assert field in candidate, f"Missing field in candidate: {field}"

    def test_optimization_golden_candidate_statuses(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        data = json.loads(golden_file.read_text())
        valid_statuses = {"complete", "analyzed", "no_data", "baseline"}
        for candidate in data["candidate_results"]:
            assert candidate["status"] in valid_statuses, f"Invalid status: {candidate['status']}"


# ── Golden Directory Tests ──────────────────────────────────────────


class TestGoldenDirectory:
    def test_golden_directory_exists(self):
        assert GOLDEN_DIR.exists()

    def test_golden_directory_has_files(self):
        json_files = list(GOLDEN_DIR.glob("*.json"))
        assert len(json_files) > 0, "No .json files in golden directory"

    def test_golden_files_are_json(self):
        for json_file in GOLDEN_DIR.glob("*.json"):
            content = json_file.read_text()
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                pytest.fail(f"{json_file} is not valid JSON: {e}")

    def test_golden_run_result_file_name(self):
        golden_file = GOLDEN_DIR / "run_result_golden.json"
        assert golden_file.exists()

    def test_golden_optimization_file_name(self):
        golden_file = GOLDEN_DIR / "optimization_result_golden.json"
        assert golden_file.exists()
