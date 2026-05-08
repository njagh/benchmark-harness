"""Tests for Milestone 18 — RunSpec and Result Schemas."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.schemas import (
    RunSpec,
    ArtifactSpec,
    RuntimeSpec,
    WorkloadSpec,
    HardwareSpec,
    StoragePolicy,
    ArtifactKind,
    ArtifactMode,
    RuntimeKind,
    LaunchMode,
    build_run_spec_from_flags,
)
from bench_harness.schemas.run_result import RunResult, RequestResult, ResultSummary
from bench_harness.schemas.model_artifact import ModelArtifact
from bench_harness.schemas.compat import (
    resolve_schema_version,
    migrate_result_v0_to_v1,
    SchemaVersionError,
    KNOWN_VERSIONS,
)


# ── RunSpec ────────────────────────────────────────────────────────


class TestRunSpecValidation:
    def test_valid_run_spec(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test-model",
                model_id="test-model",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.schema_version == "llm_bench.run_spec.v1"
        assert spec.name == "test-model-smoke"
        assert spec.project == "test_project"
        assert spec.artifact.kind == ArtifactKind.hf_checkpoint

    def test_invalid_name_too_short(self):
        with pytest.raises(Exception):
            RunSpec(
                name="a",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
            )

    def test_invalid_name_uppercase(self):
        with pytest.raises(Exception):
            RunSpec(
                name="Test-Model",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
            )

    def test_invalid_name_hyphen_first(self):
        with pytest.raises(Exception):
            RunSpec(
                name="-test-model",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
            )

    def test_default_runtime(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.runtime.kind == RuntimeKind.openai_compatible
        assert spec.runtime.launch == LaunchMode.existing

    def test_default_workload_defaults(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.workload.max_tokens == 256
        assert spec.workload.temperature == 0.0
        assert spec.workload.num_runs == 1
        assert spec.workload.concurrency == 1

    def test_workload_task_dir(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(
                prompt_suite="smoke",
                task_dir="/tasks/smoke",
            ),
        )
        assert spec.workload.task_dir == "/tasks/smoke"
        assert spec.workload.prompt_suite == "smoke"

    def test_workload_task_dir_none_by_default(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.workload.task_dir is None

    def test_workload_task_dir_yaml_roundtrip(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
                model_id="test-model",
            ),
            workload=WorkloadSpec(
                prompt_suite="smoke",
                task_dir="/tasks/smoke",
                max_tokens=512,
                num_runs=2,
            ),
        )
        yaml_str = spec.to_yaml()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml_path = f.name
            f.write(yaml_str)
        reloaded = RunSpec.from_yaml(yaml_path)
        assert reloaded.workload.task_dir == "/tasks/smoke"
        assert reloaded.workload.max_tokens == 512
        assert reloaded.workload.num_runs == 2
        Path(yaml_path).unlink()

    def test_endpoint_requires_url(self):
        with pytest.raises(ValueError):
            ArtifactSpec(
                kind=ArtifactKind.openai_endpoint,
                path="local-path",
            )

    def test_endpoint_allows_url(self):
        spec = ArtifactSpec(
            kind=ArtifactKind.openai_endpoint,
            path="http://127.0.0.1:8009/v1",
        )
        assert spec.path == "http://127.0.0.1:8009/v1"


class TestRunSpecYAMLJSON:
    def test_to_yaml_from_yaml(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            tags=["test", "smoke"],
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
                model_id="test-model",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        yaml_str = spec.to_yaml()
        # Write and re-read
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml_path = f.name
            f.write(yaml_str)
        reloaded = RunSpec.from_yaml(yaml_path)
        assert reloaded.name == spec.name
        assert reloaded.project == spec.project
        assert reloaded.tags == spec.tags
        Path(yaml_path).unlink()

    def test_to_json_from_json(self):
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
                model_id="test-model",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json_path = f.name
            f.write(spec.to_json())
        reloaded = RunSpec.from_json(json_path)
        assert reloaded.name == spec.name
        assert reloaded.project == spec.project
        Path(json_path).unlink()

    def test_build_from_flags(self):
        spec = build_run_spec_from_flags(
            suite="smoke",
            models=["test-model"],
            num_runs=3,
            max_tokens=128,
            temperature=0.7,
            concurrency=2,
            context_tokens="medium",
            project="my_project",
        )
        assert spec.name == "test-model-smoke-cli"
        assert spec.project == "my_project"
        assert spec.workload.num_runs == 3
        assert spec.workload.max_tokens == 128
        assert spec.workload.temperature == 0.7
        assert spec.workload.concurrency == 2
        assert spec.hardware.profile == "medium"


# ── RunResult ──────────────────────────────────────────────────────


class TestRequestResult:
    def test_basic_request_result(self):
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
        assert result.request_id == "req-1"
        assert result.error is None


class TestResultSummary:
    def test_empty_requests(self):
        summary = ResultSummary.from_requests([])
        assert summary.mean_ttft_ms == 0
        assert summary.success_rate == 0.0
        assert summary.oom_count == 0
        assert summary.timeout_count == 0

    def test_single_request(self):
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
        assert summary.oom_count == 0

    def test_with_errors(self):
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
                ttft_ms=60.0,
                decode_ms=190.0,
                total_wall_ms=240.0,
                tokens_per_second_decode=105.0,
                tokens_per_second_wall=82.0,
                finish_reason="stop",
                error="timeout exceeded",
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.success_rate == 0.0
        assert summary.oom_count == 1
        assert summary.timeout_count == 1
        assert summary.mean_ttft_ms == 55.0

    def test_with_quality_scores(self):
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
                quality_score=0.7,
            ),
        ]
        summary = ResultSummary.from_requests(requests)
        assert summary.qualitative_score == 0.8
        assert summary.quality_stddev > 0

    def test_run_result_finalize(self):
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
        assert run.summary is None
        run.finalize()
        assert run.summary is not None
        assert run.summary.success_rate == 1.0

    def test_write_to_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
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
            run.write_to_directory(run_dir)
            assert (run_dir / "metrics.jsonl").exists()
            assert (run_dir / "summary.json").exists()
            assert (run_dir / "run_result.json").exists()


# ── ModelArtifact ──────────────────────────────────────────────────


class TestModelArtifact:
    def test_valid_artifact(self):
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            model_id="test-model",
        )
        assert artifact.schema_version == "llm_bench.model_artifact.v1"
        assert artifact.durable is True

    def test_to_json(self):
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        data = json.loads(artifact.to_json())
        assert data["artifact_id"] == "art-1"


# ── Schema Compat ──────────────────────────────────────────────────


class TestSchemaCompat:
    def test_known_versions(self):
        assert "llm_bench.run_spec.v1" in KNOWN_VERSIONS
        assert "llm_bench.run_result.v1" in KNOWN_VERSIONS
        assert "llm_bench.model_artifact.v1" in KNOWN_VERSIONS

    def test_resolve_known_version(self):
        data = {"schema_version": "llm_bench.run_spec.v1"}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v1"

    def test_resolve_missing_version(self):
        data = {"name": "test"}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v1"

    def test_resolve_unknown_version(self):
        data = {"schema_version": "unknown.schema.v1"}
        with pytest.raises(SchemaVersionError):
            resolve_schema_version(data)

    def test_migrate_result_v0_to_v1(self):
        old_data = {
            "id": "run-1",
            "project": "legacy",
            "metrics": [
                {
                    "request_id": "req-1",
                    "prompt_id": "prompt-1",
                    "prompt_tokens": 10,
                    "generated_tokens": 20,
                    "ttft_ms": 50.0,
                    "decode_ms": 200.0,
                    "total_wall_ms": 250.0,
                    "tokens_per_second_decode": 100.0,
                    "tokens_per_second_wall": 80.0,
                    "finish_reason": "stop",
                }
            ],
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["schema_version"] == "llm_bench.run_result.v1"
        assert migrated["run_id"] == "run-1"
        assert migrated["project"] == "legacy"
        assert len(migrated["per_request"]) == 1

    def test_migrate_preserves_existing_fields(self):
        old_data = {
            "schema_version": "llm_bench.run_result.v1",
            "run_id": "existing-id",
            "project": "existing-project",
            "per_request": [],
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["run_id"] == "existing-id"
        assert migrated["project"] == "existing-project"
        assert migrated["schema_version"] == "llm_bench.run_result.v1"
