"""Tests for RunSpec — model validation, serialization, enum conversion,
and build helpers."""

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


# ── RunSpec model ────────────────────────────────────────────────────


class TestRunSpec:
    def test_valid_run_spec(self):
        """A valid RunSpec is created successfully."""
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

    def test_required_name_field(self):
        """name field is required."""
        with pytest.raises(Exception):
            RunSpec(
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_required_project_field(self):
        """project field is required."""
        with pytest.raises(Exception):
            RunSpec(
                name="test-run",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_required_artifact_field(self):
        """artifact field is required."""
        with pytest.raises(Exception):
            RunSpec(
                name="test-run",
                project="p",
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_required_workload_field(self):
        """workload is required and prompt_suite is required within it."""
        with pytest.raises(Exception):
            RunSpec(
                name="test-run",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
            )

    def test_defaults_schema_version(self):
        """schema_version defaults to llm_bench.run_spec.v1."""
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.schema_version == "llm_bench.run_spec.v1"

    def test_defaults_tags_empty_list(self):
        """tags defaults to empty list."""
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.tags == []

    def test_custom_tags(self):
        """Custom tags are stored."""
        spec = RunSpec(
            name="test-model-smoke",
            project="p",
            tags=["test", "benchmark", "smoke"],
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.tags == ["test", "benchmark", "smoke"]

    def test_name_validation_rejects_single_char(self):
        """name must be at least 2 characters."""
        with pytest.raises(Exception):
            RunSpec(
                name="a",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_name_validation_rejects_uppercase(self):
        """name must be lowercase."""
        with pytest.raises(Exception):
            RunSpec(
                name="Test-Model",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_name_validation_rejects_hyphen_first(self):
        """name cannot start with hyphen."""
        with pytest.raises(Exception):
            RunSpec(
                name="-test-model",
                project="p",
                artifact=ArtifactSpec(
                    kind=ArtifactKind.hf_checkpoint,
                    path="/models/test",
                ),
                workload=WorkloadSpec(prompt_suite="smoke"),
            )

    def test_name_validation_accepts_slug(self):
        """name accepts valid slug format."""
        spec = RunSpec(
            name="test-model-v2",
            project="p",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert spec.name == "test-model-v2"

    def test_serialization_to_dict(self):
        """RunSpec can be serialized to dict."""
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            tags=["smoke"],
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
                model_id="test-model",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        data = spec.model_dump(mode="python")
        assert data["name"] == "test-model-smoke"
        assert data["project"] == "test_project"


# ── RuntimeSpec ──────────────────────────────────────────────────────


class TestRuntimeSpec:
    def test_basic_runtime_spec(self):
        """RuntimeSpec with minimal fields."""
        runtime = RuntimeSpec(kind=RuntimeKind.vllm)
        assert runtime.kind == RuntimeKind.vllm
        assert runtime.launch == LaunchMode.existing

    def test_runtime_with_launch_mode(self):
        """RuntimeSpec respects launch mode."""
        runtime = RuntimeSpec(
            kind=RuntimeKind.vllm,
            launch=LaunchMode.managed_process,
        )
        assert runtime.launch == LaunchMode.managed_process

    def test_runtime_with_host_port(self):
        """RuntimeSpec stores host and port."""
        runtime = RuntimeSpec(
            kind=RuntimeKind.openai_compatible,
            host="127.0.0.1",
            port=8009,
        )
        assert runtime.host == "127.0.0.1"
        assert runtime.port == 8009

    def test_runtime_with_model_name(self):
        """RuntimeSpec stores model_name."""
        runtime = RuntimeSpec(
            kind=RuntimeKind.vllm,
            model_name="my-model",
        )
        assert runtime.model_name == "my-model"

    def test_runtime_args_default_empty(self):
        """RuntimeSpec args defaults to empty dict."""
        runtime = RuntimeSpec(kind=RuntimeKind.vllm)
        assert runtime.args == {}

    def test_runtime_args_custom(self):
        """RuntimeSpec stores custom args."""
        runtime = RuntimeSpec(
            kind=RuntimeKind.vllm,
            args={"gpu_memory_utilization": 0.9, "max_model_len": 4096},
        )
        assert runtime.args["gpu_memory_utilization"] == 0.9

    def test_all_runtime_kinds(self):
        """All RuntimeKind enum values are available."""
        for kind in RuntimeKind:
            runtime = RuntimeSpec(kind=kind)
            assert runtime.kind == kind

    def test_all_launch_modes(self):
        """All LaunchMode enum values are available."""
        for mode in LaunchMode:
            runtime = RuntimeSpec(kind=RuntimeKind.vllm, launch=mode)
            assert runtime.launch == mode


# ── WorkloadSpec ─────────────────────────────────────────────────────


class TestWorkloadSpec:
    def test_basic_workload_spec(self):
        """WorkloadSpec with minimal required fields."""
        wl = WorkloadSpec(prompt_suite="smoke")
        assert wl.prompt_suite == "smoke"
        assert wl.max_tokens == 256
        assert wl.temperature == 0.0
        assert wl.num_runs == 1
        assert wl.concurrency == 1

    def test_custom_max_tokens(self):
        """WorkloadSpec accepts custom max_tokens."""
        wl = WorkloadSpec(prompt_suite="smoke", max_tokens=512)
        assert wl.max_tokens == 512

    def test_custom_temperature(self):
        """WorkloadSpec accepts custom temperature."""
        wl = WorkloadSpec(prompt_suite="smoke", temperature=0.7)
        assert wl.temperature == 0.7

    def test_custom_num_runs(self):
        """WorkloadSpec accepts custom num_runs."""
        wl = WorkloadSpec(prompt_suite="smoke", num_runs=5)
        assert wl.num_runs == 5

    def test_custom_concurrency(self):
        """WorkloadSpec accepts custom concurrency."""
        wl = WorkloadSpec(prompt_suite="smoke", concurrency=4)
        assert wl.concurrency == 4

    def test_task_dir_none_by_default(self):
        """task_dir is None by default."""
        wl = WorkloadSpec(prompt_suite="smoke")
        assert wl.task_dir is None

    def test_custom_task_dir(self):
        """WorkloadSpec stores custom task_dir."""
        wl = WorkloadSpec(prompt_suite="smoke", task_dir="/tasks/custom")
        assert wl.task_dir == "/tasks/custom"


# ── StoragePolicy ────────────────────────────────────────────────────


class TestStoragePolicy:
    def test_default_artifact_policy(self):
        """artifact_policy defaults to external_path."""
        policy = StoragePolicy()
        assert policy.artifact_policy == ArtifactMode.external_path

    def test_default_result_policy(self):
        """result_policy defaults to 'managed'."""
        policy = StoragePolicy()
        assert policy.result_policy == "managed"

    def test_custom_artifact_policy(self):
        """StoragePolicy accepts custom artifact_policy."""
        policy = StoragePolicy(
            artifact_policy=ArtifactMode.managed_copy,
        )
        assert policy.artifact_policy == ArtifactMode.managed_copy

    def test_custom_result_policy(self):
        """StoragePolicy accepts custom result_policy."""
        policy = StoragePolicy(result_policy="external")
        assert policy.result_policy == "external"


# ── from_yaml / to_json ──────────────────────────────────────────────


class TestSerialization:
    def test_to_yaml_from_yaml_roundtrip(self):
        """Round-trip through YAML preserves key fields."""
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            tags=["smoke", "test"],
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
                model_id="test-model",
            ),
            workload=WorkloadSpec(prompt_suite="smoke", max_tokens=512),
        )
        yaml_str = spec.to_yaml()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml_path = f.name
            f.write(yaml_str)
        try:
            reloaded = RunSpec.from_yaml(yaml_path)
            assert reloaded.name == spec.name
            assert reloaded.project == spec.project
            assert reloaded.tags == spec.tags
            assert reloaded.workload.max_tokens == 512
        finally:
            Path(yaml_path).unlink()

    def test_to_json_from_json_roundtrip(self):
        """Round-trip through JSON preserves key fields."""
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
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json_path = f.name
            f.write(spec.to_json())
        try:
            reloaded = RunSpec.from_json(json_path)
            assert reloaded.name == spec.name
            assert reloaded.project == spec.project
        finally:
            Path(json_path).unlink()

    def test_to_json_is_valid_json(self):
        """to_json produces valid JSON."""
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        data = json.loads(spec.to_json())
        assert data["name"] == "test-model-smoke"

    def test_yaml_contains_enum_values(self):
        """to_yaml serializes enums as their string values."""
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            runtime=RuntimeSpec(kind=RuntimeKind.vllm),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        yaml_str = spec.to_yaml()
        assert "hf_checkpoint" in yaml_str
        assert "vllm" in yaml_str

    def test_from_yaml_with_string_enums(self):
        """from_yaml converts string enum values back to enum members."""
        spec = RunSpec(
            name="test-model-smoke",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        yaml_str = spec.to_yaml()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml_path = f.name
            f.write(yaml_str)
        try:
            reloaded = RunSpec.from_yaml(yaml_path)
            assert isinstance(reloaded.artifact.kind, ArtifactKind)
            assert isinstance(reloaded.runtime.kind, RuntimeKind)
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_file_not_found(self):
        """from_yaml raises when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            RunSpec.from_yaml("/nonexistent/path/spec.yaml")

    def test_run_spec_with_all_fields(self):
        """RunSpec with all optional fields set."""
        spec = RunSpec(
            name="full-test",
            project="my_project",
            tags=["full", "test"],
            hardware=HardwareSpec(profile="default", expected_gpu="A100"),
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                mode=ArtifactMode.managed_copy,
                path="/models/test",
                tokenizer_path="/models/test/tokenizer",
                model_id="test-model",
                quantization="FP16",
            ),
            runtime=RuntimeSpec(
                kind=RuntimeKind.vllm,
                launch=LaunchMode.managed_process,
                host="127.0.0.1",
                port=8000,
                model_name="test-model",
                args={"max_model_len": 2048},
            ),
            workload=WorkloadSpec(
                prompt_suite="smoke",
                max_tokens=512,
                temperature=0.5,
                num_runs=3,
                concurrency=2,
                task_dir="/tasks/smoke",
            ),
            storage=StoragePolicy(
                artifact_policy=ArtifactMode.managed_copy,
                result_policy="external",
            ),
        )
        assert spec.hardware.expected_gpu == "A100"
        assert spec.artifact.mode == ArtifactMode.managed_copy
        assert spec.workload.num_runs == 3


# ── build_run_spec_from_flags ────────────────────────────────────────


class TestBuildRunSpecFromFlags:
    def test_build_default_params(self):
        """build_run_spec_from_flags with defaults."""
        spec = build_run_spec_from_flags(
            suite="smoke",
            models=["test-model"],
        )
        assert spec.name == "test-model-smoke-cli"
        assert spec.project == "default"
        assert spec.workload.max_tokens == 256
        assert spec.workload.num_runs == 1
        assert spec.hardware.profile == "small"

    def test_build_custom_params(self):
        """build_run_spec_from_flags with custom params."""
        spec = build_run_spec_from_flags(
            suite="benchmark",
            models=["big-model"],
            num_runs=10,
            max_tokens=1024,
            temperature=0.8,
            concurrency=4,
            context_tokens="large",
            project="my_project",
        )
        assert spec.name == "big-model-benchmark-cli"
        assert spec.project == "my_project"
        assert spec.workload.num_runs == 10
        assert spec.workload.max_tokens == 1024
        assert spec.workload.temperature == 0.8
        assert spec.workload.concurrency == 4
        assert spec.hardware.profile == "large"

    def test_build_empty_models(self):
        """build_run_spec_from_flags with empty models list."""
        spec = build_run_spec_from_flags(
            suite="smoke",
            models=[],
        )
        assert "unknown" in spec.name
