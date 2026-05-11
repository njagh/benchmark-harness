"""Tests for M21 — ModelOpt metadata hooks.

Covers `hooks.py`: ArtifactMetadataHook ABC and ModelOptMetadataHook.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bench_harness.hooks import ArtifactMetadataHook, ModelOptMetadataHook
from bench_harness.schemas.model_artifact import ModelArtifact
from bench_harness.schemas.run_result import RunResult


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def modelopt_meta(tmp_path: Path) -> Path:
    """Create a modelopt_meta.json file."""
    meta = {
        "modelopt_version": "0.5.0",
        "quantization_algorithm": "AWQ",
        "calibration_dataset": "wikitext-2-raw-v1",
        "base_model_id": "meta-llama/Llama-2-7b",
        "git_commit": "abc123def456",
        "num_calibration_samples": 128,
        "export_format": "gptq",
    }
    meta_path = tmp_path / "modelopt_meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path


@pytest.fixture
def valid_artifact(tmp_path: Path) -> ModelArtifact:
    """A ModelArtifact with a source path."""
    return ModelArtifact(
        artifact_id="test-model-001",
        kind="hf_checkpoint",
        mode="external_path",
        source_path=str(tmp_path / "models" / "test-model"),
    )


@pytest.fixture
def valid_run_result() -> RunResult:
    """A RunResult with artifact fingerprint."""
    return RunResult(
        run_id="run-001",
        run_spec_ref="spec-001",
        project="test_project",
        artifact_fingerprint={"detected_dtype": "float16"},
    )


# ── ArtifactMetadataHook ABC ─────────────────────────────────────────


class TestArtifactMetadataHook:
    def test_abstract_class_cannot_be_instantiated(self):
        """ArtifactMetadataHook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ArtifactMetadataHook()

    def test_concrete_implementation_works(self):
        """A concrete implementation can be instantiated and used."""
        class TestHook(ArtifactMetadataHook):
            def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
                return artifact

            def enrich_run_result(self, result: RunResult) -> RunResult:
                return result

        hook = TestHook()
        assert hook is not None

    def test_enrich_artifact_must_be_implemented(self):
        """Subclass without enrich_artifact raises TypeError."""
        class BadHook(ArtifactMetadataHook):
            def enrich_run_result(self, result: RunResult) -> RunResult:
                return result

        with pytest.raises(TypeError):
            BadHook()

    def test_enrich_run_result_must_be_implemented(self):
        """Subclass without enrich_run_result raises TypeError."""
        class BadHook(ArtifactMetadataHook):
            def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
                return artifact

        with pytest.raises(TypeError):
            BadHook()


# ── ModelOptMetadataHook ─────────────────────────────────────────────


class TestModelOptMetadataHook:
    def test_enrich_artifact_no_source(self, valid_artifact):
        """Artifact with no source_path is returned unchanged."""
        valid_artifact.source_path = ""
        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(valid_artifact)
        assert result.producing_version is None

    def test_enrich_artifact_no_meta_file(self, valid_artifact, tmp_path):
        """Artifact without modelopt_meta.json is returned unchanged."""
        valid_artifact.source_path = str(tmp_path / "no_meta")
        Path(valid_artifact.source_path).mkdir(parents=True)
        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(valid_artifact)
        assert result.producing_version is None

    def test_enrich_artifact_finds_meta(self, tmp_path):
        """Hook finds and reads modelopt_meta.json."""
        meta_dir = tmp_path / "models" / "test-model"
        meta_dir.mkdir(parents=True)
        meta_dir.joinpath("modelopt_meta.json").write_text(
            json.dumps({
                "modelopt_version": "0.5.0",
                "quantization_algorithm": "AWQ",
                "calibration_dataset": "wikitext-2-raw-v1",
                "base_model_id": "meta-llama/Llama-2-7b",
                "git_commit": "abc123def456",
                "num_calibration_samples": 128,
                "export_format": "gptq",
            })
        )

        artifact = ModelArtifact(
            artifact_id="test-model-001",
            kind="hf_checkpoint",
            mode="external_path",
            source_path=str(meta_dir),
        )

        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(artifact)

        assert result.producing_version == "0.5.0"
        assert result.quantization == "AWQ"
        assert result.model_id == "meta-llama/Llama-2-7b"
        assert result.producing_git_commit == "abc123def456"
        assert result.backend_version == "gptq"

    def test_enrich_artifact_nested_meta(self, tmp_path):
        """Hook finds modelopt_meta.json in nested directories."""
        src = tmp_path / "models" / "deep" / "nested"
        src.mkdir(parents=True)
        (src / "modelopt_meta.json").write_text(
            json.dumps({"modelopt_version": "1.0.0", "quantization_algorithm": "SqueezeLLM"})
        )

        artifact = ModelArtifact(
            artifact_id="test",
            kind="hf_checkpoint",
            mode="external_path",
            source_path=str(tmp_path / "models"),
        )

        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(artifact)

        assert result.producing_version == "1.0.0"
        assert result.quantization == "SqueezeLLM"

    def test_enrich_artifact_bad_json(self, valid_artifact, tmp_path):
        """Hook handles corrupt JSON gracefully."""
        meta_dir = tmp_path / "models" / "bad-json"
        meta_dir.mkdir(parents=True)
        (meta_dir / "modelopt_meta.json").write_text("{not valid json")
        valid_artifact.source_path = str(meta_dir)

        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(valid_artifact)
        # Should not crash, fields remain unset
        assert result.producing_version is None

    def test_enrich_artifact_preserves_existing(self, valid_artifact, tmp_path):
        """Hook preserves existing fields when no meta overrides."""
        valid_artifact.source_path = str(tmp_path / "no-meta")
        valid_artifact.producing_version = "pre-existing"
        Path(valid_artifact.source_path).mkdir(parents=True)

        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(valid_artifact)
        assert result.producing_version == "pre-existing"

    def test_enrich_run_result_no_fingerprint(self, valid_run_result):
        """Run result with empty fingerprint is returned unchanged."""
        valid_run_result.artifact_fingerprint = {}
        hook = ModelOptMetadataHook()
        result = hook.enrich_run_result(valid_run_result)
        assert result.artifact_fingerprint.get("hook") is None

    def test_enrich_run_result_with_fingerprint(self, valid_run_result):
        """Run result with fingerprint gets hook field."""
        hook = ModelOptMetadataHook()
        result = hook.enrich_run_result(valid_run_result)
        assert result.artifact_fingerprint.get("hook") == "modelopt"

    def test_enrich_run_result_preserves_existing_fingerprint(self, valid_run_result):
        """Hook does not overwrite existing fingerprint fields."""
        valid_run_result.artifact_fingerprint["detected_dtype"] = "float16"
        hook = ModelOptMetadataHook()
        result = hook.enrich_run_result(valid_run_result)
        assert result.artifact_fingerprint["detected_dtype"] == "float16"
        assert result.artifact_fingerprint["hook"] == "modelopt"

    def test_enrich_artifact_empty_source_path(self):
        """Hook handles artifact with empty source_path gracefully."""
        artifact = ModelArtifact(
            artifact_id="test",
            kind="hf_checkpoint",
            mode="external_path",
            source_path="",
        )
        hook = ModelOptMetadataHook()
        # Should not raise, source_path is empty so no meta found
        result = hook.enrich_artifact(artifact)
        assert result.source_path == ""

    def test_calibration_dataset_mapping(self, valid_artifact, tmp_path):
        """Hook maps calibration_dataset to file_list_summary field."""
        meta_dir = tmp_path / "models" / "calib"
        meta_dir.mkdir(parents=True)
        (meta_dir / "modelopt_meta.json").write_text(
            json.dumps({"calibration_dataset": "c4-dataset"})
        )
        valid_artifact.source_path = str(meta_dir)

        hook = ModelOptMetadataHook()
        result = hook.enrich_artifact(valid_artifact)
        assert result.file_list_summary == "c4-dataset"