"""Tests specifically for ArtifactMode — all 3 modes, enum serialization,
deserialization, and validation with string coercion."""

from __future__ import annotations

import json
import pytest

from bench_harness.schemas.model_artifact import (
    ModelArtifact,
    ArtifactKind,
    ArtifactMode,
)
from bench_harness.schemas.run_spec import ArtifactSpec, RunSpec, WorkloadSpec


# ── All 3 modes ──────────────────────────────────────────────────────


class TestArtifactModeAllModes:
    def test_external_path_mode(self):
        """ArtifactMode.external_path is available and has correct value."""
        mode = ArtifactMode.external_path
        assert mode.value == "external_path"
        assert isinstance(mode, str)

    def test_managed_copy_mode(self):
        """ArtifactMode.managed_copy is available and has correct value."""
        mode = ArtifactMode.managed_copy
        assert mode.value == "managed_copy"
        assert isinstance(mode, str)

    def test_managed_symlink_mode(self):
        """ArtifactMode.managed_symlink is available and has correct value."""
        mode = ArtifactMode.managed_symlink
        assert mode.value == "managed_symlink"
        assert isinstance(mode, str)

    def test_external_path_in_model_artifact(self):
        """ModelArtifact can be created with external_path mode."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.external_path,
        )
        assert artifact.mode == ArtifactMode.external_path

    def test_managed_copy_in_model_artifact(self):
        """ModelArtifact can be created with managed_copy mode."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_copy,
        )
        assert artifact.mode == ArtifactMode.managed_copy

    def test_managed_symlink_in_model_artifact(self):
        """ModelArtifact can be created with managed_symlink mode."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_symlink,
        )
        assert artifact.mode == ArtifactMode.managed_symlink

    def test_external_path_in_artifact_spec(self):
        """ArtifactSpec accepts external_path mode."""
        spec = ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path,
            path="/models/test",
        )
        assert spec.mode == ArtifactMode.external_path

    def test_managed_copy_in_artifact_spec(self):
        """ArtifactSpec accepts managed_copy mode."""
        spec = ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_copy,
            path="/models/test",
        )
        assert spec.mode == ArtifactMode.managed_copy

    def test_managed_symlink_in_artifact_spec(self):
        """ArtifactSpec accepts managed_symlink mode."""
        spec = ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            path="/models/test",
        )
        assert spec.mode == ArtifactMode.managed_symlink

    def test_modes_are_equal_comparable(self):
        """Modes can be compared with ==."""
        assert ArtifactMode.external_path == ArtifactMode.external_path
        assert ArtifactMode.managed_copy == ArtifactMode.managed_copy
        assert ArtifactMode.managed_symlink == ArtifactMode.managed_symlink

    def test_modes_are_not_equal_to_each_other(self):
        """Different modes are not equal."""
        assert ArtifactMode.external_path != ArtifactMode.managed_copy
        assert ArtifactMode.external_path != ArtifactMode.managed_symlink
        assert ArtifactMode.managed_copy != ArtifactMode.managed_symlink

    def test_modes_have_correct_values(self):
        """Each mode's .value matches its name pattern."""
        assert ArtifactMode.external_path.value == "external_path"
        assert ArtifactMode.managed_copy.value == "managed_copy"
        assert ArtifactMode.managed_symlink.value == "managed_symlink"


# ── Enum serialization / deserialization ─────────────────────────────


class TestEnumSerialization:
    def test_serialize_external_path(self):
        """external_path serializes to string 'external_path'."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.external_path,
        )
        data = json.loads(artifact.to_json())
        assert data["mode"] == "external_path"

    def test_serialize_managed_copy(self):
        """managed_copy serializes to string 'managed_copy'."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_copy,
        )
        data = json.loads(artifact.to_json())
        assert data["mode"] == "managed_copy"

    def test_serialize_managed_symlink(self):
        """managed_symlink serializes to string 'managed_symlink'."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_symlink,
        )
        data = json.loads(artifact.to_json())
        assert data["mode"] == "managed_symlink"

    def test_deserialize_string_to_mode(self):
        """String 'external_path' coerces to ArtifactMode.external_path."""
        mode = ArtifactMode("external_path")
        assert mode == ArtifactMode.external_path

    def test_deserialize_string_managed_copy(self):
        """String 'managed_copy' coerces to ArtifactMode.managed_copy."""
        mode = ArtifactMode("managed_copy")
        assert mode == ArtifactMode.managed_copy

    def test_deserialize_string_managed_symlink(self):
        """String 'managed_symlink' coerces to ArtifactMode.managed_symlink."""
        mode = ArtifactMode("managed_symlink")
        assert mode == ArtifactMode.managed_symlink

    def test_model_dump_python_serializes_mode(self):
        """model_dump(mode='python') serializes mode to string."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_copy,
        )
        data = artifact.model_dump(mode="python")
        assert data["mode"] == "managed_copy"

    def test_mode_in_artifact_spec_dump(self):
        """ArtifactSpec model_dump serializes mode to string."""
        spec = ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            path="/models/test",
        )
        data = spec.model_dump(mode="python")
        assert data["mode"] == "managed_symlink"


# ── Validation with string coercion ──────────────────────────────────


class TestStringCoercionValidation:
    def test_invalid_mode_string_raises(self):
        """Invalid mode string raises ValueError on construction."""
        with pytest.raises(ValueError):
            ArtifactMode("invalid_mode_string")

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactMode("")

    def test_uppercase_string_raises(self):
        """Uppercase string raises ValueError (case-sensitive)."""
        with pytest.raises(ValueError):
            ArtifactMode("EXTERNAL_PATH")

    def test_model_artifact_rejects_invalid_mode_from_dict(self):
        """ModelArtifact validates mode from dict constructor."""
        with pytest.raises(Exception):
            ModelArtifact(
                artifact_id="art-1",
                kind=ArtifactKind.hf_checkpoint,
                source_path="/models/test",
                mode=ArtifactMode("invalid"),
            )

    def test_artifact_spec_accepts_string_mode(self):
        """ArtifactSpec coerces string mode to enum."""
        spec = ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            mode="external_path",
            path="/models/test",
        )
        assert spec.mode == ArtifactMode.external_path

    def test_run_spec_artifact_mode_string_coercion(self):
        """RunSpec artifact's mode is coerced from string."""
        run = RunSpec(
            name="test-model",
            project="test_project",
            artifact=ArtifactSpec(
                kind=ArtifactKind.hf_checkpoint,
                mode="managed_copy",
                path="/models/test",
            ),
            workload=WorkloadSpec(prompt_suite="smoke"),
        )
        assert run.artifact.mode == ArtifactMode.managed_copy

    def test_mode_equality_after_string_coercion(self):
        """Mode after string coercion equals the enum member."""
        mode = ArtifactMode("external_path")
        assert mode is ArtifactMode.external_path
        assert mode == ArtifactMode.external_path

    def test_mode_hash_unchanged_after_coercion(self):
        """Mode hash is consistent after string coercion."""
        mode1 = ArtifactMode.external_path
        mode2 = ArtifactMode("external_path")
        assert hash(mode1) == hash(mode2)
