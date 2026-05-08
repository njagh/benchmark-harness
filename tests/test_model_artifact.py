"""Tests for model_artifact.py — ModelArtifact fields, enums, validation,
and serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.schemas.model_artifact import (
    ModelArtifact,
    ArtifactKind,
    ArtifactMode,
)


# ── ModelArtifact fields ─────────────────────────────────────────────


class TestModelArtifactFields:
    def test_all_fields_set(self):
        """ModelArtifact with all fields populated."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_copy,
            source_path="/models/test-model",
            model_id="test-model",
            quantization="FP8",
            dtype="float16",
            parameter_class="7b",
            tokenizer_path="/models/test-model/tokenizer",
            file_list_summary={"weights.bin": 1000, "config.json": 500},
            total_size_bytes=10737418240,
            config_file_hash="abc123",
            weight_manifest_hash="def456",
            created_at="2024-01-01T00:00:00",
            producing_git_commit="abc123def",
            producing_version="1.0.0",
            backend_version="v0.5.0",
            registered_at="2024-01-02T00:00:00",
            durable=True,
            artifact_warnings=["warning-1"],
        )
        assert artifact.artifact_id == "art-1"
        assert artifact.kind == ArtifactKind.hf_checkpoint
        assert artifact.mode == ArtifactMode.managed_copy
        assert artifact.model_id == "test-model"
        assert artifact.quantization == "FP8"
        assert artifact.dtype == "float16"
        assert artifact.parameter_class == "7b"
        assert artifact.tokenizer_path == "/models/test-model/tokenizer"
        assert artifact.total_size_bytes == 10737418240
        assert artifact.config_file_hash == "abc123"
        assert artifact.weight_manifest_hash == "def456"
        assert artifact.created_at == "2024-01-01T00:00:00"
        assert artifact.durable is True
        assert artifact.artifact_warnings == ["warning-1"]

    def test_required_fields(self):
        """artifact_id, kind, and source_path are required."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.artifact_id == "art-1"

    def test_defaults_schema_version(self):
        """schema_version defaults to v1."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.schema_version == "llm_bench.model_artifact.v1"

    def test_defaults_mode(self):
        """mode defaults to external_path."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.mode == ArtifactMode.external_path

    def test_defaults_durable(self):
        """durable defaults to True."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.durable is True

    def test_defaults_total_size_bytes(self):
        """total_size_bytes defaults to 0."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.total_size_bytes == 0

    def test_defaults_empty_lists(self):
        """file_list_summary and artifact_warnings default to empty dicts/lists."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.file_list_summary == {}
        assert artifact.artifact_warnings == []

    def test_defaults_none_fields(self):
        """Optional fields default to None."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        assert artifact.model_id is None
        assert artifact.quantization is None
        assert artifact.dtype is None
        assert artifact.parameter_class is None
        assert artifact.tokenizer_path is None
        assert artifact.config_file_hash is None
        assert artifact.weight_manifest_hash is None
        assert artifact.created_at is None


# ── ArtifactKind enum ────────────────────────────────────────────────


class TestArtifactKind:
    def test_all_values(self):
        """All 5 ArtifactKind values exist."""
        assert ArtifactKind.hf_checkpoint.value == "hf_checkpoint"
        assert ArtifactKind.trtllm_engine.value == "trtllm_engine"
        assert ArtifactKind.gguf.value == "gguf"
        assert ArtifactKind.vllm_endpoint.value == "vllm_endpoint"
        assert ArtifactKind.openai_endpoint.value == "openai_endpoint"

    def test_len_is_five(self):
        """ArtifactKind has exactly 5 members."""
        assert len(list(ArtifactKind)) == 5

    def test_from_string(self):
        """Can create ArtifactKind from string."""
        kind = ArtifactKind("hf_checkpoint")
        assert kind == ArtifactKind.hf_checkpoint

    def test_from_string_trtllm(self):
        kind = ArtifactKind("trtllm_engine")
        assert kind == ArtifactKind.trtllm_engine

    def test_from_string_gguf(self):
        kind = ArtifactKind("gguf")
        assert kind == ArtifactKind.gguf

    def test_from_string_vllm_endpoint(self):
        kind = ArtifactKind("vllm_endpoint")
        assert kind == ArtifactKind.vllm_endpoint

    def test_from_string_openai_endpoint(self):
        kind = ArtifactKind("openai_endpoint")
        assert kind == ArtifactKind.openai_endpoint

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactKind("invalid_kind")


# ── ArtifactMode enum ────────────────────────────────────────────────


class TestArtifactMode:
    def test_all_values(self):
        """All 3 ArtifactMode values exist."""
        assert ArtifactMode.external_path.value == "external_path"
        assert ArtifactMode.managed_copy.value == "managed_copy"
        assert ArtifactMode.managed_symlink.value == "managed_symlink"

    def test_len_is_three(self):
        """ArtifactMode has exactly 3 members."""
        assert len(list(ArtifactMode)) == 3

    def test_from_string(self):
        """Can create ArtifactMode from string."""
        mode = ArtifactMode("managed_copy")
        assert mode == ArtifactMode.managed_copy

    def test_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactMode("invalid_mode")

    def test_in_model_artifact_validation(self):
        """ModelArtifact validates mode correctly."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            mode=ArtifactMode.managed_copy,
        )
        assert artifact.mode == ArtifactMode.managed_copy


# ── from_yaml ─────────────────────────────────────────────────────────


class TestFromYaml:
    def test_from_yaml_basic(self):
        """ModelArtifact from YAML with minimal fields."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        yaml_str = artifact.to_json()  # We'll write a yaml file manually
        yaml_data = {
            "artifact_id": "art-1",
            "kind": "hf_checkpoint",
            "source_path": "/models/test",
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml_path = f.name
            yaml.dump(yaml_data, f)
        try:
            loaded = ModelArtifact.from_yaml(yaml_path)
            assert loaded.artifact_id == "art-1"
            assert loaded.kind == ArtifactKind.hf_checkpoint
            assert loaded.source_path == "/models/test"
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_with_mode(self):
        """from_yaml loads mode from string."""
        yaml_data = {
            "artifact_id": "art-2",
            "kind": "hf_checkpoint",
            "source_path": "/models/test",
            "mode": "managed_copy",
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml_path = f.name
            yaml.dump(yaml_data, f)
        try:
            loaded = ModelArtifact.from_yaml(yaml_path)
            assert loaded.mode == ArtifactMode.managed_copy
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_invalid_kind(self):
        """from_yaml with invalid kind passes through as-is."""
        yaml_data = {
            "artifact_id": "art-3",
            "kind": "invalid_kind",
            "source_path": "/models/test",
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml_path = f.name
            yaml.dump(yaml_data, f)
        try:
            with pytest.raises(Exception):
                ModelArtifact.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_file_not_found(self):
        """from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ModelArtifact.from_yaml("/nonexistent/path/artifact.yaml")


# ── to_json ──────────────────────────────────────────────────────────


class TestToJson:
    def test_to_json_basic(self):
        """to_json produces valid JSON with correct fields."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
            model_id="test-model",
        )
        data = json.loads(artifact.to_json())
        assert data["artifact_id"] == "art-1"
        assert data["kind"] == "hf_checkpoint"
        assert data["source_path"] == "/models/test"
        assert data["model_id"] == "test-model"

    def test_to_json_with_all_fields(self):
        """to_json with all fields populated."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.trtllm_engine,
            mode=ArtifactMode.managed_symlink,
            source_path="/models/engine",
            model_id="big-model",
            quantization="INT8",
            dtype="float16",
            parameter_class="70b",
            tokenizer_path="/models/engine/tokenizer",
            total_size_bytes=9999999999,
            durable=False,
        )
        data = json.loads(artifact.to_json())
        assert data["artifact_id"] == "art-1"
        assert data["kind"] == "trtllm_engine"
        assert data["mode"] == "managed_symlink"
        assert data["quantization"] == "INT8"
        assert data["dtype"] == "float16"
        assert data["parameter_class"] == "70b"
        assert data["total_size_bytes"] == 9999999999
        assert data["durable"] is False

    def test_to_json_empty_optional_fields(self):
        """to_json serializes None fields as null."""
        artifact = ModelArtifact(
            artifact_id="art-1",
            kind=ArtifactKind.hf_checkpoint,
            source_path="/models/test",
        )
        data = json.loads(artifact.to_json())
        assert data["model_id"] is None
        assert data["quantization"] is None
        assert data["artifact_warnings"] == []
        assert data["file_list_summary"] == {}
