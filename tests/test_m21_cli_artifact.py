"""Tests for M21 — CLI register-artifact integration.

Covers the CLI commands for artifact registration and inspection.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bench_harness.registry import ArtifactRegistry, manage_artifact
from bench_harness.schemas.model_artifact import ModelArtifact, ArtifactMode, ArtifactKind
from bench_harness.storage.config import StorageConfig

KIND_HF = str(ArtifactKind.hf_checkpoint)
KIND_GGUF = str(ArtifactKind.gguf)


def _make_artifact_dir(tmp_path: Path, name="test-model") -> Path:
    """Create a small directory with model files."""
    src = tmp_path / name
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    (src / "tokenizer.json").write_text("{}")
    return src


class TestArtifactCLI:
    def test_register_artifact_external(self, tmp_path):
        """register_artifact with external_path returns source path."""
        src = _make_artifact_dir(tmp_path, "external-model")
        artifact = ModelArtifact(
            artifact_id="ext-model-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path,
            source_path=str(src),
        )
        config = StorageConfig(root=tmp_path / "storage")
        result = manage_artifact(artifact, config)
        assert Path(result) == src

    def test_register_artifact_copy(self, tmp_path):
        """register_artifact with managed_copy creates a copy."""
        src = _make_artifact_dir(tmp_path, "source-model")
        artifact = ModelArtifact(
            artifact_id="copy-model-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_copy,
            source_path=str(src),
        )
        config = StorageConfig(root=tmp_path / "storage")
        result = manage_artifact(artifact, config)
        assert Path(result).exists()
        assert (Path(result) / "config.json").exists()
        assert (Path(result) / "tokenizer.json").exists()

    def test_register_artifact_symlink(self, tmp_path):
        """register_artifact with managed_symlink creates a symlink."""
        src = _make_artifact_dir(tmp_path, "source-model")
        artifact = ModelArtifact(
            artifact_id="symlink-model-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            source_path=str(src),
        )
        config = StorageConfig(root=tmp_path / "storage")
        result = manage_artifact(artifact, config)
        assert Path(result).is_symlink()

    def test_artifact_registry_roundtrip(self, tmp_path):
        """Register and lookup artifact through registry."""
        src = _make_artifact_dir(tmp_path, "reg-model")
        artifact = ModelArtifact(
            artifact_id="reg-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path,
            source_path=str(src),
        )

        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))
        registry.register(artifact)

        found = registry.lookup("reg-001")
        assert found is not None
        assert found.artifact_id == "reg-001"
        assert found.kind == ArtifactKind.hf_checkpoint
        assert found.source_path == str(src)

    def test_artifact_registry_list_all(self, tmp_path):
        """list_all returns all registered artifacts."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        for i in range(5):
            art = ModelArtifact(
                artifact_id=f"list-test-{i}",
                kind=ArtifactKind.hf_checkpoint,
                mode=ArtifactMode.external_path,
                source_path=f"/models/model-{i}",
            )
            registry.register(art)

        all_artifacts = registry.list_all()
        assert len(all_artifacts) == 5

    def test_artifact_registry_query_by_kind(self, tmp_path):
        """query() filters by kind."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        registry.register(ModelArtifact(
            artifact_id="hf-001", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models/one",
        ))
        registry.register(ModelArtifact(
            artifact_id="hf-002", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models/two",
        ))
        registry.register(ModelArtifact(
            artifact_id="gguf-001", kind=ArtifactKind.gguf,
            mode=ArtifactMode.external_path, source_path="/models/three",
        ))

        hf_results = registry.query(kind=KIND_HF)
        assert len(hf_results) == 2
        assert all(a.kind == ArtifactKind.hf_checkpoint for a in hf_results)

        gguf_results = registry.query(kind=KIND_GGUF)
        assert len(gguf_results) == 1
        assert gguf_results[0].kind == ArtifactKind.gguf

    def test_artifact_registry_query_by_project(self, tmp_path):
        """query() filters by project (substring match on model_id)."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        registry.register(ModelArtifact(
            artifact_id="proj-001", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="meta-llama/Llama-2-7b",
        ))
        registry.register(ModelArtifact(
            artifact_id="proj-002", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="mistralai/Mistral-7B-v0.1",
        ))
        registry.register(ModelArtifact(
            artifact_id="proj-003", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="gpt2",
        ))

        results = registry.query(project="llama")
        assert len(results) == 1
        assert results[0].artifact_id == "proj-001"

    def test_artifact_registry_query_no_match(self, tmp_path):
        """query() returns empty list when no matches."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))
        results = registry.query(kind=KIND_GGUF)
        assert results == []
