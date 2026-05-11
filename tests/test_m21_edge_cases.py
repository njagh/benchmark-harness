"""Tests for M21 — Edge cases for fingerprinting, symlinks, and registry query.

Covers edge cases not covered by the main artifact tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bench_harness.registry import ArtifactRegistry, manage_artifact
from bench_harness.schemas.model_artifact import ModelArtifact, ArtifactKind, ArtifactMode
from bench_harness.storage.config import StorageConfig
from bench_harness.utils.hashing import compute_artifact_fingerprint

KIND_HF = str(ArtifactKind.hf_checkpoint)
KIND_GGUF = str(ArtifactKind.gguf)


class TestComputeArtifactFingerprintEdgeCases:
    def test_fingerprint_missing_torch_dtype(self, tmp_path):
        """Fingerprint handles config.json without torch_dtype gracefully."""
        src = tmp_path / "no-dtype-model"
        src.mkdir()
        (src / "config.json").write_text(
            '{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}'
        )
        fp = compute_artifact_fingerprint(None, src)
        assert fp is not None

    def test_fingerprint_empty_config(self, tmp_path):
        """Fingerprint handles empty config.json."""
        src = tmp_path / "empty-config-model"
        src.mkdir()
        (src / "config.json").write_text("{}")
        fp = compute_artifact_fingerprint(None, src)
        assert fp is not None

    def test_fingerprint_missing_config(self, tmp_path):
        """Fingerprint works with no config.json at all."""
        src = tmp_path / "no-config-model"
        src.mkdir()
        (src / "tokenizer.json").write_text("{}")
        fp = compute_artifact_fingerprint(None, src)
        assert fp is not None

    def test_fingerprint_config_parsing_error(self, tmp_path):
        """Fingerprint handles corrupt config.json."""
        src = tmp_path / "corrupt-config-model"
        src.mkdir()
        (src / "config.json").write_text("{not json")
        fp = compute_artifact_fingerprint(None, src)
        assert fp is not None

    def test_fingerprint_large_model(self, tmp_path):
        """Fingerprint handles large artifacts efficiently without hash_weights."""
        src = tmp_path / "large-model"
        src.mkdir()
        large_file = src / "large.bin"
        large_file.write_bytes(b"x" * (50 * 1024 * 1024))  # 50MB
        (src / "config.json").write_text('{"model_type": "test"}')
        fp = compute_artifact_fingerprint(None, src, hash_weights=False)
        assert fp is not None


class TestSymlinkArtifactEdgeCases:
    def test_symlink_with_spaces_in_path(self, tmp_path):
        """managed_symlink handles paths with spaces."""
        src = tmp_path / "my model" / "with spaces"
        src.mkdir(parents=True)
        (src / "config.json").write_text("{}")

        artifact = ModelArtifact(
            artifact_id="symlink-spaces",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            source_path=str(src),
        )
        config = StorageConfig(root=tmp_path / "storage")
        result = manage_artifact(artifact, config)
        assert Path(result).exists()

    def test_symlink_different_ids(self, tmp_path):
        """managed_symlink with different artifact_ids creates separate symlinks."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "config.json").write_text("{}")

        src2 = tmp_path / "source2"
        src2.mkdir()
        (src2 / "config.json").write_text('{"different": true}')

        artifact = ModelArtifact(
            artifact_id="symlink-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            source_path=str(src),
        )
        config = StorageConfig(root=tmp_path / "storage")
        result1 = manage_artifact(artifact, config)
        assert Path(result1).is_symlink()

        artifact2 = ModelArtifact(
            artifact_id="symlink-002",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.managed_symlink,
            source_path=str(src2),
        )
        result2 = manage_artifact(artifact2, config)
        assert Path(result2).is_symlink()


class TestArtifactRegistryQueryEdgeCases:
    def test_query_by_partial_model_id(self, tmp_path):
        """query() partial match on model_id finds substring."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        registry.register(ModelArtifact(
            artifact_id="a1", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="meta-llama/Llama-2-7b",
        ))
        registry.register(ModelArtifact(
            artifact_id="a2", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="mistralai/Mistral-7B-v0.1",
        ))

        results = registry.query(project="llama")
        assert len(results) == 1
        assert results[0].artifact_id == "a1"

        results = registry.query(project="mistral")
        assert len(results) == 1
        assert results[0].artifact_id == "a2"

    def test_query_by_quantization(self, tmp_path):
        """query() filters by quantization string."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        registry.register(ModelArtifact(
            artifact_id="int4-001", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            quantization="int4",
        ))
        registry.register(ModelArtifact(
            artifact_id="int8-001", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            quantization="int8",
        ))
        registry.register(ModelArtifact(
            artifact_id="fp16-001", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            quantization="fp16",
        ))

        results = registry.query(kind=KIND_HF, quantization="int4")
        assert len(results) == 1
        assert results[0].artifact_id == "int4-001"

        results = registry.query(kind=KIND_HF, quantization="fp16")
        assert len(results) == 1
        assert results[0].artifact_id == "fp16-001"

    def test_query_combined_filters(self, tmp_path):
        """query() combines kind and project filters."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        registry.register(ModelArtifact(
            artifact_id="a1", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="meta-llama/Llama-2-7b",
        ))
        registry.register(ModelArtifact(
            artifact_id="a2", kind=ArtifactKind.gguf,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="meta-llama/Llama-2-7b",
        ))
        registry.register(ModelArtifact(
            artifact_id="a3", kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path, source_path="/models",
            model_id="gpt2",
        ))

        results = registry.query(kind=KIND_HF, project="llama")
        assert len(results) == 1
        assert results[0].artifact_id == "a1"

        results = registry.query(kind=KIND_HF, project="gpt")
        assert len(results) == 1
        assert results[0].artifact_id == "a3"

    def test_query_no_match(self, tmp_path):
        """query() returns empty when no matches."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))
        results = registry.query(kind=KIND_GGUF)
        assert results == []

    def test_registry_duplicate_registration(self, tmp_path):
        """Registering same artifact_id twice appends both entries."""
        registry = ArtifactRegistry(StorageConfig(root=tmp_path / "storage"))

        artifact = ModelArtifact(
            artifact_id="dup-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path,
            source_path="/models/v1",
        )
        registry.register(artifact)

        artifact2 = ModelArtifact(
            artifact_id="dup-001",
            kind=ArtifactKind.hf_checkpoint,
            mode=ArtifactMode.external_path,
            source_path="/models/v2",
        )
        registry.register(artifact2)

        all_artifacts = registry.list_all()
        assert len(all_artifacts) == 2

        found = registry.lookup("dup-001")
        assert found.source_path == "/models/v1"
