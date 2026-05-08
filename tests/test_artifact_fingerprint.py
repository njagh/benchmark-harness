"""Tests for artifact fingerprinting utilities in hashing.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bench_harness.utils.hashing import (
    compute_artifact_fingerprint,
    compute_config_hash,
    compute_file_hash,
    compute_weight_manifest_hash,
    scan_artifact_path,
)


# ── compute_file_hash ─────────────────────────────────────────────────


class TestComputeFileHash:
    """Test compute_file_hash with various algorithms and edge cases."""

    def test_sha256(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_file_hash(f, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_md5(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_file_hash(f, "md5")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_sha1(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_file_hash(f, "sha1")
        assert isinstance(result, str)
        assert len(result) == 40

    def test_sha512(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_file_hash(f, "sha512")
        assert isinstance(result, str)
        assert len(result) == 128

    def test_sha256_deterministic(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"deterministic content")
        r1 = compute_file_hash(f, "sha256")
        r2 = compute_file_hash(f, "sha256")
        assert r1 == r2

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_large_file(self, tmp_path: Path):
        f = tmp_path / "large.bin"
        # Write 5 MB
        chunk = b"X" * 1024 * 1024
        with open(f, "wb") as fh:
            for _ in range(5):
                fh.write(chunk)
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_file_hash_default_algorithm(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"default test")
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_known_value(self, tmp_path: Path):
        f = tmp_path / "known.bin"
        f.write_bytes(b"")
        result = compute_file_hash(f, "sha256")
        # SHA-256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ── compute_config_hash ──────────────────────────────────────────────


class TestComputeConfigHash:
    """Test compute_config_hash with existing, missing, and symlink files."""

    def test_existing_config(self, tmp_path: Path):
        cfg = tmp_path / "config.json"
        cfg.write_text('{"key": "value"}')
        result = compute_config_hash(cfg)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 64

    def test_missing_config(self, tmp_path: Path):
        cfg = tmp_path / "nonexistent.json"
        result = compute_config_hash(cfg)
        assert result is None

    def test_symlink_config(self, tmp_path: Path):
        real_cfg = tmp_path / "real_config.json"
        real_cfg.write_text('{"model": "test"}')
        link = tmp_path / "link_config.json"
        link.symlink_to(real_cfg)
        result = compute_config_hash(link)
        assert result is not None
        assert isinstance(result, str)

    def test_same_content_same_hash(self, tmp_path: Path):
        c1 = tmp_path / "c1.json"
        c2 = tmp_path / "c2.json"
        c1.write_text('{"same": true}')
        c2.write_text('{"same": true}')
        assert compute_config_hash(c1) == compute_config_hash(c2)

    def test_broken_symlink(self, tmp_path: Path):
        link = tmp_path / "broken_link"
        link.symlink_to(tmp_path / "does_not_exist")
        result = compute_config_hash(link)
        assert result is None


# ── compute_weight_manifest_hash ─────────────────────────────────────


class TestComputeWeightManifestHash:
    """Test weight manifest hashing with various directory layouts."""

    def test_single_dir(self, tmp_path: Path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "model.safetensors").write_bytes(b"fake weight data " * 100)
        result = compute_weight_manifest_hash(weights_dir)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_nested_dirs(self, tmp_path: Path):
        base = tmp_path / "weights"
        sub1 = base / "shard1"
        sub2 = base / "shard2"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        (sub1 / "layer.bin").write_bytes(b"shard1 data")
        (sub2 / "layer.bin").write_bytes(b"shard2 data")
        result = compute_weight_manifest_hash(base)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_weights"
        empty_dir.mkdir()
        result = compute_weight_manifest_hash(empty_dir)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_symlink_in_weights(self, tmp_path: Path):
        base = tmp_path / "weights"
        base.mkdir()
        real = tmp_path / "real_weights.bin"
        real.write_bytes(b"real weight data")
        link = base / "symlinked.bin"
        link.symlink_to(real)
        result = compute_weight_manifest_hash(base)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_different_contents_different_hash(self, tmp_path: Path):
        base1 = tmp_path / "base1"
        base2 = tmp_path / "base2"
        base1.mkdir()
        base2.mkdir()
        (base1 / "weights.bin").write_bytes(b"content A")
        (base2 / "weights.bin").write_bytes(b"content B different size x")
        assert compute_weight_manifest_hash(base1) != compute_weight_manifest_hash(base2)

    def test_sorting_consistency(self, tmp_path: Path):
        """Ensure hash is deterministic regardless of filesystem ordering."""
        base = tmp_path / "weights"
        base.mkdir()
        files = ["z.bin", "a.bin", "m.bin"]
        for fname in files:
            (base / fname).write_bytes(fname.encode())
        r1 = compute_weight_manifest_hash(base)
        r2 = compute_weight_manifest_hash(base)
        assert r1 == r2


# ── scan_artifact_path ───────────────────────────────────────────────


class TestScanArtifactPath:
    """Test scan_artifact_path with various directory layouts."""

    def test_simple_directory(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
        (d / "model.bin").write_bytes(b"weight data")
        result = scan_artifact_path(d)
        assert result["file_count"] == 2
        assert result["total_size_bytes"] > 0
        assert "config.json" in result["files"]
        assert "model.bin" in result["files"]
        assert "config.json" in result["config_hashes"]

    def test_subdirectories(self, tmp_path: Path):
        d = tmp_path / "artifact"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "root.bin").write_bytes(b"root")
        (sub / "child.bin").write_bytes(b"child")
        result = scan_artifact_path(d)
        assert result["file_count"] == 2
        assert "sub/child.bin" in result["files"]

    def test_config_files_detected(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        config_files = [
            "config.json",
            "model.safetensors.index.json",
            "generation_config.json",
            "config.yaml",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ]
        for cf in config_files:
            (d / cf).write_text("{}")
        result = scan_artifact_path(d)
        assert len(result["config_hashes"]) == len(config_files)
        for cf in config_files:
            assert cf in result["config_hashes"]

    def test_multiple_configs(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"model": "a"}')
        (d / "generation_config.json").write_text('{"max_new_tokens": 100}')
        result = scan_artifact_path(d)
        assert "config.json" in result["config_hashes"]
        assert "generation_config.json" in result["config_hashes"]
        assert result["config_hashes"]["config.json"] != result["config_hashes"]["generation_config.json"]

    def test_empty_directory(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        result = scan_artifact_path(d)
        assert result["file_count"] == 0
        assert result["total_size_bytes"] == 0
        assert result["files"] == {}
        assert result["config_hashes"] == {}

    def test_non_utf8_content(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "binary.bin").write_bytes(b"\x80\x81\x82\x83")
        result = scan_artifact_path(d)
        assert result["file_count"] == 1
        assert "binary.bin" in result["files"]


# ── compute_artifact_fingerprint ─────────────────────────────────────


class TestComputeArtifactFingerprint:
    """Test compute_artifact_fingerprint with various configurations."""

    def test_with_config_json(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "_name_or_path": "meta-llama/Llama-2-7b",
            "torch_dtype": "float16",
            "architectures": ["LlamaForCausalLM"],
        }))
        (d / "model.bin").write_bytes(b"weight data")
        fp = compute_artifact_fingerprint(None, d, hash_weights=False)
        assert fp["config_file_hash"] is not None
        assert fp["detected_model_id"] == "meta-llama/Llama-2-7b"
        assert fp["detected_dtype"] == "float16"
        assert fp["parameter_class"] == "LlamaForCausalLM"

    def test_without_config(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "model.bin").write_bytes(b"weight data")
        fp = compute_artifact_fingerprint(None, d, hash_weights=False)
        assert fp["config_file_hash"] is None
        assert "detected_model_id" not in fp
        assert "detected_dtype" not in fp
        assert "parameter_class" not in fp

    def test_with_large_weights(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"_name_or_path": "test"}')
        # Create a >100MB file
        large_file = d / "model.safetensors"
        with open(large_file, "wb") as f:
            f.write(b"X" * (100 * 1024 * 1024 + 1))
        fp = compute_artifact_fingerprint(None, d, hash_weights=True)
        assert "weight_file_hashes" in fp
        assert isinstance(fp["weight_file_hashes"], dict)
        assert "model.safetensors" in fp["weight_file_hashes"]

    def test_without_hash_weights(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"_name_or_path": "test"}')
        large_file = d / "model.safetensors"
        with open(large_file, "wb") as f:
            f.write(b"X" * (100 * 1024 * 1024 + 1))
        fp = compute_artifact_fingerprint(None, d, hash_weights=False)
        assert "weight_file_hashes" not in fp

    def test_fp_has_required_fields(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "weights.bin").write_bytes(b"data")
        fp = compute_artifact_fingerprint(None, d)
        required = ["config_file_hash", "weight_manifest_hash", "total_size_bytes", "file_count"]
        for field in required:
            assert field in fp, f"Missing field: {field}"

    def test_multiple_config_files(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"_name_or_path": "model-a"}')
        (d / "generation_config.json").write_text('{"max_new_tokens": 50}')
        (d / "tokenizer.json").write_text("{}")
        fp = compute_artifact_fingerprint(None, d)
        assert fp["config_file_hash"] is not None

    def test_detected_dtype_str(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "_name_or_path": "test",
            "torch_dtype": "float32",
            "architectures": ["TestModel"],
        }))
        fp = compute_artifact_fingerprint(None, d)
        assert fp["detected_dtype"] == "float32"

    def test_missing_architectures(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        (d / "config.json").write_text('{"_name_or_path": "test"}')
        fp = compute_artifact_fingerprint(None, d)
        assert fp.get("parameter_class") is None


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for hashing utilities."""

    def test_permission_error_scan(self, tmp_path: Path):
        """scan_artifact_path should not crash on unreadable files."""
        d = tmp_path / "artifact"
        d.mkdir()
        f = d / "noperm.bin"
        f.write_bytes(b"secret")
        try:
            f.chmod(0o000)
            result = scan_artifact_path(d)
            assert isinstance(result, dict)
            assert result["file_count"] >= 0
        finally:
            f.chmod(0o644)

    def test_permission_error_file_hash(self, tmp_path: Path):
        """compute_file_hash should propagate permission errors."""
        f = tmp_path / "noperm.bin"
        f.write_bytes(b"secret")
        try:
            f.chmod(0o000)
            with pytest.raises((PermissionError, OSError)):
                compute_file_hash(f)
        finally:
            f.chmod(0o644)

    def test_broken_symlink_config_hash(self, tmp_path: Path):
        result = compute_config_hash(tmp_path / "broken")
        assert result is None

    def test_unicode_paths(self, tmp_path: Path):
        d = tmp_path / "artifact_\u00e9\u00e9"
        d.mkdir()
        (d / "config.json").write_text('{"_name_or_path": "test"}')
        (d / "\u0444\u0430\u0439\u043b.bin").write_bytes(b"data")
        result = scan_artifact_path(d)
        assert result["file_count"] == 2
        fp = compute_artifact_fingerprint(None, d)
        assert fp["config_file_hash"] is not None

    def test_pathlib_path_type(self, tmp_path: Path):
        """All functions should accept Path objects."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"test")
        assert isinstance(compute_file_hash(f), str)
        assert isinstance(compute_config_hash(f), str)
        assert compute_config_hash(tmp_path / "nope") is None

    def test_deeply_nested_artifact(self, tmp_path: Path):
        base = tmp_path / "artifact"
        deep = base
        for i in range(10):
            deep = deep / f"level_{i}"
        deep.mkdir(parents=True)
        (deep / "leaf.bin").write_bytes(b"deep data")
        result = scan_artifact_path(base)
        assert result["file_count"] == 1
        # Verify the path uses actual levels, not the ellipsis shorthand
        assert "level_0/level_1/level_2/level_3/level_4/level_5/level_6/level_7/level_8/level_9/leaf.bin" in str(result["files"])

    def test_broken_symlink_in_scan(self, tmp_path: Path):
        d = tmp_path / "artifact"
        d.mkdir()
        link = d / "broken_link.bin"
        link.symlink_to(tmp_path / "nonexistent")
        result = scan_artifact_path(d)
        # Broken symlink should be skipped via OSError handling
        assert result["file_count"] == 0 or "broken_link.bin" not in result["files"]
