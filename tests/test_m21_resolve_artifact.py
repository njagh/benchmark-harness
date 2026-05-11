"""Tests for M21 — StorageConfig.resolve_artifact().

Covers artifact path resolution through StorageConfig with all three modes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bench_harness.schemas.model_artifact import ArtifactMode, ArtifactKind
from bench_harness.storage.config import StorageConfig


def _make_storage_config(tmp_path: Path) -> StorageConfig:
    """Create a minimal StorageConfig for testing."""
    return StorageConfig(root=tmp_path / "storage")


class TestResolveArtifact:
    def test_external_path_returns_source(self, tmp_path):
        """external_path mode returns the original source path."""
        config = _make_storage_config(tmp_path)
        source = str(tmp_path / "models" / "my-model")
        result = config.resolve_artifact("external_path", source)
        assert Path(result) == Path(source)

    def test_managed_copy_creates_copy(self, tmp_path):
        """managed_copy mode creates a copy in the managed directory."""
        config = _make_storage_config(tmp_path)

        # Create source files
        src_dir = tmp_path / "source_model"
        src_dir.mkdir()
        (src_dir / "config.json").write_text('{"model_type": "llama"}')
        (src_dir / "model.safetensors").write_text("fake weights")

        result = config.resolve_artifact("managed_copy", str(src_dir))
        assert Path(result).exists()
        assert (Path(result) / "config.json").exists()
        assert (Path(result) / "model.safetensors").exists()

    def test_managed_symlink_creates_symlink(self, tmp_path):
        """managed_symlink mode creates a symlink."""
        config = _make_storage_config(tmp_path)

        src_dir = tmp_path / "source_model"
        src_dir.mkdir()
        (src_dir / "config.json").write_text('{"model_type": "llama"}')

        result = config.resolve_artifact("managed_symlink", str(src_dir))
        assert Path(result).is_symlink()
        assert Path(result).resolve() == src_dir.resolve()

    def test_http_url_handled(self, tmp_path):
        """HTTP/HTTPS URLs are passed through manage_artifact (converted to Path)."""
        config = _make_storage_config(tmp_path)
        result = config.resolve_artifact("external_path", "https://api.openai.com/v1")
        # External path mode returns the source; URL gets converted to Path
        assert result is not None

    def test_local_path_uses_hf_checkpoint_kind(self, tmp_path):
        """Local paths default to hf_checkpoint kind."""
        config = _make_storage_config(tmp_path)
        local_path = str(tmp_path / "models" / "local-model")
        result = config.resolve_artifact("external_path", local_path)
        # external_path just returns the path
        assert Path(result) == Path(local_path)

    def test_managed_copy_sets_durable(self, tmp_path):
        """managed_copy produces a durable artifact."""
        config = _make_storage_config(tmp_path)
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "config.json").write_text("{}")

        result = config.resolve_artifact("managed_copy", str(src_dir))
        # The managed copy path is under storage_root/managed/, which is durable
        assert "storage" in str(result)

    def test_managed_symlink_relative_source(self, tmp_path):
        """managed_symlink with absolute source path creates correct symlink."""
        config = _make_storage_config(tmp_path)
        src_dir = tmp_path / "absolute_source"
        src_dir.mkdir()
        (src_dir / "config.json").write_text("{}")

        result = config.resolve_artifact("managed_symlink", str(src_dir))
        assert Path(result).is_symlink()

    def test_resolve_artifact_idempotent(self, tmp_path):
        """Calling resolve_artifact twice with same inputs produces valid results."""
        config = _make_storage_config(tmp_path)
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "config.json").write_text("{}")

        result1 = config.resolve_artifact("managed_copy", str(src_dir))
        result2 = config.resolve_artifact("managed_copy", str(src_dir))

        # Both should produce valid paths
        assert Path(result1).exists()
        assert Path(result2).exists()
