"""Artifact registry and artifact management utilities.

Provides a JSONL-based registry for tracking model artifacts across
projects and experiments, plus helpers for managing artifacts in
various modes (external_path, managed_copy, managed_symlink).
"""

from __future__ import annotations

import json
import shutil
import datetime as dt
from pathlib import Path
from typing import Any

from bench_harness.storage.config import StorageConfig
from bench_harness.schemas.model_artifact import ModelArtifact, ArtifactKind, ArtifactMode
from bench_harness.storage.safety import detect_ephemeral_path


class ArtifactRegistry:
    """JSONL-based registry for model artifacts."""

    def __init__(self, config: StorageConfig):
        self.path = config.registry_root / "artifacts.jsonl"

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def register(self, artifact: ModelArtifact) -> None:
        """Append artifact record to registry."""
        self._ensure_file()
        entry = artifact.model_dump(mode='python')
        entry["registered_at"] = dt.datetime.now().isoformat()
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def lookup(self, artifact_id: str) -> ModelArtifact | None:
        """Look up artifact by ID."""
        self._ensure_file()
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("artifact_id") == artifact_id:
                    return ModelArtifact(**data)
        return None

    def list_all(self) -> list[ModelArtifact]:
        """List all registered artifacts."""
        self._ensure_file()
        artifacts = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                artifacts.append(ModelArtifact(**json.loads(line)))
        return artifacts

    def query(self, kind: str | None = None, project: str | None = None,
              quantization: str | None = None) -> list[ModelArtifact]:
        """Filter by kind, project, quantization method."""
        artifacts = self.list_all()
        results = []
        for a in artifacts:
            if kind and str(a.kind) != kind:
                continue
            if project and project not in (a.model_id or ""):
                continue
            if quantization and a.quantization != quantization:
                continue
            results.append(a)
        return results


def manage_artifact(artifact: ModelArtifact, config: StorageConfig) -> Path:
    """Handle artifact based on mode.

    Returns the effective path to use for benchmarking.
    """
    source = Path(artifact.source_path)

    if artifact.mode == ArtifactMode.external_path:
        is_ephemeral, warnings = detect_ephemeral_path(artifact.source_path)
        artifact.durable = not is_ephemeral
        artifact.artifact_warnings = warnings
        return source
    elif artifact.mode == ArtifactMode.managed_copy:
        return _copy_artifact(artifact, config)
    elif artifact.mode == ArtifactMode.managed_symlink:
        return _symlink_artifact(artifact, config)
    else:
        raise ValueError(f"Unknown artifact mode: {artifact.mode}")


def _copy_artifact(artifact: ModelArtifact, config: StorageConfig) -> Path:
    """Copy artifact files to the artifact store incrementally."""
    source = Path(artifact.source_path)
    dest = config.artifacts_root / "models" / artifact.artifact_id
    dest.mkdir(parents=True, exist_ok=True)

    copied_files = set()
    if (dest / ".copied_files").exists():
        copied_files = set((dest / ".copied_files").read_text().strip().split('\n'))

    if source.is_dir():
        for item in source.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                copied = _copy_directory_incremental(item, dest_item, copied_files)
            else:
                copied = _copy_file_incremental(item, dest_item, copied_files)

        # Write manifest of copied files
        manifest_path = dest / ".copied_files"
        manifest_path.write_text('\n'.join(sorted(copied_files)))
    else:
        _copy_file_incremental(source, dest, copied_files)

    return dest


def _copy_file_incremental(source: Path, dest: Path, copied_files: set[str]) -> set[str]:
    """Copy a single file if it is new or changed."""
    rel = str(source.relative_to(source.parent.parent))
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied_files.add(rel)
    else:
        try:
            if source.stat().st_mtime != dest.stat().st_mtime or source.stat().st_size != dest.stat().st_size:
                shutil.copy2(source, dest)
                copied_files.add(rel)
        except OSError:
            shutil.copy2(source, dest)
            copied_files.add(rel)
    return copied_files


def _copy_directory_incremental(source: Path, dest: Path, copied_files: set[str]) -> set[str]:
    """Copy a directory incrementally."""
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        for item in source.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                copied_files = _copy_directory_incremental(item, dest_item, copied_files)
            else:
                copied_files = _copy_file_incremental(item, dest_item, copied_files)
    return copied_files


def _symlink_artifact(artifact: ModelArtifact, config: StorageConfig) -> Path:
    """Create a symlink to the artifact in the artifact store."""
    source = Path(artifact.source_path)
    dest = config.artifacts_root / "models" / artifact.artifact_id
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if source.is_absolute():
        dest.symlink_to(source)
    else:
        dest.symlink_to(source.resolve())

    return dest
