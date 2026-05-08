"""Artifact metadata hooks for project-specific enrichment.

Provides an ABC for hooks that can add project-specific metadata
to artifact manifests and run results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bench_harness.schemas.model_artifact import ModelArtifact
from bench_harness.schemas.run_result import RunResult


class ArtifactMetadataHook(ABC):
    """Hook to add project-specific metadata to artifact manifests."""

    @abstractmethod
    def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        """Enrich an artifact with project-specific metadata."""
        ...

    @abstractmethod
    def enrich_run_result(self, result: RunResult) -> RunResult:
        """Enrich a run result with project-specific metadata."""
        ...


class ModelOptMetadataHook(ArtifactMetadataHook):
    """Capture ModelOpt-specific metadata from quantization runs.

    Scans artifact for ModelOpt metadata files and fills in:
    - ModelOpt version
    - Quantization algorithm (AWQ, SqueezeLLM, etc.)
    - Calibration dataset name/path
    - Number of calibration samples
    - Export format
    - Base model ID
    - Quantized output path
    - Producing command
    - Producing git commit
    """

    def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        """Enrich artifact with ModelOpt-specific metadata."""
        source = artifact.source_path
        if not source:
            return artifact

        meta_path = None
        import os
        for root, dirs, files in os.walk(source):
            if 'modelopt_meta.json' in files:
                meta_path = os.path.join(root, 'modelopt_meta.json')
                break

        if meta_path:
            import json
            try:
                with open(meta_path) as f:
                    meta_data = json.load(f)
                artifact.producing_version = meta_data.get('modelopt_version', artifact.producing_version)
                artifact.quantization = meta_data.get('quantization_algorithm', artifact.quantization)
                artifact.file_list_summary = meta_data.get('calibration_dataset', artifact.file_list_summary)
                artifact.model_id = meta_data.get('base_model_id', artifact.model_id)
                artifact.producing_git_commit = meta_data.get('git_commit', artifact.producing_git_commit)
                artifact.backend_version = meta_data.get('export_format', artifact.backend_version)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        return artifact

    def enrich_run_result(self, result: RunResult) -> RunResult:
        """Enrich run result with ModelOpt-specific metadata."""
        if not result.artifact_fingerprint:
            return result

        # ModelOpt-specific fields are already captured in artifact
        result.artifact_fingerprint.setdefault('hook', 'modelopt')
        return result
