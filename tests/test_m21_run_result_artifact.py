"""Tests for M21 — RunResult artifact fields and fingerprint embedding.

Covers the artifact_durable and artifact_warnings fields on RunResult.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bench_harness.schemas.run_result import RunResult


class TestRunResultArtifactFields:
    def test_artifact_durable_defaults_to_none(self):
        """artifact_durable defaults to None."""
        result = RunResult(run_id="r1", run_spec_ref="s1", project="p1")
        assert result.artifact_durable is None

    def test_artifact_warnings_defaults_to_empty(self):
        """artifact_warnings defaults to empty list."""
        result = RunResult(run_id="r1", run_spec_ref="s1", project="p1")
        assert result.artifact_warnings == []

    def test_artifact_durable_can_be_set(self):
        """artifact_durable can be explicitly set to True/False."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_durable=True,
        )
        assert result.artifact_durable is True

    def test_artifact_warnings_can_be_set(self):
        """artifact_warnings can contain warning strings."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_warnings=["path is temporary", "non-durable storage"],
        )
        assert len(result.artifact_warnings) == 2
        assert "path is temporary" in result.artifact_warnings

    def test_fingerprint_and_artifact_fields_together(self):
        """All artifact-related fields coexist correctly."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_fingerprint={"model_id": "test-model", "dtype": "float16"},
            artifact_durable=False,
            artifact_warnings=["model not found in HF cache"],
        )
        assert result.artifact_fingerprint["model_id"] == "test-model"
        assert result.artifact_durable is False
        assert "model not found in HF cache" in result.artifact_warnings

    def test_write_to_directory_includes_artifact_fields(self, tmp_path):
        """write_to_directory() persists artifact fields in JSON."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_fingerprint={"model_id": "m1"},
            artifact_durable=True,
            artifact_warnings=["warning-1"],
        )
        run_dir = tmp_path / "runs" / "r1"
        run_dir.mkdir(parents=True)
        result.write_to_directory(run_dir)

        # Read back the saved JSON
        saved = json.loads((run_dir / "run_result.json").read_text())
        assert saved["artifact_fingerprint"]["model_id"] == "m1"
        assert saved["artifact_durable"] is True
        assert saved["artifact_warnings"] == ["warning-1"]

    def test_artifact_fields_json_serializable(self):
        """All artifact fields are JSON-serializable."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_fingerprint={
                "model_id": "test",
                "detected_dtype": "float16",
                "parameter_class": "small",
            },
            artifact_durable=None,
            artifact_warnings=[],
        )
        data = result.model_dump(mode='python')
        json.dumps(data)  # Should not raise

    def test_fingerprint_empty_by_default(self):
        """artifact_fingerprint is empty dict by default."""
        result = RunResult(run_id="r1", run_spec_ref="s1", project="p1")
        assert result.artifact_fingerprint == {}

    def test_artifact_durable_false(self):
        """artifact_durable can be explicitly False."""
        result = RunResult(run_id="r1", run_spec_ref="s1", project="p1", artifact_durable=False)
        assert result.artifact_durable is False

    def test_artifact_warnings_list_type(self):
        """artifact_warnings must be a list of strings."""
        result = RunResult(
            run_id="r1", run_spec_ref="s1", project="p1",
            artifact_warnings=["warn1", "warn2", "warn3"],
        )
        assert all(isinstance(w, str) for w in result.artifact_warnings)
