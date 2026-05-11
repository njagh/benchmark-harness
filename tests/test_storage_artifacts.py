"""Tests for M22 — storage/artifacts.py (save_run_artifact, save_judge_artifact)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bench_harness.storage.artifacts import save_run_artifact, save_judge_artifact
from bench_harness.storage.config import StorageConfig


def _make_run_result(**overrides):
    """Create a minimal mock RunResult for testing."""
    result = MagicMock()
    result.run_id = "test-run-001"
    result.suite_id = "smoke"
    result.task_id = "coding.task_001"
    result.model_alias = "gpt-4o"
    result.model_backend = "openai"
    result.prompt = "Write a function to add two numbers."
    result.raw_response = "def add(a, b): return a + b"
    result.prompt_tokens = 100
    result.completion_tokens = 50
    result.total_tokens = 150
    result.ttft_ms = 10.0
    result.prefill_ms = 5.0
    result.decode_ms = 25.0
    result.total_wall_ms = 35.0
    result.exit_status = "success"
    result.error_message = None
    result.created_at = "2025-01-01T00:00:00Z"
    result.generated_code = "def add(a, b): return a + b"
    result.code_status = "pass"
    result.tests_passed = 5
    result.tests_failed = 0
    result.tests_total = 5
    result.test_output = "All tests passed"
    result.exit_code = 0
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class TestSaveRunArtifact:
    def test_save_with_config(self, tmp_path):
        """save_run_artifact with StorageConfig writes to results_runs."""
        config = StorageConfig(root=tmp_path / "storage")
        result = _make_run_result()
        artifact_path = save_run_artifact(result, config=config)
        assert artifact_path.exists()
        assert artifact_path.name == "runs.jsonl"

        lines = artifact_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["run_id"] == "test-run-001"
        assert record["model_alias"] == "gpt-4o"

    def test_save_with_out_dir(self, tmp_path):
        """save_run_artifact with out_dir writes to that directory."""
        result = _make_run_result()
        artifact_path = save_run_artifact(result, out_dir=str(tmp_path / "custom"))
        assert artifact_path.exists()
        assert artifact_path.name == "runs.jsonl"

    def test_save_with_default_dir(self, tmp_path, monkeypatch):
        """save_run_artifact with no config or out_dir defaults to 'runs'."""
        import os
        orig_cwd = os.getcwd()
        try:
            monkeypatch.chdir(str(tmp_path))
            result = _make_run_result()
            artifact_path = save_run_artifact(result)
            assert artifact_path.exists()
            assert artifact_path.name == "runs.jsonl"
        finally:
            os.chdir(orig_cwd)
            runs_dir = tmp_path / "runs"
            if runs_dir.exists():
                import shutil
                shutil.rmtree(runs_dir)

    def test_save_includes_optional_fields(self, tmp_path):
        """save_run_artifact includes generated_code, code_status, tests fields."""
        config = StorageConfig(root=tmp_path / "storage")
        result = _make_run_result()
        save_run_artifact(result, config=config)
        artifact_path = config.results_runs / Path("runs.jsonl")
        # Find the file
        import glob
        jsonl_files = list(tmp_path.glob("**/runs.jsonl"))
        assert len(jsonl_files) > 0
        record = json.loads(jsonl_files[0].read_text().strip().split("\n")[-1])
        assert record["generated_code"] == "def add(a, b): return a + b"
        assert record["code_status"] == "pass"
        assert record["tests_passed"] == 5
        assert record["tests_failed"] == 0
        assert record["tests_total"] == 5
        assert record["test_output"] == "All tests passed"
        assert record["exit_code"] == 0

    def test_save_with_none_fields(self, tmp_path):
        """save_run_artifact omits None optional fields."""
        result = _make_run_result(
            generated_code=None,
            code_status=None,
            tests_passed=None,
            tests_failed=None,
            tests_total=None,
            test_output=None,
            exit_code=None,
        )
        config = StorageConfig(root=tmp_path / "storage")
        save_run_artifact(result, config=config)
        jsonl_files = list(tmp_path.glob("**/runs.jsonl"))
        record = json.loads(jsonl_files[0].read_text().strip().split("\n")[-1])
        assert "generated_code" not in record
        assert "code_status" not in record
        assert "tests_passed" not in record


class TestSaveJudgeArtifact:
    def test_save_all_judge_files(self, tmp_path):
        """save_judge_artifact creates raw, parsed, and prompt files."""
        config = StorageConfig(root=tmp_path / "storage")
        paths = save_judge_artifact(
            run_id="judge-001",
            config=config,
            raw_response='{"score": 0.9}',
            parsed_scores={"quality": 0.9, "completeness": 0.8},
            rubric_name="code_quality",
            judge_model="gpt-4o-judge",
            prompt="Evaluate this code",
        )
        assert "raw_response" in paths
        assert "parsed_scores" in paths
        assert "prompt" in paths
        assert paths["raw_response"].exists()
        assert paths["parsed_scores"].exists()
        assert paths["prompt"].exists()

    def test_save_raw_response_content(self, tmp_path):
        """save_judge_artifact raw file has correct content."""
        config = StorageConfig(root=tmp_path / "storage")
        paths = save_judge_artifact(
            run_id="j1",
            config=config,
            raw_response='{"quality": 1.0}',
            parsed_scores={"quality": 1.0},
        )
        content = paths["raw_response"].read_text()
        assert '"quality": 1.0' in content

    def test_save_parsed_scores_content(self, tmp_path):
        """save_judge_artifact parsed scores file has correct structure."""
        config = StorageConfig(root=tmp_path / "storage")
        paths = save_judge_artifact(
            run_id="j2",
            config=config,
            parsed_scores={"style": 0.7, "clarity": 0.9},
            rubric_name="writing",
            judge_model="claude-judge",
        )
        data = json.loads(paths["parsed_scores"].read_text())
        assert data["run_id"] == "j2"
        assert data["rubric_name"] == "writing"
        assert data["judge_model"] == "claude-judge"
        assert data["scores"] == {"style": 0.7, "clarity": 0.9}

    def test_save_prompt_content(self, tmp_path):
        """save_judge_artifact prompt file has correct content."""
        config = StorageConfig(root=tmp_path / "storage")
        paths = save_judge_artifact(
            run_id="j3",
            config=config,
            prompt="Score the following: Hello world",
        )
        assert paths["prompt"].read_text() == "Score the following: Hello world"

    def test_save_with_out_dir(self, tmp_path):
        """save_judge_artifact with out_dir writes to specified directory."""
        paths = save_judge_artifact(
            run_id="j4",
            out_dir=str(tmp_path / "judge_output"),
            parsed_scores={},
        )
        assert paths["parsed_scores"].parent == tmp_path / "judge_output"

    def test_save_missing_raw_response(self, tmp_path):
        """save_judge_artifact creates raw file path even when raw_response is None."""
        config = StorageConfig(root=tmp_path / "storage")
        paths = save_judge_artifact(
            run_id="j5",
            config=config,
            raw_response=None,
            parsed_scores={},
        )
        # raw_response path exists as a Path object even if file was not written
        assert paths["raw_response"] is not None
        # The file was NOT written when raw_response is None
        assert not paths["raw_response"].exists()
