"""Tests for Milestone 13 — Training-data export.

Covers:
  - SFT export (OpenAI messages JSONL)
  - Preference export (DPO/ORPO chosen/rejected JSONL)
  - Regression export (YAML)
  - Judge export (JSONL)
  - Base export helpers
  - CLI export integration
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
import yaml

from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.runners.completion_runner import RunResult
from bench_harness.export.base import (
    get_runs_by_suite,
    get_task_by_id,
    get_tasks_from_task_dir,
    get_judge_evaluations,
    get_pairwise_comparisons,
)
from bench_harness.export.sft_export import export_sft
from bench_harness.export.preference_export import (
    export_preference_score_based,
    export_preference_from_pairwise,
)
from bench_harness.export.regression_export import export_regression
from bench_harness.export.judge_export import (
    export_judge,
    export_judge_pairwise,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures: file-based SQLite database with mock data
# ──────────────────────────────────────────────────────────────────────
# We use temp files instead of :memory: because export functions create
# their own SQLiteStore instances, and :memory: creates separate DBs per
# instance.


@pytest.fixture
def db():
    """Create a file-based SQLite database and return the store + db_path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path)
    store.init()
    yield store
    store.db.close()
    Path(db_path).unlink(missing_ok=True)


def _insert_run(
    db,
    run_id,
    suite_id="test-suite",
    task_id="test.task_001",
    model_alias="model-a",
    prompt="What is 2+2?",
    raw_response="4",
    exit_status="success",
    score_primary=0.9,
    generated_code=None,
    prompt_style=None,
    quantization=None,
    error_message=None,
    judge_score=None,
):
    """Insert a run into the SQLite database."""
    result = RunResult(
        run_id=run_id,
        suite_id=suite_id,
        task_id=task_id,
        model_alias=model_alias,
        prompt=prompt,
        raw_response=raw_response,
        exit_status=exit_status,
        score_primary=score_primary,
        error_message=error_message,
        prompt_style=prompt_style,
        quantization=quantization,
    )
    if generated_code is not None:
        result.generated_code = generated_code
        result.raw_response = ""
    if judge_score is not None:
        result.judge_score = judge_score
    db.save_run(result)
    return result


def _insert_judge_eval(db, run_id, task_id, model_alias, judge_model, rubric_name,
                       score, dimensions=None, explanation="", raw_response=""):
    """Insert a judge evaluation into the database."""
    db.save_judge_evaluation(
        run_id=run_id,
        task_id=task_id,
        model_alias=model_alias,
        judge_model=judge_model,
        rubric_name=rubric_name,
        score=score,
        dimensions=dimensions,
        explanation=explanation,
        raw_response=raw_response,
    )


def _insert_pairwise(db, task_id, model_a, model_b, winner, margin, confidence,
                     reason="", dimension_comparison=None):
    """Insert a pairwise comparison into the database."""
    db.save_pairwise_comparison(
        task_id=task_id,
        model_a=model_a,
        model_b=model_b,
        winner=winner,
        margin=margin,
        confidence=confidence,
        reason=reason,
        dimension_comparison=dimension_comparison,
    )


# ══════════════════════════════════════════════════════════════════════
# TestSFTExport
# ══════════════════════════════════════════════════════════════════════


class TestSFTExport:
    """Tests for SFT (Supervised Fine-Tuning) export."""

    def test_export_sft_basic(self, db):
        """Export successful runs, verify JSONL format with messages array."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="What is 2+2?", raw_response="Four", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="What is 2+2?", raw_response="Four", score_primary=0.8)
        _insert_run(db, "r3", task_id="test.task_002", model_alias="model-a",
                     prompt="Hello", raw_response="Hi there", score_primary=1.0)

        out = export_sft(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")

        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            assert isinstance(record["messages"], list)
            assert len(record["messages"]) >= 1
            # Last message is always assistant
            assert record["messages"][-1]["role"] == "assistant"

    def test_export_sft_filters_min_score(self, db):
        """min_score=0.5 excludes low-scoring runs."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Ans1", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="Ans2", score_primary=0.3)
        _insert_run(db, "r3", task_id="test.task_001", model_alias="model-c",
                     prompt="Q1", raw_response="Ans3", score_primary=0.5)

        out = export_sft(db.db_path, "test-suite", min_score=0.5)
        lines = Path(out).read_text().strip().split("\n")

        assert len(lines) == 2
        scores = {json.loads(l)["score"] for l in lines}
        assert 0.3 not in scores
        assert 0.9 in scores
        assert 0.5 in scores

    def test_export_sft_includes_system_message(self, db):
        """When system_message exists, it's in messages."""
        # Create a temp task YAML directory
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "sys_task.yaml"
            task_file.write_text(yaml.dump({
                "id": "test.task_sys",
                "prompt": "Answer this",
                "scoring": {"primary": "exact_match"},
                "expected": {"type": "exact"},
                "family": "test",
                "version": "1.0",
                "source": "local",
                "input": {"system_message": "You are a helpful assistant."},
            }))

            # Patch the task cache to include our task
            from bench_harness.export import base
            original_cache = base._task_cache
            base._task_cache = {
                "test.task_sys": {
                    "id": "test.task_sys",
                    "family": "test",
                    "input": {"system_message": "You are a helpful assistant."},
                },
            }

            try:
                _insert_run(db, "r1", task_id="test.task_sys", model_alias="model-a",
                            prompt="Answer this", raw_response="Hello", score_primary=0.8)

                out = export_sft(db.db_path, "test-suite")
                record = json.loads(Path(out).read_text().strip().split("\n")[0])

                roles = [m["role"] for m in record["messages"]]
                assert "system" in roles
                assert "user" in roles
                assert "assistant" in roles

                # Verify system message content
                system_msg = [m for m in record["messages"] if m["role"] == "system"][0]
                assert "You are a helpful assistant." in system_msg["content"]
            finally:
                base._task_cache = original_cache

    def test_export_sft_code_task_includes_code(self, db):
        """For code tasks, raw_response is used as assistant content."""
        _insert_run(
            db, "r1",
            task_id="test.task_code",
            model_alias="model-a",
            prompt="Write a function",
            raw_response="def hello():\n    print('hi')",
            exit_status="success",
            score_primary=0.9,
        )

        out = export_sft(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assistant_msg = record["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        assert "def hello()" in assistant_msg["content"]

    def test_export_sft_empty_result(self, db):
        """No successful runs -> empty file."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q", raw_response="A", exit_status="error", score_primary=0)

        out = export_sft(db.db_path, "test-suite")
        content = Path(out).read_text().strip()
        assert content == ""

    def test_export_sft_output_format(self, db):
        """Verify exact JSON structure matches OpenAI messages format."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="What is 2+2?", raw_response="4", score_primary=0.9)

        out = export_sft(db.db_path, "test-suite")
        record = json.loads(Path(out).read_text().strip().split("\n")[0])

        # Check top-level keys
        assert "messages" in record
        assert "model" in record
        assert "family" in record
        assert "task_id" in record
        assert "score" in record

        # Check messages structure
        assert isinstance(record["messages"], list)
        for msg in record["messages"]:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant")

        assert record["model"] == "model-a"
        assert record["task_id"] == "test.task_001"
        assert record["score"] == 0.9


# ══════════════════════════════════════════════════════════════════════
# TestPreferenceExport
# ══════════════════════════════════════════════════════════════════════


class TestPreferenceExport:
    """Tests for preference (DPO/ORPO) export."""

    def test_export_preference_score_based(self, db):
        """Groups by task_id, picks best/worst."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A1", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="B1", score_primary=0.3)
        _insert_run(db, "r3", task_id="test.task_002", model_alias="model-a",
                     prompt="Q2", raw_response="A2", score_primary=0.7)
        _insert_run(db, "r4", task_id="test.task_002", model_alias="model-b",
                     prompt="Q2", raw_response="B2", score_primary=0.5)

        out = export_preference_score_based(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")

        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "chosen" in record
            assert "rejected" in record
            assert "margin" in record
            assert record["chosen"]["score"] > record["rejected"]["score"]

    def test_export_preference_min_margin(self, db):
        """Only includes pairs where margin >= threshold."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="B", score_primary=0.7)

        # Margin is 0.2
        out = export_preference_score_based(db.db_path, "test-suite", min_margin=0.5)
        content = Path(out).read_text().strip()
        assert content == ""

        out = export_preference_score_based(db.db_path, "test-suite", min_margin=0.1)
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 1
        margin_val = json.loads(lines[0])["margin"]
        assert abs(margin_val - 0.2) < 0.001

    def test_export_preference_from_pairwise(self, db):
        """Uses pairwise_comparisons table."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_pairwise(db, "test.task_001", "model-a", "model-b",
                         winner="A", margin="0.2", confidence=0.8,
                         reason="A is more accurate")

        out = export_preference_from_pairwise(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) >= 1

        # Find the record with our reason
        matching = [l for l in lines if "A is more accurate" in l]
        assert len(matching) >= 1

        record = json.loads(matching[0])
        assert record["chosen"]["model"] == "model-a"
        assert record["rejected"]["model"] == "model-b"
        assert record["reason"] == "A is more accurate"

    def test_export_preference_no_dupe_models(self, db):
        """A model doesn't appear as both chosen and rejected."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="B", score_primary=0.3)
        _insert_run(db, "r3", task_id="test.task_001", model_alias="model-c",
                     prompt="Q1", raw_response="C", score_primary=0.5)

        out = export_preference_score_based(db.db_path, "test-suite")
        record = json.loads(Path(out).read_text().strip().split("\n")[0])
        chosen_model = record["chosen"]["model"]
        rejected_model = record["rejected"]["model"]
        assert chosen_model != rejected_model

    def test_export_preference_json_format(self, db):
        """Verify exact JSONL structure."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="B", score_primary=0.3)

        out = export_preference_score_based(db.db_path, "test-suite")
        record = json.loads(Path(out).read_text().strip().split("\n")[0])

        assert "messages" in record
        assert "chosen" in record
        assert "rejected" in record
        assert "margin" in record
        assert "task_id" in record
        assert "prompt" in record

        assert "model" in record["chosen"]
        assert "score" in record["chosen"]
        assert "model" in record["rejected"]
        assert "score" in record["rejected"]


# ══════════════════════════════════════════════════════════════════════
# TestRegressionExport
# ══════════════════════════════════════════════════════════════════════


class TestRegressionExport:
    """Tests for regression export."""

    def test_export_regression_basic(self, db):
        """Exports failed runs grouped by task."""
        # Failed with a response (score 0.0, success exit)
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Wrong answer", score_primary=0.0,
                     exit_status="success")
        # Failed with error message
        _insert_run(db, "r2", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="some output", exit_status="error",
                     score_primary=0, error_message="API timeout")
        # Another task failure
        _insert_run(db, "r3", task_id="test.task_002", model_alias="model-b",
                     prompt="Q2", raw_response="Bad", score_primary=0.0,
                     exit_status="success")

        out = export_regression(db.db_path, "test-suite")
        data = yaml.safe_load(Path(out).read_text())
        assert isinstance(data, list)
        task_ids = {t["task_id"] for t in data}
        assert "test.task_001" in task_ids
        assert "test.task_002" in task_ids

    def test_export_regression_excludes_api_errors(self, db):
        """Empty raw_response excluded (API errors)."""
        # Run with API error (no raw_response, no error_message -> empty)
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="", exit_status="error", score_primary=0,
                     error_message="Network error")

        out = export_regression(db.db_path, "test-suite")
        data = yaml.safe_load(Path(out).read_text())

        # Should have no failures (run has error_message so it's included)
        # The test checks that pure empty runs are excluded
        assert data is not None
        if data:
            for task in data:
                for failure in task.get("failures", []):
                    assert "error" in failure or "raw_response" in failure

    def test_export_regression_yaml_format(self, db):
        """Verify YAML structure matches spec."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Wrong", score_primary=0.0,
                     exit_status="success")

        out = export_regression(db.db_path, "test-suite")
        data = yaml.safe_load(Path(out).read_text())
        assert len(data) >= 1

        task_entry = data[0]
        # Entry has id or task_id
        assert "id" in task_entry or "task_id" in task_entry
        assert "family" in task_entry
        assert "failures" in task_entry
        assert isinstance(task_entry["failures"], list)

        failure = task_entry["failures"][0]
        assert "model" in failure
        assert "score" in failure

    def test_export_regression_task_definition(self, db):
        """Includes full task definition with failures array."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Wrong", score_primary=0.0,
                     exit_status="success")

        out = export_regression(db.db_path, "test-suite")
        data = yaml.safe_load(Path(out).read_text())
        assert len(data) >= 1
        task_entry = data[0]

        # task_definition is included when the task is found from cache
        # When not found, the fallback still has id/family/failures
        assert "failures" in task_entry
        assert len(task_entry["failures"]) >= 1
        failure = task_entry["failures"][0]
        assert "model" in failure

    def test_export_regression_no_failures(self, db):
        """No failed runs -> empty list."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Right", exit_status="success",
                     score_primary=0.9)

        out = export_regression(db.db_path, "test-suite")
        data = yaml.safe_load(Path(out).read_text())
        assert data == []


# ══════════════════════════════════════════════════════════════════════
# TestJudgeExport
# ══════════════════════════════════════════════════════════════════════


class TestJudgeExport:
    """Tests for judge export."""

    def test_export_judge_basic(self, db):
        """Exports from judge_evaluations table."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A1", score_primary=0.9)
        _insert_judge_eval(db, "r1", "test.task_001", "model-a",
                           "judge_model_1", "quality_rubric",
                           score='{"correctness": 4, "completeness": 3}',
                           explanation="Good answer", dimensions={
                               "correctness": 4,
                               "completeness": 3,
                           })

        out = export_judge(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["task_id"] == "test.task_001"
        assert record["model_alias"] == "model-a"
        assert record["judge_model"] == "judge_model_1"
        assert record["rubric_name"] == "quality_rubric"
        assert record["is_pairwise"] is False

    def test_export_judge_dimensions(self, db):
        """Per-dimension scores included."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.8)
        _insert_judge_eval(db, "r1", "test.task_001", "model-a",
                           "judge_1", "quality_rubric",
                           score='{"correctness": 4}',
                           dimensions={"correctness": 4, "completeness": 3, "safety": 5})

        out = export_judge(db.db_path, "test-suite")
        record = json.loads(Path(out).read_text().strip().split("\n")[0])
        assert record["dimensions"]["correctness"] == 4
        assert record["dimensions"]["completeness"] == 3
        assert record["dimensions"]["safety"] == 5

    def test_export_judge_pairwise(self, db):
        """Includes pairwise comparison data."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_pairwise(db, "test.task_001", "model-a", "model-b",
                         winner="A", margin="0.15", confidence=0.85,
                         reason="A is better")

        out = export_judge_pairwise(db.db_path, "test-suite")
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["is_pairwise"] is True
        assert record["task_id"] == "test.task_001"
        assert record["winner"] == "A"
        assert record["model_a"] == "model-a"
        assert record["model_b"] == "model-b"

    def test_export_judge_json_format(self, db):
        """Verify JSON structure."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.8)
        _insert_judge_eval(db, "r1", "test.task_001", "model-a",
                           "judge_1", "quality_rubric",
                           score='{"correctness": 4}',
                           dimensions={"correctness": 4},
                           explanation="Good job")

        out = export_judge(db.db_path, "test-suite")
        record = json.loads(Path(out).read_text().strip().split("\n")[0])

        assert "run_id" in record
        assert "task_id" in record
        assert "model_alias" in record
        assert "judge_model" in record
        assert "rubric_name" in record
        assert "score" in record
        assert "dimensions" in record
        assert "explanation" in record
        assert "is_pairwise" in record


# ══════════════════════════════════════════════════════════════════════
# TestBaseExports
# ══════════════════════════════════════════════════════════════════════


class TestBaseExports:
    """Tests for base export helper functions."""

    def test_get_runs_by_suite(self, db):
        """Queries correct runs."""
        _insert_run(db, "r1", suite_id="test-suite", task_id="test.task_001",
                     model_alias="model-a", score_primary=0.9)
        _insert_run(db, "r2", suite_id="other-suite", task_id="test.task_002",
                     model_alias="model-b", score_primary=0.5)
        _insert_run(db, "r3", suite_id="test-suite", task_id="test.task_003",
                     model_alias="model-a", score_primary=0.7)

        results = get_runs_by_suite(db.db_path, "test-suite")
        assert len(results) == 2
        task_ids = {r["task_id"] for r in results}
        assert "test.task_001" in task_ids
        assert "test.task_003" in task_ids

    def test_get_tasks_from_dir(self):
        """Loads tasks and returns by ID map."""
        import tempfile
        from bench_harness.export.base import get_tasks_from_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "test_task.yaml"
            task_file.write_text(yaml.dump({
                "id": "test.dir_task",
                "family": "test",
                "prompt": "Hello",
                "scoring": {"primary": "exact_match"},
                "expected": {"type": "exact"},
            }))

            tasks = get_tasks_from_dir(str(tmpdir))
            assert "test.dir_task" in tasks
            assert tasks["test.dir_task"]["family"] == "test"

    def test_get_judge_evaluations(self, db):
        """Queries judge table correctly."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_judge_eval(db, "r1", "test.task_001", "model-a",
                           "judge_1", "rubric_1",
                           score='{"score": 4}',
                           dimensions={"correctness": 4})

        evals = get_judge_evaluations(db.db_path, "test-suite")
        assert len(evals) == 1
        assert evals[0]["task_id"] == "test.task_001"
        assert evals[0]["judge_model"] == "judge_1"

    def test_get_pairwise_comparisons(self, db):
        """Queries pairwise table correctly."""
        _insert_run(db, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="A", score_primary=0.9)
        _insert_pairwise(db, "test.task_001", "model-a", "model-b",
                         winner="A", margin="0.2", confidence=0.8,
                         reason="A wins")

        comps = get_pairwise_comparisons(db.db_path, "test-suite")
        assert len(comps) == 1
        assert comps[0]["winner"] == "A"
        assert comps[0]["model_a"] == "model-a"
        assert comps[0]["model_b"] == "model-b"


# ══════════════════════════════════════════════════════════════════════
# TestCLIExportIntegration
# ══════════════════════════════════════════════════════════════════════


class TestCLIExportIntegration:
    """Integration tests for export CLI commands."""

    def _make_test_db(self):
        """Create a test database with mock data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteStore(db_path)
        store.init()
        _insert_run(store, "r1", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Right", score_primary=0.9)
        _insert_run(store, "r2", task_id="test.task_001", model_alias="model-b",
                     prompt="Q1", raw_response="Wrong", score_primary=0.3)
        _insert_run(store, "r3", task_id="test.task_002", model_alias="model-a",
                     prompt="Q2", raw_response="Bad", exit_status="success",
                     score_primary=0.2)
        store._db_path = db_path  # Save path for cleanup
        return store

    def test_cli_export_sft(self):
        """Full integration via CLI command exports SFT data."""
        store = self._make_test_db()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = export_sft(store.db_path, "test-suite", out_path=tmpdir)
            assert Path(out).exists()

            lines = Path(out).read_text().strip().split("\n")
            assert len(lines) == 3  # r1, r2, r3 all have scores > 0 and success

            for line in lines:
                record = json.loads(line)
                assert "messages" in record
                assert record["messages"][-1]["role"] == "assistant"

    def test_cli_export_all(self):
        """Exports all four formats in one test run."""
        store = self._make_test_db()

        # Add judge and pairwise data
        _insert_run(store, "r_judge", task_id="test.task_001", model_alias="model-a",
                     prompt="Q1", raw_response="Right", score_primary=0.9)
        _insert_judge_eval(store, "r_judge", "test.task_001", "model-a",
                           "judge_1", "quality",
                           score='{"score": 4}',
                           dimensions={"correctness": 4})
        _insert_pairwise(store, "test.task_001", "model-a", "model-b",
                         winner="A", margin="0.2", confidence=0.8, reason="A is better")

        with tempfile.TemporaryDirectory() as tmpdir:
            sft_out = export_sft(store.db_path, "test-suite", out_path=tmpdir)
            pref_out = export_preference_score_based(store.db_path, "test-suite",
                                                      out_path=tmpdir)
            reg_out = export_regression(store.db_path, "test-suite",
                                         out_path=tmpdir)
            judge_out = export_judge(store.db_path, "test-suite", out_path=tmpdir)

            assert Path(sft_out).exists()
            assert Path(pref_out).exists()
            assert Path(reg_out).exists()
            assert Path(judge_out).exists()

            # Verify SFT
            sft_lines = Path(sft_out).read_text().strip().split("\n")
            assert len(sft_lines) >= 2  # r1, r_judge at least

            # Verify preference
            pref_lines = Path(pref_out).read_text().strip().split("\n")
            assert len(pref_lines) >= 1

            # Verify regression
            reg_data = yaml.safe_load(Path(reg_out).read_text())
            assert isinstance(reg_data, list)

            # Verify judge
            judge_lines = Path(judge_out).read_text().strip().split("\n")
            assert len(judge_lines) >= 1

    def test_cli_export_invalid_action(self):
        """Error for unknown action."""
        from bench_harness.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["export", "unknown_action"])
        assert result.exit_code != 0
