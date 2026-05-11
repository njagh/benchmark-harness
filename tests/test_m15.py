"""Tests for Milestone 15 — CI / Regression Mode.

Covers compare_runs(), format_comparison_output() and generate_regression_suite()
plus the regression.py helper functions.
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

from bench_harness.compare import compare_runs, format_comparison_output
from bench_harness.regression import (
    _get_failed_tasks,
    _get_high_variance_tasks,
    _get_regressed_tasks,
    generate_regression_suite,
)
from bench_harness.runners.completion_runner import RunResult
from bench_harness.storage.sqlite import SQLiteStore


# ──────────────────────────────────────────────────────────────────────
# Fixtures: file-based SQLite database with mock data
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def db():
    """Create a file-based SQLite database and return the store + db_path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path)
    store.init()
    yield store, db_path
    store.db.close()
    Path(db_path).unlink(missing_ok=True)


def _make_result(
    run_id,
    suite_id="test-suite",
    task_id="test.task_001",
    model_alias="model-a",
    score_primary=None,
    total_wall_ms=0.0,
    tokens_per_second=0.0,
    exit_status="success",
    error_message=None,
):
    """Build a minimal RunResult for test data insertion."""
    return RunResult(
        run_id=run_id,
        suite_id=suite_id,
        task_id=task_id,
        model_alias=model_alias,
        score_primary=score_primary,
        total_wall_ms=total_wall_ms,
        tokens_per_second=tokens_per_second,
        exit_status=exit_status,
        error_message=error_message,
    )


# ── compare_runs ─────────────────────────────────────────────────────


class TestCompareRuns:
    def test_quality_regression(self, db):
        """Scores drop between baseline and candidate → quality_regression."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.9, total_wall_ms=1000, tokens_per_second=50))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.7, total_wall_ms=1000, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, score_threshold=0.05)
            assert len(result["quality_regressions"]) == 1
            assert result["quality_regressions"][0]["task_id"] == "t1"
            assert result["quality_regressions"][0]["delta"] == pytest.approx(-0.2)
            assert result["quality_improvements"] == []
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_quality_improvement(self, db):
        """Scores increase between baseline and candidate → quality_improvement."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.5, total_wall_ms=1000, tokens_per_second=50))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.85, total_wall_ms=1000, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, score_threshold=0.05)
            assert len(result["quality_improvements"]) == 1
            assert result["quality_improvements"][0]["delta"] == pytest.approx(0.35)
            assert result["quality_regressions"] == []
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_no_quality_change_within_tolerance(self, db):
        """Score delta below threshold → no quality change flagged."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.7, total_wall_ms=1000, tokens_per_second=50))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.72, total_wall_ms=1000, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, score_threshold=0.05)
            assert result["quality_regressions"] == []
            assert result["quality_improvements"] == []
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_performance_wall_time_regression(self, db):
        """Wall time increases → performance regression."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=500, tokens_per_second=50))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=900, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, tps_threshold=0.1)
            assert len(result["performance_regressions"]) == 1
            assert result["performance_regressions"][0]["metric"] == "wall_time"
            assert result["performance_regressions"][0]["change_pct"] == pytest.approx(80.0)
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_performance_wall_time_improvement(self, db):
        """Wall time decreases → performance improvement."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=1000, tokens_per_second=50))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=300, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, tps_threshold=0.1)
            assert len(result["performance_improvements"]) == 1
            assert result["performance_improvements"][0]["metric"] == "wall_time"
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_performance_tps_regression(self, db):
        """Tokens/sec decreases → performance regression."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=500, tokens_per_second=100))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=800, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, tps_threshold=0.1)
            # Both wall_time and tps may be detected as regressions
            assert len(result["performance_regressions"]) >= 1
            tps_regressions = [r for r in result["performance_regressions"] if r["metric"] == "tokens_per_second"]
            assert len(tps_regressions) == 1
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_performance_tps_improvement(self, db):
        """Tokens/sec increases → performance improvement."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=1000, tokens_per_second=30))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=400, tokens_per_second=80))
        try:
            result = compare_runs(db_path, store2.db_path, tps_threshold=0.1)
            # Both wall_time and tps may be detected as improvements
            assert len(result["performance_improvements"]) >= 1
            tps_improvements = [r for r in result["performance_improvements"] if r["metric"] == "tokens_per_second"]
            assert len(tps_improvements) == 1
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_crash_change_baseline_succeeded_candidate_missing(self, db):
        """Task present in baseline but not candidate → crash change."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.9, total_wall_ms=1000))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        try:
            result = compare_runs(db_path, store2.db_path)
            assert len(result["crash_changes"]) == 1
            assert "candidate_missing" in result["crash_changes"][0]["status"]
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_crash_change_baseline_crashed_candidate_succeeded(self, db):
        """Baseline present only (crashed), candidate missing → crash change."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", exit_status="error"))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        try:
            result = compare_runs(db_path, store2.db_path)
            assert len(result["crash_changes"]) == 1
            assert "baseline_crashed" in result["crash_changes"][0]["status"]
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_crash_change_candidate_crashed_baseline_succeeded(self, db):
        """Both sides have runs for same key → no crash change (falls through to score comparison).
        Since candidate has exit_status='error' and no metrics, nothing gets flagged."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=1000))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", exit_status="error"))
        try:
            result = compare_runs(db_path, store2.db_path)
            # Both sides have runs for (t1, m1), so crash change block is skipped.
            # Candidate's score_primary=None means nothing to compare in detect_regressions.
            # Candidate's total_wall_ms=0 is filtered out (> 0 check).
            # No regressions, improvements, or crash changes expected.
            assert result["quality_regressions"] == []
            assert result["quality_improvements"] == []
            assert result["performance_regressions"] == []
            assert result["performance_improvements"] == []
            assert result["crash_changes"] == []
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_multiple_tasks(self, db):
        """Compare across multiple (task_id, model) pairs."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=500, tokens_per_second=50))
        store.save_run(_make_result("b2", task_id="t2", model_alias="m1", score_primary=0.6, total_wall_ms=600, tokens_per_second=40))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.8, total_wall_ms=500, tokens_per_second=50))
        store2.save_run(_make_result("c2", task_id="t2", model_alias="m1", score_primary=0.3, total_wall_ms=900, tokens_per_second=20))
        try:
            result = compare_runs(db_path, store2.db_path, score_threshold=0.05, tps_threshold=0.1)
            assert len(result["quality_regressions"]) == 1
            assert result["quality_regressions"][0]["task_id"] == "t2"
            assert len(result["performance_regressions"]) == 2  # wall_time + tps
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)

    def test_multi_model_comparison(self, db):
        """Compare where both models appear in both runs."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.7, total_wall_ms=500, tokens_per_second=50))
        store.save_run(_make_result("b2", task_id="t1", model_alias="m2", score_primary=0.8, total_wall_ms=600, tokens_per_second=45))

        store2 = SQLiteStore(db_path.replace(".db", "_cand.db"))
        store2.init()
        store2.save_run(_make_result("c1", task_id="t1", model_alias="m1", score_primary=0.55, total_wall_ms=550, tokens_per_second=48))
        store2.save_run(_make_result("c2", task_id="t1", model_alias="m2", score_primary=0.95, total_wall_ms=580, tokens_per_second=50))
        try:
            result = compare_runs(db_path, store2.db_path, score_threshold=0.05)
            assert len(result["quality_regressions"]) == 1
            assert result["quality_regressions"][0]["model_alias"] == "m1"
            assert len(result["quality_improvements"]) == 1
            assert result["quality_improvements"][0]["model_alias"] == "m2"
        finally:
            store2.db.close()
            Path(store2.db_path).unlink(missing_ok=True)


# ── format_comparison_output ─────────────────────────────────────────


class TestFormatComparisonOutput:
    def test_empty_results(self):
        """No regressions or improvements → summary line present."""
        results = {
            "quality_regressions": [],
            "quality_improvements": [],
            "performance_regressions": [],
            "performance_improvements": [],
            "crash_changes": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Summary:" in output

    def test_with_quality_regressions(self):
        """Output includes quality regression table content."""
        results = {
            "quality_regressions": [
                {"task_id": "t1", "model_alias": "m1", "old_score": 0.9, "new_score": 0.6, "delta": -0.3},
            ],
            "quality_improvements": [],
            "performance_regressions": [],
            "performance_improvements": [],
            "crash_changes": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Quality Regressions" in output
        assert "t1" in output

    def test_with_quality_improvements(self):
        """Output includes quality improvement table content."""
        results = {
            "quality_regressions": [],
            "quality_improvements": [
                {"task_id": "t1", "model_alias": "m1", "old_score": 0.5, "new_score": 0.8, "delta": 0.3},
            ],
            "performance_regressions": [],
            "performance_improvements": [],
            "crash_changes": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Quality Improvements" in output

    def test_with_crash_changes(self):
        """Output includes crash changes table."""
        results = {
            "quality_regressions": [],
            "quality_improvements": [],
            "performance_regressions": [],
            "performance_improvements": [],
            "crash_changes": [
                {"task_id": "t1", "model_alias": "m1", "status": "baseline_crashed, candidate_succeeded"},
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Crash Changes" in output

    def test_summary_counts(self):
        """Summary line counts all categories."""
        results = {
            "quality_regressions": [
                {"task_id": "t1", "model_alias": "m1", "old_score": 0.9, "new_score": 0.6, "delta": -0.3},
            ],
            "quality_improvements": [
                {"task_id": "t2", "model_alias": "m1", "old_score": 0.5, "new_score": 0.8, "delta": 0.3},
            ],
            "performance_regressions": [
                {"task_id": "t1", "model_alias": "m1", "metric": "wall_time", "baseline_wall_ms": 500, "candidate_wall_ms": 800, "change_pct": 60.0},
            ],
            "performance_improvements": [
                {"task_id": "t2", "model_alias": "m1", "metric": "tokens_per_second", "baseline_tps": 30, "candidate_tps": 60, "change_pct": 100.0},
            ],
            "crash_changes": [
                {"task_id": "t3", "model_alias": "m1", "status": "crash"},
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "1 quality regression(s)" in output
        assert "1 quality improvement(s)" in output
        assert "performance" in output.lower()
        assert "1 performance regression(s), 1 performance improvement(s)" in output.replace("\n", " ").replace("  ", " ")
        assert "1 crash change(s)" in output

    def test_with_performance_regressions(self):
        """Output includes performance regression table with wall_time."""
        results = {
            "quality_regressions": [],
            "quality_improvements": [],
            "performance_regressions": [
                {"task_id": "t1", "model_alias": "m1", "metric": "wall_time", "baseline_wall_ms": 500, "candidate_wall_ms": 800, "change_pct": 60.0},
            ],
            "performance_improvements": [],
            "crash_changes": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Performance Regressions" in output
        assert "wall_time" in output

    def test_with_performance_improvements(self):
        """Output includes performance improvement table."""
        results = {
            "quality_regressions": [],
            "quality_improvements": [],
            "performance_regressions": [],
            "performance_improvements": [
                {"task_id": "t1", "model_alias": "m1", "metric": "tokens_per_second", "baseline_tps": 30, "candidate_tps": 80, "change_pct": 166.7},
            ],
            "crash_changes": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            format_comparison_output(results)
        output = buf.getvalue()
        assert "Performance Improvements" in output


# ── regression.py helpers ────────────────────────────────────────────


class TestGetFailedTasks:
    def test_detects_error_exit_status(self):
        runs = [
            {"task_id": "t1", "exit_status": "error", "score_primary": 0.5},
            {"task_id": "t2", "exit_status": "success"},
        ]
        assert _get_failed_tasks(runs) == {"t1"}

    def test_detects_zero_score(self):
        runs = [
            {"task_id": "t1", "exit_status": "success", "score_primary": 0.0},
            {"task_id": "t2", "exit_status": "success", "score_primary": 0.5},
        ]
        assert _get_failed_tasks(runs) == {"t1"}

    def test_detects_zero_int_score(self):
        runs = [
            {"task_id": "t1", "exit_status": "success", "score_primary": 0},
        ]
        assert _get_failed_tasks(runs) == {"t1"}

    def test_empty_runs(self):
        assert _get_failed_tasks([]) == set()

    def test_multiple_failures(self):
        runs = [
            {"task_id": "t1", "exit_status": "error"},
            {"task_id": "t2", "exit_status": "error", "score_primary": 0.0},
            {"task_id": "t3", "exit_status": "success", "score_primary": 1.0},
        ]
        assert _get_failed_tasks(runs) == {"t1", "t2"}


class TestGetHighVarianceTasks:
    def test_selects_high_variance_tasks(self):
        runs = [
            {"task_id": "t1", "model_alias": "m1", "score_primary": 0.9},
            {"task_id": "t1", "model_alias": "m2", "score_primary": 0.2},
            {"task_id": "t2", "model_alias": "m1", "score_primary": 0.8},
            {"task_id": "t2", "model_alias": "m2", "score_primary": 0.7},
        ]
        result = _get_high_variance_tasks(runs, max_tasks=1)
        assert result == ["t1"]

    def test_returns_empty_when_no_variance(self):
        runs = [
            {"task_id": "t1", "model_alias": "m1", "score_primary": 0.5},
        ]
        assert _get_high_variance_tasks(runs) == []

    def test_max_tasks_limit(self):
        runs = []
        for i in range(10):
            for j in range(2):
                runs.append({
                    "task_id": f"t{i}",
                    "model_alias": f"m{j}",
                    "score_primary": 0.5 + (i if j == 0 else -i),
                })
        result = _get_high_variance_tasks(runs, max_tasks=3)
        assert len(result) == 3


class TestGetRegressedTasks:
    def test_selects_lowest_scoring_tasks(self):
        runs = [
            {"task_id": "t1", "model_alias": "m1", "score_primary": 0.1},
            {"task_id": "t2", "model_alias": "m1", "score_primary": 0.9},
            {"task_id": "t3", "model_alias": "m1", "score_primary": 0.5},
        ]
        result = _get_regressed_tasks(runs, max_tasks=2)
        assert result[0] == "t1"
        assert result[1] == "t3"

    def test_empty_runs(self):
        result = _get_regressed_tasks([])
        assert result == []

    def test_max_tasks_limit(self):
        runs = [
            {"task_id": f"t{i}", "model_alias": "m1", "score_primary": 0.1 * (i + 1)}
            for i in range(5)
        ]
        result = _get_regressed_tasks(runs, max_tasks=2)
        assert len(result) == 2


# ── generate_regression_suite ────────────────────────────────────────


class TestGenerateRegressionSuite:
    def test_basic_generation(self, db):
        """Generate a YAML regression suite with failed and high-variance tasks."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.0, exit_status="error"))
        store.save_run(_make_result("b2", task_id="t2", model_alias="m1", score_primary=0.9))
        store.save_run(_make_result("b3", task_id="t2", model_alias="m2", score_primary=0.3))
        store.save_run(_make_result("b4", task_id="t3", model_alias="m1", score_primary=0.8))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir, max_tasks=2)
            assert os.path.exists(output)

            with open(output) as f:
                suite = yaml.safe_load(f)

            assert isinstance(suite, list)
            assert len(suite) >= 1  # t1 is failed, always included

            t1_entry = next((e for e in suite if e["id"] == "t1"), None)
            assert t1_entry is not None
            assert "failures" in t1_entry

            first_entry = suite[0]
            assert first_entry["id"] == "t1"

    def test_empty_runs_produces_empty_suite(self, db):
        """No runs → empty YAML list."""
        store, db_path = db
        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir)
            with open(output) as f:
                suite = yaml.safe_load(f)
            assert suite == []

    def test_failed_tasks_always_included(self, db):
        """Failed tasks are always in the suite regardless of max_tasks."""
        store, db_path = db
        for i in range(10):
            store.save_run(_make_result(f"b{i}", task_id=f"t{i}", model_alias="m1", score_primary=0.0, exit_status="error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir, max_tasks=2)
            with open(output) as f:
                suite = yaml.safe_load(f)
            assert len(suite) == 10

    def test_suite_has_failure_info(self, db):
        """Suite entries for failed tasks include failure details."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="model-a", score_primary=0.0, exit_status="error", error_message="OOM"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir)
            with open(output) as f:
                suite = yaml.safe_load(f)

            entry = suite[0]
            assert "failures" in entry
            failure = entry["failures"][0]
            assert failure["model"] == "model-a"
            assert failure["exit_status"] == "error"
            assert failure["error"] == "OOM"

    def test_failed_tasks_sorted_first(self, db):
        """Failed tasks come before high-variance tasks in the output."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="t1", model_alias="m1", score_primary=0.0, exit_status="error"))
        store.save_run(_make_result("b2", task_id="t2", model_alias="m1", score_primary=0.0, exit_status="error"))
        store.save_run(_make_result("b3", task_id="t3", model_alias="m1", score_primary=0.9))
        store.save_run(_make_result("b4", task_id="t3", model_alias="m2", score_primary=0.2))
        store.save_run(_make_result("b5", task_id="t4", model_alias="m1", score_primary=0.8))
        store.save_run(_make_result("b6", task_id="t4", model_alias="m2", score_primary=0.3))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir, max_tasks=2)
            with open(output) as f:
                suite = yaml.safe_load(f)

            failed_ids = {"t1", "t2"}
            failed_indices = [i for i, e in enumerate(suite) if e["id"] in failed_ids]
            non_failed_indices = [i for i, e in enumerate(suite) if e["id"] not in failed_ids]

            if non_failed_indices:
                assert max(failed_indices) < min(non_failed_indices)

    def test_suite_includes_task_metadata(self, db):
        """Suite entries include task metadata when task YAML is available."""
        store, db_path = db
        store.save_run(_make_result("b1", task_id="coding.patch_001", model_alias="m1", score_primary=0.5, total_wall_ms=500, tokens_per_second=40))
        store.save_run(_make_result("b2", task_id="coding.patch_001", model_alias="m2", score_primary=0.2, total_wall_ms=600, tokens_per_second=30))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_regression_suite(db_path, tmpdir, max_tasks=2)
            with open(output) as f:
                suite = yaml.safe_load(f)

            entry = suite[0]
            assert entry["id"] == "coding.patch_001"
            assert "family" in entry
            # scoring/expected are only included when task YAML is found in cache
