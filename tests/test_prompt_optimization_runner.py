"""Tests for prompt optimization runner module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from bench_harness.prompt_optimization.analysis import PromptAnalysis, analyze_style_data
from bench_harness.prompt_optimization.proposals import TemplateProposal, TemplateRegistry
from bench_harness.prompt_optimization.runner import PromptOptimizationRunner


# ── PromptOptimizationRunner init tests ──────────────────────────────


class TestPromptOptimizationRunnerInit:
    def test_init_default(self):
        runner = PromptOptimizationRunner()
        assert runner._results == []
        assert runner.base_runner is None

    def test_init_multiple(self):
        runner1 = PromptOptimizationRunner()
        runner2 = PromptOptimizationRunner()
        assert runner1._results is not runner2._results


# ── analyze method tests ─────────────────────────────────────────────


class TestRunnerAnalyze:
    def test_analyze_uses_benchmark_db(self, tmp_path):
        runner = PromptOptimizationRunner()
        db_path = str(tmp_path / "benchmark.db")
        mock_db = MagicMock()
        mock_db.get_runs.return_value = [
            {"run_id": "r1", "task_id": "test.coding.t1", "prompt_style": "plain", "score_primary": 0.8, "suite_id": "suite-a"},
        ]
        with patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db):
            result = runner.analyze(db_path, suite_id="suite-a")
        assert isinstance(result, PromptAnalysis)
        assert result.total_style_runs == 1

    def test_analyze_empty_db(self, tmp_path):
        runner = PromptOptimizationRunner()
        result = analyze_style_data([])
        assert result.total_style_runs == 0

    def test_analyze_passes_suite_id(self, tmp_path):
        runner = PromptOptimizationRunner()
        result = analyze_style_data([], suite_id="my-suite")
        assert result.total_style_runs == 0

    def test_analyze_passes_min_runs(self, tmp_path):
        runner = PromptOptimizationRunner()
        result = analyze_style_data([], min_runs_per_style=5)
        assert result.total_style_runs == 0


# ── run_candidates metadata tests ────────────────────────────────────


class TestRunCandidatesMetadata:
    def test_run_candidates_no_baselines_defaults_to_plain(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_candidate(TemplateProposal(name="candidate-1", baseline="plain"))
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = []
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert result == []

    def test_run_candidates_no_tasks(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_baseline("plain")
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = []
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert result == []

    def test_run_candidates_with_existing_data(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_candidate(TemplateProposal(
            name="candidate-1",
            baseline="plain",
            instructions="Test candidate",
        ))
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = [
            {"run_id": "existing-1", "task_id": "test.task_001", "prompt_style": "plain", "score_primary": 0.8},
            {"run_id": "existing-2", "task_id": "test.task_002", "prompt_style": "plain", "score_primary": 0.9},
        ]
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[{"id": "test.task_001"}]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert len(result) == 1
        assert result[0]["name"] == "candidate-1"
        assert result[0]["baseline"] == "plain"
        assert result[0]["score"] == 0.0
        assert abs(result[0]["baseline_score"] - 0.85) < 0.01

    def test_run_candidates_candidate_with_data(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_candidate(TemplateProposal(
            name="candidate-1",
            baseline="plain",
            instructions="Test candidate",
        ))
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = [
            {"run_id": "c1-1", "task_id": "test.task_001", "prompt_style": "candidate-1", "score_primary": 0.95},
            {"run_id": "c1-2", "task_id": "test.task_002", "prompt_style": "candidate-1", "score_primary": 0.9},
            {"run_id": "p1-1", "task_id": "test.task_001", "prompt_style": "plain", "score_primary": 0.8},
        ]
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[{"id": "test.task_001"}]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert len(result) == 1
        assert abs(result[0]["score"] - 0.925) < 0.01
        assert abs(result[0]["baseline_score"] - 0.8) < 0.01
        assert abs(result[0]["score_delta"] - 0.125) < 0.01

    def test_run_candidates_status_analyzed(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_candidate(TemplateProposal(name="candidate-1", baseline="plain"))
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = [
            {"run_id": "p1-1", "task_id": "test.task_001", "prompt_style": "plain", "score_primary": 0.8},
        ]
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[{"id": "test.task_001"}]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert len(result) == 1
        # candidate-1 has no runs, baseline "plain" has 1 run -> status is "no_data"
        assert result[0]["status"] in ("analyzed", "no_data")

    def test_run_candidates_status_no_data(self, tmp_path):
        runner = PromptOptimizationRunner()
        registry = TemplateRegistry()
        registry.add_candidate(TemplateProposal(name="candidate-1", baseline="plain"))
        db_path = str(tmp_path / "benchmark.db")
        (tmp_path / "tasks" / "smoke").mkdir(parents=True)
        mock_db = MagicMock()
        mock_db.get_runs.return_value = [
            {"run_id": "p1-1", "task_id": "test.task_001", "prompt_style": "plain", "score_primary": 0.8},
        ]
        with (
            patch("bench_harness.storage.sqlite.BenchmarkDB", return_value=mock_db),
            patch("bench_harness.tasks.loaders.load_tasks", return_value=[{"id": "test.task_001"}]),
        ):
            result = runner.run_candidates(
                registry=registry,
                db_path=db_path,
                model_aliases=["test-model"],
                suite_id="suite-a",
            )
        assert len(result) == 1
        assert result[0]["status"] in ("analyzed", "no_data")
        assert result[0]["run_count"] == 0


# ── _compute_candidate_results tests ─────────────────────────────────


class TestComputeCandidateResults:
    def test_compute_candidate_results_basic(self):
        runner = PromptOptimizationRunner()
        run_results = [
            {"prompt_style": "candidate-1", "score_primary": 0.9},
            {"prompt_style": "candidate-1", "score_primary": 0.8},
            {"prompt_style": "plain", "score_primary": 0.7},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        result = runner._compute_candidate_results(run_results, ["plain"], candidates)
        assert len(result) == 2
        c1 = [r for r in result if r["name"] == "candidate-1"][0]
        assert abs(c1["score"] - 0.85) < 0.01
        assert abs(c1["score_delta"] - 0.15) < 0.01
        assert c1["status"] == "complete"

    def test_compute_candidate_results_baseline_only(self):
        runner = PromptOptimizationRunner()
        run_results = [
            {"prompt_style": "plain", "score_primary": 0.7},
            {"prompt_style": "plain", "score_primary": 0.8},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        result = runner._compute_candidate_results(run_results, ["plain"], candidates)
        baselines = [r for r in result if r["status"] == "baseline"]
        assert len(baselines) == 1
        assert baselines[0]["name"] == "plain"
        assert baselines[0]["score"] == 0.75

    def test_compute_candidate_results_no_scores(self):
        runner = PromptOptimizationRunner()
        run_results = [
            {"prompt_style": "candidate-1", "score_primary": None},
            {"prompt_style": "plain", "score_primary": None},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        result = runner._compute_candidate_results(run_results, ["plain"], candidates)
        c1 = [r for r in result if r["name"] == "candidate-1"][0]
        assert c1["score"] == 0.0
        assert c1["status"] == "complete"

    def test_compute_candidate_results_multiple_baselines(self):
        runner = PromptOptimizationRunner()
        run_results = [
            {"prompt_style": "candidate-1", "score_primary": 0.9},
            {"prompt_style": "plain", "score_primary": 0.8},
            {"prompt_style": "repl", "score_primary": 0.75},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        result = runner._compute_candidate_results(run_results, ["plain", "repl"], candidates)
        names = {r["name"] for r in result}
        assert "plain" in names
        assert "repl" in names


# ── _save_results tests ──────────────────────────────────────────────


class TestSaveResults:
    def test_save_results_creates_files(self, tmp_path):
        runner = PromptOptimizationRunner()
        output_dir = str(tmp_path / "output")
        run_results = [
            {"prompt_style": "candidate-1", "score_primary": 0.9, "run_id": "r1"},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        runner._save_results(output_dir, run_results, candidates)
        assert (tmp_path / "output" / "optimization_runs.json").exists()
        assert (tmp_path / "output" / "optimization_summary.json").exists()

    def test_save_results_writes_valid_json(self, tmp_path):
        runner = PromptOptimizationRunner()
        output_dir = str(tmp_path / "output")
        run_results = [
            {"prompt_style": "candidate-1", "score_primary": 0.9, "run_id": "r1"},
        ]
        candidates = [TemplateProposal(name="candidate-1", baseline="plain")]
        runner._save_results(output_dir, run_results, candidates)
        runs_data = json.loads((tmp_path / "output" / "optimization_runs.json").read_text())
        assert isinstance(runs_data, list)
        assert len(runs_data) == 1
        summary_data = json.loads((tmp_path / "output" / "optimization_summary.json").read_text())
        assert isinstance(summary_data, list)

    def test_save_results_creates_directory(self, tmp_path):
        runner = PromptOptimizationRunner()
        output_dir = str(tmp_path / "nested" / "output")
        run_results = []
        candidates = [TemplateProposal(name="c1", baseline="plain")]
        runner._save_results(output_dir, run_results, candidates)
        assert (tmp_path / "nested" / "output").exists()


# ── generate_report tests ────────────────────────────────────────────


class TestGenerateReport:
    def test_report_empty(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        report = runner.generate_report(analysis, [])
        assert "## Prompt Optimization Report" in report
        assert "Total runs analyzed: 0" in report

    def test_report_with_analysis(self, sample_style_runs):
        runner = PromptOptimizationRunner()
        analysis = analyze_style_data(sample_style_runs)
        report = runner.generate_report(analysis, [])
        assert "Total runs analyzed: 6" in report
        assert "Best style overall: **repl**" in report
        assert "Styles found: architect, plain, repl" in report

    def test_report_with_candidates(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "repl-terse",
                "score": 0.85,
                "baseline": "repl",
                "baseline_score": 0.8,
                "score_delta": 0.05,
                "run_count": 10,
                "status": "complete",
                "instructions": "Test candidate",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Candidate Template Results" in report
        assert "repl-terse" in report
        assert "0.850" in report

    def test_report_with_analyzed_candidates(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "analyzed-candidate",
                "score": 0.75,
                "baseline": "plain",
                "baseline_score": 0.8,
                "score_delta": -0.05,
                "run_count": 5,
                "status": "analyzed",
                "instructions": "Analyzed from data",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Analyzed from Existing Data" in report
        assert "analyzed-candidate" in report

    def test_report_with_no_data_candidates(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "no-data-candidate",
                "score": 0.0,
                "baseline": "plain",
                "score_delta": 0.0,
                "run_count": 0,
                "status": "no_data",
                "instructions": "No data",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Candidates with No Data" in report
        assert "no-data-candidate" in report

    def test_report_with_baseline_entries(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "plain",
                "score": 0.8,
                "run_count": 10,
                "status": "baseline",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Baseline Styles" in report
        assert "plain" in report

    def test_report_with_insufficient_data(self, sample_style_runs):
        runner = PromptOptimizationRunner()
        analysis = analyze_style_data(sample_style_runs, min_runs_per_style=100)
        candidate_results = []
        report = runner.generate_report(analysis, candidate_results)
        assert "insufficient data" in report.lower()

    def test_report_with_recommended_candidate(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "great-candidate",
                "score": 0.95,
                "baseline": "plain",
                "baseline_score": 0.8,
                "score_delta": 0.15,
                "run_count": 10,
                "status": "complete",
                "instructions": "Great improvement",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Recommendations:" in report
        assert "great-candidate" in report

    def test_report_no_recommendations_below_threshold(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "small-candidate",
                "score": 0.81,
                "baseline": "plain",
                "baseline_score": 0.8,
                "score_delta": 0.01,
                "run_count": 10,
                "status": "complete",
                "instructions": "Small delta",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "No candidates exceeded the 0.05 improvement threshold" in report

    def test_report_family_rankings_table(self, sample_style_runs):
        runner = PromptOptimizationRunner()
        analysis = analyze_style_data(sample_style_runs)
        report = runner.generate_report(analysis, [])
        assert "Best Styles Per Task Family" in report
        assert "| Task Family | Best Style | Avg Score | Margin |" in report

    def test_report_score_variance_table(self, sample_style_runs):
        runner = PromptOptimizationRunner()
        analysis = analyze_style_data(sample_style_runs)
        report = runner.generate_report(analysis, [])
        assert "Score Variance by Style" in report
        assert "| Style | Variance |" in report

    def test_report_positive_delta_format(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "positive-candidate",
                "score": 0.95,
                "baseline": "plain",
                "baseline_score": 0.8,
                "score_delta": 0.15,
                "run_count": 5,
                "status": "complete",
                "instructions": "Test",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "+0.150" in report

    def test_report_negative_delta_format(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "negative-candidate",
                "score": 0.7,
                "baseline": "plain",
                "baseline_score": 0.8,
                "score_delta": -0.1,
                "run_count": 5,
                "status": "complete",
                "instructions": "Test",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "-0.100" in report

    def test_report_full_run_results(self):
        runner = PromptOptimizationRunner()
        analysis = PromptAnalysis()
        candidate_results = [
            {
                "name": "candidate-a",
                "score": 0.85,
                "baseline": "plain",
                "baseline_score": 0.75,
                "score_delta": 0.1,
                "run_count": 8,
                "status": "complete",
                "instructions": "Test A",
            },
            {
                "name": "candidate-b",
                "score": 0.9,
                "baseline": "repl",
                "baseline_score": 0.8,
                "score_delta": 0.1,
                "run_count": 6,
                "status": "complete",
                "instructions": "Test B",
            },
        ]
        report = runner.generate_report(analysis, candidate_results)
        assert "Live Run Results" in report
        assert "| Candidate | Score | Baseline | Delta | Runs |" in report
