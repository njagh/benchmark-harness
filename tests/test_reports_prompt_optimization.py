"""Tests for Milestone 20 — Prompt Optimization Report Module.

Covers `reports/prompt_optimization.py`: generate_optimization_report()
and generate_optimization_summary().
"""

from __future__ import annotations

import json

import pytest

from bench_harness.prompt_optimization.analysis import PromptAnalysis
from bench_harness.reports.prompt_optimization import (
    generate_optimization_report,
    generate_optimization_summary,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_analysis(
    family_rankings=None,
    best_style_overall="plain",
    best_style_by_family=None,
    style_variances=None,
    insufficient_data=None,
    all_styles=None,
    total_style_runs=10,
) -> PromptAnalysis:
    return PromptAnalysis(
        family_rankings=family_rankings or {},
        best_style_overall=best_style_overall,
        best_style_by_family=best_style_by_family or {},
        style_variances=style_variances or {},
        insufficient_data=insufficient_data or [],
        all_styles=all_styles or [],
        total_style_runs=total_style_runs,
    )


def _make_candidate(
    name="cand-v1",
    baseline="plain",
    score=0.85,
    baseline_score=0.75,
    run_count=5,
    status="complete",
    instructions="",
):
    delta = score - baseline_score if baseline_score is not None else None
    return {
        "name": name,
        "task_family": "coding",
        "baseline": baseline,
        "instructions": instructions,
        "score": score,
        "baseline_score": baseline_score,
        "score_delta": delta,
        "run_count": run_count,
        "status": status,
    }


# ── generate_optimization_report ─────────────────────────────────────


class TestGenerateOptimizationReport:
    def test_empty_report(self):
        """Empty analysis and no candidates → minimal report."""
        analysis = _make_analysis()
        report = generate_optimization_report(analysis, [])
        assert "## Prompt Optimization Report" in report
        assert "**Total runs analyzed:** 10" in report

    def test_with_family_rankings(self):
        """Report includes family rankings table."""
        analysis = _make_analysis(
            family_rankings={
                "coding": [("repl", 0.85, 0.1)],
                "debugging": [("architect", 0.75, 0.05)],
            },
            all_styles=["plain", "repl", "architect"],
        )
        report = generate_optimization_report(analysis, [])
        assert "Best Styles Per Task Family" in report
        assert "| coding | repl | 0.850 | +0.100 |" in report
        assert "| debugging | architect | 0.750 | +0.050 |" in report

    def test_with_score_variance(self):
        """Report includes score variance table."""
        analysis = _make_analysis(
            style_variances={"plain": 0.02, "repl": 0.15, "architect": 0.08},
            all_styles=["plain", "repl", "architect"],
        )
        report = generate_optimization_report(analysis, [])
        assert "Score Variance by Style" in report
        assert "| repl | 0.1500 |" in report
        assert "| plain | 0.0200 |" in report

    def test_with_insufficient_data(self):
        """Report includes insufficient data warning."""
        analysis = _make_analysis(
            insufficient_data=["coding", "debugging"],
            all_styles=["plain", "repl"],
        )
        report = generate_optimization_report(analysis, [])
        assert "Insufficient data" in report
        assert "coding" in report
        assert "debugging" in report

    def test_with_live_run_results(self):
        """Report includes live run results table."""
        candidates = [
            _make_candidate(name="cand-v1", score=0.9, baseline_score=0.75, run_count=10, status="complete", instructions="Test candidate"),
            _make_candidate(name="cand-v2", score=0.7, baseline_score=0.75, run_count=8, status="complete", instructions="Below baseline"),
        ]
        analysis = _make_analysis(all_styles=["plain", "cand-v1", "cand-v2"])
        report = generate_optimization_report(analysis, candidates)
        assert "Live Run Results" in report
        assert "| cand-v1 | 0.900 | 0.750" in report
        assert "| cand-v2 | 0.700 | 0.750" in report

    def test_with_analyzed_results(self):
        """Report includes analyzed-from-existing-data table."""
        candidates = [
            _make_candidate(name="cand-v1", score=0.8, baseline_score=0.7, status="analyzed"),
        ]
        report = generate_optimization_report(_make_analysis(), candidates)
        assert "Analyzed from Existing Data" in report

    def test_with_no_data_candidates(self):
        """Report lists candidates with no data."""
        candidates = [
            _make_candidate(name="new-style", score=0.0, run_count=0, status="no_data"),
        ]
        report = generate_optimization_report(_make_analysis(), candidates)
        assert "Candidates with No Data" in report
        assert "`new-style`" in report

    def test_with_baseline_entries(self):
        """Report includes baseline styles section."""
        candidates = [
            _make_candidate(name="cand-v1", score=0.85, baseline_score=0.75, status="complete"),
            _make_candidate(name="plain", score=0.75, status="baseline"),
        ]
        report = generate_optimization_report(_make_analysis(), candidates)
        assert "Baseline Styles" in report
        assert "| plain | 0.750 |" in report

    def test_recommendations_above_threshold(self):
        """Candidates above 0.05 threshold get recommendations."""
        candidates = [
            _make_candidate(
                name="super-cand",
                score=0.9,
                baseline_score=0.7,
                instructions="Best style ever",
                status="complete",
            ),
        ]
        report = generate_optimization_report(_make_analysis(), candidates)
        assert "**Recommendations:**" in report
        assert "super-cand" in report
        assert "+0.200" in report

    def test_no_recommendations_below_threshold(self):
        """No recommendations when no candidates beat threshold."""
        candidates = [
            _make_candidate(score=0.78, baseline_score=0.75, status="complete"),
        ]
        report = generate_optimization_report(_make_analysis(), candidates)
        assert "No candidates exceeded the 0.05 improvement threshold" in report

    def test_with_best_style_by_family(self):
        """Report includes best style by family recommendations."""
        analysis = _make_analysis(
            best_style_by_family={"coding": "repl", "debugging": "architect"},
            all_styles=["plain", "repl", "architect"],
        )
        report = generate_optimization_report(analysis, [])
        assert "Recommended Style by Family" in report
        assert "**coding:**" in report
        assert "`repl`" in report
        assert "**debugging:**" in report
        assert "`architect`" in report

    def test_with_suite_id(self):
        """Report includes suite_id in header."""
        analysis = _make_analysis(all_styles=["plain"])
        report = generate_optimization_report(analysis, [], suite_id="coding_benchmark")
        assert "coding_benchmark" in report

    def test_report_order(self):
        """Sections appear in correct order: analysis, family, variance, candidates."""
        analysis = _make_analysis(
            family_rankings={"coding": [("repl", 0.85, 0.1)]},
            style_variances={"plain": 0.02},
            all_styles=["plain", "repl"],
        )
        candidates = [_make_candidate(status="complete")]
        report = generate_optimization_report(analysis, candidates)
        sections = [
            report.index("Analysis Summary"),
            report.index("Best Styles Per Task Family"),
            report.index("Score Variance by Style"),
            report.index("Candidate Template Results"),
        ]
        assert sections == sorted(sections)

    def test_empty_candidate_results(self):
        """Empty candidate list → no candidate sections."""
        analysis = _make_analysis(all_styles=["plain"])
        report = generate_optimization_report(analysis, [])
        assert "Candidate Template Results" not in report
        assert "Live Run Results" not in report
        assert "Analy zed from Existing Data" not in report


# ── generate_optimization_summary ────────────────────────────────────


class TestGenerateOptimizationSummary:
    def test_empty_summary(self):
        """Empty data → minimal summary with recommendation."""
        analysis = _make_analysis()
        summary = generate_optimization_summary(analysis, [])
        assert summary["total_runs"] == 10
        assert summary["candidates"] == []
        assert summary["recommendation"]["action"] == "insufficient_data"

    def test_summary_with_candidates(self):
        """Summary includes candidate details."""
        candidates = [
            _make_candidate(name="cand-v1", score=0.85, baseline_score=0.75, status="complete"),
        ]
        analysis = _make_analysis(all_styles=["plain", "cand-v1"])
        summary = generate_optimization_summary(analysis, candidates)
        assert len(summary["candidates"]) == 1
        c = summary["candidates"][0]
        assert c["name"] == "cand-v1"
        assert c["score"] == 0.85
        assert c["score_delta"] == pytest.approx(0.1)
        assert c["status"] == "complete"

    def test_recommendation_adopt(self):
        """Candidates above threshold → recommendation action is 'adopt'."""
        candidates = [
            _make_candidate(name="super", score=0.9, baseline_score=0.7, status="complete"),
        ]
        summary = generate_optimization_summary(_make_analysis(), candidates)
        assert summary["recommendation"]["action"] == "adopt"
        assert "super" in summary["recommendation"]["candidates"]

    def test_recommendation_no_change(self):
        """No candidates above threshold → action is 'no_change'."""
        candidates = [
            _make_candidate(score=0.78, baseline_score=0.75, status="complete"),
        ]
        summary = generate_optimization_summary(_make_analysis(), candidates)
        assert summary["recommendation"]["action"] == "no_change"

    def test_family_rankings_serialization(self):
        """Family rankings are serialized correctly."""
        analysis = _make_analysis(
            family_rankings={
                "coding": [("repl", 0.85, 0.1), ("plain", 0.75, 0.0)],
            },
        )
        summary = generate_optimization_summary(analysis, [])
        coding = summary["family_rankings"]["coding"]
        assert len(coding) == 2
        assert coding[0]["style"] == "repl"
        assert coding[0]["avg_score"] == 0.85
        assert coding[0]["margin"] == 0.1

    def test_summary_json_serializable(self):
        """Full summary is JSON-serializable."""
        analysis = _make_analysis(
            family_rankings={"coding": [("repl", 0.85, 0.1)]},
            style_variances={"plain": 0.02, "repl": 0.15},
            insufficient_data=["debugging"],
            all_styles=["plain", "repl"],
            best_style_overall="repl",
            best_style_by_family={"coding": "repl"},
            total_style_runs=42,
        )
        candidates = [
            _make_candidate(name="cand-v1", score=0.85, baseline_score=0.75, instructions="Test"),
            _make_candidate(name="plain", score=0.75, status="baseline"),
        ]
        summary = generate_optimization_summary(analysis, candidates)
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        assert parsed["total_runs"] == 42
        assert len(parsed["candidates"]) == 2

    def test_candidate_without_baseline_score(self):
        """Baseline-only entries have no baseline_score field."""
        candidates = [
            _make_candidate(name="plain", score=0.75, baseline_score=None, status="baseline"),
        ]
        summary = generate_optimization_summary(_make_analysis(), candidates)
        c = summary["candidates"][0]
        assert "baseline_score" not in c
        assert c["score"] == 0.75

    def test_candidate_with_instructions(self):
        """Candidates with instructions include them."""
        candidates = [
            _make_candidate(name="cand", score=0.8, baseline_score=0.7, instructions="Best style"),
        ]
        summary = generate_optimization_summary(_make_analysis(), candidates)
        assert summary["candidates"][0]["instructions"] == "Best style"
