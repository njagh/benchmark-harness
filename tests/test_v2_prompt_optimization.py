"""Tests for Milestone 20 — _append_prompt_optimization in v2.py report.

Covers the integration between prompt_optimization data and the v2 report.
"""

from __future__ import annotations

import pytest

from bench_harness.reports.v2 import _append_prompt_optimization


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_model_stats(opt_runs=None, other_runs=None):
    """Create a model_stats dict with optional optimization and regular runs."""
    stats = {}
    if opt_runs:
        stats["opt-model"] = opt_runs
    if other_runs:
        stats["other-model"] = other_runs
    return stats


def _make_run(
    run_id="run-001",
    task_id="test.task_001",
    suite_id="suite-optimization-abc123",
    model_alias="test-model",
    prompt_style="cand-v1",
    score_primary=0.85,
    completion_tokens=100,
    total_wall_ms=200,
):
    return {
        "run_id": run_id,
        "task_id": task_id,
        "suite_id": suite_id,
        "model_alias": model_alias,
        "prompt_style": prompt_style,
        "score_primary": score_primary,
        "completion_tokens": completion_tokens,
        "total_wall_ms": total_wall_ms,
        "exit_status": "success",
    }


# ── Tests ────────────────────────────────────────────────────────────


class TestAppendPromptOptimization:
    def test_no_optimization_data(self):
        """No optimization runs → no section appended."""
        stats = _make_model_stats(
            other_runs=[_make_run(suite_id="regular-suite", prompt_style="plain", score_primary=0.8)]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert lines == []

    def test_optimization_runs_appended(self):
        """Optimization runs → section with candidate table appended."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.75, run_id="r1"),
                _make_run(prompt_style="cand-v1", score_primary=0.85, run_id="r2"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("## Prompt Optimization" in line for line in lines)
        assert any("cand-v1" in line for line in lines)

    def test_multiple_optimization_suites(self):
        """Multiple optimization suites are all analyzed together."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(suite_id="suite-optimization-aaa", prompt_style="plain", score_primary=0.7, run_id="r1"),
                _make_run(suite_id="suite-optimization-bbb", prompt_style="cand-v1", score_primary=0.8, run_id="r2"),
                _make_run(suite_id="suite-optimization-aaa", prompt_style="cand-v2", score_primary=0.6, run_id="r3"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("## Prompt Optimization" in line for line in lines)
        # Both candidates should appear
        assert any("cand-v1" in line for line in lines)
        assert any("cand-v2" in line for line in lines)

    def test_no_plans_baseline(self):
        """When plain baseline score is 0 (no plain runs), candidates still show."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="cand-v1", score_primary=0.85, run_id="r1"),
                _make_run(prompt_style="cand-v2", score_primary=0.9, run_id="r2"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("## Prompt Optimization" in line for line in lines)
        assert any("cand-v1" in line for line in lines)

    def test_no_candidate_styles(self):
        """Only baseline styles (all known styles) → no section."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.75, run_id="r1"),
                _make_run(prompt_style="repl", score_primary=0.8, run_id="r2"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        # Known styles only, no candidates → section should not appear
        assert lines == []

    def test_mixed_baselines_and_candidates(self):
        """Both baselines and candidates → candidates are shown."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.7, run_id="r1"),
                _make_run(prompt_style="repl", score_primary=0.75, run_id="r2"),
                _make_run(prompt_style="my-custom-style", score_primary=0.85, run_id="r3"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("## Prompt Optimization" in line for line in lines)
        # my-custom-style is a candidate (not in known_styles)
        assert any("my-custom-style" in line for line in lines)

    def test_recommendations_above_threshold(self):
        """Candidates above 0.05 threshold trigger recommendation."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.7, run_id="r1"),
                _make_run(prompt_style="super-style", score_primary=0.85, run_id="r2"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("super-style" in line for line in lines)
        # The rich report should include recommendation
        report_text = "\n".join(lines)
        assert "super-style" in report_text

    def test_no_runs(self):
        """Empty model_stats → no section."""
        lines: list[str] = []
        _append_prompt_optimization(lines, {})
        assert lines == []

    def test_optimization_runs_with_missing_fields(self):
        """Optimization runs with missing optional fields handled gracefully."""
        stats = _make_model_stats(
            opt_runs=[
                {"run_id": "r1", "suite_id": "suite-optimization-x", "score_primary": 0.8, "prompt_style": "cand-v1"},
                {"run_id": "r2", "suite_id": "suite-optimization-x", "prompt_style": "plain"},  # no score
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        assert any("## Prompt Optimization" in line for line in lines)

    def test_empty_optimization_runs(self):
        """Optimization runs but all scores are None."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="cand-v1", score_primary=None, run_id="r1"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        # No scored runs → no candidates → no section
        assert lines == []

    def test_with_non_optimization_runs_also_present(self):
        """Regular runs alongside optimization runs don't interfere."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.75, run_id="r1"),
                _make_run(prompt_style="cand-v1", score_primary=0.85, run_id="r2"),
            ],
            other_runs=[
                _make_run(suite_id="regular-suite", prompt_style="plain", score_primary=0.9, run_id="r10"),
                _make_run(suite_id="regular-suite", prompt_style="repl", score_primary=0.95, run_id="r11"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        # Only optimization data should be used for the section
        assert any("## Prompt Optimization" in line for line in lines)
        report_text = "\n".join(lines)
        # cand-v1 from optimization should be present
        assert "cand-v1" in report_text

    def test_report_structure_has_expected_sections(self):
        """Report contains all expected sections from the module."""
        stats = _make_model_stats(
            opt_runs=[
                _make_run(prompt_style="plain", score_primary=0.7, run_id="r1"),
                _make_run(prompt_style="repl", score_primary=0.75, run_id="r2"),
                _make_run(prompt_style="cand-v1", score_primary=0.85, run_id="r3"),
            ]
        )
        lines: list[str] = []
        _append_prompt_optimization(lines, stats)
        report_text = "\n".join(lines)
        assert "## Prompt Optimization Report" in report_text
        assert "Analysis Summary" in report_text
