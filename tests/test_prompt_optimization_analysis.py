"""Tests for prompt optimization analysis module."""

from __future__ import annotations

import pytest
from bench_harness.prompt_optimization.analysis import (
    PromptAnalysis,
    _avg,
    _stddev,
    _extract_family_from_task_id,
    best_style_family,
    detect_insufficient_data,
    analyze_style_data,
)


# ── Helper function tests ────────────────────────────────────────────


class TestAvg:
    def test_avg_normal(self):
        assert _avg([1.0, 2.0, 3.0]) == 2.0

    def test_avg_single(self):
        assert _avg([5.0]) == 5.0

    def test_avg_empty(self):
        assert _avg([]) == 0.0

    def test_avg_floats(self):
        assert abs(_avg([0.333, 0.667]) - 0.5) < 0.001


class TestStddev:
    def test_stddev_normal(self):
        result = _stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.0) < 0.01

    def test_stddev_single(self):
        assert _stddev([5.0]) == 0.0

    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_stddev_two_values(self):
        result = _stddev([10.0, 20.0])
        assert abs(result - 5.0) < 0.01


# ── _extract_family_from_task_id tests ──────────────────────────────

# Branch 1 (>=3 parts): skip = {"smoke", "local", "public", "synthetic"}
# Branch 2 (>=2 parts): skip = {"factual", "json", "python", "debug", "instruction"}
#   Also skipped: candidates with "_" in them, or matching /^[a-zA-Z]+_\d+$/
# Branch 3 (underscore split): skip = {"factual", "json", "python", "debug", "code", "instruction"}


class TestExtractFamily:
    def test_extract_family_three_parts(self):
        result = _extract_family_from_task_id("test.coding.task_001")
        assert result == "coding"

    def test_extract_family_three_parts_smoke_skipped_branch1_falls_through(self):
        result = _extract_family_from_task_id("test.smoke.task_001")
        assert result == "smoke"

    def test_extract_family_three_parts_local_skipped_branch1_falls_through(self):
        result = _extract_family_from_task_id("test.local.task_001")
        assert result == "local"

    def test_extract_family_three_parts_public_skipped_branch1_falls_through(self):
        result = _extract_family_from_task_id("test.public.task_001")
        assert result == "public"

    def test_extract_family_three_parts_synthetic_skipped_branch1_falls_through(self):
        result = _extract_family_from_task_id("test.synthetic.task_001")
        assert result == "synthetic"

    def test_extract_family_docker_compose(self):
        result = _extract_family_from_task_id("benchmark.docker_compose.task_001")
        assert result == "docker_compose"

    def test_extract_family_litellm_routing(self):
        result = _extract_family_from_task_id("benchmark.litellm_routing.task_001")
        assert result == "litellm_routing"

    def test_extract_family_two_parts(self):
        result = _extract_family_from_task_id("test.coding")
        assert result == "coding"

    def test_extract_family_two_parts_no_underscore(self):
        result = _extract_family_from_task_id("test.mytask")
        assert result == "mytask"

    def test_extract_family_underscore_suffix_underscore_split(self):
        result = _extract_family_from_task_id("test.factual_001")
        assert result == "001"

    def test_extract_family_two_parts_pattern_underscore_split(self):
        result = _extract_family_from_task_id("test.task_001")
        assert result == "001"

    def test_extract_family_branch2_factual_skipped(self):
        result = _extract_family_from_task_id("test.factual")
        assert result is None

    def test_extract_family_branch2_json_skipped(self):
        result = _extract_family_from_task_id("test.json")
        assert result is None

    def test_extract_family_branch2_python_skipped(self):
        result = _extract_family_from_task_id("test.python")
        assert result is None

    def test_extract_family_branch2_debug_skipped(self):
        result = _extract_family_from_task_id("test.debug")
        assert result is None

    def test_extract_family_branch2_instruction_skipped(self):
        result = _extract_family_from_task_id("test.instruction")
        assert result is None

    def test_extract_family_branch1_all_skips_fall_through(self):
        for skip_family in ("smoke", "local", "public", "synthetic"):
            result = _extract_family_from_task_id(f"test.{skip_family}.task_001")
            assert result == skip_family


# ── best_style_family tests ──────────────────────────────────────────


class TestBestStyleFamily:
    def test_best_style_basic(self, sample_style_runs):
        family_runs = [
            r for r in sample_style_runs
            if "task_001" in r["task_id"]
        ]
        best_style, best_score, margin = best_style_family(family_runs)
        assert best_style == "repl"
        assert abs(best_score - 0.9) < 0.01

    def test_best_style_reversed(self, sample_style_runs):
        family_runs = [
            r for r in sample_style_runs
            if "task_002" in r["task_id"]
        ]
        best_style, best_score, margin = best_style_family(family_runs)
        assert best_style == "repl"
        assert abs(margin - 0.15) < 0.01

    def test_best_style_empty_runs(self):
        best_style, best_score, margin = best_style_family([])
        assert best_style is None
        assert best_score == 0
        assert margin == 0

    def test_best_style_missing_score(self):
        runs = [
            {"prompt_style": "plain", "score_primary": None},
            {"prompt_style": "repl"},
        ]
        best_style, best_score, margin = best_style_family(runs)
        assert best_style is None

    def test_best_style_insufficient_runs(self, sample_style_runs):
        runs = [
            r for r in sample_style_runs
            if r["task_id"] == "test.coding.task_003"
        ]
        best_style, best_score, margin = best_style_family(runs, min_runs_per_style=2)
        assert best_style is None
        assert best_score == 0
        assert margin == 0

    def test_best_style_single_style(self):
        runs = [
            {"prompt_style": "plain", "score_primary": 0.8},
            {"prompt_style": "plain", "score_primary": 0.9},
        ]
        best_style, best_score, margin = best_style_family(runs)
        assert best_style == "plain"
        assert abs(best_score - 0.85) < 0.01
        assert margin == 0.0

    def test_best_style_handles_none_score(self):
        runs = [
            {"prompt_style": "plain", "score_primary": 0.8},
            {"prompt_style": "repl", "score_primary": None},
        ]
        best_style, best_score, margin = best_style_family(runs)
        assert best_style == "plain"
        assert abs(best_score - 0.8) < 0.01

    def test_best_style_tie_breaking(self):
        runs = [
            {"prompt_style": "a", "score_primary": 0.5},
            {"prompt_style": "b", "score_primary": 0.5},
        ]
        best_style, best_score, margin = best_style_family(runs)
        assert abs(best_score - 0.5) < 0.01
        assert best_style in ("a", "b")
        assert margin == 0.0


# ── detect_insufficient_data tests ───────────────────────────────────


class TestDetectInsufficientData:
    def test_sufficient_data(self, sample_style_runs):
        result = detect_insufficient_data(sample_style_runs, min_runs_per_style=1)
        assert "coding" not in result

    def test_insufficient_data_high_threshold(self, sample_style_runs):
        result = detect_insufficient_data(sample_style_runs, min_runs_per_style=10)
        assert "coding" in result

    def test_insufficient_data_partial(self):
        runs = [
            {"task_id": "test.family.t1", "prompt_style": "plain", "score_primary": 0.8},
            {"task_id": "test.family.t2", "prompt_style": "repl", "score_primary": 0.9},
        ]
        result = detect_insufficient_data(runs, min_runs_per_style=3)
        assert "family" in result

    def test_no_prompt_style(self):
        runs = [
            {"task_id": "test.family.t1"},
        ]
        result = detect_insufficient_data(runs)
        assert result == []

    def test_all_styles_sufficient(self, sample_style_runs):
        result = detect_insufficient_data(sample_style_runs, min_runs_per_style=2)
        assert "coding" in result  # architect only has 1 run


# ── analyze_style_data tests ─────────────────────────────────────────


class TestAnalyzeStyleData:
    def test_analyze_empty(self):
        result = analyze_style_data([])
        assert isinstance(result, PromptAnalysis)
        assert result.total_style_runs == 0
        assert result.all_styles == []
        assert result.best_style_overall == ""

    def test_analyze_basic(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        assert result.total_style_runs == 6
        assert "plain" in result.all_styles
        assert "repl" in result.all_styles
        assert "architect" in result.all_styles
        assert len(result.family_rankings) >= 1

    def test_analyze_suite_filter(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs, suite_id="suite-a")
        assert result.total_style_runs == 6

    def test_analyze_suite_filter_no_match(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs, suite_id="nonexistent")
        assert result.total_style_runs == 0
        assert result.all_styles == []

    def test_analyze_best_style_overall(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        assert result.best_style_overall == "repl"

    def test_analyze_best_style_by_family(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        assert "coding" in result.best_style_by_family
        assert result.best_style_by_family["coding"] == "repl"

    def test_analyze_style_variances(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        assert "plain" in result.style_variances
        assert "repl" in result.style_variances
        assert "architect" in result.style_variances

    def test_analyze_variances_non_negative(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        for style, var in result.style_variances.items():
            assert var >= 0

    def test_analyze_insufficient_data(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs, min_runs_per_style=10)
        assert "coding" in result.insufficient_data

    def test_analyze_minimal(self, minimal_runs):
        result = analyze_style_data(minimal_runs)
        assert result.total_style_runs == 1
        assert result.all_styles == ["plain"]
        assert result.best_style_overall == "plain"

    def test_analyze_family_rankings_structure(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        for family, rankings in result.family_rankings.items():
            for rank_entry in rankings:
                assert len(rank_entry) == 3
                style_name, avg_score, margin = rank_entry
                assert isinstance(style_name, str)
                assert isinstance(avg_score, float)
                assert isinstance(margin, float)

    def test_analyze_missing_fields_robust(self, runs_with_missing_fields):
        result = analyze_style_data(runs_with_missing_fields)
        assert result.total_style_runs == 3
        assert "plain" in result.all_styles

    def test_analyze_sorted_styles(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        assert result.all_styles == sorted(result.all_styles)

    def test_analyze_margin_calculation(self, sample_style_runs):
        result = analyze_style_data(sample_style_runs)
        for family, rankings in result.family_rankings.items():
            if len(rankings) >= 2:
                for rank_entry in rankings:
                    assert rank_entry[2] >= 0
