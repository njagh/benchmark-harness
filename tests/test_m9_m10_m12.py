"""Tests for Milestones 9, 10, and 12 — Long-context benchmarks, quantization comparison, and report v2.

M9 — Long-context benchmark suite (20 tests)
  - TestContextPacker: pack_small, pack_large, empty_files, exceeds_budget, add_distractors,
    apply_relevant_fact_placement_beginning, middle, end
  - TestContextSizeSweepRunner: runs_all_sizes, tags_results, error_handling
  - TestContextBudgetMapping: small/medium/large/xlarge budget values

M10 — Quantization comparison (15 tests)
  - TestQuantizationComparisonReport: summary_sections, quality_delta, best_by_family,
    sensitivity_analysis, speed_quality_frontier
  - TestQuantizationTracking: runresult_has_quantization, cli_passes_quantization
  - TestQuantizationTasks: tasks_load, tasks_have_scorers, suite_in_config

M12 — Report v2 (20 tests)
  - TestReportV2: executive_summary, model_comparison, best_by_family, failure_analysis,
    regression_detection, discriminating_tasks
  - TestReportHelpers: group_by_model, group_by_family, compute_score_variance,
    find_pareto_frontier, cluster_failures, detect_regressions
  - TestMarkdownIntegration: generate_report_v2_flag, sections_filter, generate_report_legacy
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import yaml

from bench_harness.runners.completion_runner import RunResult
from bench_harness.metrics.context_packer import (
    ContextPacker,
    CONTEXT_BUDGETS,
)
from bench_harness.runners.context_sweep_runner import ContextSizeSweepRunner
from bench_harness.reports.quantization_comparison import (
    generate_quantization_report,
    _avg,
    _score_values,
    _is_reference_quantization,
    _extract_family_from_task_id,
)
from bench_harness.reports.v2 import (
    generate_report_v2,
    _get_score_for_model,
    _get_speed_for_model,
    _get_context_for_model,
    _get_quantization_for_model,
    _format_delta,
    _format_safe_pct,
)
from bench_harness.reports.helpers import (
    group_by_model,
    group_by_family,
    group_by_quantization,
    compute_score_variance_by_task,
    find_pareto_frontier,
    cluster_failures,
    detect_regressions,
)


# ──────────────────────────────────────────────────────────────────────
# Helper: minimal run result dicts
# ──────────────────────────────────────────────────────────────────────

def _make_run_result(
    task_id: str = "test.task_001",
    model_alias: str = "model-a",
    score: float = 0.8,
    tps: float = 50.0,
    status: str = "success",
    quantization: str | None = None,
    context_tokens: str | None = None,
    prompt_tokens: int = 100,
    exit_code: int | None = None,
    error_message: str | None = None,
    prompt_style: str | None = None,
) -> dict[str, Any]:
    """Create a minimal run result dict for testing."""
    return {
        "run_id": f"run-{task_id}-{model_alias}",
        "suite_id": "test-suite",
        "task_id": task_id,
        "model_alias": model_alias,
        "score_primary": score,
        "tokens_per_second": tps,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + 50,
        "completion_tokens": 50,
        "total_wall_ms": 1000.0,
        "exit_status": status,
        "quantization": quantization,
        "context_tokens": context_tokens,
        "prompt_style": prompt_style,
    }


def _make_error_run_result(
    task_id: str = "test.task_001",
    model_alias: str = "model-a",
    error_message: str = "API timeout",
    context_tokens: str | None = None,
) -> dict[str, Any]:
    """Create an error run result dict for testing."""
    return {
        "run_id": f"run-{task_id}-{model_alias}-err",
        "suite_id": "test-suite",
        "task_id": task_id,
        "model_alias": model_alias,
        "exit_status": "error",
        "error_message": error_message,
        "quantization": None,
        "context_tokens": context_tokens,
    }


def _make_run_result_v2(
    task_id: str = "test.task_001",
    model_alias: str = "model-a",
    score: float = 0.8,
    tps: float = 50.0,
    status: str = "success",
    quantization: str | None = None,
    prompt_tokens: int = 100,
) -> dict[str, Any]:
    """Create a run result dict compatible with v2 report."""
    return {
        "run_id": f"run-{task_id}-{model_alias}",
        "suite_id": "test-suite",
        "task_id": task_id,
        "model_alias": model_alias,
        "score_primary": score,
        "tokens_per_second": tps,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + 50,
        "completion_tokens": 50,
        "total_wall_ms": 1000.0,
        "exit_status": status,
        "quantization": quantization,
        "context_tokens": None,
        "prompt_style": None,
    }


# ══════════════════════════════════════════════════════════════════════
# M9 — Long-context benchmark suite
# ══════════════════════════════════════════════════════════════════════

# ── TestContextPacker ─────────────────────────────────────────────────

class TestContextPacker:
    """Test ContextPacker packing, distractors, and fact placement."""

    def test_pack_small(self):
        """pack with small budget returns ~1024 tokens worth of files."""
        packer = ContextPacker()
        small_content = "The quick brown fox jumps over the lazy dog. " * 500
        files = [{"name": "f1.txt", "content": small_content}]
        result = packer.pack(files, target_budget="small")
        assert result != ""
        from bench_harness.tasks.prompt_templates import _estimate_tokens
        tokens = _estimate_tokens(result)
        # Small budget is 1024 tokens; header overhead adds ~10%
        assert tokens >= 600
        assert tokens <= 1500

    def test_pack_large(self):
        """pack with large budget includes more content than small."""
        packer = ContextPacker()
        big_content = "The quick brown fox jumps over the lazy dog. " * 500
        files = [{"name": "big.txt", "content": big_content}]
        small_result = packer.pack(files, target_budget="small")
        large_result = packer.pack(files, target_budget="large")
        from bench_harness.tasks.prompt_templates import _estimate_tokens
        assert _estimate_tokens(large_result) >= _estimate_tokens(small_result)
        # Large budget (16384) should include more than small (1024)
        assert _estimate_tokens(large_result) > 2 * _estimate_tokens(small_result)

    def test_pack_empty_files(self):
        """empty file list returns empty string."""
        packer = ContextPacker()
        result = packer.pack([], target_budget="large")
        assert result == ""

    def test_pack_exceeds_budget(self):
        """when files exceed budget, truncates properly."""
        packer = ContextPacker()
        huge_content = "word " * 5000  # ~5000 tokens
        files = [{"name": "huge.txt", "content": huge_content}]
        result = packer.pack(files, target_budget="small", max_budget=2048)
        assert result != ""
        from bench_harness.tasks.prompt_templates import _estimate_tokens
        tokens = _estimate_tokens(result)
        # Should be truncated to fit within budget
        assert tokens <= 2100  # small budget + some overhead

    def test_add_distractors(self):
        """adds distractor files before relevant files."""
        packer = ContextPacker()
        files = [
            {"name": "relevant.txt", "content": "The answer is 42."},
        ]
        extended = packer.add_distractors(files, num_distractors=3, target_budget="large")
        assert len(extended) == 4  # 3 distractors + 1 relevant
        # Distractors should come first
        assert "distractor_1" in extended[0]["name"]
        assert "distractor_3" in extended[2]["name"]
        # Relevant file should be last
        assert "relevant.txt" == extended[3]["name"]

    def test_apply_relevant_fact_placement_beginning(self):
        """moves relevant fact to beginning."""
        packer = ContextPacker()
        content = "## File: context.txt\n```\nSome context here.\n```\n\nSome extra text."
        fact = "## File: fact.txt\n```\nThe answer is 42.\n```"
        result = packer.apply_relevant_fact_placement(content, target_position="beginning", fact_text=fact)
        # Fact should appear first
        assert result.startswith("## File: fact.txt")
        assert "context.txt" in result
        assert result.index("fact.txt") < result.index("context.txt")

    def test_apply_relevant_fact_placement_middle(self):
        """moves relevant fact to middle."""
        packer = ContextPacker()
        content = "## File: before.txt\n```\nBefore content.\n```\n\n## File: after.txt\n```\nAfter content.\n```"
        fact = "## File: fact.txt\n```\nThe answer is 42.\n```"
        result = packer.apply_relevant_fact_placement(content, target_position="middle", fact_text=fact)
        # Fact should be roughly in the middle
        fact_idx = result.index("fact.txt")
        half = len(result) // 2
        # Allow some slack
        assert abs(fact_idx - half) < len(result) // 3

    def test_apply_relevant_fact_placement_end(self):
        """keeps relevant fact at end (appends if not found in content)."""
        packer = ContextPacker()
        content = "## File: context.txt\n```\nSome context here.\n```"
        fact = "## File: fact.txt\n```\nThe answer is 42.\n```"
        result = packer.apply_relevant_fact_placement(content, target_position="end", fact_text=fact)
        # Fact should appear last
        assert result.rstrip().endswith("```")
        assert "The answer is 42." in result
        assert result.index("fact.txt") > result.index("context.txt")

    def test_apply_relevant_fact_placement_empty(self):
        """empty content returns empty string."""
        packer = ContextPacker()
        result = packer.apply_relevant_fact_placement("", target_position="beginning")
        assert result == ""

    def test_extract_last_file_block(self):
        """extracts the last file block from mixed content."""
        packer = ContextPacker()
        content = (
            "## File: first.txt\n```\nFirst content\n```\n\n"
            "## File: second.txt\n```\nSecond content\n```\n\n"
            "## File: third.txt\n```\nThird content\n```\n"
        )
        last = packer._extract_last_file_block(content)
        assert "third.txt" in last


# ── TestContextBudgetMapping ──────────────────────────────────────────

class TestContextBudgetMapping:
    """Test CONTEXT_BUDGETS mapping values."""

    def test_get_context_budget_small(self):
        """returns 1024 for small."""
        assert CONTEXT_BUDGETS["small"] == 1024

    def test_get_context_budget_medium(self):
        """returns 4096 for medium."""
        assert CONTEXT_BUDGETS["medium"] == 4096

    def test_get_context_budget_large(self):
        """returns 16384 for large."""
        assert CONTEXT_BUDGETS["large"] == 16384

    def test_get_context_budget_xlarge(self):
        """returns 65536 for xlarge."""
        assert CONTEXT_BUDGETS["xlarge"] == 65536

    def test_unknown_budget_returns_none(self):
        """unknown bucket name returns None from get()."""
        assert CONTEXT_BUDGETS.get("nonexistent") is None


# ── TestContextSizeSweepRunner ────────────────────────────────────────

class TestContextSizeSweepRunner:
    """Test ContextSizeSweepRunner with mocked base runner."""

    async def _mock_runner_factory(self, exit_status="success"):
        mock_result = MagicMock()
        mock_result.exit_status = exit_status
        mock_result.context_tokens = None
        mock_result.estimated_prompt_tokens = None
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)
        return mock_runner, mock_result

    def test_sweep_initializes_sizes(self):
        """ContextSizeSweepRunner stores configured sizes."""
        mock_runner = MagicMock()
        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small", "large"])
        assert sweep.sizes == ["small", "large"]

    def test_sweep_default_sizes(self):
        """Default sizes includes all four context buckets."""
        mock_runner = MagicMock()
        sweep = ContextSizeSweepRunner(mock_runner)
        assert sweep.sizes == ["small", "medium", "large", "xlarge"]

    async def test_sweep_runs_all_sizes(self):
        """sweep runs with all context sizes producing one result per size."""
        mock_runner, mock_result = await self._mock_runner_factory()
        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small", "medium", "large"])
        task = {"id": "task_001", "prompt": "Test prompt", "input": {}}
        params = {"model_alias": "test-model"}

        results = await sweep.run_sweep([task], ["test-model"], params, suite_id="sweep")
        assert len(results) == 3

    async def test_sweep_tags_results(self):
        """each result has correct context_size metadata."""
        from bench_harness.runners.completion_runner import RunResult
        from datetime import datetime, timezone

        def make_result(context_tokens, est_tokens):
            r = RunResult(
                run_id="mock", suite_id="sweep", task_id="task_001",
                model_alias="test-model", exit_status="success",
                context_tokens=context_tokens,
                estimated_prompt_tokens=est_tokens,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            return r

        async def mock_run(task, params, suite_id=""):
            return make_result(
                params.get("context_tokens"),
                params.get("estimated_prompt_tokens"),
            )

        mock_runner = MagicMock()
        mock_runner.run = mock_run
        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small", "medium"])
        task = {"id": "task_001", "prompt": "Test prompt", "input": {}}
        params = {"model_alias": "test-model"}

        results = await sweep.run_sweep([task], ["test-model"], params, suite_id="sweep")
        sizes_found = {r.context_tokens for r in results}
        assert sizes_found == {"small", "medium"}

    async def test_sweep_error_handling(self):
        """errors in one size don't fail the whole sweep."""
        mock_runner = MagicMock()
        # Fail only for "large" size
        async def failing_run(task, params, suite_id=""):
            size = params.get("context_tokens", "")
            if size == "large":
                raise ConnectionError("API down")
            mr = MagicMock()
            mr.exit_status = "success"
            mr.context_tokens = params.get("context_tokens")
            mr.estimated_prompt_tokens = 100
            return mr
        mock_runner.run = failing_run

        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small", "medium", "large"])
        task = {"id": "task_001", "prompt": "Test prompt", "input": {}}
        params = {"model_alias": "test-model"}

        results = await sweep.run_sweep([task], ["test-model"], params, suite_id="sweep")
        # Should get 3 results: 2 success + 1 error
        assert len(results) == 3
        success_count = sum(1 for r in results if r.exit_status == "success")
        error_count = sum(1 for r in results if r.exit_status == "error")
        assert success_count == 2
        assert error_count == 1

    async def test_sweep_multi_task_multi_model(self):
        """sweep with multiple tasks and models produces correct count."""
        mock_runner, mock_result = await self._mock_runner_factory()
        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small", "large"])
        tasks = [
            {"id": "task_a", "prompt": "A", "input": {}},
            {"id": "task_b", "prompt": "B", "input": {}},
        ]
        params = {"model_alias": "m1"}

        results = await sweep.run_sweep(tasks, ["m1"], params, suite_id="sweep")
        assert len(results) == 4  # 2 tasks x 2 sizes

    async def test_sweep_error_result_has_metadata(self):
        """error results from sweep still have context_tokens tagged."""
        mock_runner = MagicMock()

        async def always_fail(task, params, suite_id=""):
            raise RuntimeError("API unavailable")
        mock_runner.run = always_fail

        sweep = ContextSizeSweepRunner(mock_runner, sizes=["small"])
        task = {"id": "task_001", "prompt": "Test prompt", "input": {}}
        params = {"model_alias": "test-model"}

        results = await sweep.run_sweep([task], ["test-model"], params, suite_id="sweep")
        assert len(results) == 1
        assert results[0].exit_status == "error"
        assert results[0].context_tokens == "small"

    def test_sweep_has_packer(self):
        """sweep runner has a ContextPacker instance."""
        mock_runner = MagicMock()
        sweep = ContextSizeSweepRunner(mock_runner)
        assert isinstance(sweep.packer, ContextPacker)



# ══════════════════════════════════════════════════════════════════════
# M10 — Quantization comparison
# ══════════════════════════════════════════════════════════════════════

# ── TestQuantizationComparisonReport ──────────────────────────────────

class TestQuantizationComparisonReport:
    """Test quantization comparison report generation."""

    def _make_q_run(
        self,
        task_id: str = "quant.math_001",
        model_alias: str = "model-a",
        score: float = 0.8,
        quantization: str | None = None,
        tps: float = 50.0,
        wall_ms: float = 1000.0,
    ) -> dict[str, Any]:
        return {
            "run_id": f"run-q-{task_id}-{quantization or 'none'}",
            "suite_id": "quant-test",
            "task_id": task_id,
            "model_alias": model_alias,
            "score_primary": score,
            "quantization": quantization,
            "tokens_per_second": tps,
            "total_wall_ms": wall_ms,
            "exit_status": "success",
        }

    def test_quant_summary_sections(self):
        """report has Quantization Summary section."""
        runs = [
            self._make_q_run(quantization="FP16", score=0.9, tps=30.0),
            self._make_q_run(quantization="FP16", score=0.85, tps=32.0),
            self._make_q_run(quantization="INT4", score=0.7, tps=80.0),
            self._make_q_run(quantization="INT4", score=0.65, tps=85.0),
        ]
        report = generate_quantization_report(runs)
        assert "Quantization Summary" in report
        assert "FP16" in report
        assert "INT4" in report

    def test_quality_delta_section(self):
        """report has Quality Delta section with score differences."""
        runs = [
            self._make_q_run(quantization="FP16", score=0.9, tps=30.0),
            self._make_q_run(quantization="FP8", score=0.8, tps=50.0),
            self._make_q_run(quantization="INT4", score=0.6, tps=90.0),
        ]
        report = generate_quantization_report(runs)
        assert "Quality Delta" in report
        assert "Delta" in report
        assert "FP8" in report
        assert "INT4" in report

    def test_best_by_family_section(self):
        """report identifies best quantization per family."""
        runs = [
            self._make_q_run(task_id="quant.math_reasoning.001", quantization="FP16", score=0.9, tps=30.0),
            self._make_q_run(task_id="quant.math_reasoning.001", quantization="INT4", score=0.7, tps=80.0),
            self._make_q_run(task_id="quant.code_generation.001", quantization="FP16", score=0.85, tps=35.0),
            self._make_q_run(task_id="quant.code_generation.001", quantization="FP8", score=0.75, tps=60.0),
        ]
        report = generate_quantization_report(runs)
        assert "Best Quantization by Task Family" in report
        assert "math_reasoning" in report

    def test_sensitivity_analysis(self):
        """report ranks families by quantization sensitivity."""
        runs = [
            self._make_q_run(task_id="quant.math_reasoning.001", quantization="FP16", score=0.95, tps=30.0),
            self._make_q_run(task_id="quant.math_reasoning.001", quantization="INT4", score=0.50, tps=80.0),
            self._make_q_run(task_id="quant.code_generation.001", quantization="FP16", score=0.90, tps=35.0),
            self._make_q_run(task_id="quant.code_generation.001", quantization="INT4", score=0.80, tps=75.0),
        ]
        report = generate_quantization_report(runs)
        assert "Sensitivity Analysis" in report
        assert "Sensitivity Rankings" in report
        assert "math_reasoning" in report

    def test_speed_quality_frontier(self):
        """report includes Pareto frontier analysis."""
        runs = [
            self._make_q_run(quantization="FP16", score=0.9, tps=30.0),
            self._make_q_run(quantization="FP8", score=0.8, tps=60.0),
            self._make_q_run(quantization="INT4", score=0.6, tps=100.0),
        ]
        report = generate_quantization_report(runs)
        assert "Speed/Quality Frontier" in report
        assert "Pareto" in report or "parieto" in report.lower()

    def test_quant_report_single_quantization(self):
        """report returns brief message when only one quantization level exists."""
        runs = [
            self._make_q_run(quantization="FP16", score=0.9),
        ]
        report = generate_quantization_report(runs)
        assert "Only one quantization level detected" in report

    def test_quant_report_no_quant_runs(self):
        """report returns empty string when no runs have quantization."""
        runs = [
            self._make_q_run(quantization=None, score=0.9),
            self._make_q_run(quantization=None, score=0.8),
        ]
        report = generate_quantization_report(runs)
        assert report == ""

    def test_is_reference_quantization_fp16(self):
        """FP16 is identified as reference."""
        assert _is_reference_quantization("FP16") is True

    def test_is_reference_quantization_int4(self):
        """INT4 is not a reference."""
        assert _is_reference_quantization("INT4") is False

    def test_is_reference_quantization_none(self):
        """None quantization is treated as reference."""
        assert _is_reference_quantization(None) is True

    def test_is_reference_quantization_fp8(self):
        """FP8 is not reference."""
        assert _is_reference_quantization("FP8") is False

    def test_score_values_filters_none(self):
        """_score_values excludes None scores."""
        runs = [
            {"score_primary": 0.8},
            {"score_primary": None},
            {"score_primary": 0.6},
        ]
        vals = _score_values(runs)
        assert vals == [0.8, 0.6]

    def test_avg_empty(self):
        """_avg returns 0.0 for empty list."""
        assert _avg([]) == 0.0

    def test_avg_values(self):
        """_avg computes correct average."""
        assert _avg([10.0, 20.0, 30.0]) == 20.0


# ── TestQuantizationTracking ─────────────────────────────────────────

class TestQuantizationTracking:
    """Test that RunResult and CLI flow support quantization field."""

    def test_runresult_has_quantization(self):
        """RunResult accepts quantization field."""
        result = RunResult(
            run_id="r-001",
            suite_id="quant-test",
            task_id="quant.math_001",
            model_alias="test-model",
            exit_status="success",
            quantization="INT4",
        )
        assert result.quantization == "INT4"

    def test_runresult_quantization_default_none(self):
        """RunResult quantization defaults to None."""
        result = RunResult(
            run_id="r-002",
            suite_id="test",
            task_id="t1",
            model_alias="m1",
            exit_status="success",
        )
        assert result.quantization is None

    def test_runresult_quantization_serialization(self):
        """RunResult with quantization serializes correctly to dict."""
        result = RunResult(
            run_id="r-003",
            suite_id="quant-test",
            task_id="quant.code_001",
            model_alias="model-a",
            exit_status="success",
            quantization="FP8",
        )
        result_dict = {
            "run_id": result.run_id,
            "task_id": result.task_id,
            "model_alias": result.model_alias,
            "quantization": result.quantization,
            "exit_status": result.exit_status,
        }
        assert result_dict["quantization"] == "FP8"
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)
        assert parsed["quantization"] == "FP8"

    def test_runresult_quantization_set_after_creation(self):
        """Quantization can be set after RunResult creation (like sweep runner does)."""
        result = RunResult(
            run_id="r-004",
            suite_id="sweep",
            task_id="t1",
            model_alias="m1",
            exit_status="success",
        )
        result.quantization = "GPTQ-Int4"
        assert result.quantization == "GPTQ-Int4"



# ── TestQuantizationTasks ────────────────────────────────────────────

class TestQuantizationTasks:
    """Test that quantization test tasks load correctly from the task dir."""

    def test_quantization_tasks_load(self):
        """3 quantization test tasks load correctly."""
        from bench_harness.tasks.loaders import load_tasks
        task_dir = Path(__file__).parent.parent / "tasks" / "quantization_test"
        tasks = load_tasks(str(task_dir))
        assert len(tasks) == 3
        task_ids = {t["id"] for t in tasks}
        assert "quant.math_reasoning_001" in task_ids
        assert "quant.code_generation_001" in task_ids
        assert "quant.format_following_001" in task_ids

    def test_quantization_tasks_have_scorers(self):
        """tasks have valid scorer configs."""
        from bench_harness.tasks.loaders import load_tasks
        task_dir = Path(__file__).parent.parent / "tasks" / "quantization_test"
        tasks = load_tasks(str(task_dir))
        for task in tasks:
            assert "scoring" in task
            assert "primary" in task["scoring"]
            scorer_name = task["scoring"]["primary"]
            assert isinstance(scorer_name, str)
            assert len(scorer_name) > 0

    def test_quantization_suite_in_config(self):
        """quantization_test suite exists in suites.yaml."""
        from bench_harness.config import load_suite_config, get_suite
        config = load_suite_config()
        suite = get_suite(config, "quantization_test")
        assert suite is not None
        assert suite.get("task_dir") == "tasks/quantization_test"
        assert suite.get("description") is not None

    def test_quantization_task_schema_valid(self):
        """All quantization tasks have required schema fields."""
        from bench_harness.tasks.loaders import load_tasks
        from bench_harness.tasks.loaders import REQUIRED_TASK_KEYS
        task_dir = Path(__file__).parent.parent / "tasks" / "quantization_test"
        tasks = load_tasks(str(task_dir))
        for task in tasks:
            for key in REQUIRED_TASK_KEYS:
                assert key in task, f"Task {task['id']} missing required key: {key}"

    def test_quantization_task_families(self):
        """Each quantization task has a family field."""
        from bench_harness.tasks.loaders import load_tasks
        task_dir = Path(__file__).parent.parent / "tasks" / "quantization_test"
        tasks = load_tasks(str(task_dir))
        families = [t.get("family") for t in tasks]
        assert "math_reasoning" in families
        assert "code_generation" in families
        assert "format_following" in families



# ══════════════════════════════════════════════════════════════════════
# M12 — Report v2
# ══════════════════════════════════════════════════════════════════════

# ── TestReportV2 ──────────────────────────────────────────────────────

class TestReportV2:
    """Test v2 report generation."""

    def test_v2_executive_summary(self):
        """report has Executive Summary section."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9, tps=50.0),
            _make_run_result_v2(task_id="test.task_a", model_alias="model-b", score=0.7, tps=80.0),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["executive_summary"],
            )
            assert "Executive Summary" in report
            assert "Best Overall Score" in report
            assert "model-a" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_model_comparison(self):
        """report has cross-model ranking table."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9, tps=50.0),
            _make_run_result_v2(task_id="test.task_a", model_alias="model-b", score=0.7, tps=80.0),
            _make_run_result_v2(task_id="test.task_a", model_alias="model-c", score=0.8, tps=60.0),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["model_comparison"],
            )
            assert "Model Comparison" in report
            # model-a should be ranked first (highest score)
            assert report.index("model-a") < report.index("model-c")
            assert report.index("model-c") < report.index("model-b")
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_best_by_family(self):
        """report has Best Model by Task Family section."""
        runs = [
            _make_run_result_v2(task_id="local.docker_compose.fix_001", model_alias="model-a", score=0.9),
            _make_run_result_v2(task_id="local.docker_compose.fix_001", model_alias="model-b", score=0.7),
            _make_run_result_v2(task_id="local.git.trouble_001", model_alias="model-a", score=0.6),
            _make_run_result_v2(task_id="local.git.trouble_001", model_alias="model-b", score=0.85),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["best_by_family"],
            )
            assert "Best Model by Task Family" in report
            assert "docker_compose" in report
            assert "git" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_failure_analysis(self):
        """report has Failure Clustering section."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9, status="success"),
            _make_run_result_v2(task_id="test.task_b", model_alias="model-b", score=None, status="error"),
        ]
        # Add error details
        runs[1]["error_message"] = "Connection timeout"
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["failure_analysis"],
            )
            assert "Failure Analysis" in report
            assert "Connection timeout" in report or "Failed runs" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_regression_detection(self):
        """report detects regressions when prior_runs provided."""
        new_runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.6),
        ]
        old_runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                new_runs, "v2-test", model_config, out_path,
                sections=["regression_detection"],
                prior_runs=old_runs,
            )
            assert "Regression Detection" in report
            assert "REGRESSION" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_discriminating_tasks(self):
        """report identifies most discriminating tasks."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.95),
            _make_run_result_v2(task_id="test.task_a", model_alias="model-b", score=0.10),
            _make_run_result_v2(task_id="test.task_b", model_alias="model-a", score=0.80),
            _make_run_result_v2(task_id="test.task_b", model_alias="model-b", score=0.70),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["discriminating_tasks"],
            )
            assert "Most Discriminating Tasks" in report
            # task_a should appear as more discriminating (0.95 vs 0.10 = big gap)
            assert "task_a" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_default_sections(self):
        """Default sections include all expected report sections."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9),
            _make_run_result_v2(task_id="test.task_a", model_alias="model-b", score=0.7),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(runs, "v2-test", model_config, out_path)
            # All default sections should be present
            assert "Executive Summary" in report
            assert "Model Comparison" in report
            assert "Best Model by Task Family" in report
            assert "Speed/Quality Frontier" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_model_info_in_report(self):
        """Report includes model info table when models_config is provided."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9),
        ]
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "FP16", "notes": "test model"},
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["executive_summary"],
            )
            assert "Models" in report
            assert "vllm" in report
            assert "FP16" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_regression_no_prior_runs(self):
        """report handles missing prior_runs gracefully (no regressions section)."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["regression_detection"],
                prior_runs=None,
            )
            # Should not crash — when no prior runs, the section is empty
            assert "Regression Detection" not in report or "No significant" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_regression_improvement_detected(self):
        """report detects improvements when prior_runs provided and score went up."""
        new_runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.95),
        ]
        old_runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.7),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                new_runs, "v2-test", model_config, out_path,
                sections=["regression_detection"],
                prior_runs=old_runs,
            )
            assert "IMPROVED" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_report_written_to_file(self):
        """generate_report_v2 writes the report to the output file."""
        runs = [
            _make_run_result_v2(task_id="test.task_a", model_alias="model-a", score=0.9),
        ]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "v2-test", model_config, out_path,
                sections=["executive_summary"],
            )
            file_content = Path(out_path).read_text()
            assert "Executive Summary" in file_content
            # File content may differ by trailing newline
            assert file_content.strip() == report.strip()
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_format_delta_positive(self):
        """_format_delta adds + sign for positive values."""
        assert _format_delta(0.5) == "+0.500"

    def test_format_delta_negative(self):
        """_format_delta uses - sign for negative values."""
        assert _format_delta(-0.3) == "-0.300"

    def test_format_delta_zero(self):
        """_format_delta with zero has no sign."""
        assert _format_delta(0.0) == "0.000"

    def test_get_score_for_model(self):
        """_get_score_for_model computes average score correctly."""
        runs = [
            {"score_primary": 0.8},
            {"score_primary": 0.6},
        ]
        assert _get_score_for_model(runs) == 0.7

    def test_get_score_for_model_no_scores(self):
        """_get_score_for_model returns 0.0 when no scores."""
        assert _get_score_for_model([]) == 0.0

    def test_get_score_for_model_skips_none(self):
        """_get_score_for_model skips runs with None scores."""
        runs = [
            {"score_primary": 0.8},
            {"score_primary": None},
        ]
        assert _get_score_for_model(runs) == 0.8



# ── TestReportHelpers ────────────────────────────────────────────────

class TestReportHelpers:
    """Test helper functions used by v2 report."""

    def test_group_by_model(self):
        """groups runs by model_alias."""
        runs = [
            {"run_id": "r1", "model_alias": "model-a"},
            {"run_id": "r2", "model_alias": "model-b"},
            {"run_id": "r3", "model_alias": "model-a"},
        ]
        grouped = group_by_model(runs)
        assert set(grouped.keys()) == {"model-a", "model-b"}
        assert len(grouped["model-a"]) == 2
        assert len(grouped["model-b"]) == 1

    def test_group_by_model_unknown_alias(self):
        """groups runs with missing model_alias under 'unknown'."""
        runs = [{"run_id": "r1"}]
        grouped = group_by_model(runs)
        assert "unknown" in grouped
        assert len(grouped["unknown"]) == 1

    def test_group_by_family(self):
        """groups runs by task family."""
        runs = [
            {"run_id": "r1", "task_id": "local.docker_compose.fix_001"},
            {"run_id": "r2", "task_id": "local.docker_compose.fix_002"},
            {"run_id": "r3", "task_id": "local.git.trouble_001"},
        ]
        grouped = group_by_family(runs)
        assert "docker_compose" in grouped
        assert "git" in grouped
        assert len(grouped["docker_compose"]) == 2

    def test_group_by_family_unknown_task(self):
        """runs with unparseable task_id go to 'other'."""
        runs = [
            {"run_id": "r1", "task_id": "simple"},
        ]
        grouped = group_by_family(runs)
        assert "other" in grouped

    def test_compute_score_variance(self):
        """computes variance per task correctly."""
        runs = [
            _make_run_result_v2(task_id="task_a", model_alias="model-a", score=0.9),
            _make_run_result_v2(task_id="task_a", model_alias="model-b", score=0.3),
            _make_run_result_v2(task_id="task_b", model_alias="model-a", score=0.8),
            _make_run_result_v2(task_id="task_b", model_alias="model-b", score=0.7),
        ]
        results = compute_score_variance_by_task(runs)
        assert len(results) == 2
        # task_a has wider spread (0.9 vs 0.3 = range 0.6)
        # task_b has narrower spread (0.8 vs 0.7 = range 0.1)
        assert results[0]["task_id"] == "task_a"
        assert results[1]["task_id"] == "task_b"

    def test_compute_score_variance_single_model(self):
        """tasks with only one model are excluded."""
        runs = [
            _make_run_result_v2(task_id="task_a", model_alias="model-a", score=0.9),
        ]
        results = compute_score_variance_by_task(runs)
        assert len(results) == 0

    def test_compute_score_variance_sorted_desc(self):
        """results sorted by variance descending."""
        runs = []
        for i in range(3):
            for model in ["model-a", "model-b"]:
                # Create spread that decreases with i
                score_a = 0.9 - i * 0.2
                score_b = 0.1 + i * 0.2
                runs.append(
                    _make_run_result_v2(
                        task_id=f"task_{i}",
                        model_alias="model-a",
                        score=max(score_a, 0.0),
                    )
                )
                runs.append(
                    _make_run_result_v2(
                        task_id=f"task_{i}",
                        model_alias="model-b",
                        score=max(score_b, 0.0),
                    )
                )
        results = compute_score_variance_by_task(runs)
        # Check that variance is descending
        for i in range(len(results) - 1):
            assert results[i]["variance"] >= results[i + 1]["variance"]

    def test_find_pareto_frontier(self):
        """identifies Pareto-optimal models."""
        runs = [
            _make_run_result_v2(task_id="t", model_alias="fast-worse", score=0.5, tps=100.0),
            _make_run_result_v2(task_id="t", model_alias="slow-better", score=0.9, tps=20.0),
            _make_run_result_v2(task_id="t", model_alias="mid", score=0.7, tps=50.0),
            _make_run_result_v2(task_id="t", model_alias="dominated", score=0.4, tps=30.0),
        ]
        frontier = find_pareto_frontier(runs)
        frontier_models = {f["model"] for f in frontier}
        # fast-worse (low score, high tps) should be Pareto optimal
        # slow-better (high score, low tps) should be Pareto optimal
        # mid might be Pareto depending on exact values
        assert "fast-worse" in frontier_models
        assert "slow-better" in frontier_models

    def test_find_pareto_frontier_single_model(self):
        """single model is always on frontier."""
        runs = [
            _make_run_result_v2(task_id="t", model_alias="only", score=0.8, tps=50.0),
        ]
        frontier = find_pareto_frontier(runs)
        assert len(frontier) == 1
        assert frontier[0]["model"] == "only"

    def test_find_pareto_frontier_all_dominated(self):
        """one model can dominate all others."""
        runs = [
            _make_run_result_v2(task_id="t", model_alias="dominant", score=0.9, tps=100.0),
            _make_run_result_v2(task_id="t", model_alias="weak-slow", score=0.3, tps=10.0),
            _make_run_result_v2(task_id="t", model_alias="weak-fast", score=0.4, tps=50.0),
        ]
        frontier = find_pareto_frontier(runs)
        assert len(frontier) == 1
        assert frontier[0]["model"] == "dominant"

    def test_find_pareto_frontier_sorted_by_score(self):
        """frontier results are sorted by score descending."""
        runs = [
            _make_run_result_v2(task_id="t", model_alias="low-score", score=0.3, tps=100.0),
            _make_run_result_v2(task_id="t", model_alias="high-score", score=0.9, tps=5.0),
        ]
        frontier = find_pareto_frontier(runs)
        assert len(frontier) == 2
        assert frontier[0]["model"] == "high-score"
        assert frontier[1]["model"] == "low-score"

    def test_cluster_failures(self):
        """clusters failures by error pattern."""
        runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.9, status="success"),
            _make_error_run_result(task_id="t2", model_alias="m-b", error_message="Connection timeout"),
            _make_error_run_result(task_id="t3", model_alias="m-c", error_message="Connection timeout"),
            _make_error_run_result(task_id="t4", model_alias="m-d", error_message="API rate limit exceeded"),
        ]
        clusters = cluster_failures(runs)
        assert len(clusters) == 2
        # All "Connection timeout" errors should be in the same cluster
        for cluster_name, cluster_runs in clusters.items():
            if "Connection timeout" in cluster_name:
                assert len(cluster_runs) == 2

    def test_cluster_failures_empty(self):
        """no failures returns empty dict."""
        runs = [
            _make_run_result_v2(status="success"),
            _make_run_result_v2(status="success"),
        ]
        clusters = cluster_failures(runs)
        assert clusters == {}

    def test_cluster_failures_unknown_errors(self):
        """errors without messages go to 'unknown errors'."""
        runs = [
            _make_error_run_result(
                task_id="t1", model_alias="m-a", error_message="",
            ),
            _make_error_run_result(
                task_id="t2", model_alias="m-b", error_message="",
            ),
        ]
        clusters = cluster_failures(runs)
        assert "unknown errors" in clusters
        assert len(clusters["unknown errors"]) == 2

    def test_detect_regressions(self):
        """detects score drops below tolerance."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.5),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.9),
        ]
        regressions = detect_regressions(new_runs, old_runs)
        assert len(regressions) == 1
        assert regressions[0]["status"] == "REGRESSION"
        assert abs(regressions[0]["delta"] - (-0.4)) < 0.01

    def test_detect_regressions_improvement(self):
        """detects improvements when new score is higher."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.9),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.5),
        ]
        regressions = detect_regressions(new_runs, old_runs)
        assert len(regressions) == 1
        assert regressions[0]["status"] == "IMPROVED"

    def test_detect_regressions_no_change(self):
        """no regression detected when scores are the same."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.8),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.8),
        ]
        regressions = detect_regressions(new_runs, old_runs)
        assert len(regressions) == 0

    def test_detect_regressions_within_tolerance(self):
        """change within tolerance is not flagged."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.76),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.8),
        ]
        regressions = detect_regressions(new_runs, old_runs, tolerance=0.05)
        assert len(regressions) == 0

    def test_detect_regressions_sorted_by_delta(self):
        """regressions are sorted by delta (worst first)."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.3),
            _make_run_result_v2(task_id="t2", model_alias="m-a", score=0.7),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.9),
            _make_run_result_v2(task_id="t2", model_alias="m-a", score=0.8),
        ]
        regressions = detect_regressions(new_runs, old_runs)
        assert len(regressions) == 2
        assert regressions[0]["delta"] < regressions[1]["delta"]

    def test_detect_regressions_missing_key(self):
        """regressions contain all expected keys."""
        new_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.5),
        ]
        old_runs = [
            _make_run_result_v2(task_id="t1", model_alias="m-a", score=0.9),
        ]
        regressions = detect_regressions(new_runs, old_runs)
        for r in regressions:
            assert "task_id" in r
            assert "model_alias" in r
            assert "old_score" in r
            assert "new_score" in r
            assert "delta" in r
            assert "status" in r

    def test_group_by_quantization(self):
        """groups runs by quantization type."""
        runs = [
            {"run_id": "r1", "quantization": "FP16"},
            {"run_id": "r2", "quantization": "INT4"},
            {"run_id": "r3", "quantization": "FP16"},
            {"run_id": "r4", "quantization": None},
        ]
        grouped = group_by_quantization(runs)
        assert "FP16" in grouped
        assert "INT4" in grouped
        assert "unquantized" in grouped
        assert len(grouped["FP16"]) == 2
        assert len(grouped["unquantized"]) == 1



# ── TestMarkdownIntegration ──────────────────────────────────────────

class TestMarkdownIntegration:
    """Test that report generation integrates correctly with file output."""

    def test_generate_report_v2_flag(self):
        """generate_report(v2=True) produces v2 report."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-a", score=0.9),
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-b", score=0.7),
        ]
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "FP16", "notes": ""},
                "model-b": {"backend": "litellm", "quantization": "INT4", "notes": ""},
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(runs, "v2-flag-test", model_config, out_path, v2=True)
            assert "Benchmark Report v2: v2-flag-test" in report
            assert "Executive Summary" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_generate_report_sections_filter(self):
        """passing sections argument limits output."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-a", score=0.9),
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-b", score=0.7),
        ]
        model_config = {"models": {}}

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(
                runs, "sections-filter-test", model_config, out_path,
                v2=True,
                sections=["executive_summary"],
            )
            assert "Executive Summary" in report
            # Model comparison should NOT be present since we filtered it
            assert "Model Comparison" not in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_generate_report_legacy(self):
        """generate_report(v2=False) produces legacy format."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-a", score=0.9),
        ]
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "FP16", "notes": ""},
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(runs, "legacy-test", model_config, out_path, v2=False)
            # Legacy format should not have v2 header
            assert "Benchmark Report v2:" not in report
            assert "Benchmark Report: legacy-test" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_generate_report_v2_writes_to_file(self):
        """generate_report(v2=True) writes output file."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-a", score=0.9),
        ]
        model_config = {"models": {}}

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(
                runs, "file-test", model_config, out_path, v2=True,
                sections=["executive_summary"],
            )
            written = Path(out_path).read_text()
            assert "Executive Summary" in written
            assert written.strip() == report.strip()
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_generate_report_v2_model_config(self):
        """v2 report includes model details from models_config."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run_result_v2(task_id="local.test.t1", model_alias="model-a", score=0.9),
        ]
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "FP16", "notes": "test"},
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(
                runs, "model-config-test", model_config, out_path, v2=True,
                sections=["executive_summary"],
            )
            assert "vllm" in report
            assert "FP16" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_format_safe_pct(self):
        """_format_safe_pct formats percentages correctly."""
        assert _format_safe_pct(10, 10) == "100%"
        assert _format_safe_pct(5, 10) == "50%"
        assert _format_safe_pct(0, 10) == "0%"
        assert _format_safe_pct(0, 0) == "N/A"

    def test_get_quantization_for_model(self):
        """_get_quantization_for_model returns most common quantization."""
        runs = [
            {"quantization": "FP16"},
            {"quantization": "FP16"},
            {"quantization": "INT4"},
        ]
        assert _get_quantization_for_model(runs) == "FP16"

    def test_get_quantization_for_model_empty(self):
        """_get_quantization_for_model returns empty string for no runs."""
        assert _get_quantization_for_model([]) == ""

    def test_get_context_for_model(self):
        """_get_context_for_model returns max prompt_tokens."""
        runs = [
            {"prompt_tokens": 100},
            {"prompt_tokens": 500},
            {"prompt_tokens": 200},
        ]
        assert _get_context_for_model(runs) == 500

    def test_get_context_for_model_no_runs(self):
        """_get_context_for_model returns 0 for no runs."""
        assert _get_context_for_model([]) == 0

    def test_v2_report_no_sections_provided(self):
        """v2 report with empty sections list produces minimal output."""
        runs = [_make_run_result_v2()]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(runs, "empty-test", model_config, out_path, sections=[])
            # Should just have the header, no sections
            assert "Benchmark Report v2: empty-test" in report
            assert "Executive Summary" not in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_v2_report_invalid_section_ignored(self):
        """v2 report ignores invalid section names."""
        runs = [_make_run_result_v2()]
        model_config = {"models": {}}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name
        try:
            report = generate_report_v2(
                runs, "invalid-section-test", model_config, out_path,
                sections=["executive_summary", "nonexistent_section"],
            )
            assert "Executive Summary" in report
            # Should not crash, invalid section is just skipped
        finally:
            Path(out_path).unlink(missing_ok=True)

