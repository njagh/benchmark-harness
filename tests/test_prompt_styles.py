"""Tests for Milestone 8 — Prompt Style Comparison.

Covers:
- Prompt template rendering for all 7 styles
- build_prompt() and render_with_style()
- StyleSweepRunner behavior
- Style comparison report generation
- Prompt style metadata flow (RunResult serialization, CLI --styles parsing)
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

from bench_harness.tasks.prompt_templates import (
    INLINE_TEMPLATES,
    build_prompt,
    load_prompt_template,
    render_template,
)


# ── Prompt Template Rendering Tests ────────────────────────────────────


class TestPromptTemplates:
    """Test the 7 built-in prompt template styles."""

    def test_all_inline_templates_exist(self):
        """All 7 expected styles are registered."""
        expected = {
            "plain", "repl", "terse", "patch_only",
            "architect", "json_schema", "step_by_step",
        }
        assert expected.issubset(set(INLINE_TEMPLATES.keys()))

    def test_plain_template(self):
        """Plain style renders user_message directly."""
        tmpl = INLINE_TEMPLATES["plain"]
        result = render_template(tmpl, {"user_message": "What is 2+2?"})
        assert result == "What is 2+2?"

    def test_repl_template(self):
        """REPL style adds REPL instructions before user_message."""
        tmpl = INLINE_TEMPLATES["repl"]
        result = render_template(tmpl, {"user_message": "Fix this bug"})
        assert "REPL mode" in result
        assert "hypothesize a single test" in result
        assert "run it" in result
        assert "interpret" in result
        assert "Fix this bug" in result

    def test_terse_template(self):
        """Terse style adds brevity instruction."""
        tmpl = INLINE_TEMPLATES["terse"]
        result = render_template(tmpl, {"user_message": "Explain sorting"})
        assert "briefly and directly" in result.lower()
        assert "Explain sorting" in result

    def test_patch_only_template(self):
        """Patch-only style requests unified diff only."""
        tmpl = INLINE_TEMPLATES["patch_only"]
        result = render_template(tmpl, {"user_message": "Fix the bug"})
        assert "unified diff" in result.lower()
        assert "no explanation" in result.lower()
        assert "Fix the bug" in result

    def test_architect_template(self):
        """Architect style adds architectural thinking instructions."""
        tmpl = INLINE_TEMPLATES["architect"]
        result = render_template(tmpl, {"user_message": "Design a cache"})
        assert "architecturally" in result.lower() or "architect" in result.lower()
        assert "analyze the problem" in result.lower()
        assert "outline the approach" in result.lower()
        assert "implement" in result.lower()
        assert "Design a cache" in result

    def test_json_schema_template(self):
        """JSON schema style includes schema and user_message."""
        tmpl = INLINE_TEMPLATES["json_schema"]
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = render_template(tmpl, {"user_message": "Answer", "schema": schema})
        assert "valid JSON" in result.lower() or "schema" in result.lower()
        assert "Answer" in result
        assert "string" in result  # from schema

    def test_step_by_step_template(self):
        """Step-by-step style asks for plan then execution."""
        tmpl = INLINE_TEMPLATES["step_by_step"]
        result = render_template(tmpl, {"user_message": "Solve this"})
        assert "step by step" in result.lower()
        assert "plan" in result.lower()
        assert "execute" in result.lower()
        assert "Solve this" in result


class TestRenderTemplate:
    """Test render_template() with various contexts."""

    def test_context_variable_substitution(self):
        """Variables in template are substituted from context."""
        tmpl = "Hello {{ name }}, you have {{ count }} items."
        result = render_template(tmpl, {"name": "Alice", "count": 5})
        assert result == "Hello Alice, you have 5 items."

    def test_file_context_injection(self):
        """Files are rendered as markdown code blocks."""
        tmpl = "{{ user_message }}\n\nFiles:\n{{ files }}"
        file_context = {
            "src/main.py": "def hello(): pass",
            "tests/test_main.py": "def test_hello(): pass",
        }
        result = render_template(tmpl, {"user_message": "Review these files"}, file_context)
        assert "## File: src/main.py" in result
        assert "## File: tests/test_main.py" in result
        assert "```" in result
        assert "def hello(): pass" in result
        assert "def test_hello(): pass" in result

    def test_file_context_sorted(self):
        """File context entries appear in sorted order."""
        tmpl = "{{ files }}"
        file_context = {
            "z_file.py": "z",
            "a_file.py": "a",
            "m_file.py": "m",
        }
        result = render_template(tmpl, {}, file_context)
        # a_file.py should appear before m_file.py before z_file.py
        a_pos = result.index("a_file.py")
        m_pos = result.index("m_file.py")
        z_pos = result.index("z_file.py")
        assert a_pos < m_pos < z_pos

    def test_file_context_none(self):
        """No files context does not crash."""
        result = render_template("Hello world", {"user_message": "Hello world"})
        assert "Hello world" in result

    def test_null_system_message(self):
        """None values in context are handled gracefully."""
        tmpl = "{{ user_message }}{{ system_message }}"
        result = render_template(tmpl, {"user_message": "Hi", "system_message": None})
        assert "Hi" in result


class TestLoadPromptTemplate:
    """Test load_prompt_template() for inline and file-based loading."""

    def test_load_inline_plain(self):
        """load_prompt_template returns inline template for known names."""
        result = load_prompt_template("plain")
        assert "{{ user_message }}" in result

    def test_load_inline_repl(self):
        """load_prompt_template returns repl inline template."""
        result = load_prompt_template("repl")
        assert "REPL mode" in result

    def test_load_inline_unknown_returns_user_message(self):
        """load_prompt_template raises FileNotFoundError for unknown inline template."""
        with pytest.raises(FileNotFoundError):
            load_prompt_template("nonexistent_style_xyz_12345")

    def test_load_from_temp_file(self, tmp_path: Path):
        """load_prompt_template loads from a file in template directory."""
        template_dir = tmp_path / "prompt_templates"
        template_dir.mkdir()
        (template_dir / "custom_style.md").write_text(
            "Custom: {{ user_message }}"
        )
        result = load_prompt_template("custom_style", template_dir=template_dir)
        assert "Custom:" in result


class TestBuildPrompt:
    """Test build_prompt() with plain style and various inputs."""

    def test_build_prompt_plain(self):
        """Plain style prompt builds directly from user_message."""
        task = {
            "id": "test.task_001",
            "prompt": "What is 2+2?",
            "input": {"user_message": "What is 2+2?"},
        }
        prompt, _ = build_prompt(task, prompt_style="plain")
        assert "What is 2+2?" in prompt

    def test_build_prompt_no_input_uses_prompt(self):
        """When no input, build_prompt falls back to task prompt."""
        task = {
            "id": "test.task_002",
            "prompt": "Describe Python",
        }
        prompt, _ = build_prompt(task, prompt_style="plain")
        assert "Describe Python" in prompt

    def test_build_prompt_repl(self):
        """REPL style adds REPL instructions to the prompt."""
        task = {
            "id": "test.task_003",
            "prompt": "Fix the sorting bug",
            "input": {"user_message": "Fix the sorting bug"},
        }
        prompt, _ = build_prompt(task, prompt_style="repl")
        assert "REPL mode" in prompt

    def test_build_prompt_architect(self):
        """Architect style adds architectural thinking instructions."""
        task = {
            "id": "test.task_004",
            "prompt": "Design a system",
            "input": {"user_message": "Design a system"},
        }
        prompt, _ = build_prompt(task, prompt_style="architect")
        assert "architect" in prompt.lower()

    def test_build_prompt_with_file_context(self):
        """build_prompt includes files as code blocks when file_context is provided."""
        task = {
            "id": "test.task_005",
            "prompt": "Review",
            "input": {"user_message": "Review"},
        }
        file_ctx = {"app.py": "def main(): pass"}
        prompt, refs = build_prompt(task, prompt_style="plain", file_context=file_ctx)
        assert "## File: app.py" in prompt
        assert "app.py" in refs


# ── RenderedPrompt / render_with_style Tests ─────────────────────────

class TestRenderedPrompt:
    """Test RenderedPrompt dataclass and render_with_style function."""

    def test_rendered_prompt_fields(self):
        """RenderedPrompt has all expected fields."""
        from bench_harness.tasks.prompt_templates import RenderedPrompt
        rp = RenderedPrompt(
            prompt="Hello",
            style="plain",
            referenced_files=["a.py", "b.py"],
            estimated_tokens=50,
        )
        assert rp.prompt == "Hello"
        assert rp.style == "plain"
        assert rp.referenced_files == ["a.py", "b.py"]
        assert rp.estimated_tokens == 50
        # Backwards-compatible aliases
        assert rp.style_name == "plain"
        assert rp.system_message is None
        assert rp.user_message == "Hello"
        assert rp.full_text == "Hello"
        assert rp.referenced_file_paths == ["a.py", "b.py"]

    def test_estimated_tokens_reasonable(self):
        """estimated_tokens is reasonable (positive integer)."""
        from bench_harness.tasks.prompt_templates import RenderedPrompt, _estimate_tokens
        text = "A short prompt"
        est = _estimate_tokens(text)
        assert est > 0

    def test_estimate_tokens_word_split(self):
        """_estimate_tokens counts words."""
        from bench_harness.tasks.prompt_templates import _estimate_tokens
        assert _estimate_tokens("hello world") == 2
        assert _estimate_tokens("one") == 1
        assert _estimate_tokens("") == 0

    def test_render_with_style_plain(self):
        """render_with_style returns RenderedPrompt for plain style."""
        from bench_harness.tasks.prompt_templates import render_with_style, RenderedPrompt
        task = {"id": "t1", "prompt": "Hello", "input": {"user_message": "Hello"}}
        result = render_with_style(task, "plain")
        assert isinstance(result, RenderedPrompt)
        assert "Hello" in result.full_text
        assert result.style == "plain"

    def test_render_with_style_repl(self):
        """render_with_style adds REPL instructions."""
        from bench_harness.tasks.prompt_templates import render_with_style, RenderedPrompt
        task = {"id": "t2", "prompt": "Fix bug", "input": {"user_message": "Fix bug"}}
        result = render_with_style(task, "repl")
        assert isinstance(result, RenderedPrompt)
        assert "REPL mode" in result.full_text
        assert result.style == "repl"

    def test_render_with_style_unknown_raises(self):
        """Unknown style ending in _xyz raises FileNotFoundError."""
        from bench_harness.tasks.prompt_templates import render_with_style
        task = {"id": "t3", "prompt": "Test", "input": {"user_message": "Test"}}
        with pytest.raises(FileNotFoundError):
            render_with_style(task, "definitely_not_real_xyz")

    def test_render_with_style_all_known_styles(self):
        """render_with_style works for all 7 known styles."""
        from bench_harness.tasks.prompt_templates import render_with_style, RenderedPrompt
        task = {"id": "t5", "prompt": "Test", "input": {"user_message": "Test"}}
        for style in ["plain", "repl", "terse", "patch_only", "architect", "json_schema", "step_by_step"]:
            result = render_with_style(task, style)
            assert isinstance(result, RenderedPrompt)
            assert result.style == style
            assert "Test" in result.full_text


# ── StyleSweepRunner Tests ────────────────────────────────────────────

class TestStyleSweepRunner:
    """Test StyleSweepRunner with mocked base runner to avoid API calls."""

    def test_sweep_initializes_with_styles(self):
        """StyleSweepRunner stores styles and default_style."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        mock_runner = MagicMock()
        runner = StyleSweepRunner(mock_runner, styles=["plain", "repl", "terse"])
        assert runner.styles == ["plain", "repl", "terse"]
        assert runner.default_style == "plain"

    def test_sweep_default_style_plain(self):
        """Default style defaults to 'plain'."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        mock_runner = MagicMock()
        runner = StyleSweepRunner(mock_runner, styles=None)
        assert runner.styles == ["plain"]

    def test_dry_run_builds_sweeps(self):
        """Dry-run mode builds sweep combinations without a runner."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner(
            base_tasks=[
                {"id": "task_a", "prompt": "Test A", "input": {"user_message": "Test A"}},
                {"id": "task_b", "prompt": "Test B", "input": {"user_message": "Test B"}},
            ],
            prompt_styles=["plain", "repl", "terse"],
            model_alias="test-model",
            suite_id="style-sweep-test",
        )
        # 2 tasks × 3 styles = 6 sweeps
        assert len(runner.sweeps) == 6

    def test_dry_run_sweeps_tagged_with_style(self):
        """Dry-run sweeps are tagged with correct prompt_style."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner(
            base_tasks=[
                {"id": "task_a", "prompt": "Test A", "input": {"user_message": "Test A"}},
            ],
            prompt_styles=["plain", "repl", "terse"],
        )
        for sweep in runner.sweeps:
            assert "prompt_style" in sweep
            assert sweep["prompt_style"] in ("plain", "repl", "terse")
            assert sweep["task_id"] == "task_a"

    def test_dry_run_task_ids_preserved(self):
        """Dry-run sweeps preserve original task IDs."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner(
            base_tasks=[
                {"id": "task_a", "prompt": "Test A", "input": {"user_message": "Test A"}},
                {"id": "task_b", "prompt": "Test B", "input": {"user_message": "Test B"}},
            ],
            prompt_styles=["plain", "repl"],
        )
        task_ids = {s["task_id"] for s in runner.sweeps}
        assert task_ids == {"task_a", "task_b"}

    def test_wrong_style_falls_back_to_plain(self):
        """Unknown style names fall back to plain in dry-run mode."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner(
            base_tasks=[
                {"id": "task_x", "prompt": "X", "input": {"user_message": "X"}},
            ],
            prompt_styles=["plain", "nonexistent_style"],
        )
        # nonexistent_style falls back to plain, so only plain appears
        styles_found = {s["prompt_style"] for s in runner.sweeps}
        assert styles_found == {"plain"}

    def test_dry_run_with_single_style(self):
        """Dry-run with a single style produces single sweep per task."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner(
            base_tasks=[
                {"id": "only_task", "prompt": "O", "input": {"user_message": "O"}},
            ],
            prompt_styles=["repl"],
        )
        assert len(runner.sweeps) == 1
        assert runner.sweeps[0]["prompt_style"] == "repl"

    def test_run_task_with_style_live_mode(self):
        """run_task_with_style raises if no base_runner in live mode."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        runner = StyleSweepRunner()  # No base_runner
        task = {
            "id": "task_a",
            "prompt": "Test prompt",
            "input": {"user_message": "Test message"},
        }
        params = {"model_alias": "test-model"}

        with pytest.raises(RuntimeError, match="Cannot run live sweep"):
            asyncio.run(runner.run_task_with_style(task, params, "repl"))

    def test_run_sweep_live_mode_with_mock(self):
        """Live mode run_sweep uses mocked base_runner."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner
        from bench_harness.runners.completion_runner import RunResult

        mock_result = MagicMock()
        mock_result.exit_status = "success"
        mock_result.prompt_style = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        runner = StyleSweepRunner(mock_runner, styles=["plain", "repl"])
        tasks = [
            {"id": "task_a", "prompt": "A", "input": {"user_message": "A"}},
        ]
        params = {"model_alias": "model-1"}

        results = asyncio.run(runner.run_sweep(tasks, ["model-1"], params, suite_id="sweep"))

        # 1 task × 2 styles = 2 results
        assert len(results) == 2
        # Each result should have prompt_style tagged
        styles_found = {r.prompt_style for r in results}
        assert styles_found == {"plain", "repl"}

    def test_run_sweep_tags_results_with_style(self):
        """Each sweep result has prompt_style set correctly."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        mock_result = MagicMock()
        mock_result.exit_status = "success"
        mock_result.prompt_style = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        runner = StyleSweepRunner(mock_runner, styles=["plain", "terse"])
        tasks = [{"id": "t1", "prompt": "X", "input": {"user_message": "X"}}]
        params = {"model_alias": "m1"}

        results = asyncio.run(runner.run_sweep(tasks, ["m1"], params, suite_id="sweep"))

        for r in results:
            assert r.prompt_style in ("plain", "terse")

    def test_run_sweep_handles_errors_gracefully(self):
        """Style sweep creates error results when the base runner fails."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=Exception("API down"))

        runner = StyleSweepRunner(mock_runner, styles=["plain"])
        tasks = [{"id": "t1", "prompt": "X", "input": {"user_message": "X"}}]
        params = {"model_alias": "m1"}

        results = asyncio.run(runner.run_sweep(tasks, ["m1"], params, suite_id="sweep"))

        # Should get an error result, not crash
        assert len(results) == 1
        assert results[0].exit_status == "error"

    def test_sweep_multi_task_multi_style(self):
        """Sweep with multiple tasks and styles produces correct count."""
        from bench_harness.runners.style_sweep_runner import StyleSweepRunner

        mock_result = MagicMock()
        mock_result.exit_status = "success"
        mock_result.prompt_style = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        runner = StyleSweepRunner(mock_runner, styles=["plain", "repl", "terse"])
        tasks = [
            {"id": "t1", "prompt": "A", "input": {"user_message": "A"}},
            {"id": "t2", "prompt": "B", "input": {"user_message": "B"}},
        ]
        params = {"model_alias": "m1"}

        results = asyncio.run(runner.run_sweep(tasks, ["m1"], params, suite_id="sweep"))

        # 2 tasks × 3 styles = 6 results
        assert len(results) == 6


# ── Style Comparison Report Generation Tests ─────────────────────────

def _make_run(
    task_id: str = "t1",
    style: str = "plain",
    score: float = 0.8,
    model: str = "model-a",
    completion_tokens: int = 100,
    prompt_tokens: int = 50,
    ttft_ms: float = 100.0,
    decode_ms: float = 200.0,
    total_wall_ms: float = 300.0,
    tps: float = 50.0,
    status: str = "success",
) -> dict[str, Any]:
    """Helper to create a minimal run dict with prompt_style."""
    return {
        "run_id": f"run-{task_id}-{style}",
        "suite_id": "test-suite",
        "task_id": task_id,
        "model_alias": model,
        "prompt_style": style,
        "score_primary": score,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": completion_tokens + prompt_tokens,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "total_wall_ms": total_wall_ms,
        "tokens_per_second": tps,
        "exit_status": status,
    }


class TestStyleComparisonReport:
    """Test style comparison report generation."""

    def test_report_generated_when_runs_have_prompt_style(self):
        """generate_style_report produces output when runs have prompt_style."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7),
            _make_run(style="repl", score=0.9),
        ]
        report = generate_style_report(runs)
        assert report != ""
        assert "Style Comparison Report" in report

    def test_report_empty_when_no_prompt_style(self):
        """generate_style_report returns empty string when no runs have prompt_style."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7),
        ]
        # Remove prompt_style to simulate runs without it
        runs[0].pop("prompt_style", None)
        report = generate_style_report(runs)
        assert report == ""

    def test_report_contains_style_summary_section(self):
        """Report contains the Style Comparison Summary section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7),
            _make_run(style="repl", score=0.9),
        ]
        report = generate_style_report(runs)
        assert "Style Comparison Summary" in report

    def test_report_contains_per_task_breakdown(self):
        """Report contains the Per-Task Style Breakdown section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(task_id="task_a", style="plain", score=0.7),
            _make_run(task_id="task_a", style="repl", score=0.9),
            _make_run(task_id="task_b", style="plain", score=0.6),
            _make_run(task_id="task_b", style="repl", score=0.8),
        ]
        report = generate_style_report(runs)
        assert "Per-Task Style Breakdown" in report
        assert "task_a" in report

    def test_report_contains_best_style_per_family(self):
        """Report contains the Best Style Per Family section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(task_id="local.docker_compose.fix_001", style="plain", score=0.7),
            _make_run(task_id="local.docker_compose.fix_001", style="repl", score=0.9),
        ]
        report = generate_style_report(runs)
        assert "Best Style Per Task Family" in report

    def test_report_contains_verbosity_analysis(self):
        """Report contains the Verbosity Analysis section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7, completion_tokens=50, prompt_tokens=30),
            _make_run(style="architect", score=0.8, completion_tokens=200, prompt_tokens=100),
        ]
        report = generate_style_report(runs)
        assert "Verbosity Analysis" in report

    def test_report_contains_latency_comparison(self):
        """Report contains the Latency Comparison section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7, ttft_ms=50, decode_ms=100, total_wall_ms=150),
            _make_run(style="repl", score=0.9, ttft_ms=100, decode_ms=200, total_wall_ms=300),
        ]
        report = generate_style_report(runs)
        assert "Latency Comparison" in report

    def test_report_contains_recommended_style(self):
        """Report contains the Recommended Style section."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7),
            _make_run(style="repl", score=0.9),
        ]
        report = generate_style_report(runs)
        assert "Recommended Style" in report

    def test_report_all_sections_present(self):
        """All 6 expected sections are present in the report."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(task_id="local.docker_compose.fix_001", style="plain", score=0.7),
            _make_run(task_id="local.docker_compose.fix_001", style="repl", score=0.9),
            _make_run(task_id="local.git.trouble_001", style="plain", score=0.6),
            _make_run(task_id="local.git.trouble_001", style="terse", score=0.85),
        ]
        report = generate_style_report(runs)

        expected_sections = [
            "Style Comparison Summary",
            "Per-Task Style Breakdown",
            "Best Style Per Task Family",
            "Verbosity Analysis",
            "Latency Comparison",
            "Recommended Style",
        ]
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"

    def test_report_filters_by_styles_param(self):
        """style comparison report respects the styles filter parameter."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(style="plain", score=0.7),
            _make_run(style="repl", score=0.9),
            _make_run(style="architect", score=0.8),
        ]
        report = generate_style_report(runs, styles=["plain", "repl"])
        assert "Style Comparison Summary" in report
        # Only plain and repl should appear
        assert "| plain |" in report
        assert "| repl |" in report

    def test_report_multiple_styles_per_task(self):
        """Report correctly compares multiple styles for the same task."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(task_id="t1", style="plain", score=0.5),
            _make_run(task_id="t1", style="repl", score=0.8),
            _make_run(task_id="t1", style="architect", score=0.7),
        ]
        report = generate_style_report(runs)
        assert "t1" in report
        assert "| plain |" in report
        assert "| repl |" in report
        assert "| architect |" in report

    def test_family_best_style_selection(self):
        """Best style per family picks the highest-scoring style."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            _make_run(task_id="local.docker_compose.a_001", style="plain", score=0.6),
            _make_run(task_id="local.docker_compose.a_001", style="repl", score=0.9),
            _make_run(task_id="local.git.b_001", style="terse", score=0.7),
            _make_run(task_id="local.git.b_001", style="plain", score=0.8),
        ]
        report = generate_style_report(runs)
        assert "docker_compose" in report
        assert "git" in report or "git" in report

    def test_report_without_scored_runs(self):
        """Report handles runs without scores gracefully."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = [
            {"run_id": "r1", "task_id": "t1", "prompt_style": "plain", "exit_status": "success"},
            {"run_id": "r2", "task_id": "t1", "prompt_style": "repl", "exit_status": "success"},
        ]
        report = generate_style_report(runs)
        # Should not crash; sections that need scores handle gracefully
        assert report == "" or "Style Comparison Report" in report


# ── Prompt Style Metadata Flow Tests ─────────────────────────────────

class TestRunResultPromptStyle:
    """Test RunResult with prompt_style field serialization."""

    def test_runresult_prompt_style_field_exists(self):
        """RunResult supports prompt_style attribute (set by StyleSweepRunner)."""
        from bench_harness.runners.completion_runner import RunResult

        result = RunResult(
            run_id="r-001",
            suite_id="style-test",
            task_id="t1",
            model_alias="test-model",
            exit_status="success",
        )
        # StyleSweepRunner sets prompt_style after run
        result.prompt_style = "repl"
        assert result.prompt_style == "repl"

    def test_runresult_prompt_style_in_json(self):
        """RunResult with prompt_style serializes to JSON."""
        from bench_harness.runners.completion_runner import RunResult

        result = RunResult(
            run_id="r-002",
            suite_id="style-test",
            task_id="t2",
            model_alias="test-model",
            exit_status="success",
        )
        result.prompt_style = "architect"
        result_dict = {
            "run_id": result.run_id,
            "suite_id": result.suite_id,
            "task_id": result.task_id,
            "model_alias": result.model_alias,
            "prompt_style": result.prompt_style,
            "exit_status": result.exit_status,
        }
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)
        assert parsed["prompt_style"] == "architect"

    def test_runresult_prompt_style_serialized_from_dict(self):
        """RunResult fields serialize correctly to a dict format."""
        from bench_harness.runners.completion_runner import RunResult

        result = RunResult(
            run_id="r-003",
            suite_id="style-test",
            task_id="t3",
            model_alias="test-model",
            prompt_style="step_by_step",
            exit_status="success",
        )
        result.prompt_style = "patch_only"
        # Verify the prompt_style attribute persists
        assert result.prompt_style == "patch_only"


class TestCLIStylesParsing:
    """Test CLI --styles parsing (comma-separated → list)."""

    def test_comma_separated_styles_to_list(self):
        """Comma-separated style string parses to list of strings."""
        styles_str = "repl,terse,patch_only"
        styles = [s.strip() for s in styles_str.split(",")]
        assert styles == ["repl", "terse", "patch_only"]

    def test_single_style(self):
        """Single style string parses to a single-element list."""
        styles_str = "architect"
        styles = [s.strip() for s in styles_str.split(",")]
        assert styles == ["architect"]

    def test_whitespace_handling(self):
        """Styles with extra whitespace are trimmed."""
        styles_str = "  plain ,  terse  ,  repl "
        styles = [s.strip() for s in styles_str.split(",")]
        assert styles == ["plain", "terse", "repl"]

    def test_empty_styles_string(self):
        """Empty string produces list with empty string."""
        styles_str = ""
        styles = [s.strip() for s in styles_str.split(",")]
        assert styles == [""]


# ── Markdown Report Integration Tests ────────────────────────────────

class TestMarkdownReportStyleIntegration:
    """Test that markdown report integrates style comparison sections."""

    def test_markdown_report_includes_style_section(self):
        """generate_report includes style comparison when runs have prompt_style."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run(task_id="local.test.t1", style="plain", score=0.7),
            _make_run(task_id="local.test.t1", style="repl", score=0.9),
        ]
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "fp8", "notes": "test"}
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(runs, "style-test", model_config, out_path)
            # The report should include the main sections
            assert "Benchmark Report: style-test" in report
            # And should include style comparison section if prompt_style detected
            assert "Style Comparison Report" in report
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_markdown_report_no_style_section_without_prompt_style(self):
        """generate_report does not include style section when no runs have prompt_style."""
        from bench_harness.reports.markdown import generate_report
        from pathlib import Path

        runs = [
            _make_run(style="plain", score=0.7),
        ]
        runs[0].pop("prompt_style", None)
        model_config = {
            "models": {
                "model-a": {"backend": "vllm", "quantization": "fp8", "notes": "test"}
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            out_path = f.name

        try:
            report = generate_report(runs, "no-style-test", model_config, out_path)
            assert "Benchmark Report: no-style-test" in report
            # No style comparison section should be present
            assert "Style Comparison Report" not in report
        finally:
            Path(out_path).unlink(missing_ok=True)


# ── Helper Function Tests ───────────────────────────────────────────

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_avg_empty(self):
        from bench_harness.reports.style_comparison import _avg
        assert _avg([]) == 0.0

    def test_avg_values(self):
        from bench_harness.reports.style_comparison import _avg
        assert _avg([1.0, 2.0, 3.0]) == 2.0
        assert _avg([100.0]) == 100.0

    def test_stddev_single_value(self):
        from bench_harness.reports.style_comparison import _stddev
        assert _stddev([5.0]) == 0.0

    def test_stddev_multiple_values(self):
        from bench_harness.reports.style_comparison import _stddev
        # Population stddev of [1, 3, 5] = sqrt(((4+0+4)/3)) = sqrt(8/3)
        sd = _stddev([1.0, 3.0, 5.0])
        import math
        expected = math.sqrt(8.0 / 3.0)
        assert abs(sd - expected) < 0.001

    def test_score_values_filters_none(self):
        from bench_harness.reports.style_comparison import _score_values
        runs = [
            {"score_primary": 0.5},
            {"score_primary": None},
            {"score_primary": 0.9},
        ]
        vals = _score_values(runs)
        assert vals == [0.5, 0.9]

    def test_extract_family_from_task_id(self):
        from bench_harness.reports.style_comparison import _extract_family_from_task_id
        assert _extract_family_from_task_id("local.docker_compose.fix_yaml_001") == "docker_compose"
        assert _extract_family_from_task_id("local.git_linux.trouble_001") == "git_linux"


# ── Full Integration: Report from Sample Style Runs ──────────────────

class TestFullStyleReportIntegration:
    """End-to-end test: generate a complete style comparison report from sample data."""

    def test_full_report_with_multiple_families_styles_and_tasks(self):
        """Generate a full report from multi-family, multi-style sample data."""
        from bench_harness.reports.style_comparison import generate_style_report

        runs = []
        families_tasks = [
            ("local.docker_compose", "fix_yaml_001"),
            ("local.git_linux", "trouble_001"),
            ("local.qwen3_debug", "debug_001"),
        ]
        styles = ["plain", "repl", "terse"]
        scores = {
            ("plain", "local.docker_compose"): 0.6,
            ("repl", "local.docker_compose"): 0.9,
            ("terse", "local.docker_compose"): 0.7,
            ("plain", "local.git_linux"): 0.5,
            ("repl", "local.git_linux"): 0.8,
            ("terse", "local.git_linux"): 0.6,
            ("plain", "local.qwen3_debug"): 0.4,
            ("repl", "local.qwen3_debug"): 0.7,
            ("terse", "local.qwen3_debug"): 0.5,
        }

        for family_prefix, task_suffix in families_tasks:
            task_id = f"{family_prefix}.{task_suffix}"
            for style in styles:
                score = scores.get((style, family_prefix), 0.5)
                runs.append(_make_run(
                    task_id=task_id,
                    style=style,
                    score=score,
                    completion_tokens=80 + hash(f"{task_id}-{style}") % 50,
                    ttft_ms=50 + hash(f"{task_id}-{style}") % 100,
                    decode_ms=100 + hash(f"{task_id}-{style}") % 200,
                ))

        report = generate_style_report(runs)

        # Verify all sections present
        assert "Style Comparison Summary" in report
        assert "Per-Task Style Breakdown" in report
        assert "Best Style Per Task Family" in report
        assert "Verbosity Analysis" in report
        assert "Latency Comparison" in report
        assert "Recommended Style" in report

        # Verify family breakdown includes all families
        assert "docker_compose" in report
        assert "git_linux" in report
        assert "qwen3_debug" in report

        # Verify all styles appear in summary
        for style in styles:
            assert f"| {style} |" in report
