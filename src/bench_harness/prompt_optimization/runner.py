"""Prompt optimization runner — execute candidate templates and compare against baselines."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_harness.prompt_optimization.analysis import PromptAnalysis, analyze_style_data
from bench_harness.prompt_optimization.proposals import TemplateProposal, TemplateRegistry
from bench_harness.runners.style_sweep_runner import StyleSweepRunner

logger = logging.getLogger(__name__)


class PromptOptimizationRunner:
    """Runs candidate prompt templates and compares them against baselines.

    Usage:
        runner = PromptOptimizationRunner()

        # 1. Analyze existing data
        analysis = runner.analyze(db_path, suite_id="coding_benchmark")

        # 2. Load custom proposals
        registry = TemplateRegistry()
        registry.add_baseline("plain")
        proposals = load_proposals_from_yaml("proposals.yaml")
        registry.add_candidates(proposals)

        # 3. Run candidates
        results = runner.run_candidates(
            registry=registry,
            db_path="benchmark.db",
            model_aliases=["agent-code"],
            suite_id="coding_benchmark",
        )

        # 4. Generate report
        report = runner.generate_report(analysis, results)
    """

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self.base_runner: Any = None

    def analyze(
        self,
        db_path: str,
        suite_id: str = "",
        min_runs_per_style: int = 1,
    ) -> PromptAnalysis:
        """Analyze existing benchmark results for style comparison data.

        Args:
            db_path: Path to the benchmark SQLite database.
            suite_id: Suite to filter by (empty = all suites).
            min_runs_per_style: Minimum runs per style.

        Returns:
            PromptAnalysis with style rankings.
        """
        from bench_harness.storage.sqlite import BenchmarkDB
        db = BenchmarkDB(db_path)
        runs = db.get_runs(suite_id=suite_id)
        return analyze_style_data(runs, suite_id=suite_id, min_runs_per_style=min_runs_per_style)

    def run_candidates(
        self,
        registry: TemplateRegistry,
        db_path: str,
        model_aliases: list[str],
        suite_id: str = "",
        task_ids: list[str] | None = None,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run candidate templates against tasks, comparing to baselines.

        This uses StyleSweepRunner internally to execute the sweep.

        Args:
            registry: TemplateRegistry with baselines and candidates.
            db_path: Path to benchmark DB (to load tasks).
            model_aliases: List of model aliases to run against.
            suite_id: Suite to filter tasks by.
            task_ids: Optional specific task IDs to run (default = all tasks in suite).
            output_dir: Optional directory to save optimization results.

        Returns:
            List of result dicts with style, score, run_count, score_delta.
        """
        from bench_harness.storage.sqlite import BenchmarkDB
        from bench_harness.tasks.loaders import load_tasks

        db = BenchmarkDB(db_path)
        baselines = registry.get_baselines()
        candidates = registry.get_candidates()

        if not baselines:
            logger.warning("No baselines defined. Defaulting to ['plain'].")
            baselines = ["plain"]

        all_styles = list(baselines)
        for c in candidates:
            all_styles.append(c.name)

        # Load tasks for this suite
        task_dirs = []
        if suite_id:
            # Try loading from standard task directory structure
            for subdir in ("tasks/smoke", "tasks/coding_smoke", "tasks/local_coding_agent_v1",
                           "tasks/public_baseline", "tasks/synthetic", "tasks/agent_safety"):
                full_dir = Path(db_path).parent / subdir
                if full_dir.exists():
                    task_dirs.append(str(full_dir))

        all_tasks: list[dict[str, Any]] = []
        for td in task_dirs:
            loaded = load_tasks(td)
            if task_ids:
                loaded = [t for t in loaded if t.get("id") in task_ids]
            if suite_id:
                # Filter tasks that could belong to this suite
                # (actual suite filtering happens at run time via runner config)
                pass
            all_tasks.extend(loaded)

        if not all_tasks:
            logger.warning("No tasks found. Optimization will produce empty results.")
            return []

        # Build a StyleSweepRunner with all styles (baselines + candidates)
        sweep_runner = StyleSweepRunner(
            base_runner=self.base_runner,
            styles=all_styles,
            default_style="plain",
        )

        # We need the completion runner to actually execute — if not available,
        # we return metadata results instead of actual runs
        if sweep_runner.base_runner is None:
            logger.info(
                "No base CompletionRunner available for live execution. "
                "Generating metadata results from existing data."
            )
            return self._run_metadata_comparison(
                baselines, candidates, all_tasks, model_aliases, suite_id, db
            )

        # Execute sweep
        params = {
            "max_tokens": 1024,
            "temperature": 0,
            "num_runs": 1,
        }
        try:
            results = sweep_runner.run_sweep(
                tasks=all_tasks,
                model_aliases=model_aliases,
                params=params,
                suite_id=f"{suite_id}-optimization-{uuid.uuid4().hex[:8]}",
            )
        except Exception as e:
            logger.error("Style sweep failed: %s", e)
            return []

        # Convert RunResult objects to dicts and tag with candidate metadata
        run_results: list[dict[str, Any]] = []
        for r in results:
            result_dict = {
                "run_id": r.run_id,
                "suite_id": r.suite_id,
                "task_id": r.task_id,
                "model_alias": r.model_alias,
                "prompt_style": r.prompt_style,
                "score_primary": r.score_primary,
                "score_secondary": r.score_secondary,
                "completion_tokens": r.completion_tokens,
                "total_wall_ms": r.total_wall_ms,
                "ttft_ms": r.ttft_ms,
                "exit_status": r.exit_status,
            }
            run_results.append(result_dict)

        self._results = run_results

        # Save results to output_dir if specified
        if output_dir:
            self._save_results(output_dir, run_results, candidates)

        return self._compute_candidate_results(run_results, baselines, candidates)

    def _run_metadata_comparison(
        self,
        baselines: list[str],
        candidates: list[TemplateProposal],
        tasks: list[dict[str, Any]],
        model_aliases: list[str],
        suite_id: str,
        db: Any,
    ) -> list[dict[str, Any]]:
        """When no live runner is available, analyze existing data for candidate comparisons.

        Args:
            baselines: Baseline style names.
            candidates: Candidate proposals.
            tasks: Available tasks.
            model_aliases: Model aliases.
            suite_id: Suite identifier.
            db: BenchmarkDB instance.

        Returns:
            List of candidate result dicts with analysis from existing data.
        """
        # Load existing runs
        existing_runs = db.get_runs(suite_id=suite_id)

        results: list[dict[str, Any]] = []

        for candidate in candidates:
            # Count how many runs exist for this candidate style
            candidate_runs = [
                r for r in existing_runs
                if r.get("prompt_style") == candidate.name and r.get("score_primary") is not None
            ]

            candidate_score = 0.0
            if candidate_runs:
                candidate_score = sum(r["score_primary"] for r in candidate_runs) / len(candidate_runs)

            # Find baseline runs
            baseline_runs = [
                r for r in existing_runs
                if r.get("prompt_style") == candidate.baseline and r.get("score_primary") is not None
            ]
            baseline_score = 0.0
            if baseline_runs:
                baseline_score = sum(r["score_primary"] for r in baseline_runs) / len(baseline_runs)

            delta = candidate_score - baseline_score

            results.append({
                "name": candidate.name,
                "task_family": candidate.task_family,
                "baseline": candidate.baseline,
                "instructions": candidate.instructions,
                "score": candidate_score,
                "baseline_score": baseline_score,
                "score_delta": delta,
                "run_count": len(candidate_runs),
                "status": "analyzed" if candidate_runs else "no_data",
            })

            logger.info(
                "Candidate %s: score=%.3f baseline=%s (%s=%.3f) delta=%.3f runs=%d",
                candidate.name, candidate_score, candidate.baseline,
                candidate.baseline, baseline_score, delta, len(candidate_runs),
            )

        return results

    def _compute_candidate_results(
        self,
        run_results: list[dict[str, Any]],
        baselines: list[str],
        candidates: list[TemplateProposal],
    ) -> list[dict[str, Any]]:
        """Compute candidate comparison results from run data.

        Args:
            run_results: List of run result dicts.
            baselines: Baseline style names.
            candidates: Candidate proposals.

        Returns:
            List of candidate result summaries.
        """
        # Group by style
        by_style: dict[str, list[dict[str, Any]]] = {}
        for r in run_results:
            style = r.get("prompt_style")
            if style is None:
                continue
            if style not in by_style:
                by_style[style] = []
            by_style[style].append(r)

        results: list[dict[str, Any]] = []

        for candidate in candidates:
            candidate_runs = [
                r for r in by_style.get(candidate.name, [])
                if r.get("score_primary") is not None
            ]
            candidate_score = 0.0
            if candidate_runs:
                candidate_score = sum(r["score_primary"] for r in candidate_runs) / len(candidate_runs)

            baseline_runs = [
                r for r in by_style.get(candidate.baseline, [])
                if r.get("score_primary") is not None
            ]
            baseline_score = 0.0
            if baseline_runs:
                baseline_score = sum(r["score_primary"] for r in baseline_runs) / len(baseline_runs)

            results.append({
                "name": candidate.name,
                "task_family": candidate.task_family,
                "baseline": candidate.baseline,
                "instructions": candidate.instructions,
                "score": candidate_score,
                "baseline_score": baseline_score,
                "score_delta": candidate_score - baseline_score,
                "run_count": len(candidate_runs),
                "status": "complete",
            })

        # Also include baseline styles that weren't part of any candidate
        for baseline in baselines:
            if not any(r["name"] == baseline for r in results):
                baseline_runs = [
                    r for r in by_style.get(baseline, [])
                    if r.get("score_primary") is not None
                ]
                baseline_score = 0.0
                if baseline_runs:
                    baseline_score = sum(r["score_primary"] for r in baseline_runs) / len(baseline_runs)
                results.append({
                    "name": baseline,
                    "task_family": "",
                    "baseline": None,
                    "instructions": "Baseline style",
                    "score": baseline_score,
                    "baseline_score": None,
                    "score_delta": None,
                    "run_count": len(baseline_runs),
                    "status": "baseline",
                })

        return results

    def _save_results(
        self,
        output_dir: str,
        run_results: list[dict[str, Any]],
        candidates: list[TemplateProposal],
    ) -> None:
        """Save optimization results to JSON files in the output directory."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save all run results
        runs_file = out_path / "optimization_runs.json"
        runs_file.write_text(json.dumps(run_results, indent=2, default=str))

        # Save candidate summaries
        summaries = self._compute_candidate_results(
            run_results,
            [r.baseline for r in candidates],
            candidates,
        )
        summary_file = out_path / "optimization_summary.json"
        summary_file.write_text(json.dumps(summaries, indent=2))

    def generate_report(
        self,
        analysis: PromptAnalysis,
        candidate_results: list[dict[str, Any]],
    ) -> str:
        """Generate a markdown report from analysis and candidate results.

        Args:
            analysis: PromptAnalysis from existing data.
            candidate_results: Results from _compute_candidate_results().

        Returns:
            Markdown report string.
        """
        lines: list[str] = []
        lines.append("## Prompt Optimization Report")
        lines.append("")

        # Section 1: Analysis Summary
        lines.append("### Analysis Summary")
        lines.append("")
        lines.append(f"Total runs analyzed: {analysis.total_style_runs}")
        lines.append(f"Styles found: {', '.join(analysis.all_styles) if analysis.all_styles else 'none'}")
        lines.append(f"Best style overall: **{analysis.best_style_overall or 'N/A'}**")
        lines.append("")

        if analysis.insufficient_data:
            lines.append(
                f"⚠️ Task families with insufficient data (< 3 runs per style): "
                f"{', '.join(analysis.insufficient_data)}"
            )
            lines.append("")

        # Section 2: Best Styles Per Family
        if analysis.family_rankings:
            lines.append("### Best Styles Per Task Family")
            lines.append("")
            lines.append("| Task Family | Best Style | Avg Score | Margin |")
            lines.append("|---|---|---|---|")
            for family, rankings in sorted(analysis.family_rankings.items()):
                if rankings:
                    best = rankings[0]
                    lines.append(
                        f"| {family} | {best[0]} | {best[1]:.3f} | {best[2]:.3f} |"
                    )
            lines.append("")

        # Section 3: Style Score Variance
        if analysis.style_variances:
            lines.append("### Score Variance by Style")
            lines.append("")
            lines.append("Higher variance means the style's performance is more task-dependent.")
            lines.append("")
            lines.append("| Style | Variance |")
            lines.append("|---|---|")
            for style, var in sorted(analysis.style_variances.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {style} | {var:.4f} |")
            lines.append("")

        # Section 4: Candidate Results
        if candidate_results:
            lines.append("### Candidate Template Results")
            lines.append("")

            # Separate candidates by status
            complete = [r for r in candidate_results if r.get("status") == "complete"]
            analyzed = [r for r in candidate_results if r.get("status") == "analyzed"]
            no_data = [r for r in candidate_results if r.get("status") == "no_data"]
            baselines = [r for r in candidate_results if r.get("status") == "baseline"]

            if complete:
                lines.append("#### Live Run Results")
                lines.append("")
                lines.append("| Candidate | Score | Baseline | Delta | Runs |")
                lines.append("|---|---|---|---|---|")
                for r in sorted(complete, key=lambda x: x.get("score_delta", 0), reverse=True):
                    delta_str = f"{r['score_delta']:+.3f}" if r.get("score_delta") is not None else "N/A"
                    baseline_str = r.get('baseline_score', '')
                    if baseline_str is not None:
                        baseline_str = f"{baseline_str:.3f}"
                    lines.append(
                        f"| {r['name']} | {r.get('score', 0):.3f} | {baseline_str} "
                        f"({r.get('baseline', '-')}) | {delta_str} | {r.get('run_count', 0)} |"
                    )
                lines.append("")

                # Recommendations for candidates with positive delta
                recommended = [r for r in complete if r.get("score_delta", 0) > 0.05]
                if recommended:
                    lines.append("**Recommendations:**")
                    lines.append("")
                    for r in recommended:
                        lines.append(
                            f"- `{r['name']}` scores **{r['score_delta']:+.3f}** vs "
                            f"baseline `{r['baseline']}` ({r.get('instructions', '')})"
                        )
                    lines.append("")
                else:
                    lines.append(
                        "**Note:** No candidates exceeded the 0.05 improvement threshold. "
                        "Current baselines are performing well."
                    )
                    lines.append("")

            if analyzed:
                lines.append("#### Analyzed from Existing Data")
                lines.append("")
                lines.append("| Candidate | Score | Baseline | Delta | Runs |")
                lines.append("|---|---|---|---|---|")
                for r in sorted(analyzed, key=lambda x: x.get("score_delta", 0), reverse=True):
                    delta_str = f"{r['score_delta']:+.3f}" if r.get("score_delta") is not None else "N/A"
                    baseline_str = f"{r.get('baseline_score', 0):.3f}" if r.get('baseline_score') is not None else "N/A"
                    lines.append(
                        f"| {r['name']} | {r['score']:.3f} | {baseline_str} "
                        f"({r.get('baseline', '-')}) | {delta_str} | {r.get('run_count', 0)} |"
                    )
                lines.append("")

            if no_data:
                lines.append("#### Candidates with No Data")
                lines.append("")
                for r in no_data:
                    lines.append(f"- `{r['name']}` — no runs found for this style")
                lines.append("")

            if baselines:
                lines.append("#### Baseline Styles")
                lines.append("")
                lines.append("| Style | Avg Score | Runs |")
                lines.append("|---|---|---|")
                for r in sorted(baselines, key=lambda x: x.get("score", 0), reverse=True):
                    lines.append(
                        f"| {r['name']} | {r.get('score', 0):.3f} | {r.get('run_count', 0)} |"
                    )
                lines.append("")

        return "\n".join(lines)
