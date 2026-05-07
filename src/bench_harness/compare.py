"""Compare two benchmark runs and detect regressions."""

from __future__ import annotations

import statistics
from typing import Any

from bench_harness.reports.helpers import detect_regressions
from bench_harness.storage.sqlite import SQLiteStore


def compare_runs(
    baseline_db: str,
    candidate_db: str,
    score_threshold: float = 0.05,
    tps_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare two benchmark runs and detect quality/performance regressions.

    Groups by (task_id, model_alias) and compares average scores and timing
    metrics between baseline and candidate runs.

    Args:
        baseline_db: Path to baseline benchmark.db.
        candidate_db: Path to candidate benchmark.db.
        score_threshold: Minimum absolute score change to flag (default 0.05).
        tps_threshold: Relative TPS change threshold (default 0.1 = 10%).

    Returns:
        Dict with keys:
            quality_regressions: list of dicts with task_id, model, old/new scores, delta
            quality_improvements: list of dicts (same structure)
            performance_regressions: list of dicts with timing regressions
            performance_improvements: list of dicts with timing improvements
            crash_changes: list of dicts where model crashed in one but not the other
    """
    baseline_store = SQLiteStore(baseline_db)
    candidate_store = SQLiteStore(candidate_db)

    baseline_runs = baseline_store.get_runs()
    candidate_runs = candidate_store.get_runs()

    # Aggregate baseline by (task_id, model_alias)
    baseline_agg: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in baseline_runs:
        key = (run.get("task_id", "unknown"), run.get("model_alias", "unknown"))
        if key not in baseline_agg:
            baseline_agg[key] = []
        baseline_agg[key].append(run)

    # Aggregate candidate by (task_id, model_alias)
    candidate_agg: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in candidate_runs:
        key = (run.get("task_id", "unknown"), run.get("model_alias", "unknown"))
        if key not in candidate_agg:
            candidate_agg[key] = []
        candidate_agg[key].append(run)

    # All (task_id, model_alias) keys across both runs
    all_keys = set(baseline_agg.keys()) | set(candidate_agg.keys())

    quality_regressions = []
    quality_improvements = []
    performance_regressions = []
    performance_improvements = []
    crash_changes = []

    for key in sorted(all_keys):
        task_id, model_alias = key
        baseline_list = baseline_agg.get(key, [])
        candidate_list = candidate_agg.get(key, [])

        if not baseline_list or not candidate_list:
            # Crash change: present in one, absent in other
            if baseline_list and not candidate_list:
                baseline_scores = [r.get("score_primary") for r in baseline_list if r.get("score_primary") is not None]
                baseline_has_error = any(r.get("exit_status") == "error" for r in baseline_list)
                if baseline_has_error:
                    crash_changes.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "status": "baseline_crashed, candidate_succeeded",
                    })
                elif baseline_scores:
                    crash_changes.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "status": "baseline_completed, candidate_missing",
                    })
            elif candidate_list and not baseline_list:
                candidate_scores = [r.get("score_primary") for r in candidate_list if r.get("score_primary") is not None]
                candidate_has_error = any(r.get("exit_status") == "error" for r in candidate_list)
                if candidate_has_error:
                    crash_changes.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "status": "candidate_crashed, baseline_succeeded",
                    })
                elif candidate_scores:
                    crash_changes.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "status": "candidate_completed, baseline_missing",
                    })
            continue

        # Compare quality scores using detect_regressions helper
        score_regressions = detect_regressions(
            candidate_list, baseline_list, tolerance=score_threshold,
        )
        for sr in score_regressions:
            sr["task_id"] = task_id
            sr["model_alias"] = model_alias
            if sr["status"] == "REGRESSION":
                quality_regressions.append(sr)
            else:
                quality_improvements.append(sr)

        # Compare performance metrics
        baseline_wall = []
        baseline_tps = []
        candidate_wall = []
        candidate_tps = []

        for r in baseline_list:
            wall = r.get("total_wall_ms")
            if wall is not None and wall > 0:
                baseline_wall.append(wall)
            tps = r.get("tokens_per_second")
            if tps is not None and tps > 0:
                baseline_tps.append(tps)

        for r in candidate_list:
            wall = r.get("total_wall_ms")
            if wall is not None and wall > 0:
                candidate_wall.append(wall)
            tps = r.get("tokens_per_second")
            if tps is not None and tps > 0:
                candidate_tps.append(tps)

        if baseline_wall and candidate_wall:
            baseline_avg_wall = statistics.mean(baseline_wall)
            candidate_avg_wall = statistics.mean(candidate_wall)
            if baseline_avg_wall > 0:
                wall_change = (candidate_avg_wall - baseline_avg_wall) / baseline_avg_wall
                if wall_change > tps_threshold:
                    performance_regressions.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "baseline_wall_ms": round(baseline_avg_wall, 2),
                        "candidate_wall_ms": round(candidate_avg_wall, 2),
                        "change_pct": round(wall_change * 100, 2),
                        "metric": "wall_time",
                    })
                elif wall_change < -tps_threshold:
                    performance_improvements.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "baseline_wall_ms": round(baseline_avg_wall, 2),
                        "candidate_wall_ms": round(candidate_avg_wall, 2),
                        "change_pct": round(wall_change * 100, 2),
                        "metric": "wall_time",
                    })

        if baseline_tps and candidate_tps:
            baseline_avg_tps = statistics.mean(baseline_tps)
            candidate_avg_tps = statistics.mean(candidate_tps)
            if baseline_avg_tps > 0:
                tps_change = (candidate_avg_tps - baseline_avg_tps) / baseline_avg_tps
                if tps_change < -tps_threshold:
                    performance_regressions.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "baseline_tps": round(baseline_avg_tps, 2),
                        "candidate_tps": round(candidate_avg_tps, 2),
                        "change_pct": round(tps_change * 100, 2),
                        "metric": "tokens_per_second",
                    })
                elif tps_change > tps_threshold:
                    performance_improvements.append({
                        "task_id": task_id,
                        "model_alias": model_alias,
                        "baseline_tps": round(baseline_avg_tps, 2),
                        "candidate_tps": round(candidate_avg_tps, 2),
                        "change_pct": round(tps_change * 100, 2),
                        "metric": "tokens_per_second",
                    })

    return {
        "quality_regressions": quality_regressions,
        "quality_improvements": quality_improvements,
        "performance_regressions": performance_regressions,
        "performance_improvements": performance_improvements,
        "crash_changes": crash_changes,
    }


def format_comparison_output(results: dict[str, Any]) -> str:
    """Format comparison results as a human-readable string.

    Args:
        results: Dict returned by compare_runs().

    Returns:
        Formatted string for display.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    output_lines = []

    # Quality regressions
    if results["quality_regressions"]:
        table = Table(title="Quality Regressions")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Old Score", justify="right")
        table.add_column("New Score", justify="right")
        table.add_column("Delta", justify="right")
        for r in results["quality_regressions"]:
            table.add_row(
                r["task_id"],
                r["model_alias"],
                f"{r['old_score']:.3f}",
                f"{r['new_score']:.3f}",
                f"{r['delta']:+.3f}",
            )
        console.print(table)

    # Quality improvements
    if results["quality_improvements"]:
        table = Table(title="Quality Improvements")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Old Score", justify="right")
        table.add_column("New Score", justify="right")
        table.add_column("Delta", justify="right")
        for r in results["quality_improvements"]:
            table.add_row(
                r["task_id"],
                r["model_alias"],
                f"{r['old_score']:.3f}",
                f"{r['new_score']:.3f}",
                f"{r['delta']:+.3f}",
            )
        console.print(table)

    # Performance regressions
    if results["performance_regressions"]:
        table = Table(title="Performance Regressions")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Metric", style="yellow")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Change", justify="right")
        for r in results["performance_regressions"]:
            if r["metric"] == "wall_time":
                table.add_row(
                    r["task_id"],
                    r["model_alias"],
                    "wall_time",
                    f"{r['baseline_wall_ms']:.0f}ms",
                    f"{r['candidate_wall_ms']:.0f}ms",
                    f"+{r['change_pct']:.1f}%",
                )
            else:
                table.add_row(
                    r["task_id"],
                    r["model_alias"],
                    "tokens/sec",
                    f"{r['baseline_tps']:.1f}",
                    f"{r['candidate_tps']:.1f}",
                    f"{r['change_pct']:+.1f}%",
                )
        console.print(table)

    # Performance improvements
    if results["performance_improvements"]:
        table = Table(title="Performance Improvements")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Metric", style="green")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Change", justify="right")
        for r in results["performance_improvements"]:
            if r["metric"] == "wall_time":
                table.add_row(
                    r["task_id"],
                    r["model_alias"],
                    "wall_time",
                    f"{r['baseline_wall_ms']:.0f}ms",
                    f"{r['candidate_wall_ms']:.0f}ms",
                    f"{r['change_pct']:.1f}%",
                )
            else:
                table.add_row(
                    r["task_id"],
                    r["model_alias"],
                    "tokens/sec",
                    f"{r['baseline_tps']:.1f}",
                    f"{r['candidate_tps']:.1f}",
                    f"{r['change_pct']:+.1f}%",
                )
        console.print(table)

    # Crash changes
    if results["crash_changes"]:
        table = Table(title="Crash Changes")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Change", style="yellow")
        for r in results["crash_changes"]:
            table.add_row(
                r["task_id"],
                r["model_alias"],
                r["status"],
            )
        console.print(table)

    # Summary
    total_quality = len(results["quality_regressions"]) + len(results["quality_improvements"])
    total_perf = len(results["performance_regressions"]) + len(results["performance_improvements"])
    total_crash = len(results["crash_changes"])

    console.print()
    console.print(
        f"[cyan]Summary:[/cyan] "
        f"{len(results['quality_regressions'])} quality regression(s), "
        f"{len(results['quality_improvements'])} quality improvement(s), "
        f"{len(results['performance_regressions'])} performance regression(s), "
        f"{len(results['performance_improvements'])} performance improvement(s), "
        f"{total_crash} crash change(s) "
        f"across {total_quality + total_perf + total_crash} comparison(s)."
    )

    return str(console.file) if hasattr(console, "file") else ""
