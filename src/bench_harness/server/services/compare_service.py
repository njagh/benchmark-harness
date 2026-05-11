"""Service for comparing benchmark runs."""

from __future__ import annotations

from typing import Any

from bench_harness.compare import compare_runs
from bench_harness.storage.sqlite import SQLiteStore


def compare_run_dbs(
    baseline_db: str,
    candidate_db: str,
    score_threshold: float = 0.05,
    tps_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare two benchmark databases and return structured results."""
    result = compare_runs(baseline_db, candidate_db, score_threshold, tps_threshold)

    # Add summaries
    baseline_store = SQLiteStore(baseline_db)
    candidate_store = SQLiteStore(candidate_db)

    baseline_summary = baseline_store.get_run_summary()
    candidate_summary = candidate_store.get_run_summary()

    return {
        "quality_regressions": [
            {
                "task_id": r.get("task_id", ""),
                "model_alias": r.get("model_alias", ""),
                "old_score": r.get("old_score"),
                "new_score": r.get("new_score"),
                "delta": r.get("delta", 0),
                "status": "REGRESSION",
                "risk": _risk_level(r.get("delta", 0), r.get("old_score", 1)),
            }
            for r in result.get("quality_regressions", [])
        ],
        "quality_improvements": [
            {
                "task_id": r.get("task_id", ""),
                "model_alias": r.get("model_alias", ""),
                "old_score": r.get("old_score"),
                "new_score": r.get("new_score"),
                "delta": r.get("delta", 0),
                "status": "IMPROVEMENT",
                "risk": _risk_level(r.get("delta", 0), r.get("old_score", 1)),
            }
            for r in result.get("quality_improvements", [])
        ],
        "performance_regressions": [
            {
                "task_id": r.get("task_id", ""),
                "model_alias": r.get("model_alias", ""),
                "baseline_value": r.get("baseline_wall_ms") or r.get("baseline_tps"),
                "candidate_value": r.get("candidate_wall_ms") or r.get("candidate_tps"),
                "change_pct": r.get("change_pct", 0),
                "metric": r.get("metric", "wall_time"),
            }
            for r in result.get("performance_regressions", [])
        ],
        "performance_improvements": [
            {
                "task_id": r.get("task_id", ""),
                "model_alias": r.get("model_alias", ""),
                "baseline_value": r.get("baseline_wall_ms") or r.get("baseline_tps"),
                "candidate_value": r.get("candidate_wall_ms") or r.get("candidate_tps"),
                "change_pct": r.get("change_pct", 0),
                "metric": r.get("metric", "wall_time"),
            }
            for r in result.get("performance_improvements", [])
        ],
        "crash_changes": [
            {
                "task_id": r.get("task_id", ""),
                "model_alias": r.get("model_alias", ""),
                "status": r.get("status", ""),
            }
            for r in result.get("crash_changes", [])
        ],
        "baseline_summary": _compute_summary(baseline_summary),
        "candidate_summary": _compute_summary(candidate_summary),
    }


def _risk_level(delta: float, old_score: float | None) -> str:
    """Compute risk level based on score delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta >= 0.2:
        return "high"
    elif abs_delta >= 0.1:
        return "medium"
    return "low"


def _compute_summary(summary_rows: list[dict]) -> dict:
    """Compute aggregate summary from SQLiteStore.get_run_summary()."""
    if not summary_rows:
        return {"model_alias": "none", "tasks_run": 0, "passed": 0, "failed": 0, "avg_score": None}

    total_tasks = sum(r.get("tasks_run", 0) for r in summary_rows)
    total_passed = sum(r.get("passed", 0) for r in summary_rows)
    total_failed = sum(r.get("failed", 0) for r in summary_rows)
    avg_ttft = sum(r.get("avg_ttft_ms", 0) for r in summary_rows) / max(len(summary_rows), 1)
    avg_score = sum(r.get("avg_score", 0) for r in summary_rows) / max(len(summary_rows), 1)

    return {
        "models": [r.get("model_alias", "") for r in summary_rows],
        "tasks_run": total_tasks,
        "passed": total_passed,
        "failed": total_failed,
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_score": round(avg_score, 3),
    }
