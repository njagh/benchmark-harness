"""Comparison routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from bench_harness.server.models.schemas import APIResponse
from bench_harness.server.services.compare_service import compare_run_dbs
from bench_harness.server.services.runner_service import list_completed_runs

compare_bp = Blueprint("compare", __name__, url_prefix="/api/compare")


@compare_bp.route("", methods=["GET"])
def list_runs_for_compare():
    """List runs that can be selected for comparison."""
    runs = list_completed_runs(current_app.storage_config.root, limit=100)
    return jsonify(APIResponse.ok(runs).model_dump())


@compare_bp.route("", methods=["POST"])
def compare_runs():
    """Compare two benchmark runs."""
    data = request.get_json(force=True)
    baseline_id = data.get("baseline_run_id", "")
    candidate_id = data.get("candidate_run_id", "")
    score_threshold = data.get("score_threshold", 0.05)
    tps_threshold = data.get("tps_threshold", 0.1)

    if not baseline_id or not candidate_id:
        return jsonify(APIResponse.error("Both baseline_run_id and candidate_run_id required").model_dump()), 400

    baseline_db = _find_run_db(baseline_id)
    candidate_db = _find_run_db(candidate_id)

    if not baseline_db:
        return jsonify(APIResponse.error(f"Baseline run not found: {baseline_id}").model_dump()), 404
    if not candidate_db:
        return jsonify(APIResponse.error(f"Candidate run not found: {candidate_id}").model_dump()), 404

    result = compare_run_dbs(baseline_db, candidate_db, score_threshold, tps_threshold)
    return jsonify(APIResponse.ok(result).model_dump())


def _find_run_db(run_id: str) -> str | None:
    """Find the benchmark.db file for a run ID."""
    from pathlib import Path
    results_runs = current_app.storage_config.results_runs
    for date_dir in results_runs.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if not run_dir.is_dir():
                continue
            db_path = run_dir / "benchmark.db"
            if db_path.exists():
                if run_id in run_dir.name or run_id == run_dir.name:
                    return str(db_path)
    return None
