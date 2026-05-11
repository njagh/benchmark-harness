"""Results viewing routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request, send_file

from bench_harness.server.models.schemas import APIResponse
from bench_harness.server.services.runner_service import get_run_results, get_run_summary, get_timing_summary

results_bp = Blueprint("results", __name__, url_prefix="/api/results")


@results_bp.route("/runs/<run_id>", methods=["GET"])
def get_run_results_by_id(run_id: str):
    """Get all results for a specific run."""
    # Search for the database
    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    results = get_run_results(db_path)
    return jsonify(APIResponse.ok(results).model_dump())


@results_bp.route("/runs/<run_id>/summary", methods=["GET"])
def get_run_summary_by_id(run_id: str):
    """Get summary for a specific run."""
    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    summary = get_run_summary(db_path)
    return jsonify(APIResponse.ok(summary).model_dump())


@results_bp.route("/runs/<run_id>/timing", methods=["GET"])
def get_run_timing(run_id: str):
    """Get timing summary for a specific run."""
    model = request.args.get("model")
    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    timing = get_timing_summary(db_path, model_alias=model)
    return jsonify(APIResponse.ok(timing).model_dump())


@results_bp.route("/runs/<run_id>/report", methods=["GET"])
def get_run_report(run_id: str):
    """Serve the markdown report for a run."""
    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    report_path = str(db_path).replace("benchmark.db", "report.md")
    if not report_path.endswith(".md"):
        parts = str(db_path).rsplit("/", 1)
        report_path = parts[0] + "/report.md" if len(parts) > 1 else "runs/report.md"

    import os
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=f"report-{run_id}.md")
    return jsonify(APIResponse.error("Report not found").model_dump()), 404


@results_bp.route("/runs/<run_id>/runs.jsonl", methods=["GET"])
def get_run_artifacts(run_id: str):
    """Serve the JSONL artifacts for a run."""
    import os
    parts = str(_find_run_db(run_id) or "").rsplit("/", 1)
    run_dir = parts[0] if len(parts) > 1 else "runs"
    jsonl_path = run_dir + "/runs.jsonl"

    if os.path.exists(jsonl_path):
        return send_file(jsonl_path, as_attachment=True, download_name=f"runs-{run_id}.jsonl")
    return jsonify(APIResponse.error("Artifacts not found").model_dump()), 404


@results_bp.route("/runs/<run_id>/export/<export_type>", methods=["GET"])
def export_run_data(run_id: str, export_type: str):
    """Export run data in various formats."""
    from bench_harness.server.services.export_service import (
        export_sft_data,
        export_preference_data,
        export_regression_data,
        export_judge_data,
    )

    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    suite_id = request.args.get("suite_id", "smoke")
    min_score = request.args.get("min_score", 0.0, type=float)
    min_margin = request.args.get("min_margin", 0.1, type=float)

    exporters = {
        "sft": lambda: export_sft_data(db_path, suite_id, min_score=min_score),
        "preference": lambda: export_preference_data(db_path, suite_id, min_margin=min_margin),
        "regression": lambda: export_regression_data(db_path, suite_id),
        "judge": lambda: export_judge_data(db_path, suite_id),
    }

    exporter = exporters.get(export_type)
    if not exporter:
        return jsonify(APIResponse.error(f"Unknown export type: {export_type}").model_dump()), 400

    try:
        output_path = exporter()
        return jsonify(APIResponse.ok({"path": output_path, "export_type": export_type}).model_dump())
    except Exception as e:
        return jsonify(APIResponse.error(f"Export failed: {e}").model_dump()), 500


def _find_run_db(run_id: str) -> str | None:
    """Find the benchmark.db file for a run ID."""
    import os
    from pathlib import Path

    # Direct match first
    results_runs = current_app.storage_config.results_runs
    for date_dir in results_runs.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if not run_dir.is_dir():
                continue
            db_path = run_dir / "benchmark.db"
            if db_path.exists():
                # Check if run_id matches or is a substring
                if run_id in run_dir.name or run_id == run_dir.name:
                    return str(db_path)
    return None
