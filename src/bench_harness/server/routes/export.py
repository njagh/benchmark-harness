"""Export routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request, send_file

from bench_harness.server.models.schemas import APIResponse
from bench_harness.server.services.export_service import (
    export_sft_data,
    export_preference_data,
    export_regression_data,
    export_judge_data,
)

export_bp = Blueprint("export", __name__, url_prefix="/api/export")


@export_bp.route("/<export_type>", methods=["POST"])
def trigger_export(export_type: str):
    """Trigger an export for a given run."""
    data = request.get_json(force=True)
    run_id = data.get("run_id", "")
    suite_id = data.get("suite_id", "smoke")
    min_score = data.get("min_score", 0.0)
    min_margin = data.get("min_margin", 0.1)

    if not run_id:
        return jsonify(APIResponse.error("run_id required").model_dump()), 400

    db_path = _find_run_db(run_id)
    if not db_path:
        return jsonify(APIResponse.error("Run results not found").model_dump()), 404

    try:
        exporters = {
            "sft": lambda: export_sft_data(db_path, suite_id, min_score=min_score),
            "preference": lambda: export_preference_data(db_path, suite_id, min_margin=min_margin),
            "regression": lambda: export_regression_data(db_path, suite_id),
            "judge": lambda: export_judge_data(db_path, suite_id),
        }

        exporter = exporters.get(export_type)
        if not exporter:
            return jsonify(APIResponse.error(f"Unknown export type: {export_type}").model_dump()), 400

        output_path = exporter()
        # Send the file
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify(APIResponse.error(f"Export failed: {e}").model_dump()), 500


def _find_run_db(run_id: str) -> str | None:
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
