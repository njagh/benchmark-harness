"""Run management routes."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from bench_harness.server.services.runner_service import (
    launch_run,
    get_run_state,
    cancel_run,
    list_completed_runs,
    _active_runs,
)
from bench_harness.server.models.schemas import APIResponse

logger = logging.getLogger(__name__)

runs_bp = Blueprint("runs", __name__, url_prefix="/api/runs")


@runs_bp.route("", methods=["GET"])
def list_runs():
    """List completed runs."""
    limit = request.args.get("limit", 50, type=int)
    runs = list_completed_runs(current_app.storage_config.root, limit=limit)
    return jsonify([r for r in runs])


@runs_bp.route("", methods=["POST"])
def create_run():
    """Start a new benchmark run from a config."""
    data = request.get_json(force=True)

    config_id = data.get("config_id", "")
    if config_id:
        # Load config from saved configs
        from bench_harness.server.services.config_service import get_config
        config = get_config(current_app.storage_config.root, config_id)
        if not config:
            return jsonify(APIResponse.error("Config not found").model_dump()), 404
        config_data = config.model_dump()
    else:
        config_data = data

    result = launch_run(config_data, current_app.storage_config.root)
    return jsonify(APIResponse.ok(result).model_dump()), 202


@runs_bp.route("/active", methods=["GET"])
def list_active_runs():
    """List currently running benchmarks."""
    active = []
    for run_id, state in _active_runs.items():
        if state.status == "running":
            active.append({
                "run_id": run_id,
                "status": state.status,
                "model_alias": state.model_alias,
                "completed_tasks": state.completed_tasks,
                "total_tasks": state.total_tasks,
            })
    return jsonify(active)


@runs_bp.route("/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """Get the state of a run (active or completed)."""
    state = get_run_state(run_id)
    if not state:
        return jsonify(APIResponse.error("Run not found").model_dump()), 404
    return jsonify(APIResponse.ok(state).model_dump())


@runs_bp.route("/<run_id>/cancel", methods=["POST"])
def cancel_run_by_id(run_id: str):
    """Cancel a running benchmark."""
    if cancel_run(run_id):
        return jsonify(APIResponse.ok(message="Run cancelled").model_dump())
    return jsonify(APIResponse.error("Run not found or not running").model_dump()), 404


@runs_bp.route("/<run_id>/events", methods=["GET"])
def get_run_events(run_id: str):
    """Get events for a running benchmark."""
    state = get_run_state(run_id)
    if not state:
        return jsonify(APIResponse.error("Run not found").model_dump()), 404
    return jsonify(state.get("events", []))
