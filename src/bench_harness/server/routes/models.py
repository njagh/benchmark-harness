"""Metadata routes (models, suites, scorers, rubrics)."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

from bench_harness.server.models.schemas import APIResponse
from bench_harness.server.services.metadata_service import (
    list_models,
    list_suites,
    list_scorers,
    list_rubrics,
    discover_task_families,
    list_prompt_styles,
)

models_bp = Blueprint("metadata", __name__, url_prefix="/api")


@models_bp.route("/models", methods=["GET"])
def get_models():
    """List all configured models."""
    return jsonify(APIResponse.ok(list_models()).model_dump())


@models_bp.route("/suites", methods=["GET"])
def get_suites():
    """List all available suites."""
    return jsonify(APIResponse.ok(list_suites()).model_dump())


@models_bp.route("/scorers", methods=["GET"])
def get_scorers():
    """List all available scorers."""
    return jsonify(APIResponse.ok(list_scorers()).model_dump())


@models_bp.route("/rubrics", methods=["GET"])
def get_rubrics():
    """List all available rubrics."""
    return jsonify(APIResponse.ok(list_rubrics()).model_dump())


@models_bp.route("/task-families", methods=["GET"])
def get_task_families():
    """Discover task families from task directories."""
    task_dir = request.args.get("task_dir") if hasattr(request, 'args') else None
    return jsonify(APIResponse.ok(discover_task_families(task_dir)).model_dump())


@models_bp.route("/prompt-styles", methods=["GET"])
def get_prompt_styles():
    """List available prompt styles."""
    return jsonify(APIResponse.ok(list_prompt_styles()).model_dump())


@models_bp.route("/storage-info", methods=["GET"])
def get_storage_info():
    """Return resolved storage configuration."""
    return jsonify(APIResponse.ok(current_app.storage_info).model_dump())
