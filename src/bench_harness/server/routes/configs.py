"""Config CRUD routes."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, jsonify, request

from bench_harness.server.services.config_service import (
    list_configs,
    get_config,
    save_config,
    delete_config,
    build_run_spec,
    to_saved_config,
)
from bench_harness.server.models.schemas import (
    SavedConfig,
    APIResponse,
    ConfigTemplate,
)

configs_bp = Blueprint("configs", __name__, url_prefix="/api/configs")


# ── Templates ──────────────────────────────────────────────────────

TEMPLATES: list[ConfigTemplate] = [
    ConfigTemplate(
        name="smoke",
        description="Quick smoke test — 1 run, 256 tokens",
        preset={
            "workload": {"prompt_suite": "smoke", "max_tokens": 256, "temperature": 0.0, "num_runs": 1, "concurrency": 1},
        },
    ),
    ConfigTemplate(
        name="full_benchmark",
        description="Full coding benchmark — 3 runs, judge enabled",
        preset={
            "workload": {"prompt_suite": "coding_smoke", "max_tokens": 512, "temperature": 0.0, "num_runs": 3, "concurrency": 1},
            "advanced": {"judge": True},
        },
    ),
    ConfigTemplate(
        name="style_comparison",
        description="Compare all prompt styles",
        preset={
            "workload": {"prompt_suite": "smoke", "max_tokens": 256, "temperature": 0.0, "num_runs": 1, "concurrency": 1},
            "advanced": {"styles": ["plain", "step_by_step", "json_schema", "architect", "patch_only", "terse", "repl"]},
        },
    ),
    ConfigTemplate(
        name="context_sweep",
        description="Test across all context sizes",
        preset={
            "workload": {"prompt_suite": "smoke", "max_tokens": 256, "temperature": 0.0, "num_runs": 1, "concurrency": 1},
            "advanced": {"context_sizes": ["small", "medium", "large", "xlarge"]},
        },
    ),
]


@configs_bp.route("/templates", methods=["GET"])
def list_templates():
    """List available config templates."""
    return jsonify([t.model_dump() for t in TEMPLATES])


@configs_bp.route("", methods=["GET"])
def get_configs():
    """List all saved configs."""
    config = current_app.storage_config
    configs = list_configs(config.root)
    return jsonify([c.model_dump() for c in configs])


@configs_bp.route("", methods=["POST"])
def create_config():
    """Create a new config."""
    data = request.get_json(force=True)
    config = SavedConfig(**data)
    result = save_config(current_app.storage_config.root, config)
    return jsonify(APIResponse.ok(result.model_dump(), "Config created").model_dump())


@configs_bp.route("/<config_id>", methods=["GET"])
def get_config_by_id(config_id: str):
    """Get a single config."""
    config = get_config(current_app.storage_config.root, config_id)
    if not config:
        return jsonify(APIResponse.error("Config not found").model_dump()), 404
    return jsonify(APIResponse.ok(config.model_dump()).model_dump())


@configs_bp.route("/<config_id>", methods=["PUT"])
def update_config(config_id: str):
    """Update an existing config."""
    existing = get_config(current_app.storage_config.root, config_id)
    if not existing:
        return jsonify(APIResponse.error("Config not found").model_dump()), 404

    data = request.get_json(force=True)
    data["id"] = config_id
    config = SavedConfig(**data)
    result = save_config(current_app.storage_config.root, config, is_update=True)
    return jsonify(APIResponse.ok(result.model_dump(), "Config updated").model_dump())


@configs_bp.route("/<config_id>", methods=["DELETE"])
def delete_config_by_id(config_id: str):
    """Delete a config."""
    if delete_config(current_app.storage_config.root, config_id):
        return jsonify(APIResponse.ok(message="Config deleted").model_dump())
    return jsonify(APIResponse.error("Config not found").model_dump()), 404


@configs_bp.route("/<config_id>/duplicate", methods=["POST"])
def duplicate_config(config_id: str):
    """Duplicate a config with a new ID."""
    existing = get_config(current_app.storage_config.root, config_id)
    if not existing:
        return jsonify(APIResponse.error("Config not found").model_dump()), 404

    # Deep copy and clear ID
    data = existing.model_dump()
    data.pop("id", None)
    data.pop("created_at", None)
    data.pop("updated_at", None)
    data.pop("last_run_at", None)
    data["name"] = data.get("name", "") + "-copy"

    new_config = SavedConfig(**data)
    result = save_config(current_app.storage_config.root, new_config)
    return jsonify(APIResponse.ok(result.model_dump(), "Config duplicated").model_dump())


@configs_bp.route("/<config_id>/spec", methods=["GET"])
def get_run_spec(config_id: str):
    """Get the RunSpec YAML for a config."""
    config = get_config(current_app.storage_config.root, config_id)
    if not config:
        return jsonify(APIResponse.error("Config not found").model_dump()), 404

    spec = build_run_spec(config)
    return jsonify({
        "yaml": spec.to_yaml(),
        "json": json.loads(spec.to_json()),
    })
