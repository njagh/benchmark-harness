"""Storage resolution and safety helpers for the web server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from bench_harness.storage.config import StorageConfig
from bench_harness.storage.safety import check_storage_root


def resolve_storage_config(
    storage_root: Optional[str] = None,
    allow_unsafe: bool = False,
) -> StorageConfig:
    """Resolve storage config with priority: explicit > env > project > default."""
    if storage_root:
        return StorageConfig.from_cli(Path(storage_root), allow_unsafe=allow_unsafe)
    return StorageConfig.from_env(allow_unsafe=allow_unsafe)


def get_storage_info(storage_config: StorageConfig) -> dict:
    """Return a dict of resolved storage paths for the UI."""
    return {
        "root": str(storage_config.root),
        "artifacts_root": str(storage_config.artifacts_root),
        "results_root": str(storage_config.results_root),
        "results_runs": str(storage_config.results_runs),
        "registry_root": str(storage_config.registry_root),
        "logs_root": str(storage_config.logs_root),
        "resolved_from": _resolve_source(),
    }


def _resolve_source() -> str:
    """Return which source the storage root was resolved from."""
    if os.environ.get("LLM_BENCH_STORAGE_ROOT"):
        return "env_var"
    # Check project config
    try:
        from bench_harness.storage.config import _find_project_config
        if _find_project_config():
            return "project_config"
    except Exception:
        pass
    return "default"
