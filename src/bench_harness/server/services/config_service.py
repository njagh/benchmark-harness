"""CRUD service for saved benchmark configs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench_harness.schemas import RunSpec
from bench_harness.server.models.schemas import SavedConfig


CONFIGS_DIR_NAME = "saved-configs"


def _configs_dir(storage_root: Path) -> Path:
    return storage_root / CONFIGS_DIR_NAME


def _config_path(storage_root: Path, config_id: str) -> Path:
    return _configs_dir(storage_root) / f"{config_id}.json"


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_count_for_config(storage_root: Path, config_id: str) -> int:
    """Count how many times this config has been used in runs."""
    results_runs = storage_root / "results" / "runs"
    if not results_runs.exists():
        return 0
    count = 0
    for date_dir in sorted(results_runs.iterdir()):
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            resolved = run_dir / "resolved_spec.yaml"
            if resolved.exists():
                try:
                    spec = RunSpec.from_yaml(resolved)
                    if spec.name == config_id:
                        count += 1
                except Exception:
                    pass
    return count


def list_configs(storage_root: Path) -> list[SavedConfig]:
    """List all saved configs, sorted by updated_at descending."""
    config_dir = _configs_dir(storage_root)
    if not config_dir.exists():
        return []

    configs = []
    for f in sorted(config_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if f.suffix == ".json" and f.is_file():
            try:
                data = json.loads(f.read_text())
                data["run_count"] = _run_count_for_config(storage_root, data["id"])
                cfg = SavedConfig(**data)
                configs.append(cfg)
            except Exception:
                pass
    return configs


def get_config(storage_root: Path, config_id: str) -> Optional[SavedConfig]:
    """Load a single config by ID."""
    path = _config_path(storage_root, config_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["run_count"] = _run_count_for_config(storage_root, config_id)
    return SavedConfig(**data)


def save_config(
    storage_root: Path,
    config: SavedConfig,
    is_update: bool = False,
) -> SavedConfig:
    """Save a config to disk. Creates a new ID if not set."""
    if not config.id:
        config.id = _make_id()
    config.updated_at = _now_iso()
    if not is_update:
        config.created_at = config.updated_at

    path = _config_path(storage_root, config.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
    return config


def delete_config(storage_root: Path, config_id: str) -> bool:
    """Delete a config file. Returns True if deleted."""
    path = _config_path(storage_root, config_id)
    if path.exists():
        path.unlink()
        return True
    return False


def build_run_spec(config: SavedConfig) -> RunSpec:
    """Convert a SavedConfig to a RunSpec for execution."""
    artifact_kwargs = {
        "kind": config.artifact.kind,
        "mode": config.artifact.mode,
        "path": config.artifact.path,
    }
    if config.artifact.tokenizer_path:
        artifact_kwargs["tokenizer_path"] = config.artifact.tokenizer_path
    if config.artifact.model_id:
        artifact_kwargs["model_id"] = config.artifact.model_id
    if config.artifact.quantization:
        artifact_kwargs["quantization"] = config.artifact.quantization

    runtime_kwargs = {
        "kind": config.runtime.kind,
        "launch": config.runtime.launch,
    }
    if config.runtime.host:
        runtime_kwargs["host"] = config.runtime.host
    if config.runtime.port:
        runtime_kwargs["port"] = config.runtime.port
    if config.runtime.model_name:
        runtime_kwargs["model_name"] = config.runtime.model_name
    if config.runtime.args:
        runtime_kwargs["args"] = config.runtime.args

    workload_kwargs = {
        "prompt_suite": config.workload.prompt_suite,
        "max_tokens": config.workload.max_tokens,
        "temperature": config.workload.temperature,
        "num_runs": config.workload.num_runs,
        "concurrency": config.workload.concurrency,
    }
    if config.workload.task_dir:
        workload_kwargs["task_dir"] = config.workload.task_dir

    return RunSpec(
        name=config.id or config.name,
        project=config.project,
        tags=config.tags,
        artifact=type("ArtifactSpec", (), artifact_kwargs)(),
        runtime=type("RuntimeSpec", (), runtime_kwargs)(),
        workload=type("WorkloadSpec", (), workload_kwargs)(),
        storage=type("StoragePolicy", (), {
            "artifact_policy": config.storage.artifact_policy,
            "result_policy": config.storage.result_policy,
        })(),
    )


def to_saved_config(spec: RunSpec) -> SavedConfig:
    """Convert a RunSpec to a SavedConfig for display/editing."""
    artifact = SavedConfig(
        artifact=type("ArtifactConfig", (), {
            "kind": spec.artifact.kind.value if hasattr(spec.artifact.kind, 'value') else str(spec.artifact.kind),
            "mode": spec.artifact.mode.value if hasattr(spec.artifact.mode, 'value') else str(spec.artifact.mode),
            "path": spec.artifact.path,
            "tokenizer_path": spec.artifact.tokenizer_path,
            "model_id": spec.artifact.model_id,
            "quantization": spec.artifact.quantization,
        })(),
        runtime=type("RuntimeConfig", (), {
            "kind": spec.runtime.kind.value if hasattr(spec.runtime.kind, 'value') else str(spec.runtime.kind),
            "launch": spec.runtime.launch.value if hasattr(spec.runtime.launch, 'value') else str(spec.runtime.launch),
            "host": spec.runtime.host,
            "port": spec.runtime.port,
            "model_name": spec.runtime.model_name,
            "args": spec.runtime.args,
        })(),
        workload=type("WorkloadConfig", (), {
            "prompt_suite": spec.workload.prompt_suite,
            "max_tokens": spec.workload.max_tokens,
            "temperature": spec.workload.temperature,
            "num_runs": spec.workload.num_runs,
            "concurrency": spec.workload.concurrency,
            "task_dir": spec.workload.task_dir,
        })(),
        storage=type("StoragePolicyConfig", (), {
            "artifact_policy": spec.storage.artifact_policy,
            "result_policy": spec.storage.result_policy,
        })(),
        name=spec.name,
        project=spec.project,
        tags=spec.tags,
    )
    return artifact
