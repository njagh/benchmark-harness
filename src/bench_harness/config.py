"""Configuration loader for models, suites, and datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_config_dir() -> Path:
    """Find the configs/ directory."""
    root = _project_root()
    config_dir = root / "configs"
    if config_dir.exists():
        return config_dir
    # Fallback: look in cwd
    cwd_config = Path.cwd() / "configs"
    if cwd_config.exists():
        return cwd_config
    raise FileNotFoundError(
        "Cannot find configs/ directory. Expected at project root."
    )


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = _find_config_dir() / filepath
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: str | None = None) -> dict[str, Any]:
    """Load and return configs/models.yaml."""
    config_path = path or "models.yaml"
    data = load_yaml(config_path)
    if "models" not in data:
        raise ValueError("models.yaml must contain a 'models' key")
    return data


def get_model(config: dict[str, Any], alias: str) -> dict[str, Any] | None:
    """Look up a model config by alias. Returns None if not found."""
    models = config.get("models", {})
    return models.get(alias)


def load_suite_config(path: str | None = None) -> dict[str, Any]:
    """Load and return configs/suites.yaml."""
    config_path = path or "suites.yaml"
    data = load_yaml(config_path)
    if "suites" not in data:
        raise ValueError("suites.yaml must contain a 'suites' key")
    return data


def get_suite(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Look up a suite config by name. Returns None if not found.

    Suite dict may include: description, task_dir, families (list),
    max_concurrency, default_runs, default_temperature.
    """
    suites = config.get("suites", {})
    return suites.get(name)


def load_dataset_config(path: str | None = None) -> dict[str, Any]:
    """Load and return configs/datasets.yaml."""
    config_path = path or "datasets.yaml"
    data = load_yaml(config_path)
    # datasets key is optional — file may be a stub
    return data


def get_dataset(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Look up a dataset config by name. Returns None if not found."""
    datasets = config.get("datasets", {})
    return datasets.get(name)


def resolve_task_dir(suite_name: str) -> Path:
    """Resolve the task directory path for a given suite.

    Looks up the suite config and returns the absolute path to the task_dir.
    Path is resolved relative to the project root.
    """
    suite_config = load_suite_config()
    suite = get_suite(suite_config, suite_name)
    if suite is None:
        raise ValueError(f"Suite '{suite_name}' not found in suites.yaml")

    task_dir_str = suite.get("task_dir", "tasks/smoke")
    task_dir = _project_root() / task_dir_str
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return task_dir
