"""Service for loading metadata (models, suites, scorers, rubrics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench_harness.config import (
    load_model_config,
    get_model,
    load_suite_config,
    get_suite,
    load_scorer_config,
    load_rubric_config,
    get_rubric,
)


def list_models() -> list[dict[str, Any]]:
    """Return list of model configs as dicts with alias."""
    try:
        config = load_model_config()
        models = config.get("models", {})
        result = []
        for alias, cfg in models.items():
            result.append({
                "alias": alias,
                **cfg,
            })
        return result
    except FileNotFoundError:
        return []


def get_model_by_alias(alias: str) -> dict[str, Any] | None:
    """Look up a specific model by alias."""
    try:
        config = load_model_config()
        return get_model(config, alias)
    except FileNotFoundError:
        return None


def list_suites() -> list[dict[str, Any]]:
    """Return list of suite configs as dicts with name."""
    try:
        config = load_suite_config()
        suites = config.get("suites", {})
        result = []
        for name, cfg in suites.items():
            result.append({
                "name": name,
                **cfg,
            })
        return result
    except FileNotFoundError:
        return []


def get_suite_by_name(name: str) -> dict[str, Any] | None:
    """Look up a specific suite by name."""
    try:
        config = load_suite_config()
        return get_suite(config, name)
    except FileNotFoundError:
        return None


def list_scorers() -> list[dict[str, Any]]:
    """Return list of scorer configs."""
    try:
        config = load_scorer_config()
        scorers = config.get("scorers", {})
        result = []
        for name, cfg in scorers.items():
            result.append({
                "name": name,
                **cfg,
            })
        return result
    except FileNotFoundError:
        return []


def list_rubrics() -> list[dict[str, Any]]:
    """Return list of rubric configs."""
    try:
        config = load_rubric_config()
        if isinstance(config, dict):
            result = []
            for name, cfg in config.items():
                if isinstance(cfg, dict) and "dimensions" in cfg:
                    result.append({
                        "name": name,
                        **cfg,
                    })
            return result
        return []
    except FileNotFoundError:
        return []


def discover_task_families(task_dir: str | None = None) -> list[dict[str, Any]]:
    """Discover task families from task directories."""
    families: dict[str, dict] = {}

    # Standard task directories
    task_dirs = [
        "tasks/smoke",
        "tasks/local_coding_agent_v1",
        "tasks/coding_benchmark",
        "tasks/public_baseline",
    ]
    if task_dir:
        task_dirs.insert(0, task_dir)

    for td in task_dirs:
        base = Path(td)
        if not base.exists():
            continue
        for family_dir in base.iterdir():
            if not family_dir.is_dir():
                continue
            task_files = list(family_dir.glob("*.yaml")) + list(family_dir.glob("*.yml"))
            family_name = family_dir.name
            if family_name not in families:
                families[family_name] = {"family": family_name, "count": 0, "tasks": [], "source": str(family_dir.parent)}
            families[family_name]["count"] += len(task_files)
            for tf in task_files:
                families[family_name]["tasks"].append(tf.stem)

    return sorted(families.values(), key=lambda x: x["family"])


def list_prompt_styles() -> list[str]:
    """List available prompt template styles."""
    try:
        from bench_harness.tasks.prompt_templates import list_styles
        return list_styles()
    except Exception:
        return ["plain", "step_by_step", "json_schema", "architect", "patch_only", "terse", "repl"]
