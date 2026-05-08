"""Storage configuration and root resolution for the benchmark harness.

Resolves storage root from multiple sources (CLI flag > env var > project
config > default) and provides namespace paths for artifacts, results,
registry, logs, cache, and tmp directories.
"""

from __future__ import annotations

import hashlib
import logging
import os
import datetime as dt
from pathlib import Path
from typing import Any

import yaml

from bench_harness.storage.safety import check_storage_root

logger = logging.getLogger(__name__)

# Default XDG data home fallback
_DEFAULT_XDG_DATA_HOME = Path.home() / ".local" / "share"
_DEFAULT_STORAGE_ROOT = _DEFAULT_XDG_DATA_HOME / "llm-bench"
_PROJECT_CONFIG_FILENAME = ".llm-bench.yaml"
_MIN_FREE_SPACE_GB = 10


class StorageConfig:
    """Encapsulates storage root resolution and namespace management.

    Resolution priority:
    1. ``from_cli(root)`` — explicit path
    2. ``LLM_BENCH_STORAGE_ROOT`` environment variable
    3. ``.llm-bench.yaml`` in cwd or parent dirs (project config)
    4. Default: ``~/.local/share/llm-bench`` (or ``$XDG_DATA_HOME/llm-bench``)

    Attributes:
        root: The resolved storage root path.
    """

    def __init__(
        self,
        root: Path,
        allow_unsafe: bool = False,
    ) -> None:
        self.root = root.resolve()
        self._allow_unsafe = allow_unsafe

    # ── namespace properties ──────────────────────────────────────

    @property
    def artifacts_root(self) -> Path:
        return self.root / "artifacts"

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    @property
    def registry_root(self) -> Path:
        return self.root / "registry"

    @property
    def logs_root(self) -> Path:
        return self.root / "logs"

    @property
    def cache_root(self) -> Path:
        return self.root / "cache"

    @property
    def tmp_root(self) -> Path:
        return self.root / "tmp"

    # ── artifact sub-namespaces ───────────────────────────────────

    @property
    def artifacts_models(self) -> Path:
        return self.artifacts_root / "models"

    @property
    def artifacts_engines(self) -> Path:
        return self.artifacts_root / "engines"

    @property
    def artifacts_tokenizers(self) -> Path:
        return self.artifacts_root / "tokenizers"

    @property
    def artifacts_calibration(self) -> Path:
        return self.artifacts_root / "calibration"

    @property
    def artifacts_runtime_builds(self) -> Path:
        return self.artifacts_root / "runtime-builds"

    # ── result sub-namespaces ─────────────────────────────────────

    @property
    def results_runs(self) -> Path:
        return self.results_root / "runs"

    @property
    def results_summaries(self) -> Path:
        return self.results_root / "summaries"

    @property
    def results_comparisons(self) -> Path:
        return self.results_root / "comparisons"

    # ── resolution class methods ──────────────────────────────────

    @classmethod
    def from_cli(cls, root: Path, allow_unsafe: bool = False) -> StorageConfig:
        """Resolve from an explicit CLI-provided path."""
        resolved = root.resolve()
        check_storage_root(resolved, allow_unsafe=allow_unsafe)
        return cls(root=resolved, allow_unsafe=allow_unsafe)

    @classmethod
    def from_env(cls, allow_unsafe: bool = False) -> StorageConfig:
        """Resolve from ``LLM_BENCH_STORAGE_ROOT`` env var or default."""
        env_root = os.environ.get("LLM_BENCH_STORAGE_ROOT")
        if env_root:
            resolved = Path(env_root).resolve()
            check_storage_root(resolved, allow_unsafe=allow_unsafe)
            return cls(root=resolved, allow_unsafe=allow_unsafe)

        default = _get_default_storage_root()
        check_storage_root(default, allow_unsafe=allow_unsafe)
        return cls(root=default, allow_unsafe=allow_unsafe)

    @classmethod
    def from_project(cls, allow_unsafe: bool = False) -> StorageConfig | None:
        """Resolve from ``.llm-bench.yaml`` in cwd or any parent directory.

        Returns ``None`` if no project config is found.
        """
        config_path = _find_project_config()
        if config_path is None:
            return None

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        project = data.get("project", {})
        if not project:
            return None

        root_str = project.get("default_storage_root")
        if not root_str:
            return None

        resolved = Path(root_str).resolve()
        check_storage_root(resolved, allow_unsafe=allow_unsafe)
        return cls(root=resolved, allow_unsafe=allow_unsafe)

    def ensure_namespaces(self) -> None:
        """Create all namespace directories if they do not exist."""
        for ns in (
            self.artifacts_root,
            self.results_root,
            self.registry_root,
            self.logs_root,
            self.cache_root,
            self.tmp_root,
        ):
            ns.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, run_name: str) -> Path:
        """Create an immutable per-run result directory.

        Directory layout:
        ``<results_root>/runs/<YYYY-MM-DD>/<name>__<ISO-timestamp>__<short-hash>/``
        """
        today = dt.datetime.now().strftime("%Y-%m-%d")
        timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        hash_input = f"{run_name}__{timestamp}".encode("utf-8")
        short_hash = hashlib.sha256(hash_input).hexdigest()[:8]

        run_dir_name = f"{run_name}__{timestamp}__{short_hash}"
        run_dir = self.results_runs / today / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created run directory: %s", run_dir)
        return run_dir

    def write_resolved_spec(self, spec: Any, run_dir: Path) -> Path:
        """Write a resolved run spec to a run directory as YAML.

        Returns the path to the written file.
        """
        spec_path = run_dir / "resolved_spec.yaml"
        if hasattr(spec, 'model_dump'):
            data = spec.model_dump(mode='python')
        else:
            data = spec
        spec_path.write_text(yaml.dump(data, default_flow_style=False))
        return spec_path


    def resolve_artifact(self, artifact_mode: str, artifact_path: str) -> Path:
        """Resolve an artifact path based on its mode.

        Returns the effective path for benchmarking.
        """
        from bench_harness.registry import manage_artifact
        from bench_harness.schemas.model_artifact import ModelArtifact, ArtifactMode, ArtifactKind

        kind = ArtifactKind.hf_checkpoint
        if artifact_path.startswith(('http://', 'https://')):
            kind = ArtifactKind.openai_endpoint

        art = ModelArtifact(
            artifact_id=f"temp-{kind.value}-{hash(artifact_path) & 0xFFFFFFFF:08x}",
            kind=kind,
            mode=ArtifactMode(artifact_mode),
            source_path=artifact_path,
        )
        return manage_artifact(art, self)


def _get_default_storage_root() -> Path:
    """Return the default storage root following XDG conventions."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "llm-bench"
    return _DEFAULT_STORAGE_ROOT


def _find_project_config() -> Path | None:
    """Search for ``.llm-bench.yaml`` in cwd and parent directories."""
    current = Path.cwd()
    while current != current.parent:
        candidate = current / _PROJECT_CONFIG_FILENAME
        if candidate.exists():
            return candidate
        current = current.parent
    return None
