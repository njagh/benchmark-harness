"""High-level benchmark runner — orchestration layer.

Usage::

    from bench_harness import BenchmarkRunner, RunSpec, StorageConfig

    config = StorageConfig(root="/path/to/storage")
    runner = BenchmarkRunner(storage=config)
    spec = RunSpec.from_yaml("my-run.yaml")
    result = runner.run(spec)
    print(result.summary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bench_harness.hooks import ArtifactMetadataHook
from bench_harness.registry import ArtifactRegistry, manage_artifact
from bench_harness.runners import RUNNER_REGISTRY, RuntimeRunner, get_runner
from bench_harness.schemas import RunSpec, RunResult
from bench_harness.storage.config import StorageConfig
from bench_harness.storage.safety import check_storage_root

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """High-level benchmark orchestrator.

    Ties together storage configuration, artifact registration, runner
    selection, and result persistence into a single ``run()`` call.

    Args:
        storage: StorageConfig for result storage.
        hook: Optional ArtifactMetadataHook for extra run metadata.
        artifact_registry: Optional pre-configured ArtifactRegistry.
            If not provided, one is created from the storage config.
    """

    def __init__(
        self,
        storage: StorageConfig | None = None,
        hook: ArtifactMetadataHook | None = None,
        artifact_registry: ArtifactRegistry | None = None,
    ):
        self.storage = storage or StorageConfig()
        self.hook = hook
        self.registry = artifact_registry or ArtifactRegistry(self.storage)

    def run(self, spec: RunSpec) -> RunResult:
        """Run a benchmark according to the given spec.

        Steps:
        1. Resolve and register artifact.
        2. Select and configure the runner.
        3. Execute tasks.
        4. Write results to storage.

        Args:
            spec: Resolved run specification.

        Returns:
            RunResult with per-request data and summary.
        """
        # Step 1: Register artifact
        artifact = spec.artifact
        if artifact is not None:
            self.registry.register(artifact)
            managed_path = manage_artifact(artifact, self.storage)

            if self.hook and hasattr(self.hook, "on_artifact_registered"):
                self.hook.on_artifact_registered(artifact, managed_path)

        # Step 2: Select runner
        runtime = spec.runtime
        runner_kind = runtime.kind if runtime else "openai_compatible"

        if runner_kind not in RUNNER_REGISTRY:
            raise ValueError(
                f"Unknown runner kind: {runner_kind}. "
                f"Available: {list(RUNNER_REGISTRY.keys())}"
            )

        runner = get_runner(runner_kind, self.storage)

        # Step 3: Prepare runner
        if hasattr(runner, "prepare") and callable(runner.prepare):
            runner.prepare(spec)

        # Step 4: Run the workload
        result = runner.run_workload(spec)

        # Step 5: Apply hook enrichment
        if self.hook:
            result = self.hook.enrich_run_result(result, spec)

        # Step 6: Write results
        self._write_result(result, spec)

        return result

    def _write_result(self, result: RunResult, spec: RunSpec) -> None:
        """Write run result to storage."""
        try:
            check_storage_root(self.storage.root)
        except (ValueError, OSError) as e:
            logger.warning("Storage root check failed: %s", e)

        run_dir = self.storage.create_run_dir(spec.run.name or "run")
        result.write_to_directory(Path(run_dir))
        logger.info("Wrote result to %s", run_dir)
