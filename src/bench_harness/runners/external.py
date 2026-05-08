from __future__ import annotations

from pathlib import Path

from bench_harness.runners.base import ProcessHandle, RuntimeRunner
from bench_harness.runners.openai_compatible import OpenAICompatibleRunner
from bench_harness.schemas.run_result import RunResult
from bench_harness.schemas.run_spec import RunSpec
from bench_harness.storage.config import StorageConfig


class ExternalRunner(OpenAICompatibleRunner):
    """Record commands and benchmark an external endpoint without owning lifecycle."""

    @property
    def kind(self) -> str:
        return "external"

    def prepare(self, spec: RunSpec) -> dict:
        prep = super().prepare(spec)
        prep["producing_command"] = spec.runtime.args.get("producing_command", "unknown")
        return prep
