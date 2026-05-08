from __future__ import annotations

from pathlib import Path

from bench_harness.runners.base import ProcessHandle, RuntimeRunner
from bench_harness.schemas.run_result import RunResult
from bench_harness.schemas.run_spec import RunSpec
from bench_harness.storage.config import StorageConfig


class LlamaCPPRunner(RuntimeRunner):
    """Launch and benchmark llama.cpp servers (stub)."""

    @property
    def kind(self) -> str:
        return "llamacpp"

    def prepare(self, spec: RunSpec) -> dict:
        raise RuntimeError(
            "llama.cpp runner requires llama-cpp-python server binary. "
            "Install with: pip install llama-cpp-python"
        )

    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        raise RuntimeError("llama.cpp not available.")

    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        return False

    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        raise RuntimeError("llama.cpp not available.")

    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        return {}
