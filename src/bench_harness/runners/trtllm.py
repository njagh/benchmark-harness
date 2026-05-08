from __future__ import annotations

from pathlib import Path

from bench_harness.runners.base import ProcessHandle, RuntimeRunner
from bench_harness.schemas.run_result import RunResult
from bench_harness.schemas.run_spec import RunSpec
from bench_harness.storage.config import StorageConfig


class TRTLLMRunner(RuntimeRunner):
    """Launch and benchmark TensorRT-LLM engines (stub)."""

    @property
    def kind(self) -> str:
        return "trtllm"

    def prepare(self, spec: RunSpec) -> dict:
        raise RuntimeError(
            "TRT-LLM runner requires tensorrt_llm package. "
            "Install with: pip install tensorrt_llm"
        )

    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        raise RuntimeError("TRT-LLM not available. Install tensorrt_llm first.")

    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        return False

    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        raise RuntimeError("TRT-LLM not available.")

    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        return {}
