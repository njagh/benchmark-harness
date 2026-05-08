from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import Popen
from typing import Any

from bench_harness.storage.config import StorageConfig
from bench_harness.schemas.run_spec import RunSpec
from bench_harness.schemas.run_result import RunResult


@dataclass
class ProcessHandle:
    proc: Popen
    host: str
    port: int
    ready_url: str


class RuntimeRunner(ABC):
    """Base class for model serving runtime runners."""

    def __init__(self, config: StorageConfig):
        self.config = config

    @abstractmethod
    def prepare(self, spec: RunSpec) -> dict:
        """Validate spec, resolve artifact path, prepare environment.
        Returns runtime-specific prep dict."""

    @abstractmethod
    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        """Launch the runtime server. Returns process handle if managed, None if external.
        Must bind to spec.runtime.host:spec.runtime.port."""

    @abstractmethod
    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        """Poll the runtime until it responds or timeout."""

    @abstractmethod
    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        """Execute the workload against the running runtime.
        Returns aggregated RunResult."""

    @abstractmethod
    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        """Collect server logs, /v1/models response, runtime version info.
        Returns {filename: content} to write to result_dir."""

    def shutdown(self, spec: RunSpec, prep: dict, handle: ProcessHandle | None) -> None:
        """Stop the runtime if it was managed. No-op for external/endpoint runners."""
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        """Return the runtime kind string, e.g. 'vllm', 'trtllm'."""
