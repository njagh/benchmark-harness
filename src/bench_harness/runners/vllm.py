from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from bench_harness.runners.base import ProcessHandle, RuntimeRunner
from bench_harness.schemas.run_result import RunResult
from bench_harness.schemas.run_spec import RunSpec, LaunchMode
from bench_harness.storage.config import StorageConfig


class VLLMRunner(RuntimeRunner):
    """Launch and benchmark vLLM servers."""

    @property
    def kind(self) -> str:
        return "vllm"

    def prepare(self, spec: RunSpec) -> dict:
        model_path = spec.artifact.path

        if spec.runtime.launch == LaunchMode.managed_process:
            vllm_args = ["vllm", "serve", model_path]
            if spec.runtime.host:
                vllm_args.extend(["--host", spec.runtime.host])
            if spec.runtime.port:
                vllm_args.extend(["--port", str(spec.runtime.port)])
            for k, v in spec.runtime.args.items():
                vllm_args.extend([f"--{k}", str(v)])
            return {
                "model_path": model_path,
                "vllm_command": vllm_args,
                "endpoint": f"http://{spec.runtime.host or '127.0.0.1'}:{spec.runtime.port or 8000}/v1",
                "model_name": spec.runtime.model_name or spec.artifact.model_id or model_path,
            }
        else:
            return OpenAICompatibleRunner.prepare(self, spec)

    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        if spec.runtime.launch != LaunchMode.managed_process:
            return None

        cmd = prep["vllm_command"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        host = spec.runtime.host or "127.0.0.1"
        port = spec.runtime.port or 8000
        return ProcessHandle(
            proc=proc,
            host=host,
            port=port,
            ready_url=f"http://{host}:{port}/v1",
        )

    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        return OpenAICompatibleRunner.wait_until_ready(self, spec, prep, timeout)

    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        return OpenAICompatibleRunner.run_workload(self, spec, prep, result_dir)

    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        logs = OpenAICompatibleRunner.collect_logs(self, spec, prep, result_dir)
        logs["server.log"] = "vLLM server logs captured at launch time"
        return logs

    def shutdown(self, spec: RunSpec, prep: dict, handle: ProcessHandle | None) -> None:
        if handle and handle.proc.poll() is None:
            handle.proc.terminate()
            try:
                handle.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
