from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.runners.base import ProcessHandle, RuntimeRunner
from bench_harness.runners.completion_runner import CompletionRunner
from bench_harness.schemas.run_result import RunResult, RequestResult
from bench_harness.schemas.run_spec import RunSpec
from bench_harness.storage.config import StorageConfig
from bench_harness.tasks.loaders import load_tasks


class OpenAICompatibleRunner(RuntimeRunner):
    """Benchmark already-running OpenAI-compatible endpoints."""

    @property
    def kind(self) -> str:
        return "openai_compatible"

    def prepare(self, spec: RunSpec) -> dict:
        prep = {}
        endpoint = spec.artifact.path
        if spec.runtime.host and spec.runtime.port:
            endpoint = f"http://{spec.runtime.host}:{spec.runtime.port}/v1"
        prep["endpoint"] = endpoint
        prep["model_name"] = spec.runtime.model_name or spec.artifact.model_id or "unknown"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{endpoint}/models")
                if resp.status_code == 200:
                    prep["v1_models"] = resp.json()
        except Exception:
            prep["v1_models"] = None
        return prep

    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        return None

    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        endpoint = prep["endpoint"]
        start = time.time()
        with httpx.Client(timeout=5.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = client.post(f"{endpoint}/chat/completions", json={
                        "model": prep["model_name"],
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    })
                    if resp.status_code in (200, 400):
                        return True
                except Exception:
                    pass
                time.sleep(2)
        return False

    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        model_name = prep["model_name"]
        client = OpenAICompatClient(
            base_url=prep["endpoint"],
            api_key=spec.runtime.args.get("api_key", "not-needed"),
            model=model_name,
        )
        runner = CompletionRunner(client, fallback_tokenizer=None)

        task_dir = spec.workload.task_dir
        if task_dir is not None:
            tasks = load_tasks(task_dir)
        else:
            tasks = load_tasks(spec.workload.prompt_suite)

        results = []
        for task in tasks:
            for run_idx in range(spec.workload.num_runs):
                task_result = asyncio.run(runner.run(
                    task,
                    params={
                        "max_tokens": spec.workload.max_tokens,
                        "temperature": spec.workload.temperature,
                    },
                    suite_id=spec.name,
                ))
                req = RequestResult(
                    request_id=f"{task.get('id', 'unknown')}-{run_idx}",
                    prompt_id=task.get("id", "unknown"),
                    prompt_tokens=getattr(task_result, 'prompt_tokens', 0),
                    generated_tokens=getattr(task_result, 'completion_tokens', 0),
                    ttft_ms=getattr(task_result, 'ttft_ms', 0),
                    decode_ms=getattr(task_result, 'decode_ms', 0),
                    total_wall_ms=getattr(task_result, 'total_wall_ms', 0),
                    tokens_per_second_decode=getattr(task_result, 'tokens_per_second', 0),
                    tokens_per_second_wall=getattr(task_result, 'tokens_per_second_total', 0),
                    finish_reason="stop",
                    error=None,
                    quality_score=getattr(task_result, 'score_primary', None),
                )
                results.append(req)

        rr = RunResult(
            run_id=spec.name,
            run_spec_ref="run_spec.yaml",
            project=spec.project,
            per_request=results,
        )
        return rr.finalize()

    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        logs = {}
        if prep.get("v1_models"):
            import json
            logs["v1_models.json"] = json.dumps(prep["v1_models"])
        return logs
