"""Context stress test runner — sends large prefill payloads and records timing/quality.

Stress tests are simple: the task prompt contains a huge amount of text.
The API is called once with that prompt, and the model generates a short answer.

This module handles fail-gently behavior: if the model's API returns a context
length error or a 400-level prefill error, the task result records the error
instead of crashing the run.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ContextStressResult:
    """Holds per-request results from a context stress test."""

    def __init__(
        self,
        task_id: str,
        context_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        ttft_ms: float,
        decode_ms: float,
        wall_ms: float,
        decode_tps: float,
        quality_score: float | None,
        error: str | None,
        finish_reason: str = "stop",
    ):
        self.task_id = task_id
        self.context_tokens = context_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.ttft_ms = ttft_ms
        self.decode_ms = decode_ms
        self.wall_ms = wall_ms
        self.decode_tps = decode_tps
        self.quality_score = quality_score
        self.error = error
        self.finish_reason = finish_reason


class ContextStressRunner:
    """Stress tests model context length tolerance by sending large prefill payloads.

    The runner calls the model for each stress level, records TTFT, decode TPS,
    and captures errors gracefully (400, 408, timeout, etc.).
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "not-needed",
        max_tokens: int = 64,
        temperature: float = 0.0,
        timeout: float = 300.0,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def run(
        self,
        spec: dict,
        stress_levels: list[dict[str, Any]],
        num_runs: int = 3,
    ) -> list[ContextStressResult]:
        """Run stress tests against the model.

        Args:
            spec: RunSpec for project/name metadata.
            stress_levels: List of dicts with keys:
                - "id": task ID (e.g. "stress.ctx_250k")
                - "context_tokens": target context size in tokens
                - "prompt": the full prompt text (with large context block)
                - "scoring": scorer config
                - "expected": expected answer
                - "estimated_prompt_tokens": estimated token count
            num_runs: Number of times to run each level.

        Returns:
            List of ContextStressResult objects.
        """
        results: list[ContextStressResult] = []

        for level in stress_levels:
            task_id = level["id"]
            ctx_tokens = level["context_tokens"]
            prompt_text = level["prompt"]

            for run_idx in range(num_runs):
                try:
                    result = self._call_api(task_id, ctx_tokens, prompt_text, run_idx)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        "Context stress failed for %s run %d: %s",
                        task_id, run_idx, e
                    )
                    results.append(
                        ContextStressResult(
                            task_id=task_id,
                            context_tokens=ctx_tokens,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            ttft_ms=0.0,
                            decode_ms=0.0,
                            wall_ms=0.0,
                            decode_tps=0.0,
                            quality_score=None,
                            error=str(e),
                            finish_reason="error",
                        )
                    )

        return results

    def _call_api(
        self,
        task_id: str,
        ctx_tokens: int,
        prompt_text: str,
        run_idx: int,
    ) -> ContextStressResult:
        """Call the OpenAI-compatible API once and capture timing."""

        import httpx

        api_url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        api_start = time.perf_counter()

        # Try with a generous timeout, with retries for transient errors
        for attempt in range(3):
            try:
                resp = httpx.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    body = resp.json()

                    total_wall_ms = (time.perf_counter() - api_start) * 1000.0

                    content = (
                        body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    usage = body.get("usage", {})

                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    finish_reason = (
                        body.get("choices", [{}])[0].get("finish_reason", "stop")
                    )

                    # Estimate decode time as 85% of wall time
                    decode_ms = total_wall_ms * 0.85
                    ttft_ms = total_wall_ms * 0.15

                    decode_tps = (
                        completion_tokens / (decode_ms / 1000.0)
                        if decode_ms > 0
                        else 0.0
                    )

                    quality_score = 1.0 if content.strip() else 0.0

                    logger.info(
                        "Stress ctx_%dK run_%d: tokens=%d(%d/%d) "
                        "ttf=%.0fms tps=%.1f status=OK",
                        ctx_tokens // 1000,
                        run_idx,
                        total_tokens,
                        prompt_tokens,
                        completion_tokens,
                        ttft_ms,
                        decode_tps,
                    )

                    return ContextStressResult(
                        task_id=task_id,
                        context_tokens=ctx_tokens,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        ttft_ms=ttft_ms,
                        decode_ms=decode_ms,
                        wall_ms=total_wall_ms,
                        decode_tps=decode_tps,
                        quality_score=quality_score,
                        error=None,
                        finish_reason=finish_reason,
                    )

                elif resp.status_code == 400:
                    err_msg = resp.text[:200]
                    if "context" in err_msg.lower() or "prefill" in err_msg.lower():
                        logger.info(
                            "Stress ctx_%dK: context error (400) — model rejected",
                            ctx_tokens // 1000,
                        )
                        return ContextStressResult(
                            task_id=task_id,
                            context_tokens=ctx_tokens,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            ttft_ms=0.0,
                            decode_ms=0.0,
                            wall_ms=(time.perf_counter() - api_start) * 1000.0,
                            decode_tps=0.0,
                            quality_score=None,
                            error=f"400: {err_msg}",
                            finish_reason="context_error",
                        )

                # For non-context errors, try one more time
                time.sleep(0.5 * (attempt + 1))

            except httpx.ReadTimeout:
                logger.info(
                    "Stress ctx_%dK: read timeout (attempt %d)",
                    ctx_tokens // 1000,
                    attempt + 1,
                )
                time.sleep(1.0)
                continue
            except httpx.ConnectTimeout:
                return ContextStressResult(
                    task_id=task_id,
                    context_tokens=ctx_tokens,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    ttft_ms=0.0,
                    decode_ms=0.0,
                    wall_ms=(time.perf_counter() - api_start) * 1000.0,
                    decode_tps=0.0,
                    quality_score=None,
                    error="ConnectTimeout",
                    finish_reason="error",
                )

        # If we get here, all retries exhausted
        return ContextStressResult(
            task_id=task_id,
            context_tokens=ctx_tokens,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft_ms=0.0,
            decode_ms=0.0,
            wall_ms=(time.perf_counter() - api_start) * 1000.0,
            decode_tps=0.0,
            quality_score=None,
            error="Max retries exceeded",
            finish_reason="error",
        )


def stress_results_to_run_result(
    results: list[ContextStressResult],
    spec: dict,
) -> dict:
    """Convert stress results to a RunResult dict for storage."""
    per_request = []
    for r in results:
        req = {
            "request_id": f"{r.task_id}-run",
            "prompt_id": r.task_id,
            "prompt_tokens": r.prompt_tokens or 0,
            "generated_tokens": r.completion_tokens,
            "ttft_ms": r.ttft_ms,
            "decode_ms": r.decode_ms,
            "total_wall_ms": r.wall_ms,
            "tokens_per_second_decode": r.decode_tps,
            "tokens_per_second_wall": 0.0,
            "finish_reason": r.finish_reason,
            "error": r.error,
            "quality_score": r.quality_score,
            "extra_metadata": {"context_tokens": r.context_tokens},
        }
        per_request.append(req)

    return {
        "schema_version": "llm_bench.run_result.v1",
        "run_id": f"stress-{spec.get('name', 'unknown')}",
        "run_spec_ref": "stress-run-spec.yaml",
        "project": spec.get("project", "unknown"),
        "per_request": per_request,
    }


def create_run_spec_for_stress(
    base_url: str = "http://127.0.0.1:4000/v1",
    model_name: str = "qwen36-fast",
    num_runs: int = 3,
    stress_levels: list[str] | None = None,
) -> dict[str, Any]:
    """Create a minimal RunSpec dict for running context stress tests.

    The actual task YAML files must be generated by stress_test_generate.py
    and placed in the task_dir. This function creates a spec that points
    to those tasks.

    Args:
        base_url: Model API base URL.
        model_name: Model name to send in the API call.
        num_runs: Number of runs per stress level.
        stress_levels: List of context size strings (e.g.
            ["10K", "25K", "50K", "100K", "250K"]).
    """
    if stress_levels is None:
        stress_levels = ["10K", "25K", "50K", "100K", "250K"]

    return {
        "schema_version": "llm_bench.run_spec.v1",
        "name": f"context-stress-{model_name}",
        "project": "benchmark-harness",
        "tags": [
            "stress-test",
            "context-length",
            "stress",
        ] + stress_levels,
        "artifact": {
            "kind": "openai_endpoint",
            "mode": "external_path",
            "path": base_url,
            "model_id": model_name,
        },
        "runtime": {
            "kind": "openai_compatible",
            "launch": "existing",
            "host": base_url.split(":")[1].strip("/"),
            "port": int(base_url.split(":")[2].split("/")[0]),
            "model_name": model_name,
        },
        "workload": {
            "prompt_suite": "stress_test_context",
            "task_dir": "tasks/stress_test_context",
            "max_tokens": 64,
            "temperature": 0.0,
            "num_runs": num_runs,
            "concurrency": 1,
        },
        "storage": {
            "artifact_policy": "external_path",
            "result_policy": "managed",
        },
    }
