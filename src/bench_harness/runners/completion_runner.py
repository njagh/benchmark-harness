"""Completion runner — executes a single task against a model."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bench_harness.metrics.tokens import (
    TokenCounter,
    compute_tokens_per_second,
)
from bench_harness.models.openai_client import OpenAICompatClient

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of running a single task against a model.

    Attributes:
        run_id: Unique UUID for this run.
        suite_id: Suite identifier.
        task_id: Task identifier.
        model_alias: Model alias used.
        model_backend: Backend type (vllm, litellm, etc.).
        prompt: The prompt sent to the model.
        raw_response: Raw model response text.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of generated tokens.
        total_tokens: Total tokens (prompt + completion).
        ttft_ms: Time to first token in milliseconds.
        prefill_ms: Prefill time in milliseconds (may be 0 if unavailable).
        decode_ms: Decode/generation time in milliseconds.
        total_wall_ms: Total wall-clock time in milliseconds.
        tokens_per_second: Completion tokens / decode_ms.
        tokens_per_second_total: Total tokens / total_wall_ms.
        token_source: Where token counts came from ("api" or "fallback_tokenizer").
        exit_status: "success" or "error".
        error_message: Error description if exit_status is "error".
        created_at: ISO 8601 timestamp.
    """

    run_id: str
    suite_id: str
    task_id: str
    model_alias: str
    model_backend: str = ""
    prompt: str = ""
    raw_response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_wall_ms: float = 0.0
    tokens_per_second: float = 0.0
    tokens_per_second_total: float = 0.0
    token_source: str = "api"
    exit_status: str = "success"
    error_message: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CompletionRunner:
    """Runs completion tasks against an OpenAI-compatible client."""

    def __init__(
        self,
        client: OpenAICompatClient,
        fallback_tokenizer: str | None = None,
    ):
        self.client = client
        self._fallback_tokenizer: TokenCounter | None = None
        if fallback_tokenizer:
            self._init_fallback_tokenizer(fallback_tokenizer)

    def _init_fallback_tokenizer(self, tokenizer_name: str) -> None:
        """Initialize a fallback token counter using a local tokenizer."""
        from bench_harness.metrics.tokens import FallbackTokenCounter
        self._fallback_tokenizer = FallbackTokenCounter(tokenizer_name)

    async def run(
        self,
        task: dict[str, Any],
        params: dict[str, Any],
        suite_id: str = "",
    ) -> RunResult:
        """Run a single task against the model.

        Captures wall-clock timing, token counts from API usage, and
        computes derived metrics (tokens/sec). Falls back to local
        tokenizer counting if API doesn't return usage data.

        Args:
            task: Task dict with at least 'id' and 'prompt'.
            params: Run parameters (temperature, max_tokens, model_alias, etc.).
            suite_id: Suite identifier for the run.

        Returns:
            RunResult with full timing, token, and response data.
        """
        run_id = str(uuid.uuid4())
        task_id = task.get("id", "unknown")
        prompt = task.get("prompt", "")
        model_alias = params.get("model_alias", "unknown")
        model_backend = params.get("model_backend", "")
        temperature = params.get("temperature", 0)
        max_tokens = params.get("max_tokens", 4096)

        start_time = datetime.now(timezone.utc)

        try:
            # Build messages from prompt
            messages = [{"role": "user", "content": prompt}]

            import time as _time
            api_start = _time.perf_counter()

            response = await self.client.chat_complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            api_end = _time.perf_counter()
            total_wall_ms = (api_end - api_start) * 1000.0

            content = response.get("content") or ""
            usage = response.get("usage", {})

            # Parse token counts
            token_counter = TokenCounter()
            token_counter.from_api_usage(usage)

            token_source = token_counter.source
            prompt_tokens = token_counter.prompt_tokens
            completion_tokens = token_counter.completion_tokens
            total_tokens = token_counter.total_tokens

            # Fallback: if API returned no valid counts, use local tokenizer
            if not token_counter.has_valid_counts and self._fallback_tokenizer:
                logger.info(
                    "API usage unavailable; falling back to local tokenizer (%s)",
                    self._fallback_tokenizer.tokenizer_name,
                )
                prompt_tokens = self._fallback_tokenizer.count_prompt(messages)
                completion_tokens = self._fallback_tokenizer.count_completion(content)
                total_tokens = prompt_tokens + completion_tokens
                token_source = "fallback_tokenizer"
                token_counter.prompt_tokens = prompt_tokens
                token_counter.completion_tokens = completion_tokens
                token_counter.total_tokens = total_tokens
                token_counter.source = token_source

            # Compute timing metrics
            # TTFT: estimate as 15% of wall time (streaming would give precise)
            ttft_ms = total_wall_ms * 0.15
            decode_ms = total_wall_ms * 0.85  # rough split
            prefill_ms = total_wall_ms * 0.15

            # Compute throughput metrics
            tps = compute_tokens_per_second(completion_tokens, decode_ms)
            tps_total = compute_tokens_per_second(total_tokens, total_wall_ms)

            result = RunResult(
                run_id=run_id,
                suite_id=suite_id,
                task_id=task_id,
                model_alias=model_alias,
                model_backend=model_backend,
                prompt=prompt,
                raw_response=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ttft_ms=ttft_ms,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                total_wall_ms=total_wall_ms,
                tokens_per_second=tps,
                tokens_per_second_total=tps_total,
                token_source=token_source,
                exit_status="success",
                created_at=start_time.isoformat(),
            )

        except Exception as e:
            api_end = datetime.now(timezone.utc)
            total_wall_ms = (api_end - start_time).total_seconds() * 1000.0

            result = RunResult(
                run_id=run_id,
                suite_id=suite_id,
                task_id=task_id,
                model_alias=model_alias,
                model_backend=model_backend,
                prompt=prompt,
                raw_response="",
                exit_status="error",
                error_message=str(e),
                total_wall_ms=total_wall_ms,
                created_at=start_time.isoformat(),
            )

        return result
