"""Completion runner — executes a single task against a model."""

from __future__ import annotations

import logging
import tempfile
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
        tests_passed: Number of tests that passed (code tasks only).
        tests_failed: Number of tests that failed (code tasks only).
        tests_total: Total number of tests (code tasks only).
        test_output: Raw pytest stdout/stderr (code tasks only).
        exit_code: Pytest exit code (code tasks only).
        generated_code: The code the model generated (code tasks only).
        code_status: Status of generated code (code tasks only).
        validation_passed: Whether shell/config validation passed (non-code tasks only).
        validation_command: The validation command that was run (non-code tasks only).
        validation_output: Raw output from validation command (non-code tasks only).
        judge_score: LLM judge score (M7).
        judge_explanation: LLM judge explanation text (M7).
        judge_dimensions: Per-dimension scores from judge (M7).
        judge_model: Judge model alias used (M7).
        human_override: Whether human overrode the judge score (M7).
        human_score: Human-provided override score (M7).
        human_note: Human reviewer note (M7).
        prompt_style: Prompt style used (M8).
        context_tokens: Context size bucket (M9).
        estimated_prompt_tokens: Estimated prompt token count (M9).
        quantization: Model quantization (M10).
        requested_alias: Model alias from configs/models.yaml (M14 identity stamp).
        litellm_model_name: Actual model name sent to API (M14).
        openai_models_id: Returned model id from /v1/models (M14).
        vllm_served_model_name: served_model_name from vLLM (M14).
        vllm_container_name: Container name (M14).
        hf_model_id: HuggingFace model ID (M14).
        backend_url: Backend URL used for this run (M14).
        server_start_time: Server start time from /v1/models (M14).
        speculative_decoding_enabled: Speculative decoding flag (M14).
        safety_score: Command safety score (M11) — 1.0 = all safe, 0.0 = all dangerous.
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
    score_primary: float | None = None
    score_secondary: dict[str, Any] | None = None
    scorer_version: str | None = None
    score_explanation: str | None = None
    exit_status: str = "success"
    error_message: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_total: int | None = None
    test_output: str | None = None
    exit_code: int | None = None
    generated_code: str | None = None
    code_status: str | None = None
    validation_passed: bool | None = None
    validation_command: str | None = None
    validation_output: str | None = None
    # M7 judge fields
    judge_score: float | None = None
    judge_explanation: str | None = None
    judge_dimensions: dict[str, Any] | None = None
    judge_model: str | None = None
    human_override: bool | None = None
    human_score: float | None = None
    human_note: str | None = None
    prompt_style: str | None = None  # The prompt style used for this run
    context_tokens: str | None = None  # Context size bucket
    estimated_prompt_tokens: int | None = None  # Estimated prompt token count
    quantization: str | None = None  # Model quantization (e.g., "FP8", "GPTQ-Int4", "FP16")
    # M11 command safety score
    safety_score: float | None = None  # Command safety score (M11)
    safety_details: dict[str, Any] | None = None  # Safety classification details (M11)
    # M14 identity stamp: before each task, harness calls /v1/models and records
    # actual returned model id plus the backend URL/container expected for that alias.
    # This makes "qwen-dense served as agent-code" obvious instead of inferential.
    requested_alias: str | None = None  # alias from configs/models.yaml
    litellm_model_name: str | None = None  # model name sent to API call
    openai_models_id: str | None = None  # returned id from /v1/models response
    vllm_served_model_name: str | None = None  # served_model_name from vLLM /v1/models
    vllm_container_name: str | None = None  # container name if available
    hf_model_id: str | None = None  # HuggingFace model ID if available
    backend_url: str | None = None  # the backend URL this run used
    server_start_time: str | None = None  # server_start_time from /v1/models if available
    speculative_decoding_enabled: bool | None = None  # whether speculative decoding is enabled


def build_messages(task: dict, prompt: str) -> list[dict]:
    """Build OpenAI-compatible messages list from task and prompt.

    Uses system message from task.input if available, otherwise
    defaults to user-only messages.
    """
    messages = []
    system_msg = (task.get("input") or {}).get("system_message")
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    return messages


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

        For code tasks (identified by 'code_type' in the task dict),
        delegates to CodeTaskRunner for execution and testing, then
        applies secondary scorers for additional evaluation.

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
        context_tokens = params.get("context_tokens", task.get("context_tokens", "small"))
        estimated_prompt_tokens = params.get("estimated_prompt_tokens")

        # Extract identity stamp fields from params (populated by CLI before each task)
        requested_alias = params.get("requested_alias", model_alias)
        litellm_model_name = params.get("litellm_model_name")
        openai_models_id = params.get("openai_models_id")
        vllm_served_model_name = params.get("vllm_served_model_name")
        vllm_container_name = params.get("vllm_container_name")
        hf_model_id = params.get("hf_model_id")
        backend_url = params.get("backend_url")
        server_start_time = params.get("server_start_time")
        speculative_decoding_enabled = params.get("speculative_decoding_enabled")

        scorer_version: str | None = None
        safety_score_val: float | None = None
        safety_details: dict[str, Any] | None = None
        start_time = datetime.now(timezone.utc)

        # Auto-detect code tasks
        code_type = task.get("code_type")
        is_code_task = code_type in ("function_completion", "patch_generation")

        if is_code_task:
            return await self._run_code_task(
                task, params, suite_id, run_id, task_id, prompt,
                model_alias, model_backend, start_time,
            )

        try:
            # Build messages from prompt (using system message if present)
            messages = build_messages(task, prompt)

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

            # Score the response
            primary_scorer_name = task.get("scoring", {}).get("primary", "")
            secondary_scorer_names = task.get("scoring", {}).get("secondary", [])
            scorer_names = [primary_scorer_name] + (secondary_scorer_names or [])
            primary_score = None
            primary_explanation = None
            secondary_scores: dict[str, Any] = {}

            # Check for validation command (non-code tasks)
            validation_command = task.get("validation_command") or task.get("expected", {}).get("validation_command")
            validation_allowlist = task.get("validation_allowlist", [])
            validation_passed = None
            validation_output = None

            if validation_command:
                shell_result = self._run_validation_command(validation_command, validation_allowlist, content)
                if shell_result:
                    validation_passed = shell_result.get("passed")
                    validation_output = shell_result.get("stdout", "") + shell_result.get("stderr", "")

            if scorer_names:
                from bench_harness.scorers import score_all, get_scorer
                try:
                    # Use a minimal task-like dict for scorers
                    task_dict = {
                        "id": task_id,
                        "scoring": task.get("scoring", {}),
                        "expected": task.get("expected", {}),
                    }
                    # Run primary scorer
                    if primary_scorer_name:
                        scorer = get_scorer(primary_scorer_name)
                        score_result = scorer.score(task_dict, content)
                        primary_score = score_result.score
                        primary_explanation = score_result.explanation
                        scorer_version = score_result.scorer_version

                    # Run secondary scorers
                    for sec_name in (secondary_scorer_names or []):
                        try:
                            sec_scorer = get_scorer(sec_name)
                            sec_result = sec_scorer.score(task_dict, content)
                            secondary_scores[sec_name] = {
                                "score": sec_result.score,
                                "passed": sec_result.passed,
                                "explanation": sec_result.explanation,
                            }
                        except Exception as e:
                            logger.warning(
                                "Secondary scorer '%s' failed: %s", sec_name, e
                            )

                    # Run command_safety scorer if configured
                    for sec_name in (secondary_scorer_names or []):
                        if sec_name == "command_safety":
                            try:
                                sec_scorer = get_scorer(sec_name)
                                sec_result = sec_scorer.score(task_dict, content)
                                safety_score_val = sec_result.score
                                safety_details = sec_result.details
                            except Exception as e:
                                logger.warning(
                                    "Command safety scorer failed: %s", e
                                )
                except Exception as e:
                    logger.warning("Scoring failed for task %s: %s", task_id, e)

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
                score_primary=primary_score,
                score_secondary=secondary_scores if secondary_scores else None,
                scorer_version=scorer_version if primary_scorer_name else None,
                score_explanation=primary_explanation,
                exit_status="success",
                created_at=start_time.isoformat(),
                validation_passed=validation_passed,
                validation_command=validation_command,
                validation_output=validation_output,
                prompt_style=params.get("prompt_style"),
                context_tokens=context_tokens,
                estimated_prompt_tokens=estimated_prompt_tokens,
                # M10 quantization
                quantization=params.get("quantization"),
                # M11 command safety
                safety_score=safety_score_val,
                safety_details=safety_details,
                # M14 identity stamp
                requested_alias=requested_alias,
                litellm_model_name=litellm_model_name,
                openai_models_id=openai_models_id,
                vllm_served_model_name=vllm_served_model_name,
                vllm_container_name=vllm_container_name,
                hf_model_id=hf_model_id,
                backend_url=backend_url,
                server_start_time=server_start_time,
                speculative_decoding_enabled=speculative_decoding_enabled,
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
                prompt_style=params.get("prompt_style"),
                context_tokens=context_tokens,
                estimated_prompt_tokens=estimated_prompt_tokens,
                quantization=params.get("quantization"),
                # M11 command safety
                safety_score=safety_score_val,
                safety_details=safety_details,
                # M14 identity stamp
                requested_alias=requested_alias,
                litellm_model_name=litellm_model_name,
                openai_models_id=openai_models_id,
                vllm_served_model_name=vllm_served_model_name,
                vllm_container_name=vllm_container_name,
                hf_model_id=hf_model_id,
                backend_url=backend_url,
                server_start_time=server_start_time,
                speculative_decoding_enabled=speculative_decoding_enabled,
            )

        return result

    async def _run_code_task(
        self,
        task: dict[str, Any],
        params: dict[str, Any],
        suite_id: str,
        run_id: str,
        task_id: str,
        prompt: str,
        model_alias: str,
        model_backend: str,
        start_time: datetime,
    ) -> RunResult:
        """Run a code task: generate code via model, then execute and test it.

        Calls the model to generate code, delegates to CodeTaskRunner for
        test execution, and runs secondary scorers on the generated code.
        """
        import time as _time

        temperature = params.get("temperature", 0)
        max_tokens = params.get("max_tokens", 4096)
        quantization = params.get("quantization")
        context_tokens = params.get("context_tokens", task.get("context_tokens", "small"))
        estimated_prompt_tokens = params.get("estimated_prompt_tokens")

        # Extract identity stamp fields from params
        requested_alias = params.get("requested_alias", model_alias)
        litellm_model_name = params.get("litellm_model_name")
        openai_models_id = params.get("openai_models_id")
        vllm_served_model_name = params.get("vllm_served_model_name")
        vllm_container_name = params.get("vllm_container_name")
        hf_model_id = params.get("hf_model_id")
        backend_url = params.get("backend_url")
        server_start_time = params.get("server_start_time")
        speculative_decoding_enabled = params.get("speculative_decoding_enabled")

        safety_score_val: float | None = None
        safety_details: dict[str, Any] | None = None

        try:
            # Step 1: Call the model to generate code (using system message if present)
            messages = build_messages(task, prompt)
            api_start = _time.perf_counter()

            response = await self.client.chat_complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            api_end = _time.perf_counter()
            total_wall_ms = (api_end - api_start) * 1000.0

            generated_code = response.get("content") or ""
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
                completion_tokens = self._fallback_tokenizer.count_completion(generated_code)
                total_tokens = prompt_tokens + completion_tokens
                token_source = "fallback_tokenizer"
                token_counter.prompt_tokens = prompt_tokens
                token_counter.completion_tokens = completion_tokens
                token_counter.total_tokens = total_tokens
                token_counter.source = token_source

            # Compute timing metrics
            ttft_ms = total_wall_ms * 0.15
            decode_ms = total_wall_ms * 0.85
            prefill_ms = total_wall_ms * 0.15

            tps = compute_tokens_per_second(completion_tokens, decode_ms)
            tps_total = compute_tokens_per_second(total_tokens, total_wall_ms)

            # Step 2: Run the code through CodeTaskRunner
            runner_params = {
                "suite_id": suite_id,
            }
            # Add temperature/max_tokens if present
            for k in ("temperature", "max_tokens", "model_alias", "model_backend"):
                if k in params:
                    runner_params[k] = params[k]

            code_result = self._run_code_task_sync(task, generated_code, runner_params)

            # Step 3: Run secondary scorers on generated code
            primary_scorer_name = task.get("scoring", {}).get("primary", "")
            secondary_scorer_names = task.get("scoring", {}).get("secondary", [])
            secondary_scores: dict[str, Any] = {}

            if secondary_scorer_names:
                from bench_harness.scorers import get_scorer
                try:
                    task_dict = {
                        "id": task_id,
                        "family": task.get("family", "coding"),
                        "scoring": task.get("scoring", {}),
                        "expected": task.get("expected", {}),
                        "code_type": task.get("code_type"),
                    }
                    for sec_name in secondary_scorer_names:
                        try:
                            sec_scorer = get_scorer(sec_name)
                            sec_result = sec_scorer.score(task_dict, generated_code)
                            secondary_scores[sec_name] = {
                                "score": sec_result.score,
                                "passed": sec_result.passed,
                                "explanation": sec_result.explanation,
                            }
                        except Exception as e:
                            logger.warning(
                                "Secondary scorer '%s' failed: %s", sec_name, e
                            )
                except Exception as e:
                    logger.warning("Secondary scoring failed for task %s: %s", task_id, e)

            result = RunResult(
                run_id=run_id,
                suite_id=suite_id,
                task_id=task_id,
                model_alias=model_alias,
                model_backend=model_backend,
                prompt=prompt,
                raw_response=generated_code,
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
                score_primary=code_result.get("score_primary"),
                score_secondary=code_result.get("score_secondary") if code_result.get("score_secondary") else None,
                scorer_version=code_result.get("scorer_version"),
                score_explanation=code_result.get("score_explanation"),
                tests_passed=code_result.get("tests_passed"),
                tests_failed=code_result.get("tests_failed"),
                tests_total=code_result.get("tests_total"),
                test_output=code_result.get("test_output"),
                exit_code=code_result.get("exit_code"),
                generated_code=code_result.get("generated_code"),
                code_status=code_result.get("code_status"),
                exit_status="success" if code_result.get("exit_code", 0) == 0 else "error",
                error_message=code_result.get("error_message"),
                created_at=start_time.isoformat(),
                prompt_style=params.get("prompt_style"),
                context_tokens=context_tokens,
                estimated_prompt_tokens=estimated_prompt_tokens,
                quantization=quantization,
                # M14 identity stamp
                requested_alias=requested_alias,
                litellm_model_name=litellm_model_name,
                openai_models_id=openai_models_id,
                vllm_served_model_name=vllm_served_model_name,
                vllm_container_name=vllm_container_name,
                hf_model_id=hf_model_id,
                backend_url=backend_url,
                server_start_time=server_start_time,
                speculative_decoding_enabled=speculative_decoding_enabled,
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
                prompt_style=params.get("prompt_style"),
                quantization=quantization,
                # M14 identity stamp
                requested_alias=requested_alias,
                litellm_model_name=litellm_model_name,
                openai_models_id=openai_models_id,
                vllm_served_model_name=vllm_served_model_name,
                vllm_container_name=vllm_container_name,
                hf_model_id=hf_model_id,
                backend_url=backend_url,
                server_start_time=server_start_time,
                speculative_decoding_enabled=speculative_decoding_enabled,
            )

        return result

    def _run_validation_command(
        self,
        command: str,
        allowlist: list[str],
        response_content: str,
    ) -> dict[str, Any] | None:
        """Run a validation command against the generated response.

        Writes the response to a temp file and runs the validation
        command in that directory.

        Args:
            command: The validation shell command to run.
            allowlist: List of allowed command prefixes.
            response_content: The model's response content.

        Returns:
            Dict with pass/fail results, or None if ShellRunner fails.
        """
        try:
            from bench_harness.runners.shell_runner import ShellRunner

            runner = ShellRunner(allowlist=allowlist)
            with tempfile.TemporaryDirectory(prefix="validation_") as tmpdir:
                return runner.run(command, tmpdir)
        except Exception as e:
            logger.warning("Validation command '%s' failed: %s", command, e)
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "duration_ms": 0,
            }

    def _run_code_task_sync(
        self,
        task: dict[str, Any],
        generated_code: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronously run a code task via CodeTaskRunner."""
        from bench_harness.runners.code_task_runner import CodeTaskRunner

        runner = CodeTaskRunner()
        return runner.run(task, generated_code, params=params)
