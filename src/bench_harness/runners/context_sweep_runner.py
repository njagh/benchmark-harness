"""Context size sweep runner — executes tasks across context sizes."""

from __future__ import annotations

import logging
from typing import Any

from bench_harness.metrics.context_packer import ContextPacker

logger = logging.getLogger(__name__)


class ContextSizeSweepRunner:
    """Runs tasks across multiple context sizes for comparison.

    Usage:
        sweep = ContextSizeSweepRunner(base_runner, sizes=["small", "medium", "large", "xlarge"])
        results = await sweep.run(tasks, models, params)
    """

    def __init__(
        self,
        base_runner: Any,
        sizes: list[str] | None = None,
    ):
        """Initialize the context size sweep runner.

        Args:
            base_runner: A CompletionRunner instance to execute individual runs.
            sizes: List of context size bucket names. Defaults to all 4 sizes.
        """
        self.base_runner = base_runner
        self.sizes = sizes or ["small", "medium", "large", "xlarge"]
        self.packer = ContextPacker()

    async def run_task_with_size(
        self,
        task: dict[str, Any],
        params: dict[str, Any],
        size: str,
        suite_id: str = "",
    ) -> Any:
        """Run a single task with a specific context size.

        Args:
            task: Task dict.
            params: Run parameters.
            size: Context size bucket ("small", "medium", "large", "xlarge").
            suite_id: Suite identifier.

        Returns:
            RunResult with context_size and estimated_prompt_tokens tagged.
        """
        # 1. Get the file context from task.input.files
        file_context: dict[str, str] = {}
        task_input = task.get("input") or {}
        files = task_input.get("files") or []

        if files:
            from pathlib import Path
            base_dir = task.get("_base_dir")
            for fpath in files:
                full_path = Path(base_dir) / fpath if base_dir else Path(fpath)
                if full_path.exists():
                    try:
                        file_context[fpath] = full_path.read_text()
                    except OSError as e:
                        logger.warning("Could not read context file %s: %s", fpath, e)

        # 2. Convert to list of dicts for ContextPacker
        file_dicts = [{"name": k, "content": v} for k, v in file_context.items()]

        # 3. Use ContextPacker to pack files to the target size
        packed_context = self.packer.pack(file_dicts, target_budget=size)

        # 4. Estimate prompt tokens
        prompt_template = task.get("prompt", "")
        prompt_with_context = (
            f"{prompt_template}\n\n{packed_context}" if packed_context else prompt_template
        )
        estimated_prompt_tokens = _estimate_tokens_from_text(prompt_with_context)

        # 5. Build a modified task with the context-augmented prompt
        task_copy = dict(task)
        task_copy["prompt"] = prompt_with_context

        # 6. Update params with context metadata
        run_params = dict(params)
        run_params["context_tokens"] = size
        run_params["estimated_prompt_tokens"] = estimated_prompt_tokens

        # 7. Call base runner
        result = await self.base_runner.run(task_copy, run_params, suite_id=suite_id)

        # 8. Tag result with context_size and estimated_prompt_tokens
        result.context_tokens = size
        result.estimated_prompt_tokens = estimated_prompt_tokens

        logger.info(
            "Context sweep: task=%s size=%s model=%s est_tokens=%d status=%s",
            task.get("id"),
            size,
            params.get("model_alias", "unknown"),
            estimated_prompt_tokens,
            result.exit_status,
        )

        return result

    async def run_sweep(
        self,
        tasks: list[dict[str, Any]],
        model_aliases: list[str],
        params: dict[str, Any],
        suite_id: str = "",
    ) -> list[Any]:
        """Run task x size x model sweep.

        Args:
            tasks: List of task dicts.
            model_aliases: List of model alias strings.
            params: Base run parameters.
            suite_id: Suite identifier.

        Returns:
            List of RunResult objects, each tagged with context_size.
        """
        all_results: list[Any] = []

        for task in tasks:
            task_id = task.get("id", "unknown")
            logger.info(
                "Context sweep: running task %s across %d sizes",
                task_id,
                len(self.sizes),
            )

            for size in self.sizes:
                for alias in model_aliases:
                    run_params = dict(params)
                    run_params["model_alias"] = alias

                    try:
                        result = await self.run_task_with_size(
                            task, run_params, size, suite_id
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "Context sweep failed: task=%s size=%s model=%s error=%s",
                            task_id,
                            size,
                            alias,
                            e,
                        )
                        from bench_harness.runners.completion_runner import RunResult
                        from datetime import datetime, timezone
                        import uuid

                        error_result = RunResult(
                            run_id=str(uuid.uuid4()),
                            suite_id=suite_id,
                            task_id=task_id,
                            model_alias=alias,
                            prompt="",
                            exit_status="error",
                            error_message=str(e),
                            context_tokens=size,
                            estimated_prompt_tokens=0,
                            created_at=datetime.now(timezone.utc).isoformat(),
                        )
                        all_results.append(error_result)

        return all_results


def _estimate_tokens_from_text(text: str) -> int:
    """Rough token count estimate using word splitting."""
    return len(text.split())
