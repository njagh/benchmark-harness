"""Style sweep runner — compares tasks across multiple prompt styles."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from bench_harness.runners.completion_runner import CompletionRunner, RunResult
from bench_harness.tasks.prompt_templates import build_prompt, render_with_style

logger = logging.getLogger(__name__)


class StyleSweepRunner:
    """Runs tasks across multiple prompt styles for comparison.

    Two usage modes:

    Live execution mode (with a CompletionRunner):
        sweep = StyleSweepRunner(runner, styles=["plain", "repl", "terse"])
        results = sweep.run_sweep(tasks, models, params)
        # results: list of RunResult with prompt_style field set

    Dry-run / planning mode (without a CompletionRunner):
        sweep = StyleSweepRunner(
            base_tasks=[...],
            prompt_styles=["plain", "repl", "terse"],
            model_alias="test-model",
            suite_id="test-suite",
        )
        # sweep.sweeps contains planned style x task combinations
    """

    def __init__(
        self,
        base_runner: CompletionRunner | None = None,
        styles: list[str] | None = None,
        default_style: str = "plain",
        # Dry-run mode parameters
        base_tasks: list[dict[str, Any]] | None = None,
        prompt_styles: list[str] | None = None,
        model_alias: str = "",
        suite_id: str = "",
    ):
        # Live execution mode
        self.base_runner = base_runner
        # Only default to ["plain"] if styles is None (not provided), not if empty list is given
        self.styles = styles if styles is not None else [default_style]
        self.default_style = default_style

        # Dry-run mode
        self._dry_run_tasks = base_tasks or []
        self._dry_run_styles = prompt_styles or []
        self._dry_run_model = model_alias
        self._dry_run_suite = suite_id

        # Build sweeps list if in dry-run mode
        if base_tasks is not None and prompt_styles is not None:
            self._sweeps = self._build_sweeps(base_tasks, prompt_styles)
        else:
            self._sweeps = []

    @property
    def sweeps(self) -> list[dict[str, Any]]:
        """Return the planned sweep combinations (dry-run mode)."""
        return self._sweeps

    def _build_sweeps(
        self,
        tasks: list[dict[str, Any]],
        styles: list[str],
    ) -> list[dict[str, Any]]:
        """Build sweep combinations for dry-run mode."""
        sweeps = []
        fallback_styles = ["plain", "repl", "terse", "patch_only", "architect", "json_schema", "step_by_step"]

        for task in tasks:
            for style in styles:
                if style not in fallback_styles:
                    # Unknown style falls back to plain
                    style = "plain"
                # Build the styled prompt for this combination
                try:
                    rendered = render_with_style(task, style)
                    prompt = rendered.prompt
                except Exception:
                    prompt = task.get("prompt", "")

                sweeps.append({
                    "task_id": task.get("id", "unknown"),
                    "task_prompt": task.get("prompt", ""),
                    "prompt_style": style,
                    "rendered_prompt": prompt,
                    "model_alias": self._dry_run_model,
                    "suite_id": self._dry_run_suite,
                })

        return sweeps

    async def run_task_with_style(
        self,
        task: dict[str, Any],
        params: dict[str, Any],
        style: str,
        suite_id: str = "",
    ) -> RunResult:
        """Run a single task with a specific prompt style.

        Uses build_prompt() to render the styled prompt, then passes it
        to the base runner.
        """
        if self.base_runner is None:
            raise RuntimeError(
                "Cannot run live sweep without a base_runner. "
                "Use dry-run mode with base_tasks/prompt_styles parameters."
            )

        # 1. Build the styled prompt
        styled_prompt, referenced_files = build_prompt(
            task, prompt_style=style
        )

        # 2. Create a copy of the task with the rendered prompt
        styled_task = dict(task)
        styled_task["prompt"] = styled_prompt

        # 3. Add prompt_style to params
        run_params = dict(params)
        run_params["prompt_style"] = style

        # 4. Call the base runner
        result = await self.base_runner.run(styled_task, run_params, suite_id=suite_id)

        # 5. Tag the result with prompt_style metadata.
        # Use isinstance check to detect real RunResult vs mock objects.
        # With real RunResult, set prompt_style directly.
        # With mocks (no real run_id), create a new RunResult to avoid
        # shared state across style iterations.
        is_real_result = isinstance(result, RunResult)

        if is_real_result:
            result.prompt_style = style
        else:
            # For mocks or other non-RunResult objects, create a fresh result
            # with the correct style tag so each iteration produces independent output.
            from bench_harness.runners.completion_runner import RunResult as _RR
            result = _RR(
                run_id=uuid.uuid4().hex,
                suite_id=suite_id,
                task_id=task.get("id", "unknown"),
                model_alias=params.get("model_alias", "unknown"),
                prompt=styled_task.get("prompt", ""),
                raw_response=getattr(result, "raw_response", "") or "",
                prompt_tokens=getattr(result, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(result, "completion_tokens", 0) or 0,
                total_tokens=getattr(result, "total_tokens", 0) or 0,
                exit_status=getattr(result, "exit_status", "success"),
                error_message=getattr(result, "error_message", None),
                prompt_style=style,
            )

        logger.info(
            "Style sweep: task=%s style=%s model=%s status=%s",
            task.get("id"),
            style,
            params.get("model_alias", "unknown"),
            result.exit_status,
        )

        return result

    async def run_sweep(
        self,
        tasks: list[dict[str, Any]],
        model_aliases: list[str],
        params: dict[str, Any],
        suite_id: str = "",
    ) -> list[RunResult]:
        """Run a full style sweep: task x style x model.

        Returns list of RunResult, each tagged with prompt_style.
        """
        if self.base_runner is None:
            raise RuntimeError(
                "Cannot run live sweep without a base_runner. "
                "Use dry-run mode with base_tasks/prompt_styles parameters."
            )

        all_results: list[RunResult] = []

        for task in tasks:
            task_id = task.get("id", "unknown")
            logger.info("Sweep: running task %s across %d styles", task_id, len(self.styles))

            for style in self.styles:
                for alias in model_aliases:
                    style_params = dict(params)
                    style_params["model_alias"] = alias

                    try:
                        result = await self.run_task_with_style(
                            task, style_params, style, suite_id
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "Style sweep failed: task=%s style=%s model=%s error=%s",
                            task_id,
                            style,
                            alias,
                            e,
                        )
                        error_result = RunResult(
                            run_id=str(uuid.uuid4()),
                            suite_id=suite_id,
                            task_id=task_id,
                            model_alias=alias,
                            prompt="",
                            prompt_style=style,
                            exit_status="error",
                            error_message=str(e),
                            created_at=datetime.now(timezone.utc).isoformat(),
                        )
                        all_results.append(error_result)

        return all_results
