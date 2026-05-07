"""Adapter for running public benchmarks via lm-evaluation-harness.

Wraps an OpenAI-compatible endpoint as an lm_eval model and runs
standard benchmark subsets (MMLU, GPQA, BBH, MATH) with small sample
sizes for quick reference scoring.

lm_eval is an optional dependency. If not installed the adapter
gracefully fails with a clear error message.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task definitions: (lm_eval task name, shot count, max samples)
# ---------------------------------------------------------------------------

TASK_DEFS: dict[str, dict[str, Any]] = {
    "mmlu_college_math": {
        "lm_eval_task": "mmlu_college_math",
        "shots": 5,
        "max_samples": 200,
        "family": "mmlu",
    },
    "mmlu_high_school_cs": {
        "lm_eval_task": "mmlu_high_school_cs",
        "shots": 5,
        "max_samples": 100,
        "family": "mmlu",
    },
    "mmlu_machine_learning": {
        "lm_eval_task": "mmlu_machine_learning",
        "shots": 5,
        "max_samples": 200,
        "family": "mmlu",
    },
    "gpqa_diamond": {
        "lm_eval_task": "gpqa_diamond",
        "shots": 5,
        "max_samples": 100,
        "family": "gpqa",
    },
    "bbh_code_generation": {
        "lm_eval_task": "bbh_code_generation",
        "shots": 3,
        "max_samples": 50,
        "family": "bbh",
    },
}


@dataclass
class LMEvalResult:
    """Result from a single lm_eval benchmark task."""

    task_id: str
    model_alias: str
    accuracy: float | None = None
    samples_run: int = 0
    error_count: int = 0
    error_message: str | None = None
    suite_id: str = ""
    family: str = ""
    shots: int = 0
    max_samples: int = 0
    created_at: str = ""


def _check_lm_eval_available() -> bool:
    """Return True if lm-evaluation-harness is installed."""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        return False


def get_lm_eval_error() -> str:
    """Return a user-friendly message explaining how to install lm_eval."""
    return (
        "The 'lm_eval' package is not installed. "
        "Install it with:\n\n"
        "    pip install lm-evaluation-harness\n\n"
        "This module is optional — the benchmark harness works without it."
    )


class LMEvalAdapter:
    """Thin adapter layer around lm-evaluation-harness.

    Takes an OpenAI-compatible endpoint URL + model name, runs selected
    benchmark tasks, and returns structured results.

    Args:
        base_url: OpenAI-compatible API base URL (e.g. vLLM endpoint).
        model: Model name sent to the API.
        max_samples: Global cap on samples per task (overrides per-task defaults).
        runs: Number of independent runs per task.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_samples: int | None = None,
        runs: int = 1,
    ):
        self.base_url = base_url
        self.model = model
        self.max_samples = max_samples
        self.runs = runs
        self._available = _check_lm_eval_available()

    @property
    def is_available(self) -> bool:
        return self._available

    def validate_endpoint(self) -> dict[str, Any]:
        """Probe the endpoint to make sure it responds.

        Returns:
            Dict with 'ok' and 'error' keys.
        """
        if not self._available:
            return {"ok": False, "error": get_lm_eval_error()}

        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/models",
                timeout=15,
                headers={"User-Agent": "bench-harness-lm-eval"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"ok": True, "models": data.get("data", [])}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_task(
        self,
        task_name: str,
        task_def: dict[str, Any],
        suite_id: str = "public_baseline",
    ) -> LMEvalResult:
        """Run a single lm_eval benchmark task.

        Args:
            task_name: Internal task name (e.g. 'mmlu_college_math').
            task_def: Task definition from TASK_DEFS.
            suite_id: Suite identifier for storage.

        Returns:
            LMEvalResult with accuracy, samples_run, error info.
        """
        if not self._available:
            return LMEvalResult(
                task_id=f"public.{task_name}",
                model_alias=self.model,
                error_count=1,
                error_message=get_lm_eval_error(),
                suite_id=suite_id,
                family=task_def.get("family", "unknown"),
            )

        lm_task = task_def["lm_eval_task"]
        shots = task_def["shots"]
        max_s = self.max_samples or task_def.get("max_samples", 100)

        logger.info(
            "Running lm_eval task=%s shots=%d max_samples=%d model=%s",
            lm_task, shots, max_s, self.model,
        )

        try:
            import lm_eval
            from lm_eval import simple_evaluate

            results = simple_evaluate(
                model="openai_completions",
                model_args={
                    "model": self.model,
                    "baseurl": self.base_url,
                    "truncate": True,
                },
                tasks=[lm_task],
                num_fewshot=shots,
                max_memory_per_gpu=None,
                batch_size=1,
                log_samples=True,
                verbosity="WARNING",
            )

            task_results = results.get("results", {}).get(lm_task, {})
            samples_key = f"requests_complete/{lm_task}"
            samples_run = task_results.get(samples_key, 0)
            if samples_run == 0:
                samples_run = task_results.get("samples", max_s)
            # Use effective_num_docs as a fallback
            if samples_run == 0:
                samples_run = task_results.get("effective_num_docs", task_results.get("samples", max_s))

            accuracy = task_results.get("acc", None)
            if accuracy is not None:
                accuracy = float(accuracy)

            return LMEvalResult(
                task_id=f"public.{task_name}",
                model_alias=self.model,
                accuracy=accuracy,
                samples_run=samples_run,
                error_count=0,
                suite_id=suite_id,
                family=task_def.get("family", "unknown"),
                shots=shots,
                max_samples=max_s,
            )

        except Exception as e:
            logger.error("lm_eval failed for %s: %s", lm_task, e, exc_info=True)
            return LMEvalResult(
                task_id=f"public.{task_name}",
                model_alias=self.model,
                samples_run=0,
                error_count=1,
                error_message=str(e),
                suite_id=suite_id,
                family=task_def.get("family", "unknown"),
                shots=shots,
                max_samples=max_s,
            )

    def run_suite(
        self,
        task_names: list[str] | None = None,
        suite_id: str = "public_baseline",
    ) -> list[LMEvalResult]:
        """Run a subset or all available benchmark tasks.

        Args:
            task_names: Task names to run. None = run all available tasks.
            suite_id: Suite identifier for storage.

        Returns:
            List of LMEvalResult objects.
        """
        if task_names is None:
            task_names = list(TASK_DEFS.keys())

        results: list[LMEvalResult] = []
        for name in task_names:
            if name not in TASK_DEFS:
                logger.warning("Unknown task: %s — skipping", name)
                continue
            result = self.run_task(name, TASK_DEFS[name], suite_id=suite_id)
            results.append(result)
        return results


def run_public_benchmarks(
    endpoint_url: str,
    model_name: str,
    task_names: list[str],
    max_samples: int | None = None,
    runs: int = 1,
    suite_id: str = "public_baseline",
) -> list[LMEvalResult]:
    """Convenience function to run public benchmark tasks.

    Args:
        endpoint_url: OpenAI-compatible API base URL.
        model_name: Model name to evaluate.
        task_names: List of task names from TASK_DEFS.
        max_samples: Override per-task max samples.
        runs: Number of runs per task.
        suite_id: Suite identifier.

    Returns:
        List of LMEvalResult objects.
    """
    adapter = LMEvalAdapter(
        base_url=endpoint_url,
        model=model_name,
        max_samples=max_samples,
        runs=runs,
    )

    if not adapter.is_available:
        results = []
        for name in task_names:
            task_def = TASK_DEFS.get(name, {})
            results.append(LMEvalResult(
                task_id=f"public.{name}",
                model_alias=model_name,
                error_count=1,
                error_message=get_lm_eval_error(),
                suite_id=suite_id,
                family=task_def.get("family", "unknown"),
            ))
        return results

    all_results: list[LMEvalResult] = []
    for _run in range(runs):
        run_results = adapter.run_suite(task_names=task_names, suite_id=suite_id)
        # If multiple runs, keep last run's results or merge
        if len(all_results) == 0:
            all_results = run_results
        else:
            # Update with latest run
            for new_r in run_results:
                for i, old_r in enumerate(all_results):
                    if old_r.task_id == new_r.task_id:
                        all_results[i] = new_r
                        break

    return all_results
