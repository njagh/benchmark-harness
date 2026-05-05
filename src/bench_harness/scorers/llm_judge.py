"""LLM Judge scorer — evaluates responses using an external judge LLM against a rubric."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.prompt_templates import render_template
from bench_harness.tasks.task_schema import Task

logger = logging.getLogger(__name__)


def _find_rubrics_file() -> Path:
    root = Path(__file__).resolve().parent.parent.parent.parent
    for candidate in [
        root / "configs" / "judge_rubrics.yaml",
        Path.cwd() / "configs" / "judge_rubrics.yaml",
    ]:
        if candidate.exists():
            return candidate
    return root / "configs" / "judge_rubrics.yaml"


def _find_template_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent.parent
    for candidate in [
        root / "configs" / "judge_prompts",
        Path.cwd() / "configs" / "judge_prompts",
    ]:
        if candidate.exists():
            return candidate
    return root / "configs" / "judge_prompts"


def load_rubric(rubric_name: str, rubrics_path: Path | None = None) -> dict[str, Any]:
    if rubrics_path is None:
        rubrics_path = _find_rubrics_file()

    data = yaml.safe_load(rubrics_path.read_text())
    rubrics = data.get("rubrics", {})
    if rubric_name not in rubrics:
        raise KeyError(
            f"Rubric '{rubric_name}' not found in {rubrics_path}. "
            f"Available: {list(rubrics.keys())}"
        )
    return rubrics[rubric_name]


def _build_rubric_prompt(rubric: dict[str, Any], task: Task, raw_response: str) -> str:
    template_path = _find_template_dir() / "rubric_judge.md"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Judge prompt template not found at {template_path}"
        )

    task_description = task.prompt or task.input.user_message if task.input else task.prompt or ""

    context: dict[str, Any] = {
        "rubric_name": rubric["name"],
        "rubric_description": rubric.get("description", ""),
        "max_scale": rubric.get("max_scale", 5),
        "dimensions": rubric["dimensions"],
        "task_description": task_description[:2000] if task_description else "No task description available.",
        "raw_response": raw_response[:8000],
    }

    return render_template(template_path.read_text(), context)


async def _call_judge(client: OpenAICompatClient, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    result = await client.chat_complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    error = result.get("error")
    if error:
        raise RuntimeError(f"Judge API error: {error}")

    content = result.get("content")
    if not content:
        raise RuntimeError("Judge returned empty response")

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Judge returned invalid JSON: {content[:200]!r}") from e

    return parsed


@register_scorer
class LLMJudgeScorer(BaseScorer):
    """Evaluates a model's response using an external judge LLM against a rubric.

    The scorer sends the rubric definition, task info, and model response
    to a judge model (typically a larger/more capable model) which returns
    dimension scores with reasoning.

    Supports self-consistency: the judge is called multiple times and scores
    are averaged with stddev computed per dimension.
    """

    name = "llm_judge"
    version = "1.0"

    def __init__(
        self,
        client: OpenAICompatClient | None = None,
        rubric_name: str = "default",
        self_consistency_rounds: int = 1,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str = "not-needed",
        model: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.rubric_name = rubric_name
        self.self_consistency_rounds = self_consistency_rounds
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens

        if client is not None:
            self.client: OpenAICompatClient | None = client
        elif base_url is not None:
            self.client = OpenAICompatClient(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        else:
            self.client = None

    def _load_rubric(self) -> dict[str, Any]:
        try:
            rubrics_path = _find_rubrics_file()
            return load_rubric(self.rubric_name, rubrics_path)
        except KeyError as e:
            logger.warning("Rubric not found: %s", e)
            return self._fallback_rubric()

    def _fallback_rubric(self) -> dict[str, Any]:
        return {
            "name": "default",
            "description": "Default rubric for general responses",
            "max_scale": 5,
            "dimensions": {
                "overall_quality": {
                    "weight": 1.0,
                    "scale": 5,
                    "description": "Overall quality of the response",
                },
            },
        }

    def _safe_parse_dimension_scores(self, parsed: Any) -> tuple[dict[str, Any], float, str | None]:
        if not isinstance(parsed, dict):
            return {}, 0.0, "Judge returned non-dict response"

        dimensions = parsed.get("dimensions", {})
        total_score = parsed.get("total_score", 0.0)
        summary = parsed.get("summary")

        if not isinstance(dimensions, dict):
            return {}, total_score, summary

        normalized_dims: dict[str, Any] = {}
        rubric_dims = self._load_rubric().get("dimensions", {})
        expected_dim_names = set(rubric_dims.keys())
        found_dim_names = set(dimensions.keys())

        missing = expected_dim_names - found_dim_names
        for dim_name in missing:
            dim_spec = rubric_dims[dim_name]
            max_scale = dim_spec.get("scale", dim_spec.get("max_scale", 5))
            normalized_dims[dim_name] = {
                "score": 0.0,
                "reason": "Dimension missing from judge output",
            }

        for dim_name, dim_data in dimensions.items():
            if isinstance(dim_data, dict):
                normalized_dims[dim_name] = {
                    "score": dim_data.get("score", 0.0),
                    "reason": dim_data.get("reason", ""),
                }
            elif isinstance(dim_data, (int, float)):
                normalized_dims[dim_name] = {
                    "score": float(dim_data),
                    "reason": "Score provided without reason",
                }
            else:
                normalized_dims[dim_name] = {
                    "score": 0.0,
                    "reason": "Invalid dimension data type",
                }

        return normalized_dims, total_score, summary

    def _compute_weighted_score(self, dims: dict[str, Any], rubric: dict[str, Any]) -> float:
        rubric_dims = rubric.get("dimensions", {})
        total_weight = 0.0
        weighted_sum = 0.0

        for dim_name, dim_spec in rubric_dims.items():
            weight = dim_spec.get("weight", 1.0 / len(rubric_dims)) if rubric_dims else 1.0
            max_scale = dim_spec.get("scale", dim_spec.get("max_scale", 5))
            dim_data = dims.get(dim_name, {})

            if isinstance(dim_data, dict):
                raw_score = dim_data.get("score", 0.0)
            else:
                raw_score = 0.0

            normalized = raw_score / max_scale if max_scale > 0 else 0.0
            weighted_sum += normalized * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _build_score_result(
        self,
        dimension_scores: dict[str, Any],
        total_score_raw: float,
        summary: str | None,
        rubric: dict[str, Any],
        stddevs: dict[str, float] | None,
        rounds: int,
        start: float,
        explanations: list[str] | None = None,
        error: str | None = None,
    ) -> ScoreResult:
        weight_score = self._compute_weighted_score(dimension_scores, rubric)
        max_scale = rubric.get("max_scale", 5)
        normalized_score = min(1.0, weight_score / max_scale) if max_scale > 0 else weight_score

        details: dict[str, Any] = {
            "dimensions": {},
            "total_score_raw": total_score_raw,
            "rubric_name": rubric["name"],
            "rounds": rounds,
        }

        for dim_name, dim_data in dimension_scores.items():
            dim_details: dict[str, Any] = {}
            if isinstance(dim_data, dict):
                dim_details["score"] = dim_data.get("score", 0.0)
                dim_details["reason"] = dim_data.get("reason", "")
            else:
                dim_details["score"] = 0.0
                dim_details["reason"] = "Invalid dimension data"

            if stddevs and dim_name in stddevs:
                dim_details["stddev"] = stddevs[dim_name]

            details["dimensions"][dim_name] = dim_details

        if explanations:
            details["round_explanations"] = explanations

        passed = normalized_score >= 0.5
        explanation = summary or f"Weighted score: {weight_score:.2f} / {max_scale}"
        if error:
            explanation = error

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=round(normalized_score, 4),
            passed=passed,
            details=details,
            explanation=explanation,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 1),
        )

    async def _score_single_round(self, prompt: str) -> tuple[dict[str, Any], float, str | None, str | None]:
        parsed = await _call_judge(
            self.client,
            prompt,
            self.judge_temperature,
            self.judge_max_tokens,
        )
        dims, total_score, summary = self._safe_parse_dimension_scores(parsed)
        return dims, total_score, summary, None

    async def _score_with_consistency(self, prompt: str) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], list[str | None], list[str | None]]:
        raw_dims: list[dict[str, Any]] = []
        explanations: list[str | None] = []
        errors: list[str | None] = []

        for i in range(self.self_consistency_rounds):
            try:
                dims, total_score, summary, error = await self._score_single_round(prompt)
                raw_dims.append(dims)
                explanations.append(summary)
                errors.append(error)
                if error:
                    logger.warning("Judge round %d/%d error: %s", i + 1, self.self_consistency_rounds, error)
            except Exception as e:
                raw_dims.append({})
                explanations.append(f"Round {i+1} error: {e}")
                errors.append(str(e))
                logger.error("Judge round %d/%d failed: %s", i + 1, self.self_consistency_rounds, e)

        dim_sets = [set(d.keys()) for d in raw_dims if d]
        all_dims = set()
        for d in dim_sets:
            all_dims |= d

        aggregated: dict[str, float] = {}
        stddevs: dict[str, float] = {}

        for dim_name in all_dims:
            scores = []
            for dims in raw_dims:
                dim_data = dims.get(dim_name, {})
                if isinstance(dim_data, dict):
                    scores.append(float(dim_data.get("score", 0.0)))
                elif isinstance(dim_data, (int, float)):
                    scores.append(float(dim_data))
                else:
                    scores.append(0.0)

            if scores:
                mean = sum(scores) / len(scores)
                aggregated[dim_name] = mean
                if len(scores) > 1:
                    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                    stddevs[dim_name] = variance ** 0.5
                else:
                    stddevs[dim_name] = 0.0

        total_scores = []
        for dims in raw_dims:
            if isinstance(dims, dict) and "total_score" in dims:
                total_scores.append(float(dims["total_score"]))
        if total_scores:
            aggregated["_total_score"] = sum(total_scores) / len(total_scores)
        else:
            aggregated["_total_score"] = 0.0

        return aggregated, stddevs, raw_dims, explanations, errors

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        task = _normalize_task(task)
        start = time.perf_counter()

        if self.client is None:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={},
                explanation="No judge client configured",
            )

        try:
            rubric = self._load_rubric()
        except (KeyError, FileNotFoundError, yaml.YAMLError) as e:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Failed to load rubric: {e}",
            )

        try:
            prompt = _build_rubric_prompt(rubric, task, raw_response)
        except FileNotFoundError as e:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Failed to build judge prompt: {e}",
            )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if self.self_consistency_rounds > 1:
                    aggregated, stddevs, raw_dims, explanations, errors = loop.run_until_complete(
                        self._score_with_consistency(prompt)
                    )

                    dims: dict[str, Any] = {}
                    for dim_name, mean_score in aggregated.items():
                        if dim_name.startswith("_"):
                            continue
                        dim_stddev = stddevs.get(dim_name, 0.0)
                        dims[dim_name] = {
                            "score": round(mean_score, 2),
                            "reason": f"Self-consistent across {self.self_consistency_rounds} rounds (stddev: {dim_stddev:.2f})",
                        }

                    total_raw = aggregated.get("_total_score", 0.0)
                    summary = f"Averaged {self.self_consistency_rounds} rounds with stddev computed per dimension"

                    non_error_explanations = [e for e in explanations if e is not None]
                    has_errors = any(e is not None for e in errors)

                    error_msg = None
                    if has_errors and not non_error_explanations:
                        error_msg = "All consistency rounds failed"

                    result = self._build_score_result(
                        dims, total_raw, summary, rubric, stddevs,
                        self.self_consistency_rounds, start,
                        explanations if non_error_explanations else None,
                        error_msg,
                    )
                    return result
                else:
                    dims, total_score, summary, error = loop.run_until_complete(
                        self._score_single_round(prompt)
                    )

                    if error:
                        return self._build_score_result(
                            {}, 0.0, summary, rubric, None, 1, start,
                            [error], error=error,
                        )

                    return self._build_score_result(
                        dims, total_score, summary, rubric, None, 1, start,
                        [summary] if summary else None,
                    )
            finally:
                loop.close()

        except Exception as e:
            logger.error("LLM judge scoring failed: %s", e)
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Judge scoring error: {e}",
                duration_ms=round((time.perf_counter() - start) * 1000.0, 1),
            )
