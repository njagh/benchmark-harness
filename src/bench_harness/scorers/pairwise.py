"""Pairwise scorer — compares two model responses using an external judge LLM."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

from bench_harness.models.openai_client import OpenAICompatClient
from bench_harness.scorers.base import BaseScorer, ScoreResult, register_scorer, _normalize_task
from bench_harness.tasks.task_schema import Task

from .llm_judge import _build_rubric_prompt, load_rubric, _find_rubrics_file

logger = logging.getLogger(__name__)


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two models.

    Attributes:
        task_id: The task identifier.
        model_a: Name/ID of the first model.
        model_b: Name/ID of the second model.
        winner: Which model won ("a", "b", or "tie").
        margin: Description of the margin of victory.
        confidence: Confidence in the judgment (0-1).
        reason: Human-readable explanation.
        dimension_comparison: Per-dimension score breakdown.
        raw_response: The raw judge output for debugging.
    """

    task_id: str
    model_a: str
    model_b: str
    winner: str
    margin: str
    confidence: float
    reason: str
    dimension_comparison: dict[str, Any]
    raw_response: str | None = None


@register_scorer
class PairwiseScorer(BaseScorer):
    """Scores responses by pairwise comparison using a judge LLM.

    The raw_response is expected to be a JSON string containing both
    model responses and metadata:
    {
        "response_a": "...",
        "response_b": "...",
        "model_a": "...",
        "model_b": "...",
    }
    Or these fields can come from the task dict.

    The scorer sends both responses to a judge model which evaluates
    them against a rubric and returns a comparison.
    """

    name = "pairwise"
    version = "1.0"

    def __init__(
        self,
        client: OpenAICompatClient | None = None,
        rubric_name: str = "default",
        base_url: str | None = None,
        api_key: str = "not-needed",
        model: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.rubric_name = rubric_name

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
            "description": "Default rubric for pairwise comparison",
            "max_scale": 5,
            "dimensions": {
                "overall_quality": {
                    "weight": 1.0,
                    "scale": 5,
                    "description": "Overall quality comparison",
                },
            },
        }

    def _extract_comparison_data(
        self, task: Task, raw_response: str
    ) -> tuple[str, str, str, str]:
        """Extract response_a, response_b, model_a, model_b from raw_response or task.

        Returns:
            (response_a, response_b, model_a, model_b)
        """
        model_a = "model_a"
        model_b = "model_b"
        response_a = ""
        response_b = ""

        data = {}
        try:
            if raw_response.strip():
                data = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Could not parse raw_response as JSON: %s", e)

        response_a = data.get("response_a", "") or ""
        response_b = data.get("response_b", "") or ""
        model_a = data.get("model_a", model_a)
        model_b = data.get("model_b", model_b)

        if not response_a or not response_b:
            task_dict = task.to_dict() if hasattr(task, "to_dict") else task
            if isinstance(task_dict, dict):
                response_a = data.get("response_a", task_dict.get("response_a", response_a))
                response_b = data.get("response_b", task_dict.get("response_b", response_b))
                model_a = data.get("model_a", task_dict.get("model_a", model_a))
                model_b = data.get("model_b", task_dict.get("model_b", model_b))

        return response_a, response_b, model_a, model_b

    def _build_pairwise_prompt(
        self, rubric: dict[str, Any], task: Task, response_a: str, response_b: str, model_a: str, model_b: str
    ) -> str:
        template_dir = _find_template_dir_from_llm_judge()
        template_path = template_dir / "pairwise_judge.md"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Pairwise judge prompt template not found at {template_path}"
            )

        task_description = task.prompt or task.input.user_message if task.input else task.prompt or ""

        context: dict[str, Any] = {
            "rubric_name": rubric["name"],
            "rubric_description": rubric.get("description", ""),
            "max_scale": rubric.get("max_scale", 5),
            "dimensions": rubric["dimensions"],
            "task_description": task_description[:2000] if task_description else "No task description available.",
            "response_a": response_a[:4000],
            "response_b": response_b[:4000],
            "model_a": model_a,
            "model_b": model_b,
        }

        from bench_harness.tasks.prompt_templates import render_template
        return render_template(template_path.read_text(), context)

    def _safe_parse_pairwise(self, parsed: Any) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {
                "winner": "tie",
                "margin": "Unable to determine",
                "confidence": 0.0,
                "reason": "Judge returned non-dict response",
                "dimension_comparison": {},
            }

        winner = parsed.get("winner", "tie")
        if winner not in ("a", "b", "tie"):
            winner = "tie"

        margin = parsed.get("margin", "Unknown margin")
        confidence = parsed.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = parsed.get("reason", "No reason provided")
        dimension_comparison = parsed.get("dimension_comparison", {})

        normalized_dims: dict[str, Any] = {}
        if isinstance(dimension_comparison, dict):
            for dim_name, dim_data in dimension_comparison.items():
                if isinstance(dim_data, dict):
                    normalized_dims[dim_name] = {
                        "score_a": dim_data.get("score_a", 0.0),
                        "score_b": dim_data.get("score_b", 0.0),
                        "winner": dim_data.get("winner", "tie"),
                    }
                else:
                    normalized_dims[dim_name] = {
                        "score_a": 0.0,
                        "score_b": 0.0,
                        "winner": "tie",
                    }

        return {
            "winner": winner,
            "margin": margin,
            "confidence": confidence,
            "reason": reason,
            "dimension_comparison": normalized_dims,
        }

    def _build_pairwise_score_result(
        self,
        pairwise_data: dict[str, Any],
        rubric: dict[str, Any],
        task_id: str,
        model_a: str,
        model_b: str,
        raw_output: str,
        start: float,
    ) -> ScoreResult:
        winner = pairwise_data["winner"]
        confidence = pairwise_data["confidence"]
        reason = pairwise_data["reason"]
        margin = pairwise_data["margin"]

        passed = winner in ("a", "b")
        score = round(confidence, 4) if winner in ("a", "b") else 0.5

        details: dict[str, Any] = {
            "winner": winner,
            "margin": margin,
            "confidence": confidence,
            "reason": reason,
            "dimension_comparison": pairwise_data["dimension_comparison"],
            "rubric_name": rubric["name"],
            "task_id": task_id,
            "model_a": model_a,
            "model_b": model_b,
        }

        explanation = (
            f"{winner.upper()} wins: {margin} ({reason})"
        )

        return ScoreResult(
            scorer_name=self.name,
            scorer_version=self.version,
            score=score,
            passed=passed,
            details=details,
            explanation=explanation,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 1),
        )

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
            response_a, response_b, model_a, model_b = self._extract_comparison_data(task, raw_response)
        except Exception as e:
            logger.error("Failed to extract comparison data: %s", e)
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Failed to extract comparison data: {e}",
            )

        if not response_a or not response_b:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={},
                explanation="Missing one or both responses for pairwise comparison",
            )

        try:
            rubric = self._load_rubric()
        except (KeyError, FileNotFoundError) as e:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Failed to load rubric: {e}",
            )

        try:
            prompt = self._build_pairwise_prompt(rubric, task, response_a, response_b, model_a, model_b)
        except FileNotFoundError as e:
            return ScoreResult(
                scorer_name=self.name,
                scorer_version=self.version,
                score=0.0,
                passed=False,
                details={"error": str(e)},
                explanation=f"Failed to build pairwise prompt: {e}",
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            pairwise_data = loop.run_until_complete(self._score_pairwise(prompt))
        except Exception as e:
            logger.error("Pairwise scoring failed: %s", e)
            pairwise_data = {
                "winner": "tie",
                "margin": "Error during scoring",
                "confidence": 0.0,
                "reason": f"Judge error: {e}",
                "dimension_comparison": {},
            }
        finally:
            loop.close()

        task_id = task.id if hasattr(task, "id") else task.get("id", "unknown")
        return self._build_pairwise_score_result(
            pairwise_data, rubric, task_id, model_a, model_b, None, start,
        )

    async def _score_pairwise(self, prompt: str) -> dict[str, Any]:
        result = await self.client.chat_complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
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

        return self._safe_parse_pairwise(parsed)


def _find_template_dir_from_llm_judge():
    """Find the judge_prompts directory by walking up from this module."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent.parent
    for candidate in [
        root / "configs" / "judge_prompts",
        Path.cwd() / "configs" / "judge_prompts",
    ]:
        if candidate.exists():
            return candidate
    return root / "configs" / "judge_prompts"
