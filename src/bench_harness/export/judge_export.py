"""Judge export — JSONL of judge evaluation records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bench_harness.export.base import (
    get_judge_evaluations,
    get_pairwise_comparisons,
    get_runs_by_suite,
)

logger = logging.getLogger(__name__)


def _resolve_output_path(out_path: str | None, filename: str) -> Path:
    """Resolve output path, defaulting to exports/{filename}."""
    if out_path is None:
        return Path("exports") / filename
    p = Path(out_path)
    if p.is_dir():
        return p / filename
    return p.parent / filename if p.parent != p else p


def export_judge(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
) -> str:
    """Export judge evaluations as JSONL.

    Combines judge_evaluations table with runs data.

    Output format per line:
    {
        "run_id": "...",
        "task_id": "...",
        "model_alias": "...",
        "judge_model": "...",
        "rubric_name": "...",
        "score": "...",
        "dimensions": {"correctness": 4, "completeness": 3, ...},
        "explanation": "...",
        "is_pairwise": false,
        "pairwise": null
    }
    """
    judge_evals = get_judge_evaluations(db_path, suite_id)
    runs = get_runs_by_suite(db_path, suite_id)

    # Build a lookup of runs by run_id for enrichment
    runs_by_id = {r.get("run_id", ""): r for r in runs}

    records: list[dict[str, Any]] = []
    for eval_rec in judge_evals:
        run_id = eval_rec.get("run_id", "")
        run = runs_by_id.get(run_id, {})

        dimensions_json = eval_rec.get("dimensions_json")
        dimensions: dict[str, Any] = {}
        if dimensions_json:
            try:
                dimensions = json.loads(dimensions_json)
            except (json.JSONDecodeError, TypeError):
                dimensions = {}

        record: dict[str, Any] = {
            "run_id": run_id,
            "task_id": eval_rec.get("task_id", ""),
            "model_alias": eval_rec.get("model_alias", ""),
            "judge_model": eval_rec.get("judge_model", ""),
            "rubric_name": eval_rec.get("rubric_name", ""),
            "score": eval_rec.get("score", 0),
            "dimensions": dimensions,
            "explanation": eval_rec.get("explanation", ""),
            "is_pairwise": False,
            "pairwise": None,
        }

        # Add run-level enrichment
        if run:
            record["prompt"] = run.get("prompt", "")
            record["raw_response"] = run.get("raw_response", "")
            record["score_primary"] = run.get("score_primary")

        records.append(record)

    output_file = _resolve_output_path(out_path, "judge_scores.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d judge evaluation records to %s", len(records), output_file)
    return str(output_file)


def export_judge_pairwise(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
) -> str:
    """Export pairwise comparisons from judge as JSONL.

    Output format per line:
    {
        "task_id": "...",
        "model_a": "model-a",
        "model_b": "model-b",
        "winner": "A",
        "margin": "0.15",
        "confidence": 0.85,
        "reason": "...",
        "dimension_comparison": {"correctness": {"A": 4, "B": 3}},
        "is_pairwise": true
    }
    """
    comparisons = get_pairwise_comparisons(db_path, suite_id)

    records: list[dict[str, Any]] = []
    for comp in comparisons:
        dim_comparison_json = comp.get("dimension_comparison_json")
        dimension_comparison: dict[str, Any] = {}
        if dim_comparison_json:
            try:
                dimension_comparison = json.loads(dim_comparison_json)
            except (json.JSONDecodeError, TypeError):
                dimension_comparison = {}

        record: dict[str, Any] = {
            "task_id": comp.get("task_id", ""),
            "model_a": comp.get("model_a", ""),
            "model_b": comp.get("model_b", ""),
            "winner": comp.get("winner", ""),
            "margin": comp.get("margin", ""),
            "confidence": comp.get("confidence", 0),
            "reason": comp.get("reason", ""),
            "dimension_comparison": dimension_comparison,
            "is_pairwise": True,
            "judge_model": comp.get("judge_model", ""),
            "human_override": comp.get("human_override", 0),
            "human_winner": comp.get("human_winner"),
        }
        records.append(record)

    output_file = _resolve_output_path(out_path, "judge_pairwise.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d pairwise records to %s", len(records), output_file)
    return str(output_file)
