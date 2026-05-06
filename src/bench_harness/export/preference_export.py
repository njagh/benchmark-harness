"""Preference export — DPO/ORPO chosen/rejected JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bench_harness.export.base import (
    get_runs_by_suite,
    get_pairwise_comparisons,
    get_task_by_id,
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


def _build_messages(task: dict, prompt: str, response: str) -> list[dict]:
    """Build messages list with optional system message."""
    messages: list[dict] = []
    system_msg = task.get("input", {}).get("system_message") if task.get("input") else None
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


def export_preference_score_based(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
    min_margin: float = 0.0,
) -> str:
    """Export preference data from score-based runs.

    Groups runs by task_id, picks best (chosen) and worst (rejected)
    model per group. Only includes pairs where margin >= min_margin.

    Output format per line:
    {
        "messages": [...],
        "chosen": {"model": "model-a", "score": 0.9},
        "rejected": {"model": "model-b", "score": 0.3},
        "margin": 0.6,
        "task_id": "...",
        "prompt": "..."
    }
    """
    runs = get_runs_by_suite(db_path, suite_id)

    # Group by task_id
    groups: dict[str, list[dict]] = {}
    for run in runs:
        if run.get("exit_status") != "success":
            continue
        score = run.get("score_primary")
        if score is None:
            continue
        task_id = run.get("task_id", "")
        groups.setdefault(task_id, []).append(run)

    records: list[dict[str, Any]] = []
    for task_id, task_runs in groups.items():
        if len(task_runs) < 2:
            continue

        sorted_runs = sorted(
            task_runs,
            key=lambda r: r.get("score_primary") or 0,
            reverse=True,
        )

        chosen_run = sorted_runs[0]
        rejected_run = sorted_runs[-1]

        chosen_score = float(chosen_run.get("score_primary") or 0)
        rejected_score = float(rejected_run.get("score_primary") or 0)
        margin = chosen_score - rejected_score

        if margin < min_margin:
            continue

        task = get_task_by_id(db_path, task_id) or {}
        prompt = chosen_run.get("prompt", "")

        messages = _build_messages(
            task, prompt, chosen_run.get("raw_response", "")
        )

        record: dict[str, Any] = {
            "messages": messages,
            "chosen": {
                "model": chosen_run.get("model_alias", ""),
                "score": chosen_score,
            },
            "rejected": {
                "model": rejected_run.get("model_alias", ""),
                "score": rejected_score,
            },
            "margin": margin,
            "task_id": task_id,
            "prompt": prompt,
        }
        records.append(record)

    output_file = _resolve_output_path(out_path, "preference_chosen_rejected.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d preference records (score-based) to %s", len(records), output_file)
    return str(output_file)


def export_preference_from_pairwise(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
) -> str:
    """Export preference data from pairwise comparisons.

    Uses the pairwise_comparisons table directly.

    Output format per line:
    {
        "chosen": {"model": "winner_model"},
        "rejected": {"model": "loser_model"},
        "reason": "...",
        "task_id": "...",
        "prompt": "...",
        "confidence": 0.85,
        "margin": "0.2"
    }
    """
    comparisons = get_pairwise_comparisons(db_path, suite_id)

    records: list[dict[str, Any]] = []
    for comp in comparisons:
        winner = comp.get("winner", "")
        model_a = comp.get("model_a", "")
        model_b = comp.get("model_b", "")

        if winner == "A":
            chosen_model = model_a
            rejected_model = model_b
        elif winner == "B":
            chosen_model = model_b
            rejected_model = model_a
        else:
            continue

        task_id = comp.get("task_id", "")
        prompt = comp.get("prompt", "")
        reason = comp.get("reason", "")

        record: dict[str, Any] = {
            "chosen": {
                "model": chosen_model,
            },
            "rejected": {
                "model": rejected_model,
            },
            "reason": reason,
            "task_id": task_id,
            "prompt": prompt,
            "confidence": comp.get("confidence", 0),
            "margin": comp.get("margin", ""),
        }
        records.append(record)

    output_file = _resolve_output_path(out_path, "preference_pairwise.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d preference records (pairwise) to %s", len(records), output_file)
    return str(output_file)
