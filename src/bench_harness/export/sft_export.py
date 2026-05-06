"""Supervised Fine-Tuning export — OpenAI messages-format JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bench_harness.export.base import get_runs_by_suite, get_task_by_id

logger = logging.getLogger(__name__)


def _resolve_output_path(out_path: str | None, filename: str) -> Path:
    """Resolve output path, defaulting to exports/{filename}."""
    if out_path is None:
        return Path("exports") / filename
    p = Path(out_path)
    if p.is_dir():
        return p / filename
    return p.parent / filename if p.parent != p else p


def export_sft(
    db_path: str,
    suite_id: str,
    out_path: str | None = None,
    min_score: float = 0.0,
    include_system_messages: bool = True,
) -> str:
    """Export successful runs as OpenAI-messages-format JSONL for SFT.

    Format per line:
    {
        "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "model": "agent-code",
        "family": "docker_compose",
        "task_id": "...",
        "score": 1.0,
        "prompt_style": "plain",
        "quantization": "FP8"
    }

    Filters:
    - exit_status == "success"
    - score_primary >= min_score

    For code tasks: also include generated_code in messages if available.
    """
    runs = get_runs_by_suite(db_path, suite_id)

    records: list[dict[str, Any]] = []
    for run in runs:
        exit_status = run.get("exit_status", "")
        if exit_status != "success":
            continue

        score_primary = run.get("score_primary")
        if score_primary is None:
            continue
        if float(score_primary) < min_score:
            continue

        raw_response = run.get("raw_response", "")
        if not raw_response:
            continue

        task_id = run.get("task_id", "")
        task = get_task_by_id(db_path, task_id)
        if task is None:
            task = {}

        family = task.get("family", "unknown")
        prompt = run.get("prompt", "")
        model_alias = run.get("model_alias", "")
        prompt_style = run.get("prompt_style")
        quantization = run.get("quantization")

        messages: list[dict[str, str]] = []

        if include_system_messages:
            system_msg = task.get("input", {}).get("system_message") if task.get("input") else None
            if system_msg:
                messages.append({"role": "system", "content": system_msg})

        messages.append({"role": "user", "content": prompt})

        if run.get("generated_code"):
            assistant_content = run["generated_code"]
        else:
            assistant_content = raw_response

        messages.append({"role": "assistant", "content": assistant_content})

        record: dict[str, Any] = {
            "messages": messages,
            "model": model_alias,
            "family": family,
            "task_id": task_id,
            "score": score_primary,
        }

        if prompt_style:
            record["prompt_style"] = prompt_style
        if quantization:
            record["quantization"] = quantization

        records.append(record)

    output_file = _resolve_output_path(out_path, "sft_openai_messages.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d SFT records to %s", len(records), output_file)
    return str(output_file)
