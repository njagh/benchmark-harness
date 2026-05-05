"""JSONL artifact output for benchmark runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bench_harness.runners.completion_runner import RunResult

logger = logging.getLogger(__name__)


def save_run_artifact(result: RunResult, out_dir: str) -> Path:
    """Write a single run result as a JSONL line to the output directory.

    Uses a consolidated runs.jsonl file in append mode.

    Args:
        result: The run result to write.
        out_dir: Output directory path.

    Returns:
        Path to the artifact file.
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    artifact_path = output / "runs.jsonl"

    record = {
        "run_id": result.run_id,
        "suite_id": result.suite_id,
        "task_id": result.task_id,
        "model_alias": result.model_alias,
        "model_backend": result.model_backend,
        "prompt": result.prompt,
        "raw_response": result.raw_response,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "ttft_ms": result.ttft_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "total_wall_ms": result.total_wall_ms,
        "exit_status": result.exit_status,
        "error_message": result.error_message,
        "created_at": result.created_at,
    }

    with open(artifact_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.debug("Wrote artifact: %s", artifact_path)
    return artifact_path
