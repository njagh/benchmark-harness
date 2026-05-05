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

    if result.generated_code is not None:
        record["generated_code"] = result.generated_code
    if result.code_status is not None:
        record["code_status"] = result.code_status
    if result.tests_passed is not None:
        record["tests_passed"] = result.tests_passed
    if result.tests_failed is not None:
        record["tests_failed"] = result.tests_failed
    if result.tests_total is not None:
        record["tests_total"] = result.tests_total
    if result.test_output is not None:
        record["test_output"] = result.test_output
    if result.exit_code is not None:
        record["exit_code"] = result.exit_code

    with open(artifact_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.debug("Wrote artifact: %s", artifact_path)
    return artifact_path


def save_judge_artifact(
    run_id: str,
    out_dir: str,
    raw_response: str | None,
    parsed_scores: dict[str, Any],
    rubric_name: str,
    judge_model: str,
    prompt: str,
) -> dict[str, Path]:
    """Save judge-specific artifacts for a run.

    Writes three files:
    - judge_raw_{run_id}.json: Raw judge LLM response
    - judge_parsed_{run_id}.json: Parsed scores
    - judge_prompt_{run_id}.txt: The prompt sent to the judge

    Args:
        run_id: The run ID this evaluation corresponds to.
        out_dir: Output directory path.
        raw_response: Raw judge model response text.
        parsed_scores: Dict of parsed judge scores.
        rubric_name: Name of the rubric used.
        judge_model: Judge model alias.
        prompt: The prompt sent to the judge.

    Returns:
        Dict mapping file type to Path.
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Save raw response
    raw_path = output / f"judge_raw_{run_id}.json"
    if raw_response is not None:
        raw_path.write_text(raw_response)
        logger.debug("Wrote raw judge response: %s", raw_path)
    paths["raw_response"] = raw_path

    # Save parsed scores
    parsed_path = output / f"judge_parsed_{run_id}.json"
    parsed_data = {
        "run_id": run_id,
        "rubric_name": rubric_name,
        "judge_model": judge_model,
        "scores": parsed_scores,
    }
    parsed_path.write_text(json.dumps(parsed_data, indent=2, default=str))
    logger.debug("Wrote parsed judge scores: %s", parsed_path)
    paths["parsed_scores"] = parsed_path

    # Save judge prompt
    prompt_path = output / f"judge_prompt_{run_id}.txt"
    prompt_path.write_text(prompt)
    logger.debug("Wrote judge prompt: %s", prompt_path)
    paths["prompt"] = prompt_path

    return paths
