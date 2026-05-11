"""Service for triggering exports from the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bench_harness.export.sft_export import export_sft
from bench_harness.export.preference_export import export_preference_score_based
from bench_harness.export.regression_export import export_regression
from bench_harness.export.judge_export import export_judge


def export_sft_data(
    db_path: str,
    suite_id: str,
    min_score: float = 0.0,
) -> str:
    """Export SFT data. Returns output path."""
    out_dir = str(Path(db_path).parent / "exports")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = export_sft(db_path, suite_id, out_dir + "/sft_openai_messages.jsonl", min_score=min_score)
    return str(path)


def export_preference_data(
    db_path: str,
    suite_id: str,
    min_margin: float = 0.1,
) -> str:
    """Export preference data. Returns output path."""
    out_dir = str(Path(db_path).parent / "exports")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = export_preference_score_based(db_path, suite_id, out_dir + "/preference_chosen_rejected.jsonl", min_margin=min_margin)
    return str(path)


def export_regression_data(
    db_path: str,
    suite_id: str,
) -> str:
    """Export regression data. Returns output path."""
    out_dir = str(Path(db_path).parent / "exports")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = export_regression(db_path, suite_id, out_dir + "/regression_tasks.yaml")
    return str(path)


def export_judge_data(
    db_path: str,
    suite_id: str,
) -> str:
    """Export judge data. Returns output path."""
    out_dir = str(Path(db_path).parent / "exports")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = export_judge(db_path, suite_id, out_dir + "/judge_scores.jsonl")
    return str(path)
