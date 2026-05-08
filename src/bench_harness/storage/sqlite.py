"""SQLite storage for benchmark run results.

Supports schema migration for adding new timing and token metric columns.
"""

from __future__ import annotations

import json
import logging
import datetime as dt
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from sqlite_utils import Database

from bench_harness.runners.completion_runner import RunResult

try:
    from bench_harness.storage.config import StorageConfig
except ImportError:
    StorageConfig = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Persist benchmark run results to SQLite using sqlite-utils.

    Supports schema migration for adding new columns (M3 timing fields).
    """

    # Columns that should exist on the runs table (M3 timing fields)
    # Use sqlite-utils COLUMN_TYPE_MAPPING keys (Python types or mapped strings)
    _RUNS_EXTRA_COLUMNS = {
        "tokens_per_second": "float",
        "tokens_per_second_total": "float",
        "token_source": "str",
        "chunk_count": "int",
    }

    # M4 scoring columns
    _RUNS_SCORE_COLUMNS = {
        "score_primary": "float",
        "score_secondary": "text",
        "scorer_version": "text",
        "score_explanation": "text",
    }

    # M7 judge columns on runs table
    _RUNS_JUDGE_COLUMNS = {
        "judge_score": "float",
        "judge_explanation": "text",
        "judge_dimensions": "text",
        "judge_model": "text",
        "human_override": "int",
        "human_score": "float",
        "human_note": "text",
    }

    # M8 prompt style column
    _RUNS_STYLE_COLUMNS = {
        "prompt_style": "text",
    }

    # M9 context size columns
    _RUNS_CONTEXT_COLUMNS = {
        "context_tokens": "str",
        "estimated_prompt_tokens": "int",
    }

    # M10 quantization column
    _RUNS_QUANT_COLUMNS = {
        "quantization": "str",
    }

    # M11 command safety columns
    _RUNS_SAFETY_COLUMNS = {
        "safety_score": "float",
        "safety_details": "text",
    }

    # M14 identity stamp columns on runs table
    _RUNS_IDENTITY_COLUMNS = {
        "requested_alias": "text",
        "litellm_model_name": "text",
        "openai_models_id": "text",
        "vllm_served_model_name": "text",
        "vllm_container_name": "text",
        "hf_model_id": "text",
        "backend_url": "text",
        "server_start_time": "text",
        "speculative_decoding_enabled": "int",
    }

    # M7 columns to add to score_details table
    _SCORE_DETAILS_EXTRA_COLUMNS = {
        "human_override": "int",
        "human_score": "float",
        "human_note": "text",
    }

    # Columns that should exist on the environments table (M3 env fields)
    _ENV_EXTRA_COLUMNS = {
        "vllm_version": "str",
        "litellm_version": "str",
        "model_path": "str",
        "quantization": "str",
        "max_model_len": "int",
        "gpu_memory_utilization": "float",
        "served_port": "int",
    }

    def __init__(
        self,
        db_path: str | None = None,
        config: "StorageConfig | None" = None,
    ) -> None:
        if config is not None:
            self.db_path = config.results_runs / "benchmark.db"
        elif db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path("runs/benchmark.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(str(self.db_path), memory=False)

    def init(self) -> None:
        """Create tables and migrate schema if needed."""
        self._create_runs_table()
        self._create_environments_table()
        self._migrate_runs_schema()
        self._migrate_environments_schema()
        self._create_run_timings_table()
        self._migrate_run_timings_schema()
        self._create_score_details_table()
        self._create_indexes()
        self._create_judge_evaluations_table()
        self._create_pairwise_comparisons_table()
        logger.info("SQLite store initialized: %s", self.db_path)

    def _create_runs_table(self) -> None:
        """Create the runs table if it doesn't exist."""
        self.db["runs"].create(
            {
                "run_id": str,
                "suite_id": str,
                "task_id": str,
                "model_alias": str,
                "model_backend": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
                "ttft_ms": float,
                "prefill_ms": float,
                "decode_ms": float,
                "total_wall_ms": float,
                "exit_status": str,
                "raw_response": str,
                "score_primary": float,
                "scorer_version": str,
                "score_secondary": str,
                "score_explanation": str,
                "error_message": str,
                "created_at": str,
            },
            pk="run_id",
            if_not_exists=True,
        )

    def _create_environments_table(self) -> None:
        """Create the environments table if it doesn't exist."""
        self.db["environments"].create(
            {
                "id": str,
                "host": str,
                "os": str,
                "gpu_name": str,
                "cuda_version": str,
                "harness_commit": str,
                "created_at": str,
            },
            pk="id",
            if_not_exists=True,
        )

    def _migrate_runs_schema(self) -> None:
        """Add any missing timing/token and scoring columns to the runs table."""
        existing = set(self.db["runs"].columns)
        for col, col_type in self._RUNS_EXTRA_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e
                    )
        for col, col_type in self._RUNS_SCORE_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding score column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e
                    )
        for col, col_type in self._RUNS_JUDGE_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding judge column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )
        for col, col_type in self._RUNS_STYLE_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding style column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )
        for col, col_type in self._RUNS_CONTEXT_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding context column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )
        for col, col_type in self._RUNS_QUANT_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding quantization column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )
        for col, col_type in self._RUNS_SAFETY_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding safety column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )
        for col, col_type in self._RUNS_IDENTITY_COLUMNS.items():
            if col not in existing:
                logger.info("Migrating runs table: adding identity column %s", col)
                try:
                    self.db["runs"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to runs table: %s", col, e,
                    )

    def _migrate_environments_schema(self) -> None:
        """Add any missing env columns to the environments table."""
        existing = set(self.db["environments"].columns)
        for col, col_type in self._ENV_EXTRA_COLUMNS.items():
            if col not in existing:
                logger.info(
                    "Migrating environments table: adding column %s", col
                )
                try:
                    self.db["environments"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to environments table: %s",
                        col, e,
                    )

    def _create_run_timings_table(self) -> None:
        """Create the run_timings table for detailed timing data."""
        self.db["run_timings"].create(
            {
                "run_id": str,
                "suite_id": str,
                "task_id": str,
                "model_alias": str,
                "ttft_ms": float,
                "prefill_ms": float,
                "decode_ms": float,
                "total_wall_ms": float,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
                "tokens_per_second": float,
                "tokens_per_second_total": float,
                "token_source": str,
                "context_tokens": str,
                "estimated_prompt_tokens": int,
                "quantization": str,
                "chunk_count": int,
                "created_at": str,
                # M14 identity stamp
                "requested_alias": str,
                "litellm_model_name": str,
                "openai_models_id": str,
                "vllm_served_model_name": str,
                "vllm_container_name": str,
                "hf_model_id": str,
                "backend_url": str,
                "server_start_time": str,
                "speculative_decoding_enabled": int,
            },
            pk="run_id",
            if_not_exists=True,
        )

    def _migrate_run_timings_schema(self) -> None:
        """Add identity stamp columns to run_timings table."""
        existing = set(self.db["run_timings"].columns)
        identity_cols = {
            "requested_alias": "text",
            "litellm_model_name": "text",
            "openai_models_id": "text",
            "vllm_served_model_name": "text",
            "vllm_container_name": "text",
            "hf_model_id": "text",
            "backend_url": "text",
            "server_start_time": "text",
            "speculative_decoding_enabled": "int",
        }
        for col, col_type in identity_cols.items():
            if col not in existing:
                logger.info(
                    "Migrating run_timings table: adding identity column %s", col
                )
                try:
                    self.db["run_timings"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to run_timings table: %s",
                        col, e,
                    )

    def _create_score_details_table(self) -> None:
        """Create the score_details table for per-scorer breakdown."""
        self.db["score_details"].create(
            {
                "id": int,
                "run_id": str,
                "scorer_name": str,
                "score": float,
                "passed": int,
                "details_json": str,
                "explanation": str,
            },
            pk="id",
            if_not_exists=True,
        )
        # Migrate human override columns onto score_details
        existing = set(self.db["score_details"].columns)
        for col, col_type in self._SCORE_DETAILS_EXTRA_COLUMNS.items():
            if col not in existing:
                logger.info(
                    "Migrating score_details table: adding column %s", col
                )
                try:
                    self.db["score_details"].add_column(col, col_type)
                except Exception as e:
                    logger.warning(
                        "Failed to add column %s to score_details table: %s",
                        col, e,
                    )

    def _create_indexes(self) -> None:
        """Create indexes for common query patterns."""
        self.db["runs"].create_index(
            ["suite_id", "model_alias"], if_not_exists=True
        )
        self.db["run_timings"].create_index(
            ["suite_id", "model_alias"], if_not_exists=True
        )

    def _create_judge_evaluations_table(self) -> None:
        """Create the judge_evaluations table for LLM judge data."""
        self.db["judge_evaluations"].create(
            {
                "id": int,
                "run_id": str,
                "task_id": str,
                "model_alias": str,
                "judge_model": str,
                "rubric_name": str,
                "score": str,
                "dimensions_json": str,
                "explanation": str,
                "raw_response": str,
                "created_at": str,
            },
            pk="id",
            if_not_exists=True,
        )

    def _create_pairwise_comparisons_table(self) -> None:
        """Create the pairwise_comparisons table for pairwise judge data."""
        self.db["pairwise_comparisons"].create(
            {
                "id": int,
                "task_id": str,
                "model_a": str,
                "model_b": str,
                "winner": str,
                "margin": str,
                "confidence": float,
                "reason": str,
                "dimension_comparison_json": str,
                "raw_judge_response": str,
                "judge_model": str,
                "human_override": int,
                "human_winner": str,
                "human_note": str,
                "created_at": str,
            },
            pk="id",
            if_not_exists=True,
        )
        # Indexes for pairwise comparisons
        self.db["pairwise_comparisons"].create_index(
            ["task_id", "judge_model"], if_not_exists=True
        )

    def save_run(self, result: RunResult) -> None:
        """Save a single run result with all timing and token fields."""
        judge_dimensions_json = None
        if result.judge_dimensions is not None:
            judge_dimensions_json = json.dumps(result.judge_dimensions)

        score_secondary_json = None
        if result.score_secondary is not None:
            score_secondary_json = json.dumps(result.score_secondary)

        self.db["runs"].insert(
            {
                "run_id": result.run_id,
                "suite_id": result.suite_id,
                "task_id": result.task_id,
                "model_alias": result.model_alias,
                "model_backend": result.model_backend,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "ttft_ms": result.ttft_ms,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
                "total_wall_ms": result.total_wall_ms,
                "exit_status": result.exit_status,
                "raw_response": result.raw_response,
                "score_primary": result.score_primary,
                "scorer_version": result.scorer_version,
                "score_secondary": score_secondary_json,
                "score_explanation": result.score_explanation,
                "error_message": result.error_message,
                "created_at": result.created_at,
                # M3 timing fields
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_total": result.tokens_per_second_total,
                "token_source": result.token_source,
                "chunk_count": 0,
                # M7 judge fields
                "judge_score": result.judge_score,
                "judge_explanation": result.judge_explanation,
                "judge_dimensions": judge_dimensions_json,
                "judge_model": result.judge_model,
                "human_override": 1 if result.human_override else 0,
                "human_score": result.human_score,
                "human_note": result.human_note,
                # M8 prompt style
                "prompt_style": result.prompt_style,
                # M9 context size
                "context_tokens": result.context_tokens,
                "estimated_prompt_tokens": result.estimated_prompt_tokens,
                # M10 quantization
                "quantization": result.quantization,
                # M11 command safety
                "safety_score": result.safety_score,
                "safety_details": (
                    json.dumps(result.safety_details)
                    if result.safety_details is not None
                    else None
                ),
                # M14 identity stamp
                "requested_alias": result.requested_alias,
                "litellm_model_name": result.litellm_model_name,
                "openai_models_id": result.openai_models_id,
                "vllm_served_model_name": result.vllm_served_model_name,
                "vllm_container_name": result.vllm_container_name,
                "hf_model_id": result.hf_model_id,
                "backend_url": result.backend_url,
                "server_start_time": result.server_start_time,
                "speculative_decoding_enabled": (
                    1 if result.speculative_decoding_enabled else 0
                ),
            },
        )

        # Also save to run_timings for detailed timing analysis
        self.save_run_timing(result)

        # Save score details to score_details table
        if result.score_secondary:
            self.save_score_details(result.run_id, result.score_secondary)

    def save_run_timing(self, result: RunResult) -> None:
        """Save timing data to the run_timings table."""
        self.db["run_timings"].insert(
            {
                "run_id": result.run_id,
                "suite_id": result.suite_id,
                "task_id": result.task_id,
                "model_alias": result.model_alias,
                "ttft_ms": result.ttft_ms,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
                "total_wall_ms": result.total_wall_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_total": result.tokens_per_second_total,
                "token_source": result.token_source,
                "chunk_count": 0,
                "created_at": result.created_at,
                "context_tokens": result.context_tokens,
                "estimated_prompt_tokens": result.estimated_prompt_tokens,
            },
        )

    def save_runs(self, results: list[RunResult]) -> None:
        """Bulk insert run results."""
        rows = []
        for r in results:
            judge_dimensions_json = None
            if r.judge_dimensions is not None:
                judge_dimensions_json = json.dumps(r.judge_dimensions)
            rows.append(
                {
                    "run_id": r.run_id,
                    "suite_id": r.suite_id,
                    "task_id": r.task_id,
                    "model_alias": r.model_alias,
                    "model_backend": r.model_backend,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "ttft_ms": r.ttft_ms,
                    "prefill_ms": r.prefill_ms,
                    "decode_ms": r.decode_ms,
                    "total_wall_ms": r.total_wall_ms,
                    "exit_status": r.exit_status,
                    "raw_response": r.raw_response,
                    "score_primary": None,
                    "scorer_version": None,
                    "score_secondary": None,
                    "score_explanation": None,
                    "error_message": r.error_message,
                    "created_at": r.created_at,
                    "tokens_per_second": r.tokens_per_second,
                    "tokens_per_second_total": r.tokens_per_second_total,
                    "token_source": r.token_source,
                    "chunk_count": 0,
                    # M7 judge fields
                    "judge_score": r.judge_score,
                    "judge_explanation": r.judge_explanation,
                    "judge_dimensions": judge_dimensions_json,
                    "judge_model": r.judge_model,
                    "human_override": 1 if r.human_override else 0,
                    "human_score": r.human_score,
                    "human_note": r.human_note,
                    # M8 prompt style
                    "prompt_style": r.prompt_style,
                    # M9 context size
                    "context_tokens": r.context_tokens,
                    "estimated_prompt_tokens": r.estimated_prompt_tokens,
                    # M10 quantization
                    "quantization": r.quantization,
                    # M11 command safety
                    "safety_score": r.safety_score,
                    "safety_details": (
                        json.dumps(r.safety_details)
                        if r.safety_details is not None
                        else None
                    ),
                    # M14 identity stamp
                    "requested_alias": r.requested_alias,
                    "litellm_model_name": r.litellm_model_name,
                    "openai_models_id": r.openai_models_id,
                    "vllm_served_model_name": r.vllm_served_model_name,
                    "vllm_container_name": r.vllm_container_name,
                    "hf_model_id": r.hf_model_id,
                    "backend_url": r.backend_url,
                    "server_start_time": r.server_start_time,
                    "speculative_decoding_enabled": (
                        1 if r.speculative_decoding_enabled else 0
                    ),
                }
            )
        self.db["runs"].insert_all(rows)

    def save_environment(self, snapshot: dict[str, Any]) -> None:
        """Save an environment snapshot."""
        self.db["environments"].insert(
            {
                "id": snapshot.get("id", "default"),
                "host": snapshot.get("host", ""),
                "os": snapshot.get("os", ""),
                "gpu_name": snapshot.get("gpu_name", ""),
                "cuda_version": snapshot.get("cuda_version", ""),
                "harness_commit": snapshot.get("harness_commit", ""),
                "created_at": snapshot.get("created_at", ""),
                # M3 env fields
                "vllm_version": snapshot.get("vllm_version"),
                "litellm_version": snapshot.get("litellm_version"),
                "model_path": snapshot.get("model_path"),
                "quantization": snapshot.get("quantization"),
                "max_model_len": snapshot.get("max_model_len"),
                "gpu_memory_utilization": snapshot.get("gpu_memory_utilization"),
                "served_port": snapshot.get("served_port"),
            },
        )

    def get_runs(
        self,
        suite_id: str | None = None,
        model_alias: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query run records with optional filters."""
        query = "SELECT * FROM runs"
        conditions = []
        params: list[Any] = []

        if suite_id:
            conditions.append("suite_id = ?")
            params.append(suite_id)
        if model_alias:
            conditions.append("model_alias = ?")
            params.append(model_alias)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        rows = list(self.db.query(query, params))

        # Auto-parse JSON text fields back to dicts
        json_fields = {"score_secondary", "judge_dimensions"}
        for row in rows:
            for field in json_fields:
                if field in row and isinstance(row[field], str):
                    try:
                        row[field] = json.loads(row[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

        return rows

    def get_run_summary(self, suite_id: str | None = None) -> list[dict[str, Any]]:
        """Get a summary grouped by model_alias."""
        query = """
            SELECT
                model_alias,
                COUNT(*) as tasks_run,
                SUM(CASE WHEN exit_status = 'success' THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN exit_status = 'error' THEN 1 ELSE 0 END) as failed,
                AVG(ttft_ms) as avg_ttft_ms,
                AVG(total_wall_ms) as avg_wall_ms
            FROM runs
        """
        conditions = []
        params: list[Any] = []

        if suite_id:
            conditions.append("suite_id = ?")
            params.append(suite_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY model_alias ORDER BY model_alias"

        return list(self.db.query(query, params))

    def save_run_timing(
        self,
        run_id: str | RunResult,
        timing_data: dict[str, Any] | None = None,
    ) -> None:
        """Save timing data to the run_timings table.

        Can be called as save_run_timing(result) with a RunResult,
        or save_run_timing(run_id, timing_data) with explicit fields.

        Args:
            run_id: RunResult object or run ID string.
            timing_data: Dict with timing fields (used if run_id is a string).
        """
        if isinstance(run_id, RunResult):
            result = run_id
            self.db["run_timings"].insert(
                {
                    "run_id": result.run_id,
                    "suite_id": result.suite_id,
                    "task_id": result.task_id,
                    "model_alias": result.model_alias,
                    "ttft_ms": result.ttft_ms,
                    "prefill_ms": result.prefill_ms,
                    "decode_ms": result.decode_ms,
                    "total_wall_ms": result.total_wall_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "tokens_per_second": result.tokens_per_second,
                    "tokens_per_second_total": result.tokens_per_second_total,
                    "token_source": result.token_source,
                    "chunk_count": 0,
                    "created_at": result.created_at,
                    "context_tokens": result.context_tokens,
                    "estimated_prompt_tokens": result.estimated_prompt_tokens,
                    # M14 identity stamp
                    "requested_alias": result.requested_alias,
                    "litellm_model_name": result.litellm_model_name,
                    "openai_models_id": result.openai_models_id,
                    "vllm_served_model_name": result.vllm_served_model_name,
                    "vllm_container_name": result.vllm_container_name,
                    "hf_model_id": result.hf_model_id,
                    "backend_url": result.backend_url,
                    "server_start_time": result.server_start_time,
                    "speculative_decoding_enabled": (
                        1 if result.speculative_decoding_enabled else 0
                    ),
                },
            )
        else:
            td = timing_data or {}
            self.db["run_timings"].insert(
                {
                    "run_id": run_id,
                    "suite_id": td.get("suite_id", ""),
                    "task_id": td.get("task_id", ""),
                    "model_alias": td.get("model_alias", ""),
                    "ttft_ms": td.get("ttft_ms", 0),
                    "prefill_ms": td.get("prefill_ms", 0),
                    "decode_ms": td.get("decode_ms", 0),
                    "total_wall_ms": td.get("total_wall_ms", 0),
                    "prompt_tokens": td.get("prompt_tokens", 0),
                    "completion_tokens": td.get("completion_tokens", 0),
                    "total_tokens": td.get("total_tokens", 0),
                    "tokens_per_second": td.get("tokens_per_second", 0),
                    "tokens_per_second_total": td.get(
                        "tokens_per_second_total", 0
                    ),
                    "token_source": td.get("token_source", "api"),
                    "chunk_count": td.get("chunk_count", 0),
                    "created_at": td.get("created_at", ""),
                },
            )

    def save_score_details(
        self,
        run_id: str,
        score_secondary: dict[str, Any],
    ) -> None:
        """Save individual scorer details to the score_details table.

        Args:
            run_id: The run ID to associate with.
            score_secondary: Dict of {scorer_name: ScoreResult} or {scorer_name: score_dict}.
        """
        for scorer_name, score_data in score_secondary.items():
            if isinstance(score_data, dict):
                self.db["score_details"].insert(
                    {
                        "run_id": run_id,
                        "scorer_name": scorer_name,
                        "score": score_data.get("score", 0),
                        "passed": score_data.get("passed", False),
                        "details_json": json.dumps(
                            score_data.get("details", {}), default=str
                        ),
                        "explanation": score_data.get("explanation"),
                    },
                )

    def get_timing_summary(
        self,
        model_alias: str | None = None,
        suite_id: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated timing metrics per model.

        Returns:
            Dict with per-model aggregates: mean, min, max, p95 for TTFT,
            decode, wall time, and tokens/sec.
        """
        query = """
            SELECT
                model_alias,
                COUNT(*) as run_count,
                AVG(ttft_ms) as mean_ttft_ms,
                MIN(ttft_ms) as min_ttft_ms,
                MAX(ttft_ms) as max_ttft_ms,
                AVG(decode_ms) as mean_decode_ms,
                AVG(total_wall_ms) as mean_wall_ms,
                AVG(tokens_per_second) as mean_tps,
                AVG(tokens_per_second_total) as mean_tps_total,
                AVG(prompt_tokens) as mean_prompt_tokens,
                AVG(completion_tokens) as mean_completion_tokens
            FROM run_timings
        """
        conditions = []
        params: list[Any] = []

        if suite_id:
            conditions.append("suite_id = ?")
            params.append(suite_id)
        if model_alias:
            conditions.append("model_alias = ?")
            params.append(model_alias)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY model_alias ORDER BY model_alias"

        rows = list(self.db.query(query, params))

        # Compute P95 for each model by fetching raw values
        for row in rows:
            model = row["model_alias"]
            # P95 TTFT
            p95_query = """
                SELECT ttft_ms FROM run_timings
                WHERE model_alias = ?
                ORDER BY ttft_ms ASC
            """
            p95_rows = list(self.db.query(p95_query, [model]))
            if p95_rows:
                n = len(p95_rows)
                p95_idx = int(n * 0.95) - 1
                p95_idx = max(0, min(p95_idx, n - 1))
                row["p95_ttft_ms"] = p95_rows[p95_idx]["ttft_ms"]
            else:
                row["p95_ttft_ms"] = 0.0

        return rows

    def save_judge_evaluation(
        self,
        run_id: str,
        task_id: str,
        model_alias: str,
        judge_model: str,
        rubric_name: str,
        score: str,
        dimensions: dict[str, Any] | None = None,
        explanation: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        """Save an LLM judge evaluation to the judge_evaluations table.

        Args:
            run_id: The run ID this evaluation corresponds to.
            task_id: Task identifier.
            model_alias: Model being judged.
            judge_model: The judge model that produced the score.
            rubric_name: Name of the rubric used.
            score: JSON string of the judge score.
            dimensions: Optional dimension scores dict.
            explanation: Optional judge explanation text.
            raw_response: Raw judge model response.
        """
        dimensions_json = None
        if dimensions is not None:
            dimensions_json = json.dumps(dimensions)

        self.db["judge_evaluations"].insert(
            {
                "run_id": run_id,
                "task_id": task_id,
                "model_alias": model_alias,
                "judge_model": judge_model,
                "rubric_name": rubric_name,
                "score": score,
                "dimensions_json": dimensions_json,
                "explanation": explanation,
                "raw_response": raw_response,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def save_pairwise_comparison(
        self,
        task_id: str,
        model_a: str,
        model_b: str,
        winner: str,
        margin: str,
        confidence: float,
        reason: str | None = None,
        dimension_comparison: dict[str, Any] | None = None,
        raw_judge_response: str | None = None,
        judge_model: str | None = None,
        human_override: bool = False,
        human_winner: str | None = None,
        human_note: str | None = None,
    ) -> None:
        """Save a pairwise comparison to the pairwise_comparisons table.

        Args:
            task_id: Task identifier.
            model_a: First model alias.
            model_b: Second model alias.
            winner: "A", "B", or "tie".
            margin: Margin of victory as string.
            confidence: Confidence score 0.0-1.0.
            reason: Judge's reasoning.
            dimension_comparison: Per-dimension comparison scores.
            raw_judge_response: Raw judge response text.
            judge_model: Judge model that produced the comparison.
            human_override: Whether a human overrode the result.
            human_winner: Human-decided winner if overridden.
            human_note: Human reviewer note.
        """
        dim_comparison_json = None
        if dimension_comparison is not None:
            dim_comparison_json = json.dumps(dimension_comparison)

        self.db["pairwise_comparisons"].insert(
            {
                "task_id": task_id,
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "margin": margin,
                "confidence": confidence,
                "reason": reason,
                "dimension_comparison_json": dim_comparison_json,
                "raw_judge_response": raw_judge_response,
                "judge_model": judge_model,
                "human_override": 1 if human_override else 0,
                "human_winner": human_winner,
                "human_note": human_note,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def get_judge_evaluations(self, suite_id: str) -> list[dict[str, Any]]:
        """Get all judge evaluations for a suite.

        Args:
            suite_id: Suite identifier.

        Returns:
            List of judge evaluation records.
        """
        query = """
            SELECT je.*, r.task_id as run_task_id, r.model_alias as run_model_alias
            FROM judge_evaluations je
            JOIN runs r ON je.run_id = r.run_id
            WHERE r.suite_id = ?
            ORDER BY je.created_at DESC
        """
        return list(self.db.query(query, [suite_id]))

    def get_pairwise_comparisons(self, suite_id: str) -> list[dict[str, Any]]:
        """Get all pairwise comparisons for a suite.

        Args:
            suite_id: Suite identifier.

        Returns:
            List of pairwise comparison records.
        """
        query = """
            SELECT pc.*
            FROM pairwise_comparisons pc
            JOIN runs r ON pc.task_id = r.task_id
            WHERE r.suite_id = ?
            ORDER BY pc.created_at DESC
        """
        return list(self.db.query(query, [suite_id]))

    def get_runs_by_task_family(
        self,
        suite_id: str,
        family: str,
    ) -> list[dict[str, Any]]:
        """Get runs filtered by task family.

        Extracts the family from the task_id prefix (e.g., 'local.docker_compose.*'
        matches family 'docker_compose').

        Args:
            suite_id: Suite identifier.
            family: Task family name (e.g., 'docker_compose', 'litellm_routing').

        Returns:
            List of run records matching the suite and family.
        """
        query = """
            SELECT * FROM runs
            WHERE suite_id = ?
            AND (
                task_id LIKE ?
                OR task_id LIKE ?
            )
            ORDER BY created_at DESC
        """
        params: list[Any] = [
            suite_id,
            f"{family}%",
            f"%.{family}.%",
        ]
        return list(self.db.query(query, params))
BenchmarkDB = SQLiteStore
