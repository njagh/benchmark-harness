"""SQLite storage for benchmark run results.

Supports schema migration for adding new timing and token metric columns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlite_utils import Database

from bench_harness.runners.completion_runner import RunResult

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

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(str(self.db_path), memory=False)

    def init(self) -> None:
        """Create tables and migrate schema if needed."""
        self._create_runs_table()
        self._create_environments_table()
        self._migrate_runs_schema()
        self._migrate_environments_schema()
        self._create_run_timings_table()
        self._create_score_details_table()
        self._create_indexes()
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
                "chunk_count": int,
                "created_at": str,
            },
            pk="run_id",
            if_not_exists=True,
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

    def _create_indexes(self) -> None:
        """Create indexes for common query patterns."""
        self.db["runs"].create_index(
            ["suite_id", "model_alias"], if_not_exists=True
        )
        self.db["run_timings"].create_index(
            ["suite_id", "model_alias"], if_not_exists=True
        )

    def save_run(self, result: RunResult) -> None:
        """Save a single run result with all timing and token fields."""
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
                "score_secondary": json.dumps(result.score_secondary) if result.score_secondary else None,
                "score_explanation": result.score_explanation,
                "error_message": result.error_message,
                "created_at": result.created_at,
                # M3 timing fields
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_total": result.tokens_per_second_total,
                "token_source": result.token_source,
                "chunk_count": 0,
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
            },
        )

    def save_runs(self, results: list[RunResult]) -> None:
        """Bulk insert run results."""
        rows = []
        for r in results:
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

        return list(self.db.query(query, params))

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
