"""Tests for M14 identity stamp — /v1/models introspection before each task."""

import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bench_harness.cli import _parse_models_response
from bench_harness.runners.completion_runner import RunResult


# ── CLI _parse_models_response tests ────────────────────────────────────


class TestParseModelsResponse:
    """Tests for the CLI helper that parses /v1/models responses."""

    def test_empty_data_returns_defaults(self):
        """Empty data list → all identity fields are None."""
        result = _parse_models_response({"data": []}, "http://localhost:4000/v1")
        assert result["openai_models_id"] is None
        assert result["vllm_served_model_name"] is None
        assert result["backend_url"] == "http://localhost:4000/v1"

    def test_error_returns_defaults(self):
        """Error in response → all identity fields are None."""
        result = _parse_models_response(
            {"error": "model not found"}, "http://localhost:4000/v1"
        )
        assert result["openai_models_id"] is None

    def test_vllm_response_with_served_model(self):
        """vLLM-style response with served_model_name."""
        resp = {
            "data": [
                {
                    "id": "qwen-35b",
                    "object": "model",
                    "owned_by": "vllm",
                    "served_model_name": "Qwen/Qwen3-35B-A3B",
                    "container_name": "vllm-qwen3",
                    "hf_model_id": "Qwen/Qwen3-35B-A3B",
                    "server_start_time": 1700000000,
                    "speculative_decoder_id": None,
                }
            ]
        }
        result = _parse_models_response(resp, "http://spark.local:4000/v1")
        assert result["openai_models_id"] == "qwen-35b"
        assert result["vllm_served_model_name"] == "Qwen/Qwen3-35B-A3B"
        assert result["vllm_container_name"] == "vllm-qwen3"
        assert result["hf_model_id"] == "Qwen/Qwen3-35B-A3B"
        assert result["server_start_time"] == 1700000000
        assert result["speculative_decoding_enabled"] is False
        assert result["backend_url"] == "http://spark.local:4000/v1"

    def test_speculative_decoding_enabled(self):
        """speculative_decoder_id present → enabled is True."""
        resp = {
            "data": [
                {
                    "id": "qwen-35b",
                    "served_model_name": "Qwen/Qwen3-35B-A3B",
                    "speculative_decoder_id": "eagle-token",
                }
            ]
        }
        result = _parse_models_response(resp, "http://localhost:4000/v1")
        assert result["speculative_decoding_enabled"] is True

    def test_multiple_model_entries_first_one(self):
        """Only the first model entry is used."""
        resp = {
            "data": [
                {"id": "first-model", "served_model_name": "first-served"},
                {"id": "second-model", "served_model_name": "second-served"},
            ]
        }
        result = _parse_models_response(resp, "http://localhost:4000/v1")
        assert result["openai_models_id"] == "first-model"
        assert result["vllm_served_model_name"] == "first-served"

    def test_model_with_no_id_skipped(self):
        """Model entries without 'id' are skipped."""
        resp = {
            "data": [
                {"object": "model"},  # no id
                {"id": "valid-model", "served_model_name": "valid-served"},
            ]
        }
        result = _parse_models_response(resp, "http://localhost:4000/v1")
        assert result["openai_models_id"] == "valid-model"


# ── RunResult identity stamp fields ─────────────────────────────────────


class TestRunResultIdentityStamp:
    """Verify RunResult dataclass has identity stamp fields."""

    def test_result_has_identity_fields(self):
        """RunResult can be constructed with identity fields."""
        result = RunResult(
            run_id="test-run-1",
            suite_id="test_suite",
            task_id="test.task_001",
            model_alias="test-model",
            requested_alias="test-alias",
            litellm_model_name="Qwen/Qwen3-35B",
            openai_models_id="qwen-35b",
            vllm_served_model_name="Qwen/Qwen3-35B-A3B",
            vllm_container_name="vllm-test",
            hf_model_id="Qwen/Qwen3-35B-A3B",
            backend_url="http://test:4000/v1",
            server_start_time="1700000000",
            speculative_decoding_enabled=False,
        )
        assert result.requested_alias == "test-alias"
        assert result.litellm_model_name == "Qwen/Qwen3-35B"
        assert result.openai_models_id == "qwen-35b"
        assert result.vllm_served_model_name == "Qwen/Qwen3-35B-A3B"
        assert result.vllm_container_name == "vllm-test"
        assert result.hf_model_id == "Qwen/Qwen3-35B-A3B"
        assert result.backend_url == "http://test:4000/v1"
        assert result.server_start_time == "1700000000"
        assert result.speculative_decoding_enabled is False

    def test_result_identity_fields_default_none(self):
        """Identity fields default to None when not provided."""
        result = RunResult(
            run_id="test-run-2",
            suite_id="test_suite",
            task_id="test.task_002",
            model_alias="test-model",
        )
        assert result.requested_alias is None
        assert result.openai_models_id is None
        assert result.vllm_served_model_name is None


# ── SQLite persistence of identity stamp ────────────────────────────────


class TestSQLiteIdentityPersistence:
    """Verify identity fields are persisted to and read from SQLite."""

    def _make_result(self, **overrides):
        """Create a RunResult with identity stamp fields."""
        default = {
            "run_id": "test-uuid-1",
            "suite_id": "test_suite",
            "task_id": "test.task_001",
            "model_alias": "test-model",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "requested_alias": "test-alias",
            "litellm_model_name": "Qwen/Qwen3-35B",
            "openai_models_id": "qwen-35b",
            "vllm_served_model_name": "Qwen/Qwen3-35B-A3B",
            "vllm_container_name": "vllm-test",
            "hf_model_id": "Qwen/Qwen3-35B-A3B",
            "backend_url": "http://test:4000/v1",
            "server_start_time": "1700000000",
            "speculative_decoding_enabled": True,
        }
        default.update(overrides)
        return RunResult(**default)

    def test_save_run_persists_identity(self, tmp_path):
        """save_run() writes identity fields to SQLite."""
        from bench_harness.storage.sqlite import SQLiteStore

        db_path = str(tmp_path / "test_identity.db")
        store = SQLiteStore(db_path)
        store.init()

        result = self._make_result(run_id="uuid-1")
        store.save_run(result)

        rows = store.get_runs(suite_id="test_suite")
        assert len(rows) == 1
        row = rows[0]
        assert row["requested_alias"] == "test-alias"
        assert row["litellm_model_name"] == "Qwen/Qwen3-35B"
        assert row["openai_models_id"] == "qwen-35b"
        assert row["vllm_served_model_name"] == "Qwen/Qwen3-35B-A3B"
        assert row["vllm_container_name"] == "vllm-test"
        assert row["hf_model_id"] == "Qwen/Qwen3-35B-A3B"
        assert row["backend_url"] == "http://test:4000/v1"
        assert row["server_start_time"] == "1700000000"
        assert row["speculative_decoding_enabled"] == 1

    def test_save_run_timing_persists_identity(self, tmp_path):
        """save_run_timing() writes identity fields to run_timings."""
        from bench_harness.storage.sqlite import SQLiteStore

        db_path = str(tmp_path / "test_timing.db")
        store = SQLiteStore(db_path)
        store.init()

        result = self._make_result(run_id="uuid-2")
        store.save_run_timing(result)

        rows = list(store.db["run_timings"].rows)
        assert len(rows) == 1
        row = next(iter(rows))
        assert row["requested_alias"] == "test-alias"
        assert row["openai_models_id"] == "qwen-35b"
        assert row["speculative_decoding_enabled"] == 1

    def test_speculative_decoding_false_stored_as_zero(self, tmp_path):
        """speculative_decoding_enabled=False → stored as integer 0."""
        from bench_harness.storage.sqlite import SQLiteStore

        db_path = str(tmp_path / "test_spec.db")
        store = SQLiteStore(db_path)
        store.init()

        result = self._make_result(
            run_id="uuid-3", speculative_decoding_enabled=False
        )
        store.save_run(result)

        rows = store.get_runs(suite_id="test_suite")
        assert len(rows) == 1
        assert rows[0]["speculative_decoding_enabled"] == 0


class TestIdentityStampMismatchDetection:
    """Test that identity stamp reveals when model alias doesn't match served model."""

    def test_alias_mismatches_openai_models_id(self):
        """When alias='qwen-dense' but openai_models_id='agent-code', detect mismatch."""
        result = RunResult(
            run_id="test-mismatch",
            suite_id="test",
            task_id="test.task",
            model_alias="qwen-dense",
            requested_alias="qwen-dense",
            litellm_model_name="qwen-dense",
            openai_models_id="Qwen/Qwen3-35B-A3B",  # This is actually agent-code's model
            vllm_served_model_name="agent-code",
            backend_url="http://spark:4000/v1",
        )
        # The identity stamp shows the real model is different from the alias
        assert result.requested_alias == "qwen-dense"
        assert result.openai_models_id == "Qwen/Qwen3-35B-A3B"
        assert result.vllm_served_model_name == "agent-code"
        # These fields together reveal the mismatch
        assert result.vllm_served_model_name != result.requested_alias

    def test_alias_matches_openai_models_id(self):
        """When everything is consistent, no mismatch."""
        result = RunResult(
            run_id="test-match",
            suite_id="test",
            task_id="test.task",
            model_alias="agent-code",
            requested_alias="agent-code",
            litellm_model_name="agent-code",
            openai_models_id="agent-code",
            vllm_served_model_name="agent-code",
            backend_url="http://spark:4000/v1",
        )
        assert result.requested_alias == result.openai_models_id
        assert result.vllm_served_model_name == result.requested_alias
