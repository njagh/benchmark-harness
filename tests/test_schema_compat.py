"""Tests for schemas/compat.py — schema versioning, validation,
migration, and exceptions."""

from __future__ import annotations

import pytest

from bench_harness.schemas.compat import (
    KNOWN_VERSIONS,
    SchemaVersionError,
    resolve_schema_version,
    migrate_result_v0_to_v1,
)


# ── KNOWN_VERSIONS ───────────────────────────────────────────────────


class TestKnownVersions:
    def test_contains_run_spec_v1(self):
        """KNOWN_VERSIONS contains the run_spec v1 version."""
        assert "llm_bench.run_spec.v1" in KNOWN_VERSIONS

    def test_contains_run_result_v1(self):
        """KNOWN_VERSIONS contains the run_result v1 version."""
        assert "llm_bench.run_result.v1" in KNOWN_VERSIONS

    def test_contains_model_artifact_v1(self):
        """KNOWN_VERSIONS contains the model_artifact v1 version."""
        assert "llm_bench.model_artifact.v1" in KNOWN_VERSIONS

    def test_is_a_set(self):
        """KNOWN_VERSIONS is a set."""
        assert isinstance(KNOWN_VERSIONS, set)

    def test_has_three_members(self):
        """KNOWN_VERSIONS has exactly 3 members."""
        assert len(KNOWN_VERSIONS) == 3


# ── SchemaVersionError ───────────────────────────────────────────────


class TestSchemaVersionError:
    def test_is_exception(self):
        """SchemaVersionError is an Exception subclass."""
        assert issubclass(SchemaVersionError, Exception)

    def test_can_be_raised_and_caught(self):
        """SchemaVersionError can be raised and caught."""
        with pytest.raises(SchemaVersionError):
            raise SchemaVersionError("test error message")

    def test_error_message_preserved(self):
        """Error message is preserved."""
        try:
            raise SchemaVersionError("version not supported")
        except SchemaVersionError as exc:
            assert "version not supported" in str(exc)


# ── validate_schema_version / resolve_schema_version ─────────────────


class TestResolveSchemaVersion:
    def test_known_run_spec_version(self):
        """Known run_spec version passes."""
        data = {"schema_version": "llm_bench.run_spec.v1"}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v1"

    def test_known_run_result_version(self):
        """Known run_result version passes."""
        data = {"schema_version": "llm_bench.run_result.v1"}
        assert resolve_schema_version(data) == "llm_bench.run_result.v1"

    def test_known_model_artifact_version(self):
        """Known model_artifact version passes."""
        data = {"schema_version": "llm_bench.model_artifact.v1"}
        assert resolve_schema_version(data) == "llm_bench.model_artifact.v1"

    def test_unknown_version_raises(self):
        """Unknown schema version raises SchemaVersionError."""
        data = {"schema_version": "unknown.schema.v1"}
        with pytest.raises(SchemaVersionError, match="Unknown schema base"):
            resolve_schema_version(data)

    def test_missing_version_returns_default(self):
        """Missing schema_version returns default."""
        data = {"name": "test"}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v1"

    def test_empty_dict_returns_default(self):
        """Empty dict returns default version."""
        assert resolve_schema_version({}) == "llm_bench.run_spec.v1"

    def test_no_schema_version_key_returns_default(self):
        """Dict without schema_version key returns default."""
        data = {"project": "my_project", "tags": ["test"]}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v1"

    def test_known_base_different_number(self):
        """Known base with different version number passes."""
        data = {"schema_version": "llm_bench.run_spec.v2"}
        assert resolve_schema_version(data) == "llm_bench.run_spec.v2"

    def test_unknown_base_raises(self):
        """Unknown base raises SchemaVersionError."""
        data = {"schema_version": "foo.bar.v1"}
        with pytest.raises(SchemaVersionError):
            resolve_schema_version(data)

    def test_version_with_dot_notation(self):
        """Version string with dots in base name is rejected if unknown."""
        data = {"schema_version": "custom.run_spec.v1"}
        with pytest.raises(SchemaVersionError):
            resolve_schema_version(data)


# ── migrate_result_v0_to_v1 ─────────────────────────────────────────


class TestMigrateResultV0ToV1:
    def test_adds_schema_version(self):
        """Migration adds schema_version if missing."""
        old_data = {"id": "run-1", "metrics": []}
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["schema_version"] == "llm_bench.run_result.v1"

    def test_preserves_existing_schema_version(self):
        """Migration preserves existing schema_version."""
        old_data = {
            "schema_version": "llm_bench.run_result.v1",
            "id": "run-1",
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["schema_version"] == "llm_bench.run_result.v1"

    def test_maps_id_to_run_id(self):
        """Migration maps 'id' to 'run_id'."""
        old_data = {"id": "run-1", "metrics": []}
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["run_id"] == "run-1"

    def test_preserves_existing_run_id(self):
        """Migration preserves existing run_id."""
        old_data = {
            "run_id": "existing-id",
            "metrics": [],
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["run_id"] == "existing-id"

    def test_maps_metrics_to_per_request(self):
        """Migration maps 'metrics' to 'per_request'."""
        old_data = {"id": "run-1", "metrics": [{"request_id": "r1"}]}
        migrated = migrate_result_v0_to_v1(old_data)
        assert len(migrated["per_request"]) == 1
        assert migrated["per_request"][0]["request_id"] == "r1"

    def test_preserves_existing_per_request(self):
        """Migration preserves existing per_request."""
        old_data = {
            "per_request": [{"request_id": "existing"}],
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert len(migrated["per_request"]) == 1
        assert migrated["per_request"][0]["request_id"] == "existing"

    def test_sets_default_project(self):
        """Migration sets default project when missing."""
        old_data = {"id": "run-1", "metrics": []}
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["project"] == "legacy"

    def test_preserves_existing_project(self):
        """Migration preserves existing project."""
        old_data = {
            "project": "my_project",
            "id": "run-1",
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["project"] == "my_project"

    def test_preserves_other_fields(self):
        """Migration preserves fields not explicitly migrated."""
        old_data = {
            "id": "run-1",
            "metrics": [],
            "custom_field": "custom_value",
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["custom_field"] == "custom_value"

    def test_empty_metrics_becomes_empty_per_request(self):
        """Empty metrics list becomes empty per_request."""
        old_data = {"id": "run-1", "metrics": []}
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["per_request"] == []

    def test_full_migration(self):
        """Full migration from v0 to v1 format."""
        old_data = {
            "id": "run-1",
            "project": "legacy",
            "metrics": [
                {
                    "request_id": "req-1",
                    "prompt_id": "prompt-1",
                    "ttft_ms": 50.0,
                }
            ],
        }
        migrated = migrate_result_v0_to_v1(old_data)
        assert migrated["schema_version"] == "llm_bench.run_result.v1"
        assert migrated["run_id"] == "run-1"
        assert migrated["project"] == "legacy"
        assert len(migrated["per_request"]) == 1
