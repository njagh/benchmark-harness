"""Schema versioning and migration compatibility layer."""

from __future__ import annotations

import re
from typing import Any


KNOWN_VERSIONS = {
    "llm_bench.run_spec.v1",
    "llm_bench.run_result.v1",
    "llm_bench.model_artifact.v1",
}


class SchemaVersionError(Exception):
    """Raised when a schema version is incompatible."""
    pass


def resolve_schema_version(data: dict) -> str:
    """Resolve schema version from data dict. Returns the version string."""
    version = data.get("schema_version")
    if not version:
        return "llm_bench.run_spec.v1"
    version_base = re.sub(r'\.v\d+$', '', version)
    known_bases = {re.sub(r'\.v\d+$', '', v) for v in KNOWN_VERSIONS}
    if version_base not in known_bases:
        raise SchemaVersionError(f"Unknown schema base: {version_base}")
    return version


def migrate_result_v0_to_v1(data: dict) -> dict:
    """Migrate pre-M18 result dict to v1 format."""
    migrated = dict(data)
    if "schema_version" not in migrated:
        migrated["schema_version"] = "llm_bench.run_result.v1"
    if "run_id" not in migrated:
        migrated["run_id"] = migrated.get("id", "unknown")
    if "project" not in migrated:
        migrated["project"] = migrated.get("project", "legacy")
    if "per_request" not in migrated:
        migrated["per_request"] = migrated.get("metrics", [])
    return migrated
