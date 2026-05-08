"""ModelArtifact schema — metadata about a model artifact for the registry."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ArtifactKind(str, Enum):
    hf_checkpoint = "hf_checkpoint"
    trtllm_engine = "trtllm_engine"
    gguf = "gguf"
    vllm_endpoint = "vllm_endpoint"
    openai_endpoint = "openai_endpoint"


class ArtifactMode(str, Enum):
    external_path = "external_path"
    managed_copy = "managed_copy"
    managed_symlink = "managed_symlink"


class ModelArtifact(BaseModel):
    schema_version: str = "llm_bench.model_artifact.v1"
    artifact_id: str
    kind: ArtifactKind
    mode: ArtifactMode = ArtifactMode.external_path
    source_path: str
    model_id: str | None = None
    quantization: str | None = None
    dtype: str | None = None
    parameter_class: str | None = None
    tokenizer_path: str | None = None
    file_list_summary: dict[str, int] = Field(default_factory=dict)
    total_size_bytes: int = 0
    config_file_hash: str | None = None
    weight_manifest_hash: str | None = None
    created_at: str | None = None
    producing_git_commit: str | None = None
    producing_version: str | None = None
    backend_version: str | None = None
    registered_at: str | None = None
    durable: bool = True
    artifact_warnings: list[str] = Field(default_factory=list)

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: ArtifactMode) -> ArtifactMode:
        allowed = {
            ArtifactMode.external_path,
            ArtifactMode.managed_copy,
            ArtifactMode.managed_symlink,
        }
        if v not in allowed:
            raise ValueError(
                f"Invalid artifact mode: {v}. Must be one of {allowed}"
            )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelArtifact":
        path = Path(path)
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        # Convert string enum values back to enum members
        if isinstance(data, dict):
            if "kind" in data and isinstance(data["kind"], str):
                try:
                    data["kind"] = ArtifactKind(data["kind"])
                except ValueError:
                    pass
            if "mode" in data and isinstance(data["mode"], str):
                try:
                    data["mode"] = ArtifactMode(data["mode"])
                except ValueError:
                    pass
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode='python'), indent=2
        )
