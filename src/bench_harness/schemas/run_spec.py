"""RunSpec schema — structured benchmark run specification."""

from __future__ import annotations

import json
import re
import yaml
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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


class RuntimeKind(str, Enum):
    openai_compatible = "openai_compatible"
    vllm = "vllm"
    trtllm = "trtllm"
    llamacpp = "llamacpp"
    external = "external"


class LaunchMode(str, Enum):
    managed_process = "managed_process"
    existing = "existing"

_ENUM_MAP: dict[str, dict[str, Any]] | None = None


def _get_enum_map() -> dict[str, dict[str, Any]]:
    global _ENUM_MAP
    if _ENUM_MAP is None:
        _ENUM_MAP = {
            "kind": {m.value: m for m in ArtifactKind},
            "mode": {m.value: m for m in ArtifactMode},
        }
    return _ENUM_MAP


def _convert_enums(data: Any) -> Any:
    """Recursively convert string enum values back to enum members."""
    if isinstance(data, dict):
        converted = {}
        for k, v in data.items():
            if k == "kind" and isinstance(v, str):
                enum_map = _get_enum_map()
                if v in enum_map.get("kind", {}):
                    converted[k] = enum_map["kind"][v]
                else:
                    converted[k] = v
            elif k == "mode" and isinstance(v, str):
                enum_map = _get_enum_map()
                if v in enum_map.get("mode", {}):
                    converted[k] = enum_map["mode"][v]
                else:
                    converted[k] = v
            else:
                converted[k] = _convert_enums(v)
        return converted
    if isinstance(data, list):
        return [_convert_enums(item) for item in data]
    return data



class StoragePolicy(BaseModel):
    artifact_policy: ArtifactMode = ArtifactMode.external_path
    result_policy: str = "managed"


class HardwareSpec(BaseModel):
    profile: str = "default"
    expected_gpu: str | None = None


class ArtifactSpec(BaseModel):
    kind: ArtifactKind
    mode: ArtifactMode = ArtifactMode.external_path
    path: str
    tokenizer_path: str | None = None
    model_id: str | None = None
    quantization: str | None = None

    @model_validator(mode='after')
    def validate_path(self) -> "ArtifactSpec":
        if self.kind in (ArtifactKind.openai_endpoint, ArtifactKind.vllm_endpoint):
            if not self.path.startswith(('http://', 'https://')):
                raise ValueError(
                    f"Endpoint artifact kind requires URL path, got: {self.path}"
                )
        return self


class RuntimeSpec(BaseModel):
    kind: RuntimeKind
    launch: LaunchMode = LaunchMode.existing
    host: str | None = None
    port: int | None = None
    model_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class WorkloadSpec(BaseModel):
    prompt_suite: str
    max_tokens: int = 256
    temperature: float = 0.0
    num_runs: int = 1
    concurrency: int = 1
    task_dir: str | None = None


class RunSpec(BaseModel):
    schema_version: str = "llm_bench.run_spec.v1"
    name: str
    project: str
    tags: list[str] = Field(default_factory=list)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    artifact: ArtifactSpec
    runtime: RuntimeSpec = Field(
        default_factory=lambda: RuntimeSpec(kind=RuntimeKind.openai_compatible)
    )
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    storage: StoragePolicy = Field(default_factory=StoragePolicy)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', v):
            raise ValueError(
                'name must be slug-formatted (lowercase, hyphens, alphanumeric)'
            )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunSpec":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        data = _convert_enums(data)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunSpec":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self) -> str:
        import enum
        data = self.model_dump(mode='python', by_alias=False, exclude_none=False)

        def _serialize(data):
            if isinstance(data, enum.Enum):
                return data.value
            if isinstance(data, dict):
                return {k: _serialize(v) for k, v in data.items()}
            if isinstance(data, list):
                return [_serialize(item) for item in data]
            return data

        return yaml.dump(_serialize(data), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode='python'), indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunSpec":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        # Convert string enum values back to enum members
        data = _convert_enums(data)
        return cls(**data)


def build_run_spec_from_flags(
    suite: str,
    models: list[str],
    num_runs: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.0,
    concurrency: int = 1,
    context_tokens: str = "small",
    project: str = "default",
) -> RunSpec:
    """Build a RunSpec from CLI flag arguments (backward compat)."""
    model_name = models[0] if models else "unknown"
    return RunSpec(
        name=f"{model_name}-{suite}-cli",
        project=project,
        artifact=ArtifactSpec(
            kind=ArtifactKind.hf_checkpoint,
            path=model_name,
            model_id=model_name,
        ),
        runtime=RuntimeSpec(
            kind=RuntimeKind.openai_compatible,
            launch=LaunchMode.existing,
        ),
        workload=WorkloadSpec(
            prompt_suite=suite,
            max_tokens=max_tokens,
            temperature=temperature,
            num_runs=num_runs,
            concurrency=concurrency,
        ),
        hardware=HardwareSpec(profile=context_tokens),
    )
