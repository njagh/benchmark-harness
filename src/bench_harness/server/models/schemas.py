"""Request and response schemas for the server API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Config models ──────────────────────────────────────────────────


class ArtifactConfig(BaseModel):
    kind: str = "openai_endpoint"
    mode: str = "external_path"
    path: str = ""
    tokenizer_path: Optional[str] = None
    model_id: Optional[str] = None
    quantization: Optional[str] = None


class RuntimeConfig(BaseModel):
    kind: str = "openai_compatible"
    launch: str = "existing"
    host: Optional[str] = None
    port: Optional[int] = None
    model_name: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)


class WorkloadConfig(BaseModel):
    prompt_suite: str = "smoke"
    max_tokens: int = 256
    temperature: float = 0.0
    num_runs: int = 1
    concurrency: int = 1
    task_dir: Optional[str] = None


class StoragePolicyConfig(BaseModel):
    artifact_policy: str = "external_path"
    result_policy: str = "managed"


class SavedConfig(BaseModel):
    id: str = ""
    name: str = ""
    project: str = "default"
    tags: list[str] = Field(default_factory=list)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    storage: StoragePolicyConfig = Field(default_factory=StoragePolicyConfig)
    advanced: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    run_count: int = 0
    last_run_at: Optional[str] = None


class ConfigTemplate(BaseModel):
    name: str
    description: str
    preset: dict[str, Any]


# ── Run models ─────────────────────────────────────────────────────


class RunStatus(BaseModel):
    run_id: str = ""
    config_id: Optional[str] = None
    config_name: Optional[str] = None
    status: str = "pending"
    model_alias: str = ""
    suite_id: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results_path: Optional[str] = None
    error_message: Optional[str] = None
    current_task: Optional[str] = None
    current_style: Optional[str] = None


class RunListItem(BaseModel):
    run_id: str = ""
    config_id: Optional[str] = None
    config_name: Optional[str] = None
    status: str = ""
    model_alias: str = ""
    suite_id: str = ""
    tasks_run: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    avg_score: Optional[float] = None
    avg_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results_path: Optional[str] = None


# ── Results models ─────────────────────────────────────────────────


class RunSummaryItem(BaseModel):
    model_alias: str = ""
    tasks_run: int = 0
    passed: int = 0
    failed: int = 0
    avg_ttft_ms: Optional[float] = None
    avg_wall_ms: Optional[float] = None
    avg_score: Optional[float] = None


class TaskResult(BaseModel):
    run_id: str = ""
    task_id: str = ""
    model_alias: str = ""
    exit_status: str = ""
    score_primary: Optional[float] = None
    score_secondary: Optional[dict] = None
    score_explanation: Optional[str] = None
    ttft_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    total_wall_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    prompt_style: Optional[str] = None
    context_tokens: Optional[str] = None
    quantization: Optional[str] = None
    safety_score: Optional[float] = None
    created_at: Optional[str] = None
    judge_score: Optional[float] = None


class TimingSummaryItem(BaseModel):
    model_alias: str = ""
    run_count: int = 0
    mean_ttft_ms: Optional[float] = None
    min_ttft_ms: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    p95_ttft_ms: Optional[float] = None
    mean_decode_ms: Optional[float] = None
    mean_wall_ms: Optional[float] = None
    mean_tps: Optional[float] = None
    mean_prompt_tokens: Optional[float] = None
    mean_completion_tokens: Optional[float] = None


# ── Comparison models ──────────────────────────────────────────────


class DeltaEntry(BaseModel):
    task_id: str = ""
    model_alias: str = ""
    old_score: Optional[float] = None
    new_score: Optional[float] = None
    delta: float = 0.0
    status: str = ""  # "REGRESSION" or "IMPROVEMENT"
    risk: str = ""    # "high", "medium", "low"


class PerformanceDeltaEntry(BaseModel):
    task_id: str = ""
    model_alias: str = ""
    baseline_value: Optional[float] = None
    candidate_value: Optional[float] = None
    change_pct: float = 0.0
    metric: str = ""  # "wall_time" or "tokens_per_second"


class CrashChangeEntry(BaseModel):
    task_id: str = ""
    model_alias: str = ""
    status: str = ""


class CompareResponse(BaseModel):
    quality_regressions: list[DeltaEntry] = Field(default_factory=list)
    quality_improvements: list[DeltaEntry] = Field(default_factory=list)
    performance_regressions: list[PerformanceDeltaEntry] = Field(default_factory=list)
    performance_improvements: list[PerformanceDeltaEntry] = Field(default_factory=list)
    crash_changes: list[CrashChangeEntry] = Field(default_factory=list)
    baseline_summary: Optional[dict] = None
    candidate_summary: Optional[dict] = None


# ── Metadata models ────────────────────────────────────────────────


class ModelInfo(BaseModel):
    alias: str = ""
    provider: str = ""
    base_url: str = ""
    model: str = ""
    backend: str = ""
    quantization: Optional[str] = None
    notes: str = ""


class SuiteInfo(BaseModel):
    name: str = ""
    description: str = ""
    task_dir: str = ""
    families: list[str] = Field(default_factory=list)
    max_concurrency: int = 0
    default_runs: int = 1
    default_temperature: float = 0.0


class ScorerInfo(BaseModel):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class RubricInfo(BaseModel):
    name: str = ""
    description: str = ""
    dimensions: dict[str, Any] = Field(default_factory=dict)


class TaskFamilyInfo(BaseModel):
    family: str = ""
    count: int = 0
    tasks: list[str] = Field(default_factory=list)


# ── API response helpers ───────────────────────────────────────────


class APIResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Any = None

    @classmethod
    def ok(cls, data: Any = None, message: str = "") -> "APIResponse":
        return cls(success=True, message=message, data=data)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "APIResponse":
        return cls(success=False, message=message, data=data)
