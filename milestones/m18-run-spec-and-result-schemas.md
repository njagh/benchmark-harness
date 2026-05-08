# Milestone 18 — RunSpec and Result Schemas ✅ DONE

## Goal

Define structured, versioned schemas for benchmark run specifications and run results. These schemas replace the ad-hoc dict/YAML approach and become the single source of truth for what a benchmark run is, what artifact it targets, and what it measured. The schemas support YAML and JSON input, validation with clear error messages, and schema versioning for future compatibility.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestone 17: `StorageConfig`, storage root resolution, immutable result directories

## Acceptance Criteria

- [x] `RunSpec` Pydantic model validates all required fields and gives clear error messages.
- [x] `RunResult` Pydantic model captures all per-request and summary metrics.
- [x] Run specs can be loaded from YAML or JSON files.
- [x] CLI can execute `llm-bench bench-run spec.yaml` and load the same spec via Python API.
- [x] Every spec and result includes a `schema_version` field (e.g., `llm_bench.run_spec.v1`).
- [x] Old run results can be read with a compatibility shim for schema changes.
- [x] Breaking schema changes produce explicit migration errors.
- [x] The `run` command (flag-based) builds a `RunSpec` internally via `build_run_spec_from_flags()`.
- [x] `WorkloadSpec` supports optional `task_dir` for explicit task directory override.

- `RunSpec` Pydantic model validates all required fields and gives clear error messages.
- `RunResult` Pydantic model captures all per-request and summary metrics.
- Run specs can be loaded from YAML or JSON files.
- CLI can execute `llm-bench run spec.yaml` and load the same spec via Python API.
- Every spec and result includes a `schema_version` field (e.g., `llm_bench.run_spec.v1`).
- Old run results can be read with a compatibility shim for schema changes.
- Breaking schema changes produce explicit migration errors.

### 18.1 Define `RunSpec` schema ✅ DONE

- [x] `ArtifactSpec` with `kind`, `mode`, `path`, `tokenizer_path`, `model_id`, `quantization`
- [x] `RuntimeSpec` with `kind`, `launch`, `host`, `port`, `model_name`, `args`
- [x] `WorkloadSpec` with `prompt_suite`, `max_tokens`, `temperature`, `num_runs`, `concurrency`, `task_dir`
- [x] `HardwareSpec` with `profile`, `expected_gpu`
- [x] `StoragePolicy` with `artifact_policy`, `result_policy`
- [x] `RunSpec` top-level with `schema_version`, `name`, `project`, `tags`, `hardware`, `artifact`, `runtime`, `workload`, `storage`
- [x] `name` validator: slug-formatted (lowercase, hyphens, alphanumeric)
- [x] Endpoint artifacts require URL paths; file artifacts accept filesystem paths
- [x] `ArtifactKind` enum: `hf_checkpoint`, `trtllm_engine`, `gguf`, `vllm_endpoint`, `openai_endpoint`
- [x] `RuntimeKind` enum: `openai_compatible`, `vllm`, `trtllm`, `llamacpp`, `external`
- [x] `LaunchMode` enum: `managed_process`, `existing`
- [x] `build_run_spec_from_flags()` — backward compat from CLI flags

### 18.2 Define `RunResult` schema ✅ DONE

**File:** `src/bench_harness/schemas/run_spec.py`

Pydantic model representing a complete benchmark run specification:

```python
RunSpec (top level)
  schema_version: str  # "llm_bench.run_spec.v1"
  name: str  # "qwen14b-int4-awq-3070ti-smoke"
  project: str  # "modelopt_3070ti"
  tags: list[str]  # ["modelopt", "int4", "awq", "3070ti"]

  hardware: HardwareSpec
    profile: str  # "3070ti-8gb"
    expected_gpu: str  # "RTX 3070 Ti"

  artifact: ArtifactSpec
    kind: str  # "hf_checkpoint" | "trtllm_engine" | "gguf" | "openai_endpoint" | "vllm_endpoint"
    mode: str  # "external_path" | "managed_copy" | "managed_symlink"
    path: str  # "/path/to/model" or "http://127.0.0.1:8009/v1"
    tokenizer_path: str | None
    model_id: str | None
    quantization: str | None

  runtime: RuntimeSpec
    kind: str  # "openai_compatible" | "vllm" | "trtllm" | "llamacpp" | "external"
    launch: str  # "managed_process" | "existing"
    host: str | None
    port: int | None
    model_name: str | None
    args: dict[str, Any]  # runtime-specific kwargs

  workload: WorkloadSpec
    prompt_suite: str  # suite name or list of task IDs
    max_tokens: int
    temperature: float
    num_runs: int
    concurrency: int

  storage: StoragePolicy
    artifact_policy: str  # "external_path" | "managed_copy" | "managed_symlink"
    result_policy: str  # "managed"
```

**Validation rules:**
- `name` is required and must be slug-formatted (lowercase, hyphens, alphanumeric)
- `project` is required
- `artifact.kind` must be one of the allowed kinds
- `artifact.mode` must be one of the allowed modes
- If `runtime.launch == "managed_process"`, then `runtime.kind` determines which launcher to use
- If `artifact.kind` is an endpoint type, `artifact.path` must be a URL
- If `artifact.kind` is a file type, `artifact.path` must be a filesystem path
- `workload.prompt_suite` resolves to a suite name in `configs/suites.yaml` or a list of task IDs

**Files created:**
- `src/bench_harness/schemas/__init__.py`
- `src/bench_harness/schemas/run_spec.py` — `RunSpec`, `ArtifactSpec`, `RuntimeSpec`, `WorkloadSpec`, `HardwareSpec`, `StoragePolicy`

### 18.2 Define `RunResult` schema

**File:** `src/bench_harness/schemas/run_result.py`

Per-request result:
```python
RequestResult
  request_id: str
  prompt_id: str
  prompt_tokens: int
  generated_tokens: int
  ttft_ms: float
  decode_ms: float
  total_wall_ms: float
  tokens_per_second_decode: float
  tokens_per_second_wall: float
  finish_reason: str
  error: str | None
  peak_gpu_memory_mb: float | None
  quality_score: float | None
  quality_explanation: str | None
```

Summary result (aggregated across all requests):
```python
RunResult
  schema_version: str  # "llm_bench.run_result.v1"
  run_id: str
  run_spec_ref: str  # path to the run_spec.yaml
  project: str
  artifact_fingerprint: dict  # hash + metadata of what was benchmarked

  per_request: list[RequestResult]
  summary: ResultSummary

ResultSummary
  mean_ttft_ms: float
  median_ttft_ms: float
  p95_ttft_ms: float
  mean_decode_tps: float
  median_decode_tps: float
  p95_decode_tps: float
  mean_wall_tps: float
  median_wall_tps: float
  p95_wall_tps: float
  success_rate: float
  oom_count: int
  timeout_count: int
  peak_vram_mb: float
  average_vram_mb: float
  qualitative_score: float | None
  quality_stddev: float
```

**Files created:**
- `src/bench_harness/schemas/run_result.py` — `RunResult`, `RequestResult`, `ResultSummary`

### 18.3 Define `ModelArtifact` schema

**File:** `src/bench_harness/schemas/model_artifact.py`

Captures metadata about a model artifact for the registry:
```python
ModelArtifact
  schema_version: str
  artifact_id: str
  kind: str  # "hf_checkpoint" | "trtllm_engine" | "gguf" | "vllm_endpoint" | "openai_endpoint"
  mode: str  # "external_path" | "managed_copy" | "managed_symlink"
  source_path: str  # filesystem path or URL
  model_id: str | None
  quantization: str | None
  dtype: str | None
  parameter_class: str | None
  tokenizer_path: str | None
  file_list_summary: dict[str, int]  # {filename: size_bytes}
  total_size_bytes: int
  config_file_hash: str | None
  weight_manifest_hash: str | None
  created_at: str | None
  producing_git_commit: str | None
  producing_version: str | None  # ModelOpt version, TensorRT-LLM version, etc.
  backend_version: str | None  # vLLM version, llama.cpp version, etc.
  durable: bool  # True if path is not ephemeral
```

**Files created:**
- `src/bench_harness/schemas/model_artifact.py` — `ModelArtifact`

### 18.4 Add schema versioning and migration

Each schema version string format: `llm_bench.<schema_type>.v<version>`

- `llm_bench.run_spec.v1`
- `llm_bench.run_result.v1`
- `llm_bench.model_artifact.v1`

**Migration shim:**
- `schemas/compat.py` — `resolve_schema_version(data: dict) -> type`
- When reading old results, detect if `schema_version` is missing (pre-M18 runs) and map to v1 with defaults for missing fields.
- If a new version is encountered that is incompatible, raise `SchemaVersionError` with a message indicating which fields changed.

**Files created:**
- `src/bench_harness/schemas/compat.py` — `resolve_schema_version()`, `migrate_result_v0_to_v1()`

### 18.5 Load `RunSpec` from YAML/JSON

- `RunSpec.from_yaml(path: str | Path) -> RunSpec`
- `RunSpec.from_json(path: str | Path) -> RunSpec`
- `RunSpec.model_dump_yaml() -> str` — serializes back to YAML for `resolved_spec.yaml`

**Validation errors:**
- Use Pydantic `ValidationError` with field-level detail
- CLI displays errors in a readable format using `rich`
- Errors identify the exact field and why it failed validation

**Files modified:**
- `src/bench_harness/schemas/run_spec.py` — add `.from_yaml()`, `.from_json()` class methods

### 18.6 Wire `RunSpec` into the CLI and runner

- The `llm-bench run spec.yaml` command loads the spec via `RunSpec.from_yaml()`
- If no spec file is provided, fall back to CLI flags (for backward compatibility): `--suite`, `--models`, `--runs`, `--context-tokens`
- Build a `RunSpec` from CLI flags by mapping:
  - `--suite` → `workload.prompt_suite`
  - `--models` → `artifact` block (multiple artifacts supported)
  - `--runs` → `workload.num_runs`
  - `--context-tokens` → resolved to `hardware` budget
- The runner receives `RunSpec` and extracts what it needs

**Files modified:**
- `src/bench_harness/cli.py` — `run` command: parse spec file or build from flags
- `src/bench_harness/runners/completion_runner.py` — accept `RunSpec` as primary input

### 18.7 Write `resolved_spec.yaml` to result directories

When a run starts, write the (possibly resolved) spec to the result directory:
- `run_spec.yaml` — the raw user-provided spec
- `resolved_spec.yaml` — after defaults, suite expansion, and CLI flag merging

**Files modified:**
- `src/bench_harness/storage/config.py` — `StorageConfig.write_resolved_spec(spec: RunSpec, run_dir: Path)`

## Files Created

- `src/bench_harness/schemas/__init__.py`
- `src/bench_harness/schemas/run_spec.py`
- `src/bench_harness/schemas/run_result.py`
- `src/bench_harness/schemas/model_artifact.py`
- `src/bench_harness/schemas/compat.py`

## Files Modified

- `src/bench_harness/cli.py` — `run` command: spec file loading, fallback to CLI flags
- `src/bench_harness/runners/completion_runner.py` — accept `RunSpec`
- `src/bench_harness/storage/sqlite.py` — store RunSpec JSON in `runs` table
- `src/bench_harness/storage/artifacts.py` — write `resolved_spec.yaml`
- `src/bench_harness/__init__.py` — export `RunSpec`, `RunResult`, `ModelArtifact`

### 18.3 Define `ModelArtifact` schema ✅ DONE

### 18.4 Add schema versioning and migration ✅ DONE

- [x] `resolve_schema_version(data: dict) -> str` — extracts and validates version string
- [x] `migrate_result_v0_to_v1(data: dict) -> dict` — migrates pre-M18 result dicts to v1
- [x] `SchemaVersionError` for unknown schema bases
- [x] `KNOWN_VERSIONS` set: `llm_bench.run_spec.v1`, `llm_bench.run_result.v1`, `llm_bench.model_artifact.v1`

### 18.5 Load `RunSpec` from YAML/JSON ✅ DONE

- [x] `RunSpec.from_yaml(path)` — loads YAML, converts string enum values to members
- [x] `RunSpec.from_json(path)` — loads JSON
- [x] `RunSpec.to_yaml()` — serializes with enum values
- [x] `RunSpec.to_json()` — serializes to JSON

### 18.6 Wire `RunSpec` into the CLI and runner ✅ DONE

- [x] `cli.py run` command: `spec` argument accepts YAML/JSON path
- [x] Falls back to `build_run_spec_from_flags()` when no spec file
- [x] `run_spec.py` exported from `bench_harness.schemas` and `bench_harness`

### 18.7 Write `resolved_spec.yaml` to result directories ✅ DONE

- [x] `StorageConfig.write_resolved_spec(spec, run_dir)` writes YAML

### 18.8 Add `WorkloadSpec.task_dir` field ✅ DONE

- [x] Optional `task_dir: str | None = None` allows explicit task directory override
- [x] `OpenAICompatibleRunner.run_workload()` uses `task_dir` when present

## Tests

- [x] `tests/test_m18.py` — 29 tests covering:
  - RunSpec validation (valid, invalid names, defaults, endpoint URL requirements)
  - YAML/JSON roundtrip serialization
  - `build_run_spec_from_flags()` with all parameters
  - `WorkloadSpec.task_dir` — default None, explicit value, YAML roundtrip
  - `RequestResult` basic construction
  - `ResultSummary.from_requests()` — empty, single, errors, quality scores
  - `RunResult.finalize()` — summary computation
  - `RunResult.write_to_directory()` — files written
  - `ModelArtifact` schema validation and JSON serialization
  - Schema compatibility: known versions, resolve, migration, field preservation

## Notes

- This milestone is backward-compatible. The CLI still accepts `--suite` and `--models` flags; those build a `RunSpec` internally.
- The `RunResult` Pydantic model is written to both `summary.json` and `metrics.jsonl` in the result directory.
- The `ModelArtifact` schema feeds into the artifact registry (M21).
