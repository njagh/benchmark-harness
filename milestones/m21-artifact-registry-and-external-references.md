# Milestone 21 — Artifact Registry and External References

## Goal

Implement an explicit artifact registry that tracks model artifacts across projects and experiments. Support three artifact modes (`external_path`, `managed_copy`, `managed_symlink`) and capture sufficient metadata to distinguish between two runs against different temporary builds. Results must remain interpretable even after temporary artifacts are deleted.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestone 17: `StorageConfig`, artifact namespace (`<root>/artifacts/`, `<root>/registry/`)
- Milestone 18: `ModelArtifact` schema
- Milestone 20: `register-artifact` and `inspect-artifact` CLI commands (structural), fingerprint logic

## Acceptance Criteria

- Artifact registry lives at `<storage-root>/registry/artifacts.jsonl`.
- Three artifact modes work correctly: `external_path`, `managed_copy`, `managed_symlink`.
- Harness can benchmark a model from `/tmp`, `/mnt/datasets-big`, HF cache, or experiment workspace.
- Harness records whether artifact path is external, copied, or symlinked in run results.
- Harness warns if an external artifact path looks temporary or non-durable.
- Two runs against different temporary builds are distinguishable via fingerprint.
- A run result remains interpretable after the temporary artifact is deleted (because metadata is captured).
- Manifest hash is stable enough for comparison but does not require hashing hundreds of GB by default.

## Subtasks

### 21.1 Implement artifact registry

**File:** `src/bench_harness/registry.py`

JSONL-based registry at `<storage-root>/registry/artifacts.jsonl`:

```python
class ArtifactRegistry:
    """JSONL-based registry for model artifacts."""

    def __init__(self, config: StorageConfig):
        self.path = config.registry_root / "artifacts.jsonl"

    def register(self, artifact: ModelArtifact) -> None:
        """Append artifact record to registry."""
        ...

    def lookup(self, artifact_id: str) -> ModelArtifact | None:
        """Look up artifact by ID."""
        ...

    def list_all(self) -> list[ModelArtifact]:
        """List all registered artifacts."""
        ...

    def query(self, kind: str | None = None, project: str | None = None,
              quantization: str | None = None) -> list[ModelArtifact]:
        """Filter by kind, project, quantization method."""
        ...
```

Each JSONL line contains a `ModelArtifact` dict with an added `registered_at` timestamp.

### 21.2 Implement artifact fingerprinting

**File:** `src/bench_harness/utils/hashing.py`

Capture enough metadata to identify what was actually benchmarked:

**Minimum fingerprint fields:**
- `config_file_hash` — SHA-256 of `config.json` or equivalent
- `weight_manifest_hash` — SHA-256 of weight file list + sizes (fast, no actual hashing of weights)
- `total_size_bytes` — total artifact directory size
- `file_count` — number of files
- `detected_dtype` — detected from model weights or config
- `detected_model_id` — from config or filename patterns
- `backend_version` — vLLM, TRT-LLM, llama.cpp, or ModelOpt version if available

**Hashing strategy:**
- Do NOT hash all weight files by default (too slow for large models)
- Instead, hash weight file paths + sizes (manifest hash) — stable enough for comparison
- Optional: `--hash-weights` flag for full weight file hashing (for final releases)
- Config file is always hashed (small, deterministic)

**Fingerprint generation:**
```python
def compute_artifact_fingerprint(artifact: ModelArtifact, scan_path: Path,
                                  hash_weights: bool = False) -> dict:
    """Compute a fingerprint dict for an artifact.
    Returns dict with config_hash, manifest_hash, size, file_count, etc."""
    ...
```

### 21.3 Implement artifact copy and symlink modes

**File:** `src/bench_harness/registry.py`

```python
def manage_artifact(artifact: ModelArtifact, config: StorageConfig) -> Path:
    """Handle artifact based on mode.
    Returns the effective path to use for benchmarking."""
    if artifact.mode == "external_path":
        _warn_if_ephemeral(artifact.path)
        return Path(artifact.path)
    elif artifact.mode == "managed_copy":
        return _copy_artifact(artifact, config)
    elif artifact.mode == "managed_symlink":
        return _symlink_artifact(artifact, config)
```

**Copy mode:**
- Copies model config files and weight files to `<artifacts_root>/models/<artifact_id>/`
- Copies are incremental — only new/changed files are re-copied
- Manifest file records which files were copied

**Symlink mode:**
- Creates symlink at `<artifacts_root>/models/<artifact_id>` → source path
- Symlink is durable even if source is moved (as long as source path is recorded)

### 21.4 Wire artifact reference into `RunSpec` and run results

- `RunSpec.artifact` already has `mode` and `path` fields from M18
- When a run starts:
  1. Resolve artifact via `manage_artifact(spec.artifact, storage_config)`
  2. Compute fingerprint via `compute_artifact_fingerprint()`
  3. Write `artifact_manifest.json` to result directory
  4. Record `artifact_fingerprint` in `RunResult`
- The fingerprint becomes part of the run result's immutable identity

**File:** `src/bench_harness/storage/config.py` — add `resolve_artifact()` helper

### 21.5 Add ModelOpt-specific metadata hooks

**File:** `src/bench_harness/registry.py` — or a new `src/bench_harness/hooks.py`

When registering or benchmarking ModelOpt-produced artifacts, capture:
- ModelOpt version
- Quantization algorithm (AWQ, SqueezeLLM, etc.)
- Calibration dataset name/path
- Number of calibration samples
- Export format
- Base model ID
- Quantized output path
- Producing command
- Producing git commit

**Hook interface:**
```python
class ArtifactMetadataHook(ABC):
    """Hook to add project-specific metadata to artifact manifests."""
    def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact: ...
    def enrich_run_result(self, result: RunResult) -> RunResult: ...
```

**ModelOpt hook:**
```python
class ModelOptMetadataHook(ArtifactMetadataHook):
    """Capture ModelOpt-specific metadata from quantization runs."""
    def enrich_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        # Scan artifact for ModelOpt metadata files (e.g., modelopt_meta.json)
        # Fill in quantization algorithm, calibration dataset, ModelOpt version, etc.
        ...
```

**Default:** No hooks installed by default. ModelOpt project can install the hook via config:
```yaml
# .llm-bench.yaml
hooks:
  - module: modelopt_bench.hooks
    class: ModelOptMetadataHook
```

### 21.6 Ephemeral artifact detection

**File:** `src/bench_harness/storage/safety.py` — extend `detect_ephemeral_path()`

Warn if artifact path is under:
- `/tmp`, `/var/tmp`
- Known build cache paths
- Missing path (path doesn't exist)
- Docker container filesystem path (contains `/var/lib/docker/` or similar)
- Container layer paths

The warning includes a `durable: false` flag in the artifact manifest.

**Run result metadata:**
```json
{
  "artifact_durable": true,
  "artifact_warnings": []
}
```

or

```json
{
  "artifact_durable": false,
  "artifact_warnings": [
    "Artifact path /tmp/modelopt-runs/qwen-14b-int4 is under /tmp and may be deleted"
  ]
}
```

## Files Created

- `src/bench_harness/registry.py` — `ArtifactRegistry`, `manage_artifact()`
- `src/bench_harness/hooks.py` — `ArtifactMetadataHook` ABC
- `src/bench_harness/utils/hashing.py` — artifact fingerprinting (moved/enhanced from M20)

## Files Modified

- `src/bench_harness/schemas/model_artifact.py` — add `registered_at`, `durable`, `artifact_warnings`
- `src/bench_harness/schemas/run_result.py` — add `artifact_fingerprint`, `artifact_durable`, `artifact_warnings`
- `src/bench_harness/storage/config.py` — `resolve_artifact()` helper
- `src/bench_harness/storage/sqlite.py` — store artifact fingerprint in `runs` table
- `src/bench_harness/cli.py` — `register-artifact` uses `ArtifactRegistry`
- `src/bench_harness/__init__.py` — export `ArtifactRegistry`, `ArtifactMetadataHook`

## Tests

- `tests/test_artifact_registry.py` — register, lookup, list, query
- `tests/test_artifact_fingerprint.py` — hash computation, manifest hash stability
- `tests/test_artifact_modes.py` — external_path, managed_copy, managed_symlink
- `tests/test_ephemeral_detection.py` — warnings for /tmp, docker paths, missing paths
- `tests/test_modelopt_hooks.py` — ModelOpt metadata enrichment

## Notes

- The registry is JSONL (append-only) — no locks needed, concurrent writes are safe.
- `artifact_id` is generated as `<kind>-<model_id>-<quantization>-<short-hash>` for human readability.
- The fingerprint is computed once at benchmark time and embedded in the result. Even if the artifact is deleted afterward, the result directory has everything needed to understand what was tested.
- ModelOpt hooks are optional and loaded from `.llm-bench.yaml` if specified.
