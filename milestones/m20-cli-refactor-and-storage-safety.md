# Milestone 20 — CLI Refactor and Storage Safety

## Goal

Refactor the CLI around a clean command structure aligned with the BUILD_LIBRARY.md specification. Add `--dry-run` to all mutable commands, ensure the CLI never silently writes large files into the source repo, and enforce storage safety rules.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestone 17: `StorageConfig`, `init-storage`, `storage-info`
- Milestone 18: `RunSpec` schema
- Milestone 19: `RuntimeRunner` interface

## Acceptance Criteria

- CLI commands match BUILD_LIBRARY.md spec:
  - `llm-bench init-storage --root /path`
  - `llm-bench storage-info`
  - `llm-bench register-artifact artifact.yaml`
  - `llm-bench inspect-artifact /path/to/model`
  - `llm-bench run spec.yaml`
  - `llm-bench run spec.yaml --storage-root /path`
  - `llm-bench list-runs --project name`
  - `llm-bench summarize --project name`
  - `llm-bench compare RUN_ID_A RUN_ID_B`
  - `llm-bench export-summary --project name --format markdown`
- `--dry-run` exists for `run`, `register-artifact`, and `inspect-artifact`.
- CLI prints resolved storage locations before running benchmarks.
- CLI never silently writes large files into the source repo.
- Unsafe storage root is warned/refused with clear message.
- Ephemeral artifact paths are warned about.
- `--allow-unsafe-storage-root` override works.
- Backward-compatible: old `bench-harness run --suite ... --models ...` still works via `RunSpec` build from flags.

## Subtasks

### 20.1 Refactor CLI command structure

**File:** `src/bench_harness/cli.py`

Reorganize the CLI into a command tree:

```
llm-bench
  init-storage [--root PATH] [--dry-run] [--allow-unsafe-storage-root]
  storage-info
  register-artifact ARTIFACT_YAML [--dry-run] [--allow-unsafe-storage-root]
  inspect-artifact PATH [--dry-run]
  run SPEC_YAML [--storage-root PATH] [--dry-run] [--allow-unsafe-storage-root]
    (backward compat: run --suite SUITE --models MODEL ...)
  list-runs [--project NAME] [--storage-root PATH]
  summarize --project NAME [--storage-root PATH] [--format json|markdown]
  compare RUN_ID_A RUN_ID_B [--storage-root PATH]
  export-summary --project NAME --format markdown|json [--storage-root PATH]
  # Legacy commands (preserved for backward compat)
  list-tasks
  show-task TASK_ID
  judge list-rubrics
  judge show-rubric RUBRIC_NAME
  export sft|preference|regression|judge|all ...
  analyze notebook --db PATH
  compare-runs RUN_DB_A RUN_DB_B
  regression suite --db PATH
  run-lm-eval --suite SUITE --models MODEL ...
```

Key changes:
- All commands that write data accept `--storage-root` override
- All commands resolve `StorageConfig` before executing
- Storage info printed before any write operation

### 20.2 Add `--dry-run` support

For `run`, `register-artifact`, `inspect-artifact`:

- `--dry-run` resolves `StorageConfig` fully
- Validates `RunSpec` if provided
- Prints what would happen:
  - Resolved storage root
  - Target result directory path
  - Artifact source path and mode
  - Runtime to be used
  - Workload summary (suite, num_runs, concurrency)
  - Whether the server would be launched or connected to existing
  - Files that would be created/overwritten
- Does NOT execute any benchmark or write any results/logs

**Implementation:**
```python
def dry_run(spec: RunSpec, config: StorageConfig) -> None:
    """Print what a run would do without executing it."""
    run_dir = config.create_run_dir(spec.name)
    click.echo(f"[dry-run] Storage root: {config.root}")
    click.echo(f"[dry-run] Result directory: {run_dir}")
    click.echo(f"[dry-run] Artifact: {spec.artifact.kind} at {spec.artifact.path}")
    click.echo(f"[dry-run] Runtime: {spec.runtime.kind} (launch={spec.runtime.launch})")
    click.echo(f"[dry-run] Workload: {spec.workload.prompt_suite}, {spec.workload.num_runs} runs")
    click.echo(f"[dry-run] Would write results to: {run_dir}")
```

### 20.3 Add `register-artifact` command

```bash
llm-bench register-artifact artifact.yaml [--dry-run] [--storage-root PATH]
```

- Reads artifact spec from YAML (see BUILD_LIBRARY.md artifact spec example)
- Validates the artifact path exists or is a valid URL
- If not `--dry-run`:
  - If `managed_copy`: copies selected files into `artifacts/models/`
  - If `managed_symlink`: creates symlink in `artifacts/models/`
  - If `external_path`: records path in registry without copying
  - Captures artifact fingerprint (hashes, metadata)
  - Writes to `registry/artifacts.jsonl`
- `--dry-run` prints what would be registered and where

**Files created:**
- `src/bench_harness/registry.py` — `register_artifact()`, `artifact_fingerprint()`, `hash_artifact()`

### 20.4 Add `inspect-artifact` command

```bash
llm-bench inspect-artifact /path/to/model [--dry-run]
```

- Scans a model artifact path and prints metadata:
  - Kind detection (HF checkpoint vs GGUF vs TRT-LLM engine)
  - Files and sizes
  - Config files found and their hashes
  - Weight files found and their hashes
  - Total size
  - Ephemeral path warning if under `/tmp` or Docker overlay
  - Dtype, model ID if detectable from config
- `--dry-run` is implicit (inspect never writes)

**Files created:**
- `src/bench_harness/utils/hashing.py` — `compute_config_hash()`, `compute_weight_manifest_hash()`, `scan_artifact_path()`

### 20.5 Add `list-runs` command

```bash
llm-bench list-runs [--project NAME] [--storage-root PATH]
```

- Lists all run result directories under `<storage_root>/results/runs/`
- Shows: run ID, name, project, date, artifact kind, runtime, status
- `--project` filters by project name
- Output in Rich table format (consistent with existing CLI style)

### 20.6 Add `summarize` command

```bash
llm-bench summarize --project NAME [--storage-root PATH] [--format json|markdown]
```

- Loads all runs for the project from `results/runs/`
- Aggregates summary metrics across runs
- Shows: per-model average TTFT, decode TPS, success rate
- Quality delta between best and worst model
- `-f markdown` produces a report similar to existing markdown reports

### 20.7 Add `export-summary` command

```bash
llm-bench export-summary --project NAME --format markdown [--storage-root PATH]
```

- Exports the project summary to a file
- Default output: `<storage_root>/results/summaries/<project>_summary.md`
- JSON format produces machine-readable output

### 20.8 Add `compare` command

```bash
llm-bench compare RUN_ID_A RUN_ID_B [--storage-root PATH]
```

- Compares two runs by ID from the results store
- Loads both `summary.json` files and the `metrics.jsonl`
- Shows quality regressions/improvements and performance changes
- Uses existing `compare.py` logic but operates on the new result directory structure

### 20.9 Storage safety enforcement in CLI

Every command that writes data must:
1. Resolve `StorageConfig`
2. Call `check_storage_root(config.root)` before proceeding
3. If unsafe, print a Rich-formatted warning and ask for confirmation
4. Accept `--allow-unsafe-storage-root` to override
5. Check artifact paths for ephemerality and warn

**CLI safety decorator:**
```python
def unsafe_storage_guard(f):
    """Decorator that checks storage root safety before command execution."""
    @functools.wraps(f)
    def wrapper(*args, allow_unsafe=False, storage_root=None, **kwargs):
        if storage_root:
            config = StorageConfig(root=storage_root)
        else:
            config = StorageConfig.from_env()
        if not allow_unsafe:
            safety.check_storage_root(config.root)
        return f(*args, config=config, **kwargs)
    return wrapper
```

### 20.10 Preserve backward-compatible commands

All existing `bench-harness` commands must continue to work:
- `bench-harness run --suite smoke --models agent-code` → builds `RunSpec` from flags
- `bench-harness compare runs/baseline/benchmark.db runs/candidate/benchmark.db` → uses old SQLite path directly
- `bench-harness export sft --suite coding_benchmark` → works with old DB format
- `bench-harness analyze notebook --db runs/.../benchmark.db` → works with old DB
- `bench-harness run-lm-eval` → preserved

These are aliases on the same CLI app. The `bench-harness` and `llm-bench` scripts both invoke the same app.

## Files Created

- `src/bench_harness/registry.py` — artifact registration, fingerprint computation
- `src/bench_harness/utils/hashing.py` — artifact hashing utilities (moved from ad-hoc code)
- `src/bench_harness/utils/paths.py` — path resolution helpers (moved from ad-hoc code)

## Files Modified

- `src/bench_harness/cli.py` — complete refactor into new command tree
- `src/bench_harness/__init__.py` — new public exports
- `src/bench_harness/storage/sqlite.py` — accept `StorageConfig` in all methods
- `src/bench_harness/storage/artifacts.py` — accept `StorageConfig`

## Tests

- `tests/test_cli_dry_run.py` — dry-run mode for `run`, `register-artifact`, `inspect-artifact`
- `tests/test_cli_storage_safety.py` — unsafe root rejection, override behavior
- `tests/test_cli_backward_compat.py` — old `bench-harness run --suite` still works
- `tests/test_artifact_registration.py` — register-artifact with all three modes

## Notes

- This milestone is the biggest CLI change. The existing `bench_harness.cli` is ~1100 lines and must be reorganized while preserving all existing commands.
- The command tree follows the BUILD_LIBRARY.md spec as closely as possible. Commands not in the spec (export, analyze, run-lm-eval) are preserved as backward-compatible legacy commands.
- `--storage-root` on every command is the escape hatch for one-off runs against a specific storage location.
- `--dry-run` is the safety net — users can always preview what would happen.
