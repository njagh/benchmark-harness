# Milestone 17 — Package and Storage Abstraction ✅ DONE

## Goal

Convert the benchmark harness into a reusable, importable Python package with a storage layer independent of the repository source tree. This milestone lays the foundation for M18–M23 by establishing `StorageConfig`, resolving storage roots from multiple sources, and separating artifact storage from results storage.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestone 1: CLI runner, SQLite storage, markdown report (existing code to refactor)
- Milestone 3: Timing and token metrics (will be accessed via new storage layer)
- BUILD_LIBRARY.md sections 1.1, 2.1, 2.2, 8.1

## Acceptance Criteria

- [x] `pip install -e .` works. The package installs as `llm_bench` (in addition to existing `bench_harness`).
- [x] `python -m llm_bench --help` prints CLI help.
- [x] `python -c "from bench_harness import StorageConfig; print(StorageConfig.from_env())"` works.
- [x] Storage root resolves via priority: CLI flag > env var > per-project config > default.
- [x] No benchmark code writes to repo-relative paths except tests/examples.
- [x] Default storage root is `~/.local/share/llm-bench`, not a subdirectory of the git repo.
- [x] Unsafe storage roots (inside git repo, `/tmp`, virtualenv, Docker overlay) are rejected by default with a clear warning.
- [x] `--allow-unsafe-storage-root` override exists.
- [x] Artifact store and results store are separate namespaces under the storage root.
- [x] A `dry-run` mode exists for storage initialization.

## Subtasks

### 17.1 Create `llm_bench` namespace package alongside `bench_harness` ✅ DONE

- [x] `src/llm_bench/__init__.py` — re-exports public API from `bench_harness`
- [x] `src/llm_bench/cli.py` — thin wrapper that imports `bench_harness.cli`
- [x] `src/llm_bench/__main__.py` — module entry point
- [x] `pyproject.toml` — add `llm_bench` package find + `llm-bench` entry point

### 17.2 Implement `StorageConfig` ✅ DONE

### 17.3 Implement storage root resolution and safety checks ✅ DONE

### 17.4 Create `init-storage` CLI command ✅ DONE

### 17.5 Create `storage-info` CLI command ✅ DONE

### 17.6 Migrate existing result-writing code to use `StorageConfig` ✅ DONE

- [x] `sqlite.py` — accepts optional `StorageConfig` with legacy path fallback
- [x] `artifacts.py` — accepts optional `StorageConfig` with legacy path fallback
- [x] `cli.py run` command — creates run directory, writes resolved spec

### 17.7 Create immutable per-run result directories ✅ DONE

- `pip install -e .` works. The package installs as `llm_bench` (in addition to existing `bench_harness`).
- `python -m llm_bench --help` prints CLI help.
- `python -c "from llm_bench import StorageConfig; print(StorageConfig.from_env())"` works.
- Storage root resolves via priority: CLI flag > env var > per-project config > default.
- No benchmark code writes to repo-relative paths except tests/examples.
- Default storage root is `~/.local/share/llm-bench`, not a subdirectory of the git repo.
- Unsafe storage roots (inside git repo, `/tmp`, virtualenv, Docker overlay) are rejected by default with a clear warning.
- `--allow-unsafe-storage-root` override exists.
- Artifact store and results store are separate namespaces under the storage root.
- A `dry-run` mode exists for storage initialization.

## Subtasks

### 17.1 Create `llm_bench` namespace package alongside `bench_harness`

- Create `src/llm_bench/` as an alias namespace that re-exports from `bench_harness`.
- Keep `bench_harness` as the internal package for backward compatibility with existing scripts.
- Add `llm_bench` entry point to `pyproject.toml`: `llm-bench = "llm_bench.cli:main"`.
- Update `pyproject.toml` to include both `bench-harness` and `llm-bench` scripts.

**Files:**
- `src/llm_bench/__init__.py` — re-exports public API from `bench_harness`
- `src/llm_bench/cli.py` — symlink or thin wrapper that imports `bench_harness.cli`
- `pyproject.toml` — add `llm_bench` package find + `llm-bench` entry point

### 17.2 Implement `StorageConfig`

Create a `StorageConfig` class that encapsulates storage root resolution and namespace management.

**Resolution priority:**
1. `--storage-root` CLI flag (passed at runtime)
2. `LLM_BENCH_STORAGE_ROOT` environment variable
3. Per-project `.llm-bench.yaml` config file
4. Default: `~/.local/share/llm-bench` (XDG_DATA_HOME / fallback)

**Public API:**
```python
from llm_bench import StorageConfig

config = StorageConfig.from_env()              # from env var or default
config = StorageConfig(root="/mnt/datasets-big/llm-bench")  # explicit
config = StorageConfig.from_project()          # from .llm-bench.yaml in cwd or parents
```

**Fields:**
- `root: Path` — the resolved storage root
- `artifacts_root: Path` — `<root>/artifacts/`
- `results_root: Path` — `<root>/results/`
- `registry_root: Path` — `<root>/registry/`
- `logs_root: Path` — `<root>/logs/`
- `cache_root: Path` — `<root>/cache/`
- `tmp_root: Path` — `<root>/tmp/`

**Artifact sub-namespaces:**
- `artifacts/models/`
- `artifacts/engines/`
- `artifacts/tokenizers/`
- `artifacts/calibration/`
- `artifacts/runtime-builds/`

**Results sub-namespaces:**
- `results/runs/`
- `results/summaries/`
- `results/comparisons/`

**Files:**
- `src/bench_harness/storage/config.py` — `StorageConfig` class
- `src/bench_harness/storage/config.py` — unsafe-path detection logic
- `src/bench_harness/__init__.py` — export `StorageConfig`

### 17.3 Implement storage root resolution and safety checks

**Unsafe path detection — reject by default:**
- Inside the harness git repo (`git rev-parse --show-toplevel`)
- `/tmp`, `/var/tmp`
- Inside any virtualenv (check `sys.prefix` or `VIRTUAL_ENV` env var)
- Inside a Docker overlay path (check `/proc/1/cgroup` or `/.dockerenv`)
- Path on filesystem with less than N GB free space (configurable, default 10 GB)

**Override:**
- Pass `--allow-unsafe-storage-root` or `StorageConfig(..., allow_unsafe=True)`

**Files:**
- `src/bench_harness/storage/safety.py` — `is_unsafe_path()`, `check_storage_root()`, `detect_ephemeral_path()`

### 17.4 Create `init-storage` CLI command

```bash
llm-bench init-storage --root /mnt/datasets-big/llm-bench --dry-run
```

- Creates all namespace directories under the storage root.
- Writes a `.llm-bench.yaml` in the current working directory for project-level config.
- `--dry-run` shows what would be created without creating anything.
- Prints the resolved storage locations before any creation.

**Files:**
- `src/bench_harness/cli.py` — add `init_storage` command

### 17.5 Create `storage-info` CLI command

```bash
llm-bench storage-info
```

- Prints the resolved storage root, all namespaces, and whether each is a valid path.
- Shows artifact policy and result policy for the current project.
- Shows warnings for unsafe roots or ephemeral artifact paths.

**Files:**
- `src/bench_harness/cli.py` — add `storage_info` command

### 17.6 Migrate existing result-writing code to use `StorageConfig`

Walk through all files that write results or artifacts and replace hard-coded paths:

- `src/bench_harness/storage/sqlite.py` — accept `StorageConfig` instead of `Path`; store DB at `results/runs/<date>/` instead of `runs/benchmark.db`
- `src/bench_harness/storage/artifacts.py` — accept `StorageConfig`; write JSONL to `results/runs/<date>/` namespace
- `src/bench_harness/cli.py` — the `run` command: create result directory under `results_root`, pass `StorageConfig` to `SQLiteStore`, `CompletionRunner`, artifact savers
- `src/bench_harness/reports/` — update report generators to write to `results_root/summaries/`
- `src/bench_harness/export/` — update exporters to accept `StorageConfig`
- `src/bench_harness/compare.py` — accept `StorageConfig` or explicit path

**Key migration rule:**
- Every function that previously accepted `Path | str` for output should now accept `StorageConfig` with an optional relative path suffix.
- Historical data at old locations (e.g., `./runs/benchmark.db`) must still be readable for migration purposes.

### 17.7 Create immutable per-run result directories

Each run produces a unique result directory:

```
<storage-root>/results/runs/<YYYY-MM-DD>/<run-name>__<ISO-timestamp>__<short-hash>/
  run_spec.yaml
  resolved_spec.yaml
  artifact_manifest.json
  environment.json
  hardware.json
  metrics.jsonl
  summary.json
  stdout.log
  stderr.log
  server.log
  notes.md
```

**Files:**
- `src/bench_harness/storage/config.py` — `StorageConfig.create_run_dir(run_name) -> Path`
- `src/bench_harness/storage/config.py` — deterministic hash suffix from spec content

## Files Created

- `src/bench_harness/storage/config.py` — `StorageConfig`, namespace paths
- `src/bench_harness/storage/safety.py` — unsafe path detection, ephemeral artifact detection
- `src/llm_bench/__init__.py` — public API re-exports
- `src/llm_bench/cli.py` — CLI wrapper
- `src/bench_harness/cli.py` — updated with `init-storage`, `storage-info`, migration to `StorageConfig`
- `src/bench_harness/__init__.py` — updated exports

## Files Modified

- `pyproject.toml` — add `llm_bench` package + entry point
- `src/bench_harness/storage/sqlite.py` — accept `StorageConfig`
- `src/bench_harness/storage/artifacts.py` — accept `StorageConfig`
- `src/bench_harness/reports/*.py` — accept `StorageConfig`
- `src/bench_harness/export/*.py` — accept `StorageConfig`
- `src/bench_harness/compare.py` — accept `StorageConfig`
- `src/bench_harness/cli.py` — migrate to `StorageConfig`, add new CLI commands
- `src/bench_harness/__init__.py` — export `StorageConfig`

## Tests

- [x] `tests/test_m17_storage.py` — 26 tests covering:
  - StorageConfig resolution from all sources (env, CLI, project config, default)
  - Unsafe path detection (git repo, /tmp, virtualenv, Docker overlay, low disk)
  - Ephemeral artifact path detection
  - Immutable per-run directory creation
  - `resolved_spec.yaml` writing
  - Namespace directory creation and existence checks
  - CLI `init-storage` and `storage-info` commands
  - Docker overlay detection uses source-code introspection (runtime path unavailable in sandboxed env)

## Notes

- This milestone is purely infrastructure. No benchmark behavior changes; it replaces where data is written.
- Historical runs at `./runs/benchmark.db` must still be readable — `SQLiteStore.get_runs()` should accept an explicit path override.
- M18 builds on this by defining the `RunSpec` schema that will flow through `StorageConfig.create_run_dir()` and result directories.
