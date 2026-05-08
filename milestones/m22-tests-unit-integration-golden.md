# Milestone 22 — Tests (Unit, Integration, Golden)

## Goal

Add comprehensive tests that cover the new library infrastructure: storage resolution, schemas, artifact manifests, runner interface, and CLI commands. Tests must run without GPU access and must not download large models.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestones M17–M21: all new modules

## Acceptance Criteria

- All tests run without GPU.
- No tests download large models.
- CI can run on an ordinary CPU machine.
- Storage resolution is fully tested: all priority levels, unsafe paths, overrides.
- Schema validation is tested: valid specs, invalid specs, missing required fields.
- Artifact manifest creation is tested: all three modes (external, copy, symlink).
- Runner interface is tested with mocked endpoints.
- Golden output tests verify `summary.json`, markdown report, compare output, artifact manifests.
- Test coverage of new modules (M17–M21) is > 80%.

## Subtasks

### 22.1 Unit tests for storage layer

**File:** `tests/test_storage_config.py`

- `test_storage_root_from_env` — `LLM_BENCH_STORAGE_ROOT` env var
- `test_storage_root_from_flag` — `StorageConfig(root="/explicit/path")`
- `test_storage_root_from_project_config` — `.llm-bench.yaml` in cwd and parent dirs
- `test_storage_root_default` — falls back to `~/.local/share/llm-bench`
- `test_namespace_paths` — all sub-namespace paths are correct
- `test_create_run_dir` — creates immutable per-run directory, returns path
- `test_run_dir_hash_uniqueness` — same spec produces same hash, different specs produce different hashes
- `test_run_dir_not_overwritten` — running twice creates two directories

**File:** `tests/test_storage_safety.py`

- `test_unsafe_git_repo_path` — path inside git repo is rejected
- `test_unsafe_tmp_path` — `/tmp` is rejected
- `test_unsafe_virtualenv_path` — path inside venv is rejected
- `test_unsafe_docker_path` — Docker overlay path is rejected
- `test_allow_unsafe_override` — `allow_unsafe=True` bypasses all checks
- `test_safe_external_path` — `/mnt/datasets-big/` is accepted
- `test_low_disk_space_warning` — < 10 GB free triggers warning

### 22.2 Unit tests for schemas

**File:** `tests/test_run_spec.py`

- `test_valid_run_spec` — minimal valid spec loads correctly
- `test_run_spec_from_yaml` — YAML file loads and validates
- `test_run_spec_from_json` — JSON file loads and validates
- `test_run_spec_missing_required_fields` — clear error for missing `name`, `project`, `artifact.path`
- `test_run_spec_invalid_artifact_kind` — rejects unknown kind
- `test_run_spec_invalid_artifact_mode` — rejects unknown mode
- `test_run_spec_endpoint_path` — URL artifact path validates
- `test_run_spec_file_path` — filesystem artifact path validates
- `test_run_spec_serialization` — `model_dump_yaml()` round-trips correctly
- `test_run_spec_cli_flag_build` — building RunSpec from CLI flags produces valid spec

**File:** `tests/test_run_result.py`

- `test_result_summary_from_requests` — computes mean/median/p95 correctly
- `test_result_summary_with_errors` — handles failed requests
- `test_result_summary_empty` — handles zero requests gracefully
- `test_request_result_serialization` — round-trips JSON correctly

**File:** `tests/test_model_artifact.py`

- `test_valid_model_artifact` — all required fields
- `test_model_artifact_optional_fields` — missing optional fields are allowed
- `test_artifact_mode_validation` — validates mode values
- `test_artifact_kind_validation` — validates kind values
- `test_durable_detection` — external path vs /tmp path sets `durable` correctly

**File:** `tests/test_schema_compat.py`

- `test_resolve_known_version` — `llm_bench.run_spec.v1` resolves correctly
- `test_resolve_missing_version` — missing version maps to v1 with defaults
- `test_resolve_future_version` — incompatible future version raises error
- `test_migrate_result_v0_to_v1` — pre-M18 result dict migrates to v1

### 22.3 Unit tests for artifact registry

**File:** `tests/test_artifact_registry.py`

- `test_register_artifact` — appends to JSONL
- `test_lookup_by_id` — finds registered artifact
- `test_lookup_missing` — returns None for unregistered ID
- `test_list_all` — returns all registered artifacts
- `test_query_by_kind` — filters by artifact kind
- `test_query_by_quantization` — filters by quantization method
- `test_query_combined` — multiple filters

**File:** `tests/test_artifact_fingerprint.py`

- `test_compute_config_hash` — consistent hash of config.json
- `test_compute_manifest_hash` — hash of file list + sizes
- `test_fingerprint_stability` — same artifact produces same fingerprint on repeated calls
- `test_different_artifacts_different_fingerprints` — different models produce different fingerprints

**File:** `tests/test_artifact_modes.py`

- `test_external_path_mode` — records path, doesn't copy
- `test_managed_copy_mode` — copies files to artifact store
- `test_managed_symlink_mode` — creates symlink
- `test_managed_copy_incremental` — re-copy skips unchanged files
- `test_managed_symlink_broken_source` — warning for broken symlink

**File:** `tests/test_ephemeral_detection.py`

- `test_warn_tmp_path` — /tmp path triggers warning
- `test_warn_var_tmp_path` — /var/tmp path triggers warning
- `test_warn_docker_path` — Docker overlay path triggers warning
- `test_warn_missing_path` — non-existent path triggers warning
- `test_safe_mnt_path` — /mnt path does not trigger warning
- `test_durable_flag_in_manifest` — ephemeral artifact has `durable: false`

### 22.4 Integration tests with mocked endpoints

**File:** `tests/test_mock_benchmark.py`

Use `aiohttp` or `httpx` mock server to create a fake OpenAI-compatible endpoint:

- `test_end_to_end_benchmark` — fake server responds with completions, full run completes, result directory created
- `test_end_to_end_with_judge` — fake server + fake judge server, judge evaluation recorded
- `test_end_to_end_with_code_task` — fake server returns code, tests run (mocked test runner)
- `test_run_spec_from_file` — load spec from YAML, run against fake server
- `test_multiple_artifacts` — run against two artifacts, two result directories

**File:** `tests/test_runner_factory.py`

- `test_get_known_runner` — returns correct runner class
- `test_get_unknown_runner` — raises ValueError
- `test_openai_compatible_runner_kind` — returns "openai_compatible"
- `test_vllm_runner_kind` — returns "vllm"

**File:** `tests/test_runner_lifecycle.py`

Mock lifecycle tests for `RuntimeRunner`:

- `test_openai_runner_lifecycle` — prepare → wait → run → collect → shutdown (no-op for external)
- `test_runner_collects_v1_models` — captures `/v1/models` response
- `test_runner_collects_server_logs` — writes server.log to result dir

### 22.5 Golden output tests

**File:** `tests/test_golden_outputs.py`

Compare actual output against expected baseline:

- `test_golden_summary_json` — run a mocked benchmark, compare `summary.json` against golden file
- `test_golden_markdown_report` — compare generated markdown report against golden
- `test_golden_compare_output` — compare CLI compare output against golden
- `test_golden_artifact_manifest` — compare `artifact_manifest.json` against golden
- `test_golden_run_spec_yaml` — compare `resolved_spec.yaml` against golden

**Golden files stored in:** `tests/golden/`
- `golden_summary.json`
- `golden_report.md`
- `golden_compare.txt`
- `golden_artifact_manifest.json`
- `golden_resolved_spec.yaml`

Golden files are updated by running `pytest --update-golden` (flag recognized in CI only).

### 22.6 CLI tests

**File:** `tests/test_cli_integration.py`

Use `typer`'s test client or `subprocess` to test CLI commands:

- `test_cli_help` — `--help` works for main app and subcommands
- `test_init_storage_creates_dirs` — `init-storage --root /tmp/test-xxx` creates namespace dirs
- `test_storage_info_prints_root` — `storage-info` prints resolved root
- `test_run_dry_run_no_write` — `run --dry-run` doesn't create result directories
- `test_register_artifact_dry_run` — `register-artifact --dry-run` doesn't write to registry
- `test_inspect_artifact_shows_metadata` — `inspect-artifact /path/to/fake/model` shows files
- `test_list_runs_empty` — no runs registered, empty table
- `test_list_runs_with_projects` — shows filtered runs
- `test_compare_runs` — compares two result directories
- `test_export_summary_markdown` — exports summary to markdown file
- `test_backward_compat_run` — `run --suite smoke --models test` still works
- `test_unsafe_storage_rejected` — storage root in git repo is rejected
- `test_unsafe_storage_override` — `--allow-unsafe-storage-root` works

### 22.7 Test configuration

**File:** `tests/conftest.py`

Shared fixtures:
- `tmp_storage_root` — creates a temporary storage root for each test
- `mock_server` — pytest fixture for a fake OpenAI-compatible server
- `fake_run_spec` — returns a valid `RunSpec` for testing
- `fake_artifact` — returns a `ModelArtifact` for testing
- `golden_dir` — path to `tests/golden/` for comparing outputs
- `storage_config` — `StorageConfig(root=tmp_storage_root)` fixture

**File:** `pytest.ini` or `pyproject.toml [tool.pytest]`

- Add `tests/golden/` to `.gitignore` for auto-updated golden files
- CI runs `pytest --basetemp=/tmp/pytest-tmp` for clean temp directories

## Files Created

- `tests/test_storage_config.py`
- `tests/test_storage_safety.py`
- `tests/test_run_spec.py`
- `tests/test_run_result.py`
- `tests/test_model_artifact.py`
- `tests/test_schema_compat.py`
- `tests/test_artifact_registry.py`
- `tests/test_artifact_fingerprint.py`
- `tests/test_artifact_modes.py`
- `tests/test_ephemeral_detection.py`
- `tests/test_mock_benchmark.py`
- `tests/test_runner_factory.py`
- `tests/test_runner_lifecycle.py`
- `tests/test_golden_outputs.py`
- `tests/test_cli_integration.py`
- `tests/conftest.py` — shared fixtures
- `tests/golden/golden_summary.json`
- `tests/golden/golden_report.md`
- `tests/golden/golden_compare.txt`
- `tests/golden/golden_artifact_manifest.json`
- `tests/golden/golden_resolved_spec.yaml`

## Notes

- All tests use `tmp_path` or `tmp_storage_root` fixture for isolation.
- Mock servers use `pytest-asyncio` for async endpoint testing.
- No test depends on network access or model downloads.
- Golden tests compare against baselines that are generated from the first passing run.
- `--update-golden` flag updates golden files (use sparingly, review diff before committing).
