# Milestone 5 — Coding task runner

## Goal

Evaluate code outputs using executable tests in isolated temp directories.

## Phase

Phase B — Coding usefulness (Milestone 4 of 7 in phase)

## Tasks

### Code task format

- [x] Add `code_type` field to `Task` schema (values: "function_completion", "patch_generation").
- [x] Add `function_signature`, `entry_point`, `test_framework`, `test_code` fields to `TaskExpected` schema.
- [x] Support backward-compatible YAML loading for code tasks.

### Function completion tasks

- [x] Write generated code + test code to isolated temp directory.
- [x] Run pytest via subprocess with `sys.executable -m pytest`.
- [x] Capture pytest stdout/stderr output.
- [x] Parse pytest output for pass/fail counts and individual test results.
- [x] Detect syntax errors in generated code before test execution.
- [x] Handle import errors gracefully (score=0, code_status="import_error").

### Patch generation tasks

- [x] Initialize temp git repo for patch context files.
- [x] Write model-generated patch diff to temp dir.
- [x] Run `git apply --check` to detect patch apply failure.
- [x] Run `git apply` to apply patch on success.
- [x] Detect unrelated file changes via `git diff --name-only` vs expected test_files.
- [x] Run tests after successful patch application.

### Test execution

- [x] Implement `TestExecutionResult` dataclass for test output parsing.
- [x] Implement `_parse_pytest_output()` for stdout/stderr parsing.
- [x] Implement `_detect_syntax_error()` for Python syntax error detection.
- [x] Isolate each code task run in its own temp directory.
- [x] Save artifacts to temp dir (generated_code.py, test_generated.py, test_output.txt, patch.diff).

### CodeTaskRunner integration

- [x] Create `CodeTaskRunner` class in `src/bench_harness/runners/code_task_runner.py`.
- [x] Auto-detect code tasks in `CompletionRunner.run()` via `code_type` field.
- [x] Delegates to `CodeTaskRunner` for code tasks, existing path unchanged for non-code tasks.
- [x] Run secondary scorers (format_compliance, etc.) on generated code via scorer registry.
- [x] Store code-specific fields on `RunResult`: tests_passed, tests_failed, tests_total, test_output, exit_code, generated_code, code_status.

### Unit test scorer

- [x] Create `unit_test` scorer in `src/bench_harness/scorers/unit_test.py`.
- [x] Validates tasks with `expected.type == "unit_test"`.
- [x] Checks generated code for expected patterns (function definition).
- [x] Scores 0.5 for pattern match, 0.0 for no match.

### Artifact storage

- [x] Extend `save_run_artifact()` with code fields when present.
- [x] Include generated_code, code_status, tests_passed/failed/total in JSONL output.

### Coding smoke tasks

- [x] Create `tasks/coding_smoke/simple_math.yaml` — `plus(a, b)` function completion.
- [x] Create `tasks/coding_smoke/string_ops.yaml` — `reverse_string(s)` function completion.
- [x] Create `tasks/coding_smoke/list_utils.yaml` — `find_max(lst)` function completion.
- [x] Create `tasks/coding_smoke/bugfix_simple.yaml` — `add(a, b)` patch generation.

### CLI integration

- [x] `list-tasks` discovers and displays coding_smoke tasks.
- [x] `run --suite coding_smoke` executes coding tasks.
- [x] `show-task smoke.code_math_001` displays code task details.

### Testing

- [x] `tests/test_code_task_runner.py` — 35 tests covering all scenarios.
  - Function completion: passing, syntax error, all fail, partial pass.
  - Patch generation: successful apply, apply failure, unrelated file changes.
  - Artifact file creation verification.
  - Edge cases: empty code, no test_code, whitespace-only, exit codes.

## Acceptance criteria

- [x] Can run HumanEval/MBPP-style local tasks with real test execution.
- [x] Can score pass/fail from unit tests (fraction of tests passed).
- [x] Raw generated code, test code, and test logs are saved as artifacts.
- [x] All 4 coding smoke tasks execute successfully with correct scoring.
- [x] `pytest tests/test_code_task_runner.py` passes with full coverage.
- [x] `python -m bench_harness list-tasks` shows coding tasks.
