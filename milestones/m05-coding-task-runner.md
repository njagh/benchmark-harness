# Milestone 5 — Coding Task Runner

## Goal

Evaluate code outputs using executable tests. Support function-completion tasks (HumanEval/MBPP-style) and patch-generation tasks with isolated execution, artifact capture, and deterministic scoring.

## Phase

Phase B — Real coding usefulness (Milestone 3 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap, runner, storage)
- Milestone 2 (task schema with test_files and scoring metadata)
- Milestone 3 (timing metrics)
- Milestone 4 (basic scorers — regex scorer used for code extraction)

### Leveraged Libraries

This milestone leverages existing libraries rather than building execution infrastructure from scratch:

- **bigcode-tools** (`bigcode-evaluation-hub`): Provides HumanEval/MBPP evaluation infrastructure including function extraction, isolated execution, timeout handling, and pass@k computation. Use for function-completion tasks.
- **unidiff**: Parses unified diffs into structured objects (chunks, lines, file paths). Use for patch extraction and minimality scoring.
- **eval_plus** (`evalplus`): Deobfuscated HumanEval+/MBPP+ with stronger test suites. Use as the test source for coding tasks rather than the original HumanEval tests.

---

## Subtasks

### 5.1 Define code task format

**File:** `src/bench_harness/tasks/task_schema.py` (update)

**New schema fields for code tasks:**

```python
class CodeTaskInput(TaskInput):
    stub_code: str | None = None        # function stub for completion tasks
    original_file: str | None = None     # path to original file for patch tasks
    test_file: str | None = None         # path to test file to run
    setup_script: str | None = None      # optional setup script path

class CodeTaskExpected(TaskExpected):
    test_files: list[str] | None = None  # paths to test files
    test_command: str | None = None       # command to run tests (default: pytest)
    timeout_seconds: float = 30.0         # execution timeout per test
    expected_files: list[str] | None = None  # files that should exist after generation
    allowed_file_changes: list[str] | None = None  # files the model is allowed to modify
```

**Task YAML example (function completion):**
```yaml
id: coding.humaneval.style_001
family: coding
category: function_completion
prompt_template: function_completion.md
input:
  user_message: "Complete the following function:"
  stub_code: |
    def two_sum(nums: list[int], target: int) -> list[int]:
        '''Given an array of integers, return indices of the two numbers
        that add up to target.'''
expected:
  type: unit_test
  test_files:
    - "tasks/coding_smoke/tests/test_two_sum.py"
  test_command: "python -m pytest {test_file} -v"
  timeout_seconds: 15
scoring:
  primary: unit_tests
  secondary:
    - format_compliance
risk_level: low
```

**Task YAML example (patch generation):**
```yaml
id: coding.patch.fix_bug_001
family: coding
category: patch_generation
prompt_template: patch_generation.md
input:
  user_message: "Fix the bug in this file:"
  original_file: "tasks/coding_smoke/fixtures/buggy_add.py"
  files:
    - "tasks/coding_smoke/fixtures/buggy_add.py"
expected:
  type: unit_test
  test_files:
    - "tasks/coding_smoke/tests/test_add_fixed.py"
  test_command: "python -m pytest {test_file} -v"
  timeout_seconds: 10
  allowed_file_changes:
    - "buggy_add.py"
scoring:
  primary: unit_tests
  secondary:
    - patch_minimality
    - unrelated_changes
risk_level: low
```

**Actions:**
- [ ] Extend Task schema with code-specific fields
- [ ] Add `category` field values: `function_completion`, `patch_generation`, `file_creation`
- [ ] Update loader to handle code task fields
- [ ] Add `bigcode-tools` and `evalplus` to pyproject.toml as optional dependencies (they have heavier deps; keep as `[project.optional-dependencies]`)

### 5.2 Implement code runner base class

**File:** `src/bench_harness/runners/code_runner.py`

**Class:** `CodeRunner`

```python
class CodeRunner:
    def __init__(self, client: OpenAICompatClient, work_dir: str | None = None):
        self.client = client
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="bench_code_")

    def run(self, task: Task, params: dict) -> CodeRunResult:
        """Run a coding task: generate code, extract it, execute tests, return result."""

    def cleanup(self):
        """Remove temporary work directory."""
```

**Class:** `CodeRunResult`

```python
@dataclass
class CodeRunResult:
    run_id: str
    suite_id: str
    task_id: str
    model_alias: str
    task_category: str          # function_completion, patch_generation
    raw_response: str
    extracted_code: str         # code extracted from response
    extraction_method: str      # how code was extracted (code_fence, full_response, etc.)
    generated_file_path: str | None  # path where code was written
    test_stdout: str            # captured stdout from test run
    test_stderr: str            # captured stderr from test run
    test_exit_code: int         # 0 = pass, non-zero = fail
    tests_passed: int           # number of tests that passed
    tests_failed: int           # number of tests that failed
    tests_total: int            # total number of tests
    patch_applies: bool | None  # for patch tasks: whether diff applied cleanly
    unrelated_file_changes: list[str] | None  # files changed outside allowed list
    score_primary: float        # 0.0 or 1.0 (all tests pass or not)
    timing: StreamMetrics | None
    token_usage: TokenCounter | None
    error_message: str | None
    created_at: str
```

**Actions:**
- [ ] Implement `CodeRunner` class with `run()` and `cleanup()`
- [ ] Implement `CodeRunResult` dataclass
- [ ] Create isolated temp directory per task execution
- [ ] Support configurable work directory for debugging

### 5.3 Implement code extraction from LLM response

**File:** `src/bench_harness/runners/code_extractor.py`

**Functions:**

```python
def extract_code_blocks(response: str, language: str | None = None) -> list[str]:
    """Extract fenced code blocks from markdown response.
    
    Returns list of code strings. If language is specified, only
    returns blocks matching that language tag.
    """

def extract_python_function(response: str, function_name: str | None = None) -> str | None:
    """Extract a Python function definition from response text.
    
    Handles:
    - Code in ```python ... ``` blocks
    - Bare function definitions
    - Multiple functions (returns first or named one)
    """

def extract_unified_diff(response: str) -> str | None:
    """Extract a unified diff patch from response text.
    
    Handles:
    - Code-fenced diff blocks
    - Bare diff output starting with --- or +++
    """

def best_effort_code_extract(response: str) -> tuple[str, str]:
    """Best-effort extraction returning (code, method_used).
    
    Tries in order: code fence, bare function, full response.
    """
```

bigcode-tools provides `extract_code()` utilities for HumanEval-style extraction. Adapt these rather than writing from scratch. The harness's `code_extractor.py` should wrap or extend bigcode-tools extraction with additional support for patch-only and multi-file responses.

**Actions:**
- [ ] Implement code fence extraction with language filter
- [ ] Implement Python function extraction with regex
- [ ] Implement unified diff extraction
- [ ] Implement best-effort extraction fallback chain
- [ ] Log extraction method used for debugging

### 5.4 Implement function-completion task execution

**File:** `src/bench_harness/runners/code_runner.py` (extend)

**Method:** `run_function_completion(task: CodeTaskInput, raw_response: str, work_dir: str) -> CodeRunResult`

**Execution flow:**
1. Extract code from response using `extract_python_function()` or `extract_code_blocks()`
2. Combine stub code + generated code into a single file
3. Write file to temp work directory
4. Copy test file(s) to work directory
5. Execute test command in work directory using `subprocess.run()`
6. Capture stdout, stderr, exit code
7. Parse pytest output for pass/fail counts
8. Return `CodeRunResult`

bigcode-tools provides `execute_sample()` which handles isolated temp directory creation, test file copying, subprocess execution with timeout, and output parsing. Use as the base execution layer. The harness wraps it to capture additional metrics (TTFT, GPU memory) and to integrate with the scorer pipeline.

**Test execution details:**
```python
result = subprocess.run(
    test_command,
    shell=True,
    cwd=work_dir,
    capture_output=True,
    text=True,
    timeout=timeout_seconds,
    env=env  # restricted environment
)
```

**Pytest output parsing:**
- Look for `N passed, M failed` pattern
- Look for `FAILED` lines for individual test names
- Fall back to exit code (0 = all pass)

**Actions:**
- [ ] Implement function-completion execution flow
- [ ] Implement subprocess execution with timeout
- [ ] Implement pytest output parsing
- [ ] Handle `subprocess.TimeoutExpired` gracefully
- [ ] Restrict environment variables in subprocess

### 5.5 Implement patch-generation task execution

**File:** `src/bench_harness/runners/code_runner.py` (extend)

**Method:** `run_patch_generation(task: CodeTaskInput, raw_response: str, work_dir: str) -> CodeRunResult`

**Execution flow:**
1. Extract unified diff from response using `extract_unified_diff()`
2. Copy original file(s) to work directory
3. Apply patch using `subprocess.run(["git", "apply", ...])` or `patch` command
4. If patch fails, mark `patch_applies = False` and skip test execution
5. If patch succeeds, copy test file(s) and run tests
6. Check for unrelated file changes:
   - Compare files in work directory before/after patch
   - Flag any modified files not in `allowed_file_changes`

Use `unidiff` library to parse unified diffs from model responses. `unidiff.parse_patch()` returns structured Chunk objects with added/removed/lines. Combine with `filecmp` (stdlib) for unrelated-change detection.

**Patch application:**
```python
patch_result = subprocess.run(
    ["git", "apply", "--check", patch_file],
    cwd=work_dir,
    capture_output=True,
    text=True
)
# If --check passes, apply for real
subprocess.run(
    ["git", "apply", patch_file],
    cwd=work_dir,
    capture_output=True,
    text=True
)
```

**Unrelated change detection:**
```python
def detect_unrelated_changes(
    work_dir: str,
    allowed_files: list[str],
    baseline_files: list[str]
) -> list[str]:
    """Return list of changed files not in allowed set."""
```

**Actions:**
- [ ] Implement patch extraction and application
- [ ] Implement `git apply --check` validation
- [ ] Implement unrelated file change detection
- [ ] Handle patch format variations (1-based vs 0-based line numbers)

### 5.6 Implement unit-test scorer

**File:** `src/bench_harness/scorers/unit_tests.py`

**Class:** `UnitTestScorer(BaseScorer)`

```python
class UnitTestScorer(BaseScorer):
    name = "unit_tests"

    def score(self, task: Task, raw_response: str, code_run_result: CodeRunResult | None = None) -> ScoreResult:
        """Score based on unit test pass/fail results."""
```

**Behavior:**
- If `code_run_result` is provided, use its test results directly
- Score is `tests_passed / tests_total` (0.0 to 1.0)
- `passed` is `True` only if all tests pass
- `details` includes per-test pass/fail if available

**Actions:**
- [ ] Implement scorer using `CodeRunResult` data
- [ ] Register with `@register_scorer`
- [ ] Handle case where no tests ran (score = 0.0, explanation = error message)

### 5.7 Implement patch minimality scorer

**File:** `src/bench_harness/scorers/patch_minimality.py`

**Class:** `PatchMinimalityScorer(BaseScorer)`

```python
class PatchMinimalityScorer(BaseScorer):
    name = "patch_minimality"

    def score(self, task: Task, raw_response: str, code_run_result: CodeRunResult | None = None) -> ScoreResult:
        """Score patch by lines changed. Fewer lines = higher score (if tests pass)."""
```

**Behavior:**
- Parse unified diff to count added/removed/changed lines
- Score is inversely proportional to lines changed
- Baseline: tasks include `expected_lines_changed` as reference
- Score = `min(expected_lines_changed, actual_lines_changed) / max(expected_lines_changed, actual_lines_changed)`
- If tests fail, score = 0.0 regardless

**Actions:**
- [ ] Implement diff line counting
- [ ] Implement minimality scoring formula
- [ ] Register with `@register_scorer`

### 5.8 Implement artifact storage for code tasks

**File:** `src/bench_harness/storage/artifacts.py` (update)

**New function:**
```python
def save_code_artifacts(
    result: CodeRunResult,
    out_dir: str,
    raw_response: str,
    extracted_code: str,
    test_output: str
) -> dict[str, str]:
    """Save all artifacts for a code task run.
    
    Returns dict of artifact name -> file path.
    """
```

**Artifacts saved:**
- `{out_dir}/raw_response_{task_id}.txt` — full LLM response
- `{out_dir}/extracted_code_{task_id}.py` — extracted code
- `{out_dir}/test_output_{task_id}.txt` — test stdout/stderr
- `{out_dir}/patch_{task_id}.diff` — generated patch (if patch task)
- `{out_dir}/result_{task_id}.json` — structured CodeRunResult as JSON

**Actions:**
- [ ] Implement artifact save function
- [ ] Save all artifacts with deterministic naming
- [ ] Update runner to call `save_code_artifacts` after each code task

### 5.9 Extend SQLite schema for code tasks

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `runs` table:**
```sql
ALTER TABLE runs ADD COLUMN task_category TEXT;
ALTER TABLE runs ADD COLUMN extracted_code TEXT;
ALTER TABLE runs ADD COLUMN extraction_method TEXT;
ALTER TABLE runs ADD COLUMN tests_passed INTEGER;
ALTER TABLE runs ADD COLUMN tests_failed INTEGER;
ALTER TABLE runs ADD COLUMN tests_total INTEGER;
ALTER TABLE runs ADD COLUMN test_exit_code INTEGER;
ALTER TABLE runs ADD COLUMN patch_applies INTEGER;
ALTER TABLE runs ADD COLUMN unrelated_changes TEXT;  -- JSON list
ALTER TABLE runs ADD COLUMN generated_file_path TEXT;
ALTER TABLE runs ADD COLUMN test_stdout_path TEXT;
ALTER TABLE runs ADD COLUMN test_stderr_path TEXT;
```

**Methods (add):**
- `save_code_run(result: CodeRunResult)` — saves code-specific fields
- `get_code_runs(suite_id: str | None, category: str | None) -> list[dict]`

**Actions:**
- [ ] Implement schema migration
- [ ] Implement `save_code_run` method
- [ ] Store extracted code in SQLite (may be large, consider compression)

### 5.10 Add coding_smoke suite config and tasks

**File:** `configs/suites.yaml` (update)

```yaml
suites:
  # ... existing ...
  coding_smoke:
    description: "Fast coding quality regression"
    task_dir: "tasks/coding_smoke"
    families:
      - coding
      - debugging
    max_concurrency: 2
    default_runs: 1
    default_temperature: 0
    runner: code_runner
```

**Task directory:** `tasks/coding_smoke/`

**Sample tasks to create (6 minimum):**

1. `tasks/coding_smoke/function_fibonacci.yaml` — write fibonacci function
2. `tasks/coding_smoke/function_two_sum.yaml` — write two_sum function
3. `tasks/coding_smoke/fix_buggy_add.yaml` — fix a-b to a+b
4. `tasks/coding_smoke/explain_stacktrace.yaml` — explain a Python stack trace
5. `tasks/coding_smoke/improve_benchmark.yaml` — improve a timing script
6. `tasks/coding_smoke/identify_transformer_bug.yaml` — identify bug in attention code

**Test fixtures:**
```
tasks/coding_smoke/
  function_fibonacci.yaml
  function_two_sum.yaml
  fix_buggy_add.yaml
  explain_stacktrace.yaml
  improve_benchmark.yaml
  identify_transformer_bug.yaml
  fixtures/
    buggy_add.py
    benchmark_script.py
    attention_module.py
  tests/
    test_fibonacci.py
    test_two_sum.py
    test_add_fixed.py
```

**Actions:**
- [ ] Create all 6 task YAML files
- [ ] Create corresponding fixture files
- [ ] Create corresponding test files
- [ ] Update suites.yaml with coding_smoke suite
- [ ] Wire `runner: code_runner` in suite config to the CLI

### 5.11 Wire code runner into CLI

**File:** `src/bench_harness/cli.py` (update)

**Updates:**
- Suite config `runner` field determines which runner to use
- `runner: completion_runner` (default) uses `CompletionRunner`
- `runner: code_runner` uses `CodeRunner`
- CLI passes `--work-dir` option for code runner (for debugging)

**New CLI flag:**
```
--work-dir   PATH   Working directory for code tasks (default: auto tempdir)
```

**Actions:**
- [ ] Update CLI to select runner based on suite config
- [ ] Add `--work-dir` flag
- [ ] Ensure code runner artifacts are saved alongside JSONL

### 5.12 Add code runner tests

**File:** `tests/test_code_runner.py`

**Tests:**
- `test_extract_code_block_python` — extracts Python code from markdown fence
- `test_extract_code_block_no_fence` — extracts bare function definition
- `test_extract_unified_diff` — extracts diff from response
- `test_extract_diff_from_fence` — extracts diff from ```diff fence
- `test_best_effort_fallback` — falls through extraction chain
- `test_function_completion_execution` — end-to-end function completion in temp dir
- `test_function_completion_timeout` — handles subprocess timeout
- `test_patch_generation_apply` — applies patch and runs tests
- `test_patch_generation_fail` — detects patch application failure
- `test_unrelated_change_detection` — detects changes outside allowed files
- `test_unit_test_scorer_all_pass` — scores 1.0 when all tests pass
- `test_unit_test_scorer_partial` — scores fraction when some tests fail
- `test_unit_test_scorer_no_result` — handles missing CodeRunResult
- `test_patch_minimality_scorer` — scores inversely on lines changed
- `test_code_artifact_save` — all artifacts written to disk

**Actions:**
- [ ] Implement all tests
- [ ] Use temp directories for execution tests
- [ ] Mock subprocess for tests that don't need real execution
- [ ] Ensure no network calls in tests

---

## Acceptance Criteria Checklist

- [ ] HumanEval/MBPP-style function completion tasks run end-to-end with test execution
- [ ] Patch-generation tasks extract diff, apply patch, and run tests
- [ ] Test pass/fail is scored deterministically
- [ ] Raw generated code, test logs, and patches are saved as artifacts
- [ ] Unrelated file changes are detected and reported
- [ ] `coding_smoke` suite runs with at least 6 tasks
- [ ] Code runner artifacts are stored in SQLite and on disk
- [ ] `pytest tests/test_code_runner.py` passes

## Estimated Effort

3–4 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/runners/code_runner.py` | To create |
| `src/bench_harness/runners/code_extractor.py` | To create |
| `src/bench_harness/scorers/unit_tests.py` | To create |
| `src/bench_harness/scorers/patch_minimality.py` | To create |
| `src/bench_harness/storage/artifacts.py` | Update (code artifacts) |
| `src/bench_harness/storage/sqlite.py` | Update (code schema migration) |
| `src/bench_harness/tasks/task_schema.py` | Update (code task fields) |
| `src/bench_harness/cli.py` | Update (code runner selection) |
| `configs/suites.yaml` | Update (coding_smoke suite) |
| `tasks/coding_smoke/*.yaml` (×6) | To create |
| `tasks/coding_smoke/fixtures/*.py` | To create |
| `tasks/coding_smoke/tests/test_*.py` | To create |
| `tests/test_code_runner.py` | To create |
