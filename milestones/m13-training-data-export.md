# Milestone 13 — Training-Data Export

## Goal

Turn benchmark results into future fine-tuning and preference datasets. Produce SFT, DPO/ORPO, regression, and judge-labeled training data from completed benchmark runs.

## Phase

Phase D — Data flywheel (Milestone 1 of 3 in phase)

## Dependencies

- Milestone 7 (LLM judge integration — judge_evaluations and pairwise_comparisons tables)
- Milestone 5 (Code task runner — generated_code field in run results)
- Milestone 1 (SQLite storage — persistent run data)

---

### Leveraged Libraries

- **sqlite-utils** (already in pyproject.toml): SQLite storage for query access
- **yaml** (stdlib + pyyaml in pyproject.toml): YAML output for regression tasks
- **typer** (already in pyproject.toml): CLI for export subcommand
- **json** (stdlib): JSONL output for SFT, preference, and judge exports

---

## Subtasks

### 13.1 Implement base export helpers

**File:** `src/bench_harness/export/base.py` (update)

**Functions:**
```python
def get_runs_by_suite(db_path: str, suite_id: str) -> list[dict]
def get_task_by_id(db_path: str, task_id: str) -> dict | None
def get_tasks_from_task_dir(task_dir: str) -> dict[str, dict]
def get_judge_evaluations(db_path: str, suite_id: str) -> list[dict]
def get_pairwise_comparisons(db_path: str, suite_id: str) -> list[dict]
```

**Actions:**
- [x] Add get_runs_by_suite() to query runs table by suite_id
- [x] Add get_task_by_id() to look up task YAML files by task ID
- [x] Add get_tasks_from_task_dir() to load and index tasks from directories
- [x] Add get_judge_evaluations() to query judge_evaluations joined with runs
- [x] Add get_pairwise_comparisons() to query pairwise_comparisons joined with runs

### 13.2 Implement SFT export

**File:** `src/bench_harness/export/sft_export.py` (new)

**Function:**
```python
def export_sft(db_path, suite_id, out_path=None, min_score=0.0, include_system_messages=True) -> str
```

**Behavior:**
- Filters runs to exit_status == "success" and score_primary >= min_score
- Builds OpenAI messages format: [system?, user, assistant]
- For code tasks, uses generated_code as assistant content
- Includes model, family, task_id, score, prompt_style, quantization metadata
- Outputs JSONL to exports/sft_openai_messages.jsonl

**Actions:**
- [x] Filter successful runs with min_score threshold
- [x] Build messages array with optional system message from task input
- [x] Use generated_code as assistant content for code tasks
- [x] Include model, family, task_id, score metadata
- [x] Include prompt_style and quantization when available
- [x] Write JSONL output file

### 13.3 Implement preference export (score-based)

**File:** `src/bench_harness/export/preference_export.py` (new)

**Function:**
```python
def export_preference_score_based(db_path, suite_id, out_path=None, min_margin=0.0) -> str
```

**Behavior:**
- Groups successful runs by (task_id, suite_id)
- Picks highest-scored model as "chosen" and lowest as "rejected"
- Only includes pairs where margin >= min_margin
- Output format: messages, chosen {model, score}, rejected {model, score}, margin, task_id, prompt

**Actions:**
- [x] Group runs by (task_id, suite_id)
- [x] Sort by score and pick best/worst model
- [x] Apply min_margin filter to exclude close pairs
- [x] Include messages, chosen, rejected, margin, task_id, prompt in output
- [x] Write JSONL output file

### 13.4 Implement preference export (pairwise)

**File:** `src/bench_harness/export/preference_export.py` (update — add function)

**Function:**
```python
def export_preference_from_pairwise(db_path, suite_id, out_path=None) -> str
```

**Behavior:**
- Uses pairwise_comparisons table directly
- Maps winner field to chosen/rejected model
- Includes reason, confidence, margin

**Actions:**
- [x] Query pairwise_comparisons for suite
- [x] Map winner A/B to chosen/rejected model
- [x] Skip ties (winner == "tie")
- [x] Include reason, confidence, margin in output
- [x] Write JSONL output file

### 13.5 Implement regression export

**File:** `src/bench_harness/export/regression_export.py` (new)

**Function:**
```python
def export_regression(db_path, suite_id, out_path=None) -> str
```

**Behavior:**
- Groups failed runs by task_id
- Excludes runs with empty raw_response and empty error_message (pure API errors)
- Includes full task_definition with failures array
- Output format: YAML with task_id, family, prompt, expected, task_definition, failures

**Actions:**
- [x] Group failed runs by task_id
- [x] Exclude pure API errors (empty raw_response and error_message)
- [x] Include full task_definition from task YAML files
- [x] Include failures array with model, error/raw_response/score per failure
- [x] Write YAML output file

### 13.6 Implement judge export

**File:** `src/bench_harness/export/judge_export.py` (new)

**Functions:**
```python
def export_judge(db_path, suite_id, out_path=None) -> str
def export_judge_pairwise(db_path, suite_id, out_path=None) -> str
```

**Behavior (export_judge):**
- Queries judge_evaluations joined with runs
- Parses dimensions_json into structured dict
- Includes run-level enrichment (prompt, raw_response, score_primary)
- Marks is_pairwise=False

**Behavior (export_judge_pairwise):**
- Queries pairwise_comparisons
- Parses dimension_comparison_json
- Marks is_pairwise=True

**Actions:**
- [x] Query judge_evaluations for suite
- [x] Parse dimensions_json into structured dict
- [x] Enrich with run-level data (prompt, raw_response)
- [x] Include is_pairwise=False and pairwise=null
- [x] Write judge JSONL output file
- [x] Export pairwise comparisons as separate JSONL
- [x] Parse dimension_comparison_json for pairwise output

### 13.7 Add CLI export subcommand

**File:** `src/bench_harness/cli.py` (update)

**New command:**
```bash
bench-harness export --action sft --suite test-suite --out exports/
bench-harness export --action preference --suite test-suite --min-margin 0.1
bench-harness export --action preference-pairwise --suite test-suite
bench-harness export --action regression --suite test-suite
bench-harness export --action judge --suite test-suite
bench-harness export --action judge-pairwise --suite test-suite
bench-harness export --action all --suite test-suite --out exports/
```

**Actions:**
- [x] Add export subcommand with --action flag (sft, preference, preference-pairwise, regression, judge, judge-pairwise, all)
- [x] Add --action all to export all four formats
- [x] Add --min-margin flag for preference export
- [x] Add --out flag for output directory
- [x] Add --db flag for database path (defaults to runs/*/benchmark.db)
- [x] Error on unknown action

### 13.8 Add comprehensive tests

**File:** `tests/test_export.py` (new)

**Test categories:**
- SFT export: basic, min_score filter, system_message, code task, empty result, format
- Preference export: score-based grouping, min_margin, pairwise, no dupe models, format
- Regression export: basic grouping, API error exclusion, YAML format, task_definition, empty
- Judge export: basic, dimensions, pairwise, format
- Base helpers: get_runs_by_suite, get_tasks_from_dir, get_judge_evaluations, get_pairwise_comparisons
- CLI integration: SFT export, all formats, invalid action

**Actions:**
- [x] Test SFT export with basic successful runs
- [x] Test SFT min_score filter excludes low-scoring runs
- [x] Test SFT includes system message when present in task
- [x] Test SFT code tasks include generated_code as assistant content
- [x] Test SFT empty result when no successful runs
- [x] Test SFT output format matches OpenAI messages JSONL
- [x] Test preference score-based groups by task_id, picks best/worst
- [x] Test preference min_margin filter
- [x] Test preference from pairwise table
- [x] Test preference no model appears as both chosen and rejected
- [x] Test preference JSON format
- [x] Test regression exports failed runs grouped by task
- [x] Test regression excludes API errors (empty raw_response)
- [x] Test regression YAML format with task_definition and failures
- [x] Test regression no failures produces empty list
- [x] Test judge export from judge_evaluations table
- [x] Test judge dimensions parsing
- [x] Test judge pairwise export
- [x] Test judge JSON format
- [x] Test base get_runs_by_suite queries correctly
- [x] Test base get_tasks_from_dir loads tasks from directory
- [x] Test base get_judge_evaluations queries correctly
- [x] Test base get_pairwise_comparisons queries correctly
- [x] Test CLI export sft integration
- [x] Test CLI export all four formats
- [x] Test CLI export invalid action error

---

## Acceptance Criteria Checklist

- [x] A failed benchmark task can become a regression test
- [x] A high-quality response can become an SFT example
- [x] A model comparison can become preference data
- [x] SFT export produces valid OpenAI messages-format JSONL
- [x] SFT export supports min_score filtering
- [x] SFT export includes system messages when available
- [x] SFT export includes generated_code for code tasks
- [x] Preference export groups by task_id and picks best/worst models
- [x] Preference export supports min_margin threshold
- [x] Preference export from pairwise uses pairwise_comparisons table
- [x] Regression export groups failed runs by task with full task_definition
- [x] Regression export excludes pure API errors
- [x] Judge export includes per-dimension scores
- [x] All exports produce valid JSONL/YAML output
- [x] All exports tested with in-memory SQLite database (no mocking)
- [x] 27 tests covering all export functions and base helpers
- [x] CLI export subcommand with action-based dispatch
- [x] CLI supports --action all for bulk export

---

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/export/base.py` | Updated (added get_judge_evaluations, get_pairwise_comparisons) |
| `src/bench_harness/export/__init__.py` | Created (module init with exports) |
| `src/bench_harness/export/sft_export.py` | Created (SFT JSONL export) |
| `src/bench_harness/export/preference_export.py` | Created (DPO/ORPO JSONL export) |
| `src/bench_harness/export/regression_export.py` | Created (regression YAML export) |
| `src/bench_harness/export/judge_export.py` | Created (judge JSONL export) |
| `src/bench_harness/cli.py` | Updated (added export subcommand) |
| `tests/test_export.py` | Created (27 tests for all export functions) |
| `ROADMAP.md` | Updated (M13 marked done with detailed acceptance criteria) |
| `README.md` | Updated (status line and milestone table) |
