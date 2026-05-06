# Milestone 10 — Quantization Comparison Suite

## Goal

Measure quality and performance impact of quantization, identifying whether quantization degrades coding, math, formatting, or instruction-following capability, and finding the best quantized model per task family.

## Phase

Phase C — Deep comparison (Milestone 2 of 3 in phase)

## Dependencies

- Milestone 3 (timing metrics — TTFT, decode, tokens/sec)
- Milestone 5 (coding task runner — code-specific quantization impact)
- Milestone 7 (LLM judge integration — judge-scored comparison)
- Milestone 8 (prompt style comparison — style isolation during comparison)

---

### Leveraged Libraries

- **typer** (already in pyproject.toml): CLI for quantization-aware runs
- **rich** (already in pyproject.toml): Terminal output for comparison progress
- **pynvml** (in pyproject.toml optional `[long-context]`): GPU memory impact of quantization
- **duckdb** (in pyproject.toml optional `[long-context]`): SQL queries on quantized run comparisons

---

## Subtasks

### 10.1 Add quantization field to RunResult and SQLite schema

**File:** `src/bench_harness/models/run_result.py` (update)

**RunResult additions:**
```python
class RunResult(BaseModel):
    ...
    quantization: str | None = None  # "FP8", "NVFP4", "GPTQ_Int4", "none", etc.
```

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Migrations:**
- `runs` table: add `quantization TEXT` column
- `run_timings` table: add `quantization TEXT` column
- Safe migration on both fresh and existing databases

**Actions:**
- [x] Add quantization field to RunResult
- [x] Add JSON schema and serialization support
- [x] Add quantization column to runs table (safe migration)
- [x] Add quantization column to run_timings table (safe migration)
- [x] Extract quantization from model config during run

### 10.2 Add CLI quantization-aware model config extraction

**File:** `src/bench_harness/cli.py` (update)

**Behavior:**
- CLI automatically extracts quantization from model config (`configs/models.yaml`)
- The `notes` field on models contains quantization info (e.g., "Qwen3.6-35B-A3B-FP8 via LiteLLM")
- Parse quantization scheme from model name/notes when not explicitly set

**Actions:**
- [x] Implement quantization extraction from model config
- [x] Parse common quantization patterns (FP8, NVFP4, GPTQ_Int4, AWQ, etc.)
- [x] Set quantization field on RunResult when not explicitly provided
- [x] Document quantization field in CLI help

### 10.3 Create quantization-sensitive task suite

**Directory:** `tasks/synthetic/quantization_sensitive/`

**Task types:**
1. `math_reasoning` — multi-step math problems sensitive to numeric precision
2. `code_generation` — coding tasks where quantization may break syntax/logic
3. `format_compliance` — strict output format (JSON, markdown, numbered lists)
4. `instruction_following` — multi-constraint tasks where quantization loses details
5. `long_context_retrieval` — long-context fact retrieval (requires M9 context packer)
6. `hallucination_resistance` — tasks where quantization may increase fabrication

**Actions:**
- [x] Define YAML task files for each quantization-sensitive category
- [x] Include ground truth answers for exact/deterministic scoring
- [x] Include rubric dimensions for judge scoring
- [x] Register in task registry under `synthetic/quantization_sensitive` family

### 10.4 Implement quantization comparison report

**File:** `src/bench_harness/reports/quantization_comparison.py` (new)

**Function:**
```python
def generate_quantization_report(runs, suite_id="", prior_runs=None) -> str
```

**Sections:**
1. **Quantization Summary** — per quantization: avg score, avg speed, task count
2. **Quality Delta Analysis** — score difference from reference (FP8 or dense) quantization
3. **Best Quantization Per Task Family** — which quantization wins per family
4. **Speed/Quality Pareto Frontier** — models plotted by speed vs quality by quantization
5. **Sensitivity Analysis** — which task families are most/least affected by quantization

**Actions:**
- [x] Implement generate_quantization_report() with all 5 sections
- [x] Implement _group_by_quantization() to partition runs
- [x] Implement _quality_delta() to compute score difference from reference
- [x] Implement _best_quantization_per_family() to find top quantization by family
- [x] Implement _pareto_frontier() to identify speed/quality efficient models
- [x] Implement _sensitivity_analysis() to rank families by quantization impact
- [x] Handle runs without quantization metadata gracefully (return empty string)

### 10.5 Integrate quantization report into markdown report

**File:** `src/bench_harness/reports/markdown.py` (update)

**Changes:**
- Add `_append_quantization_comparison(lines, runs)` helper
- Detect runs with quantization metadata
- Call generate_quantization_report() when quantization metadata is present
- Append after coding agent ranking section (before style comparison)

**Actions:**
- [x] Add _append_quantization_comparison() helper function
- [x] Import generate_quantization_report in markdown.py
- [x] Integrate into generate_report() flow
- [x] Auto-detect quantization metadata and append comparison sections

### 10.6 Add comprehensive tests

**File:** `tests/test_quantization_comparison.py` (new)

**Test categories:**
- RunResult quantization field serialization
- Quantization extraction from model config notes
- Quantization comparison report: all 5 sections present
- Quality delta calculation with known deltas
- Best quantization per family identification
- Pareto frontier computation
- Sensitivity analysis ranking families correctly
- SQLite schema migrations for quantization columns
- Markdown report integration with quantization sections

**Actions:**
- [x] Test RunResult quantization field serialization to JSON
- [x] Test quantization extraction from various config formats
- [x] Test quantization comparison report generation with sample data
- [x] Test quality delta computation accuracy
- [x] Test best quantization per family with known data
- [x] Test Pareto frontier identification
- [x] Test sensitivity analysis ranking
- [x] Test SQLite migrations for quantization columns
- [x] Test markdown report integration (with/without quantization)

---

## Acceptance Criteria Checklist

- [x] quantization field added to RunResult and SQLite schemas
- [x] CLI automatically extracts quantization from model config
- [x] Quantization-sensitive task suite covers math, code, format, instruction-following
- [x] Quantization comparison report has all 5 sections (summary, quality delta, best per family, Pareto frontier, sensitivity)
- [x] Report shows whether quantization hurts coding, math, long-context, or formatting
- [x] Report identifies best quantized model by task family
- [x] Quantization impact is separated from prompt/backend differences (matched task sets, same sampling)
- [x] Markdown report auto-detects quantization metadata and appends comparison sections
- [x] SQLite migrations for quantization columns are safe on fresh and existing databases
- [x] All tests pass (quantization comparison, report generation, integration)

---

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/models/run_result.py` | Updated (quantization field) |
| `src/bench_harness/storage/sqlite.py` | Updated (runs/run_timings migrations for quantization) |
| `src/bench_harness/reports/quantization_comparison.py` | Created (generate_quantization_report with 5 sections) |
| `src/bench_harness/reports/markdown.py` | Updated (_append_quantization_comparison integration) |
| `src/bench_harness/cli.py` | Updated (quantization extraction from config) |
| `tasks/synthetic/quantization_sensitive/` | Created (quantization-sensitive task definitions) |
| `tests/test_quantization_comparison.py` | Created (comprehensive comparison tests) |
| `ROADMAP.md` | Updated (M10 marked done with detailed acceptance criteria) |
| `README.md` | Updated (status line and milestone table) |
