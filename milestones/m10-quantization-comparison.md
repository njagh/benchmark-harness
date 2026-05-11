# Milestone 10 — Quantization Comparison Suite

## Goal

Measure quality and performance impact of quantization by running matched task sets across quantized model variants and producing quality delta reports with speed/quality frontier plots.

## Phase

Phase C — Deep model/backend comparison (Milestone 2 of 4 in phase)

## Dependencies

- Milestone 1: CLI runner, SQLite storage, Markdown report
- Milestone 3: Timing and token metrics
- Milestone 4: Basic scorers
- Milestone 5: Coding task runner
- Milestone 9: Long-context benchmark suite (for long-context quantization comparison)

---

### Data Access (STORAGE_PLAN)

Per STORAGE_PLAN.md: quantization comparison runs require pinned, reproducible datasets. Each quantized variant runs against the same local dataset copies:

```
/mnt/datasets-big/evals/quant_comparison_v1/
  tasks.jsonl
  manifest.json
```

The manifest pins:
- Dataset name, source, revision
- Checksum of the local file
- Task count and categories

This ensures quantization quality deltas are measured against identical inputs, not varying remote data.

---

## Subtasks

### 10.1 Add quantization metadata to model config

**File:** `configs/models.yaml` (extend)

**New fields for each model entry:**

```yaml
models:
  qwen35-35b-fp8:
    provider: openai_compatible
    base_url: "http://localhost:4000/v1"
    model: "qwen35-35b-fp8"
    backend: vllm
    quantization:
      scheme: FP8
      bits: 8
      type: model_weights
      framework: vllm
    base_model_family: "Qwen3.5-35B"
    parameters_total: 35000000000
    parameters_active: 35000000000
    architecture: dense
    gpu_memory_required_gb: 70
    notes: "Qwen3.5-35B full precision FP8"

  qwen35-35b-nvfp4:
    provider: openai_compatible
    base_url: "http://localhost:4000/v1"
    model: "qwen35-35b-nvfp4"
    backend: vllm
    quantization:
      scheme: NVFP4
      bits: 4
      type: model_weights
      framework: vllm
    base_model_family: "Qwen3.5-35B"
    parameters_total: 35000000000
    parameters_active: 35000000000
    architecture: dense
    gpu_memory_required_gb: 40
    notes: "Qwen3.5-35B NVIDIA FP4 quantized"

  qwen35-35b-gptq-int4:
    provider: openai_compatible
    base_url: "http://localhost:4000/v1"
    model: "qwen35-35b-gptq-int4"
    backend: vllm
    quantization:
      scheme: GPTQ
      bits: 4
      type: model_weights
      framework: autoawq
    base_model_family: "Qwen3.5-35B"
    parameters_total: 35000000000
    parameters_active: 35000000000
    architecture: dense
    gpu_memory_required_gb: 35
    notes: "Qwen3.5-35B GPTQ Int4"
```

**Actions:**
- [ ] Add quantization block to each model config entry
- [ ] Add `base_model_family` field for grouping variants
- [ ] Document all supported quantization schemes

### 10.2 Extend SQLite schema for quantization tracking

**File:** `src/bench_harness/storage/sqlite.py`

**New table:** `model_quantization_metadata`

```sql
CREATE TABLE model_quantization_metadata (
    run_id TEXT,
    model_alias TEXT NOT NULL,
    base_model_family TEXT NOT NULL,
    quantization_scheme TEXT NOT NULL,
    quantization_bits INTEGER,
    architecture TEXT,          -- dense | moe
    parameters_total INTEGER,
    parameters_active INTEGER,
    gpu_memory_required_gb REAL,
    created_at TEXT NOT NULL
);
```

**New columns in `runs` table (via ALTER TABLE or new table):**
```sql
ALTER TABLE runs ADD COLUMN quantization_scheme TEXT;
ALTER TABLE runs ADD COLUMN base_model_family TEXT;
```

**New methods in `SQLiteStore`:**
```python
def save_quantization_metadata(self, run_id: str, model_config: dict)
def get_quantization_comparisons(self, base_family: str) -> list[dict]
```

**Actions:**
- [ ] Add new table and columns to `SQLiteStore.init()`
- [ ] Implement save/query methods
- [ ] Add index on `(base_model_family, quantization_scheme)`

### 10.3 Define quantization-sensitive task suite

**File:** `configs/suites.yaml` (extend)

```yaml
  quantization_comparison:
    description: "Measure quality impact of quantization across model variants"
    task_dir: "tasks/synthetic/quantization_sensitive"
    max_concurrency: 2
    default_runs: 3
    requires_matched_tasks: true
    task_categories:
      - math_reasoning
      - coding
      - instruction_following
      - long_context_retrieval
      - json_formatting
      - hallucination_factuality
```

**Directory:** `tasks/synthetic/quantization_sensitive/`

**Task distribution (at least 20 tasks total, ~3-4 per category):**

| Category | Task Count | Description |
|---|---|---|
| `math_reasoning` | 4 | Multi-step arithmetic, algebra, logic puzzles |
| `coding` | 4 | Function generation, bug fix, algorithm implementation |
| `instruction_following` | 4 | Complex formatting, constraint satisfaction, IFEval-style |
| `long_context_retrieval` | 3 | Needle-in-haystack at 32k and 64k tokens |
| `json_formatting` | 3 | Nested JSON, strict schema compliance |
| `hallucination_factuality` | 3 | Questions with known facts, detect fabrication |

**Example task file:** `tasks/synthetic/quantization_sensitive/math_multistep_001.yaml`
```yaml
id: quant.math_multistep_001
family: quantization_comparison
category: math_reasoning
prompt: |
  A store sells apples at $3 each and oranges at $2 each.
  If you buy 15 apples and 10 oranges, what is the total cost?
  Then, if the store gives a 12% discount on purchases over $50,
  what is your final cost? Show your work.
expected:
  type: exact
  answer: "53.4"
  tolerance: 0.1
scoring:
  primary: exact_match
  secondary:
    - regex            # checks for correct intermediate steps
```

**Actions:**
- [ ] Create at least 20 task YAML files across 6 categories
- [ ] Ensure tasks are quantization-sensitive (precision matters for math, formatting strictness)
- [ ] Tag each task with its category

### 10.4 Implement matched-task runner for quantization comparison

**File:** `src/bench_harness/runners/quantization_runner.py` (new)

**Class:** `QuantizationRunner`

**Methods:**
```python
class QuantizationRunner:
    """Runs the same tasks across quantized variants with matched settings."""

    def __init__(self, store: SQLiteStore, config: dict):
        self.store = store
        self.config = config

    def run_matched_suite(
        self,
        base_family: str,
        tasks: list[dict],
        temperature: float = 0,
        max_tokens: int = 4096,
        runs: int = 3,
    ) -> list[dict]:
        """
        Run same task set across all quantized variants of a model family.
        Guarantees: same sampling, same prompts, same scoring, same run count.
        """

    def _get_variants(self, base_family: str) -> list[dict]:
        """Find all model configs sharing the same base_model_family."""

    def _ensure_matched_settings(self, variants: list[dict]) -> dict:
        """
        Verify all variants use identical:
        - temperature
        - max_tokens
        - prompt templates
        - run count
        Raises error on mismatch.
        """
```

**CLI integration in `src/bench_harness/cli.py`:**
```
--quant-compare       TEXT   Base model family for quantization comparison
                           (e.g., "Qwen3.5-35B")
--quant-baseline      TEXT   Reference quantization scheme for delta calculation
                           (e.g., "FP8")
```

**Actions:**
- [ ] Implement `QuantizationRunner` class
- [ ] Implement matched-settings verification
- [ ] Wire into CLI with new arguments
- [ ] Tag each run with quantization metadata before saving

### 10.5 Implement quality delta scorer

**File:** `src/bench_harness/scorers/quantization_delta.py` (new)

**Class:** `QuantizationDeltaScorer`

**Methods:**
```python
class QuantizationDeltaScorer:
    """Computes quality deltas between quantization variants."""

    def __init__(self, store: SQLiteStore):
        self.store = store

    def compute_deltas(
        self,
        base_family: str,
        baseline_scheme: str,
        task_categories: list[str] | None = None,
    ) -> QuantizationDeltaReport:
        """
        For each task category, compute:
        - delta in primary_score (quantized - baseline)
        - delta in pass_rate
        - delta in format_failure_rate
        - delta in tokens_per_second
        - delta in TTFT
        """

    def compute_per_task_deltas(self, base_family: str, baseline_scheme: str) -> list[dict]:
        """Per-task quality delta for fine-grained analysis."""

    def identify_degraded_tasks(self, base_family: str, threshold: float = -0.1) -> list[dict]:
        """Return tasks where quantization caused quality drop > threshold."""
```

**QuantizationDeltaReport dataclass:**
```python
@dataclass
class QuantizationDeltaReport:
    base_family: str
    baseline_scheme: str
    comparisons: list[QuantizationComparison]
    category_deltas: dict[str, CategoryDelta]
    degraded_tasks: list[dict]
    unacceptable_regimes: list[str]  # quant schemes that fail for coding-agent use
```

**Actions:**
- [ ] Implement delta computation across all metrics
- [ ] Implement per-category aggregation
- [ ] Implement degradation threshold detection

### 10.6 Implement speed/quality frontier plot

**File:** `src/bench_harness/replots/plots.py` (extend or new)

**Class/Function:** `SpeedQualityFrontierPlot`

```python
def generate_speed_quality_frontier(
    db_path: str,
    base_family: str,
    out_path: str,
) -> str:
    """
    Generate scatter plot with:
    - x-axis: tokens per second (or TTFT)
    - y-axis: primary_score (or composite score)
    - Each point: one quantization variant
    - Points labeled with quant scheme and bits
    - Frontier curve connecting Pareto-optimal points

    Returns path to generated PNG.
    """
```

**Additional charts:**

```python
def generate_quality_delta_bar_chart(
    db_path: str,
    base_family: str,
    baseline_scheme: str,
    out_path: str,
) -> str:
    """
    Grouped bar chart showing quality delta by task category.
    One group per category, one bar per quantization variant.
    """

def generate_quantization_heatmap(
    db_path: str,
    out_path: str,
) -> str:
    """
    Heatmap: rows = models, columns = task categories.
    Cell value = primary_score, color-coded.
    Enables spotting which quant schemes hurt which categories.
    """
```

**Actions:**
- [ ] Implement speed/quality frontier scatter plot with Pareto frontier
- [ ] Implement quality delta grouped bar chart
- [ ] Implement quantization heatmap
- [ ] Use matplotlib/seaborn for chart generation
- [ ] Save charts as PNG to output directory

### 10.7 Generate quantization comparison report

**File:** `src/bench_harness/reports/quantization_report.py` (new)

**Function:** `generate_quantization_report(db_path: str, out_dir: str, base_family: str)`

**Report sections:**

1. **Executive Summary**
   - Which quantization schemes are acceptable for coding-agent use
   - Which schemes show unacceptable degradation

2. **Quality Deltas by Task Category** — table and bar chart
   - Rows: task categories (math, coding, instruction, long-context, json, factuality)
   - Columns: quantization schemes
   - Cells: score delta from baseline

3. **Speed/Quality Frontier** — scatter plot
   - Pareto-optimal quantization configurations highlighted

4. **Per-Task Degradation** — sorted table
   - Tasks ranked by largest quality delta
   - Flags tasks where quantization caused regression > threshold

5. **Hallucination and Format Impact**
   - Does quantization increase hallucination rate?
   - Does quantization increase format failure rate?

6. **Recommendations by Use Case**
   - Best quantization for coding tasks
   - Best quantization for long-context tasks
   - Best quantization for fast prototyping

**Markdown table format:**

```markdown
## Quality Deltas by Task Category

Baseline: FP8

| Task Category | NVFP4 | GPTQ-Int4 | Notes |
|---|---|---|---|
| Math Reasoning | -0.03 | -0.12 | GPTQ-Int4 degrades arithmetic |
| Coding | -0.02 | -0.05 | Both acceptable for coding |
| Instruction Following | -0.01 | -0.08 | GPTQ-Int4 format failures increase |
| Long Context | -0.04 | -0.15 | NVFP4 moderate drop, GPTQ severe |
| JSON Formatting | -0.01 | -0.10 | GPTQ-Int4 schema violations |
| Factuality | +0.00 | -0.02 | Negligible impact |
```

**Actions:**
- [ ] Implement report generation with all sections
- [ ] Auto-generate charts inline or as referenced files
- [ ] Integrate with CLI: auto-generate when `--suite quantization_comparison` completes

### 10.8 Add script for quantization sweep

**File:** `scripts/run_quantization_compare.sh`

**Content:**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness run \
  --suite quantization_comparison \
  --quant-compare "Qwen3.5-35B" \
  --quant-baseline "FP8" \
  --runs 3 \
  --out "runs/$(date +%Y-%m-%d)-quant-compare"
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 10.9 Add tests

**File:** `tests/test_quantization.py`

**Tests:**
- `test_quantization_config_fields` — model config has required quantization fields
- `test_matched_settings_verification` — runner rejects mismatched settings
- `test_quantization_delta_computation` — delta computed correctly for synthetic data
- `test_degraded_task_detection` — tasks below threshold are identified
- `test_sqlite_quantization_schema` — table creation and round-trip
- `test_variant_discovery` — all variants of a base family are found
- `test_speed_quality_frontier_chart` — chart generated without error
- `test_quantization_report` — report generated with expected sections

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for temp DB with synthetic quantization data
- [ ] Mock chart output path for chart tests

---

## Acceptance Criteria Checklist

- [ ] Report shows whether quantization hurts coding, math, long-context, or formatting
- [ ] Report identifies best quantized model by task family
- [ ] Quantization impact is separated from prompt/backend differences
- [ ] Quality deltas are reported per task category
- [ ] Speed/quality frontier chart is generated
- [ ] Unacceptable quantization regimes for coding-agent use are identified
- [ ] Same tasks run with same settings across all quantized variants
- [ ] `pytest tests/test_quantization.py` passes

## Estimated Effort

3–4 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `configs/models.yaml` | Extend |
| `configs/suites.yaml` | Extend |
| `src/bench_harness/runners/quantization_runner.py` | To create |
| `src/bench_harness/scorers/quantization_delta.py` | To create |
| `src/bench_harness/reports/quantization_report.py` | To create |
| `src/bench_harness/reports/plots.py` | Extend |
| `src/bench_harness/storage/sqlite.py` | Extend |
| `src/bench_harness/cli.py` | Extend |
| `tasks/synthetic/quantization_sensitive/*.yaml` (×20) | To create |
| `scripts/run_quantization_compare.sh` | To create |
| `tests/test_quantization.py` | To create |
