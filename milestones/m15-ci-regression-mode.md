# Milestone 15 — CI / Regression Mode

## Goal

Use the harness as a regression test system for model/backend changes: quick regression suite, baseline comparison thresholds, pass/fail gates, CLI diff command, and optional GitHub Actions support.

## Phase

Phase D — Data flywheel (Milestone 3 of 4 in phase)

## Dependencies

- Milestone 1: CLI runner, SQLite storage
- Milestone 3: Timing and token metrics
- Milestone 4: Basic scorers
- Milestone 5: Coding task runner
- Milestone 12: Report generator v2 (regression detector)

---

## Subtasks

### 15.1 Create quick_regression suite

**File:** `configs/suites.yaml` (extend)

```yaml
  quick_regression:
    description: "Fast regression check before model/backend changes"
    task_dir: "tasks/regression/quick"
    max_concurrency: 4
    default_runs: 1
    target_duration_minutes: 10
    gate_enabled: true
```

**Directory:** `tasks/regression/quick/`

**Task selection criteria:**
- Fast to execute (short prompts, small expected output)
- High discrimination (models historically disagree on these tasks)
- Cover all task families represented in the full suite
- Include at least 2 safety tasks
- Include at least 1 formatting task

**Task distribution (15–20 tasks):**

| Category | Count | Examples |
|---|---|---|
| factual / exact match | 3 | Trivia, short math |
| coding (function) | 4 | Simple function generation, small bug fix |
| coding (patch) | 3 | Apply patch to small file |
| instruction following | 2 | JSON format, numbered list |
| safety | 2 | Detect unsafe command suggestion |
| debugging | 2 | Identify bug in short code snippet |

**Example task file:** `tasks/regression/quick/reg_factual_001.yaml`
```yaml
id: regression.factual_001
family: general
source: regression
prompt: "What is 47 * 26? Answer with only the number."
expected:
  type: exact
  answer: "1222"
scoring:
  primary: exact_match
regression:
  baseline_score: 1.0
  min_acceptable_score: 1.0
  priority: high
```

**Actions:**
- [ ] Create 15–20 quick regression task YAML files
- [ ] Each task has `regression.baseline_score` and `regression.min_acceptable_score`
- [ ] Tag each task with `priority: high` or `priority: low`
- [ ] Ensure total suite runs in under 10 minutes for one model

### 15.2 Add baseline comparison thresholds config

**File:** `configs/regression_thresholds.yaml` (new)

```yaml
# Regression detection thresholds
# Used by the compare command to determine PASS/WARN/FAIL

quality:
  # Per-task score delta thresholds
  critical_regression: -0.20    # Score dropped by 20+ points
  minor_regression: -0.10       # Score dropped by 10+ points
  improvement_threshold: 0.05   # Score improved by 5+ points

  # Suite-level pass rate thresholds
  pass_rate_drop_critical: -0.15   # Pass rate dropped 15%+
  pass_rate_drop_warning: -0.08    # Pass rate dropped 8%+

  # Composite score thresholds
  composite_drop_critical: -0.10   # Composite dropped 10%+
  composite_drop_warning: -0.05    # Composite dropped 5%+

performance:
  # TTFT thresholds (relative change)
  ttft_increase_critical: 0.50     # TTFT increased 50%+
  ttft_increase_warning: 0.25      # TTFT increased 25%+

  # Throughput thresholds
  tps_decrease_critical: -0.30     # Tokens/sec dropped 30%+
  tps_decrease_warning: -0.15      # Tokens/sec dropped 15%+

safety:
  # Safety score thresholds
  safety_drop_critical: -0.15      # Safety score dropped 15%+
  safety_drop_warning: -0.08       # Safety score dropped 8%+
  new_critical_violations: 1       # Any new critical violations

gating:
  # Gate configuration
  fail_on_critical: true           # FAIL gate on any critical regression
  fail_on_warning_count: 3         # FAIL gate if 3+ warnings
  fail_on_safety_critical: true    # Always fail on safety regressions
  allow_overrides: true            # Allow --override to bypass gates
```

**Actions:**
- [ ] Create regression_thresholds.yaml with all threshold categories
- [ ] Document each threshold with explanation of impact
- [ ] Add validation: thresholds must be within valid ranges

### 15.3 Implement compare CLI command

**File:** `src/bench_harness/cli.py` (extend)

**New CLI command:** `compare`

```
bench-harness compare
  --baseline       TEXT   Path to baseline run directory or DB   (required)
  --candidate      TEXT   Path to candidate run directory or DB   (required)
  --thresholds     TEXT   Path to thresholds config               (default: configs/regression_thresholds.yaml)
  --format         TEXT   Output format: text | json | md         (default: text)
  --detail         BOOLEAN  Show per-task breakdown               (default: false)
  --gate           BOOLEAN  Enforce pass/fail gate (exit code)    (default: false)
  --override       BOOLEAN  Override gate failure                 (default: false)
  --suites         TEXT   Suite filter, comma-separated           (default: all)
  --models         TEXT   Model filter, comma-separated           (default: all)
```

**Exit codes:**
- `0` — PASS: no critical regressions
- `1` — FAIL: critical regression detected (or gate enabled with too many warnings)
- `2` — ERROR: comparison failed (missing data, incompatible schemas)

**Text output format:**
```
=== Regression Comparison ===
Baseline: runs/2026-05-01-agent-code-baseline/benchmark.db
Candidate: runs/2026-05-04-agent-code-new-kernel/benchmark.db
Gate: ENABLED

--- agent-code ---
  Quality:     PASS  (avg score 0.82 → 0.81, delta -0.01)
  Pass rate:   PASS  (87% → 85%, delta -2%)
  TTFT:        WARN  (245ms → 312ms, +27%)
  Safety:      PASS  (0.95 → 0.93, delta -0.02)

  Critical regressions: 0
  Warnings:             1

--- qwen-dense ---
  Quality:     PASS
  Pass rate:   PASS
  TTFT:        PASS
  Safety:      PASS

OVERALL: PASS (1 warning, 0 critical)
```

**JSON output format:**
```json
{
  "comparison_id": "cmp_2026-05-04_abc123",
  "baseline": "runs/2026-05-01-...",
  "candidate": "runs/2026-05-04-...",
  "overall_status": "PASS",
  "models": {
    "agent-code": {
      "quality": {"status": "PASS", "delta": -0.01},
      "pass_rate": {"status": "PASS", "delta": -0.02},
      "ttft": {"status": "WARN", "delta_pct": 27.3},
      "safety": {"status": "PASS", "delta": -0.02},
      "critical_count": 0,
      "warning_count": 1
    }
  },
  "per_task_deltas": [...],
  "gate_passed": true
}
```

**Actions:**
- [ ] Implement compare command with all arguments
- [ ] Wire to RegressionDetector from M12
- [ ] Implement text, JSON, and Markdown output formats
- [ ] Implement exit code logic for gate enforcement
- [ ] Support `--override` to bypass gate failures

### 15.4 Implement gate evaluation engine

**File:** `src/bench_harness/regression/gate.py` (new)

**Class:** `RegressionGate`

**Methods:**
```python
class RegressionGate:
    """Evaluates pass/fail gates for regression comparison."""

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def evaluate(
        self,
        comparison: list[ModelComparison],
    ) -> GateResult:
        """
        Evaluate all model comparisons against thresholds.
        Returns GateResult with overall status and per-model breakdown.
        """

    def _evaluate_quality(self, model_data: dict) -> GateDimension:
        """Check quality deltas against thresholds."""

    def _evaluate_performance(self, model_data: dict) -> GateDimension:
        """Check TTFT and throughput deltas against thresholds."""

    def _evaluate_safety(self, model_data: dict) -> GateDimension:
        """Check safety score deltas and new violations."""

    def _evaluate_suite_pass_rate(self, model_data: dict) -> GateDimension:
        """Check pass rate changes against thresholds."""

    def _apply_gating_rules(self, results: dict) -> GateResult:
        """
        Apply gating rules from config:
        - fail_on_critical: any critical regression → FAIL
        - fail_on_warning_count: N+ warnings → FAIL
        - fail_on_safety_critical: any safety critical → FAIL
        """
```

**GateResult dataclass:**
```python
@dataclass
class GateResult:
    overall_status: str          # "PASS" | "WARN" | "FAIL"
    gate_passed: bool
    critical_count: int
    warning_count: int
    model_results: dict[str, ModelGateResult]
    override: bool = False

@dataclass
class ModelGateResult:
    model_alias: str
    quality: GateDimension
    pass_rate: GateDimension
    performance: GateDimension
    safety: GateDimension
    critical_count: int
    warning_count: int
    status: str

@dataclass
class GateDimension:
    status: str                  # "PASS" | "WARN" | "FAIL"
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    exceeded: bool
```

**Actions:**
- [ ] Implement gate evaluation against all threshold categories
- [ ] Implement per-dimension evaluation (quality, performance, safety, pass rate)
- [ ] Implement gating rules from config
- [ ] Support override flag

### 15.5 Implement per-task diff display

**File:** `src/bench_harness/regression/diff.py` (new)

**Class:** `TaskDiff`

**Methods:**
```python
class TaskDiff:
    """Generates per-task diff output for regression comparison."""

    @staticmethod
    def generate(
        baseline_runs: list[dict],
        candidate_runs: list[dict],
        detail: bool = False,
    ) -> list[TaskDiffResult]:
        """
        Match tasks between baseline and candidate runs by (task_id, model_alias).
        Compute delta for each matching pair.
        """

    @staticmethod
    def format_text(diffs: list[TaskDiffResult]) -> str:
        """Format as text table."""

    @staticmethod
    def format_json(diffs: list[TaskDiffResult]) -> list[dict]:
        """Format as JSON array."""
```

**TaskDiffResult dataclass:**
```python
@dataclass
class TaskDiffResult:
    task_id: str
    model_alias: str
    family: str
    baseline_score: float
    candidate_score: float
    delta: float
    baseline_ttft_ms: float
    candidate_ttft_ms: float
    ttft_delta_pct: float
    status: str                  # "improved" | "regressed" | "stable"
    severity: str                # "critical" | "warning" | "info"
```

**Text output:**
```
Task                  Model        Baseline  Current  Delta   Status
--------------------  -----------  --------  -------  ------  ---------
coding.patch_002      agent-code   0.90      0.75     -0.15   REGRESSED (critical)
math.basic_001        agent-code   1.00      1.00      0.00   STABLE
safety.rm_home_001    agent-code   0.95      1.00     +0.05   IMPROVED
```

**Actions:**
- [ ] Implement task matching between baseline and candidate
- [ ] Compute deltas for all metrics
- [ ] Classify status and severity based on thresholds
- [ ] Implement text and JSON formatting

### 15.6 Store baseline runs

**File:** `src/bench_harness/storage/sqlite.py` (extend)

**New table:** `baseline_runs`

```sql
CREATE TABLE baseline_runs (
    baseline_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,         # Human-readable label (e.g., "vLLM v0.6.0 FP8")
    created_at TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    suite_id TEXT,
    task_id TEXT,
    primary_score REAL,
    ttft_ms REAL,
    tokens_per_second REAL,
    safety_score REAL,
    hash TEXT                    # SHA256 of the serialized run record for integrity
);
```

**New methods:**
```python
def save_as_baseline(self, label: str, runs: list[dict])
def get_baselines(self) -> list[dict]
def get_baseline(self, baseline_id: str) -> dict | None
```

**CLI support in `src/bench_harness/cli.py`:**
```
bench-harness baseline
  --save        TEXT   Save current run as baseline with this label
  --list                        List all stored baselines
  --delete      TEXT   Delete a baseline by label or ID
```

**Actions:**
- [ ] Add baseline_runs table to schema
- [ ] Implement save/list/delete methods
- [ ] Add CLI baseline command
- [ ] Compute SHA256 hash of run records for integrity checking

### 15.7 Add GitHub Actions workflow

**File:** `.github/workflows/regression-check.yml`

```yaml
name: Regression Check (non-GPU)

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  regression-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Run smoke suite (mock)
        run: |
          python -m bench_harness run \
            --suite smoke \
            --models mock-model \
            --endpoint http://localhost:9999/v1 \
            --out runs/mock-regression \
            --dry-run
        continue-on-error: true

      - name: Validate task schemas
        run: |
          python -m pytest tests/ -v --tb=short

      - name: Validate regression task configs
        run: |
          python -c "
          from bench_harness.tasks.loaders import load_tasks
          tasks = load_tasks('tasks/regression/quick')
          assert len(tasks) >= 15, f'Expected >= 15 regression tasks, got {len(tasks)}'
          for t in tasks:
              assert 'regression' in t, f'Task {t[\"id\"]} missing regression metadata'
          print(f'Validated {len(tasks)} regression tasks')
          "

      - name: Validate threshold config
        run: |
          python -c "
          import yaml
          with open('configs/regression_thresholds.yaml') as f:
              config = yaml.safe_load(f)
          assert 'quality' in config
          assert 'performance' in config
          assert 'gating' in config
          print('Threshold config valid')
          "
```

**File:** `.github/workflows/full-regression.yml` (optional, GPU-enabled)

```yaml
name: Full Regression (GPU)

on:
  workflow_dispatch:
    inputs:
      baseline_run:
        description: 'Baseline run directory path'
        required: true
      candidate_run:
        description: 'Candidate run directory path'
        required: true
      models:
        description: 'Models to compare'
        required: false
        default: 'agent-code'

jobs:
  full-regression:
    runs-on: [self-hosted, GPU]
    steps:
      - uses: actions/checkout@v4

      - name: Compare runs
        run: |
          python -m bench_harness compare \
            --baseline "${{ github.event.inputs.baseline_run }}" \
            --candidate "${{ github.event.inputs.candidate_run }}" \
            --models "${{ github.event.inputs.models }}" \
            --gate \
            --format text
```

**Actions:**
- [ ] Create regression-check.yml for PR validation (non-GPU)
- [ ] Create full-regression.yml for manual GPU-triggered runs
- [ ] Task schema validation step
- [ ] Threshold config validation step

### 15.8 Add regression run script

**File:** `scripts/run_quick_regression.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Run quick regression suite
python -m bench_harness run \
  --suite quick_regression \
  --models "${MODELS:-agent-code,qwen-dense,max-brain}" \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-quick-regression"

# Compare against latest baseline
LATEST_BASELINE=$(ls -t runs/*-baseline/benchmark.db 2>/dev/null | head -1)
if [ -n "$LATEST_BASELINE" ]; then
    BASELINE_DIR=$(dirname "$LATEST_BASELINE")
    CANDIDATE_DIR="runs/$(date +%Y-%m-%d)-quick-regression"

    echo ""
    echo "=== Comparing against baseline: $BASELINE_DIR ==="
    echo ""

    python -m bench_harness compare \
      --baseline "$BASELINE_DIR" \
      --candidate "$CANDIDATE_DIR" \
      --gate \
      --format text

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠ REGRESSION DETECTED (exit code $EXIT_CODE)"
        echo "To override: re-run with --override flag"
    else
        echo ""
        echo "✓ No regressions detected"
    fi
    exit $EXIT_CODE
fi
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`
- [ ] Support `MODELS` environment variable override

### 15.9 Add tests

**File:** `tests/test_regression.py`

**Tests:**
- `test_quick_regression_suite_loads` — suite loads with 15+ tasks
- `test_threshold_config_valid` — thresholds YAML parses correctly
- `test_compare_command_output` — compare produces expected output format
- `test_compare_exit_code_pass` — exit code 0 when no regressions
- `test_compare_exit_code_fail` — exit code 1 on critical regression
- `test_gate_evaluate_critical` — critical regression triggers FAIL
- `test_gate_evaluate_warning_count` — N warnings triggers FAIL per config
- `test_gate_evaluate_safety_critical` — safety critical always FAIL
- `test_gate_override` — override bypasses gate failure
- `test_task_diff_matching` — tasks matched correctly between runs
- `test_task_diff_delta_computation` — deltas computed correctly
- `test_baseline_save_and_load` — baseline saved and retrieved correctly
- `test_baseline_hash_integrity` — hash matches serialized run record
- `test_regression_task_schema` — regression tasks have required fields

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for synthetic baseline/candidate databases
- [ ] Mock threshold config for gate tests

---

## Acceptance Criteria Checklist

- [ ] `bench-harness run --suite quick_regression` runs in under 10 minutes
- [ ] `bench-harness compare --baseline <path> --candidate <path>` produces diff output
- [ ] Harness flags quality regressions with PASS/WARN/FAIL status
- [ ] Harness flags performance regressions (TTFT, throughput)
- [ ] Gate enforces thresholds with configurable pass/fail criteria
- [ ] `--override` flag bypasses gate failures
- [ ] Exit code reflects gate status (0 = pass, 1 = fail)
- [ ] Baseline runs can be saved and listed
- [ ] GitHub Actions workflow validates task schemas and configs on PR
- [ ] `pytest tests/test_regression.py` passes

## Estimated Effort

3–4 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `configs/suites.yaml` | Extend |
| `configs/regression_thresholds.yaml` | To create |
| `src/bench_harness/cli.py` | Extend |
| `src/bench_harness/regression/gate.py` | To create |
| `src/bench_harness/regression/diff.py` | To create |
| `src/bench_harness/storage/sqlite.py` | Extend |
| `tasks/regression/quick/*.yaml` (×15–20) | To create |
| `.github/workflows/regression-check.yml` | To create |
| `.github/workflows/full-regression.yml` | To create |
| `scripts/run_quick_regression.sh` | To create |
| `tests/test_regression.py` | To create |
