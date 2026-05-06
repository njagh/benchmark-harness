# Milestone 12 — Report Generator v2

## Goal

Make benchmark results easy to interpret with a modular, comprehensive report that answers practical questions about model quality, speed, tradeoffs, and regressions.

## Phase

Phase C — Deep comparison (Milestone 3 of 3 in phase)

## Dependencies

- Milestone 7 (LLM judge integration — judge scores for comparison)
- Milestone 8 (prompt style comparison — style analysis integration)
- Milestone 9 (long-context benchmark suite — context analysis integration)
- Milestone 10 (quantization comparison suite — quantization analysis integration)

---

### Leveraged Libraries

- **duckdb** (in pyproject.toml optional `[long-context]`): SQL queries on SQLite for report data
- **matplotlib** / **seaborn** (in pyproject.toml optional `[long-context]`): Chart generation for reports
- **great-tables** (in pyproject.toml optional `[long-context]`): Polished report tables
- **typer** (already in pyproject.toml): CLI for --report-v2, --sections, --prior-runs flags
- **rich** (already in pyproject.toml): Terminal output

---

## Subtasks

### 12.1 Create reports/helpers.py

**File:** `src/bench_harness/reports/helpers.py` (new)

**Functions:**
```python
def group_runs(runs, group_by) -> dict  # group by model, family, quantization, etc.
def mean_std(values) -> tuple  # mean and standard deviation
def variance(values) -> float  # variance for discriminating tasks
def identify_pareto_frontier(items, x_key, y_key) -> list  # speed/quality Pareto frontier
def detect_clustering(items, key) -> dict  # cluster failures by pattern
def detect_regression(actual, expected, tolerance=0.05) -> dict  # regression vs prior run
def top_n(items, key, n=5, reverse=True) -> list  # top/bottom N by field
def margin(a, b) -> float  # margin between two scores
```

**Actions:**
- [x] Implement group_runs() for model, family, quantization, context bucket grouping
- [x] Implement mean_std() for score and timing aggregation
- [x] Implement variance() for discriminating task identification
- [x] Implement identify_pareto_frontier() for speed/quality frontier
- [x] Implement detect_clustering() for failure pattern grouping
- [x] Implement detect_regression() for prior run comparison
- [x] Implement top_n() for ranking helpers
- [x] Implement margin() for best-model margin analysis

### 12.2 Create reports/v2.py

**File:** `src/bench_harness/reports/v2.py` (new)

**Class:**
```python
class ReportV2:
    def __init__(self, runs, sections=None, prior_runs=None)
    def generate() -> str  # Full report generation
```

**Sections (each as a method, independently enableable/disabled):**
1. **Executive Summary** — top-line winners across all dimensions
2. **Model Comparison Cross-Ranking** — ranked table with quantization/context metadata
3. **Best Model by Task Family** — winner per family with margin analysis
4. **Speed/Quality Pareto Frontier** — which models sit on the efficient frontier
5. **Context Analysis** — quality vs context length, breakpoints (from M9)
6. **Quantization Comparison** — quality deltas, best quant per family (from M10)
7. **Prompt Style Analysis** — best prompt per family, verbosity comparison (from M8)
8. **Judge Analysis** — judge-scored task breakdowns, rubric dimensions (from M7)
9. **Failure Clustering** — top failure patterns with examples
10. **Regression Detection** — quality/speed changes vs prior run
11. **Discriminating Tasks** — tasks with highest score variance across models

**Behavior:**
- Sections are individually enabled/disabled via `sections` parameter
- Defaults to all sections when not specified
- Prior runs data is optional, used for regression detection
- Legacy markdown report is used as fallback when v2 is not requested

**Actions:**
- [x] Implement ReportV2 class with modular section architecture
- [x] Implement _append_executive_summary() with top-line winners
- [x] Implement _append_model_comparison() with cross-ranking table
- [x] Implement _append_best_family() with margin analysis
- [x] Implement _append_pareto_frontier() with efficient frontier identification
- [x] Implement _append_context_analysis() integrating M9 context reports
- [x] Implement _append_quantization_comparison() integrating M10 reports
- [x] Implement _append_prompt_style_analysis() integrating M8 reports
- [x] Implement _append_judge_analysis() integrating M7 judge data
- [x] Implement _append_failure_clustering() grouping failures by pattern
- [x] Implement _append_regression_detection() comparing against prior run
- [x] Implement _append_discriminating_tasks() ranking by score variance
- [x] Support sections parameter for selective section inclusion

### 12.3 Update markdown report to delegate to v2

**File:** `src/bench_harness/reports/markdown.py` (update)

**Changes:**
- Add `_generate_v2_report()` call when --report-v2 is set
- Fall back to legacy report generator when v2 is not requested
- Pass sections filter and prior runs data to v2

**Actions:**
- [x] Add report_v2 parameter to generate_report()
- [x] Add sections parameter to generate_report()
- [x] Add prior_runs parameter to generate_report()
- [x] Import and call ReportV2 when report_v2=True
- [x] Pass sections and prior_runs through to v2
- [x] Maintain legacy report generation as default fallback

### 12.4 Add CLI flags for v2 report

**File:** `src/bench_harness/cli.py` (update)

**New flags:**
```bash
--report-v2        BOOLEAN        Use modular report v2 (default: false)
--sections         TEXT           Comma-separated sections to include (default: all)
--prior-runs       TEXT           Path to prior run directory for regression comparison
```

**Actions:**
- [x] Add --report-v2 flag to run command
- [x] Add --sections flag (comma-separated section names)
- [x] Add --prior-runs flag (path to prior run for regression detection)
- [x] Wire flags into report generation flow
- [x] Validate section names against known sections

### 12.5 Add comprehensive tests

**File:** `tests/test_report_v2.py` (new)

**Test categories:**
- ReportV2 initialization and section filtering
- Executive summary with known data
- Model cross-ranking table generation
- Best family identification with margins
- Pareto frontier computation with known data
- Context analysis integration from sample runs
- Quantization comparison integration from sample runs
- Prompt style analysis integration from sample runs
- Judge analysis integration from sample data
- Failure clustering by error pattern
- Regression detection with known deltas
- Discriminating tasks identification by variance
- Legacy fallback when report_v2=False
- CLI flag parsing (--report-v2, --sections, --prior-runs)
- Markdown report integration with v2 sections

**Actions:**
- [x] Test ReportV2 with all sections enabled
- [x] Test ReportV2 with selective sections
- [x] Test executive summary identifies correct winners
- [x] Test model cross-ranking table with metadata
- [x] Test best family with margin analysis
- [x] Test Pareto frontier identification
- [x] Test context analysis section generation
- [x] Test quantization comparison section generation
- [x] Test prompt style analysis section generation
- [x] Test judge analysis section generation
- [x] Test failure clustering with sample failures
- [x] Test regression detection with known prior run
- [x] Test discriminating tasks by score variance
- [x] Test legacy fallback when report_v2=False
- [x] Test CLI flag parsing for all three flags
- [x] Test markdown report integration with v2 sections

---

## Acceptance Criteria Checklist

- [x] Report answers: best overall, fastest, unsafe, most discriminating tasks, best prompt style, quantization impact, long-context degradation
- [x] Modular architecture: sections can be enabled/disabled independently via --sections flag
- [x] Regression detection compares against previous benchmark run via --prior-runs
- [x] Executive Summary provides top-line winners across all dimensions
- [x] Model comparison cross-ranking includes quantization and context metadata
- [x] Best model by task family includes margin analysis
- [x] Speed/quality Pareto frontier identifies efficient models
- [x] Context analysis from M9 integrated into v2 report
- [x] Quantization comparison from M10 integrated into v2 report
- [x] Prompt style analysis from M8 integrated into v2 report
- [x] Judge analysis from M7 integrated into v2 report
- [x] Failure clustering groups failures by error pattern
- [x] Discriminating tasks identified by highest score variance
- [x] Markdown report delegates to v2 when --report-v2 is set
- [x] Legacy markdown report works as fallback when v2 is not requested
- [x] CLI --report-v2, --sections, and --prior-runs flags all work correctly
- [x] All tests pass (v2 report, helpers, integration, CLI flags)

---

## Known Sections

| Section ID | Description | Depends On |
|---|---|---|
| `executive_summary` | Top-line winners across all dimensions | none |
| `model_comparison` | Cross-ranking table with metadata | none |
| `best_family` | Best model per task family with margins | none |
| `pareto_frontier` | Speed/quality efficient frontier | none |
| `context_analysis` | Quality vs context length, breakpoints | M9 |
| `quantization_comparison` | Quality deltas, best quant per family | M10 |
| `prompt_style_analysis` | Best prompt per family, verbosity | M8 |
| `judge_analysis` | Judge-scored task breakdowns | M7 |
| `failure_clustering` | Failure patterns with examples | none |
| `regression_detection` | Changes vs prior run | prior_runs |
| `discriminating_tasks` | Highest score variance tasks | none |

---

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/reports/helpers.py` | Created (grouping, variance, clustering, regression detection) |
| `src/bench_harness/reports/v2.py` | Created (modular section-based ReportV2 class) |
| `src/bench_harness/reports/markdown.py` | Updated (delegates to v2, legacy fallback) |
| `src/bench_harness/cli.py` | Updated (--report-v2, --sections, --prior-runs flags) |
| `tests/test_report_v2.py` | Created (comprehensive v2 report tests) |
| `ROADMAP.md` | Updated (M12 marked done with detailed acceptance criteria) |
| `README.md` | Updated (status line and milestone table) |
