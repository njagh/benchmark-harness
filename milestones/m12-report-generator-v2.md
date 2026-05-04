# Milestone 12 — Report Generator v2

## Goal

Make benchmark results easy to interpret with model comparison tables, suite-level summaries, best-model-by-task-family, speed/quality frontier, failure clustering, and regression detection versus prior runs. Outputs Markdown and HTML.

## Phase

Phase C / Phase D bridge — Makes all prior milestones actionable

## Dependencies

- Milestone 1: SQLite storage, Markdown report v1
- Milestone 3: Timing and token metrics
- Milestone 4: Basic scorers
- Milestone 6: Local coding-agent suite
- Milestone 7: LLM judge integration
- Milestone 9: Long-context benchmark suite
- Milestone 10: Quantization comparison suite
- Milestone 11: Agent safety scoring

---

### Leveraged Libraries

- **DuckDB** (`duckdb`): Query SQLite benchmark database with SQL for report data aggregation. DuckDB can attach SQLite databases directly and run complex aggregation queries without ETL. Use for all report data queries.
- **matplotlib + seaborn** (`matplotlib`, `seaborn`): Chart generation for speed/quality frontier, failure clustering heatmaps, etc. Save as PNG for Markdown embedding and SVG for HTML.
- **great-tables** (`great-tables`): Generate polished Markdown and HTML tables from pandas DataFrames. Superior to pandas default table rendering for reports.

---

## Subtasks

### 12.1 Redesign report data layer

**File:** `src/bench_harness/reports/report_data.py` (new)

**Class:** `ReportDataLayer`

**Methods:**
```python
class ReportDataLayer:
    """Queries SQLite and assembles structured data for all report sections."""

    def __init__(self, db_path: str):
        self.con = duckdb.connect()
        self.con.execute(f"ATTACH '{db_path}' AS benchmark (TYPE SQLITE)")

    def get_run_summary(self) -> RunSummary:
        """Aggregate summary: suites run, models compared, total tasks."""

    def get_model_comparison(self, suite_id: str | None = None) -> list[ModelComparisonRow]:
        """Per-model aggregates: tasks run, passed, failed, avg scores, timings."""

    def get_task_family_rankings(self) -> dict[str, list[ModelComparisonRow]]:
        """Best model per task family, sorted by primary score."""

    def get_speed_quality_data(self) -> list[SpeedQualityPoint]:
        """Data for speed/quality frontier: (model, score, tokens_per_second, ttft)."""

    def get_failure_clustering(self) -> list[FailureCluster]:
        """Group failures by task family and failure type."""

    def get_regression_data(self, baseline_db_path: str) -> list[RegressionDelta]:
        """Compare current run against baseline database."""

    def get_quantization_summary(self) -> dict[str, QuantDelta]:
        """Summary of quantization impact (from M10 data)."""

    def get_long_context_summary(self) -> dict[str, ContextDegradation]:
        """Summary of context degradation (from M9 data)."""

    def get_safety_summary(self) -> list[SafetyRanking]:
        """Safety rankings (from M11 data)."""
```

**Data classes:**
```python
@dataclass
class RunSummary:
    date: str
    host: str
    suites_run: list[str]
    models_compared: list[str]
    total_tasks: int
    total_runs: int
    environment_snapshot: dict

@dataclass
class ModelComparisonRow:
    model_alias: str
    backend: str
    quantization: str | None
    tasks_run: int
    tasks_passed: int
    pass_rate: float
    avg_primary_score: float
    avg_ttft_ms: float
    avg_decode_ms: float
    avg_tokens_per_second: float
    avg_total_tokens: float
    format_failure_count: int
    safety_score: float | None

@dataclass
class FailureCluster:
    family: str
    failure_type: str        # "low_score" | "format_failure" | "timeout" | "safety_violation"
    count: int
    affected_models: list[str]
    example_task_ids: list[str]

@dataclass
class RegressionDelta:
    model_alias: str
    task_id: str
    baseline_score: float
    current_score: float
    delta: float
    severity: str            # "regression" | "improvement" | "stable"
```

**Actions:**
- [ ] Implement all query methods with parameterized SQL
- [ ] Handle missing data gracefully (NULL scores, missing suites)
- [ ] Add caching layer to avoid repeated queries during report generation
- [ ] Add unit tests for query correctness with synthetic data

### 12.2 Add model comparison tables

**File:** `src/bench_harness/reports/markdown.py` (extend)

**New report section: Overall Model Comparison**

```markdown
## Model Comparison Summary

| Model | Backend | Quant | Tasks | Pass Rate | Avg Score | Avg TTFT | Tok/s | Safety |
|---|---|---|---|---|---|---|---|---|
| agent-code | vllm | FP8 | 120 | 87% | 0.82 | 245ms | 42.1 | 0.95 |
| qwen-dense | vllm | FP8 | 120 | 79% | 0.74 | 198ms | 51.3 | 0.82 |
| max-brain | vllm | GPTQ-Int4 | 120 | 91% | 0.88 | 892ms | 18.7 | 0.90 |
```

**New section: Best Model by Task Family**

```markdown
## Best Model by Task Family

| Task Family | Best Quality | Best Fast Model | Quality/Sec Winner | Notes |
|---|---|---|---|---|
| coding | max-brain | agent-code | agent-code | max-brain +6% but 3x slower |
| shell_debugging | agent-code | qwen-dense | qwen-dense | qwen-dense adequate for simple fixes |
| long_context | max-brain | agent-code | agent-code | quality drops at 128k for all |
| instruction_following | max-brain | agent-code | agent-code | |
| math_reasoning | max-brain | agent-code | agent-code | |
```

**Actions:**
- [ ] Implement Markdown table generation for model comparison
- [ ] Compute "best fast model" as top-scoring model under a latency threshold
- [ ] Compute quality/sec as avg_score * tokens_per_second / max_tokens_per_second
- [ ] Generate per-family best-model table

### 12.3 Add suite-level summaries

**File:** `src/bench_harness/reports/markdown.py` (extend)

**Per-suite summary section:**

```markdown
## Suite: coding_smoke

- Tasks: 30
- Models: 3
- Best: max-brain (pass rate 93%)
- Fastest: qwen-dense (avg TTFT 189ms)
- Best quality/sec: agent-code

### Per-Task Results

| Task | agent-code | qwen-dense | max-brain |
|---|---|---|---|
| smoke.python_001 | PASS (0.95) | PASS (0.88) | PASS (0.97) |
| coding.patch_002 | PASS (0.90) | FAIL (0.42) | PASS (0.95) |
| ... | ... | ... | ... |
```

**Actions:**
- [ ] Generate per-suite sections with task-by-model score matrices
- [ ] Color-code PASS/FAIL in HTML output
- [ ] Include timing summary per suite

### 12.4 Add speed/quality frontier chart

**File:** `src/bench_harness/reports/plots.py` (extend or new)

```python
def generate_speed_quality_frontier(
    data: list[SpeedQualityPoint],
    out_path: str,
) -> str:
    """
    Scatter plot with Pareto frontier.
    - x-axis: tokens per second (log scale)
    - y-axis: primary composite score
    - Each point: one model, labeled
    - Frontier curve: connects Pareto-optimal points
    - Regions labeled: "quality zone", "speed zone", "balanced"
    """
```

**Additional frontier variant: quality vs latency**
```python
def generate_quality_latency_frontier(
    data: list[QualityLatencyPoint],
    out_path: str,
) -> str:
    """
    Scatter plot:
    - x-axis: TTFT (milliseconds, log scale)
    - y-axis: primary score
    - Shows tradeoff between speed and quality
    """
```

**Actions:**
- [ ] Implement speed/quality frontier with Pareto-optimal curve
- [ ] Implement quality/latency frontier
- [ ] Add region labels and annotations
- [ ] Support both PNG output (for Markdown) and SVG (for HTML)

### 12.5 Add failure clustering analysis

**File:** `src/bench_harness/reports/failure_analysis.py` (new)

**Class:** `FailureAnalyzer`

**Methods:**
```python
class FailureAnalyzer:
    """Groups and analyzes benchmark failures."""

    def cluster_failures(self, runs: list[dict]) -> list[FailureCluster]:
        """
        Group failures by:
        1. Task family (which categories are models struggling with?)
        2. Failure type (format, low score, timeout, safety)
        3. Model-specific (which model fails most?)
        """

    def get_most_discriminating_tasks(self, runs: list[dict], top_n: int = 10) -> list[dict]:
        """
        Tasks where models disagree most.
        High-variance tasks are the most discriminating.
        Returns tasks sorted by score standard deviation across models.
        """

    def get_consensus_failures(self, runs: list[dict]) -> list[dict]:
        """
        Tasks that ALL models fail.
        May indicate bad tasks, impossible prompts, or systemic issues.
        """
```

**Report section output:**

```markdown
## Failure Analysis

### Most Discriminating Tasks
Tasks where models show the most variance (best for evaluating model differences):

| Task | Score Range | Best Model | Worst Model | Std Dev |
|---|---|---|---|---|
| coding.patch_002 | 0.42–0.95 | max-brain | qwen-dense | 0.28 |
| ... | ... | ... | ... | ... |

### Consensus Failures
Tasks all models failed on (possible task issues):

| Task | Family | Avg Score | Most Common Failure |
|---|---|---|---|
| ... | ... | ... | ... |

### Failure Clusters by Family

| Family | Total Fails | Most Affected Model | Primary Failure Type |
|---|---|---|---|
| ... | ... | ... | ... |
```

**Actions:**
- [ ] Implement failure clustering by family and type
- [ ] Implement discriminating task detection (variance-based)
- [ ] Implement consensus failure detection
- [ ] Generate report section

### 12.6 Add regression detection versus prior run

**File:** `src/bench_harness/reports/regression_detector.py` (new)

**Class:** `RegressionDetector`

**Methods:**
```python
class RegressionDetector:
    """Detects quality and performance regressions vs baseline."""

    def __init__(self, current_db: str, baseline_db: str):
        self.current = ReportDataLayer(current_db)
        self.baseline = ReportDataLayer(baseline_db)

    def detect_regressions(self) -> list[RegressionDelta]:
        """
        Compare current runs against baseline:
        - Per-model score delta
        - Per-suite pass rate delta
        - Per-task score delta
        - Performance metric delta (TTFT, tokens/sec)
        """

    def get_summary(self) -> RegressionSummary:
        """
        Overall assessment:
        - Any quality regressions? (score delta < -0.05)
        - Any performance regressions? (TTFT delta > 20%)
        - Any improvements?
        - Net assessment: PASS / WARN / FAIL
        """

    def get_changed_tasks(self) -> list[dict]:
        """Tasks where score changed significantly."""
```

**RegressionSummary dataclass:**
```python
@dataclass
class RegressionSummary:
    overall_status: str       # "PASS" | "WARN" | "FAIL"
    quality_regressions: int
    quality_improvements: int
    performance_regressions: int
    performance_improvements: int
    worst_regression: RegressionDelta | None
    details: list[RegressionDelta]
```

**Report section:**
```markdown
## Regression Analysis (vs baseline: runs/2026-05-01-agent-code-baseline)

**Overall Status: WARN**

- Quality regressions: 2
- Quality improvements: 5
- Performance regressions: 1
- Worst regression: `coding.patch_002` agent-code: -0.15 (0.90 → 0.75)

### Quality Regressions

| Task | Model | Baseline | Current | Delta |
|---|---|---|---|---|
| coding.patch_002 | agent-code | 0.90 | 0.75 | -0.15 |
| ... | ... | ... | ... | ... |
```

**Actions:**
- [ ] Implement cross-database comparison queries
- [ ] Implement regression severity classification
- [ ] Generate report section with PASS/WARN/FAIL status
- [ ] Wire into CLI `compare` command (from M15)

### 12.7 Implement HTML report

**File:** `src/bench_harness/reports/html.py` (new)

**Function:** `generate_html_report(data_layer: ReportDataLayer, out_path: str)`

**HTML report features:**
- Embedded CSS for clean styling
- Collapsible sections per suite
- Color-coded PASS/FAIL indicators
- Inline SVG/PNG charts
- Model comparison tables with sorting
- Failure examples with expandable raw output
- Navigation sidebar with jump links

**Template structure:**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Benchmark Report: {suite} — {date}</title>
  <style>/* embedded CSS */</style>
</head>
<body>
  <nav>/* sidebar with section links */</nav>
  <main>
    <section id="summary">/* model comparison table */</section>
    <section id="rankings">/* best by family */</section>
    <section id="frontier">/* speed/quality chart */</section>
    <section id="suites">/* per-suite breakdown */</section>
    <section id="failures">/* failure analysis */</section>
    <section id="regressions">/* regression detection */</section>
    <section id="quantization">/* quantization summary */</section>
    <section id="long-context">/* context degradation */</section>
    <section id="safety">/* safety rankings */</section>
    <section id="recommendations">/* generated recommendations */</section>
  </main>
</body>
</html>
```

**Actions:**
- [ ] Implement HTML template with embedded CSS
- [ ] Support chart embedding as base64 PNG or inline SVG
- [ ] Add collapsible sections for per-suite detail
- [ ] Add color coding: green for pass, red for fail, yellow for warning

### 12.8 Add recommendation engine

**File:** `src/bench_harness/reports/recommendations.py` (new)

**Class:** `RecommendationEngine`

**Methods:**
```python
class RecommendationEngine:
    """Generates model selection recommendations from benchmark data."""

    def generate_recommendations(
        self,
        data_layer: ReportDataLayer,
    ) -> str:
        """
        Generate natural-language recommendations:
        - Default coding assistant: best quality/sec model
        - Hard debugging: best quality model (ignoring speed)
        - Long-context tasks: model with best context breakpoint
        - Quick snippets: fastest adequate model
        - Unsafe models: flag models with low safety scores
        - Quantized alternatives: recommend acceptable quant variants
        """

    def _recommend_coding_default(self, models: list[ModelComparisonRow]) -> str:
        """Best quality/sec model for general coding."""

    def _recommend_hard_debugging(self, models: list[ModelComparisonRow]) -> str:
        """Best quality model for complex debugging."""

    def _recommend_quick_option(self, models: list[ModelComparisonRow]) -> str:
        """Fastest model above quality threshold."""
```

**Report section output:**
```markdown
## Recommendations

**Default coding assistant:** agent-code
- Best quality/sec balance
- Strong safety record (0.95)
- Handles all coding task families well

**Hard debugging:** max-brain
- Highest overall score (0.88)
- Best at root-cause analysis in long error logs
- Tradeoff: 3.6x higher TTFT than agent-code

**Quick shell snippets:** qwen-dense
- Fastest adequate option (TTFT 189ms)
- Acceptable for low-risk edits
- Not recommended for complex debugging

**Unsafe or unreliable:** (none flagged)

**Quantization note:** GPTQ-Int4 shows unacceptable degradation
in math reasoning (-0.12) and long context (-0.15). NVFP4 is acceptable.
```

**Actions:**
- [ ] Implement recommendation logic with configurable thresholds
- [ ] Generate natural-language recommendations
- [ ] Include safety and quantization caveats

### 12.9 Extend CLI for report generation

**File:** `src/bench_harness/cli.py` (extend)

**New CLI command:** `report`

```
bench-harness report
  --db            TEXT   Path to SQLite database      (required)
  --baseline-db   TEXT   Path to baseline database     (optional)
  --format        TEXT   Output format: md | html | both (default: both)
  --out           PATH   Output directory              (default: same as run dir)
  --suites        TEXT   Suite filter, comma-separated (default: all)
```

**Actions:**
- [ ] Add `report` command to typer CLI
- [ ] Wire report data layer → all report generators → output
- [ ] Support standalone report generation from any completed run DB

### 12.10 Add tests

**File:** `tests/test_reports.py`

**Tests:**
- `test_report_data_layer_summary` — correct aggregation from synthetic data
- `test_model_comparison_table` — table rows match expected values
- `test_task_family_rankings` — best model per family computed correctly
- `test_failure_clustering` — failures grouped correctly
- `test_regression_detection` — regressions identified correctly
- `test_regression_no_change` — no regressions when data is identical
- `test_html_report_generation` — HTML file created with expected sections
- `test_recommendations_generated` — recommendation text contains model names
- `test_speed_quality_chart` — chart PNG generated without error

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for synthetic SQLite databases

---

## Acceptance Criteria Checklist

- [ ] Report answers: which model is best overall for local coding?
- [ ] Report answers: which model is best fast option?
- [ ] Report answers: which model is unsafe or unreliable?
- [ ] Report answers: which tasks are most discriminating?
- [ ] Report answers: which prompt style works best?
- [ ] Report answers: which quantization changed quality?
- [ ] Report answers: which model degrades under long context?
- [ ] Model comparison tables are generated
- [ ] Speed/quality frontier chart is generated
- [ ] Failure clustering analysis is included
- [ ] Regression detection versus prior run is included
- [ ] Both Markdown and HTML output are generated
- [ ] Recommendations section provides actionable guidance
- [ ] `pytest tests/test_reports.py` passes

## Estimated Effort

4–5 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/reports/report_data.py` | To create |
| `src/bench_harness/reports/markdown.py` | Extend |
| `src/bench_harness/reports/html.py` | To create |
| `src/bench_harness/reports/plots.py` | Extend |
| `src/bench_harness/reports/failure_analysis.py` | To create |
| `src/bench_harness/reports/regression_detector.py` | To create |
| `src/bench_harness/reports/recommendations.py` | To create |
| `src/bench_harness/cli.py` | Extend |
| `tests/test_reports.py` | To create |
