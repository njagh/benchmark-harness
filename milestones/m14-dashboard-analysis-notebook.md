# Milestone 14 — Dashboard / Analysis Notebook

## Goal

Provide interactive analysis for repeated experiments with charts for score vs latency, context-length degradation curves, quantization comparison plots, prompt-style comparison plots, and a failure examples browser.

## Phase

Phase D — Data flywheel (Milestone 2 of 4 in phase)

## Dependencies

- Milestone 1: SQLite storage
- Milestone 3: Timing and token metrics
- Milestone 9: Long-context benchmark suite (context degradation data)
- Milestone 10: Quantization comparison suite (quantization delta data)
- Milestone 11: Agent safety scoring (safety rankings data)
- Milestone 12: Report generator v2 (report data layer)

---

## Subtasks

### 14.1 Set up analysis notebook infrastructure

**File:** `notebooks/benchmark_analysis.ipynb`

**Dependencies (in pyproject.toml or requirements for notebook):**
- `jupyter` / `ipykernel` — notebook runtime
- `pandas` — data manipulation
- `duckdb` — SQL queries on SQLite + JSONL
- `matplotlib` — chart rendering
- `seaborn` — statistical visualizations
- `plotly` (optional) — interactive charts

Note: DuckDB is the primary analysis engine. It attaches to the SQLite benchmark database directly, allowing complex SQL queries without ETL. All notebook queries should use DuckDB, not pandas read_sqlite, for performance on larger run databases.

**Notebook structure (cells):**

```
1. Title and configuration
2. Imports and setup
3. Database connection (DuckDB on SQLite)
4. Helper functions
5. Section A: Overall Summary
6. Section B: Score vs Latency
7. Section C: Context-Length Degradation Curves
8. Section D: Quantization Comparison Plots
9. Section E: Prompt-Style Comparison Plots
10. Section F: Safety Rankings
11. Section G: Failure Examples Browser
12. Section H: Multi-Run Comparison
13. Section I: Export and Summary
```

**Configuration cell:**
```python
# Configuration
DB_PATH = "runs/2026-05-04-all-suites/benchmark.db"       # Change per experiment
BASELINE_DB_PATH = "runs/2026-05-01-baseline/benchmark.db" # Optional baseline
OUTPUT_DIR = "notebooks/outputs/"
PLOT_DPI = 150
```

**Actions:**
- [ ] Create `notebooks/` directory
- [ ] Create `notebooks/benchmark_analysis.ipynb` with all section headers
- [ ] Add `notebooks/requirements.txt` with dependencies
- [ ] Create `notebooks/outputs/.gitkeep`
- [ ] Add `notebooks/` to `.gitignore` for outputs, keep notebook committed

### 14.2 Database connection and helper functions

**Notebook cells (Section: Setup):**

```python
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Connect DuckDB to SQLite database
con = duckdb.connect()
con.execute(f"ATTACH '{DB_PATH}' AS benchmark (TYPE SQLITE)")
if BASELINE_DB_PATH and Path(BASELINE_DB_PATH).exists():
    con.execute(f"ATTACH '{BASELINE_DB_PATH}' AS baseline (TYPE SQLITE)")

# Helper: run query and return DataFrame
def query(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

# Helper: load JSONL artifacts
def load_artifacts(run_dir: str) -> pd.DataFrame:
    path = Path(run_dir) / "runs.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return pd.DataFrame(records)
```

**Actions:**
- [ ] Implement DuckDB connection with SQLite attach
- [ ] Implement helper functions for querying and artifact loading
- [ ] Add error handling for missing databases

### 14.3 Section A: Overall Summary

**Notebook cells:**

```python
# A1: Run summary
summary = query("""
    SELECT
        suite_id,
        model_alias,
        COUNT(*) as total_tasks,
        AVG(primary_score) as avg_score,
        AVG(CASE WHEN primary_score >= 0.8 THEN 1 ELSE 0 END) as pass_rate,
        AVG(ttft_ms) as avg_ttft,
        AVG(tokens_per_second) as avg_tps,
        MIN(primary_score) as min_score,
        MAX(primary_score) as max_score
    FROM benchmark.runs
    GROUP BY suite_id, model_alias
    ORDER BY suite_id, avg_score DESC
""")
summary

# A2: Model comparison table
model_summary = query("""
    SELECT
        model_alias,
        COUNT(*) as tasks_run,
        ROUND(AVG(primary_score), 3) as avg_score,
        ROUND(AVG(ttft_ms), 1) as avg_ttft_ms,
        ROUND(AVG(tokens_per_second), 1) as avg_tps,
        ROUND(STDDEV(primary_score), 3) as score_stddev,
        SUM(CASE WHEN primary_score < 0.5 THEN 1 ELSE 0 END) as failures
    FROM benchmark.runs
    GROUP BY model_alias
    ORDER BY avg_score DESC
""")
model_summary
```

**Actions:**
- [ ] Implement summary queries with GROUP BY suite_id, model_alias
- [ ] Display as formatted DataFrames
- [ ] Add markdown cells explaining each table

### 14.4 Section B: Score vs Latency

**Chart specifications:**

```python
# B1: Score vs tokens/sec scatter
scatter_data = query("""
    SELECT
        model_alias,
        primary_score,
        tokens_per_second,
        ttft_ms,
        task_id,
        family
    FROM benchmark.runs
    WHERE primary_score IS NOT NULL
""")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
for model in scatter_data['model_alias'].unique():
    subset = scatter_data[scatter_data['model_alias'] == model]
    ax.scatter(
        subset['tokens_per_second'],
        subset['primary_score'],
        label=model,
        alpha=0.6,
        s=50,
    )
ax.set_xlabel('Tokens per Second')
ax.set_ylabel('Primary Score')
ax.set_title('Score vs Speed by Model')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}score_vs_speed.png', dpi=PLOT_DPI)
plt.show()

# B2: Score vs TTFT scatter (with Pareto frontier)
pareto_data = scatter_data.groupby('model_alias').agg({
    'primary_score': 'mean',
    'ttft_ms': 'mean',
}).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
for _, row in pareto_data.iterrows():
    ax.scatter(row['ttft_ms'], row['primary_score'], s=200, zorder=5)
    ax.annotate(row['model_alias'],
                (row['ttft_ms'], row['primary_score']),
                textcoords="offset points", xytext=(10, 5),
                fontsize=10, fontweight='bold')
ax.set_xscale('log')
ax.set_xlabel('TTFT (ms)')
ax.set_ylabel('Avg Primary Score')
ax.set_title('Model Quality vs Latency')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}score_vs_ttft.png', dpi=PLOT_DPI)
plt.show()
```

**Chart specs:**
- **Score vs Speed scatter**: x=tokens/sec, y=primary_score, points colored by model, alpha=0.6 for density visibility
- **Score vs TTFT with Pareto**: x=TTFT (log scale), y=avg primary score, annotated points, Pareto frontier line connecting optimal points
- **Per-family breakdown**: faceted scatter plot, one facet per task family

**Actions:**
- [ ] Implement score vs tokens/sec scatter plot
- [ ] Implement score vs TTFT with Pareto frontier
- [ ] Implement per-family faceted chart
- [ ] Save charts to `notebooks/outputs/`

### 14.5 Section C: Context-Length Degradation Curves

**Chart specifications:**

```python
# C1: Quality degradation curve
if 'long_context_runs' in [t[0] for t in con.execute("SELECT name FROM benchmark.sqlite_master WHERE type='table'").fetchall()]:
    degradation = query("""
        SELECT
            model_alias,
            context_size_tokens,
            AVG(primary_score) as avg_score,
            STDDEV(primary_score) as score_stddev,
            AVG(prefill_ms) as avg_prefill_ms,
            AVG(decode_ms) as avg_decode_ms,
            AVG(gpu_memory_used_peak_mb) as avg_gpu_peak_mb
        FROM benchmark.long_context_runs
        GROUP BY model_alias, context_size_tokens
        ORDER BY model_alias, context_size_tokens
    """)

    # Quality vs context length
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for model in degradation['model_alias'].unique():
        subset = degradation[degradation['model_alias'] == model].sort_values('context_size_tokens')
        ax.plot(
            subset['context_size_tokens'],
            subset['avg_score'],
            marker='o',
            label=model,
            linewidth=2,
        )
        # Confidence interval shading
        ax.fill_between(
            subset['context_size_tokens'],
            subset['avg_score'] - subset['score_stddev'],
            subset['avg_score'] + subset['score_stddev'],
            alpha=0.15,
        )
    ax.set_xlabel('Context Size (tokens)')
    ax.set_ylabel('Avg Primary Score')
    ax.set_title('Quality Degradation vs Context Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(degradation['context_size_tokens'].unique())
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}quality_degradation.png', dpi=PLOT_DPI)
    plt.show()

    # Speed degradation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for model in degradation['model_alias'].unique():
        subset = degradation[degradation['model_alias'] == model].sort_values('context_size_tokens')
        ax1.plot(subset['context_size_tokens'], subset['avg_prefill_ms'],
                 marker='o', label=model)
        ax2.plot(subset['context_size_tokens'], subset['avg_gpu_peak_mb'],
                 marker='s', label=model)
    ax1.set_ylabel('Avg Prefill Time (ms)')
    ax1.set_title('Prefill Time vs Context Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel('Avg Peak GPU Memory (MB)')
    ax2.set_xlabel('Context Size (tokens)')
    ax2.set_title('GPU Memory Pressure vs Context Length')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}speed_degradation.png', dpi=PLOT_DPI)
    plt.show()
```

**Chart specs:**
- **Quality degradation**: line chart per model, x=context_size_tokens, y=avg primary score, with stddev confidence band
- **Prefill time**: line chart per model, x=context_size_tokens, y=avg prefill_ms
- **GPU memory**: line chart per model, x=context_size_tokens, y=avg peak GPU memory
- **Context breakpoint table**: DataFrame showing each model's context size where score drops below 80% of 2k baseline

**Actions:**
- [ ] Implement quality degradation curve with confidence bands
- [ ] Implement prefill time degradation chart
- [ ] Implement GPU memory pressure chart
- [ ] Compute and display context breakpoint table
- [ ] Handle case where long_context_runs table doesn't exist (skip gracefully)

### 14.6 Section D: Quantization Comparison Plots

**Chart specifications:**

```python
# D1: Quality delta grouped bar chart
quant_data = query("""
    SELECT
        base_model_family,
        quantization_scheme,
        ROUND(AVG(primary_score), 3) as avg_score,
        ROUND(AVG(ttft_ms), 1) as avg_ttft,
        ROUND(AVG(tokens_per_second), 1) as avg_tps
    FROM benchmark.runs
    GROUP BY base_model_family, quantization_scheme
    ORDER BY base_model_family, avg_score DESC
""")

if not quant_data.empty:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(data=quant_data, x='quantization_scheme', y='avg_score',
                hue='base_model_family', ax=ax)
    ax.set_xlabel('Quantization Scheme')
    ax.set_ylabel('Avg Primary Score')
    ax.set_title('Quality by Quantization Scheme')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}quant_quality.png', dpi=PLOT_DPI)
    plt.show()

    # D2: Speed/quality frontier for quantized variants
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for family in quant_data['base_model_family'].unique():
        subset = quant_data[quant_data['base_model_family'] == family]
        for _, row in subset.iterrows():
            ax.scatter(row['avg_tps'], row['avg_score'], s=150, zorder=5)
            label = f"{family}/{row['quantization_scheme']}"
            ax.annotate(label, (row['avg_tps'], row['avg_score']),
                       textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel('Tokens per Second')
    ax.set_ylabel('Avg Primary Score')
    ax.set_title('Speed/Quality Frontier — Quantized Variants')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}quant_frontier.png', dpi=PLOT_DPI)
    plt.show()

    # D3: Quantization heatmap by task category
    quant_cat = query("""
        SELECT
            quantization_scheme,
            category,
            ROUND(AVG(primary_score), 3) as avg_score
        FROM benchmark.runs
        WHERE category IS NOT NULL
        GROUP BY quantization_scheme, category
    """)
    if not quant_cat.empty:
        heatmap = quant_cat.pivot(index='category', columns='quantization_scheme', values='avg_score')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.heatmap(heatmap, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
        ax.set_title('Score Heatmap: Quantization × Task Category')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}quant_heatmap.png', dpi=PLOT_DPI)
        plt.show()
```

**Chart specs:**
- **Quality by scheme**: grouped bar chart, x=quantization_scheme, y=avg_score, hue=base_model_family
- **Speed/quality frontier**: scatter with annotated labels for each variant
- **Heatmap**: rows=task categories, columns=quantization schemes, cells=avg_score, color RdYlGn

**Actions:**
- [ ] Implement grouped bar chart for quality by scheme
- [ ] Implement speed/quality frontier scatter
- [ ] Implement quantization heatmap
- [ ] Handle empty data gracefully

### 14.7 Section E: Prompt-Style Comparison Plots

**Chart specifications:**

```python
# E1: Prompt style comparison by task family
prompt_data = query("""
    SELECT
        prompt_style,
        family,
        model_alias,
        ROUND(AVG(primary_score), 3) as avg_score,
        ROUND(AVG(total_tokens), 0) as avg_tokens,
        ROUND(AVG(ttft_ms), 1) as avg_ttft
    FROM benchmark.runs
    WHERE prompt_style IS NOT NULL
    GROUP BY prompt_style, family, model_alias
    ORDER BY family, avg_score DESC
""")

if not prompt_data.empty:
    # Score by prompt style
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(data=prompt_data, x='prompt_style', y='avg_score',
                hue='family', ax=ax)
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Avg Primary Score')
    ax.set_title('Score by Prompt Style and Task Family')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}prompt_style_score.png', dpi=PLOT_DPI)
    plt.show()

    # Token overhead by prompt style
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(data=prompt_data, x='prompt_style', y='avg_tokens',
                hue='family', ax=ax)
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Avg Total Tokens')
    ax.set_title('Token Overhead by Prompt Style')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}prompt_style_tokens.png', dpi=PLOT_DPI)
    plt.show()
```

**Chart specs:**
- **Score by style**: grouped bar chart, x=prompt_style, y=avg_score, hue=family
- **Token overhead**: grouped bar chart, x=prompt_style, y=avg_total_tokens
- **Safety by style**: if safety data available, bar chart of safety_score by prompt_style

**Actions:**
- [ ] Implement prompt style comparison charts
- [ ] Handle missing prompt_style column gracefully
- [ ] Add safety comparison if safety data exists

### 14.8 Section F: Safety Rankings

**Chart specifications:**

```python
# F1: Safety score bar chart
safety_data = query("""
    SELECT
        model_alias,
        ROUND(AVG(safety_score), 3) as avg_safety,
        SUM(critical_violations) as total_critical,
        SUM(high_violations) as total_high,
        SUM(medium_violations) as total_medium,
        SUM(unsafe_commands) as total_unsafe
    FROM benchmark.command_safety_results
    GROUP BY model_alias
    ORDER BY avg_safety DESC
""")

if not safety_data.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Safety scores
    bars = ax1.bar(safety_data['model_alias'], safety_data['avg_safety'],
                   color=safety_data['avg_safety'].apply(
                       lambda x: 'green' if x >= 0.9 else 'orange' if x >= 0.7 else 'red'))
    ax1.set_ylabel('Avg Safety Score')
    ax1.set_title('Command Safety by Model')
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Safe threshold')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Violation breakdown stacked bar
    safety_data[['total_critical', 'total_high', 'total_medium']].plot(
        kind='bar', stacked=True, ax=ax2,
        color=['red', 'orange', 'yellow'])
    ax2.set_ylabel('Violation Count')
    ax2.set_title('Safety Violations by Severity')
    ax2.set_xticklabels(safety_data['model_alias'], rotation=45, ha='right')
    ax2.legend(['Critical', 'High', 'Medium'])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}safety_rankings.png', dpi=PLOT_DPI)
    plt.show()
```

**Chart specs:**
- **Safety scores**: bar chart colored by safety level (green ≥ 0.9, orange ≥ 0.7, red < 0.7), with threshold lines
- **Violation breakdown**: stacked bar chart by severity (critical/high/medium)

**Actions:**
- [ ] Implement safety score bar chart with color coding
- [ ] Implement violation breakdown stacked bar
- [ ] Handle missing safety data gracefully

### 14.9 Section G: Failure Examples Browser

**Interactive display:**

```python
# G1: Failure table with expandable details
from IPython.display import HTML, display

failures = query("""
    SELECT
        r.run_id,
        r.task_id,
        r.model_alias,
        r.primary_score,
        r.exit_status,
        r.raw_response,
        r.suite_id,
        t.prompt_template
    FROM benchmark.runs r
    LEFT JOIN benchmark.task_metadata t ON r.task_id = t.task_id
    WHERE r.primary_score < 0.7 OR r.exit_status = 'error'
    ORDER BY r.primary_score ASC
    LIMIT 50
""")

if not failures.empty:
    # Display as HTML table with clickable raw response
    failure_html = """
    <style>
    .failure-table { width: 100%; border-collapse: collapse; }
    .failure-table th { background: #f0f0f0; padding: 8px; text-align: left; }
    .failure-table td { padding: 8px; border-bottom: 1px solid #ddd; }
    .failure-row { background: #fff5f5; }
    .error-row { background: #ffe0e0; }
    </style>
    <table class="failure-table">
    <tr><th>Task</th><th>Model</th><th>Score</th><th>Status</th><th>Response (first 200 chars)</th></tr>
    """
    for _, row in failures.iterrows():
        row_class = 'error-row' if row['exit_status'] == 'error' else 'failure-row'
        resp_preview = (row['raw_response'] or '')[:200].replace('<', '&lt;').replace('>', '&gt;')
        failure_html += f"""
        <tr class="{row_class}">
            <td>{row['task_id']}</td>
            <td>{row['model_alias']}</td>
            <td>{row['primary_score']:.3f}</td>
            <td>{row['exit_status']}</td>
            <td>{resp_preview}</td>
        </tr>"""
    failure_html += "</table>"
    display(HTML(failure_html))

# G2: Failure distribution by family
failure_dist = query("""
    SELECT
        family,
        COUNT(*) as failures,
        ROUND(AVG(primary_score), 3) as avg_score,
        COUNT(DISTINCT model_alias) as models_affected
    FROM benchmark.runs
    WHERE primary_score < 0.7 OR exit_status = 'error'
    GROUP BY family
    ORDER BY failures DESC
""")
failure_dist
```

**Actions:**
- [ ] Implement HTML failure table with styling
- [ ] Implement failure distribution by family
- [ ] Implement filtering controls (by model, by score threshold)
- [ ] Add raw response preview with expand option

### 14.10 Section H: Multi-Run Comparison

**Chart specifications:**

```python
# H1: Side-by-side comparison of two runs
if BASELINE_DB_PATH and Path(BASELINE_DB_PATH).exists():
    comparison = query("""
        SELECT
            'current' as run_label,
            model_alias,
            suite_id,
            ROUND(AVG(primary_score), 3) as avg_score,
            ROUND(AVG(ttft_ms), 1) as avg_ttft
        FROM benchmark.runs
        GROUP BY model_alias, suite_id

        UNION ALL

        SELECT
            'baseline' as run_label,
            model_alias,
            suite_id,
            ROUND(AVG(primary_score), 3) as avg_score,
            ROUND(AVG(ttft_ms), 1) as avg_ttft
        FROM baseline.runs
        GROUP BY model_alias, suite_id
    """)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(data=comparison, x='model_alias', y='avg_score',
                hue='run_label', ax=ax)
    ax.set_title('Score Comparison: Current vs Baseline')
    ax.set_ylabel('Avg Primary Score')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}run_comparison.png', dpi=PLOT_DPI)
    plt.show()

    # H2: Per-task delta table
    delta = query("""
        SELECT
            c.task_id,
            c.model_alias,
            c.avg_score as current_score,
            b.avg_score as baseline_score,
            ROUND(c.avg_score - b.avg_score, 3) as delta
        FROM (
            SELECT task_id, model_alias, AVG(primary_score) as avg_score
            FROM benchmark.runs GROUP BY task_id, model_alias
        ) c
        JOIN (
            SELECT task_id, model_alias, AVG(primary_score) as avg_score
            FROM baseline.runs GROUP BY task_id, model_alias
        ) b ON c.task_id = b.task_id AND c.model_alias = b.model_alias
        WHERE ABS(c.avg_score - b.avg_score) > 0.05
        ORDER BY delta ASC
    """)
    delta
```

**Actions:**
- [ ] Implement run comparison bar chart
- [ ] Implement per-task delta table sorted by magnitude
- [ ] Color-code deltas (green for improvement, red for regression)

### 14.11 Section I: Export and Summary

**Final notebook cells:**

```python
# I1: Save all charts summary
import os
print("=== Generated Charts ===")
for f in sorted(Path(OUTPUT_DIR).glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

# I2: Export key tables to CSV for external analysis
model_summary.to_csv(f'{OUTPUT_DIR}model_summary.csv', index=False)
print(f"Exported model_summary.csv")

# I3: Final recommendations text
print("=== Quick Recommendations ===")
best_overall = model_summary.iloc[0]
best_fast = model_summary.sort_values('avg_ttft_ms').iloc[0]
print(f"Best overall: {best_overall['model_alias']} (score {best_overall['avg_score']})")
print(f"Fastest: {best_fast['model_alias']} (TTFT {best_fast['avg_ttft_ms']}ms)")
```

**Actions:**
- [ ] Implement chart inventory listing
- [ ] Implement CSV export of key tables
- [ ] Generate quick recommendation text

### 14.12 Create convenience launch script

**File:** `scripts/run_analysis_notebook.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default to most recent run database
LATEST_DB=$(ls -t runs/*/benchmark.db 2>/dev/null | head -1)
if [ -z "$LATEST_DB" ]; then
    echo "No benchmark databases found in runs/"
    exit 1
fi

echo "Launching notebook with latest database: $LATEST_DB"
jupyter notebook notebooks/benchmark_analysis.ipynb
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 14.13 Add tests

**File:** `tests/test_analysis.py`

**Tests:**
- `test_duckdb_sqlite_attach` — DuckDB can attach to SQLite database
- `test_query_runs` — query helper returns DataFrame with expected columns
- `test_load_artifacts` — JSONL artifacts loaded into DataFrame
- `test_summary_query` — summary aggregation produces expected results
- `test_chart_generation_score_speed` — score vs speed chart saved as PNG
- `test_chart_generation_degradation` — degradation chart saved (with synthetic data)
- `test_failure_query` — failure query returns only low-score rows

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for synthetic databases
- [ ] Verify chart files are created

---

## Acceptance Criteria Checklist

- [ ] Can explore results without manually reading JSONL
- [ ] Can compare current run to previous runs visually
- [ ] Can identify regressions visually
- [ ] Score vs latency chart is generated
- [ ] Context-length degradation curves are generated per model
- [ ] Quantization comparison plots are generated
- [ ] Prompt-style comparison plots are generated
- [ ] Safety rankings chart is generated
- [ ] Failure examples are browsable with raw response preview
- [ ] Charts are saved to `notebooks/outputs/` as PNG
- [ ] Key tables are exportable to CSV
- [ ] `pytest tests/test_analysis.py` passes

## Estimated Effort

3–4 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `notebooks/benchmark_analysis.ipynb` | To create |
| `notebooks/requirements.txt` | To create |
| `notebooks/outputs/.gitkeep` | To create |
| `scripts/run_analysis_notebook.sh` | To create |
| `tests/test_analysis.py` | To create |
