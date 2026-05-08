# Milestone 20 — Automatic Prompt Optimization

## Goal

Use benchmark results to identify the best prompt style per task family and automatically generate + test improved prompt templates.

## The Problem

We have 7 prompt styles (plain, repl, terse, patch_only, architect, json_schema, step_by_step) and the ability to compare them via style sweeps. But we don't have a system that:

1. **Analyzes** existing benchmark results to rank styles by task family
2. **Suggests** new prompt template candidates based on patterns in the data
3. **Tests** those candidates in a structured way
4. **Proposes** a final recommended template per task family

## Design

### PromptOptimizationRunner (new)

A high-level orchestrator that:
- Takes a suite + model + task filter
- Runs a style sweep with a **dynamic candidate pool** of templates
- Compares all candidates against a baseline (default: `plain`)
- Produces an optimization report

### TemplateProposal (new dataclass)

A structured representation of a candidate prompt template:

```python
@dataclass
class TemplateProposal:
    name: str                  # e.g. "debugging-v2"
    task_family: str           # e.g. "docker_compose"
    template_str: str          # Jinja2 template string
    baseline: str = "plain"    # Baseline to compare against
    instructions: str = ""     # Why this variant was proposed
    score: float = 0.0         # Score after evaluation (set post-run)
    score_delta: float = 0.0   # Delta vs baseline
    run_count: int = 0         # Number of runs for this variant
```

### PromptAnalysis (new dataclass)

A structured analysis of benchmark results:

```python
@dataclass
class PromptAnalysis:
    """Analysis results from existing benchmark data."""
    family_rankings: dict[str, list[tuple[str, float, float]]]  # family -> [(style, avg_score, margin)]
    best_style_overall: str
    best_style_by_family: dict[str, str]  # family -> best_style
    style_variances: dict[str, float]     # style -> score_variance
    insufficient_data: list[str]          # task families with too few style comparisons
```

### The Optimization Loop

```
1. Analyze existing benchmark results
   -> Identify best style per family (if enough data exists)

2. Generate candidate proposals
   -> Based on patterns: e.g. if REPL is best for debugging, try "REPL + terse" variant
   -> Based on failure analysis: e.g. if format_compliance fails on patch_only, try "patch_only + format guard"
   -> User can also provide custom proposals via CLI

3. Run candidates against a subset of tasks
   -> Use StyleSweepRunner with candidate templates + baselines
   -> Compare scores, token usage, latency

4. Produce optimization report
   -> "For task family X, style Y scored best with margin Z"
   -> "Proposed template 'debugging-v2' scores +0.12 vs plain baseline"
   -> "Recommended: adopt Y for family X, test Z in production"
```

### CLI Interface

```bash
# Analyze existing benchmark results — shows best styles per family
bench-harness prompt-opt analyze --suite coding_benchmark --db runs/.../benchmark.db

# Propose new candidate templates (dry-run: no actual runs)
bench-harness prompt-opt propose --suite coding_benchmark --db runs/.../benchmark.db

# Run proposed candidates for evaluation
bench-harness prompt-opt run-proposed --spec proposed_templates.yaml --models agent-code

# Run a custom optimization sweep with user-specified templates
bench-harness prompt-opt run --templates custom1,custom2,repl --base-styles plain --suite coding_benchmark
```

### Custom Template Loading

Templates can be specified in a YAML spec:

```yaml
name: debugging-optimization
suites:
  - coding_benchmark
models:
  - agent-code
baselines:
  - plain
candidates:
  - name: repl-terse
    style: repl
    instructions: "REPL mode but with terse output to reduce tokens"
  - name: architect-step
    style: architect
    instructions: "Architect-style analysis followed by step-by-step execution"
  - name: format-protected-patch
    style: patch_only
    instructions: "Patch-only output with explicit format guard (explain before patch)"
  - name: json-debug
    style: json_schema
    instructions: "JSON output with debug fields (hypothesis, test, result)"
```

### Scoring Criteria for Proposals

Each candidate is evaluated on:
- **Score delta vs baseline** — primary metric, must exceed 0.05 to be recommended
- **Score stability** — low variance across tasks is preferred over high variance
- **Token efficiency** — score/token ratio
- **Family coverage** — must be tested across at least 3 tasks in the target family

### Integration with Existing System

- Reuses `StyleSweepRunner` for candidate evaluation
- Reuses `reports/style_comparison.py` for per-style comparison logic
- Reuses `reports/helpers.py` for grouping and variance calculations
- Reuses `analysis.py` `style_comparison_df()` for DataFrame queries
- Adds new section to v2 report: "Prompt Optimization"

## Tasks

### 20.1 Data structures and analysis

* [ ] Implement `PromptAnalysis` dataclass in `src/bench_harness/prompt_optimization/analysis.py`
* [ ] Implement `analyze_style_data(db_path, suite_id, min_runs_per_style=3)` — reads SQLite, computes per-family rankings
* [ ] Implement `best_style_family(task_family, family_runs)` — picks best style with margin check
* [ ] Implement `detect_insufficient_data(runs, min_runs=3)` — flags families with too few comparisons
* [ ] Add 40+ tests for analysis functions

### 20.2 Template proposals

* [ ] Implement `TemplateProposal` dataclass in `src/bench_harness/prompt_optimization/proposals.py`
* [ ] Implement `generate_proposals(runs, task_family)` — automatic suggestion based on failure patterns
  * If `patch_only` has low format_compliance: suggest `format-protected-patch`
  * If `repl` scores high but is verbose: suggest `repl-terse`
  * If `json_schema` has high format but low correctness: suggest `json-debug`
* [ ] Implement `load_proposals_from_yaml(spec_path)` — load custom proposals from YAML
* [ ] Implement `TemplateRegistry` — manage candidate + baseline templates
* [ ] Add 20+ tests for proposal system

### 20.3 Optimization runner

* [ ] Implement `PromptOptimizationRunner` in `src/bench_harness/prompt_optimization/runner.py`
* [ ] `optimize()` — orchestrates: analyze -> propose -> run -> compare
* [ ] `run_candidates(spec, runner, tasks, models)` — runs sweep with candidates + baselines
* [ ] `compare_results(candidate_results, baseline_results)` — score delta computation
* [ ] Reuse `StyleSweepRunner` internally for sweep execution
* [ ] Store optimization results in a structured JSON format in the run directory

### 20.4 CLI commands

* [ ] Add `prompt-opt` CLI group with subcommands: `analyze`, `propose`, `run-proposed`, `run`
* [ ] `bench-harness prompt-opt analyze --suite <suite> --db <db>` — show best styles per family
* [ ] `bench-harness prompt-opt propose --suite <suite> --db <db>` — show candidate suggestions
* [ ] `bench-harness prompt-opt run-proposed --spec <yaml> --models <models>` — evaluate proposals
* [ ] `bench-harness prompt-opt run --templates <names> --base-styles <names> --suite <suite> --models <models>` — custom sweep
* [ ] Rich table output for terminal display
* [ ] JSON output option: `--format json`

### 20.5 Report integration

* [ ] Add `reports/prompt_optimization.py` — generates markdown optimization report
* [ ] Sections: Analysis Summary, Best Styles Per Family, Proposed Templates, Candidate Results, Recommendations
* [ ] Integrate into v2 report with `_append_prompt_optimization()` section
* [ ] Auto-detect when optimization data exists in runs

### 20.6 Tests

* [ ] `tests/test_prompt_optimization.py` — comprehensive tests for analysis, proposals, runner
* [ ] Integration test: analyze -> propose -> run -> compare pipeline with in-memory DB
* [ ] Test custom YAML spec loading
* [ ] Test edge cases: no style data, single style, all failures

## Acceptance criteria

* [ ] `bench-harness prompt-opt analyze` shows best prompt style per task family from existing benchmark data
* [ ] `bench-harness prompt-opt propose` generates actionable candidate templates based on failure patterns
* [ ] Custom proposal YAML is loaded and validated correctly
* [ ] `bench-harness prompt-opt run` runs a style sweep with user-specified templates
* [ ] Optimization report shows score deltas, recommendations, and token efficiency
* [ ] Pipeline works end-to-end: analyze existing data -> propose -> run -> compare
* [ ] 100+ tests covering analysis, proposals, runner, and CLI

## Files to create/modify

### New files:
- `src/bench_harness/prompt_optimization/__init__.py`
- `src/bench_harness/prompt_optimization/analysis.py`
- `src/bench_harness/prompt_optimization/proposals.py`
- `src/bench_harness/prompt_optimization/runner.py`
- `src/bench_harness/reports/prompt_optimization.py`
- `milestones/m20-prompt-optimization.md`
- `tests/test_prompt_optimization.py`

### Modified files:
- `src/bench_harness/cli.py` — add `prompt-opt` CLI group
- `src/bench_harness/reports/v2.py` — add prompt optimization section

## Edge Cases to Handle

1. **No existing style data** — `analyze` should gracefully say "no style comparison data found in this suite"
2. **Single style per family** — can't recommend a best style, should flag "insufficient data"
3. **All runs failed** — optimization should skip failed tasks and report "no successful runs for analysis"
4. **Custom templates with unknown styles** — should validate against known template format
5. **Very small candidate pool** — if only 1 candidate with no baseline, can't compute delta

## Risks and Mitigations

### Risk: Style sweep runs are expensive
**Mitigation**: `analyze` and `propose` are read-only (no model calls). Only `run-proposed` and `run` hit models. Support `--dry-run` flag on all commands.

### Risk: Recommendations are too narrow
**Mitigation**: Show margins, not just winners. If the margin is small (<0.05), recommend "no clear winner — all styles perform similarly."

### Risk: Candidate templates don't generalize across families
**Mitigation**: Require evaluation across 3+ tasks in a family before making a recommendation. Show per-task results in the report.
