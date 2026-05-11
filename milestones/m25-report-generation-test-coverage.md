# Milestone 25 — Report Generation Test Coverage

## Goal

Achieve >80% test coverage on the report generation modules, focusing on
`reports/v2.py` (66%) and `reports/markdown.py` (40%). These modules
generate the benchmark reports that summarize all analysis — they are the
primary output artifacts of the harness and are not yet well-tested.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- M17–M24: all infrastructure, tests, and CLI complete

## Acceptance Criteria (Definition of Done)

- `reports/v2.py` coverage >= 95%
- `reports/markdown.py` coverage >= 92%
- `reports/helpers.py` >= 95% (maintain, already 95%)
- All existing tests continue to pass (no regressions)
- Total test count increases by >= 155 new tests

## Subtasks

### 25.1 v2 Report Section Tests

File: `tests/test_reports_v2_sections.py` — minimum 75 tests

Covers every `_append_*` section function and helper in `reports/v2.py`:

**Helper functions (7 tests):**
- `_format_delta` — positive, negative, zero
- `_format_safe_pct` — zero total, 100% safe, 50% safe
- `_get_score_for_model` — normal, no scores, all None, single run
- `_get_speed_for_model` — normal, no TPS, zero TPS
- `_get_context_for_model` — single value, multiple values, zero
- `_get_quantization_for_model` — normal, no quant, multiple values

**Section tests (68 tests):**

- `TestExecutiveSummary`: best score/safe/TPS/context/quantized, single model, no scores
- `TestModelComparison`: multi-model ranking, single model, context/quant labels
- `TestBestByFamily`: multi-family multi-model, single-model-per-family skip
- `TestSpeedQualityFrontier`: Pareto frontier, no scored data, single model
- `TestContextAnalysis`: 2+ context sizes, single size (early return), no data (early return)
- `TestQuantizationComparison`: 2+ quant levels, single quant (early return), reference selection
- `TestStyleAnalysis`: runs with prompt_style, no style data (section absent)
- `TestPromptOptimization`: optimization suite_id detected, no opt data (absent), no scored results
- `TestJudgeAnalysis`: judge_score present, no judge data (absent), malformed JSON in dimensions
- `TestFailureAnalysis`: mixed success/error, safety flags, empty errors, cluster truncation
- `TestRegressionDetection`: with/without prior runs, tolerance, sorted by delta
- `TestDiscriminatingTasks`: multiple tasks/scores, single-model tasks excluded, top-15 limit
- `TestPublicBaseline`: public. prefix runs, no public runs, score_secondary JSON vs other types
- `TestIdentityStamp`: openai_models_id/vllm_served_model_name, no identity data, alias mismatch

### 25.2 Markdown Report Tests

File: `tests/test_reports_markdown.py` — minimum 50 tests

Covers `reports/markdown.py` legacy report generation:

**`generate_report` dispatch tests (4 tests):**
- `v2=True` — delegates to `generate_report_v2`
- `v2=False` — delegates to `_generate_legacy_report`
- `sections` arg passed through
- `prior_runs` arg passed through

**Legacy report tests (25 tests):**
- `_generate_legacy_report` — header, models table, summary table, timing, per-task results, slowest tasks, scoring summary
- `_append_coding_agent_ranking` — with/without code_type runs
- `_append_judge_sections` — judge tasks, dimensions, pairwise, no judge data
- `_group_by_family` / `_extract_family_from_task_id` — various ID formats, skip prefixes

**Context length analysis tests (8 tests):**
- Multiple context sizes with breakpoint detection
- No context data
- Breakpoint thresholds (>10% drop)

**Summary print tests (5 tests):**
- `print_summary` — multiple models, mixed success/error, Rich table output

**Grouping tests (8 tests):**
- `_group_by_family` — correct extraction, "other" fallback, skip list
- `_extract_family_from_task_id` — various formats, returns None for skip prefixes

### 25.3 Report Integration Tests

File: `tests/test_reports_v2_integration.py` — minimum 30 tests

End-to-end report generation covering full report pipeline:

- `TestReportV2FullGeneration` — all default sections in one report
- `TestReportV2SectionFiltering` — single section, non-existent section, empty list
- `TestReportV2WithModelsConfig` — model info table with backend/quantization/notes
- `TestReportV2WithPriorRuns` — regression + improvement in same report
- `TestReportV2SafetyFlags` — safety_score < 1.0 in failure section
- `TestReportV2AllSectionsTogether` — verify all 14 section headers present
- `TestReportV2EdgeCases` — empty runs, all None scores, all errors, long strings

## Files Created

- `tests/test_reports_v2_sections.py`
- `tests/test_reports_markdown.py`
- `tests/test_reports_v2_integration.py`

## Definition of Done Checklist

When all of the following are true:

- [ ] v2.py >= 95% coverage
- [ ] markdown.py >= 92% coverage
- [ ] All existing 1511+ tests pass
- [ ] >= 155 new tests added
- [ ] No code changes required (tests only)
- [ ] Commit message references M25 acceptance criteria
