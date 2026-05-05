# Milestone 7 — LLM Judge Integration

## Goal

Score complex answers with explicit rubrics using an LLM-as-judge. Support structured JSON judge output, self-consistency, pairwise comparison, human override, and auditable judge logs.

## Phase

Phase D — Data flywheel (Milestone 1 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap, OpenAI client, storage)
- Milestone 2 (task schema, prompt templates)
- Milestone 3 (timing metrics)
- Milestone 4 (basic scorers — judge is a fallback when exact/executable scoring unavailable)
- Milestone 5 (coding task runner — judge scores complex code explanations)
- Milestone 6 (local coding-agent suite — judge scores rubric-based tasks)

---

### Leveraged Libraries

- **jsonschema** (already in pyproject.toml): Validate judge JSON output against expected schema
- No heavy external dependencies — judge scoring is harness-specific logic

---

## Subtasks

### 7.1 Implement judge model configuration

**File:** `configs/models.yaml` (update)

**New section:**
```yaml
judge:
  model: "max-brain"            # judge model alias (uses existing model config)
  provider: openai_compatible
  base_url: "http://spark-e287.local:4000/v1"
  temperature: 0                 # deterministic judging
  max_tokens: 2048
  self_consistency_rounds: 1    # number of independent judge calls (1 = no self-consistency)
```

**File:** `src/bench_harness/config.py` (update)

**Functions (add):**
```python
def load_judge_config(path: str) -> dict:
    """Load judge configuration from models.yaml."""

def get_judge_model(config: dict) -> dict:
    """Get judge model settings."""
```

**Actions:**
- [x] Add judge section to models.yaml
- [x] Implement config loader functions (`load_judge_config`, `load_rubric_config`, `get_rubric`)
- [x] Validate judge config has required fields (model, base_url)

### 7.2 Define rubric YAML format

**File:** `configs/judge_rubrics.yaml`

**Actions:**
- [x] Write judge_rubrics.yaml with all rubric definitions
- [x] Define rubric schema: name, description, dimensions with weight/scale/description/anchors
- [x] Ensure weights per rubric sum to 1.0

### 7.3 Implement judge prompt templates

**Directory:** `configs/judge_prompts/`

**Actions:**
- [x] Create both judge prompt templates
- [x] Ensure templates produce prompts that enforce JSON-only output
- [x] Add system message to reinforce JSON format requirement

### 7.4 Implement LLM judge scorer

**File:** `src/bench_harness/scorers/llm_judge.py`

**Actions:**
- [x] Implement `LLMJudgeScorer` with rubric-based scoring
- [x] Implement `_call_judge` with error handling and retry
- [x] Implement JSON output parsing with validation
- [x] Implement self-consistency aggregation
- [x] Implement pairwise comparison mode
- [x] Register with `@register_scorer`

### 7.5 Implement pairwise comparison scorer

**File:** `src/bench_harness/scorers/pairwise.py`

**Actions:**
- [x] Implement `PairwiseScorer` class
- [x] Implement `PairwiseResult` dataclass
- [x] Add CLI subcommand `compare-pairwise`
- [x] Wire judge model client creation

### 7.6 Add human override field

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `score_details` table:**
```sql
ALTER TABLE score_details ADD COLUMN human_override INTEGER;  -- 0/1
ALTER TABLE score_details ADD COLUMN human_score REAL;
ALTER TABLE score_details ADD COLUMN human_note TEXT;
```

**New table — `judge_evaluations`:**
```sql
CREATE TABLE judge_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    rubric_name TEXT NOT NULL,
    score TEXT NOT NULL,
    dimensions_json TEXT,
    explanation TEXT,
    raw_response TEXT,
    created_at TEXT NOT NULL
);
```

**New table — `pairwise_comparisons`:**
```sql
CREATE TABLE pairwise_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_a TEXT NOT NULL,
    model_b TEXT NOT NULL,
    winner TEXT NOT NULL,
    margin TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    dimension_comparison_json TEXT,
    raw_judge_response TEXT,
    judge_model TEXT NOT NULL,
    human_override INTEGER DEFAULT 0,
    human_winner TEXT,
    human_note TEXT,
    created_at TEXT NOT NULL
);
```

**Methods (add):**
- `save_judge_evaluation()`
- `save_pairwise_comparison()`
- `get_judge_evaluations(suite_id)`
- `get_pairwise_comparisons(suite_id)`
- `save_run()` updated to include judge columns

**Actions:**
- [x] Implement schema migration for score_details human columns
- [x] Add `judge_evaluations` table
- [x] Add `pairwise_comparisons` table
- [x] Implement save/retrieve methods

### 7.7 Add judge CLI support

**File:** `src/bench_harness/cli.py` (update)

**New CLI flags for run command:**
```
--judge          BOOLEAN   Run LLM judge scorer on scored tasks
--judge-model    TEXT      Judge model alias (default: from config)
--rubric         TEXT      Rubric name (default: from task)
--judge-rounds   INT       Self-consistency rounds (default: 1)
```

**New CLI command:**
```
bench-harness judge-config list      # list available rubrics
bench-harness judge-config show <name>  # show rubric details
```

**Actions:**
- [x] Add `--judge` flag to `run` subcommand
- [x] Add `judge-config` subcommand with list/show actions
- [x] Wire judge model client from config

### 7.8 Store judge explanations

**File:** `src/bench_harness/storage/artifacts.py` (update)

**New function:**
```python
def save_judge_artifact(
    run_id: str,
    out_dir: str,
    raw_response: str,
    parsed_scores: dict,
    rubric_name: str,
    judge_model: str,
    prompt: str,
) -> dict[str, str]:
    """Save judge output artifacts."""
```

**Artifacts saved:**
- `{out_dir}/judge_raw_{run_id}.json` — raw judge response
- `{out_dir}/judge_parsed_{run_id}.json` — parsed scores with reasons
- `{out_dir}/judge_prompt_{run_id}.txt` — full prompt sent to judge

**Actions:**
- [x] Implement judge artifact saving
- [x] Save both raw and parsed outputs for auditability
- [x] Save the judge prompt for reproducibility

### 7.9 Update Markdown report with judge scores

**File:** `src/bench_harness/reports/markdown.py` (update)

**New report sections:**

1. **Judge-Scored Tasks:**
```markdown
## Judge-Scored Tasks

| Task | Model | Judge Score | Judge Model | Dimensions |
|---|---|---|---|---|
```

2. **Judge Dimension Breakdown:**
```markdown
## Judge Dimension Breakdown

| Model | Dimension | Avg Score | Std Dev |
|---|---|---|---|
```

3. **Pairwise Comparisons:**
```markdown
## Pairwise Comparisons

| Task | Model A | Model B | Winner | Margin | Confidence |
|---|---|---|---|---|---|
```

**Actions:**
- [x] Add Judge-Scored Tasks section
- [x] Add Dimension Breakdown section with per-model averages and stddev
- [x] Add Pairwise Comparison section
- [x] Include dimension-level breakdowns
- [x] Compute self-consistency metrics

### 7.10 Add judge scorer tests

**File:** `tests/test_llm_judge.py`

**Tests:**
- `test_rubric_config_loads` — rubrics YAML loads and validates
- `test_rubric_weights_sum_to_one` — each rubric's weights sum to 1.0
- `test_judge_prompt_rendering` — rubric judge template renders with context
- `test_pairwise_prompt_rendering` — pairwise template renders correctly
- `test_judge_output_parsing` — valid JSON output parsed correctly
- `test_judge_output_validation` — out-of-range scores rejected
- `test_judge_output_extract_from_fences` — JSON in code fences extracted
- `test_self_consistency_aggregation` — multiple rounds averaged correctly
- `test_self_consistency_stddev` — standard deviation computed correctly
- `test_pairwise_result_structure` — PairwiseResult has all fields
- `test_human_override_saved` — override persists in SQLite
- `test_judge_artifact_save` — all judge artifacts written to disk

**Note:** Tests use mock client responses, not real LLM calls.

**Actions:**
- [ ] Implement all tests
- [ ] Mock OpenAI client for judge tests
- [ ] Validate rubric YAML structure in tests
- [ ] Test edge cases: malformed JSON, missing dimensions

---

## Acceptance Criteria Checklist

- [x] Complex answers receive rubric scores with per-dimension breakdown
- [x] Pairwise model comparisons can be generated with confidence scores
- [x] Judge outputs are auditable (raw response, parsed scores, prompt all saved)
- [x] Judge model can be swapped via config without code changes
- [x] Self-consistency mode supported (configurable rounds)
- [x] Human override field stores corrections to judge scores
- [x] Pairwise results stored in SQLite with dimension-level comparison
- [x] Markdown report includes judge score tables and pairwise summary
- [ ] `pytest tests/test_llm_judge.py` passes (pending LLM judge scorer implementation by subagent)
- [x] Judge is used only as fallback — exact/executable scorers take precedence

## Estimated Effort

3 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/scorers/llm_judge.py` | Created by subagent |
| `src/bench_harness/scorers/pairwise.py` | Created by subagent |
| `configs/judge_rubrics.yaml` | Created by subagent |
| `configs/judge_prompts/rubric_judge.md` | Created by subagent |
| `configs/judge_prompts/pairwise_judge.md` | Created by subagent |
| `configs/models.yaml` | Updated (judge section) |
| `src/bench_harness/config.py` | Updated (load_judge_config, load_rubric_config) |
| `src/bench_harness/storage/sqlite.py` | Updated (judge schema, new tables, migrations) |
| `src/bench_harness/storage/artifacts.py` | Updated (save_judge_artifact) |
| `src/bench_harness/cli.py` | Updated (--judge flag, judge-config command) |
| `src/bench_harness/reports/markdown.py` | Updated (judge sections) |
| `src/bench_harness/runners/completion_runner.py` | Updated (judge fields on RunResult) |
| `src/bench_harness/tasks/task_schema.py` | Already has rubric_name field |
| `ROADMAP.md` | Updated (M7 marked done) |
| `README.md` | Updated (status line, milestone table) |
