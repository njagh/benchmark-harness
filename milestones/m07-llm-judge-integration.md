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
- [ ] Add judge section to models.yaml
- [ ] Implement config loader functions
- [ ] Validate judge config has required fields (model, base_url)

### 7.2 Define rubric YAML format

**File:** `configs/judge_rubrics.yaml`

```yaml
rubrics:
  coding_explanation:
    name: "Coding Explanation Quality"
    description: "Evaluates quality of bug explanation and fix"
    dimensions:
      correctness:
        weight: 0.30
        scale: 5
        description: "Is the root cause correctly identified?"
        anchors:
          1: "Completely wrong diagnosis"
          3: "Partially correct, misses key detail"
          5: "Accurate root cause with precise explanation"
      completeness:
        weight: 0.20
        scale: 5
        description: "Does the explanation cover all aspects of the issue?"
        anchors:
          1: "Only mentions symptom"
          3: "Covers main issue but misses context"
          5: "Covers root cause, symptoms, and context"
      specificity:
        weight: 0.20
        scale: 5
        description: "Are suggestions specific and actionable?"
        anchors:
          1: "Vague, non-actionable advice"
          3: "Some specific steps but gaps remain"
          5: "Precise, step-by-step, copy-paste-ready"
      safety:
        weight: 0.15
        scale: 5
        description: "Are suggested commands safe?"
        anchors:
          1: "Suggests destructive or risky commands"
          3: "Mostly safe but one risky suggestion"
          5: "All commands are safe and reversible"
      format_compliance:
        weight: 0.10
        scale: 5
        description: "Does the response follow the requested format?"
        anchors:
          1: "Completely ignores format requirements"
          3: "Mostly follows format with minor deviations"
          5: "Perfectly follows requested format"
      minimality:
        weight: 0.05
        scale: 5
        description: "Is the fix minimal and focused?"
        anchors:
          1: "Over-engineered, changes unrelated code"
          3: "Slightly more changes than needed"
          5: "Minimal, surgical fix touching only what's needed"

  shell_troubleshooting:
    name: "Shell Troubleshooting Quality"
    dimensions:
      correctness:
        weight: 0.35
        scale: 5
      safety:
        weight: 0.25
        scale: 5
      incremental_diagnosis:
        weight: 0.20
        scale: 5
      clarity:
        weight: 0.10
        scale: 5
      command_specificity:
        weight: 0.10
        scale: 5

  architecture_plan:
    name: "Architecture Plan Quality"
    dimensions:
      correctness:
        weight: 0.25
        scale: 5
      completeness:
        weight: 0.25
        scale: 5
      specificity:
        weight: 0.20
        scale: 5
      feasibility:
        weight: 0.15
        scale: 5
      clarity:
        weight: 0.15
        scale: 5
```

**Actions:**
- [ ] Write judge_rubrics.yaml with all rubric definitions
- [ ] Define rubric schema: name, description, dimensions with weight/scale/description/anchors
- [ ] Ensure weights per rubric sum to 1.0

### 7.3 Implement judge prompt templates

**Directory:** `configs/judge_prompts/`

**File:** `configs/judge_prompts/rubric_judge.md`

```markdown
You are an expert evaluator. Score the following model response using the rubric below.

## Task
{{ task.prompt }}

{% if task.input.files %}
## Context Files
{% for file in task.input.files %}
### {{ file.name }}
{{ file.content }}
{% endfor %}
{% endif %}

## Model Response
{{ response }}

## Rubric: {{ rubric.name }}
{{ rubric.description }}

Score each dimension on a scale of 1-{{ rubric.max_scale }}:
{% for dim in rubric.dimensions %}
- **{{ dim.name }}** (weight: {{ dim.weight }}): {{ dim.description }}
  - 1: {{ dim.anchors.1 }}
  - 3: {{ dim.anchors.3 }}
  - 5: {{ dim.anchors.5 }}
{% endfor %}

## Output Format
Respond with ONLY valid JSON in this exact format:
{
  "scores": {
    "dimension_name": {"score": N, "reason": "explanation"}
  },
  "weighted_total": 0.00,
  "summary": "one-sentence overall assessment"
}
```

**File:** `configs/judge_prompts/pairwise_judge.md`

```markdown
You are an expert evaluator. Compare two model responses to the same task and determine which is better.

## Task
{{ task.prompt }}

## Response A (Model: {{ model_a }})
{{ response_a }}

## Response B (Model: {{ model_b }})
{{ response_b }}

## Output Format
Respond with ONLY valid JSON:
{
  "winner": "A" | "B" | "tie",
  "margin": "clear" | "slight" | "minimal",
  "confidence": 0.00,
  "reason": "detailed explanation of why one is better",
  "dimension_comparison": {
    "correctness": "A" | "B" | "tie",
    "safety": "A" | "B" | "tie",
    "clarity": "A" | "B" | "tie"
  }
}
```

**Actions:**
- [ ] Create both judge prompt templates
- [ ] Ensure templates produce prompts that enforce JSON-only output
- [ ] Add system message to reinforce JSON format requirement

### 7.4 Implement LLM judge scorer

**File:** `src/bench_harness/scorers/llm_judge.py`

**Class:** `LLMJudgeScorer(BaseScorer)`

```python
class LLMJudgeScorer(BaseScorer):
    name = "llm_judge"
    version = "1.0"

    def __init__(
        self,
        client: OpenAICompatClient,
        rubric_name: str | None = None,
        rubric_config: dict | None = None,
        self_consistency_rounds: int = 1,
    ):
        self.client = client
        self.rubric_name = rubric_name
        self.rubric_config = rubric_config
        self.self_consistency_rounds = self_consistency_rounds

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Score a response using LLM judge with explicit rubric."""

    async def _call_judge(self, prompt: str) -> dict:
        """Make a single judge call and parse JSON response."""

    def _parse_judge_output(self, raw_json: str) -> dict:
        """Parse and validate judge JSON output."""

    def _aggregate_self_consistency(self, results: list[dict]) -> dict:
        """Average scores across multiple judge rounds."""

    def score_pairwise(
        self,
        task: Task,
        response_a: str,
        response_b: str,
        model_a: str,
        model_b: str,
    ) -> ScoreResult:
        """Compare two responses using LLM judge."""
```

**Judge output parsing:**
```python
def _parse_judge_output(self, raw_json: str) -> dict:
    # Extract JSON from response
    json_str = extract_json(raw_json)
    data = json.loads(json_str)

    # Validate structure
    assert "scores" in data
    assert "weighted_total" in data

    # Validate each dimension score is in range
    for dim_name, dim_data in data["scores"].items():
        assert 1 <= dim_data["score"] <= self.rubric_config["max_scale"]

    return data
```

**Self-consistency aggregation:**
```python
def _aggregate_self_consistency(self, results: list[dict]) -> dict:
    """Average dimension scores across rounds, keep all individual reasons."""
    averaged_scores = {}
    for dim_name in results[0]["scores"]:
        scores = [r["scores"][dim_name]["score"] for r in results]
        reasons = [r["scores"][dim_name]["reason"] for r in results]
        averaged_scores[dim_name] = {
            "score": sum(scores) / len(scores),
            "reasons": reasons,
            "stddev": statistics.stdev(scores) if len(scores) > 1 else 0,
        }
    return {"scores": averaged_scores, "weighted_total": ...}
```

**Actions:**
- [ ] Implement `LLMJudgeScorer` with rubric-based scoring
- [ ] Implement `_call_judge` with error handling and retry
- [ ] Implement JSON output parsing with validation
- [ ] Implement self-consistency aggregation
- [ ] Implement pairwise comparison mode
- [ ] Register with `@register_scorer`

### 7.5 Implement pairwise comparison scorer

**File:** `src/bench_harness/scorers/pairwise.py`

**Class:** `PairwiseScorer(BaseScorer)`

```python
class PairwiseScorer(BaseScorer):
    name = "pairwise"
    version = "1.0"

    def __init__(self, client: OpenAICompatClient):
        self.client = client

    def score_pairwise(
        self,
        task: Task,
        response_a: str,
        response_b: str,
        model_a: str,
        model_b: str,
    ) -> PairwiseResult:
        """Run pairwise comparison and return structured result."""
```

**Class:** `PairwiseResult`

```python
@dataclass
class PairwiseResult:
    task_id: str
    model_a: str
    model_b: str
    winner: str              # "A", "B", or "tie"
    margin: str              # "clear", "slight", "minimal"
    confidence: float        # 0.0–1.0
    reason: str
    dimension_comparison: dict[str, str]  # {dimension: "A"|"B"|"tie"}
    judge_model: str
    raw_judge_response: str
    created_at: str
```

**CLI integration:**
```
bench_harness compare-pairwise \
  --task <task_id> \
  --run-a <run_id_A> \
  --run-b <run_id_B> \
  --judge max-brain
```

**Actions:**
- [ ] Implement `PairwiseScorer` class
- [ ] Implement `PairwiseResult` dataclass
- [ ] Add CLI subcommand `compare-pairwise`
- [ ] Wire judge model client creation

### 7.6 Add human override field

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `score_details` table:**
```sql
ALTER TABLE score_details ADD COLUMN human_override INTEGER;  -- 0/1
ALTER TABLE score_details ADD COLUMN human_score REAL;
ALTER TABLE score_details ADD COLUMN human_note TEXT;
```

**New table — `pairwise_comparisons`:**
```sql
CREATE TABLE pairwise_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_a TEXT NOT NULL,
    model_b TEXT NOT NULL,
    winner TEXT NOT NULL,        -- "A", "B", or "tie"
    margin TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    dimension_comparison_json TEXT,
    judge_model TEXT NOT NULL,
    raw_judge_response TEXT,
    human_override INTEGER DEFAULT 0,
    human_winner TEXT,
    human_note TEXT,
    created_at TEXT NOT NULL
);
```

**Methods (add):**
- `save_pairwise_comparison(result: PairwiseResult)`
- `override_judge_score(run_id: str, scorer_name: str, human_score: float, human_note: str)`
- `get_pairwise_results(task_id: str | None, model_a: str | None, model_b: str | None) -> list[dict]`

**Actions:**
- [ ] Implement schema migration
- [ ] Add `pairwise_comparisons` table
- [ ] Implement save/retrieve/override methods

### 7.7 Add judge self-consistency CLI support

**File:** `src/bench_harness/cli.py` (update)

**New CLI flags for judge mode:**
```
--judge-model       TEXT   Judge model alias     (default: from config)
--rubric            TEXT   Rubric name           (default: from task)
--judge-rounds      INT    Self-consistency rounds (default: 1)
--pairwise          BOOLEAN Run pairwise comparison mode
```

**New CLI command:**
```
bench_harness judge \
  --run-id <run_id> \
  --judge-model max-brain \
  --rubric coding_explanation
```

**Actions:**
- [ ] Add `judge` subcommand to CLI
- [ ] Add pairwise comparison subcommand
- [ ] Wire judge model client from config
- [ ] Support loading run result from SQLite for re-scoring

### 7.8 Store judge explanations

**File:** `src/bench_harness/storage/artifacts.py` (update)

**New function:**
```python
def save_judge_artifact(
    run_id: str,
    out_dir: str,
    judge_raw_response: str,
    parsed_scores: dict,
    rubric_name: str,
    judge_model: str,
) -> dict[str, str]:
    """Save judge output artifacts.
    
    Returns dict of artifact name -> file path.
    """
```

**Artifacts saved:**
- `{out_dir}/judge_raw_{run_id}.json` — raw judge response
- `{out_dir}/judge_parsed_{run_id}.json` — parsed scores with reasons
- `{out_dir}/judge_prompt_{run_id}.txt` — full prompt sent to judge

**Actions:**
- [ ] Implement judge artifact saving
- [ ] Save both raw and parsed outputs for auditability
- [ ] Save the judge prompt for reproducibility

### 7.9 Update Markdown report with judge scores

**File:** `src/bench_harness/reports/markdown.py` (update)

**New report sections:**

1. **Judge-Scored Tasks:**
```markdown
## Judge-Scored Tasks

| Task | Model | Judge Score | Dimensions | Judge Model |
|---|---|---|---|---|
```

2. **Dimension Breakdown (per model):**
```markdown
## Judge Dimension Breakdown — {{ model }}

| Dimension | Avg Score (1-5) | Std Dev |
|---|---|---|
```

3. **Pairwise Comparison Summary:**
```markdown
## Pairwise Comparisons

| Task | Model A | Model B | Winner | Margin | Confidence |
|---|---|---|---|---|---|
```

4. **Judge Reliability:**
```markdown
## Judge Reliability (Self-Consistency)

| Task | Model | Rounds | Score Std Dev | Consistent? |
|---|---|---|---|---|
```

**Actions:**
- [ ] Add all four report sections
- [ ] Include dimension-level breakdowns
- [ ] Compute self-consistency metrics

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

- [ ] Complex answers receive rubric scores with per-dimension breakdown
- [ ] Pairwise model comparisons can be generated with confidence scores
- [ ] Judge outputs are auditable (raw response, parsed scores, prompt all saved)
- [ ] Judge model can be swapped via config without code changes
- [ ] Self-consistency mode averages multiple judge rounds
- [ ] Human override field stores corrections to judge scores
- [ ] Pairwise results stored in SQLite with dimension-level comparison
- [ ] Markdown report includes judge score tables and pairwise summary
- [ ] `pytest tests/test_llm_judge.py` passes
- [ ] Judge is used only as fallback — exact/executable scorers take precedence

## Estimated Effort

3 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/scorers/llm_judge.py` | To create |
| `src/bench_harness/scorers/pairwise.py` | To create |
| `configs/judge_rubrics.yaml` | To create |
| `configs/judge_prompts/rubric_judge.md` | To create |
| `configs/judge_prompts/pairwise_judge.md` | To create |
| `configs/models.yaml` | Update (judge section) |
| `src/bench_harness/config.py` | Update (judge config loader) |
| `src/bench_harness/storage/sqlite.py` | Update (judge schema migration) |
| `src/bench_harness/storage/artifacts.py` | Update (judge artifacts) |
| `src/bench_harness/cli.py` | Update (judge/pairwise commands) |
| `src/bench_harness/reports/markdown.py` | Update (judge report sections) |
| `tests/test_llm_judge.py` | To create |
