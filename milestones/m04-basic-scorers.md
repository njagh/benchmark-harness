# Milestone 4 — Basic Scorers

## Goal

Support automatic deterministic scoring for simple tasks: exact match, multiple-choice, regex, JSON schema, contains/does-not-contain, and format compliance. Scorers are tested, stored in run records, and visible in reports.

## Phase

Phase B — Real coding usefulness (Milestone 2 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap, runner, storage)
- Milestone 2 (task schema with scoring metadata)
- Milestone 3 (timing metrics — needed for complete run records)

---

## Subtasks

### 4.1 Implement base scorer interface

**File:** `src/bench_harness/scorers/base.py`

**Class:** `BaseScorer`

```python
class BaseScorer(ABC):
    """Base class for all scorers."""

    name: str = "base"
    version: str = "1.0"

    @abstractmethod
    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Score a single response against a task's expected output."""

    def validate_task(self, task: Task) -> bool:
        """Check if this scorer is appropriate for the task's expected output type."""
        return True
```

**Class:** `ScoreResult`

```python
@dataclass
class ScoreResult:
    scorer_name: str
    scorer_version: str
    score: float                  # 0.0–1.0
    passed: bool                  # boolean pass/fail
    details: dict                 # scorer-specific breakdown
    explanation: str | None = None  # human-readable reason
    duration_ms: float | None = None
```

**Scorer registry:**
```python
_SCORERS: dict[str, type[BaseScorer]] = {}

def register_scorer(cls: type[BaseScorer]):
    """Register a scorer by its `name` attribute."""

def get_scorer(name: str) -> BaseScorer:
    """Look up and instantiate a scorer by name."""

def list_scorers() -> list[str]:
    """Return all registered scorer names."""
```

**Actions:**
- [ ] Implement `BaseScorer` ABC
- [ ] Implement `ScoreResult` dataclass
- [ ] Implement scorer registry with auto-registration via decorator
- [ ] Add `score_all(task, response, scorer_names)` helper that runs multiple scorers

### 4.2 Implement exact match scorer

**File:** `src/bench_harness/scorers/exact_match.py`

**Class:** `ExactMatchScorer(BaseScorer)`

```python
class ExactMatchScorer(BaseScorer):
    name = "exact_match"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Exact string match after stripping whitespace."""
```

**Behavior:**
- Strips leading/trailing whitespace from both expected answer and response
- Case-sensitive by default
- Supports `case_insensitive: true` in task's `expected` section
- Extracts the expected answer from `task.expected.answer`

**Task YAML example:**
```yaml
expected:
  type: exact
  answer: "Paris"
scoring:
  primary: exact_match
```

**Actions:**
- [ ] Implement scorer with whitespace normalization
- [ ] Support `case_insensitive` option
- [ ] Register with `@register_scorer` decorator

### 4.3 Implement multiple-choice scorer

**File:** `src/bench_harness/scorers/multiple_choice.py`

**Class:** `MultipleChoiceScorer(BaseScorer)`

```python
class MultipleChoiceScorer(BaseScorer):
    name = "multiple_choice"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Match response against a set of choices."""
```

**Behavior:**
- Task provides choices: `expected.choices: [A, B, C, D]` with `expected.answer: "B"`
- Extracts the model's selected answer by looking for the choice letter or text
- Supports formats: "B", "(B)", "Answer: B", "The answer is B"
- Returns partial credit if multiple-choice with `partial_credit: true`

**Task YAML example:**
```yaml
expected:
  type: multiple_choice
  choices:
    A: "Paris"
    B: "London"
    C: "Berlin"
    D: "Madrid"
  answer: "A"
scoring:
  primary: multiple_choice
```

**Actions:**
- [ ] Implement choice extraction logic with regex patterns
- [ ] Support letter-only, parenthesized, and verbose answer formats
- [ ] Add `details` with extracted answer and confidence

### 4.4 Implement regex scorer

**File:** `src/bench_harness/scorers/regex.py`

**Class:** `RegexScorer(BaseScorer)`

```python
class RegexScorer(BaseScorer):
    name = "regex"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Match one or more regex patterns against the response."""
```

**Behavior:**
- `expected.patterns` is a list of regex patterns
- Default mode: all patterns must match (AND)
- `mode: any` — at least one pattern must match (OR)
- `mode: all` — all patterns must match (AND, default)
- `mode: none` — none of the patterns should match (anti-patterns)
- Score is fraction of patterns matched

**Task YAML example:**
```yaml
expected:
  type: regex
  patterns:
    - "def fibonacci"
    - "return.*\\+"
  mode: all
scoring:
  primary: regex
```

**Actions:**
- [ ] Implement pattern matching with `re.search()`
- [ ] Support all three modes (all/any/none)
- [ ] Return partial score as fraction of patterns matched
- [ ] Include matched patterns in `details`

### 4.5 Implement JSON schema scorer

Use the `jsonschema` library (already in pyproject.toml) rather than hand-rolling JSON validation. The `jsonschema.validate()` and `jsonschema.Draft202012Validator` classes provide standard-compliant validation with detailed error messages.

**File:** `src/bench_harness/scorers/json_schema.py`

**Class:** `JsonSchemaScorer(BaseScorer)`

```python
class JsonSchemaScorer(BaseScorer):
    name = "json_schema"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Validate response against a JSON schema."""
```

**Behavior:**
- Extracts JSON from the response (handles markdown code fences, bare JSON, partial JSON)
- Validates against `task.expected.schema` using `jsonschema` library
- Checks required fields, types, and constraints
- `strict: true` — reject if extra fields present
- `repair: true` — attempt JSON repair before validation (score reduced if repair was needed)

**JSON extraction strategy:**
1. Look for `````json ... ``` ``` blocks
2. Look for ````` ... ``` ``` blocks
3. Look for `{ ... }` or `[ ... ]` at top level
4. Try to find balanced braces/brackets
5. Fall back to full response as JSON

**Task YAML example:**
```yaml
expected:
  type: json_schema
  schema:
    type: object
    required: [name, language, year_created]
    properties:
      name: {type: string}
      language: {type: string}
      year_created: {type: integer}
  strict: false
  repair: true
scoring:
  primary: json_schema
```

**Actions:**
- [ ] Add `jsonschema` to `pyproject.toml` dependencies
- [ ] Implement JSON extraction with all fallback strategies
- [ ] Implement schema validation
- [ ] Implement repair attempt with score penalty
- [ ] Include validation errors in `details`

### 4.6 Implement contains scorer

**File:** `src/bench_harness/scorers/contains.py`

**Class:** `ContainsScorer(BaseScorer)`

```python
class ContainsScorer(BaseScorer):
    name = "contains"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Check if response contains (or does not contain) specific strings."""
```

**Behavior:**
- `expected.patterns` — list of strings that must appear (substring match, not regex)
- `expected.absent_patterns` — list of strings that must NOT appear
- `case_insensitive: true` — case-insensitive matching
- Score is fraction of required patterns found, penalized for absent patterns found

**Task YAML example:**
```yaml
expected:
  type: contains
  patterns:
    - "a + b"
    - "def add"
  absent_patterns:
    - "a - b"
  case_insensitive: false
scoring:
  primary: contains
```

**Actions:**
- [ ] Implement substring matching
- [ ] Support both positive and negative patterns
- [ ] Compute composite score from presence/absence

### 4.7 Implement format compliance scorer

**File:** `src/bench_harness/scorers/format_compliance.py`

**Class:** `FormatComplianceScorer(BaseScorer)`

```python
class FormatComplianceScorer(BaseScorer):
    name = "format_compliance"

    def score(self, task: Task, raw_response: str) -> ScoreResult:
        """Check if response follows requested output format."""
```

**Behavior:**
- `expected.format_checks` — list of format check types:
  - `starts_with_numbered_list` — response starts with `1.` or `1)`
  - `is_markdown_table` — response contains a valid markdown table
  - `is_valid_json` — response is parseable as JSON
  - `is_code_block` — response contains a fenced code block
  - `line_count_max: N` — response has at most N lines
  - `line_count_min: N` — response has at least N lines
  - `no_conversational filler` — response does not start with "Sure!", "Of course!", etc.
  - `ends_with_keyword: X` — response ends with specific keyword

**Task YAML example:**
```yaml
expected:
  type: format
  format_checks:
    - starts_with_numbered_list
    - item_count: 3
    - no_conversational_filler
scoring:
  primary: format_compliance
```

**Actions:**
- [ ] Implement each format check as a callable
- [ ] Support `format_checks` with arguments (e.g., `line_count_max: 10`)
- [ ] Return per-check pass/fail in `details`

### 4.8 Wire scorers into runner

**File:** `src/bench_harness/runners/completion_runner.py` (update)

**Updates to `RunResult`:**
```python
@dataclass
class RunResult:
    # ... existing fields ...
    score_primary: float | None = None
    score_secondary: dict | None = None    # {scorer_name: ScoreResult}
    scorer_version: str | None = None
    score_explanation: str | None = None
```

**Updates to `CompletionRunner.run()`:**
1. After obtaining `raw_response`, look up `task.scoring.primary`
2. Call `get_scorer(primary_name).score(task, raw_response)`
3. Store result in `score_primary`
4. If `task.scoring.secondary` is set, run each secondary scorer
5. Store all results in `score_secondary`

**Actions:**
- [ ] Update `RunResult` with scoring fields
- [ ] Update `CompletionRunner.run()` to invoke scorers
- [ ] Handle scorer errors gracefully (log error, set score to `None`)

### 4.9 Update SQLite schema for scoring

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `runs` table:**
```sql
ALTER TABLE runs ADD COLUMN score_primary REAL;
ALTER TABLE runs ADD COLUMN score_secondary TEXT;   -- JSON-encoded dict
ALTER TABLE runs ADD COLUMN scorer_version TEXT;
ALTER TABLE runs ADD COLUMN score_explanation TEXT;
ALTER TABLE runs ADD COLUMN format_valid INTEGER;    -- boolean as 0/1
```

**New table — `score_details`:**
```sql
CREATE TABLE score_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    scorer_name TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    details_json TEXT,
    explanation TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**Methods (add):**
- `save_score_details(run_id: str, score_results: dict[str, ScoreResult])` — saves all scorer details
- `get_scores(suite_id: str, model_alias: str) -> list[dict]` — retrieves scores with details

**Actions:**
- [ ] Implement schema migration
- [ ] Add `score_details` table
- [ ] Implement save/retrieve methods
- [ ] Serialize `score_secondary` as JSON string in `runs` table

### 4.10 Update Markdown report with scoring

**File:** `src/bench_harness/reports/markdown.py` (update)

**Updated summary table:**
```markdown
| Model | Tasks Run | Passed | Failed | Avg Score | Avg TTFT (ms) |
|---|---|---|---|---|---|
```

**New per-task scoring table:**
```markdown
| Task | Model | Primary Score | Secondary Scores | Explanation |
|---|---|---|---|---|
```

**New format compliance section:**
```markdown
## Format Compliance

| Model | Format Valid | Format Failed Tasks |
|---|---|---|
```

**Actions:**
- [ ] Update summary table to include score column
- [ ] Add per-task scoring table
- [ ] Add format compliance summary
- [ ] Compute pass/fail from `score_primary >= 0.95` (configurable threshold)

### 4.11 Define scorer configuration

**File:** `configs/scorers.yaml` (update from stub)

```yaml
scorers:
  exact_match:
    class: "bench_harness.scorers.exact_match.ExactMatchScorer"
    default_threshold: 1.0

  multiple_choice:
    class: "bench_harness.scorers.multiple_choice.MultipleChoiceScorer"
    default_partial_credit: false

  regex:
    class: "bench_harness.scorers.regex.RegexScorer"
    default_mode: "all"

  json_schema:
    class: "bench_harness.scorers.json_schema.JsonSchemaScorer"
    default_strict: false
    default_repair: true
    repair_score_penalty: 0.5

  contains:
    class: "bench_harness.scorers.contains.ContainsScorer"
    default_case_insensitive: false

  format_compliance:
    class: "bench_harness.scorers.format_compliance.FormatComplianceScorer"
```

**Actions:**
- [ ] Write scorers.yaml with all scorer configurations
- [ ] Load scorer config in `config.py`
- [ ] Pass scorer config to scorer constructors

### 4.12 Add scorer unit tests

**File:** `tests/test_scorers.py`

**Tests:**
- `test_exact_match_pass` — exact string match passes
- `test_exact_match_whitespace` — whitespace differences handled
- `test_exact_match_case_insensitive` — case-insensitive option works
- `test_exact_match_fail` — different strings fail
- `test_multiple_choice_letter` — "A" is detected
- `test_multiple_choice_verbose` — "The answer is A" is detected
- `test_multiple_choice_wrong` — wrong choice fails
- `test_regex_all_mode` — all patterns must match
- `test_regex_any_mode` — any pattern matches
- `test_regex_none_mode` — no patterns match
- `test_json_schema_valid` — valid JSON passes schema
- `test_json_schema_invalid` — invalid JSON fails
- `test_json_schema_extract_from_fences` — JSON in markdown fences extracted
- `test_json_schema_repair` — repairable JSON passes with penalty
- `test_contains_pass` — required strings present
- `test_contains_absent_violation` — forbidden strings detected
- `test_format_numbered_list` — numbered list detected
- `test_format_line_count` — line count checks work
- `test_format_no_filler` — conversational filler detected
- `test_scorer_registry` — all scorers register and resolve by name
- `test_score_result_serialization` — ScoreResult fields serialize correctly

**Actions:**
- [ ] Implement all tests with pytest
- [ ] Use parametrized tests for multiple input variants
- [ ] Ensure tests are deterministic (no network, no randomness)

### 4.13 Note on IFEval integration

Google's IFEval benchmark (https://github.com/google-research/google-research/tree/master/instruction_following_eval) provides a standardized instruction-following test suite. Rather than reimplementing IFEval's prompt templates and scoring logic, the harness should:

1. Download IFEval data once to `/mnt/datasets-big/evals/ifeval_v1/`
2. Run IFEval as a suite using the harness's runner infrastructure
3. Store IFEval results in the harness SQLite database alongside local tasks
4. Use IFEval's own prompt format but score through the harness's scorers

This avoids duplicating IFEval's task definitions while keeping results in a unified database.

**Deferred to:** Milestone 6 (public benchmark integration)

---

## Acceptance Criteria Checklist

- [ ] All six scorers (exact_match, multiple_choice, regex, json_schema, contains, format_compliance) are implemented and registered
- [ ] Smoke tasks produce real scores with all configured scorers
- [ ] Format failures are visible in the Markdown report
- [ ] Scorers are deterministic and fully tested
- [ ] Score results are stored in SQLite with details
- [ ] Runner invokes scorers automatically based on task YAML scoring config
- [ ] `pytest tests/test_scorers.py` passes with 100% coverage on scorer modules

## Estimated Effort

2.5–3 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/scorers/base.py` | To create |
| `src/bench_harness/scorers/exact_match.py` | To create |
| `src/bench_harness/scorers/multiple_choice.py` | To create |
| `src/bench_harness/scorers/regex.py` | To create |
| `src/bench_harness/scorers/json_schema.py` | To create |
| `src/bench_harness/scorers/contains.py` | To create |
| `src/bench_harness/scorers/format_compliance.py` | To create |
| `configs/scorers.yaml` | Update (full config) |
| `src/bench_harness/runners/completion_runner.py` | Update (score integration) |
| `src/bench_harness/storage/sqlite.py` | Update (score schema migration) |
| `src/bench_harness/reports/markdown.py` | Update (score report sections) |
| `src/bench_harness/config.py` | Update (load scorer config) |
| `tests/test_scorers.py` | To create |
