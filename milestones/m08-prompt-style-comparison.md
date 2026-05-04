# Milestone 8 — Prompt Style Comparison

## Goal

Evaluate the impact of REPL mode and other prompting styles on correctness, verbosity, latency, safety, and token use. Run the same tasks with multiple prompt styles and produce a comparison report that recommends the best prompt style per task family.

## Phase

Phase B extension / Phase C prep (bridges real coding usefulness and deep comparison)

## Dependencies

- Milestone 1 (project bootstrap, CLI, storage)
- Milestone 2 (task schema, prompt templates — template system is the foundation)
- Milestone 3 (timing and token metrics — needed for latency/token comparison)
- Milestone 4 (basic scorers — correctness scoring across styles)
- Milestone 5 (coding task runner — code tasks run with different prompts)
- Milestone 6 (local coding-agent suite — task set to test against)
- Milestone 7 (LLM judge — optional for rubric-scored prompt comparison)

---

## Subtasks

### 8.1 Define prompt style variants

**File:** `configs/prompt_styles.yaml` (new)

```yaml
prompt_styles:
  plain:
    name: "Plain Direct"
    description: "Direct question, no special framing"
    template: "configs/prompt_templates/plain.md"
    system_message: null

  repl:
    name: "REPL Mode"
    description: "Hypothesize, test, interpret cycle"
    template: "configs/prompt_templates/repl.md"
    system_message: |
      Follow REPL mode: form a hypothesis, run a single targeted test,
      interpret the result. Never suggest broad destructive commands.
      Always inspect before modifying.

  terse:
    name: "Terse Answer"
    description: "Brief, minimal response"
    template: "configs/prompt_templates/terse.md"
    system_message: |
      Answer briefly and directly. No explanations, no pleasantries,
      no conversational filler. Just the answer.

  patch_only:
    name: "Patch Only"
    description: "Output only a unified diff"
    template: "configs/prompt_templates/patch_only.md"
    system_message: |
      Output ONLY a unified diff. No explanations, no markdown text
      outside the diff. No conversational filler.

  architect:
    name: "Architect Mode"
    description: "Think architecturally first, then implement"
    template: "configs/prompt_templates/architect.md"
    system_message: |
      Think about the architectural implications first. Consider edge
      cases, performance, and maintainability. Then provide the
      implementation.

  json_schema:
    name: "JSON Schema Output"
    description: "Structured JSON output matching a schema"
    template: "configs/prompt_templates/json_schema.md"
    system_message: |
      Output ONLY valid JSON matching the provided schema. No text
      outside the JSON object.

  step_by_step:
    name: "Step-by-Step Plan"
    description: "Plan step by step, then execute"
    template: "configs/prompt_templates/step_by_step.md"
    system_message: |
      First plan your approach step by step. Then execute the plan.
      Explain your reasoning for each step.
```

**Actions:**
- [ ] Write prompt_styles.yaml with all 7 style variants
- [ ] Each style has: name, description, template path, system_message
- [ ] Template files already exist from M2, update content to match style spec

### 8.2 Update prompt template renderer for styles

**File:** `src/bench_harness/tasks/prompt_templates.py` (update)

**New function:**
```python
def render_with_style(
    task: Task,
    style_name: str,
    style_config: dict,
    task_dir: str,
) -> RenderedPrompt:
    """Render a task's prompt with a specific style applied.
    
    Returns RenderedPrompt with system_message, user_message,
    and the full rendered text.
    """
```

**Class:** `RenderedPrompt`

```python
@dataclass
class RenderedPrompt:
    style_name: str
    system_message: str | None
    user_message: str
    full_text: str            # combined for display/logging
    file_context: dict       # {filename: content}
    rendered_tokens_estimate: int | None  # pre-count if tokenizer available
```

**Rendering flow:**
1. Load style config from `prompt_styles.yaml`
2. Load template from `style_config.template`
3. Inject task input (user_message, files, context)
4. Render system message from style config
5. Compute token estimate using fallback tokenizer

**Actions:**
- [ ] Implement `render_with_style()` function
- [ ] Implement `RenderedPrompt` dataclass
- [ ] Support both system message and user message rendering
- [ ] Compute pre-render token estimate

### 8.3 Implement prompt style sweep runner

**File:** `src/bench_harness/runners/style_sweep_runner.py`

**Class:** `StyleSweepRunner`

```python
class StyleSweepRunner:
    """Runs the same set of tasks across multiple prompt styles."""

    def __init__(
        self,
        client: OpenAICompatClient,
        code_runner: CodeRunner | None = None,
        scorers: dict[str, BaseScorer] | None = None,
    ):
        self.client = client
        self.code_runner = code_runner
        self.scorers = scorers or {}

    def run_sweep(
        self,
        tasks: list[Task],
        style_names: list[str],
        style_configs: dict[str, dict],
        task_dir: str,
        params: dict,
    ) -> list[RunResult]:
        """Run each task × style combination and return all results."""

    def run_task_style(
        self,
        task: Task,
        style_name: str,
        style_config: dict,
        task_dir: str,
        params: dict,
    ) -> RunResult:
        """Run a single task with a single prompt style."""
```

**Execution flow:**
1. For each task:
   a. For each style:
      - Render prompt with `render_with_style()`
      - Run completion or code execution
      - Score with configured scorers
      - Capture timing and token metrics
      - Save result with `prompt_style` field
2. Return all results as flat list (task × style matrix)

**Actions:**
- [ ] Implement `StyleSweepRunner` class
- [ ] Implement nested loop over tasks × styles
- [ ] Pass rendered prompt to runner (completion or code)
- [ ] Tag each result with `prompt_style` metadata

### 8.4 Extend RunResult with prompt style fields

**File:** `src/bench_harness/runners/completion_runner.py` (update `RunResult`)

**New fields:**
```python
@dataclass
class RunResult:
    # ... existing fields ...
    prompt_style: str | None = None          # "plain", "repl", "terse", etc.
    prompt_template_hash: str | None = None  # SHA-256 of rendered prompt
    system_message: str | None = None        # system message used
    full_prompt_text: str | None = None      # full rendered prompt for audit
```

**Actions:**
- [ ] Add new fields to `RunResult`
- [ ] Update `CompletionRunner.run()` to accept `RenderedPrompt`
- [ ] Compute prompt template hash from rendered text

### 8.5 Extend SQLite schema for prompt style

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `runs` table:**
```sql
ALTER TABLE runs ADD COLUMN prompt_style TEXT;
ALTER TABLE runs ADD COLUMN prompt_template_hash TEXT;
ALTER TABLE runs ADD COLUMN system_message TEXT;
ALTER TABLE runs ADD COLUMN full_prompt_text TEXT;
```

**New table — `prompt_style_results`:**
```sql
CREATE TABLE prompt_style_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    prompt_style TEXT NOT NULL,
    primary_score REAL,
    total_tokens INTEGER,
    ttft_ms REAL,
    total_wall_ms REAL,
    tokens_per_second REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**Methods (add):**
- `save_prompt_style_result(result: RunResult)`
- `get_style_comparison(
    task_id: str | None = None,
    model_alias: str | None = None,
    family: str | None = None
) -> list[dict]`
- `get_best_style_by_family() -> dict` — returns `{family: {style: avg_score}}`

**Actions:**
- [ ] Implement schema migration
- [ ] Add `prompt_style_results` table
- [ ] Implement save and query methods
- [ ] Add index on `(task_id, model_alias, prompt_style)`

### 8.6 Add CLI support for prompt style sweep

**File:** `src/bench_harness/cli.py` (update)

**New CLI command:**
```
bench_harness sweep-styles \
  --suite <suite_name> \
  --models <model_aliases> \
  --styles plain,repl,terse,patch_only,architect,json_schema,step_by_step \
  --runs 1 \
  --out runs/2026-05-04-style-sweep
```

**New CLI arguments:**
```
--styles          TEXT   Prompt style names, comma-separated (default: "plain,repl")
--sweep           BOOLEAN Enable style sweep mode (auto-set by sweep-styles command)
--style-report    BOOLEAN Generate style comparison report (default: true in sweep mode)
```

**Flow:**
1. Load suite → tasks
2. Load prompt styles from config
3. For each model × task × style:
   - Render prompt with style
   - Run task (completion or code)
   - Score response
   - Save result with style metadata
4. Generate style comparison report

**Actions:**
- [ ] Add `sweep-styles` subcommand
- [ ] Implement style sweep flow in CLI
- [ ] Wire `StyleSweepRunner` into command handler
- [ ] Add `--styles` argument with default

### 8.7 Implement style comparison report

**File:** `src/bench_harness/reports/style_comparison.py` (new)

**Class:** `StyleComparisonReport`

```python
class StyleComparisonReport:
    def __init__(self, results: list[RunResult], db: SQLiteStore):
        self.results = results
        self.db = db

    def generate(self, out_path: str) -> str:
        """Generate full style comparison report in Markdown."""
```

**Report sections:**

1. **Header:** sweep date, models, styles, task count
```markdown
# Prompt Style Comparison Report

Date: 2026-05-04
Models: agent-code, qwen-dense, max-brain
Styles: plain, repl, terse, patch_only, architect, json_schema, step_by_step
Tasks: 25 | Style×Task Runs: 175 per model
```

2. **Correctness by Style (overall):**
```markdown
## Correctness by Prompt Style (Overall)

| Style | agent-code | qwen-dense | max-brain |
|---|---|---|---|
| plain | 0.72 | 0.68 | 0.80 |
| repl | 0.76 | 0.70 | 0.78 |
| terse | 0.65 | 0.64 | 0.74 |
| ... | ... | ... | ... |
```

3. **Correctness by Style × Task Family:**
```markdown
## Correctness by Style and Task Family

### Debugging Tasks
| Style | agent-code | qwen-dense | max-brain |
|---|---|---|---|

### Coding Tasks
| Style | agent-code | qwen-dense | max-brain |
|---|---|---|---|

### Shell Tasks
| Style | agent-code | qwen-dense | max-brain |
|---|---|---|---|
```

4. **Token Usage by Style:**
```markdown
## Token Usage by Style

| Style | Avg Prompt Tok | Avg Completion Tok | Avg Total Tok | Token Increase vs Plain |
|---|---|---|---|---|
| plain | 45 | 120 | 165 | baseline |
| repl | 120 | 280 | 400 | +142% |
| terse | 55 | 45 | 100 | -39% |
| ... | ... | ... | ... | ... |
```

5. **Latency by Style:**
```markdown
## Latency by Style

| Style | Avg TTFT (ms) | Avg Decode (ms) | Avg Wall (ms) |
|---|---|---|---|
```

6. **Safety by Style (if safety scorer available):**
```markdown
## Safety Violations by Style

| Style | Violations | Violation Rate |
|---|---|---|
```

7. **Format Compliance by Style:**
```markdown
## Format Compliance by Style

| Style | Format Valid Rate | Common Failures |
|---|---|---|
```

8. **Quality per Token (efficiency frontier):**
```markdown
## Quality per Token

Score per 100 tokens of total usage:

| Style | agent-code | qwen-dense | max-brain |
|---|---|---|---|
```

9. **Recommended Style per Task Family:**
```markdown
## Recommended Prompt Style per Task Family

| Task Family | Recommended Style | Reason |
|---|---|---|
| debugging | repl | +5.6% correctness, worth token overhead |
| coding | plain | repl adds overhead without quality gain |
| shell | terse | fastest adequate option, -40% tokens |
| planning | architect | significantly better on complex tasks |
```

10. **Where REPL Helps (and Doesn't):**
```markdown
## REPL Mode: Where It Helps

| Task Family | Improvement | Token Cost | Verdict |
|---|---|---|---|
| debugging | +6.2% | +142% | Helpful |
| coding | +0.8% | +138% | Not worth it |
| shell | -1.2% | +150% | Hurts |
```

**Actions:**
- [ ] Implement `StyleComparisonReport` class
- [ ] Generate all 10 report sections
- [ ] Compute cross-style statistics (means, ratios, differences)
- [ ] Generate recommendation table based on highest score per family

### 8.8 Implement style comparison plots

**File:** `src/bench_harness/reports/plots.py`

**Functions:**
```python
def plot_score_by_style(
    results: list[RunResult],
    out_path: str,
) -> str:
    """Grouped bar chart: score by style for each model."""

def plot_tokens_by_style(
    results: list[RunResult],
    out_path: str,
) -> str:
    """Bar chart: average total tokens by style."""

def plot_quality_vs_tokens(
    results: list[RunResult],
    out_path: str,
) -> str:
    """Scatter plot: quality score vs total tokens, colored by style."""

def plot_latency_by_style(
    results: list[RunResult],
    out_path: str,
) -> str:
    """Box plot: wall time distribution by style."""

def plot_style_by_family_heatmap(
    results: list[RunResult],
    out_path: str,
) -> str:
    """Heatmap: rows = styles, columns = task families, cells = avg score."""
```

**Implementation details:**
- Use `matplotlib` for plot generation
- Save as PNG files
- Return file paths for report embedding
- Use consistent color palette for styles

**Actions:**
- [ ] Add `matplotlib` and `seaborn` to `pyproject.toml` dependencies
- [ ] Implement all 5 plot functions
- [ ] Generate plots during style sweep report generation
- [ ] Embed plot references in Markdown report

### 8.9 Add script for style sweep

**File:** `scripts/run_style_sweep.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness sweep-styles \
  --suite local_coding_agent \
  --models agent-code,qwen-dense,max-brain \
  --styles plain,repl,terse,patch_only,architect,json_schema,step_by_step \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-style-sweep"
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 8.10 Add style comparison tests

**File:** `tests/test_prompt_styles.py`

**Tests:**
- `test_prompt_styles_config_loads` — all 7 styles load from YAML
- `test_each_style_has_template` — every style references existing template file
- `test_render_with_style_plain` — plain style renders task prompt
- `test_render_with_style_repl` — REPL style adds system message wrapper
- `test_render_with_style_terse` — terse style includes brevity instruction
- `test_rendered_prompt_has_all_fields` — RenderedPrompt dataclass populated
- `test_prompt_template_hash_deterministic` — same prompt produces same hash
- `test_style_sweep_runner_executes_all_combinations` — tasks × styles matrix complete
- `test_result_has_style_metadata` — RunResult contains prompt_style field
- `test_sqlite_style_results_saved` — prompt_style_results table populated
- `test_style_comparison_query` — get_style_comparison returns correct data
- `test_best_style_by_family` — returns highest-scoring style per family
- `test_token_increase_computed` — token delta vs plain style calculated
- `test_recommendation_generated` — recommendation table has entry per family

**Actions:**
- [ ] Implement all tests
- [ ] Use sample tasks for rendering tests
- [ ] Mock client responses for sweep runner tests
- [ ] Test report generation with synthetic results

---

## Acceptance Criteria Checklist

- [ ] Harness can run the same tasks with all 7 prompt styles (plain, repl, terse, patch_only, architect, json_schema, step_by_step)
- [ ] Each result is tagged with prompt_style, template hash, and system message
- [ ] Correctness, token usage, latency, safety, and format compliance are compared across styles
- [ ] Quality-per-token efficiency metric is computed for each style
- [ ] Report shows where REPL mode helps and where it adds unnecessary overhead
- [ ] Recommended prompt style is reported per task family with justification
- [ ] Style comparison plots are generated (bar charts, scatter, heatmap)
- [ ] `bench_harness sweep-styles` CLI command works end-to-end
- [ ] SQLite stores per-style results for historical comparison
- [ ] `pytest tests/test_prompt_styles.py` passes

## Estimated Effort

2.5–3 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `configs/prompt_styles.yaml` | To create |
| `configs/prompt_templates/plain.md` | Update (M2 created stub) |
| `configs/prompt_templates/repl.md` | Update (M2 created stub) |
| `configs/prompt_templates/terse.md` | Update (M2 created stub) |
| `configs/prompt_templates/patch_only.md` | Update (M2 created stub) |
| `configs/prompt_templates/architect.md` | Update (M2 created stub) |
| `configs/prompt_templates/json_schema.md` | Update (M2 created stub) |
| `configs/prompt_templates/step_by_step.md` | Update (M2 created stub) |
| `src/bench_harness/tasks/prompt_templates.py` | Update (render_with_style) |
| `src/bench_harness/runners/style_sweep_runner.py` | To create |
| `src/bench_harness/runners/completion_runner.py` | Update (style fields in RunResult) |
| `src/bench_harness/storage/sqlite.py` | Update (style schema migration) |
| `src/bench_harness/reports/style_comparison.py` | To create |
| `src/bench_harness/reports/plots.py` | To create |
| `src/bench_harness/reports/markdown.py` | Update (style report integration) |
| `src/bench_harness/cli.py` | Update (sweep-styles command) |
| `scripts/run_style_sweep.sh` | To create |
| `tests/test_prompt_styles.py` | To create |
