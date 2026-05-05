# Milestone 8 — Prompt Style Comparison

## Goal

Evaluate the impact of REPL mode and other prompting styles (terse, patch_only, architect, json_schema, step_by_step, plain) on correctness, verbosity, latency, and token efficiency across benchmark tasks.

## Phase

Phase B — Coding usefulness (Milestone 4 of 4 in phase)

## Dependencies

- Milestone 2 (task schema, prompt template renderer with Jinja2)
- Milestone 3 (timing metrics — TTFT, decode time, tokens/sec)
- Milestone 4 (basic scorers — compare scores across styles)
- Milestone 6 (local coding-agent suite — real tasks for style comparison)
- Milestone 7 (LLM judge integration — judge-scored comparison)

---

### Leveraged Libraries

- **jinja2** (already in pyproject.toml): Prompt template rendering, used by M2
- **typer** (already in pyproject.toml): CLI for sweep-styles command
- **rich** (already in pyproject.toml): Terminal output for sweep progress

---

## Subtasks

### 8.1 Implement prompt template variants

**File:** `src/bench_harness/tasks/prompt_templates.py` (update)

**Built-in styles:**
- `plain` — direct answer, no instructions
- `repl` — hypothesize a single test, run it, interpret
- `terse` — brief and direct
- `patch_only` — unified diff, no explanation
- `architect` — analyze, outline, implement
- `json_schema` — valid JSON matching schema
- `step_by_step` — plan step by step, then execute

**Actions:**
- [x] Verify all 7 inline templates in INLINE_TEMPLATES dict
- [x] Verify `render_template()` handles file_context injection as markdown code blocks
- [x] Verify `load_prompt_template()` loads from inline AND file-based templates
- [x] Verify `build_prompt()` applies style wrapper to task prompts

### 8.2 Add RenderedPrompt dataclass and render_with_style()

**File:** `src/bench_harness/tasks/prompt_templates.py` (update)

**New types:**
```python
@dataclass
class RenderedPrompt:
    style_name: str
    system_message: str | None
    user_message: str
    full_text: str
    referenced_file_paths: list[str]
    estimated_tokens: int

def render_with_style(task, style, base_dir=None) -> RenderedPrompt:
```

**Actions:**
- [x] Add RenderedPrompt dataclass with all fields
- [x] Add render_with_style() that builds styled prompt + metadata
- [x] Add _estimate_tokens() helper using word splitting
- [x] Verify estimated_tokens is reasonable (positive integer)

### 8.3 Implement StyleSweepRunner

**File:** `src/bench_harness/runners/style_sweep_runner.py`

**Class:**
```python
class StyleSweepRunner:
    def __init__(self, base_runner: CompletionRunner, styles, default_style="plain")
    async def run_task_with_style(task, params, style, suite_id) -> RunResult
    async def run_sweep(tasks, model_aliases, params, suite_id) -> list[RunResult]
```

**Behavior:**
- Runs each task with each prompt style
- Uses `build_prompt()` to render styled prompts
- Tags each `RunResult` with `prompt_style` field
- Handles errors gracefully (creates error results)
- Unknown style names fall back to plain via `build_prompt`

**Actions:**
- [x] Implement StyleSweepRunner class
- [x] Implement run_task_with_style() — build_prompt → runner.run → tag prompt_style
- [x] Implement run_sweep() — task × style × model combinations
- [x] Add error handling with fallback error results

### 8.4 Add CLI support for --styles

**File:** `src/bench_harness/cli.py` (update)

**New/updated CLI flags:**
```bash
--styles    TEXT   Prompt style(s) to sweep, comma-separated (default: plain)
```

**Actions:**
- [x] Add --styles flag to run command (comma-separated string → list)
- [x] Wire StyleSweepRunner into the run command when multiple styles specified
- [x] Test CLI parsing: comma-separated → list with stripping

### 8.5 Implement style comparison report

**File:** `src/bench_harness/reports/style_comparison.py` (new)

**Function:**
```python
def generate_style_report(runs, suite_id="", styles=None) -> str
```

**Sections:**
1. **Style Comparison Summary** — per style: avg score, avg tokens, avg wall time, task count
2. **Per-Task Style Breakdown** — for each task, compare all styles side by side
3. **Best Style Per Family** — which style performs best per task family
4. **Verbosity Analysis** — token usage per style (score/token ratio, quality per 1k tokens)
5. **Latency Comparison** — TTFT and decode time per style
6. **Recommended Style** — overall recommendation based on score/token ratio

**Actions:**
- [x] Implement generate_style_report() with all 6 sections
- [x] Implement helper functions: _avg, _stddev, _extract_family_from_task_id, _score_values
- [x] Implement _append_style_summary with table
- [x] Implement _append_per_task_breakdown with multi-style comparison
- [x] Implement _append_best_style_family with per-family best style
- [x] Implement _append_verbosity_analysis with score/token ratio
- [x] Implement _append_latency_comparison with TTFT, decode, p95 wall
- [x] Implement _append_recommended_style with score/token ranking
- [x] Handle empty/unscored runs gracefully (return empty string)

### 8.6 Integrate style comparison into markdown report

**File:** `src/bench_harness/reports/markdown.py` (update)

**Changes:**
- Add `_append_style_comparison(lines, runs)` helper
- Detect runs with prompt_style metadata
- Call `generate_style_report()` when style metadata is detected
- Append after "Coding Agent Ranking" and before "Failures" (after judge sections)

**Actions:**
- [x] Add _append_style_comparison() helper function
- [x] Import generate_style_report in markdown.py
- [x] Integrate into generate_report() flow

### 8.7 Add comprehensive tests

**File:** `tests/test_prompt_styles.py` (new)

**Test categories:**
- Prompt template rendering for all 7 styles
- `render_template()` with various contexts and file injection
- `load_prompt_template()` inline and file-based loading
- `build_prompt()` with plain, repl, architect styles
- `RenderedPrompt` dataclass fields and `_estimate_tokens()`
- `render_with_style()` returns correct fields
- `StyleSweepRunner` with mocked base runner (task×style×model)
- StyleSweepRunner error handling and unknown style fallback
- Style comparison report generation (all 6 sections)
- Report with multi-family, multi-style sample data
- RunResult prompt_style serialization
- CLI --styles parsing (comma-separated → list)
- Markdown report integration with style sections

**Actions:**
- [x] Test all 7 inline templates render correctly
- [x] Test file context injection as markdown code blocks
- [x] Test load_prompt_template inline and file-based
- [x] Test build_prompt with plain, repl, architect
- [x] Test RenderedPrompt fields and _estimate_tokens
- [x] Test render_with_style for plain and repl
- [x] Test StyleSweepRunner: initialization, run_task_with_style, run_sweep
- [x] Test StyleSweepRunner error handling and unknown style fallback
- [x] Test style comparison report: all 6 sections present
- [x] Test style comparison report: multi-family data
- [x] Test RunResult prompt_style field
- [x] Test CLI --styles comma-separated parsing
- [x] Test markdown report integration (with/without prompt_style)

---

## Acceptance Criteria Checklist

- [x] All 7 prompt template styles (plain, repl, terse, patch_only, architect, json_schema, step_by_step) render correctly with context variables
- [x] File context is injected as markdown code blocks in rendered prompts
- [x] load_prompt_template supports both inline built-in and file-based templates
- [x] RenderedPrompt dataclass has all required fields (style_name, system_message, user_message, full_text, referenced_file_paths, estimated_tokens)
- [x] render_with_style() returns RenderedPrompt with correct metadata
- [x] estimated_tokens is a reasonable positive integer
- [x] StyleSweepRunner runs each task with each prompt style
- [x] StyleSweepRunner tags each result with prompt_style field
- [x] StyleSweepRunner handles unknown style names gracefully (fallback to plain)
- [x] StyleSweepRunner error handling creates error results without crashing
- [x] CLI --styles flag parses comma-separated values into list
- [x] Style comparison report has all 6 sections (summary, per-task, per-family, verbosity, latency, recommendation)
- [x] Per-task breakdown only shows tasks with multiple styles
- [x] Best style per family correctly identifies highest-scoring style
- [x] Verbosity analysis includes score/token ratio metric
- [x] Latency comparison includes TTFT, decode, and p95 wall time
- [x] Recommended style is based on score/token ratio ranking
- [x] Markdown report auto-detects prompt_style metadata and appends style comparison
- [x] RunResult supports prompt_style attribute for sweep results
- [x] RunResult with prompt_style serializes to JSON correctly
- [x] All tests pass (prompt template, build_prompt, StyleSweepRunner, report generation, integration)
- [x] REPL mode helps documented (higher scores on debugging tasks)
- [x] REPL mode overhead documented (lower score/token ratio on simple tasks)

---

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/tasks/prompt_templates.py` | Updated (RenderedPrompt, render_with_style, _estimate_tokens) |
| `src/bench_harness/runners/style_sweep_runner.py` | Created (StyleSweepRunner class) |
| `src/bench_harness/reports/style_comparison.py` | Created (generate_style_report with 6 sections) |
| `src/bench_harness/reports/markdown.py` | Updated (_append_style_comparison integration) |
| `src/bench_harness/cli.py` | Updated (--styles flag support) |
| `tests/test_prompt_styles.py` | Created (comprehensive test suite) |
| `ROADMAP.md` | Updated (M8 marked done with detailed acceptance criteria) |
| `README.md` | Updated (status line and milestone table) |
