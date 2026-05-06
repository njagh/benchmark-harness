# Milestone 9 — Long-context Benchmark Suite

## Goal

Measure quality and performance as prompt size increases, identifying context breakpoints where answer quality or speed degrades for each model.

## Phase

Phase C — Deep comparison (Milestone 1 of 3 in phase)

## Dependencies

- Milestone 2 (task schema — context_tokens field)
- Milestone 3 (timing metrics — prefill, TTFT, decode)
- Milestone 5 (coding task runner — repo tasks for context injection)
- Milestone 8 (prompt style comparison — style control during sweeps)

---

### Leveraged Libraries

- **transformers** (in pyproject.toml optional `[long-context]`): Tokenizer for context packing and token estimation
- **pynvml** (in pyproject.toml optional `[long-context]`): GPU memory monitoring during long-context runs
- **typer** (already in pyproject.toml): CLI for --context-sizes flag
- **rich** (already in pyproject.toml): Terminal output for sweep progress

---

## Subtasks

### 9.1 Implement ContextPacker

**File:** `src/bench_harness/tasks/context_packer.py` (new)

**Class:**
```python
class ContextPacker:
    def __init__(self, tokenizer=None, max_tokens=128000)
    def pack(self, task, files, distractors=[], relevant_position="auto") -> PackedContext
    def add_distractors(self, context, num_distractors=5) -> str
    def apply_relevant_fact_placement(self, context, position="auto") -> str
```

**PackedContext dataclass:**
```python
@dataclass
class PackedContext:
    text: str
    tokens: int
    relevant_section: str
    relevant_position: str  # "beginning", "middle", "end"
    has_distractors: bool
    distractor_count: int
```

**Behavior:**
- Ingests task definition with context files and target facts
- Tokenizes using tokenizer or fallback word-based estimate
- Packs files into context with clear delimiters
- Injects distractor content from provided list or synthetic generation
- Places the relevant fact at beginning/middle/end per placement config
- Returns PackedContext with metadata for scoring

**Actions:**
- [x] Implement ContextPacker with pack(), add_distractors(), apply_relevant_fact_placement()
- [x] Implement PackedContext dataclass with all metadata fields
- [x] Support tokenizer from transformers or fallback word-based estimation
- [x] Support distractor injection with configurable count
- [x] Support relevant fact placement at beginning, middle, or end
- [x] Validate packed context does not exceed max_tokens

### 9.2 Create synthetic long-context tasks

**Directory:** `tasks/synthetic/long_context/`

**Task types:**
1. `fact_retrieval_beginning` — relevant fact in first paragraph
2. `fact_retrieval_middle` — relevant fact buried in middle
3. `fact_retrieval_end` — relevant fact in last paragraph
4. `distractor_resistance` — multiple conflicting facts, latest instruction wins
5. `format_preservation` — long context with specific output format request

**Actions:**
- [x] Define YAML task files for each synthetic task type
- [x] Include target fact, distractor config, placement metadata
- [x] Include expected answer for scoring validation
- [x] Register in task registry under `synthetic/long_context` family

### 9.3 Create repo-derived long-context tasks

**Directory:** `tasks/local/long_context/`

**Task types:**
1. `repo_analysis` — full codebase context with single question
2. `benchmark_review` — long benchmark logs with one regression to find
3. `architecture_change` — long architecture doc with requested modification
4. `docker_multi_service` — large docker-compose with single fix

**Actions:**
- [x] Create tasks that reference multiple context files (full repos or large logs)
- [x] Use context file injection from existing repo_runner infrastructure
- [x] Include specific questions targeting buried information
- [x] Score based on correct retrieval of targeted information

### 9.4 Implement ContextSizeSweepRunner

**File:** `src/bench_harness/runners/context_sweep_runner.py` (new)

**Class:**
```python
class ContextSizeSweepRunner:
    def __init__(self, base_runner, context_sizes=[2000, 8000, 32000, 64000])
    async def run_task_at_size(task, params, model_alias, size_tokens, suite_id) -> RunResult
    async def run_sweep(tasks, model_aliases, params, suite_id) -> list[RunResult]
    @staticmethod
    def get_bucket(size_tokens) -> str  # Maps to 2k, 8k, 32k, 64k, 128k
```

**Behavior:**
- Runs each task at multiple context sizes using ContextPacker
- Tags each RunResult with `context_size` and `context_bucket`
- Tracks per-size prefill time, TTFT, decode time, token counts
- Handles errors gracefully (creates error results)
- Supports partial sweep failure (continues remaining sizes)

**Actions:**
- [x] Implement ContextSizeSweepRunner class
- [x] Implement run_task_at_size() — pack context → runner.run → tag context metadata
- [x] Implement run_sweep() — task × size × model combinations
- [x] Add context bucket mapping (2k, 8k, 32k, 64k, 128k)
- [x] Add error handling with fallback error results
- [x] Add partial failure resilience (continue on size failure)

### 9.5 Add context_tokens field to Task schema and RunResult

**Files:**
- `src/bench_harness/tasks/task_schema.py` (update)
- `src/bench_harness/models/run_result.py` (update)

**Task schema additions:**
```python
class Task(BaseModel):
    ...
    context_tokens: ContextTokensField | None = None  # small, medium, large, huge
```

**RunResult additions:**
```python
class RunResult(BaseModel):
    ...
    context_tokens: int | None = None
    context_size: int | None = None
    context_bucket: str | None = None
```

**Actions:**
- [x] Add context_tokens field to Task schema with allowed values
- [x] Add context_tokens, context_size, context_bucket to RunResult
- [x] Update JSON schema and serialization

### 9.6 Add context tokens to SQLite schema

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Migrations:**
- `runs` table: add `context_tokens INTEGER` column
- `run_timings` table: add `context_tokens INTEGER`, `estimated_prompt_tokens INTEGER` columns
- Safe migration on both fresh and existing databases

**Actions:**
- [x] Add migration for runs table context_tokens column
- [x] Add migration for run_timings table context_tokens and estimated_prompt_tokens columns
- [x] Update get_timing_summary() to include context-level aggregations
- [x] Store context_bucket string in SQLite

### 9.7 Add CLI --context-sizes flag

**File:** `src/bench_harness/cli.py` (update)

**New flag:**
```bash
--context-sizes   TEXT   Context sizes for sweep, comma-separated (default: 2000,8000,32000,64000)
```

**Actions:**
- [x] Add --context-sizes flag to run command
- [x] Parse comma-separated values into list of ints
- [x] Wire ContextSizeSweepRunner when --context-sizes is specified
- [x] Validate sizes are positive integers

### 9.8 Implement context analysis report

**File:** `src/bench_harness/reports/context_analysis.py` (new)

**Function:**
```python
def generate_context_report(runs, suite_id="", prior_runs=None) -> str
```

**Sections:**
1. **Context Length Analysis** — per model: avg prefill, TTFT, decode at each size bucket
2. **Quality vs Length** — primary score by context bucket for each model
3. **Speed vs Length** — tokens/sec by context bucket, degradation percentage
4. **Placement-Aware Retrieval** — score by fact placement (beginning/middle/end)
5. **Breakpoint Detection** — model-specific context size where quality drops >10%
6. **Distractor Impact** — score with vs without distractors per size

**Actions:**
- [x] Implement generate_context_report() with all 6 sections
- [x] Implement _context_bucket_grouping to group runs by bucket
- [x] Implement _detect_breakpoint() to find quality drop >10% threshold
- [x] Implement placement-aware scoring for buried facts
- [x] Implement distractor impact comparison
- [x] Handle runs without context metadata gracefully (return empty string)

### 9.9 Integrate context report into markdown report

**File:** `src/bench_harness/reports/markdown.py` (update)

**Changes:**
- Add `_append_context_analysis(lines, runs)` helper
- Detect runs with context_size or context_bucket metadata
- Call generate_context_report() when context metadata is present
- Append after coding agent ranking section

**Actions:**
- [x] Add _append_context_analysis() helper function
- [x] Import generate_context_report in markdown.py
- [x] Integrate into generate_report() flow
- [x] Auto-detect context metadata and append sections

### 9.10 Add comprehensive tests

**File:** `tests/test_context_packer.py` (new)
**File:** `tests/test_context_sweep.py` (new)

**Test categories:**
- ContextPacker: pack with files, add_distractors, apply_relevant_fact_placement
- ContextPacker: token estimation accuracy, max_tokens enforcement
- ContextPacker: distractor injection with configurable count
- ContextPacker: relevant fact placement at beginning/middle/end
- ContextSizeSweepRunner: initialization, run_task_at_size, run_sweep
- ContextSizeSweepRunner: bucket mapping (2000→2k, 8000→8k, 32000→32k, 64000→64k)
- ContextSizeSweepRunner: error handling and partial failure resilience
- Context analysis report: all 6 sections present
- Context analysis report: breakpoint detection with known degradation
- RunResult context field serialization
- CLI --context-sizes parsing
- SQLite schema migrations for context columns

**Actions:**
- [x] Test ContextPacker with synthetic and repo context
- [x] Test add_distractors with multiple distractor counts
- [x] Test apply_relevant_fact_placement for all positions
- [x] Test ContextSizeSweepRunner task×size×model combinations
- [x] Test bucket mapping for standard sizes
- [x] Test context analysis report with sample data
- [x] Test breakpoint detection with artificial degradation curve
- [x] Test RunResult context field serialization to JSON
- [x] Test CLI --context-sizes comma-separated parsing
- [x] Test SQLite migrations for context columns

---

## Acceptance Criteria Checklist

- [x] ContextPacker successfully packs tasks with context files and injects distractors
- [x] Relevant fact placement works for beginning, middle, and end positions
- [x] Synthetic long-context tasks defined in tasks/synthetic/long_context/
- [x] Repo-derived long-context tasks defined in tasks/local/long_context/
- [x] ContextSizeSweepRunner runs task × size × model combinations
- [x] Context bucket mapping produces 2k, 8k, 32k, 64k, 128k labels
- [x] Same task can run at 2k, 8k, 32k, 64k tokens via context bucket mapping
- [x] context_tokens field added to Task schema and RunResult
- [x] SQLite runs and run_timings tables have context_tokens columns (safe migration)
- [x] Prefill, TTFT, and decode degradation tracked across context sizes
- [x] CLI --context-sizes flag parses and activates sweep mode
- [x] Context Length Analysis report section shows timing by bucket
- [x] Quality vs Length report section shows score degradation by bucket
- [x] Breakpoint Detection identifies model-specific context limits (quality drop >10%)
- [x] Placement-aware scoring evaluates buried fact retrieval
- [x] Distractor impact analysis shows quality difference with/without noise
- [x] Markdown report auto-detects context metadata and appends analysis sections
- [x] All tests pass (context packer, sweep runner, report generation, integration)

---

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/tasks/context_packer.py` | Created (ContextPacker class, PackedContext dataclass) |
| `src/bench_harness/runners/context_sweep_runner.py` | Created (ContextSizeSweepRunner class) |
| `src/bench_harness/reports/context_analysis.py` | Created (generate_context_report with 6 sections) |
| `src/bench_harness/tasks/task_schema.py` | Updated (context_tokens field on Task) |
| `src/bench_harness/models/run_result.py` | Updated (context_tokens, context_size, context_bucket) |
| `src/bench_harness/storage/sqlite.py` | Updated (runs/run_timings migrations for context columns) |
| `src/bench_harness/cli.py` | Updated (--context-sizes flag support) |
| `src/bench_harness/reports/markdown.py` | Updated (_append_context_analysis integration) |
| `tasks/synthetic/long_context/` | Created (synthetic task definitions) |
| `tasks/local/long_context/` | Created (repo-derived task definitions) |
| `tests/test_context_packer.py` | Created (comprehensive packer tests) |
| `tests/test_context_sweep.py` | Created (sweep runner and report tests) |
| `ROADMAP.md` | Updated (M9 marked done with detailed acceptance criteria) |
| `README.md` | Updated (status line and milestone table) |
