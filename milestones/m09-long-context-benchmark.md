# Milestone 9 — Long-Context Benchmark Suite

## Goal

Measure quality and performance as prompt size increases, producing degradation curves and identifying model-specific context breakpoints.

## Phase

Phase C — Deep model/backend comparison (Milestone 1 of 4 in phase)

## Dependencies

- Milestone 1: CLI runner, SQLite storage, Markdown report
- Milestone 3: Timing and token metrics (TTFT, decode time, tokens/sec)
- Milestone 4: Basic scorers (exact, regex, JSON schema)
- Milestone 5: Coding task runner (for code-in-context tasks)
- Milestone 8: Prompt style comparison (prompt template infrastructure)

---

## Subtasks

### 9.1 Design long-context task schema

**File:** `src/bench_harness/tasks/task_schema.py` (extend existing)

**New fields in task YAML for long-context tasks:**

```yaml
id: longcontext.buried_fact_early_001
family: long_context
category: needle_in_haystack
source: synthetic
context:
  target_size_tokens: 32000
  needle_position: early    # early | middle | late | multi
  needle: "The secret code is ALPHA-7742"
  needle_question: "What is the secret code mentioned in the document?"
  filler_source: repo_code   # repo_code | benchmark_logs | architecture_doc | generic
  distractor_count: 5
  distractor_type: conflicting_facts  # conflicting_facts | irrelevant_code | noise
prompt_template: long_context_needle.md
expected:
  type: exact
  answer: "ALPHA-7742"
scoring:
  primary: exact_match
  secondary:
    - format_compliance
    - hallucination_flag
risk_level: low
context_tokens: large
```

**Data class extension in `src/bench_harness/tasks/task_schema.py`:**

```python
@dataclass
class LongContextTaskConfig:
    target_size_tokens: int
    needle_position: Literal["early", "middle", "late", "multi"]
    needle: str
    needle_question: str
    filler_source: str
    distractor_count: int = 0
    distractor_type: str = "irrelevant_code"
```

**Actions:**
- [ ] Add `LongContextTaskConfig` dataclass
- [ ] Extend task YAML schema validation to accept `context` field
- [ ] Add `category` and `context_tokens` fields to base task schema

### 9.2 Implement context packer

**File:** `src/bench_harness/tasks/context_packer.py`

**Class:** `ContextPacker`

**Methods:**
```python
class ContextPacker:
    """Builds long-context prompts by inserting needles into filler content."""

    def __init__(self, filler_dirs: list[str], tokenizer_name: str = "Qwen/Qwen2.5-7B"):
        self.filler_dirs = filler_dirs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_fillers(self, source: str) -> list[str]:
        """Load filler content from repo_code, benchmark_logs, or architecture_doc."""

    def count_tokens(self, text: str) -> int:
        """Count tokens using the loaded tokenizer."""

    def truncate_to_size(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target_tokens tokens."""

    def build_context(
        self,
        needle: str,
        needle_position: str,
        filler_source: str,
        target_size_tokens: int,
        distractors: list[str] | None = None,
    ) -> str:
        """
        Build a long-context prompt with needle at specified position.
        Returns full prompt string ready for model input.
        """
```

**Position logic:**
- `early`: needle placed in first 10% of tokens
- `middle`: needle placed at 40-60% of tokens
- `late`: needle placed in last 10% of tokens
- `multi`: needle repeated at early, middle, and late positions with conflicting values

**Actions:**
- [ ] Implement `ContextPacker` class
- [ ] Load filler content from `tasks/local/` source repos
- [ ] Implement token-aware truncation and padding
- [ ] Implement distractor insertion at random offsets
- [ ] Add caching: save packed contexts to `tasks/long_context/packed/` to avoid recomputation

### 9.3 Implement distractor generator

**File:** `src/bench_harness/tasks/distractor_generator.py`

**Class:** `DistractorGenerator`

**Methods:**
```python
class DistractorGenerator:
    """Generates misleading context entries to test distractor resistance."""

    def __init__(self):
        pass

    def generate_conflicting_facts(self, needle: str, count: int) -> list[str]:
        """Generate plausible but wrong alternatives to the needle value."""

    def generate_irrelevant_code_blocks(self, source_repo: str, count: int) -> list[str]:
        """Extract unrelated code blocks to pad context."""

    def insert_distractors(
        self,
        context: str,
        distractors: list[str],
        strategy: str = "scattered",  # scattered | clustered
    ) -> str:
        """Insert distractors into context at specified positions."""
```

**Distractor categories:**
```
conflicting_facts   — alternate values for the same question
irrelevant_code     — unrelated code blocks from the repo
noise               — random but plausible-looking documentation text
```

**Actions:**
- [ ] Implement distractor generation for each category
- [ ] Implement insertion strategies (scattered vs clustered)
- [ ] Ensure distractors are token-counted so total context size remains stable

### 9.4 Create synthetic long-context tasks

**Directory:** `tasks/synthetic/context_distraction/`

**Task variants (at minimum 7 task templates × 6 context sizes = 42 task instances per model):**

| Task Template ID | Needle Position | Distractor Type | Description |
|---|---|---|---|
| `longcontext.needle_early` | early | none | Relevant fact buried early in context |
| `longcontext.needle_middle` | middle | none | Relevant fact buried in middle |
| `longcontext.needle_late` | late | none | Relevant fact buried late (last 10%) |
| `longcontext.needle_multi_conflict` | multi | conflicting_facts | Multiple conflicting facts, latest wins |
| `longcontext.needle_with_irrelevant_code` | middle | irrelevant_code | Large irrelevant codebase with one relevant file |
| `longcontext.needle_in_logs` | late | noise | Long benchmark logs with one important regression |
| `longcontext.needle_in_archdoc` | early | irrelevant_code | Long architecture doc with requested change |

**Context sizes:** `2000`, `8000`, `32000`, `64000`, `128000`, `200000` (where supported)

**Example task file:** `tasks/synthetic/context_distraction/needle_early_001.yaml`
```yaml
id: longcontext.needle_early_001
family: long_context
category: needle_in_haystack
source: synthetic
context:
  target_size_tokens: 32000
  needle_position: early
  needle: "The API key prefix is XN-8847-PROD"
  needle_question: "What is the API key prefix?"
  filler_source: repo_code
  distractor_count: 0
prompt_template: long_context_needle.md
expected:
  type: exact
  answer: "XN-8847-PROD"
scoring:
  primary: exact_match
  secondary:
    - format_compliance
    - hallucination_flag
```

**Actions:**
- [ ] Create 7 synthetic task template YAMLs
- [ ] Each template supports context-size sweep via CLI parameter
- [ ] Populate filler source repos: copy representative files from `tasks/local/` directories

### 9.5 Create repo-derived long-context tasks

**Directory:** `tasks/local/long_context/`

**Task definitions:**

| Task ID | Description | Source Repo | Needle Type |
|---|---|---|---|
| `longcontext.repo.qwen3_rmsnorm` | Find RMSNorm parity bug in full qwen3 codebase | qwen3_replicate | Code bug location |
| `longcontext.repo.vllm_timing` | Find regression in vLLM benchmark log output | llm_validation | Numeric value in logs |
| `longcontext.repo.docker_compose_full` | Find misconfiguration in full Docker Compose stack | ai-stack | Config value |
| `longcontext.repo.litellm_route` | Find routing bug in full LiteLLM config | ai-stack | Config path |
| `longcontext.repo.agent_workspace` | Find bug in full agent-orchestrator workspace init | agent-orchestrator | Script path |

**Actions:**
- [ ] Create 5 repo-derived task YAMLs
- [ ] Set `filler_source` to the actual repo paths
- [ ] Ensure needle questions reference real bugs from Milestone 6 task packs

### 9.6 Add context-size sweep to CLI

**File:** `src/bench_harness/cli.py` (extend `run` command)

**New CLI arguments:**
```
--context-sizes    TEXT   Comma-separated token counts for context sweep
                         (e.g., "2000,8000,32000,64000,128000")
                         Default: None (use task's target_size_tokens)
--context-cache-dir PATH  Directory to cache packed contexts
                         (default: "tasks/long_context/packed/")
```

**Modified run flow for `--suite long_context`:**
1. For each task template:
   - For each context size in sweep:
     - Pack context using `ContextPacker`
     - Cache packed context to disk
     - For each model × runs:
       - Run task
       - Capture extended metrics (prefill_ms, decode_ms, gpu_memory_peak)

**New internal data structure:**
```python
@dataclass
class LongContextRunResult(RunResult):
    context_size_tokens: int
    needle_position: str
    distractor_count: int
    distractor_resistance: bool  # did model ignore distractors?
    cited_correct_section: bool  # did model reference the right context section?
    prefill_ms: float | None
    decode_ms: float | None
    gpu_memory_peak_mb: float | None
```

**Actions:**
- [ ] Add CLI arguments to typer command
- [ ] Implement sweep loop: task × context_size × model × runs
- [ ] Wire ContextPacker into the run pipeline
- [ ] Extend `RunResult` or create `LongContextRunResult` subclass

### 9.7 Track prefill/decode/gpu degradation

**File:** `src/bench_harness/metrics/timing.py` (extend)

**New metrics to capture per run:**

```python
@dataclass
class LongContextMetrics:
    prefill_ms: float | None        # time to process full context prompt
    decode_ms: float | None         # time to generate response tokens
    total_wall_ms: float
    tokens_per_second_prefill: float | None
    tokens_per_second_decode: float | None
    gpu_memory_used_start_mb: float | None
    gpu_memory_used_peak_mb: float | None
    gpu_memory_used_end_mb: float | None
    system_ram_used_peak_mb: float | None
    context_length_tokens: int
    response_length_tokens: int
```

**Implementation:**
- Parse vLLM response headers for prefill/decode timing where available
- Fallback: measure wall-clock time from request send to first token (≈TTFT≈prefill)
- GPU memory: use `nvidia-smi` subprocess or `pynvml` before/during/after each request
- Store all metrics in SQLite with new columns (see 9.8)

**File:** `src/bench_harness/metrics/gpu.py` (new)
```python
class GPUMonitor:
    def snapshot(self) -> dict:
        """Return current GPU memory used, utilization, temperature."""

    def monitor_during(self, callable_fn):
        """Context manager that tracks GPU metrics during execution."""
```

**Actions:**
- [ ] Extend timing capture to separate prefill from decode
- [ ] Implement `GPUMonitor` using `subprocess` calling `nvidia-smi` or `pynvml`
- [ ] Add memory snapshot before, during, and after each API call
- [ ] Extend `RunResult` to carry new metric fields

### 9.8 Extend SQLite schema for long-context

**File:** `src/bench_harness/storage/sqlite.py`

**New table:** `long_context_runs`

```sql
CREATE TABLE long_context_runs (
    run_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    context_size_tokens INTEGER NOT NULL,
    needle_position TEXT,
    distractor_count INTEGER,
    distractor_resistance INTEGER,       -- 0 or 1
    cited_correct_section INTEGER,       -- 0 or 1
    prefill_ms REAL,
    decode_ms REAL,
    total_wall_ms REAL,
    tokens_per_second_prefill REAL,
    tokens_per_second_decode REAL,
    gpu_memory_used_start_mb REAL,
    gpu_memory_used_peak_mb REAL,
    gpu_memory_used_end_mb REAL,
    system_ram_used_peak_mb REAL,
    primary_score REAL,
    hallucination_flag INTEGER,
    exit_status TEXT,
    created_at TEXT NOT NULL
);
```

**Migration:**
- Add `ALTER TABLE` with `IF NOT EXISTS` equivalent for new columns
- Or create new table and keep `runs` for non-long-context tasks
- Add indexes on `(model_alias, context_size_tokens)` and `(task_id, model_alias)`

**Actions:**
- [ ] Implement new table creation in `SQLiteStore.init()`
- [ ] Implement `save_long_context_run(result: LongContextRunResult)`
- [ ] Implement `get_degradation_data(model_alias: str) -> list[dict]` query for charting

### 9.9 Score retrieval of buried facts

**File:** `src/bench_harness/scorers/long_context_scorer.py` (new)

**Class:** `LongContextScorer`

**Methods:**
```python
class LongContextScorer(BaseScorer):
    """Scores long-context retrieval quality."""

    def score(self, result: LongContextRunResult, task: dict) -> ScorerResult:
        """
        Composite scoring for long-context tasks:
        - 40% relevant information retrieved (exact/partial match)
        - 25% distractor resistance (did not cite distractors)
        - 15% instruction following (output format, length constraints)
        - 10% answer completeness
        - 10% format compliance
        """

    def check_retrieval(self, response: str, expected_answer: str) -> float:
        """Check if response contains or references the correct needle answer."""

    def check_distractor_resistance(self, response: str, distractors: list[str]) -> bool:
        """Check if response incorrectly cites any distractor value."""

    def check_hallucination(self, response: str, context: str) -> bool:
        """Flag if response invents information not present in context."""

    def check_instruction_following(self, response: str, task: dict) -> float:
        """Check format compliance for long-context output constraints."""
```

**ScorerResult dataclass:**
```python
@dataclass
class LongContextScorerResult:
    primary_score: float          # 0-1 composite
    retrieval_score: float        # 0-1 did it find the needle?
    distractor_resistance: bool
    instruction_following: float  # 0-1
    completeness: float           # 0-1
    format_compliance: float      # 0-1
    hallucination_flag: bool
    details: dict
```

**Actions:**
- [ ] Implement `LongContextScorer` class
- [ ] Implement each sub-score with clear logic
- [ ] Register scorer in `configs/scorers.yaml`
- [ ] Add unit tests: `tests/test_long_context_scorer.py`

### 9.10 Create long-context report

**File:** `src/bench_harness/reports/long_context_report.py` (new)

**Function:** `generate_long_context_report(db_path: str, out_dir: str)`

**Report sections:**

1. **Quality vs Context Length** — line chart per model showing primary_score vs context_size_tokens
2. **Speed vs Context Length** — line chart per model showing prefill_ms and tokens/sec vs context_size_tokens
3. **Context Breakpoint Table** — for each model, the context size where quality drops below 80% of baseline (2k)
4. **Needle Position Analysis** — table showing score by needle position (early/middle/late/multi)
5. **Distractor Resistance** — bar chart showing false-positive rate by model
6. **GPU Memory Pressure** — chart of peak GPU memory vs context size
7. **Per-Model Summary** — recommended max practical context length per model

**Actions:**
- [ ] Implement report generation using SQLite queries
- [ ] Generate Markdown report with embedded charts (matplotlib → PNG → base64 or external)
- [ ] Generate CSV of raw degradation data for external analysis
- [ ] Integrate with CLI: auto-generate when `--suite long_context` completes

### 9.11 Add long-context suite config

**File:** `configs/suites.yaml` (extend)

```yaml
  long_context:
    description: "Measure quality and performance as prompt size increases"
    task_dir: "tasks/synthetic/context_distraction"
    extra_task_dirs:
      - "tasks/local/long_context"
    max_concurrency: 2
    default_runs: 1
    context_sizes: [2000, 8000, 32000, 64000, 128000]
    requires_gpu_monitoring: true
```

**File:** `configs/judge_rubrics.yaml` (extend, if needed for long-context judge scoring)

**Actions:**
- [ ] Add long_context suite to suites.yaml
- [ ] Add scorer config for long_context_scorer

### 9.12 Add script for long-context sweep

**File:** `scripts/run_long_context_sweep.sh`

**Content:**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness run \
  --suite long_context \
  --models agent-code,qwen-dense,max-brain \
  --context-sizes 2000,8000,32000,64000,128000 \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-long-context-sweep"
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 9.13 Add tests

**File:** `tests/test_long_context.py`

**Tests:**
- `test_context_packer_build` — builds context at target size, needle is present
- `test_context_packer_position` — needle appears at correct position (early/middle/late)
- `test_distractor_generator` — generates correct number of distractors
- `test_distractor_insertion` — context size remains stable after distractor insertion
- `test_long_context_scorer_correct` — correct answer scores 1.0
- `test_long_context_scorer_distractor` — response citing distractor fails distractor resistance
- `test_long_context_scorer_hallucination` — invented answer is flagged
- `test_sqlite_long_context_schema` — table creation and round-trip save/retrieve
- `test_context_size_sweep` — CLI sweep produces expected number of runs

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixtures for temp DB and temp filler directory
- [ ] Mock API client for scorer tests

---

## Acceptance Criteria Checklist

- [ ] Can run same task at 2k, 8k, 32k, 64k, 128k tokens
- [ ] Produces quality-vs-context-length line chart per model
- [ ] Produces speed-vs-context-length line chart per model
- [ ] Identifies model-specific context breakpoints (where quality < 80% baseline)
- [ ] Distractor resistance is measured and reported
- [ ] GPU memory pressure is tracked and reported
- [ ] Prefill/decode time degradation is captured separately
- [ ] Report identifies practical max context length per model
- [ ] Packed contexts are cached to avoid recomputation
- [ ] `pytest tests/test_long_context.py` passes

## Estimated Effort

4–5 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/tasks/context_packer.py` | To create |
| `src/bench_harness/tasks/distractor_generator.py` | To create |
| `src/bench_harness/tasks/task_schema.py` | Extend |
| `src/bench_harness/scorers/long_context_scorer.py` | To create |
| `src/bench_harness/metrics/gpu.py` | To create |
| `src/bench_harness/metrics/timing.py` | Extend |
| `src/bench_harness/storage/sqlite.py` | Extend |
| `src/bench_harness/reports/long_context_report.py` | To create |
| `src/bench_harness/cli.py` | Extend |
| `configs/suites.yaml` | Extend |
| `configs/scorers.yaml` | Extend |
| `tasks/synthetic/context_distraction/*.yaml` (×7) | To create |
| `tasks/local/long_context/*.yaml` (×5) | To create |
| `scripts/run_long_context_sweep.sh` | To create |
| `tests/test_long_context.py` | To create |
