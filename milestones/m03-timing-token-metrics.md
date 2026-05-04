# Milestone 3 — Timing and Token Metrics

## Goal

Capture performance metrics (TTFT, wall time, decode time, token counts) comparable to existing timing benchmarks, stored consistently in SQLite and exposed in reports.

## Phase

Phase A — Minimal useful harness (Milestone 2 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap, client, runner, SQLite storage)
- Milestone 2 (task schema and registry — needed for structured run records)

---

## Subtasks

### 3.1 Implement wall-clock timing layer

**File:** `src/bench_harness/metrics/timing.py`

**Class:** `TimingContext`

```python
@contextmanager
def timed_block(label: str = "block") -> Iterator[TimingRecord]:
    """Context manager that captures wall-clock elapsed time."""
```

**Class:** `TimingRecord`

```python
@dataclass
class TimingRecord:
    label: str
    start_wall: float          # time.monotonic()
    end_wall: float | None
    elapsed_ms: float | None
```

**Functions:**
- `measure_function(fn, *args, **kwargs) -> tuple[Any, float]` — wraps a callable, returns `(result, elapsed_ms)`
- `format_duration(ms: float) -> str` — human-readable duration string

**Actions:**
- [ ] Implement `TimingContext` context manager
- [ ] Implement `TimingRecord` dataclass
- [ ] Use `time.monotonic()` for wall-clock, `time.perf_counter()` for high-resolution
- [ ] Add unit tests for timing accuracy (sleep-based)

### 3.2 Implement streaming TTFT capture

**File:** `src/bench_harness/metrics/timing.py` (extend)

**Class:** `StreamingTimer`

```python
class StreamingTimer:
    def __init__(self):
        self.start_time: float = 0.0
        self.first_token_time: float | None = None
        self.last_chunk_time: float = 0.0
        self.chunk_count: int = 0
        self.total_text: str = ""

    def start(self):
        """Mark the start of the streaming request."""

    def on_chunk(self, text: str):
        """Called for each streamed chunk. Captures TTFT on first call."""

    def finalize(self) -> StreamMetrics:
        """Compute and return final streaming metrics."""
```

**Class:** `StreamMetrics`

```python
@dataclass
class StreamMetrics:
    ttft_ms: float                   # time to first token
    decode_ms: float | None          # time from first to last token
    total_wall_ms: float             # total request wall time
    chunk_count: int                 # number of streamed chunks
    chars_per_chunk_avg: float       # average characters per chunk
```

**Integration with `OpenAICompatClient.chat_complete_stream`:**
- Wrap the async stream iterator with `StreamingTimer`
- Call `timer.start()` before iteration begins
- Call `timer.on_chunk(chunk)` for each yielded chunk
- Call `timer.finalize()` after stream completes
- Attach `StreamMetrics` to the `RunResult`

**Actions:**
- [ ] Implement `StreamingTimer` class
- [ ] Implement `StreamMetrics` dataclass
- [ ] Update `OpenAICompatClient.chat_complete_stream()` to use `StreamingTimer`
- [ ] Update `completion_runner.py` to capture `StreamMetrics` in `RunResult`

### 3.3 Capture API token counts

**File:** `src/bench_harness/metrics/tokens.py`

**Class:** `TokenCounter`

```python
class TokenCounter:
    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def from_api_usage(self, usage: dict) -> 'TokenCounter':
        """Extract token counts from OpenAI-compatible usage object.
        
        Handles both flat dict {'prompt_tokens': N} and nested
        {'completion_tokens_details': {...}} formats used by some backends.
        """

    def from_response(self, response: Any) -> 'TokenCounter':
        """Extract from OpenAI SDK response object (response.usage)."""
```

**Functions:**
- `normalize_usage(usage: Any) -> dict` — normalizes usage objects across backend formats (vLLM, LiteLLM, OpenAI)
- `compute_tokens_per_second(tokens: int, duration_ms: float) -> float`

**Backend-specific handling:**
- **vLLM**: May include `prompt_tokens`, `completion_tokens`, `total_tokens` directly
- **LiteLLM**: Passes through upstream usage, may add `accepted_prediction_tokens`, `audio_tokens`
- **Fallback**: If `usage` is `None`, mark fields as `-1` and log warning

**Actions:**
- [ ] Implement `TokenCounter` class
- [ ] Handle missing `usage` field gracefully (set to `-1`, log warning)
- [ ] Update `completion_runner.py` to include `TokenCounter` results in `RunResult`

### 3.4 Implement fallback tokenizer counting

**File:** `src/bench_harness/metrics/tokens.py` (extend)

**Class:** `FallbackTokenCounter`

```python
class FallbackTokenCounter:
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer = get_tokenizer(tokenizer_name)

    def count_prompt(self, messages: list[dict]) -> int:
        """Count tokens in a list of chat messages."""

    def count_completion(self, text: str) -> int:
        """Count tokens in completion text."""

    def count_total(self, messages: list[dict], text: str) -> int:
        """Count total tokens (prompt + completion)."""
```

**Implementation details:**
- Use `tiktoken` library for counting
- Default to `cl100k_base` (GPT-4 family tokenizer)
- Configurable per-model in `configs/models.yaml` via `tokenizer: cl100k_base` field
- Only used as fallback when API doesn't return usage data

**Actions:**
- [ ] Add `tiktoken` to `pyproject.toml` dependencies
- [ ] Implement `FallbackTokenCounter`
- [ ] Add tokenizer selection logic: prefer API counts, fall back to local counting
- [ ] Add model-level tokenizer config in `configs/models.yaml`

### 3.5 Extend RunResult with timing and token fields

**File:** `src/bench_harness/runners/completion_runner.py` (update `RunResult`)

**Updated dataclass fields:**
```python
@dataclass
class RunResult:
    # ... existing fields ...
    # New timing fields:
    prefill_ms: float | None = None        # from API or estimated
    decode_ms: float | None = None         # decode time (streaming)
    tokens_per_second: float | None = None # completion_tokens / (decode_ms / 1000)
    tokens_per_second_total: float | None = None  # total_tokens / (total_wall_ms / 1000)
    # New token source tracking:
    token_source: str = "api"              # "api" | "fallback_tokenizer"
    chunk_count: int | None = None         # number of streaming chunks
```

**Actions:**
- [ ] Add new fields to `RunResult` dataclass
- [ ] Update `completion_runner.run()` to populate all new fields
- [ ] Ensure non-streaming path still computes `tokens_per_second` from wall time

### 3.6 Extend SQLite schema for timing metrics

**File:** `src/bench_harness/storage/sqlite.py` (update)

**Schema migration — add columns to `runs` table:**
```sql
ALTER TABLE runs ADD COLUMN prefill_ms REAL;
ALTER TABLE runs ADD COLUMN decode_ms REAL;
ALTER TABLE runs ADD COLUMN tokens_per_second REAL;
ALTER TABLE runs ADD COLUMN tokens_per_second_total REAL;
ALTER TABLE runs ADD COLUMN token_source TEXT DEFAULT 'api';
ALTER TABLE runs ADD COLUMN chunk_count INTEGER;
```

**Schema migration — extend `environments` table:**
```sql
ALTER TABLE environments ADD COLUMN vllm_version TEXT;
ALTER TABLE environments ADD COLUMN litellm_version TEXT;
ALTER TABLE environments ADD COLUMN model_path TEXT;
ALTER TABLE environments ADD COLUMN quantization TEXT;
ALTER TABLE environments ADD COLUMN max_model_len INTEGER;
ALTER TABLE environments ADD COLUMN gpu_memory_utilization REAL;
ALTER TABLE environments ADD COLUMN served_port INTEGER;
```

**New table — `run_timings`:**
```sql
CREATE TABLE run_timings (
    run_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    ttft_ms REAL NOT NULL,
    prefill_ms REAL,
    decode_ms REAL,
    total_wall_ms REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    tokens_per_second REAL,
    tokens_per_second_total REAL,
    token_source TEXT,
    chunk_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**Methods (add):**
- `save_run_timing(run_id: str, timing_data: dict)` — saves timing record
- `get_timing_summary(model_alias: str, suite_id: str | None) -> dict` — aggregates: mean/min/max/p95 TTFT, tokens/sec, wall time

**Migration strategy:**
- Use `PRAGMA table_info` to check existing columns before ALTER
- Wrap ALTER in try/except (SQLite doesn't support transactional DDL in all versions)
- Add `_migration_version` table to track applied migrations

**Actions:**
- [ ] Implement schema migration with column existence checks
- [ ] Add `run_timings` table creation
- [ ] Implement `save_run_timing`
- [ ] Implement `get_timing_summary` with SQL aggregation queries
- [ ] Add migration version tracking

### 3.7 Add per-model timing summary to report

**File:** `src/bench_harness/reports/markdown.py` (update)

**New report sections:**

1. **Timing Summary** (after existing summary table):
```markdown
## Timing Summary

| Model | Avg TTFT (ms) | P95 TTFT (ms) | Avg Decode (ms) | Avg Tokens/sec | Avg Total Wall (ms) |
|---|---|---|---|---|---|
```

2. **Per-Task Timing**:
```markdown
## Per-Task Timing

| Task | Model | TTFT (ms) | Decode (ms) | Tokens/sec | Prompt Tok | Completion Tok |
|---|---|---|---|---|---|---|
```

3. **Slowest Tasks** (top 5 by wall time):
```markdown
## Slowest Tasks

| Rank | Task | Model | Wall Time (ms) | Tokens |
|---|---|---|---|---|
```

**Actions:**
- [ ] Add timing summary section to `generate_report()`
- [ ] Add per-task timing table
- [ ] Add "slowest tasks" table sorted by `total_wall_ms`
- [ ] Compute P95 using sorted list indexing (no external stats library needed)

### 3.8 Add timing metrics CLI output

**File:** `src/bench_harness/cli.py` (update)

**New CLI flag:**
```
--timing-detail   BOOLEAN  Show per-task timing in CLI output  (default: false)
```

**Output format (when enabled):**
```
Task: smoke.factual_001 | Model: agent-code
  TTFT: 120ms | Decode: 45ms | Wall: 168ms
  Tokens: 12 prompt, 8 completion | 47.6 tok/s
```

**Actions:**
- [ ] Add `--timing-detail` flag to CLI
- [ ] Format timing output using `rich` console
- [ ] Include token source indicator (api/fallback)

### 3.9 Add timing metric tests

**File:** `tests/test_timing.py`

**Tests:**
- `test_timing_context_manager` — `timed_block` captures elapsed time correctly
- `test_streaming_timer_ttft` — `StreamingTimer` captures TTFT on first chunk
- `test_streaming_timer_decode` — `StreamingTimer` computes decode time correctly
- `test_streaming_timer_finalize` — `finalize()` returns `StreamMetrics` with all fields
- `test_token_counter_from_api` — `TokenCounter.from_api_usage()` parses standard usage
- `test_token_counter_missing_usage` — handles `None` usage gracefully
- `test_fallback_token_counter` — counts tokens when API provides no usage
- `test_tokens_per_second_computation` — `compute_tokens_per_second()` is correct
- `test_sqlite_timing_save_and_retrieve` — round-trip of timing record
- `test_sqlite_timing_summary` — `get_timing_summary` returns correct aggregates
- `test_schema_migration_safe` — migration doesn't fail on fresh or existing DB

**Actions:**
- [ ] Implement all tests
- [ ] Use `time.sleep()` for timing accuracy tests with tolerance
- [ ] Mock streaming responses for `StreamingTimer` tests
- [ ] Use temp SQLite DB for storage tests

---

## Acceptance Criteria Checklist

- [ ] Report shows TTFT, wall time, decode time, completion tokens, and tokens/sec per task
- [ ] Metrics are captured consistently across streaming and non-streaming responses
- [ ] Fallback tokenizer counting works when API returns no usage data
- [ ] SQLite `run_timings` table stores all timing metrics
- [ ] `get_timing_summary()` returns per-model aggregates (mean, min, max, p95)
- [ ] Markdown report includes timing summary table and per-task timing table
- [ ] `--timing-detail` CLI flag shows per-task timing in terminal output
- [ ] Schema migration is safe on both fresh and existing databases
- [ ] All timing metric tests pass

## Estimated Effort

2 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `src/bench_harness/metrics/timing.py` | To create |
| `src/bench_harness/metrics/tokens.py` | To create |
| `src/bench_harness/metrics/__init__.py` | To create |
| `src/bench_harness/runners/completion_runner.py` | Update (extend RunResult) |
| `src/bench_harness/models/openai_client.py` | Update (StreamingTimer integration) |
| `src/bench_harness/storage/sqlite.py` | Update (schema migration + run_timings table) |
| `src/bench_harness/reports/markdown.py` | Update (timing report sections) |
| `src/bench_harness/cli.py` | Update (--timing-detail flag) |
| `tests/test_timing.py` | To create |
| `configs/models.yaml` | Update (add tokenizer field) |
