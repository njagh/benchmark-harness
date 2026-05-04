# Milestone 1 — Project Bootstrap

## Goal

Create the repository structure, basic CLI, model config, and smoke runner.

## Phase

Phase A — Minimal useful harness (Milestone 1 of 4 in phase)

## Dependencies

None (first milestone)

---

## Subtasks

### 1.1 Create repository structure

**Files to create:**

```
benchmark-harness/
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── models.yaml
│   ├── suites.yaml
│   └── scorers.yaml
├── src/
│   └── bench_harness/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── openai_client.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   └── loaders.py
│       ├── runners/
│       │   ├── __init__.py
│       │   └── completion_runner.py
│       ├── scorers/
│       │   └── __init__.py
│       ├── metrics/
│       │   └── __init__.py
│       ├── storage/
│       │   ├── __init__.py
│       │   └── sqlite.py
│       └── reports/
│           ├── __init__.py
│           └── markdown.py
├── tasks/
│   └── smoke/
│       └── .gitkeep
├── runs/
│   └── .gitkeep
├── scripts/
│   └── run_smoke.sh
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

**Actions:**
- [ ] Create all directories
- [ ] Create all `__init__.py` stub files
- [ ] Create `.gitignore` (exclude `runs/`, `__pycache__/`, `.env`, `*.pyc`, `*.db`)
- [ ] Create `.env.example` with `OPENAI_BASE_URL`, `OPENAI_API_KEY` placeholders

### 1.2 Define pyproject.toml

**File:** `pyproject.toml`

**Required dependencies:**
- `typer[all]` — CLI framework (with async and rich support)
- `rich` — CLI output formatting, tables, progress bars
- `pyyaml` — YAML config parsing
- `pydantic` — task schema validation (used from M1 onward for configs)
- `openai` — OpenAI-compatible client
- `httpx` — async HTTP (used by openai client)
- `sqlite-utils` — SQLite helpers (bulk insert, schema management, CSV export). Use THIS over raw sqlite3.
- `jsonschema` — JSON schema validation (used by M4 scorer, but install early)
- `jinja2` — prompt template rendering (used by M2, but install early)
- `unidiff` — unified diff parsing (used by M5, but install early)
- `pytest` — testing framework
- `pytest-asyncio` — async test support

All libraries are installed at project bootstrap to avoid piecemeal dependency management. Libraries not yet used by M1 will have their import guarded by lazy loading.

**File structure:**
- `[project]` with name `bench-harness`, version `0.1.0`
- `[project.scripts]` entry point `bench-harness = "bench_harness.cli:app"`
- `[tool.setuptools.packages.find]` where `where = ["src"]`
- `[tool.pytest.ini_options]` with `testpaths = ["tests"]`

### 1.3 Define configs/models.yaml

**File:** `configs/models.yaml`

**Content spec:**

```yaml
models:
  agent-code:
    provider: openai_compatible
    base_url: "http://spark-e287.local:4000/v1"
    model: "agent-code"
    backend: vllm
    quantization: FP8
    notes: "Qwen3.6-35B-A3B-FP8 via LiteLLM"

  qwen-dense:
    provider: openai_compatible
    base_url: "http://spark-e287.local:4000/v1"
    model: "qwen-dense"
    backend: vllm
    quantization: FP8
    notes: "Qwen3.6-27B-FP8"

  max-brain:
    provider: openai_compatible
    base_url: "http://spark-e287.local:4000/v1"
    model: "max-brain"
    backend: vllm
    quantization: GPTQ-Int4
    notes: "Qwen3.5-122B-A10B GPTQ Int4"
```

**Actions:**
- [ ] Write models.yaml with above 3 model entries
- [ ] Document that additional models can be added following the same schema

### 1.4 Define configs/suites.yaml

**File:** `configs/suites.yaml`

**Content spec:**

```yaml
suites:
  smoke:
    description: "Verify harness and model endpoint work"
    task_dir: "tasks/smoke"
    max_concurrency: 4
    default_runs: 1
```

### 1.5 Implement config loader

**File:** `src/bench_harness/config.py`

**Functions:**
- `load_model_config(path: str) -> dict` — loads and returns `configs/models.yaml`
- `get_model(config: dict, alias: str) -> dict | None` — looks up a model by alias
- `load_suite_config(path: str) -> dict` — loads and returns `configs/suites.yaml`
- `get_suite(config: dict, name: str) -> dict | None` — looks up a suite by name

**Actions:**
- [ ] Implement using `pyyaml.safe_load()`
- [ ] Resolve paths relative to project root (find `configs/` sibling to `src/`)
- [ ] Add basic validation: raise if required keys missing

### 1.6 Implement OpenAI-compatible client

**File:** `src/bench_harness/models/openai_client.py`

**Class:** `OpenAICompatClient`

**Methods:**
- `__init__(base_url: str, api_key: str = "not-needed", model: str = "")`
- `chat_complete(messages: list[dict], temperature: float = 0, max_tokens: int = 4096, **kwargs) -> dict`
  - Returns dict with `content`, `usage` (prompt_tokens, completion_tokens), `finish_reason`
- `chat_complete_stream(messages: list[dict], temperature: float = 0, max_tokens: int = 4096, **kwargs) -> AsyncIterator[str]`
  - Yields content chunks as they arrive
  - Tracks time to first token internally

**Implementation details:**
- Use `openai.AsyncOpenAI(base_url=..., api_key=...)`
- Call `client.chat.completions.create()`
- Handle API errors, timeout, and retry (3 retries with exponential backoff)
- Capture `response.usage` for token counts

**Actions:**
- [ ] Implement `OpenAICompatClient` class
- [ ] Add error handling with typed exceptions (`APIConnectionError`, `APITimeoutError`)
- [ ] Add logging of request metadata (model, base_url, timestamp)

### 1.7 Implement simple completion runner

**File:** `src/bench_harness/runners/completion_runner.py`

**Class:** `CompletionRunner`

**Methods:**
- `__init__(client: OpenAICompatClient)`
- `run(task: dict, params: dict) -> RunResult`
  - `task` has `id`, `prompt`, `expected`, `scoring`
  - `params` has `temperature`, `max_tokens`, `model_alias`
  - Returns `RunResult` dataclass with fields from the data model (§5.3)

**RunResult dataclass fields:**
```python
@dataclass
class RunResult:
    run_id: str
    suite_id: str
    task_id: str
    model_alias: str
    prompt: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float
    total_wall_ms: float
    exit_status: str  # "success" | "error"
    error_message: str | None
    created_at: str  # ISO 8601
```

**Actions:**
- [ ] Define `RunResult` dataclass
- [ ] Implement `run()` with wall-clock timing
- [ ] Generate `run_id` as UUID
- [ ] Handle and capture exceptions without crashing

### 1.8 Implement SQLite storage

**File:** `src/bench_harness/storage/sqlite.py`

**Class:** `SQLiteStore`

**Schema:**

Table `runs`:
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    model_backend TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    ttft_ms REAL,
    total_wall_ms REAL,
    exit_status TEXT,
    raw_response TEXT,
    score_primary REAL,
    scorer_version TEXT,
    created_at TEXT NOT NULL
);
```

Table `environments`:
```sql
CREATE TABLE environments (
    id TEXT PRIMARY KEY,
    host TEXT,
    os TEXT,
    gpu_name TEXT,
    cuda_version TEXT,
    harness_commit TEXT,
    created_at TEXT NOT NULL
);
```

**Methods:**
- `__init__(db_path: str)`
- `init()` — create tables if not exist
- `save_run(result: RunResult)`
- `save_environment(snapshot: dict)`
- `get_runs(suite_id: str | None = None, model_alias: str | None = None) -> list[dict]`

**Actions:**
- [ ] Implement table creation with `IF NOT EXISTS`
- [ ] Implement `save_run` with parameterized queries
- [ ] Implement `get_runs` with optional WHERE filters
- [ ] Create index on `(suite_id, model_alias)`

### 1.9 Add first five smoke tasks

**Directory:** `tasks/smoke/`

Each task is a YAML file:

**Task 1 — `factual_trivial.yaml`**
```yaml
id: smoke.factual_001
family: general
prompt: "What is the capital of France? Answer in one word."
expected:
  type: exact
  answer: "Paris"
scoring:
  primary: exact_match
```

**Task 2 — `json_format.yaml`**
```yaml
id: smoke.json_001
family: format_following
prompt: |
  Output a JSON object with keys "name", "language", and "year_created"
  for the Python programming language. Output only valid JSON.
expected:
  type: json_schema
  schema:
    type: object
    required: [name, language, year_created]
scoring:
  primary: json_schema
```

**Task 3 — `python_function.yaml`**
```yaml
id: smoke.python_001
family: coding
prompt: |
  Write a Python function called fibonacci(n) that returns the nth
  Fibonacci number. n is a non-negative integer.
expected:
  type: contains
  patterns:
    - "def fibonacci"
scoring:
  primary: regex
```

**Task 4 — `debug_small.yaml`**
```yaml
id: smoke.debug_001
family: debugging
prompt: |
  This Python code has a bug:
    def add(a, b):
      return a - b
  What is the bug and how do you fix it?
expected:
  type: contains
  patterns:
    - "a + b"
scoring:
  primary: regex
```

**Task 5 — `instruction_follow.yaml`**
```yaml
id: smoke.instruction_001
family: instruction_following
prompt: |
  List exactly three benefits of unit testing. Use a numbered list.
  Do not write any other text.
expected:
  type: format
  checks:
    - starts_with_numbered_list
    - exactly_three_items
scoring:
  primary: regex
```

**Actions:**
- [ ] Create all 5 YAML files
- [ ] Create `tasks/smoke/loader.py` or update `src/bench_harness/tasks/loaders.py` to read task YAMLs from a directory

### 1.10 Implement task loader

**File:** `src/bench_harness/tasks/loaders.py`

**Functions:**
- `load_tasks(task_dir: str) -> list[dict]` — reads all `.yaml` files from directory
- `load_task(task_path: str) -> dict` — reads a single task file
- `filter_tasks(tasks: list[dict], family: str | None) -> list[dict]` — optional filter

**Actions:**
- [ ] Implement YAML loading with error handling per file
- [ ] Validate required keys: `id`, `prompt`, `scoring`
- [ ] Skip files with clear error message on load failure

### 1.11 Implement CLI

**File:** `src/bench_harness/cli.py`

**Framework:** `typer`

**Commands:**
- `app.run()` — main benchmark command

**CLI arguments:**
```
--suite        TEXT   Suite name(s), comma-separated  (default: "smoke")
--models       TEXT   Model alias(es), comma-separated (default: "agent-code")
--endpoint     TEXT   Base URL override                (default: from config)
--temperature  FLOAT  Sampling temperature             (default: 0)
--max-tokens   INT    Max output tokens                (default: 4096)
--runs         INT    Number of repetitions per task   (default: 1)
--out          PATH   Output directory                 (default: "runs/YYYY-MM-DD-<suite>")
```

**Flow:**
1. Load model config
2. Load suite config → resolve task directory
3. Load tasks from task directory
4. For each model:
   - Create client
   - For each task × runs:
     - Run completion runner
     - Save result to SQLite
     - Save raw JSONL artifact
5. Generate Markdown report
6. Print summary to stdout

**Actions:**
- [ ] Implement typer CLI with all arguments
- [ ] Wire config → tasks → runner → storage → report pipeline
- [ ] Add `--help` documentation
- [ ] Add `--dry-run` flag (prints tasks without executing)

### 1.12 Implement JSONL artifact output

**File:** `src/bench_harness/storage/artifacts.py`

**Functions:**
- `save_run_artifact(result: RunResult, out_dir: str)` — writes one JSONL line
- File naming: `{out_dir}/{suite}_{model}_{task_id}_{run_N}.jsonl`
- Or consolidated: `{out_dir}/runs.jsonl` (append mode)

**Actions:**
- [ ] Implement consolidated JSONL writer
- [ ] Ensure each line is a valid JSON object with all RunResult fields
- [ ] Create output directory if it doesn't exist

### 1.13 Implement basic Markdown report

**File:** `src/bench_harness/reports/markdown.py`

**Function:** `generate_report(runs: list[dict], out_path: str)`

**Report sections:**
1. Header: suite name, date, host
2. Models table: alias, backend, notes
3. Summary table:

| Model | Tasks Run | Passed | Failed | Avg TTFT (ms) | Avg Tokens/sec |
|---|---|---|---|---|---|

4. Per-task results:

| Task | Model | Status | TTFT | Tokens |
|---|---|---|---|---|

5. Raw output excerpts for failed tasks (first 200 chars)

**Actions:**
- [ ] Implement markdown generation with table formatting
- [ ] Compute pass/fail from scoring (for M1, "success" exit_status = pass)
- [ ] Write to both `.md` file and stdout summary

### 1.14 Implement scripts/run_smoke.sh

**File:** `scripts/run_smoke.sh`

**Content:**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness run \
  --suite smoke \
  --models agent-code,qwen-dense,max-brain \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-smoke"
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 1.15 Add basic tests

**File:** `tests/test_smoke.py`

**Tests:**
- `test_load_model_config` — loads config, returns dict with expected models
- `test_load_smoke_tasks` — loads tasks from smoke dir, returns 5+ tasks
- `test_task_has_required_fields` — each loaded task has `id`, `prompt`, `scoring`
- `test_sqlite_store_init` — creates tables without error
- `test_sqlite_save_and_retrieve` — round-trip save/get of a run record

**Actions:**
- [ ] Implement tests with pytest
- [ ] Use fixture for temp SQLite DB path
- [ ] Ensure tests pass with `pytest tests/`

### 1.16 Define dataset registry config

**File:** `configs/datasets.yaml`

**Content spec:**
```yaml
# Dataset registry — all benchmark data is loaded from local pinned files.
# Never stream from HuggingFace during measured runs.
# See STORAGE_PLAN.md for dataset layout and access patterns.

datasets:
  # Local task packs live in the repo under tasks/
  smoke_v1:
    type: local_yaml
    path: "tasks/smoke"
    description: "Five smoke tasks for harness verification"

  coding_smoke_v1:
    type: local_yaml
    path: "tasks/coding_smoke"
    description: "Coding regression smoke tasks"

  # External eval data lives in ~/datasets/evals/
  # These are populated by scripts/prepare_eval_dataset.py (M2+)
  # mmlu_pro_v1:
  #   type: jsonl
  #   path: "/home/njalbicelli/datasets/evals/mmlu_pro_v1/tasks.jsonl"
  #   manifest: "/home/njalbicelli/datasets/evals/mmlu_pro_v1/MANIFEST.json"
```

**File:** `src/bench_harness/config.py` — add function:
- `load_dataset_config(path: str) -> dict` — loads configs/datasets.yaml
- `get_dataset(config: dict, name: str) -> dict | None`

**STORAGE_PLAN integration notes:**
- Harness never pulls live from HuggingFace during benchmark runs
- All eval data is downloaded once, pinned with manifest, stored locally
- `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE` env vars documented in `.env.example`
- Dataset paths use `~/datasets/evals/` for external data, `tasks/` for local YAML tasks

**Actions:**
- [ ] Create configs/datasets.yaml
- [ ] Add dataset config loader to config.py
- [ ] Add HF cache env var docs to .env.example

---

## Acceptance Criteria Checklist

- [ ] `python -m bench_harness run --suite smoke --models agent-code` runs end-to-end
- [ ] `python -m bench_harness run --suite smoke --models agent-code,qwen-dense` runs against two models
- [ ] SQLite database is created at output path with run records
- [ ] JSONL artifacts are written to output directory
- [ ] Markdown report is generated with summary table
- [ ] All 5 smoke tasks execute and produce results
- [ ] `pytest tests/` passes
- [ ] `scripts/run_smoke.sh` works from project root

## Estimated Effort

2–3 days of focused implementation

## Files Produced by This Milestone

| File | Status |
|---|---|
| `pyproject.toml` | To create |
| `.env.example` | To create |
| `.gitignore` | To create |
| `configs/models.yaml` | To create |
| `configs/suites.yaml` | To create |
| `configs/scorers.yaml` | To create (stub) |
| `src/bench_harness/__init__.py` | To create |
| `src/bench_harness/cli.py` | To create |
| `src/bench_harness/config.py` | To create |
| `src/bench_harness/models/openai_client.py` | To create |
| `src/bench_harness/tasks/loaders.py` | To create |
| `src/bench_harness/runners/completion_runner.py` | To create |
| `src/bench_harness/storage/sqlite.py` | To create |
| `src/bench_harness/storage/artifacts.py` | To create |
| `src/bench_harness/reports/markdown.py` | To create |
| `tasks/smoke/*.yaml` (×5) | To create |
| `scripts/run_smoke.sh` | To create |
| `tests/test_smoke.py` | To create |
