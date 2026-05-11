# Web UI for Benchmark Harness — Implementation Roadmap

## Problem

Running the benchmark harness requires juggling many CLI variables — suite names, model aliases, endpoints, temperatures, context sizes, prompt styles, storage roots, judge flags, and more. This is error-prone and hard to reproduce. A web UI would let users **create configs visually**, **launch runs with one click**, and **explore results interactively** without touching the terminal.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     Browser (UI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Config   │ │  Runs    │ │ Results  │ │ Compare    │  │
│  │ Builder  │ │  Runner  │ │ Viewer   │ │ Dashboard  │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │ REST + WebSocket
┌────────────────────────▼─────────────────────────────────┐
│              Flask API Server                             │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │ Config API │ │ Run API    │ │ Results API          │  │
│  │ CRUD       │ │ Launch/WS  │ │ Query + Aggregations │  │
│  └────────────┘ └────────────┘ └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐│
│  │  Bench Harness Library (existing)                     ││
│  │  RunSpec · StorageConfig · Runner · Scorers · Reports ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│  Storage (SQLite + JSONL + file artifacts)                 │
└───────────────────────────────────────────────────────────┘
```

**Choice: Flask** over FastAPI. Reason: the existing codebase already depends on no ASGI framework. Flask is simpler, has fewer dependencies, and the app is I/O-bound (calling the existing runner), not CPU-bound on request handling. A lightweight synchronous Flask server is the lowest-friction path.

---

## Phase 0 — Foundation (Week 1)

### Goal: Working Flask server that reads/writes configs and delegates to existing library

### 0.1 Project setup
- New package subdirectory: `src/bench_harness/server/`
- `pyproject.toml` optional deps add: `flask`, `flask-cors`
- Entry point: `bench-harness web-ui` → starts Flask server
- Minimal `app.py` with `/health` endpoint

### 0.2 Config serialization bridge
- Build API that mirrors `RunSpec` fields with readable labels
- Convert JSON from UI → `RunSpec` Pydantic model → YAML
- Convert `RunSpec` → JSON for UI display
- Handle enum display (ArtifactKind, RuntimeKind, LaunchMode, ArtifactMode) as human dropdowns, not raw strings

### 0.3 Config storage
- Store configs as YAML files in a `saved-configs/` directory under the storage root
- Endpoints:
  - `GET /api/configs` — list saved configs
  - `POST /api/configs` — create a new config
  - `GET /api/configs/{id}` — load a config
  - `PUT /api/configs/{id}` — update a config
  - `DELETE /api/configs/{id}` — delete a config
  - `POST /api/configs/{id}/duplicate` — clone a config (for quick variations)

### 0.4 Load existing config sources
- `GET /api/models` — populate from `configs/models.yaml`
- `GET /api/suites` — populate from `configs/suites.yaml`
- `GET /api/scorers` — populate from `configs/scorers.yaml`
- `GET /api/task-families` — discover task families from `tasks/` directory
- `GET /api/prompt-styles` — list `configs/prompt_templates/*.md` filenames

### 0.5 Storage resolution
- Read `LLM_BENCH_STORAGE_ROOT`, `.llm-bench.yaml`, fallback default
- Surface resolved paths in UI config panel ("Results will be saved to: ...")
- Allow override via `?storage_root=` query param or server env var

---

## Phase 1 — Config Builder UI (Week 2)

### Goal: Full interactive form for creating and saving benchmark configs

### 1.1 Config Builder page

A multi-section form with collapsible panels:

**Panel: Run Identity**
- Run name (slug validation from RunSpec)
- Project name
- Tags (free-text → list, or tags from prior runs)

**Panel: Model / Artifact**
- Dropdown: model alias (from `configs/models.yaml`) — pre-fills all known fields
- Or manual entry:
  - Kind: dropdown (hf_checkpoint, trtllm_engine, gguf, vllm_endpoint, openai_endpoint)
  - Mode: dropdown (external_path, managed_copy, managed_symlink)
  - Path / URL
  - Model ID
  - Quantization
  - Tokenizer path
- Validation: endpoint kinds require `http://` or `https://` prefix (enforce `RunSpec.validate_name`)

**Panel: Runtime**
- Kind: dropdown (openai_compatible, vllm, trtllm, llamacpp, external)
- Launch: dropdown (existing, managed_process)
- Host, Port (conditional)
- Model name
- Extra args (key-value pairs)

**Panel: Workload**
- Prompt suite: dropdown (from `configs/suites.yaml`)
- Task dir: optional text
- Max tokens: number input (default 256)
- Temperature: slider (0.0–2.0, step 0.1)
- Num runs: number input (default 1)
- Concurrency: number input (default 1)

**Panel: Advanced Overrides**
- Styles: multi-select (plain, step_by_step, json_schema, architect, patch_only, terse, repl)
- Context sizes: multi-select (small, medium, large, xlarge)
- Judge: toggle (yes/no)
- Report v2: toggle
- Prior runs path: text input
- Output directory: text input (overrides storage root)

**Panel: Summary Preview**
- Live YAML preview (synced as user types)
- "Copy YAML" button
- "Save config" button → POST to `/api/configs`
- "Run now" button → POST to `/api/runs` with this config

### 1.2 Config versioning
- Each saved config gets a UUID and timestamp
- `GET /api/configs/{id}/history` — show edits over time
- Auto-save draft every 30 seconds (localStorage → server sync on close)

### 1.3 Quick templates
- "Smoke test" template pre-fills suite=smoke, max_tokens=256, runs=1
- "Full benchmark" template pre-fills suite=coding_smoke, runs=3, judge=true
- "Style comparison" template pre-fills all styles
- "Context sweep" template pre-fills all context sizes
- Templates load as starting points, then user tweaks

---

## Phase 2 — Run Runner (Week 3)

### Goal: Launch benchmark runs from UI, track progress in real-time

### 2.1 Run API
- `POST /api/runs` — accept a config JSON or config ID, create and launch a run
  - Returns `{"run_id": "...", "status": "queued"}`
  - Writes resolved spec YAML to result directory immediately
  - Calls existing `bench_harness` runner via subprocess or direct function call
- `GET /api/runs` — list recent runs with summary (name, status, model, date, tasks, passed/total)
- `GET /api/runs/{run_id}` — full run status
- `GET /api/runs/{run_id}/logs` — tail server output

### 2.2 Real-time progress (WebSocket)
- WebSocket endpoint: `/ws/runs/{run_id}`
- Server broadcasts events:
  - `{"type": "started", "run_id": "..."}`
  - `{"type": "task_started", "task_id": "...", "model": "...", "style": "..."}`
  - `{"type": "task_completed", "task_id": "...", "score": 0.85, "ttft_ms": 42, "tokens_per_second": 120}`
  - `{"type": "progress", "completed": 15, "total": 20}`
  - `{"type": "completed", "run_id": "...", "results_path": "..."}`
  - `{"type": "error", "error": "..."}`

### 2.3 Run page UI
- Shows active run with live progress bar
- Per-task status table updating in real-time:
  - Task ID | Model | Style | Context | Status | Score | TTFT | Tokens/s
- Running tasks show spinning indicator
- Completed tasks show score color-coded (green/yellow/red)
- Completed button navigates to results view
- "Copy YAML" of the config that was run

### 2.4 Run management
- `POST /api/runs/{run_id}/cancel` — signal runner to stop
- Runs table with filter: by model, suite, date range, status
- "Re-run this config" button

### 2.5 Multi-model runs
- UI allows selecting multiple models for one run
- Each model is a row in the results table
- Progress shows per-model completion

---

## Phase 3 — Results Viewer (Week 4)

### Goal: Interactive exploration of single-run results

### 3.1 Results data loading
- `GET /api/runs/{run_id}/results` — load results from SQLite database
- Parse `benchmark.db` using existing `sqlite.py` module
- Use `analysis.py` DataFrame builders for aggregations
- Cache parsed results in memory for the session

### 3.2 Results page UI

**Header section:**
- Run name, date, project, tags
- Config summary (model, suite, parameters)
- "Export" buttons (download JSONL, Markdown report, YAML)

**Executive summary cards:**
- Best scoring model (big number)
- Fastest model (lowest avg TTFT)
- Best quality/second (composite metric)
- Total tasks run / passed / failed
- Judge-averaged scores (if judge enabled)

**Model comparison table:**
- Rows: models | Cols: avg score, avg TTFT, avg tok/s, tasks passed, format failures, safety score
- Clickable to drill into per-model view

**Per-model tab views:**
- Task results table (expandable rows with raw response + explanation)
- Score distribution chart (histogram)
- Timing chart (bar: avg TTFT, decode, wall)
- Family breakdown bar chart (scores by task family)
- Failure details section

**Scoring breakdown:**
- Primary score per task with color coding
- Secondary scores (format compliance, safety, etc.)
- Judge dimensions if enabled (correctness, completeness, specificity, safety)

**Task detail modal:**
- Task ID and family
- Prompt rendered with template
- Raw model response (collapsible)
- Score and explanation
- Test output (for code tasks)
- Timing details

### 3.3 Report generation on demand
- `POST /api/runs/{run_id}/report` — generate Markdown report via existing `reports/markdown.py` or `reports/v2.py`
- Return rendered HTML for in-browser preview
- Download as `.md` file

---

## Phase 4 — Comparison Dashboard (Week 5)

### Goal: Side-by-side comparison of multiple runs

### 4.1 Comparison API
- `GET /api/runs` — filter and select runs for comparison
  - Filter by: project, date range, model, suite, tags
  - Multi-select runs
- `POST /api/compare` — accept list of run IDs
  - Loads results from both run databases
  - Returns comparison data:
    - Per-task score deltas
    - Regression detection (quality + performance)
    - Quality vs speed scatter plot data
    - Pareto frontier points

### 4.2 Comparison page UI

**Run selector:**
- Search/filter runs by name, date, model, suite
- Visual cards for each run showing mini-summary
- Multi-select up to 5 runs at once
- "Add baseline" button — marks one as reference

**Comparison view:**

*Delta table:*
- Rows: tasks | Cols: baseline score → candidate score, delta, flag (regression/stable/improved)
- Color coding: red arrows for regressions, green for improvements
- Sortable by delta magnitude
- Filter: show only regressions, show only improvements, show all

*Charts:*
- **Quality scatter:** x = baseline score, y = candidate score, dots = tasks, diagonal = no change, above diagonal = improvement
- **Speed scatter:** x = baseline TTFT, y = candidate TTFT (lower is better, inverted axis)
- **Speed/quality Pareto:** x = avg TTFT, y = avg score, dots = models, labeled
- **Family delta bars:** grouped bar chart, baseline vs candidate per task family
- **Context degradation curves:** if context sweep runs are compared
- **Quantization comparison:** if quantized models are compared

*Regression detail:*
- Table of flagged regressions (score delta beyond configurable threshold, default 10%)
- Each row: task, baseline score, candidate score, delta, task family, risk level
- Click to see prompt and both responses side by side

*Summary section:*
- "Candidate is faster by X%" / "Candidate is slower by X%"
- "Candidate scored higher on Y tasks, lower on Z"
- "Key regression: [task name] dropped from A to B"
- "Best kept: [task family] score unchanged"

### 4.3 Comparison presets
- "Compare latest vs baseline" — one-click
- "Compare all runs this week" — grouped by day
- "Compare quantization variants" — auto-detects runs with different quantization field
- "Compare prompt styles" — auto-detects style sweep runs

### 4.4 Save comparison
- `POST /api/comparisons` — save a comparison with a name
- `GET /api/comparisons` — list saved comparisons
- Comparison includes the run IDs, filters used, and generated charts data
- Sharable via URL (encoded in query string for small comparisons)

---

## Phase 5 — Config & Run History (Week 6)

### Goal: Full lifecycle management of configs, runs, and comparisons

### 5.1 Config → Run audit trail
- Each run record links back to the config ID it was created from
- Config page shows "Ran 12 times" with last run date and avg scores
- Config version shows diff when editing

### 5.2 Run history
- Calendar view or timeline of runs
- Click a day → shows all runs that day
- Click a run → goes to results viewer
- "Trend" mini-chart: avg score over time for a model

### 5.3 Export from UI
- `GET /api/runs/{run_id}/export/{format}` — trigger export
  - formats: `sft`, `preference`, `regression`, `judge`
  - Returns download link for the export file
- "Export all recent runs" — batch export for training data

### 5.4 Run scheduling (lightweight)
- "Schedule recurring run" UI (simple cron-like interface)
- Stores schedule in a JSON file
- Background thread checks every minute for scheduled runs
- Not a full scheduler — just "run every Friday at 6pm against agent-code"

### 5.5 Notifications (optional, nice-to-have)
- Email or Slack webhook on run completion
- "Run failed" alerts
- Configurable per-schedule

---

## Phase 6 — Polish & Deployment (Week 7)

### 6.1 UI polish
- Responsive design (works on laptop, not just desktop)
- Dark mode toggle
- Keyboard shortcuts: `n` = new config, `r` = run, `c` = compare
- Search/filter across configs, runs, comparisons
- Breadcrumbs and URL state (shareable URLs for specific configs/runs/comparisons)

### 6.2 Security
- API key or basic auth for self-hosting
- CORS configured for localhost + optionally remote
- No file upload — all input is JSON from forms
- Storage root safety checks (refuse dangerous paths from UI)

### 6.3 Deployment
- `bench-harness web-ui` starts the server
- Configurable host/port: `bench-harness web-ui --host 0.0.0.0 --port 8080`
- Dockerfile for containerized deployment
- Reverse proxy example (nginx or Caddy) for remote access

### 6.4 Documentation
- Inline help tooltips on every form field
- `/docs` with API documentation (Flask openapi or simple markdown)
- Quick-start guide in `server/README.md`

---

## File Structure (New Code)

```
src/bench_harness/server/
  __init__.py          # Server init, app factory
  app.py               # Flask app creation, blueprints
  routes/
    __init__.py
    configs.py         # Config CRUD endpoints
    runs.py            # Run CRUD + launch endpoints
    results.py         # Results query + aggregation
    compare.py         # Comparison endpoints
    models.py          # Model/suite/scorer metadata
    export.py          # Export trigger + download
    ws.py              # WebSocket endpoints
  templates/           # Jinja2 HTML templates
    base.html          # Layout, nav, footer
    config_builder.html
    config_list.html
    config_edit.html
    runs_list.html
    run_detail.html
    results.html
    compare.html
    comparisons_list.html
  static/
    css/
      main.css
      forms.css
      tables.css
      charts.css
      run_progress.css
    js/
      app.js           # SPA routing (minimal, or multi-page with Jinja)
      config_builder.js
      runs.js
      results.js
      compare.js
      ws.js            # WebSocket client
  services/
    __init__.py
    runner_service.py  # Wraps bench_harness runner, manages run lifecycle
    config_service.py  # CRUD + validation for configs
    results_service.py # Queries SQLite, returns structured data
    compare_service.py # Loads two+ runs, computes deltas
    export_service.py  # Triggers bench_harness exports
    ws_service.py      # Manages WebSocket connections + broadcast
  models/
    __init__.py
    schemas.py         # Request/response Pydantic models
    responses.py        # Standardized API response helpers
  utils/
    __init__.py
    storage.py         # Storage root resolution helpers
    security.py        # Auth, CORS, path safety
  __main__.py          # Entry point: python -m bench_harness.server
```

---

## Dependencies Addition

```toml
[project.optional-dependencies]
web = [
    "flask[async]",
    "flask-cors",
]
```

Entry point addition:
```toml
[project.scripts]
bench-harness = "bench_harness.cli:main"
llm-bench = "llm_bench.cli:main"
bench-harness-web = "bench_harness.server.__main__:main"
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Flask is synchronous, runs are blocking | High | Use `threading` for background run execution; WebSocket for status updates. For larger scale, migrate to `flask[async]` + `asyncio` executor. |
| Existing runner uses Rich for terminal output | Medium | Runner output goes to log files, not stdout. Server captures logs and serves via `/api/runs/{id}/logs`. |
| Large result sets overwhelm browser | Low-Medium | Paginate results table. Load full data on demand per model. Use server-side aggregation. |
| Config YAML serialization complexity | Medium | Delegate to existing `RunSpec.to_yaml()` and `RunSpec.from_yaml()`. Server only handles JSON ↔ Pydantic conversion. |
| WebSocket connection drops during long runs | Medium | Reconnect logic in JS client. Store all events in a JSONL log per run for replay. |
| Self-hosting security | Low | Basic auth + API key from day one. Storage root safety checks baked in. No file uploads from browser. |

---

## What This Does NOT Do (Out of Scope)

- **Model serving** — does not start/stop vLLM, llama.cpp, or TRT-LLM servers. The UI assumes an endpoint is already running (just like the CLI does).
- **Model downloads** — no HF model downloading or checkpoint management.
- **GPU monitoring** — no real-time GPU metrics in the UI (that's a later extension).
- **Multi-user / RBAC** — basic auth is enough for a single researcher. Full auth is future work.
- **Scheduled background runs** — lightweight cron is included, not a full job scheduler with queues.

---

## Phased Rollout Strategy

**Sprint 1 (Phase 0-1):** Can create and save configs from the UI. Still run from CLI for actual execution.

**Sprint 2 (Phase 2):** Can launch runs from UI. Live progress in browser. Replaces `bench-harness run <spec>`.

**Sprint 3 (Phase 3-4):** Can view results and compare runs from UI. Replaces `bench-harness analyze notebook` for 90% of use cases.

**Sprint 4 (Phase 5-6):** Full lifecycle — config → run → results → compare → export → schedule, all from browser.

Each sprint is independently valuable: a saved config is useful even without UI runs, a results viewer is useful even without live progress.
