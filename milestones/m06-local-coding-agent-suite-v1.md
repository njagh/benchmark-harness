# Milestone 6 — Local Coding-Agent Suite v1

## Goal

Create a high-value benchmark based on real workflows with 25 local tasks across five families, each with defined prompts, scoring, and risk levels. At least 10 tasks have executable validation.

## Phase

Phase B — Real coding usefulness (Milestone 4 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap)
- Milestone 2 (task schema, prompt templates)
- Milestone 3 (timing metrics)
- Milestone 4 (basic scorers)
- Milestone 5 (coding task runner with test execution)

---

## Subtasks

### 6.1 Create task directory structure

**Directory:** `tasks/local_coding_agent_v1/`

```
tasks/local_coding_agent_v1/
  docker_compose/
    fix_yaml_001.yaml ... fix_yaml_005.yaml
  litellm_routing/
    routing_001.yaml ... routing_005.yaml
  qwen3_debug/
    debug_001.yaml ... debug_005.yaml
  benchmark_script/
    benchmark_001.yaml ... benchmark_005.yaml
  git_linux/
    git_001.yaml ... git_002.yaml
    linux_001.yaml ... linux_003.yaml
```

**Actions:**
- [x] Create all 5 subdirectories under `tasks/local_coding_agent_v1/`
- [x] Document task naming convention

### 6.2 Create Docker Compose / YAML repair tasks (5 tasks)

**Directory:** `tasks/local_coding_agent_v1/docker_compose/`

**Tasks:**
- [x] `fix_yaml_001.yaml` — Fix invalid port mapping and redis connection in docker-compose.yml
- [x] `fix_yaml_002.yaml` — Fix YAML syntax error in docker-compose (misaligned key)
- [x] `fix_yaml_003.yaml` — Fix healthcheck configuration (wrong command, wrong interval)
- [x] `fix_yaml_004.yaml` — Fix volume mount path mismatch
- [x] `fix_yaml_005.yaml` — Add healthcheck and restart policies to service

**Scoring:** `primary: contains` with patterns checking for correct ports, healthchecks, restart policies, and absence of invalid values.

### 6.3 Create LiteLLM / model routing tasks (5 tasks)

**Directory:** `tasks/local_coding_agent_v1/litellm_routing/`

**Tasks:**
- [x] `routing_001.yaml` — Write LiteLLM router config with routing tables and fallback
- [x] `routing_002.yaml` — Add fallback model routing with retry_on_timeout
- [x] `routing_003.yaml` — Fix API key configuration (move to env var reference)
- [x] `routing_004.yaml` — Configure rate limiting (100 rpm, burst of 20)
- [x] `routing_005.yaml` — Fix CORS configuration for Open WebUI frontend

**Scoring:** `primary: contains` with patterns for required config keys.

### 6.4 Create qwen3_replicate debugging tasks (5 tasks)

**Directory:** `tasks/local_coding_agent_v1/qwen3_debug/`

**Tasks:**
- [x] `debug_001.yaml` — Fix CUDA OOM config for 72B model (vram_limit, dtype, batch size)
- [x] `debug_002.yaml` — Fix RMSNorm parity bug (missing epsilon, wrong normalization)
- [x] `debug_003.yaml` — Fix RoPE layout bug (dimension mismatch in frequencies)
- [x] `debug_004.yaml` — Fix attention mask (upper triangle should be -inf not 0)
- [x] `debug_005.yaml` — Fix HF weight loader name mapping

**Scoring:** `primary: contains` with patterns for required configuration elements.

### 6.5 Create benchmark-script modification tasks (5 tasks)

**Directory:** `tasks/local_coding_agent_v1/benchmark_script/`

**Tasks:**
- [x] `benchmark_001.yaml` — Fix benchmark runner: add error handling, retries, timeout
- [x] `benchmark_002.yaml` — Add throughput metric (requests per second) to output
- [x] `benchmark_003.yaml` — Fix token accounting (characters → tokens)
- [x] `benchmark_004.yaml` — Separate prefill vs decode time reporting
- [x] `benchmark_005.yaml` — Add CSV export to benchmark results

**Scoring:** `primary: contains` with patterns for required features.

### 6.6 Create Git/Linux troubleshooting tasks (5 tasks)

**Directory:** `tasks/local_coding_agent_v1/git_linux/`

**Tasks:**
- [x] `git_001.yaml` — Fix git pre-commit hook (regex too strict, allow conventional commits)
- [x] `git_002.yaml` — Resolve merge conflict in main.py keeping both changes
- [x] `linux_001.yaml` — Fix broken git remote URL (SSH → HTTPS)
- [x] `linux_002.yaml` — Diagnose permission denied on script (chmod +x)
- [x] `linux_003.yaml` — Fix Docker build context error (Dockerfile in subdirectory)

**Scoring:** `primary: contains` with patterns for required commands and configurations.

### 6.7 Update CLI task discovery

**File:** `src/bench_harness/cli.py`

**Actions:**
- [x] Add `tasks/local_coding_agent_v1/` to `list-tasks` auto-discovery list
- [x] Add `tasks/local_coding_agent_v1/` to `show-task` discovery list
- [x] Add `--family` option supporting comma-separated values (e.g., `--family docker_compose,git_linux`)

### 6.8 Add Coding Agent Ranking to Markdown report

**File:** `src/bench_harness/reports/markdown.py`

**Actions:**
- [x] Add `_group_by_family()` helper function for family-level grouping
- [x] Add `_extract_family_from_task_id()` helper for family extraction from task IDs
- [x] Add `_append_coding_agent_ranking()` function
- [x] Append "## Coding Agent Ranking" section when code_type is present in runs
- [x] Append "## Family Breakdown" section with per-family stats
- [x] Gracefully skip these sections for runs without code tasks

### 6.9 Mark milestone as complete

**Files:**
- [x] Update `ROADMAP.md` — mark M6 as **✅ DONE**
- [x] Update `README.md` — change status line to show M1–M6 complete
- [x] Update README milestone table: M5 to **Done**, M6 to **Done**

---

## Acceptance Criteria

- [x] 25 local tasks defined across 5 families (5 per family)
- [x] Each task has prompt, scoring config, risk level, and `code_type`
- [x] At least 10 tasks have executable validation (`code_type: patch_generation`)
- [x] `bench-harness list-tasks` discovers all 25 tasks
- [x] `bench-harness list-tasks --family docker_compose` filters correctly
- [x] Markdown report includes "Coding Agent Ranking" with per-model scores
- [x] Markdown report includes "Family Breakdown" with per-family scores
- [x] Report ranking based on test pass rate for code tasks, pattern match for config tasks

## Estimated Effort

3–4 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `tasks/local_coding_agent_v1/docker_compose/fix_yaml_00{1,2,3,4,5}.yaml` (×5) | Done |
| `tasks/local_coding_agent_v1/litellm_routing/routing_00{1,2,3,4,5}.yaml` (×5) | Done |
| `tasks/local_coding_agent_v1/qwen3_debug/debug_00{1,2,3,4,5}.yaml` (×5) | Done |
| `tasks/local_coding_agent_v1/benchmark_script/benchmark_00{1,2,3,4,5}.yaml` (×5) | Done |
| `tasks/local_coding_agent_v1/git_linux/git_00{1,2}.yaml` (×2) | Done |
| `tasks/local_coding_agent_v1/git_linux/linux_00{1,2,3}.yaml` (×3) | Done |
| `src/bench_harness/cli.py` — add discovery and --family filter | Done |
| `src/bench_harness/reports/markdown.py` — add ranking sections | Done |
| `ROADMAP.md` — mark M6 done | Done |
| `README.md` — update status and milestone table | Done |
