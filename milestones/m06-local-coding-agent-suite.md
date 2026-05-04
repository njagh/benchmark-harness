# Milestone 6 — Local Coding-Agent Suite v1

## Goal

Create a high-value benchmark based on real workflows with at least 25 local tasks across five categories, each with defined prompts, context files, expected behavior, scorers, and risk levels. At least 10 tasks must have executable validation.

## Phase

Phase B — Real coding usefulness (Milestone 4 of 4 in phase)

## Dependencies

- Milestone 1 (project bootstrap)
- Milestone 2 (task schema, prompt templates)
- Milestone 3 (timing metrics)
- Milestone 4 (basic scorers)
- Milestone 5 (coding task runner with test execution)

### Leveraged Libraries and Data Access

**Public benchmark integration (lm-eval):**
The roadmap (§6.2) references IFEval, MMLU-Pro, GPQA, BBH, MATH, HumanEval, and MBPP. Rather than hand-writing task YAMLs for these, the harness should integrate with **lm-evaluation-harness** (https://github.com/EleutherAI/lm-evaluation-harness) where practical:

1. Run public benchmarks through `lm-eval` directly (it handles prompt templates, tokenization, and scoring)
2. Store summarized metrics in the harness SQLite database
3. Use the harness's unified report generator to present public + local results together
4. This avoids duplicating thousands of benchmark tasks and their scoring logic

**Dataset access (STORAGE_PLAN):**
- All external benchmark data follows the STORAGE_PLAN.md layout: `~/datasets/evals/` for pinned local copies
- Never stream from HuggingFace during measured runs
- Each dataset has a manifest with source, revision, checksum, and sample metadata
- The harness reads only from registered local datasets (configs/datasets.yaml)

**eval_plus for coding tests:**
- Use `evalplus` (https://github.com/evalplus/eval_plus) for HumanEval+ and MBPP+ test suites
- These are deobfuscated versions with stronger tests than the originals
- Download test files once to `~/datasets/evals/evalplus/` and reference locally

---

## Subtasks

### 6.1 Define task directory structure

**Directory:** `tasks/local/`

```
tasks/local/
  docker_compose/
    task_001.yaml
    task_002.yaml
    task_003.yaml
    task_004.yaml
    task_005.yaml
    fixtures/
      docker-compose-001.yml
      docker-compose-002.yml
      ...
    tests/
      test_compose_001.py
      ...
  litellm/
    task_001.yaml
    ...
    fixtures/
      litellm-config-001.yaml
      ...
    tests/
      ...
  qwen3_replicate/
    task_001.yaml
    ...
    fixtures/
      model_implementation_buggy.py
      ...
    tests/
      ...
  benchmark_scripts/
    task_001.yaml
    ...
    fixtures/
      benchmark_script_v1.py
      ...
    tests/
      ...
  git_linux/
    task_001.yaml
    ...
    fixtures/
      ...
    tests/
      ...
```

**Actions:**
- [ ] Create all subdirectories
- [ ] Create `.gitkeep` files in fixture/test directories
- [ ] Document task naming convention: `task_NNN.yaml` within each category

### 6.2 Create Docker Compose / YAML repair tasks (5 tasks)

**Category:** `docker_compose`
**File prefix:** `tasks/local/docker_compose/`

**Task 6.2.1 — `task_001.yaml`: Fix YAML syntax error**

```yaml
id: local.docker_compose.fix_yaml_syntax_001
family: shell_debugging
category: docker
version: "1.0"
source: local
prompt_template: docker_compose_debug.md
input:
  user_message: |
    docker compose up fails with:
    yaml: line 4: did not find expected key
    Here's the docker-compose.yml:
  files:
    - "tasks/local/docker_compose/fixtures/docker-compose-001.yml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/docker_compose/tests/test_compose_001.py"
  test_command: "python -m pytest {test_file} -v"
  timeout_seconds: 10
  allowed_file_changes:
    - "docker-compose.yml"
scoring:
  primary: unit_tests
  secondary:
    - patch_minimality
risk_level: medium
context_tokens: small
```

**Fixture:** `docker-compose-001.yml` — a docker-compose file with intentional YAML syntax error (misaligned key, missing colon, etc.)

**Test:** `test_compose_001.py` — validates that the fixed YAML parses correctly and contains required services

**Actions:**
- [ ] Write task_001.yaml
- [ ] Write fixture docker-compose-001.yml with syntax error
- [ ] Write test_compose_001.py that validates fixed YAML

**Task 6.2.2 — `task_002.yaml`: Fix port conflict**

```yaml
id: local.docker_compose.fix_port_conflict_002
family: shell_debugging
category: docker
prompt_template: docker_compose_debug.md
input:
  user_message: |
    Two services in docker-compose.yml are both trying to bind port 8080.
    Fix the conflict by reassigning one service to port 8081.
  files:
    - "tasks/local/docker_compose/fixtures/docker-compose-002.yml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/docker_compose/tests/test_compose_002.py"
scoring:
  primary: unit_tests
risk_level: medium
```

**Actions:**
- [ ] Write task_002.yaml
- [ ] Write fixture with port conflict
- [ ] Write test that validates distinct ports

**Task 6.2.3 — `task_003.yaml`: Fix volume mount path**

```yaml
id: local.docker_compose.fix_volume_mount_003
family: shell_debugging
category: docker
prompt_template: docker_compose_debug.md
input:
  user_message: |
    The container can't find the mounted config file. The volume mount
    path is wrong. Fix it.
  files:
    - "tasks/local/docker_compose/fixtures/docker-compose-003.yml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/docker_compose/tests/test_compose_003.py"
scoring:
  primary: unit_tests
risk_level: low
```

**Actions:**
- [ ] Write task_003.yaml
- [ ] Write fixture with wrong volume path
- [ ] Write test validating correct path

**Task 6.2.4 — `task_004.yaml`: Fix network configuration**

```yaml
id: local.docker_compose.fix_network_004
family: shell_debugging
category: docker
prompt_template: docker_compose_debug.md
input:
  user_message: |
    Service 'api' can't reach service 'db'. The network configuration
    is incorrect. Fix the networking so they can communicate.
  files:
    - "tasks/local/docker_compose/fixtures/docker-compose-004.yml"
expected:
  type: contains
  patterns:
    - "networks"
  absent_patterns:
    - "network_mode: host"
scoring:
  primary: regex
  secondary:
    - contains
risk_level: low
```

**Actions:**
- [ ] Write task_004.yaml
- [ ] Write fixture with broken network config
- [ ] Scorer uses regex/contains (no executable test)

**Task 6.2.5 — `task_005.yaml`: Add healthcheck to service**

```yaml
id: local.docker_compose.add_healthcheck_005
family: shell_debugging
category: docker
prompt_template: docker_compose_debug.md
input:
  user_message: |
    Add a healthcheck to the 'api' service that curls localhost:8000/health
    every 30 seconds with 3 retries.
  files:
    - "tasks/local/docker_compose/fixtures/docker-compose-005.yml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/docker_compose/tests/test_compose_005.py"
scoring:
  primary: unit_tests
risk_level: low
```

**Actions:**
- [ ] Write task_005.yaml
- [ ] Write fixture without healthcheck
- [ ] Write test validating healthcheck presence

### 6.3 Create LiteLLM / model routing tasks (5 tasks)

**Category:** `litellm`
**File prefix:** `tasks/local/litellm/`

**Task 6.3.1 — `task_001.yaml`: Fix model alias mapping**

```yaml
id: local.litellm.fix_model_alias_001
family: shell_debugging
category: litellm
prompt_template: litellm_debug.md
input:
  user_message: |
    Requests to model 'my-agent' are failing with 'model not found'.
    Fix the LiteLLM config to properly route 'my-agent' to the
    vLLM endpoint at http://localhost:4000/v1 using model 'agent-code'.
  files:
    - "tasks/local/litellm/fixtures/litellm-config-001.yaml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/litellm/tests/test_litellm_001.py"
scoring:
  primary: unit_tests
risk_level: medium
```

**Actions:**
- [ ] Write task_001.yaml
- [ ] Write fixture with broken alias mapping
- [ ] Write test validating correct routing config

**Task 6.3.2 — `task_002.yaml`: Add fallback model routing**

```yaml
id: local.litellm.add_fallback_routing_002
family: shell_debugging
category: litellm
prompt_template: litellm_debug.md
input:
  user_message: |
    Add fallback routing so that if the primary model times out,
    requests fall back to the secondary model. Configure retry_on_timeout.
  files:
    - "tasks/local/litellm/fixtures/litellm-config-002.yaml"
expected:
  type: contains
  patterns:
    - "retry_on_timeout"
    - "fallback"
scoring:
  primary: regex
risk_level: low
```

**Actions:**
- [ ] Write task_002.yaml and fixture
- [ ] Scorer uses regex (no executable test)

**Task 6.3.3 — `task_003.yaml`: Fix API key configuration**

```yaml
id: local.litellm.fix_api_key_003
family: shell_debugging
category: litellm
prompt_template: litellm_debug.md
input:
  user_message: |
    The LiteLLM config has the API key hardcoded. Move it to an
    environment variable reference.
  files:
    - "tasks/local/litellm/fixtures/litellm-config-003.yaml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/litellm/tests/test_litellm_003.py"
scoring:
  primary: unit_tests
  secondary:
    - contains
risk_level: medium
```

**Actions:**
- [ ] Write task_003.yaml
- [ ] Write fixture with hardcoded API key
- [ ] Write test validating env var reference

**Task 6.3.4 — `task_004.yaml`: Configure rate limiting**

```yaml
id: local.litellm.configure_rate_limit_004
family: shell_debugging
category: litellm
prompt_template: litellm_debug.md
input:
  user_message: |
    Add rate limiting to the LiteLLM config: 100 requests per minute
    per key, with a burst of 20.
  files:
    - "tasks/local/litellm/fixtures/litellm-config-004.yaml"
expected:
  type: contains
  patterns:
    - "rpm"
    - "100"
scoring:
  primary: regex
risk_level: low
```

**Actions:**
- [ ] Write task_004.yaml and fixture
- [ ] Scorer uses regex

**Task 6.3.5 — `task_005.yaml`: Fix cross-origin routing**

```yaml
id: local.litellm.fix_cors_routing_005
family: shell_debugging
category: litellm
prompt_template: litellm_debug.md
input:
  user_message: |
    The Open WebUI frontend can't reach the LiteLLM API due to CORS.
    Fix the LiteLLM config to allow requests from http://localhost:3000.
  files:
    - "tasks/local/litellm/fixtures/litellm-config-005.yaml"
expected:
  type: unit_test
  test_files:
    - "tasks/local/litellm/tests/test_litellm_005.py"
scoring:
  primary: unit_tests
risk_level: medium
```

**Actions:**
- [ ] Write task_005.yaml
- [ ] Write fixture with missing CORS config
- [ ] Write test validating CORS headers

### 6.4 Create qwen3_replicate debugging tasks (5 tasks)

**Category:** `qwen3_replicate`
**File prefix:** `tasks/local/qwen3_replicate/`

**Task 6.4.1 — `task_001.yaml`: RMSNorm parity bug**

```yaml
id: local.qwen3_replicate.rmsnorm_bug_001
family: coding
category: debugging
prompt_template: debug_task.md
input:
  user_message: |
    The RMSNorm implementation doesn't match the reference. The output
    differs by more than 1e-5 on standard inputs. Find and fix the bug.
  files:
    - "tasks/local/qwen3_replicate/fixtures/rmsnorm_buggy.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/qwen3_replicate/tests/test_rmsnorm.py"
  test_command: "python -m pytest {test_file} -v"
  timeout_seconds: 30
scoring:
  primary: unit_tests
risk_level: high
```

**Actions:**
- [ ] Write task_001.yaml
- [ ] Write fixture with RMSNorm bug (e.g., missing epsilon, wrong normalization order)
- [ ] Write test that compares against reference implementation with tolerance

**Task 6.4.2 — `task_002.yaml`: RoPE layout bug**

```yaml
id: local.qwen3_replicate.rope_layout_bug_002
family: coding
category: debugging
prompt_template: debug_task.md
input:
  user_message: |
    The RoPE (rotary positional embedding) implementation has a tensor
    layout mismatch. The frequencies are applied to the wrong dimensions.
    Fix the bug.
  files:
    - "tasks/local/qwen3_replicate/fixtures/rope_buggy.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/qwen3_replicate/tests/test_rope.py"
scoring:
  primary: unit_tests
risk_level: high
```

**Actions:**
- [ ] Write task_002.yaml
- [ ] Write fixture with dimension mismatch in RoPE
- [ ] Write test with known input/output pairs

**Task 6.4.3 — `task_003.yaml`: GQA cache shape bug**

```yaml
id: local.qwen3_replicate.gqa_cache_bug_003
family: coding
category: debugging
prompt_template: debug_task.md
input:
  user_message: |
    The KV cache for Grouped-Query Attention has the wrong shape.
    The key cache should be [batch, num_kv_heads, seq_len, head_dim]
    but is currently [batch, seq_len, num_kv_heads, head_dim].
    Find and fix all affected code.
  files:
    - "tasks/local/qwen3_replicate/fixtures/attention_gqa_buggy.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/qwen3_replicate/tests/test_gqa_cache.py"
scoring:
  primary: unit_tests
risk_level: high
```

**Actions:**
- [ ] Write task_003.yaml
- [ ] Write fixture with transposed cache dimensions
- [ ] Write test validating cache shapes

**Task 6.4.4 — `task_004.yaml`: Attention mask bug**

```yaml
id: local.qwen3_replicate.attention_mask_bug_004
family: coding
category: debugging
prompt_template: debug_task.md
input:
  user_message: |
    The causal attention mask is allowing future tokens to be attended
    to. The upper triangle should be -inf but currently contains zeros.
    Fix the mask creation.
  files:
    - "tasks/local/qwen3_replicate/fixtures/attention_mask_buggy.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/qwen3_replicate/tests/test_attention_mask.py"
scoring:
  primary: unit_tests
risk_level: high
```

**Actions:**
- [ ] Write task_004.yaml
- [ ] Write fixture with broken causal mask
- [ ] Write test checking upper triangle values

**Task 6.4.5 — `task_005.yaml`: HF weight loader bug**

```yaml
id: local.qwen3_replicate.hf_loader_bug_005
family: coding
category: debugging
prompt_template: debug_task.md
input:
  user_message: |
    The HuggingFace weight loader is mis-mapping parameter names.
    The model expects 'q_proj.weight' but the HF checkpoint has
    'wq.weight'. Fix the name mapping.
  files:
    - "tasks/local/qwen3_replicate/fixtures/weight_loader_buggy.py"
expected:
  type: contains
  patterns:
    - "wq.weight"
    - "q_proj.weight"
scoring:
  primary: regex
risk_level: medium
```

**Actions:**
- [ ] Write task_005.yaml
- [ ] Write fixture with wrong name mapping
- [ ] Scorer uses regex (no executable test — weight loading requires full model)

### 6.5 Create benchmark-script modification tasks (5 tasks)

**Category:** `benchmark_scripts`
**File prefix:** `tasks/local/benchmark_scripts/`

**Task 6.5.1 — `task_001.yaml`: Add throughput metric**

```yaml
id: local.benchmark.add_throughput_001
family: coding
category: benchmark
prompt_template: benchmark_modify.md
input:
  user_message: |
    This benchmark script measures latency but not throughput.
    Add a throughput calculation (requests per second) to the output.
  files:
    - "tasks/local/benchmark_scripts/fixtures/benchmark_v1.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/benchmark_scripts/tests/test_throughput.py"
scoring:
  primary: unit_tests
risk_level: low
```

**Actions:**
- [ ] Write task_001.yaml
- [ ] Write fixture benchmark script without throughput
- [ ] Write test validating throughput output

**Task 6.5.2 — `task_002.yaml`: Fix token accounting**

```yaml
id: local.benchmark.fix_token_accounting_002
family: coding
category: benchmark
prompt_template: benchmark_modify.md
input:
  user_message: |
    The benchmark script's token counting is off. It's counting
    characters instead of tokens. Fix the token accounting.
  files:
    - "tasks/local/benchmark_scripts/fixtures/benchmark_token_buggy.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/benchmark_scripts/tests/test_token_count.py"
scoring:
  primary: unit_tests
risk_level: medium
```

**Actions:**
- [ ] Write task_002.yaml
- [ ] Write fixture with character counting instead of token counting
- [ ] Write test validating token count

**Task 6.5.3 — `task_003.yaml`: Parse vLLM timing output**

```yaml
id: local.benchmark.parse_vllm_timing_003
family: coding
category: benchmark
prompt_template: benchmark_modify.md
input:
  user_message: |
    Write a function that parses vLLM benchmark output and extracts
    TTFT, decode time, and tokens/sec. The output format is:
    "TTFT: 120.5ms | Decode: 45.2ms | Tokens/sec: 234.5"
  files: []
expected:
  type: unit_test
  test_files:
    - "tasks/local/benchmark_scripts/tests/test_parse_vllm.py"
scoring:
  primary: unit_tests
risk_level: low
```

**Actions:**
- [ ] Write task_003.yaml
- [ ] Write test with sample vLLM output strings

**Task 6.5.4 — `task_004.yaml`: Compare prefill vs decode performance**

```yaml
id: local.benchmark.compare_prefill_decode_004
family: coding
category: benchmark
prompt_template: benchmark_modify.md
input:
  user_message: |
    Modify this benchmark to separately report prefill (prompt processing)
    time and decode (token generation) time, and compute the ratio.
  files:
    - "tasks/local/benchmark_scripts/fixtures/benchmark_prefill_v1.py"
expected:
  type: contains
  patterns:
    - "prefill"
    - "decode"
    - "ratio"
scoring:
  primary: regex
risk_level: low
```

**Actions:**
- [ ] Write task_004.yaml
- [ ] Write fixture benchmark script
- [ ] Scorer uses regex

**Task 6.5.5 — `task_005.yaml`: Add CSV export to benchmark**

```yaml
id: local.benchmark.add_csv_export_005
family: coding
category: benchmark
prompt_template: benchmark_modify.md
input:
  user_message: |
    Add CSV export to this benchmark script. Each row should have:
    model, prompt_tokens, completion_tokens, ttft_ms, tokens_per_sec.
  files:
    - "tasks/local/benchmark_scripts/fixtures/benchmark_no_export.py"
expected:
  type: unit_test
  test_files:
    - "tasks/local/benchmark_scripts/tests/test_csv_export.py"
scoring:
  primary: unit_tests
risk_level: low
```

**Actions:**
- [ ] Write task_005.yaml
- [ ] Write fixture without CSV export
- [ ] Write test validating CSV output format

### 6.6 Create Git/Linux troubleshooting tasks (5 tasks)

**Category:** `git_linux`
**File prefix:** `tasks/local/git_linux/`

**Task 6.6.1 — `task_001.yaml`: Resolve merge conflict**

```yaml
id: local.git_linux.resolve_merge_conflict_001
family: shell_debugging
category: git
prompt_template: git_troubleshooting.md
input:
  user_message: |
    I have a merge conflict in main.py. Both branches modified the
    same function. Here's the conflicted file. Show the correct
    resolution that keeps both changes.
  files:
    - "tasks/local/git_linux/fixtures/main_conflicted.py"
expected:
  type: contains
  patterns:
    - "def feature_a"
    - "def feature_b"
  absent_patterns:
    - "<<<<<<"
    - "====="
    - ">>>>>>"
scoring:
  primary: regex
  secondary:
    - contains
risk_level: medium
```

**Actions:**
- [ ] Write task_001.yaml
- [ ] Write fixture with conflict markers
- [ ] Scorer uses regex/contains

**Task 6.6.2 — `task_002.yaml`: Fix broken git remote URL**

```yaml
id: local.git_linux.fix_remote_url_002
family: shell_debugging
category: git
prompt_template: git_troubleshooting.md
input:
  user_message: |
    git push fails with 'repository not found'. The remote URL is
    using SSH but should use HTTPS. Show the command to fix this.
  files: []
expected:
  type: contains
  patterns:
    - "git remote set-url"
scoring:
  primary: contains
risk_level: low
```

**Actions:**
- [ ] Write task_002.yaml
- [ ] Scorer uses contains (no executable test)

**Task 6.6.3 — `task_003.yaml`: Diagnose permission denied on file**

```yaml
id: local.git_linux.diagnose_permission_003
family: shell_debugging
category: linux
prompt_template: linux_troubleshooting.md
input:
  user_message: |
    I get 'Permission denied' when running python script.py.
    The file exists and has content. Diagnose and fix.
  files: []
expected:
  type: contains
  patterns:
    - "chmod"
    - "+x"
scoring:
  primary: contains
risk_level: low
```

**Actions:**
- [ ] Write task_003.yaml
- [ ] Scorer uses contains

**Task 6.6.4 — `task_004.yaml`: Fix Docker build context error**

```yaml
id: local.git_linux.fix_docker_build_004
family: shell_debugging
category: docker
prompt_template: linux_troubleshooting.md
input:
  user_message: |
    docker build fails with 'failed to solve: failed to read dockerfile'.
    The Dockerfile is in a subdirectory 'deploy/'. Show the correct
    docker build command.
  files: []
expected:
  type: contains
  patterns:
    - "-f"
    - "deploy"
scoring:
  primary: contains
risk_level: low
```

**Actions:**
- [ ] Write task_004.yaml
- [ ] Scorer uses contains

**Task 6.6.5 — `task_005.yaml`: Recover lost git commit**

```yaml
id: local.git_linux.recover_commit_005
family: shell_debugging
category: git
prompt_template: git_troubleshooting.md
input:
  user_message: |
    I accidentally ran git reset --hard and lost my last 3 commits.
    How do I recover them?
  files: []
expected:
  type: contains
  patterns:
    - "git reflog"
scoring:
  primary: contains
  secondary:
    - regex
risk_level: low
```

**Actions:**
- [ ] Write task_005.yaml
- [ ] Scorer uses contains/regex

### 6.7 Create prompt templates for local tasks

**Directory:** `configs/prompt_templates/` (extend)

**New templates:**
- `docker_compose_debug.md` — for Docker Compose debugging tasks
- `litellm_debug.md` — for LiteLLM configuration tasks
- `debug_task.md` — for code debugging tasks
- `benchmark_modify.md` — for benchmark script modification tasks
- `git_troubleshooting.md` — for Git troubleshooting tasks
- `linux_troubleshooting.md` — for Linux troubleshooting tasks

**Template content example (`docker_compose_debug.md`):**
```markdown
{{ system_message | default("You are an expert DevOps engineer. Diagnose and fix the issue below. Output only the fix as a unified diff or corrected file.") }}

{{ user_message }}

{% for file in files %}
--- File: {{ file.name }} ---
{{ file.content }}
{% endfor %}
```

**Actions:**
- [ ] Create all 6 prompt templates
- [ ] Ensure templates support file context injection
- [ ] Test template rendering with sample context

### 6.8 Add suite configuration for local coding-agent suite

**File:** `configs/suites.yaml` (update)

```yaml
suites:
  # ... existing ...
  local_coding_agent:
    description: "Real-workflow coding agent benchmark"
    task_dirs:
      - "tasks/local/docker_compose"
      - "tasks/local/litellm"
      - "tasks/local/qwen3_replicate"
      - "tasks/local/benchmark_scripts"
      - "tasks/local/git_linux"
    max_concurrency: 2
    default_runs: 1
    default_temperature: 0
    runner: auto       # code_runner for tasks with test_files, completion_runner otherwise
```

**Actions:**
- [ ] Add suite config with multiple task_dirs
- [ ] Update config loader to handle `task_dirs` (list of directories)
- [ ] Implement `runner: auto` logic — checks if task has `test_files` to decide runner

### 6.9 Add composite scoring for coding-agent

**File:** `src/bench_harness/scorers/composite.py`

**Class:** `CompositeScorer(BaseScorer)`

```python
class CompositeScorer(BaseScorer):
    name = "composite"

    def score(self, task: Task, raw_response: str, primary_result: ScoreResult, secondary_results: dict[str, ScoreResult]) -> ScoreResult:
        """Compute weighted composite score from primary + secondary scorers."""
```

**Behavior:**
- Weights come from `task.scoring.weights`
- Coding-agent composite weights:
  ```yaml
  weights:
    unit_tests: 0.50
    patch_minimality: 0.20
    format_compliance: 0.15
    contains: 0.10
    regex: 0.05
  ```
- Composite score = sum(weight_i * score_i) for scorers that produced results
- Normalize so scores sum to 1.0

**Actions:**
- [ ] Implement `CompositeScorer`
- [ ] Register with `@register_scorer`
- [ ] Add composite score to `RunResult` and SQLite

### 6.10 Update Markdown report for local suite

**File:** `src/bench_harness/reports/markdown.py` (update)

**New report sections:**

1. **Task Family Breakdown:**
```markdown
## Task Family Results

| Family | Model | Tasks | Passed | Avg Score |
|---|---|---|---|---|
```

2. **Executable vs Rubric Tasks:**
```markdown
## Validation Coverage

| Model | Executable Tasks | Passed | Rubric-Only Tasks | Avg Score |
|---|---|---|---|---|
```

3. **Risk Level Breakdown:**
```markdown
## Risk Level Performance

| Risk | Model | Tasks | Passed | Notes |
|---|---|---|---|---|
```

4. **Per-Model Recommendation:**
```markdown
## Recommendations

Based on local workload results:
- Best for Docker/compose tasks: <model>
- Best for code debugging: <model>
- Best for Git/Linux troubleshooting: <model>
```

**Actions:**
- [ ] Add all four report sections
- [ ] Compute per-family scores from task metadata
- [ ] Generate recommendation text based on highest scores per category

### 6.11 Add script for local suite

**File:** `scripts/run_local_coding_suite.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m bench_harness run \
  --suite local_coding_agent \
  --models agent-code,qwen-dense,max-brain \
  --runs 1 \
  --out "runs/$(date +%Y-%m-%d)-local-coding-agent"
```

**Actions:**
- [ ] Write script
- [ ] `chmod +x`

### 6.12 Add local suite tests

**File:** `tests/test_local_suite.py`

**Tests:**
- `test_all_25_tasks_load` — all 25 tasks load without error
- `test_task_has_required_fields` — every task has id, prompt, scoring
- `test_at_least_10_executable` — at least 10 tasks have test_files
- `test_prompt_templates_exist` — all referenced templates exist
- `test_fixture_files_exist` — all referenced fixture files exist
- `test_test_files_exist` — all referenced test files exist
- `test_risk_levels_valid` — all risk levels are low/medium/high
- `test_families_covered` — all 5 categories have at least 5 tasks

**Actions:**
- [ ] Implement all validation tests
- [ ] Run against actual task directory structure
- [ ] Fail if any cross-referenced files are missing

---

## Acceptance Criteria Checklist

- [ ] All 25 local tasks are defined with complete YAML, fixtures, and tests where applicable
- [ ] 5 Docker Compose tasks created with at least 3 having executable validation
- [ ] 5 LiteLLM tasks created with at least 3 having executable validation
- [ ] 5 qwen3_replicate tasks created with at least 4 having executable validation
- [ ] 5 benchmark script tasks created with at least 3 having executable validation
- [ ] 5 Git/Linux tasks created (regex/contains scoring)
- [ ] All 6 prompt templates created
- [ ] Suite runs end-to-end against configured models
- [ ] At least 10 tasks have executable validation
- [ ] Report ranks models by local coding-agent usefulness with per-category breakdown
- [ ] Composite scoring produces weighted scores per task
- [ ] `scripts/run_local_coding_suite.sh` works from project root
- [ ] `pytest tests/test_local_suite.py` passes

## Estimated Effort

4–5 days

## Files Produced by This Milestone

| File | Status |
|---|---|
| `tasks/local/docker_compose/task_001.yaml` through `task_005.yaml` | To create |
| `tasks/local/docker_compose/fixtures/docker-compose-*.yml` (×5) | To create |
| `tasks/local/docker_compose/tests/test_compose_*.py` (×3) | To create |
| `tasks/local/litellm/task_001.yaml` through `task_005.yaml` | To create |
| `tasks/local/litellm/fixtures/litellm-config-*.yaml` (×5) | To create |
| `tasks/local/litellm/tests/test_litellm_*.py` (×3) | To create |
| `tasks/local/qwen3_replicate/task_001.yaml` through `task_005.yaml` | To create |
| `tasks/local/qwen3_replicate/fixtures/*.py` (×5) | To create |
| `tasks/local/qwen3_replicate/tests/test_*.py` (×4) | To create |
| `tasks/local/benchmark_scripts/task_001.yaml` through `task_005.yaml` | To create |
| `tasks/local/benchmark_scripts/fixtures/*.py` (×4) | To create |
| `tasks/local/benchmark_scripts/tests/test_*.py` (×3) | To create |
| `tasks/local/git_linux/task_001.yaml` through `task_005.yaml` | To create |
| `tasks/local/git_linux/fixtures/main_conflicted.py` | To create |
| `configs/prompt_templates/docker_compose_debug.md` | To create |
| `configs/prompt_templates/litellm_debug.md` | To create |
| `configs/prompt_templates/debug_task.md` | To create |
| `configs/prompt_templates/benchmark_modify.md` | To create |
| `configs/prompt_templates/git_troubleshooting.md` | To create |
| `configs/prompt_templates/linux_troubleshooting.md` | To create |
| `configs/suites.yaml` | Update (local_coding_agent suite) |
| `src/bench_harness/scorers/composite.py` | To create |
| `src/bench_harness/reports/markdown.py` | Update (local suite report sections) |
| `scripts/run_local_coding_suite.sh` | To create |
| `tests/test_local_suite.py` | To create |
