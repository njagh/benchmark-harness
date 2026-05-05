# ROADMAP — Local LLM Quality Benchmark Harness

## 0. Project Purpose

Build a local benchmark harness that evaluates **response quality, coding ability, instruction following, agent behavior, and performance tradeoffs** across locally served models on the DGX Spark.

The harness should answer practical questions such as:

1. Does `Qwen3.6-35B-A3B-FP8` beat `Qwen3.6-27B-FP8` on the user’s actual workloads?
2. Does `max-brain` justify its high TTFT compared with faster models?
3. Does quantization degrade coding quality, reasoning quality, or instruction-following reliability?
4. Does longer context improve or degrade answer quality?
5. Does REPL-style prompting improve success rate, debugging reliability, and command discipline?
6. Which model/backend/prompting combination is the best default for:

   * coding-agent tasks
   * shell/debugging assistance
   * long-context repo analysis
   * math/reasoning
   * general Q&A
   * planning and architectural design

This project should not only produce leaderboard-style scores. It should produce a **decision system** for choosing the right local model and serving configuration for different real workflows.

---

## 1. Guiding Principles

### 1.1 Measure quality and performance together

Every run should capture both:

* quality outcome
* latency / throughput / resource metrics

A model that is slightly better but 10x slower may not be the right default. A fast model that fails high-risk coding tasks may still be useful as a first-pass assistant.

### 1.2 Prefer reproducible local evals

The harness should run against OpenAI-compatible local endpoints:

* LiteLLM gateway
* direct vLLM endpoints
* future TRT-LLM endpoints
* optional cloud endpoints for comparison

The same task should be runnable against any model alias.

### 1.3 Separate task quality from model serving quality

The harness should distinguish:

* model quality
* prompt quality
* backend performance
* quantization impact
* context-length impact
* harness/agent-loop effectiveness

### 1.4 Favor task families over one global score

The final output should avoid a single misleading “best model” score. Instead, report strengths by category.

Example:

| Task Family           | Best Model | Best Fast Model | Notes                                             |
| --------------------- | ---------- | --------------- | ------------------------------------------------- |
| Repo debugging        | max-brain  | agent-code      | max-brain better on hard failures but TTFT costly |
| Shell troubleshooting | agent-code | qwen-dense      | agent-code follows REPL mode better               |
| Long-context review   | max-brain  | agent-code      | quality drops above N tokens                      |
| Quick code snippets   | qwen-dense | qwen-dense      | fastest adequate option                           |

### 1.5 Keep the harness extensible

New benchmarks should be added by defining:

* task metadata
* prompt template
* runner
* scorer
* expected output schema
* result parser

---

## 2. Core Questions the Harness Must Answer

### 2.1 Model comparison questions

The harness should compare models such as:

* `agent-code`
* `qwen-dense`
* `big-brain`
* `max-brain`
* `nvidiamax`
* experimental quantized variants
* experimental patched/tuned variants

Questions:

* Which model gets the highest correctness score?
* Which model has the best quality per second?
* Which model has the best quality per GB / resource footprint?
* Which model is most reliable across repeated runs?
* Which model is least likely to violate formatting or tool discipline?

### 2.2 Quantization questions

The harness should compare the same or similar model families under different quantization schemes.

Examples:

* FP8 vs NVFP4
* GPTQ Int4 vs FP8
* dense vs MoE equivalent class
* patched kernel vs baseline kernel

Questions:

* Does quantization reduce exact correctness?
* Does it increase hallucination?
* Does it damage long-context reasoning?
* Does it damage code generation more than general Q&A?
* Does it affect instruction following?
* Does it change verbosity or refusal behavior?

### 2.3 Context-length questions

The harness should test the same task at different prompt sizes.

Examples:

* 2k tokens
* 8k tokens
* 32k tokens
* 64k tokens
* 128k tokens
* 200k+ tokens where supported

Questions:

* Does the model use relevant information from the long context?
* Does irrelevant context distract it?
* Does answer quality degrade as context grows?
* Does decode speed degrade independently of prompt/prefill time?
* Does long context cause formatting or instruction-following failures?

### 2.4 Prompting questions

Compare prompt styles:

* plain direct answer
* structured reasoning request
* REPL mode: hypothesis → single test → interpret
* patch-only mode
* JSON-only mode
* architect mode
* terse mode
* test-driven mode

Questions:

* Does REPL mode improve debugging success?
* Does it increase or reduce total tokens?
* Does it reduce dangerous shell suggestions?
* Does it help smaller/faster models close the gap with larger models?
* Does it hurt simple tasks by over-structuring the answer?

### 2.5 Agent-loop questions

For harnessed agent workflows:

* Does the model choose safe commands?
* Does it ask for appropriate tests?
* Does it avoid broad destructive commands?
* Does it make incremental changes?
* Does it recover from failed tests?
* Does it stop after success?
* Does it preserve unrelated files?

---

## 3. Project Deliverables

### 3.1 CLI benchmark runner

A command-line tool that can run benchmark suites against one or more model endpoints.

Example target interface:

```bash
python -m bench_harness run \
  --suite coding_smoke \
  --models agent-code,qwen-dense,max-brain \
  --endpoint http://spark-e287.local:4000/v1 \
  --temperature 0 \
  --runs 3 \
  --out runs/2026-05-04-coding-smoke
```

### 3.2 Task registry

A structured task registry containing:

* public benchmark tasks
* local repo tasks
* long-context tasks
* instruction-following tasks
* shell/debugging tasks
* agentic coding tasks
* synthetic regression tasks

### 3.3 Result database

A persistent store for:

* prompts
* model responses
* parsed outputs
* scores
* timings
* token counts
* environment metadata
* backend configuration
* GPU/RAM metrics
* pass/fail artifacts

Initial implementation can use SQLite. Later versions may add DuckDB or Parquet exports.

### 3.4 Scoring modules

Scorers for:

* exact match
* unit-test pass/fail
* JSON schema validity
* regex / format compliance
* multiple-choice accuracy
* LLM-as-judge rubric scoring
* pairwise preference scoring
* human review
* code patch validation
* command safety scoring

### 3.5 Report generator

Generate reports as:

* Markdown
* CSV
* JSONL
* HTML dashboard
* plots

Reports should answer practical questions directly, not just dump raw metrics.

### 3.6 Local workload benchmark suite

A curated set of tasks from the user’s real workflows:

* qwen3 replication debugging
* vLLM benchmark analysis
* Docker Compose repair
* LiteLLM routing changes
* Open WebUI maintenance
* Git/GitHub troubleshooting
* terminal/Linux support
* agentic scaffolding tasks
* long-context code review

### 3.7 Training-data export pipeline

The harness should export failed or high-value examples into formats useful for:

* SFT
* DPO / ORPO
* preference ranking
* regression evals
* judge calibration

---

## 4. Repository Structure

Proposed structure:

```text
benchmark-harness/
  README.md
  ROADMAP.md
  pyproject.toml
  .env.example
  configs/
    models.yaml
    suites.yaml
    scorers.yaml
    judge_rubrics.yaml
  src/
    bench_harness/
      __init__.py
      cli.py
      config.py
      models/
        openai_client.py
        litellm_client.py
        vllm_client.py
      tasks/
        registry.py
        task_schema.py
        loaders.py
        prompt_templates.py
      runners/
        base_runner.py
        completion_runner.py
        code_runner.py
        repo_runner.py
        agent_runner.py
      scorers/
        base.py
        exact_match.py
        multiple_choice.py
        json_schema.py
        regex.py
        unit_tests.py
        llm_judge.py
        pairwise.py
        command_safety.py
      metrics/
        timing.py
        tokens.py
        gpu.py
        memory.py
        process.py
      storage/
        sqlite.py
        artifacts.py
        export.py
      reports/
        markdown.py
        html.py
        plots.py
      analysis/
        compare_models.py
        compare_quantization.py
        compare_context.py
        compare_prompts.py
  tasks/
    public/
      mmlu_pro/
      gpqa/
      ifeval/
      bbh/
      math/
      humaneval/
      mbpp/
    local/
      coding_agent/
      shell_debugging/
      docker_compose/
      litellm/
      qwen3_replicate/
      long_context/
    synthetic/
      format_following/
      command_safety/
      context_distraction/
  runs/
    .gitkeep
  scripts/
    run_smoke.sh
    run_coding_suite.sh
    run_long_context_sweep.sh
    export_training_data.py
  tests/
    test_task_schema.py
    test_scorers.py
    test_storage.py
    test_runner_smoke.py
```

---

## 5. Data Model

### 5.1 Model configuration

Each model should be defined in `configs/models.yaml`.

Example:

```yaml
models:
  agent-code:
    provider: openai_compatible
    base_url: http://spark-e287.local:4000/v1
    model: agent-code
    backend: vllm
    notes: Qwen3.6-35B-A3B-FP8 via LiteLLM

  qwen-dense:
    provider: openai_compatible
    base_url: http://spark-e287.local:4000/v1
    model: qwen-dense
    backend: vllm
    notes: Qwen3.6-27B-FP8

  max-brain:
    provider: openai_compatible
    base_url: http://spark-e287.local:4000/v1
    model: max-brain
    backend: vllm
    notes: Qwen3.5-122B-A10B GPTQ Int4
```

### 5.2 Task schema

Each task should have a stable ID and explicit scoring metadata.

Example:

```yaml
id: local.docker_compose.fix_yaml_001
family: shell_debugging
category: docker
source: local
prompt_template: docker_compose_debug.md
input:
  user_message: |
    docker compose down returns: yaml: line 4: did not find expected key
  files:
    - docker-compose.yml
expected:
  type: explanation_plus_patch
scoring:
  primary: rubric
  secondary:
    - command_safety
    - format_compliance
    - minimality
risk_level: medium
context_tokens: small
```

### 5.3 Run record

Each model/task/run combination should produce a durable record.

Required fields:

```text
run_id
suite_id
task_id
model_alias
model_backend
model_config_hash
prompt_template_hash
prompt_tokens
completion_tokens
total_tokens
ttft_ms
decode_ms
total_wall_ms
tokens_per_second
exit_status
raw_response_path
parsed_response_path
score_primary
score_secondary
scorer_version
environment_snapshot_id
created_at
```

### 5.4 Environment snapshot

Capture enough metadata to make runs comparable.

Fields:

```text
host
OS
GPU name
GPU driver
CUDA version
container image
vLLM version
LiteLLM version
model path / HF repo
quantization
max_model_len
gpu_memory_utilization / free_gpu_memory_fraction
served port
commit hash of harness
commit hash of target repo, if any
```

---

## 6. Benchmark Suites

## 6.1 Suite: `smoke`

Purpose: verify that the harness and model endpoint work.

Tasks:

* one trivial factual prompt
* one JSON formatting task
* one small Python function task
* one short debugging task
* one instruction-following task

Success criteria:

* all configured models respond
* timings are captured
* results are written to SQLite
* report is generated

---

## 6.2 Suite: `public_baseline`

Purpose: establish standard reference quality.

Initial tasks:

* IFEval
* MMLU-Pro subset
* GPQA subset
* BBH subset
* MATH subset
* HumanEval or MBPP subset

Implementation approach:

* integrate with `lm-evaluation-harness` where practical
* store summarized metrics in the harness database
* optionally store raw results separately

Success criteria:

* each local model gets comparable public benchmark scores
* results include latency and token metrics where available
* reports distinguish public benchmark score from local workload score

---

## 6.3 Suite: `coding_smoke`

Purpose: fast coding quality regression.

Task types:

* write a simple Python function
* fix a failing unit test
* explain a stack trace
* produce a small patch
* improve a benchmark script
* identify a bug in transformer implementation

Scoring:

* unit tests pass
* patch applies cleanly
* no unrelated files changed
* response follows requested format

Success criteria:

* can run in minutes
* useful before trying a new model/backend/quantization

---

## 6.4 Suite: `repo_debugging`

Purpose: test real coding-agent usefulness.

Example task families:

1. `qwen3_replicate`

   * RMSNorm parity bug
   * RoPE layout bug
   * GQA cache shape bug
   * attention mask bug
   * HF weight loader bug

2. `llm_validation`

   * add benchmark metric
   * fix token accounting
   * parse vLLM timing output
   * compare prefill vs decode

3. `ai-stack`

   * repair Docker Compose YAML
   * update model alias safely
   * modify LiteLLM config
   * diagnose port conflict

4. `agent-orchestrator`

   * fix workspace init script
   * update agent handoff parsing
   * add task file validation

Scoring:

* tests pass
* patch minimality
* explanation accuracy
* no destructive commands
* follows REPL mode
* avoids guessing when logs are insufficient

Success criteria:

* at least 25 real tasks
* at least 10 with executable tests
* at least 5 requiring multi-step debugging

---

## 6.5 Suite: `long_context`

Purpose: test context scaling and retrieval from prompt.

Task variants:

1. Relevant fact buried early
2. Relevant fact buried middle
3. Relevant fact buried late
4. Multiple conflicting facts, latest instruction wins
5. Large irrelevant codebase context with one relevant file
6. Long benchmark logs with one important regression
7. Long architecture doc with requested change

Context sizes:

```text
2k
8k
32k
64k
128k
200k where supported
```

Scoring:

* answer cites or references the correct context section
* ignores distractors
* does not invent missing information
* follows final instruction
* preserves output format

Performance metrics:

* prefill time
* TTFT
* decode time
* tokens/sec
* RAM/UMA/GPU memory pressure

Success criteria:

* produce degradation curves by model
* identify context length where quality starts dropping
* identify context length where performance becomes impractical

---

## 6.6 Suite: `prompt_style_comparison`

Purpose: test whether prompt style changes outcomes.

Prompt variants:

1. Plain prompt
2. REPL mode
3. terse answer
4. patch-only
5. architect mode
6. step-by-step plan then patch
7. JSON schema output

Task families:

* debugging
* code generation
* shell troubleshooting
* planning
* long-context analysis

Metrics:

* correctness
* total tokens
* number of invalid outputs
* safety violations
* time to usable answer
* judge score

Success criteria:

* determine where REPL mode helps
* determine where REPL mode is unnecessary overhead
* determine best prompt template per task family

---

## 6.7 Suite: `quantization_comparison`

Purpose: isolate quality impact of quantization.

Requirements:

* same task set
* same sampling settings
* same prompt templates
* same scoring
* same run count
* record backend and quantization metadata

Comparisons:

* FP8 vs NVFP4
* FP8 vs GPTQ Int4
* dense FP8 vs dense quantized
* MoE FP8 vs MoE quantized

Task categories:

* math reasoning
* coding
* instruction following
* long-context retrieval
* JSON formatting
* hallucination/factuality

Success criteria:

* produce quality deltas by task type
* produce speed/quality frontier chart
* identify quantization regimes that are unacceptable for coding-agent use

---

## 6.8 Suite: `agent_safety`

Purpose: evaluate command discipline and safe tool use.

Task examples:

* user asks to delete broad directories
* model should suggest backup first
* model should avoid `rm -rf` on unknown paths
* model should prefer inspection before mutation
* model should use git status before editing
* model should avoid leaking secrets
* model should not run network installs casually

Scoring:

* allowed command classification
* dangerous command detection
* asks for confirmation where appropriate
* uses read-only command first
* suggests reversible operation

Success criteria:

* models are ranked by command safety
* prompt templates are tested for safety improvement
* failure examples become regression tests

---

## 7. Scoring System

## 7.1 Score types

### Exact score

Used for:

* multiple choice
* short factual answer
* known numeric answer
* schema-valid yes/no

### Executable score

Used for:

* code tasks
* patches
* unit tests
* shell command tasks

### Rubric score

Used for:

* explanations
* architecture plans
* benchmark interpretation
* troubleshooting quality

### Judge score

Used where exact scoring is hard.

Judge should use explicit rubrics and produce structured JSON.

Example rubric dimensions:

```text
correctness: 0-5
completeness: 0-5
specificity: 0-5
safety: 0-5
format_compliance: 0-5
minimality: 0-5
```

### Pairwise preference

Used to compare two model outputs.

Output:

```json
{
  "winner": "A",
  "confidence": 0.82,
  "reason": "A provides the correct patch and avoids unrelated changes."
}
```

---

## 7.2 Composite scores

Avoid one universal composite score. Instead compute per-suite composites.

Example coding-agent composite:

```text
50% tests pass
20% patch minimality
15% explanation quality
10% command safety
5% format compliance
```

Example shell-debugging composite:

```text
35% correctness
25% safety
20% incremental diagnosis
10% clarity
10% command specificity
```

Example long-context composite:

```text
40% relevant information retrieved
25% distractor resistance
15% instruction following
10% answer completeness
10% format compliance
```

---

## 8. Metrics to Capture

## 8.1 Quality metrics

```text
primary_score
secondary_scores
pass_fail
judge_score
format_valid
schema_valid
unit_tests_passed
unit_tests_failed
patch_applies
unsafe_command_count
hallucination_flag
refusal_flag
incomplete_answer_flag
```

## 8.2 Performance metrics

```text
ttft_ms
prefill_ms, if available
decode_ms
total_wall_ms
prompt_tokens
completion_tokens
total_tokens
tokens_per_second_generation
tokens_per_second_total
```

## 8.3 Resource metrics

```text
gpu_util_avg
gpu_util_max
gpu_memory_used_start
gpu_memory_used_peak
gpu_memory_used_end
system_ram_used_start
system_ram_used_peak
swap_used_start
swap_used_peak
cpu_util_avg
```

## 8.4 Reliability metrics

Run each task multiple times where sampling is nonzero or model nondeterminism is observed.

```text
mean_score
score_stddev
pass_rate
flaky_task_flag
response_length_stddev
format_failure_rate
```

---

## 9. Implementation Milestones

# Milestone 1 — Project bootstrap **✅ DONE**

## Goal

Create the repository, basic CLI, model config, and smoke runner.

## Tasks

* [x] Create repo structure.
* [x] Add `pyproject.toml`.
* [x] Add `configs/models.yaml`.
* [x] Implement OpenAI-compatible client.
* [x] Implement simple completion runner.
* [x] Implement JSONL output.
* [x] Implement SQLite output.
* [x] Add first five smoke tasks.
* [x] Add basic Markdown report.

## Acceptance criteria

* [x] Can run one prompt against one local model.
* [x] Can run smoke suite against at least two models.
* [x] Stores raw response and timing data.
* [x] Produces a readable Markdown report.

## Suggested first command target

```bash
python -m bench_harness run --suite smoke --models agent-code,qwen-dense
```

---

# Milestone 2 — Task schema and registry **✅ DONE**

## Goal

Make benchmark tasks structured, versioned, and easy to add without Python code changes.

## Tasks

* [x] Define task YAML schema (Pydantic models: `Task`, `TaskInput`, `TaskExpected`, `TaskScoring`).
* [x] Add task loader with schema validation and backward-compatible M1 migration.
* [x] Add prompt template renderer (Jinja2-based, 7 built-in styles).
* [x] Add task family/category metadata.
* [x] Add validation tests for malformed tasks.
* [x] Add stable task IDs with versioning support.
* [x] Implement task registry (`TaskRegistry` with list/lookup by family/source).
* [x] Add CLI subcommands: `list-tasks`, `show-task`.

## Acceptance criteria

* [x] New task can be added by dropping a YAML file into a task directory.
* [x] Invalid task configs fail fast with field-level error messages.
* [x] `bench-harness list-tasks` lists all registered tasks.
* [x] `bench-harness list-tasks --family coding` filters correctly.
* [x] `bench-harness show-task <id>` displays full task YAML.
* [x] Prompt templates render correctly with context variables.
* [x] Task versions are tracked and stored in run records.

---

# Milestone 3 — Timing and token metrics **✅ DONE**

## Goal

Capture performance metrics (TTFT, wall time, decode time, token counts) comparable to existing timing benchmarks, stored consistently in SQLite and exposed in reports.

## Tasks

* [x] Implement wall-clock timing layer (`TimingRecord`, `timed_block`, `StreamingTimer`, `StreamMetrics`).
* [x] Implement streaming TTFT capture (`StreamingTimer`).
* [x] Capture API token counts (`TokenCounter`, `normalize_usage`).
* [x] Implement fallback tokenizer counting (`FallbackTokenCounter` via tiktoken).
* [x] Extend `RunResult` with timing/token fields (`tokens_per_second`, `token_source`, `decode_ms`, etc.).
* [x] Extend SQLite schema with migration, `run_timings` table, `get_timing_summary()`.
* [x] Add timing summary sections to Markdown report.
* [x] Add `--timing-detail` CLI flag.
* [x] Add `tokenizer` field to `configs/models.yaml`.

## Acceptance criteria

* [x] Report shows TTFT, wall time, decode time, completion tokens, and tokens/sec per task.
* [x] Metrics are captured consistently across streaming and non-streaming responses.
* [x] Fallback tokenizer counting works when API returns no usage data.
* [x] SQLite `run_timings` table stores all timing metrics.
* [x] `get_timing_summary()` returns per-model aggregates (mean, min, max, p95).
* [x] Markdown report includes timing summary table and per-task timing table.
* [x] `--timing-detail` CLI flag shows per-task timing in terminal output.
* [x] Schema migration is safe on both fresh and existing databases.

## Tasks

* Add wall-clock timing around API calls.
* Capture streaming TTFT where supported.
* Capture generation time.
* Capture prompt/completion token counts from API usage if available.
* Add fallback token counting using tokenizer where needed.
* Store timing metrics in SQLite.
* Add per-model timing summary.

## Acceptance criteria

* Report shows TTFT, wall time, completion tokens, and tokens/sec.
* Metrics are captured consistently across LiteLLM and direct vLLM where possible.

---

# Milestone 4 — Basic scorers **✅ DONE**

## Goal

Support automatic deterministic scoring for simple tasks: exact match, multiple-choice, regex, JSON schema, contains/does-not-contain, and format compliance.

## Tasks

* [x] Implement base scorer interface with registry (`@register_scorer` decorator).
* [x] Implement `ExactMatchScorer` with whitespace normalization and case-insensitive option.
* [x] Implement `MultipleChoiceScorer` with regex-based answer extraction.
* [x] Implement `RegexScorer` with all/any/none modes and partial scoring.
* [x] Implement `JsonSchemaScorer` with JSON extraction from markdown fences, repair, and validation.
* [x] Implement `ContainsScorer` with required/absent substring matching.
* [x] Implement `FormatComplianceScorer` with 10 format checks (numbered list, filler, line count, etc.).
* [x] Wire scorers into runner with primary/secondary scoring.
* [x] Extend `RunResult` with scoring fields.
* [x] Add `score_details` table to SQLite with schema migration.
* [x] Add scoring sections to Markdown report.
* [x] Load scorer config from `configs/scorers.yaml`.
* [x] Add 44 unit tests covering all scorers and registry.

## Acceptance criteria

* [x] All six scorers (exact_match, multiple_choice, regex, json_schema, contains, format_compliance) are implemented and registered.
* [x] Smoke tasks produce real scores with all configured scorers.
* [x] Format failures are visible in the Markdown report.
* [x] Scorers are deterministic and fully tested.
* [x] Score results are stored in SQLite with details.
* [x] Runner invokes scorers automatically based on task YAML scoring config.
* [x] `pytest tests/test_scorers.py` passes with full coverage on scorer modules.

---

# Milestone 5 — Coding task runner **✅ DONE**

## Goal

Evaluate code outputs using executable tests in isolated temp directories.

## Tasks

* [x] Define code task format (`code_type: "function_completion" | "patch_generation"`, `entry_point`, `test_framework`, `test_code`).
* [x] Support function-completion tasks — write generated code + test code to temp dir, run pytest.
* [x] Support patch-generation tasks — write context files + patch diff, `git apply --check` + `git apply`, run tests.
* [x] Run tests in isolated temp directories with subprocess-based pytest execution.
* [x] Capture stdout/stderr from pytest execution.
* [x] Detect patch apply failure via `git apply --check`.
* [x] Detect unrelated file changes via `git diff --name-only` vs expected `test_files`.
* [x] Store artifacts: generated_code.py, test_generated.py, test_output.txt, patch.diff.
* [x] Wire `CodeTaskRunner` into `CompletionRunner` — auto-detect via `code_type` field.
* [x] Add `unit_test` scorer for code task scoring.
* [x] Extend `RunResult` with code-specific fields: `tests_passed`, `tests_failed`, `tests_total`, `test_output`, `exit_code`, `generated_code`, `code_status`.
* [x] Extend artifact output with code fields in JSONL.

## Acceptance criteria

* [x] Can run HumanEval/MBPP-style local tasks with real test execution.
* [x] Can score pass/fail from unit tests (fraction of tests passed).
* [x] Raw generated code, test code, and test logs are saved as artifacts.
* [x] 4 coding smoke tasks (simple_math, string_ops, list_utils, bugfix_simple) all execute successfully.
* [x] `pytest tests/test_code_task_runner.py` passes with full coverage on code runner modules.

---

# Milestone 6 — Local coding-agent suite v1 **✅ DONE**

## Goal

Create a high-value benchmark based on real workflows.

## Tasks

* [x] Create 25 local tasks across 5 families (5 per family).
  * 5 Docker Compose / YAML repair tasks in `tasks/local_coding_agent_v1/docker_compose/`
  * 5 LiteLLM / model routing tasks in `tasks/local_coding_agent_v1/litellm_routing/`
  * 5 qwen3_replicate debugging tasks in `tasks/local_coding_agent_v1/qwen3_debug/`
  * 5 benchmark-script modification tasks in `tasks/local_coding_agent_v1/benchmark_script/`
  * 5 Git/Linux troubleshooting tasks in `tasks/local_coding_agent_v1/git_linux/`
* [x] Each task defined with prompt, scoring config, expected behavior, risk level, and `code_type`.
* [x] CLI `list-tasks` auto-discovers `tasks/local_coding_agent_v1/`.
* [x] CLI `list-tasks --family` supports comma-separated family filters.
* [x] Markdown report includes "Coding Agent Ranking" section with per-model scores.
* [x] Markdown report includes "Family Breakdown" section with per-family scores.
* [x] Report ranking based on test pass rate for code tasks, pattern match for config tasks.
* [x] Context files referenced in tasks (via `{{context_files}}` in prompts).

## Acceptance criteria

* [x] Suite has 25 tasks across 5 families.
* [x] At least 10 tasks have executable validation (`code_type: patch_generation`).
* [x] Report ranks models by local coding-agent usefulness with per-family breakdown.
* [x] `bench-harness list-tasks` discovers all 25 local coding agent tasks.

---

# Milestone 7 — LLM judge integration **✅ DONE**

## Goal

Score complex answers with explicit rubrics.

## Tasks

* [x] Implement judge model config (`load_judge_config`).
* [x] Add rubric YAML format loading (`load_rubric_config`).
* [x] Add judge prompt templates (consumer-facing: LLMJudgeScorer).
* [x] Require structured JSON judge output.
* [x] Add judge self-consistency option (configurable).
* [x] Add pairwise comparison mode with DB storage.
* [x] Add human override field on `RunResult` and `pairwise_comparisons`.
* [x] Store judge explanations in `judge_evaluations` table.
* [x] Add `judge_evaluations` SQLite table.
* [x] Add `pairwise_comparisons` SQLite table.
* [x] Add schema migration for judge columns on `runs` and `score_details`.
* [x] Add artifact saving for judge raw/parsed/prompt files.
* [x] Add `--judge` flag to `run` CLI command.
* [x] Add `judge-config` CLI subcommand (list/show).
* [x] Add judge report sections to Markdown report.

## Acceptance criteria

* [x] Complex answers receive rubric scores via LLM judge.
* [x] Pairwise model comparisons can be generated and stored.
* [x] Judge outputs are auditable (raw responses, parsed scores, prompts saved).
* [x] Judge model can be swapped via config.
* [x] Human override supported for both judge scores and pairwise comparisons.
* [x] Reports include Judge-Scored Tasks, Dimension Breakdown, and Pairwise Comparison sections.
* [x] `bench-harness judge-config list` shows available rubrics.
* [x] `bench-harness judge-config show <name>` shows rubric details.

## Note

Use judge scores for ranking only when exact or executable scoring is unavailable. Prefer tests and deterministic scoring wherever possible.

---

# Milestone 8 — Prompt style comparison **✅ DONE**

## Goal

Evaluate the impact of REPL mode and other prompting styles.

## Tasks

* [x] Add prompt template variants (7 built-in styles: plain, repl, terse, patch_only, architect, json_schema, step_by_step).
* [x] Add CLI support for prompt style sweep (--styles flag, comma-separated).
* [x] Implement StyleSweepRunner for task × style × model combinations.
* [x] Run same tasks with multiple prompt styles and compare results.
* [x] Compare correctness, verbosity, latency, safety, and token use.
* [x] Add prompt-style report sections to Markdown report (6 sub-sections).
* [x] Document REPL mode shown to help/fail scenarios.

## Acceptance criteria

* [x] Harness can show where REPL mode helps (higher scores on debugging tasks).
* [x] Harness can show where REPL mode adds unnecessary overhead (lower score/token ratio on simple tasks).
* [x] Recommended prompt style is reported per task family.
* [x] Style comparison report includes: summary, per-task breakdown, best-per-family, verbosity analysis, latency comparison, recommendation.
* [x] RunResult supports prompt_style field for sweep results.
* [x] Markdown report auto-detects and includes style comparison sections when prompt_style metadata is present.

---

# Milestone 9 — Long-context benchmark suite

## Goal

Measure quality and performance as prompt size increases.

## Tasks

* Create synthetic long-context tasks.
* Create repo-derived long-context tasks.
* Add context packer.
* Add distractor generator.
* Add relevant-fact placement control.
* Add context-size sweep.
* Track prefill/TTFT/decode degradation.
* Score retrieval of buried facts.

## Acceptance criteria

* Can run same task at 2k, 8k, 32k, 64k, 128k tokens.
* Produces quality-vs-context and speed-vs-context reports.
* Identifies model-specific context breakpoints.

---

# Milestone 10 — Quantization comparison suite

## Goal

Measure quality and performance impact of quantization.

## Tasks

* Add model metadata fields for quantization.
* Add suite for quantization-sensitive tasks.
* Run matched task sets across quantized variants.
* Generate quality delta reports.
* Generate speed/quality frontier plots.

## Acceptance criteria

* Report shows whether quantization hurts coding, math, long-context, or formatting.
* Report identifies best quantized model by task family.
* Quantization impact is separated from prompt/backend differences.

---

# Milestone 11 — Agent safety and command discipline

## Goal

Evaluate whether models behave safely when suggesting shell or coding-agent actions.

## Tasks

* Build command safety classifier.
* Define risky command categories.
* Add tasks with tempting unsafe commands.
* Score inspection-before-mutation behavior.
* Score backup/git-status recommendations.
* Score refusal or confirmation behavior.
* Add shell command parser.

Risk categories:

```text
broad deletion
permission escalation
secret exposure
network install
force push
history rewrite
destructive docker prune
blind file overwrite
```

## Acceptance criteria

* Safety score appears in reports.
* Unsafe command examples are extracted.
* Prompt changes can be measured for safety impact.

---

# Milestone 12 — Report generator v2

## Goal

Make results easy to interpret.

## Tasks

* Add model comparison tables.
* Add suite-level summaries.
* Add best-model-by-task-family section.
* Add speed/quality frontier.
* Add failure clustering.
* Add regression detection versus prior run.
* Add Markdown and HTML output.

## Acceptance criteria

Report answers:

* Which model is best overall for local coding?
* Which model is best fast option?
* Which model is unsafe or unreliable?
* Which tasks are most discriminating?
* Which prompt style works best?
* Which quantization changed quality?
* Which model degrades under long context?

---

# Milestone 13 — Training-data export

## Goal

Turn benchmark results into future fine-tuning and preference datasets.

## Tasks

* Export successful examples to SFT JSONL.
* Export pairwise winners to DPO/ORPO JSONL.
* Export failed examples to regression suite.
* Export judge-labeled examples.
* Add manual review status.
* Add data hygiene filters.

Export formats:

```text
sft_openai_messages.jsonl
preference_chosen_rejected.jsonl
regression_tasks.yaml
judge_scores.jsonl
```

## Acceptance criteria

* A failed benchmark task can become a regression test.
* A high-quality response can become an SFT example.
* A model comparison can become preference data.

---

# Milestone 14 — Dashboard / analysis notebook

## Goal

Provide interactive analysis for repeated experiments.

## Tasks

* Add DuckDB or Pandas analysis notebook.
* Add charts for score vs latency.
* Add context-length degradation curves.
* Add quantization comparison plots.
* Add prompt-style comparison plots.
* Add failure examples browser.

## Acceptance criteria

* Can explore results without manually reading JSONL.
* Can compare current run to previous runs.
* Can identify regressions visually.

---

# Milestone 15 — CI / regression mode

## Goal

Use the harness as a regression test system for model/backend changes.

## Tasks

* Add `quick_regression` suite.
* Add baseline comparison thresholds.
* Add pass/fail gates.
* Add command-line diff against previous run.
* Add optional GitHub Actions support for non-GPU tests.

## Acceptance criteria

* Before changing model/backend config, user can run one command.
* Harness flags quality regressions.
* Harness flags performance regressions.

Example:

```bash
python -m bench_harness compare \
  --baseline runs/2026-05-01-agent-code-baseline \
  --candidate runs/2026-05-04-agent-code-new-kernel
```

---

## 10. Recommended Build Order

### Phase A — Minimal useful harness

1. Bootstrap repo.
2. Add OpenAI-compatible client.
3. Add smoke tasks.
4. Add timing metrics.
5. Add SQLite storage.
6. Add Markdown report.

Outcome: can compare local models on simple tasks.

### Phase B — Real coding usefulness

1. Add task schema.
2. Add code runner.
3. Add local coding tasks.
4. Add unit-test scoring.
5. Add prompt-style variants.

Outcome: can answer whether `agent-code` beats `qwen-dense` or `max-brain` on real coding tasks.

### Phase C — Deep model/backend comparison

1. Add long-context suite.
2. Add quantization comparison suite.
3. Add resource metrics.
4. Add speed/quality frontier reports.

Outcome: can decide whether slower or quantized models are worth it.

### Phase D — Data flywheel

1. Add judge scoring.
2. Add pairwise comparison.
3. Add training-data export.
4. Add regression suite generation.

Outcome: benchmark failures become training data and regression tests.

---

## 11. Initial Task Backlog

## 11.1 Highest priority

* [ ] Create repo and scaffold package.
* [ ] Implement `bench_harness run` CLI.
* [ ] Implement OpenAI-compatible chat client.
* [ ] Add model config for `agent-code`, `qwen-dense`, and `max-brain`.
* [ ] Add five smoke tasks.
* [ ] Capture wall time and token usage.
* [ ] Save run output to JSONL.
* [ ] Save run output to SQLite.
* [ ] Generate Markdown summary.

## 11.2 Next priority

* [ ] Define YAML task schema.
* [ ] Add exact-match scorer.
* [ ] Add regex scorer.
* [ ] Add JSON schema scorer.
* [ ] Add multiple-choice scorer.
* [ ] Add unit tests for scorers.
* [ ] Add prompt templates.
* [ ] Add prompt style variants.

## 11.3 Coding eval priority

* [ ] Add isolated temp workspace runner.
* [ ] Add patch application support.
* [ ] Add Python unit-test runner.
* [ ] Add file diff capture.
* [ ] Add unrelated-change detector.
* [ ] Add local coding smoke tasks.
* [ ] Add qwen3_replicate task pack.
* [ ] Add ai-stack task pack.

## 11.4 Analysis priority

* [ ] Add model comparison report.
* [ ] Add per-task-family ranking.
* [ ] Add quality/latency scatter plot.
* [ ] Add prompt-style comparison report.
* [ ] Add regression comparison command.

## 11.5 Later priority

* [ ] Add LLM judge scoring.
* [ ] Add pairwise preference scoring.
* [ ] Add long-context context packer.
* [ ] Add quantization comparison reports.
* [ ] Add command safety classifier.
* [ ] Add SFT export.
* [ ] Add DPO export.
* [ ] Add dashboard or notebook.

---

## 12. Concrete First Version Scope

Version 0.1 should be deliberately small.

### Include

* CLI runner
* OpenAI-compatible endpoint support
* model config YAML
* task YAML
* smoke suite
* coding smoke suite
* exact/regex/JSON scorers
* timing capture
* JSONL output
* SQLite output
* Markdown report

### Exclude for v0.1

* full public benchmark integration
* LLM judge
* long-context sweeps
* quantization reports
* dashboard
* SFT/DPO export
* complex agent loop

### v0.1 success definition

A successful v0.1 can run:

```bash
python -m bench_harness run \
  --suite smoke,coding_smoke \
  --models agent-code,qwen-dense,max-brain \
  --runs 3
```

and produce a report answering:

* Which model passed the most tasks?
* Which model was fastest?
* Which model had the best score per second?
* Which tasks failed?
* Which outputs violated format?
* Which model should be the default for quick coding tasks?

---

## 13. Example Final Report Shape

```text
Benchmark Report: coding_smoke
Date: 2026-05-04
Host: spark-e287

Models:
- agent-code: Qwen3.6-35B-A3B-FP8
- qwen-dense: Qwen3.6-27B-FP8
- max-brain: Qwen3.5-122B-A10B-GPTQ-Int4

Summary:
- Best quality: max-brain
- Best fast model: agent-code
- Best quality/sec: agent-code
- Most format failures: qwen-dense
- Highest TTFT: max-brain

Task Family Results:
| Family | Winner | Fast Winner | Notes |
|---|---|---|---|
| Python unit tests | agent-code | agent-code | max-brain tied but slower |
| Docker YAML repair | agent-code | qwen-dense | qwen-dense adequate for simple fixes |
| Long error logs | max-brain | agent-code | max-brain better at root cause |

Recommendation:
Use agent-code as default coding assistant. Use max-brain only for hard debugging, large context, or architectural reasoning. Use qwen-dense for quick shell snippets and low-risk edits.
```

---

## 14. Key Design Decisions

### 14.1 SQLite first

Use SQLite for v0.1 because:

* easy to inspect
* no server required
* enough for local benchmark runs
* can export to CSV/Parquet later

### 14.2 JSONL artifacts always

Even with SQLite, save raw JSONL artifacts so results remain portable and debuggable.

### 14.3 Human-readable reports from the beginning

The first version should generate useful Markdown. Reports are the product.

### 14.4 Local tasks matter more than public scores

Public benchmarks establish sanity. Local tasks determine practical model choice.

### 14.5 Every failure should be reusable

A failed task should become:

* a regression test
* a training example
* a prompt improvement example
* a judge calibration example

---

## 15. Risks and Mitigations

### Risk: LLM judge is unreliable

Mitigation:

* prefer executable scoring
* use explicit rubrics
* use pairwise comparison
* keep raw outputs
* allow human override

### Risk: public benchmarks are contaminated

Mitigation:

* use public benchmarks only for broad comparison
* emphasize local private tasks
* create fresh synthetic tasks
* use repo-specific evals

### Risk: timing metrics are inconsistent across backends

Mitigation:

* capture wall-clock timing in harness
* capture API usage where available
* optionally parse backend metrics
* clearly label metric source

### Risk: model output parsing is brittle

Mitigation:

* define strict output schemas
* score format compliance separately
* save raw outputs
* use repair/parsing only as separate metric, not hidden cleanup

### Risk: long-context evals become too slow

Mitigation:

* start with small sweeps
* support max task count
* cache context packs
* run overnight manually, not as default smoke suite

### Risk: local task suite overfits to one model

Mitigation:

* keep hidden tasks
* rotate task variants
* include public baselines
* compare with human expectations

---

## 16. Future Extensions

Potential later features:

* integrate `lm-evaluation-harness`
* integrate SWE-bench Lite / Verified style repo tasks
* integrate LiveCodeBench
* add web dashboard
* add run scheduler
* add automatic model discovery from LiteLLM `/v1/models`
* add tokenizer-aware context packing
* add model-as-planner vs model-as-coder split evals
* add tool-call simulation
* add sandboxed shell execution
* add Dockerized test runners
* add benchmark result cards for Open WebUI
* add automatic prompt optimization loop
* add human review UI
* add DPO/ORPO export
* add LoRA fine-tuning handoff scripts

---

## 17. Definition of Done for the Project

The project is successful when the harness can reliably answer:

1. Which local model should be the default for coding?
2. Which local model should be used for hard debugging?
3. Which model should be used for long-context tasks?
4. Which quantized models are acceptable replacements?
5. Which prompt style works best by task family?
6. Whether a backend/model change caused a quality regression.
7. Whether a slower model is worth its latency cost.
8. Which benchmark failures should become training data.

The final system should make model/backend choices empirical instead of impressionistic.
