# TODO: Generalize Benchmark Harness for ModelOpt / Temporary Model Experiments

## Objective

Refactor the existing benchmark harness so it works as a reusable library and CLI for evaluating models produced by separate experiments, including NVIDIA Model Optimizer quantization runs, temporary build directories, TensorRT-LLM engines, vLLM-served checkpoints, GGUF/offload baselines, and other future model-optimization projects.

The harness must not assume that models, generated artifacts, benchmark configs, or results live inside the benchmark harness repository.

## Core Design Goal

Separate these four concerns:

1. **Harness code** — reusable Python package and CLI.
2. **Experiment workspace** — where a specific project runs, such as `modelopt_3070ti_experiment`.
3. **Model artifacts** — large/generated outputs such as quantized checkpoints, TensorRT-LLM engines, patched HF snapshots, GGUF files, logs, calibration caches.
4. **Benchmark results** — structured, durable run records that can be compared across projects, machines, runtimes, and model variants.

The benchmark harness repo should contain code, schemas, tests, and examples. It should not be the default long-term home for large model artifacts or project-specific result archives.

---

# 1. Package / Library Refactor

## 1.1 Convert harness into importable package

Refactor the benchmark harness into a Python package with a stable public API.

Suggested layout:

```text
benchmark_harness/
  pyproject.toml
  README.md
  src/
    llm_bench/
      __init__.py
      cli.py
      config.py
      storage.py
      registry.py
      runners/
        __init__.py
        openai_compatible.py
        vllm.py
        trtllm.py
        llamacpp.py
        local_process.py
      metrics/
        __init__.py
        latency.py
        throughput.py
        memory.py
        quality.py
      schemas/
        __init__.py
        run_spec.py
        run_result.py
        model_artifact.py
      reports/
        __init__.py
        summarize.py
        compare.py
      utils/
        __init__.py
        subprocesses.py
        gpu.py
        hashing.py
        paths.py
  tests/
  examples/
```

Acceptance criteria:

* `pip install -e .` works.
* `python -m llm_bench --help` works.
* Existing benchmark scripts are either preserved as wrappers or migrated into the CLI.
* Core functionality can be imported from another project without shelling out.

## 1.2 Define public API

Expose a small public API for external projects.

Example:

```python
from llm_bench import BenchmarkRunner, RunSpec, StorageConfig

runner = BenchmarkRunner(storage=StorageConfig.from_env())
result = runner.run(spec)
```

Acceptance criteria:

* External project can construct a benchmark run from Python.
* External project can invoke the same run through CLI.
* Public API does not depend on repository-relative paths.

---

# 2. Storage Architecture

## 2.1 Create explicit storage root configuration

Add a storage layer that supports a user-configurable root independent of the benchmark harness repo.

Priority order for resolving storage root:

1. CLI flag: `--storage-root`
2. Environment variable: `LLM_BENCH_STORAGE_ROOT`
3. Per-project config file
4. Default: `~/.local/share/llm-bench` or `/mnt/datasets-big/llm-bench` if configured explicitly by the user

Do not default to a subdirectory inside the benchmark harness repo.

Acceptance criteria:

* Running benchmarks from any working directory writes to the selected storage root.
* No large artifacts are written under the harness source tree unless explicitly requested.
* A dry-run command shows exactly where artifacts and results will be stored.

## 2.2 Separate artifact store from results store

Implement separate storage namespaces:

```text
<storage-root>/
  artifacts/
    models/
    engines/
    tokenizers/
    calibration/
    runtime-builds/
  results/
    runs/
    summaries/
    comparisons/
  registry/
    models.jsonl
    artifacts.jsonl
    runs.jsonl
  logs/
  cache/
  tmp/
```

Acceptance criteria:

* Benchmark results can reference model artifacts located anywhere.
* Model artifacts can be external paths, symlinks, or registered managed artifacts.
* Results are durable even if temporary model build directories are deleted, as long as metadata was captured.

## 2.3 Support external artifact references

Do not require copying every model into harness storage.

Support three artifact modes:

1. `external_path` — model lives somewhere else; harness records path and metadata.
2. `managed_copy` — harness copies selected files into artifact storage.
3. `managed_symlink` — harness stores a symlink to the source artifact.

Example artifact spec:

```yaml
artifact:
  mode: external_path
  kind: hf_checkpoint
  path: /tmp/modelopt-runs/qwen-14b-int4-awq
  model_id: Qwen/example-14B
  quantization: int4_awq
```

Acceptance criteria:

* Harness can benchmark a model from `/tmp`, `/mnt/datasets-big`, HF cache, or experiment workspace.
* Harness records whether artifact path is external, copied, or symlinked.
* Harness warns if an external artifact path looks temporary or non-durable.

## 2.4 Capture artifact fingerprints

For every model artifact, capture enough metadata to identify what was actually benchmarked.

Minimum metadata:

* artifact kind: HF checkpoint, TensorRT-LLM engine, GGUF, vLLM endpoint, OpenAI-compatible endpoint, etc.
* source path or URL
* model ID / served model name
* quantization method
* dtype
* parameter class if known
* tokenizer path
* file list summary
* total artifact size
* config file hash
* selected weight file hashes or manifest hash
* creation timestamp if available
* git commit of producing project if available
* ModelOpt version if applicable
* TensorRT-LLM / vLLM / llama.cpp version if applicable

Acceptance criteria:

* Two runs against different temporary builds are distinguishable.
* A run result remains interpretable even after the temporary artifact is deleted.
* Manifest hash is stable enough for comparison but does not require hashing hundreds of GB by default.

---

# 3. Run Specification Schema

## 3.1 Define `RunSpec`

Create a structured run spec used by both CLI and Python API.

Suggested fields:

```yaml
run:
  name: qwen14b-int4-awq-3070ti-smoke
  project: modelopt_3070ti
  tags: [modelopt, int4, awq, 3070ti]

hardware:
  profile: 3070ti-8gb
  expected_gpu: RTX 3070 Ti

artifact:
  kind: hf_checkpoint
  mode: external_path
  path: /path/to/quantized/model
  tokenizer_path: /path/to/tokenizer

runtime:
  kind: vllm
  launch: managed_process
  host: 127.0.0.1
  port: 8000
  model_name: test-model
  args:
    max_model_len: 4096
    gpu_memory_utilization: 0.90
    quantization: null

workload:
  prompt_suite: coding_smoke
  max_tokens: 256
  temperature: 0
  num_runs: 3
  concurrency: 1

storage:
  artifact_policy: external_path
  result_policy: managed
```

Acceptance criteria:

* Run specs can be YAML or JSON.
* CLI can execute `llm-bench run spec.yaml`.
* Python API can load the same spec.
* Schema validation gives clear errors.

## 3.2 Add schema versioning

Every spec and result should include a schema version.

Example:

```yaml
schema_version: llm_bench.run_spec.v1
```

Acceptance criteria:

* Old run results can be migrated or read with compatibility shims.
* Breaking schema changes are explicit.

---

# 4. Results Format

## 4.1 Write immutable per-run result directories

Each run should produce a unique immutable result directory.

Suggested format:

```text
<storage-root>/results/runs/2026-05-08/qwen14b-int4-awq-3070ti-smoke__20260508T143012Z__abc123/
  run_spec.yaml
  resolved_spec.yaml
  artifact_manifest.json
  environment.json
  hardware.json
  metrics.jsonl
  summary.json
  stdout.log
  stderr.log
  server.log
  notes.md
```

Acceptance criteria:

* Result directories are append-only once finalized.
* Re-running the same spec creates a new run directory.
* Result summary can be loaded without reading all logs.

## 4.2 Store raw and summarized metrics

Capture both per-request metrics and aggregate summaries.

Per-request metrics:

* request ID
* prompt ID
* prompt tokens
* generated tokens
* time to first token
* decode time
* total wall time
* tokens/sec decode
* tokens/sec wall
* finish reason
* error if any
* peak GPU memory during request if available

Summary metrics:

* mean / median / p95 TTFT
* mean / median / p95 decode tokens/sec
* mean / median / p95 wall tokens/sec
* success rate
* OOM count
* timeout count
* peak VRAM
* average VRAM
* qualitative score if eval is enabled

Acceptance criteria:

* Summary can compare runs across models and runtimes.
* Raw metrics are sufficient to debug outliers.

---

# 5. Runtime Abstraction

## 5.1 Implement runner interface

Create a common runner interface for different runtime types.

Suggested interface:

```python
class RuntimeRunner:
    def prepare(self, spec): ...
    def launch(self, spec): ...
    def wait_until_ready(self, spec): ...
    def run_workload(self, spec): ...
    def collect_logs(self, spec): ...
    def shutdown(self, spec): ...
```

Initial runners:

* `openai_compatible`: benchmark an already-running endpoint.
* `vllm`: launch and benchmark vLLM.
* `trtllm`: launch and benchmark TensorRT-LLM endpoint or engine server.
* `llamacpp`: launch and benchmark llama.cpp server.
* `external`: record commands and benchmark endpoint without owning lifecycle.

Acceptance criteria:

* The ModelOpt project can benchmark already-running servers and managed launched servers.
* Runtime-specific arguments are isolated from generic benchmark logic.

## 5.2 Treat endpoint-only benchmarks as first-class

Many experiments will produce temporary servers rather than durable model files.

Support artifact kind:

```yaml
artifact:
  kind: openai_endpoint
  endpoint: http://127.0.0.1:8009/v1
  model_name: agent-code
```

Acceptance criteria:

* Harness can benchmark Spark/vLLM/LiteLLM endpoints without knowing the underlying model path.
* Harness records server `/v1/models` response when available.
* Harness records user-supplied backend metadata when provided.

---

# 6. Project Integration for ModelOpt 3070 Ti Work

## 6.1 Add example project config

Create an example config for the ModelOpt 3070 Ti project.

Example:

```yaml
project:
  name: modelopt_3070ti
  default_storage_root: /mnt/datasets-big/llm-bench
  artifact_policy: external_path
  result_policy: managed
  tags: [modelopt, 3070ti, quantization]
```

Acceptance criteria:

* The ModelOpt experiment repo can call the harness without vendoring it.
* The benchmark harness can be installed as a dependency.
* Storage root can point to external NVMe or another durable path.

## 6.2 Add ModelOpt-specific metadata hooks

When benchmarking ModelOpt-produced artifacts, capture:

* ModelOpt version
* quantization algorithm
* calibration dataset name/path
* number of calibration samples
* export format
* base model ID
* quantized output path
* producing command
* producing git commit

Acceptance criteria:

* ModelOpt run metadata is included in `artifact_manifest.json`.
* Missing metadata is allowed but clearly marked as unknown.

---

# 7. CLI Requirements

Implement or refactor the CLI around these commands:

```bash
llm-bench init-storage --root /mnt/datasets-big/llm-bench
llm-bench storage-info
llm-bench register-artifact artifact.yaml
llm-bench inspect-artifact /path/to/model
llm-bench run spec.yaml
llm-bench run spec.yaml --storage-root /mnt/datasets-big/llm-bench
llm-bench list-runs --project modelopt_3070ti
llm-bench summarize --project modelopt_3070ti
llm-bench compare RUN_ID_A RUN_ID_B
llm-bench export-summary --project modelopt_3070ti --format markdown
```

Acceptance criteria:

* CLI never silently writes large files into the source repo.
* CLI prints resolved storage locations before running.
* `--dry-run` exists for `run`, `register-artifact`, and `inspect-artifact`.

---

# 8. Storage Safety Rules

Implement guardrails.

## 8.1 Refuse dangerous storage locations by default

Warn or refuse if storage root is:

* `/tmp`
* inside the harness git repo
* inside a model build temp directory
* inside a virtualenv
* inside a Docker overlay path
* low on free disk space

Allow override with:

```bash
--allow-unsafe-storage-root
```

Acceptance criteria:

* Accidental writes to repo-local result directories are prevented.
* User can intentionally override for quick tests.

## 8.2 Detect ephemeral artifact paths

Warn if artifact path is under:

* `/tmp`
* `/var/tmp`
* a known build cache
* a deleted or missing path
* a Docker container filesystem path

Acceptance criteria:

* Temporary model artifacts can still be benchmarked.
* Result metadata clearly says whether artifact was durable at benchmark time.

---

# 9. Tests

## 9.1 Unit tests

Add tests for:

* storage root resolution
* safe/unsafe storage root detection
* artifact manifest creation
* run spec validation
* result directory creation
* external artifact references
* managed copy mode
* managed symlink mode
* schema version handling

## 9.2 Integration tests

Add lightweight integration tests using a fake OpenAI-compatible endpoint or mocked runner.

Acceptance criteria:

* Tests do not require GPU.
* Tests do not download large models.
* CI can run on ordinary CPU machine.

## 9.3 Golden output tests

Add golden tests for:

* `summary.json`
* markdown report output
* compare output
* artifact manifests

Acceptance criteria:

* Changes to output schema are visible in diffs.

---

# 10. Migration from Current Harness

## 10.1 Inventory current repo-local storage

Find all places where scripts assume paths like:

```text
./results
./runs
./models
./artifacts
./outputs
```

Replace with `StorageConfig` calls.

Acceptance criteria:

* No benchmark code writes to repo-relative paths except tests/examples.
* Existing historical results can still be read or migrated.

## 10.2 Add migration command

Optional but useful:

```bash
llm-bench migrate-results ./old-results --storage-root /mnt/datasets-big/llm-bench
```

Acceptance criteria:

* Existing result files are copied into the new storage layout.
* Original files are not deleted.
* Migration writes a manifest of what was moved.

---

# 11. Documentation

Update docs with:

1. Library usage example.
2. CLI usage example.
3. Storage model explanation.
4. Artifact policy explanation.
5. ModelOpt 3070 Ti example workflow.
6. vLLM endpoint-only workflow.
7. TensorRT-LLM engine workflow.
8. llama.cpp comparison workflow.
9. Troubleshooting storage mistakes.

Acceptance criteria:

* A new project can use the harness without modifying the harness repo.
* The storage design is understandable from README alone.

---

# 12. Suggested Implementation Order

1. Add `StorageConfig` and storage root resolution.
2. Move result writing behind storage APIs.
3. Define `RunSpec` and `RunResult` schemas.
4. Convert current benchmark script into importable runner logic.
5. Add CLI wrapper around the runner.
6. Add artifact manifest generation.
7. Add external artifact support.
8. Add immutable result directories.
9. Add endpoint-only benchmark support.
10. Add ModelOpt metadata hooks.
11. Add tests.
12. Add docs and examples.

---

# 13. Definition of Done

This task is complete when:

* The benchmark harness can be installed as a Python package.
* A separate ModelOpt experiment repo can import and call it.
* Benchmarks can be run from any working directory.
* Storage is configured explicitly and is not tied to the harness repo.
* Temporary model artifacts can be benchmarked safely.
* Run results preserve enough metadata to understand what was tested later.
* Endpoint-only, vLLM, TensorRT-LLM, and llama.cpp-style workflows are supported or cleanly stubbed.
* Existing benchmark functionality still works through the new CLI.
* Tests cover storage, schemas, artifact manifests, and at least one mocked benchmark run.

---

# Agent Prompt

You are modifying the existing benchmark harness so it becomes reusable infrastructure for multiple model optimization projects. Do not make it specific to the ModelOpt 3070 Ti experiment, but include that experiment as an example integration.

The most important architectural change is to decouple benchmark code, experiment workspaces, model artifacts, and benchmark results. Do not keep large generated models or durable benchmark results inside the benchmark harness source tree by default.

Implement the storage abstraction first, then migrate existing result-writing code to use it. Preserve existing benchmark behavior where possible, but move it behind a package API and CLI. Add tests that run without GPU access. Avoid downloading large models in tests.

When in doubt, prefer explicit configuration, durable metadata, append-only result directories, and clear warnings over convenience defaults that hide where files are written.
