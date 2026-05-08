# Benchmark Harness — Local LLM Quality Evaluation

Evaluates response quality, coding ability, instruction following, and performance
tradeoffs across locally served models. Produces actionable recommendations for
choosing the right model and serving configuration for real workflows.

## Overview

The benchmark harness is a portable, library-first toolkit for benchmarking
large language models. It supports:

- **Structured run specs** — YAML-driven configurations with full metadata
- **Multiple serving backends** — vLLM (managed or existing), TensorRT-LLM,
  llama.cpp, OpenAI-compatible endpoints
- **Safe artifact management** — managed copy, symlink, or direct external path
- **Structured results** — JSONL metrics, per-request timing, aggregate summaries
- **Artifact registry** — JSONL-based tracking across projects and experiments
- **Project hooks** — plug-in metadata enrichment (e.g., ModelOpt quantization details)

## Quick Start

```bash
# Install the harness
pip install -e .

# Run a benchmark from a spec file
llm-bench run examples/modelopt_3070ti/run-endpoint-only.yaml

# Or programmatically
python examples/library-usage.py
```

## Storage Model

The harness uses a **storage root** to organize all outputs. The root is resolved
in this priority order:

1. Explicit CLI flag (e.g., `--storage-root /path`)
2. `LLM_BENCH_STORAGE_ROOT` environment variable
3. `.llm-bench.yaml` project config in cwd or parents
4. Default: `~/.local/share/llm-bench`

The storage root contains these namespaces:

| Namespace | Path | Purpose |
|---|---|---|
| Artifacts | `<root>/artifacts/` | Model checkpoints, engines, tokens |
| Results | `<root>/results/` | Benchmark outputs |
| Runs | `<root>/results/runs/<YYYY-MM-DD>/` | Per-day, immutable run dirs |
| Summaries | `<root>/results/summaries/` | Aggregated summaries |
| Comparisons | `<root>/results/comparisons/` | Run comparison outputs |
| Registry | `<root>/registry/` | JSONL artifact registry |
| Logs | `<root>/logs/` | Server stdout/stderr |
| Cache | `<root>/cache/` | Transient build/cache data |
| Temp | `<root>/tmp/` | Working temp space |

Each run creates an immutable directory:

```
<results>/runs/2025-01-15/<name>__<timestamp>__<hash>/
```

## CLI Reference

```bash
# Run a benchmark from a spec file
llm-bench run run_spec.yaml [--dry-run]

# Show help
llm-bench --help
llm-bench run --help

# Legacy entry point (backward compatible)
bench-harness --help
```

## Python API

```python
from bench_harness import (
    StorageConfig,
    RunSpec,
    ArtifactRegistry,
    ArtifactMetadataHook,
    ModelOptMetadataHook,
)

# 1. Configure storage
config = StorageConfig.from_env()
config.ensure_namespaces()

# 2. Load a run spec from YAML
spec = RunSpec.from_yaml("my-run.yaml")

# 3. Use the artifact registry
registry = ArtifactRegistry(config)
registry.register(your_artifact)

# 4. List and query artifacts
artifacts = registry.list_all()
hf_artifacts = registry.query(kind="hf_checkpoint")
```

## Run Spec Reference

Run specs are YAML files with this structure:

```yaml
schema_version: llm_bench.run_spec.v1

run:
  name: my-run-name          # slug-formatted identifier
  project: my-project         # project name
  tags: [optional, tags]     # arbitrary labels

hardware:                      # optional
  profile: default
  expected_gpu: RTX 4090

artifact:                      # the model/engine to benchmark
  kind: hf_checkpoint          # hf_checkpoint | trtllm_engine | gguf | openai_endpoint | vllm_endpoint
  mode: external_path          # external_path | managed_copy | managed_symlink
  path: /path/to/model         # filesystem path or HTTP(S) URL
  model_id: org/model-name     # original model identifier
  quantization: int4_awq       # optional quantization label

runtime:                       # how to serve the model
  kind: vllm                   # vllm | trtllm | llamacpp | openai_compatible | external
  launch: managed_process      # managed_process | existing
  host: 127.0.0.1
  port: 8000
  model_name: my-model
  args:                        # backend-specific launch arguments
    max_model_len: 4096
    gpu_memory_utilization: 0.90

workload:                      # what to evaluate
  prompt_suite: coding_smoke   # suite of prompts to evaluate
  max_tokens: 256              # max tokens per response
  temperature: 0.0             # sampling temperature
  num_runs: 5                  # number of runs (repetitions)
  concurrency: 1               # concurrent requests

storage:                       # optional overrides
  artifact_policy: external_path
  result_policy: managed
```

## Artifact Policies

### `external_path`

Use the source path directly without copying. The artifact files are accessed
in-place. This is fastest but the source must be durable and accessible.

```yaml
artifact:
  kind: hf_checkpoint
  mode: external_path
  path: /mnt/datasets-big/models/qwen3-8b
```

### `managed_copy`

Copy artifact files to the harness artifact store (`<root>/artifacts/models/`).
Copies are incremental — only changed files are re-copied.

```yaml
artifact:
  kind: hf_checkpoint
  mode: managed_copy
  path: /mnt/datasets-big/models/qwen3-8b
```

### `managed_symlink`

Create a symlink in the artifact store pointing to the source. No data is
duplicated. Useful for large models where copy time is prohibitive.

```yaml
artifact:
  kind: hf_checkpoint
  mode: managed_symlink
  path: /mnt/datasets-big/models/qwen3-8b
```

## Project Config

A `.llm-bench.yaml` file in your project directory sets defaults:

```yaml
project:
  name: my-experiment
  default_storage_root: /mnt/datasets-big/llm-bench
  artifact_policy: managed_copy
  result_policy: managed
  tags: [experiment, quantization]
  hooks:
    - module: myproject.hooks
      class: MyMetadataHook
```

The harness searches for this file in the cwd and all parent directories.

## Example: ModelOpt 3070 Ti

See `examples/modelopt_3070ti/` for a complete workflow:

```bash
# Managed vLLM process (launches and shuts down vLLM automatically)
llm-bench run examples/modelopt_3070ti/run-vllm-smoke.yaml

# Endpoint-only (connects to already-running vLLM)
llm-bench run examples/modelopt_3070ti/run-endpoint-only.yaml
```

The included `ModelOptMetadataHook` reads `modelopt_meta.json` from quantized
models and enriches run results with quantization details.

## vLLM Endpoint Workflow

Benchmark an already-running vLLM server without launching a new process:

```yaml
artifact:
  kind: openai_endpoint
  mode: external_path
  path: http://127.0.0.1:8009/v1

runtime:
  kind: vllm
  launch: existing
  host: 127.0.0.1
  port: 8009
```

Start your vLLM server separately:

```bash
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8009
```

Then run the benchmark. The harness will not attempt to manage the server
process.

## TRT-LLM Engine Workflow

Benchmark a TensorRT-LLM engine. The runner is available as a stub with
clear error messages when invoked:

```yaml
artifact:
  kind: trtllm_engine
  mode: managed_copy
  path: /mnt/datasets-big/engines/qwen3-8b-trtllm

runtime:
  kind: trtllm
  launch: managed_process
  host: 127.0.0.1
  port: 8000
```

Run:

```bash
llm-bench run trtllm-run-spec.yaml
```

## llama.cpp Comparison Workflow

Benchmark GGUF models via llama.cpp:

```yaml
artifact:
  kind: gguf
  mode: managed_copy
  path: /mnt/datasets-big/gguf/qwen3-8b-q4_k_m.gguf

runtime:
  kind: llamacpp
  launch: managed_process
  host: 127.0.0.1
  port: 8000
```

Run:

```bash
llm-bench run llamacpp-run-spec.yaml
```

## Migration from Old Harness

### From flag-based runs

Old style:
```bash
python -m bench_harness run --suite smoke --models agent-code
```

New style:
```yaml
# run-spec.yaml
run:
  name: smoke-agent-code
  project: default
artifact:
  kind: hf_checkpoint
  mode: external_path
  path: agent-code
runtime:
  kind: openai_compatible
  launch: existing
workload:
  prompt_suite: smoke
  max_tokens: 256
  num_runs: 3
```

```bash
llm-bench run run-spec.yaml
```

### From old `bench-harness bench-run`

Old:
```bash
bench-harness bench-run run_spec.yaml
```

New (same command, just use `llm-bench`):
```bash
llm-bench run run_spec.yaml
```

The old `bench-harness` CLI remains available for backward compatibility.

## Troubleshooting

### "Unsafe storage root"

The harness rejects paths that are world-writable or inside /tmp by default.

Fix: set the storage root to a dedicated directory:

```bash
export LLM_BENCH_STORAGE_ROOT=/mnt/datasets-big/llm-bench
```

Or temporarily allow unsafe paths:

```python
config = StorageConfig.from_cli(Path("/tmp/my-storage"), allow_unsafe=True)
```

### "Ephemeral artifact path"

Running benchmarks against models in /tmp or other transient directories
generates warnings. These paths may be cleaned up before the run completes.

Fix: copy or symlink the model to a durable location first.

### "Unknown runner kind"

Available runtime kinds:

| Kind | Description |
|---|---|
| `vllm` | vLLM server (managed or existing) |
| `trtllm` | TensorRT-LLM engine |
| `llamacpp` | llama.cpp server |
| `openai_compatible` | Any OpenAI-compatible endpoint |
| `external` | External runner |

### "Schema version error"

Old run specs may use a different schema version. Use the compatibility
helpers:

```python
from bench_harness import resolve_schema_version, migrate_result_v0_to_v1
```

### "Can't find storage config"

Initialize storage by setting the environment variable:

```bash
export LLM_BENCH_STORAGE_ROOT=/path/to/storage
```

Or create a `.llm-bench.yaml` project config in your working directory.

## Testing

Run the test suite:

```bash
pytest
```

Tests cover storage resolution, schemas, artifact management, and mocked
benchmark runs.
