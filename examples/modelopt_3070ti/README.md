# ModelOpt 3070 Ti Experiment

This example demonstrates running LLM benchmarks on a machine with an
NVIDIA RTX 3070 Ti (8 GB VRAM) using ModelOpt-quantized models.

## Workflow

1. **Setup** — Install the harness as a dependency
2. **Quantize** — Run ModelOpt quantization on your model
3. **Run benchmark** — Use the included spec files to benchmark

## Prerequisites

- NVIDIA RTX 3070 Ti (or similar GPU with at least 8 GB VRAM)
- Python 3.12+
- PyTorch with CUDA support
- vLLM installed (`pip install vllm`)

## Setup

Install the benchmark harness:

```bash
pip install -e /path/to/benchmark-harness
```

Point storage to your dataset location. Either set the environment variable:

```bash
export LLM_BENCH_STORAGE_ROOT=/mnt/datasets-big/llm-bench
```

Or create a `.llm-bench.yaml` project config (already included in this
directory):

```yaml
project:
  name: modelopt-3070ti-experiment
  default_storage_root: /mnt/datasets-big/llm-bench
  artifact_policy: managed_copy
```

## Quantization with ModelOpt

Run ModelOpt's quantization pipeline on your model. The process writes a
`modelopt_meta.json` file into the output directory. The harness's
`ModelOptMetadataHook` will automatically read this file and capture
quantization details into the run metadata.

```bash
# Example (actual command depends on your ModelOpt setup)
python -m modelopt export awq \
  --model Qwen/Qwen3-8B \
  --output /mnt/datasets-big/modelopt-runs/qwen3-8b-int4-awq \
  --calibration-dataset my-calib-data
```

The output directory should contain the quantized checkpoint plus
`modelopt_meta.json`.

## Running the Examples

### Managed vLLM Process

This spec launches vLLM as a managed subprocess, runs the workload, and
shuts it down:

```bash
llm-bench run examples/modelopt_3070ti/run-vllm-smoke.yaml
```

The runner will:
1. Copy the artifact to the managed store (`managed_copy` policy)
2. Launch vLLM bound to `127.0.0.1:8000`
3. Run 5 benchmark iterations
4. Collect results into the storage root

### Endpoint-Only

This spec connects to an already-running vLLM server (not managed by
the harness):

```bash
# Ensure your vLLM server is running on port 8009
llm-bench run examples/modelopt_3070ti/run-endpoint-only.yaml
```

## Expected Output

After a run completes, you will see:

```
Run directory: /mnt/datasets-big/llm-bench/results/runs/2025-01-15/qwen3-8b-int4-3070ti-smoke__20250115T143022__a1b2c3d4/
  resolved_spec.yaml   — the run configuration that was used
  metrics.jsonl        — per-request timing and quality data
  summary.json         — aggregate statistics
  run_result.json      — full structured result
  logs/                — server stdout/stderr logs
```

The `summary.json` will contain:

```json
{
  "mean_ttft_ms": 12.4,
  "mean_decode_tps": 45.2,
  "success_rate": 1.0,
  "oom_count": 0
}
```

## ModelOpt Metadata Hook

The `ModelOptMetadataHook` in this project config scans the artifact for
`modelopt_meta.json` and enriches the run result with:

- ModelOpt version
- Quantization algorithm
- Calibration dataset info
- Base model ID
- Git commit of the producing run

This is captured in the `artifact_fingerprint` field of each `RunResult`.
