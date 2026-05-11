# ModelOpt 3070 Ti Example

Example workflow for benchmarking ModelOpt-quantized models on an RTX 3070 Ti (8GB VRAM).

## Prerequisites

- RTX 3070 Ti with 8GB VRAM
- ModelOpt quantization tools installed
- vLLM serving a quantized model

## Quick Start

### 1. Install the harness as a dependency

```bash
pip install -e /path/to/benchmark-harness
```

### 2. Initialize storage

Point storage to your dataset directory:

```bash
bench-harness init-storage --root /path/to/datasets/llm-bench
```

Or set via environment variable:

```bash
export LLM_BENCH_STORAGE_ROOT=/path/to/datasets/llm-bench
```

### 3. Run a benchmark

```bash
# vLLM managed process
bench-harness run examples/modelopt_3070ti/run-vllm-smoke.yaml

# Endpoint-only (already running server)
bench-harness run examples/modelopt_3070ti/run-endpoint-only.yaml
```

### 4. Inspect results

```bash
bench-harness summarize --project modelopt_3070ti
bench-harness compare <run_a_dir> <run_b_dir>
```

### 5. Export summary

```bash
bench-harness export-summary --project modelopt_3070ti --format markdown
```

## ModelOpt Metadata Hook

The `.llm-bench.yaml` references `ModelOptMetadataHook` from `modelopt_bench.hooks`.
This hook captures quantization details (type, bits, algorithm) into the run metadata
for later comparison across quantization strategies.

To use it, install the ModelOpt benchmark hooks package and ensure the module is
importable from your Python path.

## File Descriptions

- `.llm-bench.yaml` — Project configuration with hooks and artifact policy
- `run-vllm-smoke.yaml` — Benchmark a quantized Qwen 14B via vLLM managed process
- `run-endpoint-only.yaml` — Benchmark against an already-running OpenAI-compatible endpoint
