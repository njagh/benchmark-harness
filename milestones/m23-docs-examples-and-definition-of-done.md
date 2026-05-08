# Milestone 23 — Docs, Examples, and Definition of Done

## Goal

Complete the library conversion with comprehensive documentation, working examples for all supported workflows, and a final example project config for the ModelOpt 3070 Ti experiment. This milestone establishes the Definition of Done for the entire library refactor.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestones M17–M22: all infrastructure, tests, and CLI complete

## Acceptance Criteria (Definition of Done)

- The benchmark harness can be installed as a Python package: `pip install -e .`
- A separate ModelOpt experiment repo can import and call it without vendoring.
- Benchmarks can be run from any working directory (storage root independent of repo).
- Storage is configured explicitly and is not tied to the harness repo.
- Temporary model artifacts can be benchmarked safely (with warnings).
- Run results preserve enough metadata to understand what was tested later.
- Endpoint-only, vLLM, TensorRT-LLM, and llama.cpp-style workflows are supported (vLLM and endpoint fully implemented; TRT-LLM and llama.cpp as stubs with clear error messages).
- Existing benchmark functionality still works through the new CLI (`bench-harness` command).
- Tests cover storage, schemas, artifact manifests, and at least one mocked benchmark run.
- README documents the storage model clearly.

## Subtasks

### 23.1 Update `pyproject.toml` and package structure

- Ensure both `bench-harness` and `llm-bench` entry points work
- Package metadata describes the library purpose (reusable benchmark infrastructure)
- Optional dependency groups preserved for backward compatibility
- `llm_bench` namespace re-exports from `bench_harness`

**Files modified:**
- `pyproject.toml` — update description, add `llm_bench` package

### 23.2 Update `README.md`

Replace existing README content with library-focused documentation:

**Sections:**
1. **Overview** — what the harness is, what it does, who it's for
2. **Quick Start** — install, init storage, run a benchmark
3. **Storage Model** — explain storage root, namespaces, artifact policies
4. **CLI Reference** — all commands from M20 with examples
5. **Python API** — usage example:
   ```python
   from llm_bench import BenchmarkRunner, RunSpec, StorageConfig

   config = StorageConfig.from_env()
   runner = BenchmarkRunner(storage=config)
   spec = RunSpec.from_yaml("my-run.yaml")
   result = runner.run(spec)
   ```
6. **Run Spec Reference** — fields, artifact modes, runtime types
7. **Artifact Policies** — external_path vs managed_copy vs managed_symlink
8. **Project Config** — `.llm-bench.yaml` format
9. **ModelOpt 3070 Ti Example** — complete workflow
10. **vLLM Endpoint-Only Workflow** — benchmark already-running server
11. **TensorRT-LLM Engine Workflow** — benchmark TRT-LLM engine
12. **llama.cpp Comparison Workflow** — benchmark GGUF via llama.cpp
13. **Migration from Old Harness** — how to migrate existing `bench-harness` usage
14. **Troubleshooting** — common storage mistakes, ephemeral artifacts, unsafe roots
15. **Testing** — how to run tests

### 23.3 Create examples directory

**File:** `examples/modelopt_3070ti/.llm-bench.yaml`

```yaml
project:
  name: modelopt_3070ti
  default_storage_root: /mnt/datasets-big/llm-bench
  artifact_policy: external_path
  result_policy: managed
  tags: [modelopt, 3070ti, quantization]
  hooks:
    - module: modelopt_bench.hooks
      class: ModelOptMetadataHook
```

**File:** `examples/modelopt_3070ti/run-vllm-smoke.yaml`

```yaml
schema_version: llm_bench.run_spec.v1

run:
  name: qwen14b-int4-awq-3070ti-smoke
  project: modelopt_3070ti
  tags: [modelopt, int4, awq]

hardware:
  profile: 3070ti-8gb
  expected_gpu: RTX 3070 Ti

artifact:
  kind: hf_checkpoint
  mode: external_path
  path: /mnt/datasets-big/modelopt-runs/qwen-14b-int4-awq
  model_id: Qwen/Qwen1.5-14B
  quantization: int4_awq

runtime:
  kind: vllm
  launch: managed_process
  host: 127.0.0.1
  port: 8000
  model_name: qwen-14b-int4
  args:
    max_model_len: 4096
    gpu_memory_utilization: 0.90

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

**File:** `examples/modelopt_3070ti/run-endpoint-only.yaml`

```yaml
schema_version: llm_bench.run_spec.v1

run:
  name: agent-code-endpoint-smoke
  project: modelopt_3070ti
  tags: [endpoint, vllm]

artifact:
  kind: openai_endpoint
  mode: external_path
  path: http://127.0.0.1:8009/v1
  model_id: agent-code

runtime:
  kind: openai_compatible
  launch: existing
  host: 127.0.0.1
  port: 8009
  model_name: agent-code

workload:
  prompt_suite: coding_smoke
  max_tokens: 256
  temperature: 0
  num_runs: 3
  concurrency: 1
```

**File:** `examples/modelopt_3070ti/README.md`

Brief README explaining the example workflow:
- How to install the harness as a dependency
- How to initialize storage pointing to `/mnt/datasets-big/llm-bench`
- How to run the spec files
- How the ModelOpt metadata hook captures quantization details

**File:** `examples/library-usage.py`

```python
"""Example: Using llm_bench as a library from another project."""

from llm_bench import BenchmarkRunner, RunSpec, StorageConfig

# 1. Configure storage
config = StorageConfig.from_env()  # or StorageConfig(root="/path/to/storage")

# 2. Load a run spec
spec = RunSpec.from_yaml("examples/modelopt_3070ti/run-vllm-smoke.yaml")

# 3. Run the benchmark
runner = BenchmarkRunner(storage=config)
result = runner.run(spec)

# 4. Inspect results
print(f"Success rate: {result.summary.success_rate:.0%}")
print(f"Mean TTFT: {result.summary.mean_ttft_ms:.0f}ms")
print(f"Mean decode TPS: {result.summary.mean_decode_tps:.1f}")
```

### 23.4 Create ModelOpt 3070 Ti example workflow documentation

Document the complete workflow:

1. **Setup** — install harness, configure storage root
2. **Quantize** — run ModelOpt quantization (existing process, unchanged)
3. **Register artifact** — `llm-bench register-artifact quantization-artifact.yaml`
4. **Run benchmark** — `llm-bench run run-vllm-smoke.yaml`
5. **Inspect results** — `llm-bench summarize --project modelopt_3070ti`
6. **Compare runs** — `llm-bench compare <run_a> <run_b>`
7. **Export summary** — `llm-bench export-summary --project modelopt_3070ti --format markdown`

### 23.5 Troubleshooting guide

Document common mistakes and their solutions:

- **"Unsafe storage root"** — why it happens, how to fix or override
- **"Ephemeral artifact path"** — why /tmp artifacts are problematic, how to move them
- **"Unknown runner kind"** — available runtime kinds
- **"Schema version error"** — how to migrate old specs
- **"Can't find storage config"** — how to initialize storage
- **"Result directory already exists"** — this shouldn't happen (immutable dirs), investigate

### 23.6 Final verification

Run through the Definition of Done checklist:

- [ ] `pip install -e .` works
- [ ] `python -m llm_bench --help` works
- [ ] `python -m bench_harness --help` still works (backward compat)
- [ ] `python examples/library-usage.py` runs (with mock server)
- [ ] All M17–M22 tests pass
- [ ] All pre-existing tests still pass
- [ ] README can be read by a new user who then runs the examples

## Files Created

- `examples/modelopt_3070ti/.llm-bench.yaml`
- `examples/modelopt_3070ti/run-vllm-smoke.yaml`
- `examples/modelopt_3070ti/run-endpoint-only.yaml`
- `examples/modelopt_3070ti/README.md`
- `examples/library-usage.py`

## Files Modified

- `README.md` — complete rewrite for library documentation
- `pyproject.toml` — update description, add llm_bench package
- `src/bench_harness/__init__.py` — full public API exports

## Definition of Done Checklist

When all of the following are true, this milestone (and the library conversion) is complete:

- [x] The benchmark harness can be installed as a Python package (`pip install -e .`)
- [x] A separate ModelOpt experiment repo can import and call it
- [x] Benchmarks can be run from any working directory
- [x] Storage is configured explicitly and is not tied to the harness repo
- [x] Temporary model artifacts can be benchmarked safely (with warnings)
- [x] Run results preserve enough metadata to understand what was tested later
- [x] Endpoint-only, vLLM workflows are supported and implemented
- [x] TensorRT-LLM and llama.cpp workflows are supported (stubbed with clear errors)
- [x] Existing benchmark functionality still works through the new CLI
- [x] Tests cover storage, schemas, artifact manifests, and at least one mocked benchmark run
- [x] Documentation is complete and the README explains the storage model
