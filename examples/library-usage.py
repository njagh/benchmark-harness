"""Example: Using bench_harness as a library from another project.

This script demonstrates the high-level API for running benchmarks
programmatically, without using the CLI.

To run from outside the repo:

    pip install -e /path/to/benchmark-harness
    python examples/library-usage.py
"""

from pathlib import Path

from bench_harness import BenchmarkRunner, RunSpec, StorageConfig

# 1. Configure storage
config = StorageConfig(root=Path("/tmp/bench-storage"))

# 2. Load a run spec (or build one programmatically)
try:
    spec = RunSpec.from_yaml("examples/modelopt_3070ti/run-vllm-smoke.yaml")
except FileNotFoundError:
    print("Run spec not found — skipping actual benchmark.")
    print("With a valid spec, this would:")
    print("  1. Register and resolve the model artifact")
    print("  2. Launch the selected runner (vLLM, OpenAI, TRT-LLM, llama.cpp)")
    print("  3. Execute tasks against the model")
    print("  4. Write results to storage")
    print()
    print("Storage config: root=", config.root)
    exit(0)

# 3. Run the benchmark
runner = BenchmarkRunner(storage=config)
result = runner.run(spec)

# 4. Inspect results
print(f"Success rate: {result.summary.success_rate:.0%}")
print(f"Mean TTFT: {result.summary.mean_ttft_ms:.0f}ms")
print(f"Mean decode TPS: {result.summary.mean_decode_tps:.1f}")
