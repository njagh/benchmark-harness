# Milestone 19 — Runner Interface and Runtime Abstraction ✅ DONE

## Goal

Extract a common runner interface that abstracts over different model serving runtimes. This allows the harness to benchmark already-running endpoints, launch its own vLLM/TensorRT-LLM/llama.cpp servers, or record external commands without owning the lifecycle. Endpoint-only benchmarks become first-class.

## Phase

Phase E — Portable library infrastructure

## Dependencies

- Milestone 17: `StorageConfig`, storage layer
- Milestone 18: `RunSpec` schema

## Acceptance Criteria

- [x] `RuntimeRunner` base class with `prepare()`, `launch()`, `wait_until_ready()`, `run_workload()`, `collect_logs()`, `shutdown()` interface.
- [x] Five concrete runners: `OpenAICompatibleRunner`, `VLLMRunner`, `TRTLLMRunner`, `LlamaCPPRunner`, `ExternalRunner`.
- [x] The ModelOpt project can benchmark already-running servers (using `OpenAICompatibleRunner` with `launch="existing"`).
- [x] The ModelOpt project can have the harness launch vLLM (using `VLLMRunner` with `launch="managed_process"`).
- [x] Runtime-specific arguments are isolated from generic benchmark logic.
- [x] Endpoint-only benchmarks work — the harness can benchmark a server without knowing the underlying model path.
- [x] Harness records `/v1/models` response when benchmarking endpoints.
- [x] CLI `bench-run` command fully wires `RuntimeRunner` factory into the benchmark flow.

- `RuntimeRunner` base class with `prepare()`, `launch()`, `wait_until_ready()`, `run_workload()`, `collect_logs()`, `shutdown()` interface.
- Five concrete runners: `OpenAICompatibleRunner`, `VLLMRunner`, `TRTLLMRunner`, `LlamaCPPRunner`, `ExternalRunner`.
- The ModelOpt project can benchmark already-running servers (using `OpenAICompatibleRunner` with `launch="existing"`).
- The ModelOpt project can have the harness launch vLLM (using `VLLMRunner` with `launch="managed_process"`).
- Runtime-specific arguments are isolated from generic benchmark logic.
- Endpoint-only benchmarks work — the harness can benchmark a server without knowing the underlying model path.
- Harness records `/v1/models` response when benchmarking endpoints.

## Subtasks

### 19.1 Define `RuntimeRunner` base class

**File:** `src/bench_harness/runners/base.py`

Abstract interface for all runtime runners:

```python
class RuntimeRunner(ABC):
    """Base class for model serving runtime runners."""

    def __init__(self, config: StorageConfig):
        self.config = config

    @abstractmethod
    def prepare(self, spec: RunSpec) -> dict:
        """Validate spec, resolve artifact path, prepare environment.
        Returns runtime-specific prep dict."""

    @abstractmethod
    def launch(self, spec: RunSpec, prep: dict) -> ProcessHandle | None:
        """Launch the runtime server. Returns process handle if managed, None if external.
        Must bind to spec.runtime.host:spec.runtime.port."""

    @abstractmethod
    def wait_until_ready(self, spec: RunSpec, prep: dict, timeout: float = 120.0) -> bool:
        """Poll the runtime until it responds or timeout."""

    @abstractmethod
    def run_workload(self, spec: RunSpec, prep: dict, result_dir: Path) -> RunResult:
        """Execute the workload against the running runtime.
        Returns aggregated RunResult."""

    @abstractmethod
    def collect_logs(self, spec: RunSpec, prep: dict, result_dir: Path) -> dict:
        """Collect server logs, /v1/models response, runtime version info.
        Returns {filename: content} to write to result_dir."""

    def shutdown(self, spec: RunSpec, prep: dict, handle: ProcessHandle | None) -> None:
        """Stop the runtime if it was managed. No-op for external/endpoint runners."""
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        """Return the runtime kind string, e.g. 'vllm', 'trtllm'."""
```

**ProcessHandle:**
```python
@dataclass
class ProcessHandle:
    proc: subprocess.Popen
    host: str
    port: int
    ready_url: str
```

### 19.2 Implement `OpenAICompatibleRunner`

**File:** `src/bench_harness/runners/openai_compatible.py`

For benchmarking already-running OpenAI-compatible endpoints (vLLM, LiteLLM, Spark, etc.):
- `prepare()`: validates that artifact is an endpoint URL, fetches `/v1/models` if reachable
- `launch()`: returns `None` (no lifecycle ownership)
- `wait_until_ready()`: HTTP health check to `host:port/v1/chat/completions`
- `run_workload()`: delegates to existing `CompletionRunner` logic with `OpenAICompatClient`
- `collect_logs()`: captures `/v1/models` response, runtime version from headers
- Records server metadata in `artifact_manifest.json`

### 19.3 Implement `VLLMRunner`

**File:** `src/bench_harness/runners/vllm.py`

For launching and benchmarking vLLM servers:
- `prepare()`: resolves model artifact path, validates vLLM is installed
- `launch()`: starts `vllm serve <model_path> --host <host> --port <port> [runtime_args]` as managed process
- `wait_until_ready()`: polls until vLLM reports ready
- `run_workload()`: delegates to `OpenAICompatibleRunner.run_workload()` (vLLM is OpenAI-compatible)
- `collect_logs()`: captures vLLM server stdout, `/v1/models`, vLLM version
- Records vLLM version, container name, GPU memory utilization, max model len in manifest

**Files modified:**
- Reuses `src/bench_harness/models/openai_client.py` for the client layer

### 19.4 Implement `TRTLLMRunner`

**File:** `src/bench_harness/runners/trtllm.py`

For launching and benchmarking TensorRT-LLM engines:
- `prepare()`: validates TRT-LLM engine directory exists
- `launch()`: starts TensorRT-LLM REST server (or gRPC endpoint) as managed process
- `wait_until_ready()`: polls the server health endpoint
- `run_workload()`: uses Triton/inference server client or REST API
- `collect_logs()`: captures TRT-LLM server logs, TensorRT version, engine info
- Records TensorRT-LLM version, engine hash in manifest

**Stub initially** — full implementation if TRT-LLM binary is available, otherwise a stub that raises `RuntimeUnavailable` with install instructions.

### 19.5 Implement `LlamaCPPRunner`

**File:** `src/bench_harness/runners/llamacpp.py`

For launching and benchmarking llama.cpp servers:
- `prepare()`: validates GGUF file exists, llama.cpp server binary is available
- `launch()`: starts `llama-server -m <gguf_path> --host <host> --port <port> [runtime_args]`
- `wait_until_ready()`: polls `/health` endpoint
- `run_workload()`: uses OpenAI-compatible client (llama.cpp server exposes OpenAI API)
- `collect_logs()`: captures llama.cpp server logs, version, n-gpu-layers, ctx-size

**Stub initially** — full implementation if llama.cpp binary is available.

### 19.6 Implement `ExternalRunner`

**File:** `src/bench_harness/runners/external.py`

For recording commands and benchmarking an endpoint without owning the lifecycle:
- `prepare()`: records the user-supplied command and endpoint info
- `launch()`: returns `None` (user manages the server)
- `wait_until_ready()`: same as `OpenAICompatibleRunner`
- `run_workload()`: same as `OpenAICompatibleRunner`
- `collect_logs()`: captures the producing command, backend metadata from user, `/v1/models`
- Records `producing_command` and `producing_git_commit` in manifest

### 19.7 Implement runner factory

**File:** `src/bench_harness/runners/__init__.py`

```python
from .base import RuntimeRunner, ProcessHandle
from .openai_compatible import OpenAICompatibleRunner
from .vllm import VLLMRunner
from .trtllm import TRTLLMRunner
from .llamacpp import LlamaCPPRunner
from .external import ExternalRunner

RUNNER_REGISTRY: dict[str, type[RuntimeRunner]] = {
    "openai_compatible": OpenAICompatibleRunner,
    "vllm": VLLMRunner,
    "trtllm": TRTLLMRunner,
    "llamacpp": LlamaCPPRunner,
    "external": ExternalRunner,
}

def get_runner(kind: str, config: StorageConfig) -> RuntimeRunner:
    runner_cls = RUNNER_REGISTRY.get(kind)
    if runner_cls is None:
        raise ValueError(f"Unknown runner kind: {kind}. Available: {list(RUNNER_REGISTRY)}")
    return runner_cls(config)
```

### 19.8 Wire runners into the benchmark flow

Update `cli.py` `run` command to use `RuntimeRunner`:

```python
# Old flow (simplified):
client = OpenAICompatClient(...)
runner = CompletionRunner(client, fallback_tokenizer)
results = await runner.run(...)

# New flow:
runtime = get_runner(spec.runtime.kind, storage_config)
prep = runtime.prepare(spec)
handle = runtime.launch(spec, prep)
runtime.wait_until_ready(spec, prep)
result = runtime.run_workload(spec, prep, result_dir)
logs = runtime.collect_logs(spec, prep, result_dir)
# Write logs to result_dir
if handle:
    runtime.shutdown(spec, prep, handle)
```

The existing `CompletionRunner` logic becomes the workload execution layer inside each runner's `run_workload()`.

## Files Created

- `src/bench_harness/runners/base.py` — `RuntimeRunner` ABC, `ProcessHandle`
- `src/bench_harness/runners/openai_compatible.py` — `OpenAICompatibleRunner`
- `src/bench_harness/runners/vllm.py` — `VLLMRunner`
- `src/bench_harness/runners/trtllm.py` — `TRTLLMRunner` (stub)
- `src/bench_harness/runners/llamacpp.py` — `LlamaCPPRunner` (stub)
- `src/bench_harness/runners/external.py` — `ExternalRunner`

## Files Modified

- `src/bench_harness/runners/__init__.py` — runner factory, registry
- `src/bench_harness/runners/completion_runner.py` — keep as workload execution helper, used by runners
- `src/bench_harness/cli.py` — `run` command uses `RuntimeRunner` factory
- `src/bench_harness/__init__.py` — export `RuntimeRunner`, `ProcessHandle`

## Tests

- `tests/test_runner_factory.py` — runner registry, unknown kind error
- `tests/test_openai_compatible_runner.py` — mock endpoint benchmark
- `tests/test_external_runner.py` — endpoint-only benchmark without server

## Notes

- Existing `CompletionRunner` is refactored but preserved as the workload execution logic. It no longer handles server lifecycle.
- TRT-LLM and llama.cpp runners start as stubs. They should raise clear `RuntimeUnavailable` errors with install instructions rather than failing silently.
- The endpoint-only workflow is the most important path for ModelOpt: many experiments produce temporary servers. `OpenAICompatibleRunner` and `ExternalRunner` make this first-class.
