from bench_harness.runners.base import RuntimeRunner, ProcessHandle
from bench_harness.runners.openai_compatible import OpenAICompatibleRunner
from bench_harness.runners.vllm import VLLMRunner
from bench_harness.runners.trtllm import TRTLLMRunner
from bench_harness.runners.llamacpp import LlamaCPPRunner
from bench_harness.runners.external import ExternalRunner

RUNNER_REGISTRY: dict[str, type[RuntimeRunner]] = {
    "openai_compatible": OpenAICompatibleRunner,
    "vllm": VLLMRunner,
    "trtllm": TRTLLMRunner,
    "llamacpp": LlamaCPPRunner,
    "external": ExternalRunner,
}


def get_runner(kind: str, config) -> RuntimeRunner:
    """Get a runner by kind name."""
    runner_cls = RUNNER_REGISTRY.get(kind)
    if runner_cls is None:
        raise ValueError(
            f"Unknown runner kind: {kind}. Available: {list(RUNNER_REGISTRY.keys())}"
        )
    return runner_cls(config)


__all__ = [
    "RuntimeRunner",
    "ProcessHandle",
    "OpenAICompatibleRunner",
    "VLLMRunner",
    "TRTLLMRunner",
    "LlamaCPPRunner",
    "ExternalRunner",
    "RUNNER_REGISTRY",
    "get_runner",
]
