"""RunResult schema — structured benchmark result capture."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class RequestResult(BaseModel):
    request_id: str
    prompt_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    decode_ms: float
    total_wall_ms: float
    tokens_per_second_decode: float
    tokens_per_second_wall: float
    finish_reason: str
    error: str | None = None
    peak_gpu_memory_mb: float | None = None
    quality_score: float | None = None
    quality_explanation: str | None = None


class ResultSummary(BaseModel):
    mean_ttft_ms: float
    median_ttft_ms: float
    p95_ttft_ms: float
    mean_decode_tps: float
    median_decode_tps: float
    p95_decode_tps: float
    mean_wall_tps: float
    median_wall_tps: float
    p95_wall_tps: float
    success_rate: float
    oom_count: int = 0
    timeout_count: int = 0
    peak_vram_mb: float | None = None
    average_vram_mb: float | None = None
    qualitative_score: float | None = None
    quality_stddev: float = 0.0

    @staticmethod
    def from_requests(requests: list[RequestResult]) -> "ResultSummary":
        """Compute summary statistics from a list of per-request results."""
        if not requests:
            return ResultSummary(
                mean_ttft_ms=0,
                median_ttft_ms=0,
                p95_ttft_ms=0,
                mean_decode_tps=0,
                median_decode_tps=0,
                p95_decode_tps=0,
                mean_wall_tps=0,
                median_wall_tps=0,
                p95_wall_tps=0,
                success_rate=0.0,
                quality_stddev=0.0,
            )

        successful = [r for r in requests if r.error is None]
        oom_count = sum(
            1 for r in requests if r.error and 'OOM' in r.error.upper()
        )
        timeout_count = sum(
            1 for r in requests if r.error and 'timeout' in r.error.lower()
        )

        ttfts = [r.ttft_ms for r in requests]
        decode_tps = [
            r.tokens_per_second_decode
            for r in requests
            if r.tokens_per_second_decode > 0
        ]
        wall_tps = [
            r.tokens_per_second_wall
            for r in requests
            if r.tokens_per_second_wall > 0
        ]
        quality_scores = [
            r.quality_score for r in requests if r.quality_score is not None
        ]
        vram_values = [
            r.peak_gpu_memory_mb
            for r in requests
            if r.peak_gpu_memory_mb is not None
        ]

        def p95(data: list[float]) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(0.95 * len(sorted_data))
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return ResultSummary(
            mean_ttft_ms=statistics.mean(ttfts),
            median_ttft_ms=statistics.median(ttfts),
            p95_ttft_ms=p95(ttfts),
            mean_decode_tps=statistics.mean(decode_tps) if decode_tps else 0.0,
            median_decode_tps=(
                statistics.median(decode_tps) if decode_tps else 0.0
            ),
            p95_decode_tps=p95(decode_tps),
            mean_wall_tps=statistics.mean(wall_tps) if wall_tps else 0.0,
            median_wall_tps=(
                statistics.median(wall_tps) if wall_tps else 0.0
            ),
            p95_wall_tps=p95(wall_tps),
            success_rate=len(successful) / len(requests),
            oom_count=oom_count,
            timeout_count=timeout_count,
            peak_vram_mb=max(vram_values) if vram_values else None,
            average_vram_mb=(
                statistics.mean(vram_values) if vram_values else None
            ),
            qualitative_score=(
                statistics.mean(quality_scores) if quality_scores else None
            ),
            quality_stddev=(
                statistics.stdev(quality_scores)
                if len(quality_scores) > 1
                else 0.0
            ),
        )


class RunResult(BaseModel):
    schema_version: str = "llm_bench.run_result.v1"
    run_id: str
    run_spec_ref: str
    project: str
    artifact_fingerprint: dict[str, Any] = Field(default_factory=dict)
    artifact_durable: bool | None = None
    artifact_warnings: list[str] = Field(default_factory=list)
    per_request: list[RequestResult] = Field(default_factory=list)
    summary: ResultSummary | None = None

    def finalize(self) -> "RunResult":
        """Compute summary from per-request data."""
        self.summary = ResultSummary.from_requests(self.per_request)
        return self

    def write_to_directory(self, run_dir: Path) -> None:
        """Write result files to a run directory."""
        metrics_path = run_dir / "metrics.jsonl"
        with open(metrics_path, 'w') as f:
            for req in self.per_request:
                f.write(req.model_dump_json() + '\n')

        if self.summary:
            summary_path = run_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(
                    self.summary.model_dump(mode='python'),
                    f,
                    indent=2,
                    default=str,
                )

        result_path = run_dir / "run_result.json"
        with open(result_path, 'w') as f:
            json.dump(
                self.model_dump(mode='python'),
                f,
                indent=2,
            )
