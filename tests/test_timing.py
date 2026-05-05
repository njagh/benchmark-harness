"""Tests for Milestone 3 — Timing and Token Metrics."""

import time
import tempfile
from pathlib import Path

import pytest

from bench_harness.metrics.timing import (
    TimingRecord,
    StreamMetrics,
    StreamingTimer,
    timed_block,
    measure_function,
    format_duration,
)
from bench_harness.metrics.tokens import (
    TokenCounter,
    FallbackTokenCounter,
    normalize_usage,
    compute_tokens_per_second,
)
from bench_harness.storage.sqlite import SQLiteStore
from bench_harness.runners.completion_runner import RunResult


# ── Timing Tests ───────────────────────────────────────────────────────


class TestTimingRecord:
    def test_timing_record_end(self):
        """TimingRecord computes elapsed_ms after end()."""
        record = TimingRecord(label="test")
        assert record.elapsed_ms is None
        time.sleep(0.05)  # 50ms
        record.end()
        assert record.elapsed_ms is not None
        assert record.elapsed_ms >= 40  # tolerance for sleep

    def test_timing_record_start_wall(self):
        """TimingRecord captures start time after initialization."""
        record = TimingRecord(label="test")
        # start_wall defaults to 0.0, must call end() to set a real value
        assert record.end_wall is None
        time.sleep(0.05)
        record.end()
        assert record.start_wall >= 0  # was initialized


class TestStreamMetrics:
    def test_stream_metrics_default_values(self):
        """StreamMetrics initializes with reasonable defaults."""
        m = StreamMetrics(ttft_ms=0.0)
        assert m.chunk_count == 0
        assert m.chars_per_chunk_avg == 0.0


class TestStreamingTimer:
    def test_streaming_timer_start_and_chunks(self):
        """StreamingTimer captures first token and chunk count."""
        timer = StreamingTimer()
        timer.start()
        time.sleep(0.05)  # Simulate delay before first token
        timer.on_chunk("hello")
        timer.on_chunk(" world")
        metrics = timer.finalize()
        assert metrics.chunk_count == 2
        assert metrics.ttft_ms >= 40  # at least 40ms TTFT
        assert metrics.chars_per_chunk_avg > 0

    def test_streaming_timer_decode_time(self):
        """StreamingTimer computes decode time from first to last chunk."""
        timer = StreamingTimer()
        timer.start()
        time.sleep(0.03)
        timer.on_chunk("first")
        time.sleep(0.02)
        timer.on_chunk("second")
        metrics = timer.finalize()
        assert metrics.decode_ms is not None
        assert metrics.decode_ms >= 10  # at least 10ms decode

    def test_streaming_timer_no_chunks(self):
        """finalize returns valid metrics even with no chunks.
        Without chunks, TTFT is 0 since no token was received.
        """
        timer = StreamingTimer()
        timer.start()
        time.sleep(0.05)
        metrics = timer.finalize()
        assert metrics.ttft_ms == 0.0  # no chunks => no TTFT captured
        assert metrics.decode_ms is None  # no chunks => no decode time
        assert metrics.chunk_count == 0
        assert metrics.total_wall_ms >= 40


class TestTimedBlock:
    def test_timed_block_captures_elapsed(self):
        """timed_block context manager captures elapsed time."""
        with timed_block("sleep_test") as record:
            time.sleep(0.05)  # 50ms
        assert record.elapsed_ms is not None
        assert record.elapsed_ms >= 40

    def test_timed_block_label(self):
        """timed_block preserves label."""
        with timed_block("my_label") as record:
            pass
        assert record.label == "my_label"


class TestMeasureFunction:
    def test_measure_function_returns_result_and_time(self):
        """measure_function returns (result, elapsed_ms)."""
        result, elapsed = measure_function(lambda x: x * 2, 5)
        assert result == 10
        assert elapsed >= 0

    def test_measure_function_accuracy(self):
        """measure_function captures sleep duration within tolerance."""
        def sleep_fn():
            time.sleep(0.05)
        _, elapsed = measure_function(sleep_fn)
        assert elapsed >= 40  # at least 40ms


class TestFormatDuration:
    def test_format_ms(self):
        assert "ms" in format_duration(450)

    def test_format_seconds(self):
        assert "s" in format_duration(1500)

    def test_format_microseconds(self):
        assert "us" in format_duration(0.5)


# ── Token Counting Tests ──────────────────────────────────────────────


class TestTokenCounter:
    def test_from_api_usage_standard(self):
        """TokenCounter parses standard usage dict."""
        tc = TokenCounter()
        tc.from_api_usage({"prompt_tokens": 10, "completion_tokens": 20})
        assert tc.prompt_tokens == 10
        assert tc.completion_tokens == 20
        assert tc.total_tokens == 30
        assert tc.source == "api"
        assert tc.has_valid_counts

    def test_from_api_usage_none(self):
        """TokenCounter handles None usage gracefully."""
        tc = TokenCounter()
        tc.from_api_usage(None)
        assert tc.prompt_tokens == -1
        assert tc.source == "unknown"
        assert not tc.has_valid_counts

    def test_from_api_usage_zero_tokens(self):
        """TokenCounter handles zero token counts."""
        tc = TokenCounter()
        tc.from_api_usage({"prompt_tokens": 0, "completion_tokens": 0})
        assert tc.prompt_tokens == 0
        assert tc.total_tokens == 0
        assert tc.has_valid_counts

    def test_from_response_object(self):
        """TokenCounter extracts from response.usage object."""
        tc = TokenCounter()

        class FakeUsage:
            prompt_tokens = 5
            completion_tokens = 15
            total_tokens = 20

        class FakeResponse:
            usage = FakeUsage()

        tc.from_response(FakeResponse())
        assert tc.prompt_tokens == 5
        assert tc.completion_tokens == 15


class TestFallbackTokenCounter:
    def test_fallback_encoder_available(self):
        """FallbackTokenCounter can load tiktoken encoder."""
        fc = FallbackTokenCounter("cl100k_base")
        assert fc.encoder is not None

    def test_fallback_count_completion(self):
        """FallbackTokenCounter counts tokens in text."""
        fc = FallbackTokenCounter("cl100k_base")
        tokens = fc.count_completion("Hello world")
        assert tokens > 0

    def test_fallback_count_prompt(self):
        """FallbackTokenCounter counts tokens in messages."""
        fc = FallbackTokenCounter("cl100k_base")
        messages = [{"role": "user", "content": "What is 2+2?"}]
        tokens = fc.count_prompt(messages)
        assert tokens > 0


class TestNormalizeUsage:
    def test_normalize_none(self):
        result = normalize_usage(None)
        assert result["prompt_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_normalize_dict(self):
        result = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert result["prompt_tokens"] == 10

    def test_normalize_object(self):
        class Usage:
            prompt_tokens = 8
            completion_tokens = 12
            total_tokens = 20
        result = normalize_usage(Usage())
        assert result["total_tokens"] == 20


class TestComputeTokensPerSecond:
    def test_compute_tps_positive(self):
        tps = compute_tokens_per_second(100, 2000)
        assert tps == 50.0

    def test_compute_tps_zero_duration(self):
        tps = compute_tokens_per_second(100, 0)
        assert tps == 0.0

    def test_compute_tps_negative_duration(self):
        tps = compute_tokens_per_second(100, -100)
        assert tps == 0.0

    def test_compute_tps_zero_tokens(self):
        tps = compute_tokens_per_second(0, 1000)
        assert tps == 0.0


# ── SQLite Timing Tests ───────────────────────────────────────────────


class TestSQLiteTiming:
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteStore(db_path)
        store.init()
        yield store
        Path(db_path).unlink(missing_ok=True)

    def test_sqlite_timing_save_and_retrieve(self, temp_db):
        """Round-trip of timing record through save_run_timing."""
        result = RunResult(
            run_id="timing-001",
            suite_id="smoke",
            task_id="smoke.factual_001",
            model_alias="agent-code",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            ttft_ms=120.0,
            prefill_ms=30.0,
            decode_ms=80.0,
            total_wall_ms=200.0,
            tokens_per_second=62.5,
            tokens_per_second_total=75.0,
            token_source="api",
            exit_status="success",
        )
        temp_db.save_run(result)
        summary = temp_db.get_timing_summary(model_alias="agent-code")
        assert len(summary) == 1
        assert summary[0]["mean_ttft_ms"] == 120.0
        assert summary[0]["mean_tps"] == 62.5

    def test_sqlite_timing_summary_aggregates(self, temp_db):
        """get_timing_summary computes correct aggregates."""
        for i in range(5):
            r = RunResult(
                run_id=f"agg-{i}",
                suite_id="smoke",
                task_id=f"t{i}",
                model_alias="model-a",
                ttft_ms=100.0 + i * 10,
                decode_ms=50.0 + i * 5,
                total_wall_ms=200.0 + i * 20,
                completion_tokens=10 + i,
                total_tokens=20 + i,
                exit_status="success",
            )
            temp_db.save_run(r)

        summary = temp_db.get_timing_summary(model_alias="model-a")
        assert len(summary) == 1
        assert summary[0]["run_count"] == 5
        assert summary[0]["mean_ttft_ms"] == 120.0  # (100+110+120+130+140)/5

    def test_sqlite_timing_summary_multiple_models(self, temp_db):
        """Summary works across multiple models."""
        for i in range(2):
            r1 = RunResult(
                run_id=f"m1-{i}", suite_id="smoke", task_id="t1",
                model_alias="model-a", ttft_ms=100.0, exit_status="success",
            )
            r2 = RunResult(
                run_id=f"m2-{i}", suite_id="smoke", task_id="t1",
                model_alias="model-b", ttft_ms=200.0, exit_status="success",
            )
            temp_db.save_run(r1)
            temp_db.save_run(r2)

        summary = temp_db.get_timing_summary()
        assert len(summary) == 2

    def test_sqlite_timing_summary_with_suite_filter(self, temp_db):
        """get_timing_summary filters by suite_id."""
        r1 = RunResult(
            run_id="r1", suite_id="smoke", task_id="t1",
            model_alias="model-a", ttft_ms=100.0, exit_status="success",
        )
        r2 = RunResult(
            run_id="r2", suite_id="coding", task_id="t1",
            model_alias="model-a", ttft_ms=200.0, exit_status="success",
        )
        temp_db.save_run(r1)
        temp_db.save_run(r2)

        smoke_summary = temp_db.get_timing_summary(suite_id="smoke")
        assert len(smoke_summary) == 1
        assert smoke_summary[0]["run_count"] == 1

    def test_schema_migration_safe(self, temp_db):
        """init() is safe on already-initialized DB (no-dup error)."""
        temp_db.init()  # second init should not crash
        r = RunResult(
            run_id="safe-1", suite_id="smoke", task_id="t1",
            model_alias="model-a", exit_status="success",
        )
        temp_db.save_run(r)
        runs = temp_db.get_runs(suite_id="smoke")
        assert len(runs) == 1
