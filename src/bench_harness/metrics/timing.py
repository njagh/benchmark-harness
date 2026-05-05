"""Timing utilities and streaming metrics for benchmark runs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator


@dataclass
class TimingRecord:
    """Record of a timed block.

    Attributes:
        label: Human-readable label for the timed block.
        start_wall: time.perf_counter() value at start.
        end_wall: time.perf_counter() value at end (None if not yet ended).
        elapsed_ms: Elapsed time in milliseconds (None if not yet ended).
    """

    label: str
    start_wall: float = field(default=0.0, repr=False)
    end_wall: float | None = None
    elapsed_ms: float | None = None

    def end(self) -> None:
        """Mark the end of this timing record."""
        self.end_wall = time.perf_counter()
        self.elapsed_ms = (self.end_wall - self.start_wall) * 1000.0


@dataclass
class StreamMetrics:
    """Metrics captured from a streaming API response.

    Attributes:
        ttft_ms: Time to first token in milliseconds.
        decode_ms: Time from first to last token in milliseconds.
        total_wall_ms: Total wall-clock time of the request.
        chunk_count: Number of streamed chunks received.
        chars_per_chunk_avg: Average characters per chunk.
    """

    ttft_ms: float
    decode_ms: float | None = None
    total_wall_ms: float = 0.0
    chunk_count: int = 0
    chars_per_chunk_avg: float = 0.0


class StreamingTimer:
    """Captures fine-grained timing metrics from a streaming response.

    Usage:
        timer = StreamingTimer()
        timer.start()
        for chunk in stream:
            timer.on_chunk(chunk)
        metrics = timer.finalize()
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.first_token_time: float | None = None
        self.last_chunk_time: float = 0.0
        self.chunk_count: int = 0
        self.total_chars: int = 0

    def start(self) -> None:
        """Mark the start of the streaming request."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.chunk_count = 0
        self.total_chars = 0

    def on_chunk(self, text: str) -> None:
        """Called for each streamed chunk. Captures TTFT on first call.

        Args:
            text: The text content of the chunk.
        """
        now = time.perf_counter()
        self.last_chunk_time = now
        self.chunk_count += 1
        self.total_chars += len(text)

        if self.first_token_time is None:
            self.first_token_time = now

    def finalize(self) -> StreamMetrics:
        """Compute and return final streaming metrics.

        Returns:
            StreamMetrics with all computed timing values.
        """
        end_time = time.perf_counter()
        total_wall_ms = (end_time - self.start_time) * 1000.0

        ttft_ms = 0.0
        if self.first_token_time is not None:
            ttft_ms = (self.first_token_time - self.start_time) * 1000.0

        decode_ms: float | None = None
        if self.first_token_time is not None:
            decode_ms = (self.last_chunk_time - self.first_token_time) * 1000.0

        chars_avg = 0.0
        if self.chunk_count > 0:
            chars_avg = self.total_chars / self.chunk_count

        return StreamMetrics(
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            total_wall_ms=total_wall_ms,
            chunk_count=self.chunk_count,
            chars_per_chunk_avg=chars_avg,
        )


@contextmanager
def timed_block(label: str = "block") -> Iterator[TimingRecord]:
    """Context manager that captures wall-clock elapsed time.

    Usage:
        with timed_block("api_call") as record:
            do_something()
        print(record.elapsed_ms)  # milliseconds
    """
    record = TimingRecord(label=label, start_wall=time.perf_counter())
    try:
        yield record
    finally:
        record.end()


def measure_function(
    fn: Any, *args: Any, **kwargs: Any
) -> tuple[Any, float]:
    """Wrap a callable and return (result, elapsed_ms).

    Args:
        fn: Callable to measure.
        *args: Positional arguments passed to fn.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        Tuple of (fn result, elapsed time in milliseconds).
    """
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def format_duration(ms: float) -> str:
    """Convert milliseconds to a human-readable duration string.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Human-readable string, e.g. "1.2s", "450ms", "0.5ms".
    """
    if ms < 1.0:
        return f"{ms * 1000:.1f}us"
    elif ms < 1000.0:
        return f"{ms:.0f}ms"
    else:
        return f"{ms / 1000:.2f}s"
