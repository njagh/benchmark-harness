"""Token counting utilities for benchmark runs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TokenCounter:
    """Counts tokens from API usage data.

    Attributes:
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of generated/completion tokens.
        total_tokens: Total tokens (prompt + completion).
        source: Where token counts came from ("api" or "fallback_tokenizer").
    """

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.source: str = "api"

    def from_api_usage(self, usage: dict | None) -> "TokenCounter":
        """Extract token counts from an OpenAI-compatible usage dict.

        Handles flat dicts like {'prompt_tokens': N, 'completion_tokens': N}
        and nested usage objects from various backends.

        Args:
            usage: Usage dict from API response (may be None).

        Returns:
            Self, for method chaining.
        """
        if usage is None:
            self.prompt_tokens = -1
            self.completion_tokens = -1
            self.total_tokens = -1
            self.source = "unknown"
            logger.warning("No API usage data available; token counts set to -1")
            return self

        self.prompt_tokens = usage.get("prompt_tokens", 0) or 0
        self.completion_tokens = usage.get("completion_tokens", 0) or 0
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.source = "api"
        return self

    def from_response(self, response: Any) -> "TokenCounter":
        """Extract token counts from an OpenAI SDK response object.

        Handles response.usage which is typically a pydantic model with
        prompt_tokens, completion_tokens, total_tokens attributes.

        Args:
            response: OpenAI SDK completion response object.

        Returns:
            Self, for method chaining.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return self.from_api_usage(None)

        usage_dict = {}
        for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, attr, None)
            if val is not None:
                usage_dict[attr] = val

        return self.from_api_usage(usage_dict if usage_dict else None)

    @property
    def has_valid_counts(self) -> bool:
        """Return True if all token counts are non-negative."""
        return (
            self.prompt_tokens >= 0
            and self.completion_tokens >= 0
            and self.total_tokens >= 0
        )


class FallbackTokenCounter:
    """Counts tokens using a local tokenizer when API doesn't return usage data.

    Uses tiktoken with configurable encoding (default cl100k_base for GPT-4).
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer_name = tokenizer_name
        self._encoder = None

    @property
    def encoder(self):
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self.tokenizer_name)
            except ImportError:
                logger.warning(
                    "tiktoken not available; FallbackTokenCounter will not work. "
                    "Install with: pip install tiktoken"
                )
                self._encoder = None
        return self._encoder

    def count_prompt(self, messages: list[dict]) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Estimated token count.
        """
        if self.encoder is None:
            logger.warning("tiktoken unavailable, returning 0 for prompt token count")
            return 0

        enc = self.encoder
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(enc.encode(content))
        return total

    def count_completion(self, text: str) -> int:
        """Count tokens in completion text.

        Args:
            text: The generated text string.

        Returns:
            Estimated token count.
        """
        if self.encoder is None:
            logger.warning("tiktoken unavailable, returning 0 for completion token count")
            return 0
        return len(self.encoder.encode(text))

    def count_total(self, messages: list[dict], text: str) -> int:
        """Count total tokens (prompt + completion).

        Args:
            messages: List of prompt messages.
            text: The completion text.

        Returns:
            Total estimated token count.
        """
        return self.count_prompt(messages) + self.count_completion(text)


def normalize_usage(usage: Any) -> dict:
    """Normalize usage objects across backend formats.

    Handles:
    - OpenAI SDK response.usage object
    - Flat dict from API responses
    - LiteLLM wrapped usage
    - vLLM direct usage

    Args:
        usage: Raw usage data from any backend.

    Returns:
        Normalized dict with prompt_tokens, completion_tokens, total_tokens.
    """
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # If it's a dict already, return a copy with normalized keys
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }

    # If it's an object (e.g. OpenAI SDK), extract attributes
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def compute_tokens_per_second(tokens: int, duration_ms: float) -> float:
    """Compute tokens per second from token count and duration.

    Args:
        tokens: Number of tokens generated.
        duration_ms: Duration in milliseconds.

    Returns:
        Tokens per second, or 0.0 if duration is non-positive.
    """
    if duration_ms <= 0 or tokens <= 0:
        return 0.0
    return tokens / (duration_ms / 1000.0)
