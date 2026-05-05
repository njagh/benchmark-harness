"""OpenAI-compatible client for local model endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)


class OpenAICompatClient:
    """Thin wrapper around AsyncOpenAI for OpenAI-compatible local endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        model: str = "",
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info(
            "Initialized OpenAICompatClient: base_url=%s model=%s",
            base_url,
            model,
        )

    async def chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a chat completion call and return structured result.

        Returns:
            dict with keys: content, usage (prompt_tokens, completion_tokens),
            finish_reason, model
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            response = await self.client.chat.completions.create(**params)
            choice = response.choices[0] if response.choices else None

            usage = response.usage
            result = {
                "content": choice.message.content if choice else None,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": (
                        usage.completion_tokens if usage else 0
                    ),
                },
                "finish_reason": (
                    choice.finish_reason if choice else "unknown"
                ),
                "model": response.model if response else self.model,
            }

            logger.info(
                "chat_complete: model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s",
                self.model,
                result["usage"]["prompt_tokens"],
                result["usage"]["completion_tokens"],
                result["finish_reason"],
            )
            return result

        except (APIConnectionError, APITimeoutError) as e:
            logger.error("API error: %s", e)
            return {
                "content": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "finish_reason": "error",
                "error": str(e),
            }
        except Exception as e:
            logger.error("Unexpected error in chat_complete: %s", e)
            return {
                "content": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "finish_reason": "error",
                "error": str(e),
            }

    async def chat_complete_stream(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion. Yields content chunks as they arrive.

        Tracks time to first token (TTFT) internally.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        ttft_ms: float | None = None

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta.content:
                    if ttft_ms is None:
                        ttft_ms = 0  # First chunk received
                    yield choice.delta.content

        except (APIConnectionError, APITimeoutError) as e:
            logger.error("Stream API error: %s", e)
        except Exception as e:
            logger.error("Unexpected stream error: %s", e)
