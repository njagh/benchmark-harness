"""Tests for the OpenAI-compatible client wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APITimeoutError

from bench_harness.models.openai_client import OpenAICompatClient


@pytest.fixture
def mock_openai_response():
    choice = MagicMock()
    choice.message.content = "This is a test response."
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-model"
    return response


@pytest.fixture
def mock_openai_response_no_choices():
    response = MagicMock()
    response.choices = []
    response.usage = MagicMock(prompt_tokens=0, completion_tokens=0)
    response.model = "test-model"
    return response


@pytest.fixture
def client() -> OpenAICompatClient:
    with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai:
        mock_async_openai.return_value = MagicMock()
        return OpenAICompatClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )


class TestOpenAIClientInit:
    """Test OpenAICompatClient initialization."""

    def test_init_with_default_api_key(self):
        with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            client = OpenAICompatClient(
                base_url="http://localhost:8000/v1",
                model="my-model",
            )
            assert client.base_url == "http://localhost:8000/v1"
            assert client.model == "my-model"
            mock_async_openai.assert_called_once()
            kwargs = mock_async_openai.call_args[1]
            assert kwargs["api_key"] == "not-needed"
            assert kwargs["timeout"] == 300.0
            assert kwargs["max_retries"] == 3

    def test_init_with_custom_api_key(self):
        with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            client = OpenAICompatClient(
                base_url="http://localhost:8000/v1",
                api_key="sk-test-key-123",
                model="my-model",
            )
            assert client.base_url == "http://localhost:8000/v1"
            assert client.model == "my-model"
            mock_async_openai.assert_called_once()
            kwargs = mock_async_openai.call_args[1]
            assert kwargs["api_key"] == "sk-test-key-123"

    def test_init_with_custom_timeout_and_retries(self):
        with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            client = OpenAICompatClient(
                base_url="http://localhost:8000/v1",
                model="my-model",
                timeout=600.0,
                max_retries=5,
            )
            mock_async_openai.assert_called_once()
            kwargs = mock_async_openai.call_args[1]
            assert kwargs["timeout"] == 600.0
            assert kwargs["max_retries"] == 5

    def test_init_empty_model(self):
        with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()
            client = OpenAICompatClient(
                base_url="http://localhost:8000/v1",
            )
            assert client.model == ""

    def test_init_logs_message(self):
        with patch("bench_harness.models.openai_client.AsyncOpenAI") as mock_async_openai, \
             patch("bench_harness.models.openai_client.logger") as mock_logger:
            mock_async_openai.return_value = MagicMock()
            OpenAICompatClient(
                base_url="http://localhost:8000/v1",
                model="test-model",
            )
            mock_logger.info.assert_called_once_with(
                "Initialized OpenAICompatClient: base_url=%s model=%s",
                "http://localhost:8000/v1",
                "test-model",
            )


class TestChatComplete:
    """Test the chat_complete() method."""

    @pytest.mark.asyncio
    async def test_chat_complete_success(self, client: OpenAICompatClient, mock_openai_response):
        client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert result["content"] == "This is a test response."
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["finish_reason"] == "stop"
        assert result["model"] == "test-model"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_chat_complete_with_kwargs(self, client: OpenAICompatClient, mock_openai_response):
        client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
            extra_param="extra_value",
        )
        assert result["content"] == "This is a test response."
        client.client.chat.completions.create.assert_called_once()
        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_param"] == "extra_value"

    @pytest.mark.asyncio
    async def test_chat_complete_no_choices(self, client: OpenAICompatClient, mock_openai_response_no_choices):
        client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response_no_choices)
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["content"] is None
        assert result["finish_reason"] == "unknown"
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0

    @pytest.mark.asyncio
    async def test_chat_complete_api_connection_error(self, client: OpenAICompatClient):
        mock_request = MagicMock()
        client.client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(message="Connection error.", request=mock_request)
        )
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["content"] is None
        assert result["finish_reason"] == "error"
        assert "Connection error" in result["error"]
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0

    @pytest.mark.asyncio
    async def test_chat_complete_api_timeout_error(self, client: OpenAICompatClient):
        client.client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError("Request timed out")
        )
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["content"] is None
        assert result["finish_reason"] == "error"
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_chat_complete_generic_error(self, client: OpenAICompatClient):
        client.client.chat.completions.create = AsyncMock(
            side_effect=ValueError("Unexpected API response")
        )
        result = await client.chat_complete(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["content"] is None
        assert result["finish_reason"] == "error"
        assert "Unexpected API response" in result["error"]

    @pytest.mark.asyncio
    async def test_chat_complete_logs_on_success(self, client: OpenAICompatClient, mock_openai_response):
        with patch("bench_harness.models.openai_client.logger") as mock_logger:
            client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            await client.chat_complete(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5,
                max_tokens=100,
            )
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert call_args[0] == "chat_complete: model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s"
            assert call_args[1] == "test-model"
            assert call_args[2] == 10
            assert call_args[3] == 20
            assert call_args[4] == "stop"

    @pytest.mark.asyncio
    async def test_chat_complete_logs_on_error(self, client: OpenAICompatClient):
        with patch("bench_harness.models.openai_client.logger") as mock_logger:
            mock_request = MagicMock()
            client.client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(message="Connection error.", request=mock_request)
            )
            await client.chat_complete(
                messages=[{"role": "user", "content": "Hello"}],
            )
            mock_logger.error.assert_called_once()


class TestFetchModels:
    """Test the fetch_models() method (maps to /v1/models in the code)."""

    @pytest.mark.asyncio
    async def test_fetch_models_success(self, client: OpenAICompatClient):
        mock_model = MagicMock()
        mock_model.id = "test-model-1"
        mock_model.object = "model"
        mock_model.owned_by = "organization"

        mock_models_list = MagicMock()
        mock_models_list.data = [mock_model]
        mock_models_list.object = "list"

        client.client.models.list = AsyncMock(return_value=mock_models_list)
        result = await client.fetch_models()
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "test-model-1"
        assert result["data"][0]["owned_by"] == "organization"

    @pytest.mark.asyncio
    async def test_fetch_models_empty_list(self, client: OpenAICompatClient):
        mock_models_list = MagicMock()
        mock_models_list.data = []
        mock_models_list.object = "list"
        client.client.models.list = AsyncMock(return_value=mock_models_list)

        result = await client.fetch_models()
        assert result["data"] == []
        assert result["object"] == "list"

    @pytest.mark.asyncio
    async def test_fetch_models_connection_error(self, client: OpenAICompatClient):
        mock_request = MagicMock()
        client.client.models.list = AsyncMock(
            side_effect=APIConnectionError(message="Connection error.", request=mock_request)
        )
        result = await client.fetch_models()
        assert "error" in result
        assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_models_timeout_error(self, client: OpenAICompatClient):
        client.client.models.list = AsyncMock(
            side_effect=APITimeoutError("Request timed out")
        )
        result = await client.fetch_models()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_models_generic_error(self, client: OpenAICompatClient):
        client.client.models.list = AsyncMock(
            side_effect=ValueError("Some error")
        )
        result = await client.fetch_models()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_models_missing_object_attr(self, client: OpenAICompatClient):
        mock_model = MagicMock()
        mock_model.id = "test-model-1"
        mock_model.object = "model"
        mock_model.owned_by = "test"

        mock_models_list = MagicMock()
        mock_models_list.data = [mock_model]
        # Remove the 'object' attribute to test fallback
        del mock_models_list.object

        client.client.models.list = AsyncMock(return_value=mock_models_list)
        result = await client.fetch_models()
        assert result["object"] == "list"
        assert len(result["data"]) == 1


class TestChatCompleteStream:
    """Test the chat_complete_stream() method."""

    @pytest.mark.asyncio
    async def test_stream_success(self, client: OpenAICompatClient):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = []

        mock_stream = [mock_chunk1, mock_chunk2, mock_chunk3]

        async def mock_async_iter():
            for chunk in mock_stream:
                yield chunk

        client.client.chat.completions.create = AsyncMock(return_value=mock_async_iter())
        chunks = []
        async for chunk in client.chat_complete_stream(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, client: OpenAICompatClient):
        mock_request = MagicMock()
        client.client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(message="Connection error.", request=mock_request)
        )
        chunks = []
        async for chunk in client.chat_complete_stream(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, client: OpenAICompatClient):
        mock_chunk = MagicMock()
        mock_chunk.choices = []

        async def mock_async_iter():
            yield mock_chunk

        client.client.chat.completions.create = AsyncMock(return_value=mock_async_iter())
        chunks = []
        async for chunk in client.chat_complete_stream(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, client: OpenAICompatClient):
        client.client.chat.completions.create = AsyncMock(
            side_effect=ValueError("Stream error")
        )
        chunks = []
        async for chunk in client.chat_complete_stream(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)
        assert chunks == []
