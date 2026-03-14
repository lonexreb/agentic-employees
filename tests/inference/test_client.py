"""Tests for InferenceClient protocol and adapters — mocked underlying clients."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference.client import (
    InferenceClient,
    OllamaInferenceClient,
    OpenAIInferenceClient,
    create_client,
)

try:
    import openai  # noqa: F401
    _has_openai = True
except ImportError:
    _has_openai = False

requires_openai = pytest.mark.skipif(not _has_openai, reason="openai not installed")


# --- Protocol conformance ---


class _DummyClient:
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        return "hello"


def test_protocol_conformance():
    assert isinstance(_DummyClient(), InferenceClient)


def test_ollama_adapter_is_inference_client():
    with patch("ollama.AsyncClient"):
        client = OllamaInferenceClient(host="http://fake:11434")
    assert isinstance(client, InferenceClient)


@requires_openai
def test_openai_adapter_is_inference_client():
    with patch("openai.AsyncOpenAI"):
        client = OpenAIInferenceClient(base_url="http://fake:8000/v1")
    assert isinstance(client, InferenceClient)


# --- OllamaInferenceClient ---


async def test_ollama_chat():
    with patch("ollama.AsyncClient") as MockCls:
        mock_instance = AsyncMock()
        MockCls.return_value = mock_instance
        mock_instance.chat.return_value = {"message": {"content": "result text"}}

        client = OllamaInferenceClient(host="http://test:11434")
        result = await client.chat(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert result == "result text"
    mock_instance.chat.assert_awaited_once()
    call_kwargs = mock_instance.chat.call_args
    assert call_kwargs.kwargs["model"] == "test-model"


async def test_ollama_chat_json_mode():
    with patch("ollama.AsyncClient") as MockCls:
        mock_instance = AsyncMock()
        MockCls.return_value = mock_instance
        mock_instance.chat.return_value = {"message": {"content": '{"key": "val"}'}}

        client = OllamaInferenceClient(host="http://test:11434")
        result = await client.chat(
            model="m", messages=[{"role": "user", "content": "q"}], json_mode=True
        )

    assert result == '{"key": "val"}'
    call_kwargs = mock_instance.chat.call_args
    assert call_kwargs.kwargs.get("format") == "json"


# --- OpenAIInferenceClient ---


@requires_openai
async def test_openai_chat():
    with patch("openai.AsyncOpenAI") as MockCls:
        mock_instance = AsyncMock()
        MockCls.return_value = mock_instance

        # Build mock response
        mock_message = MagicMock()
        mock_message.content = "openai result"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIInferenceClient(base_url="http://test:8000/v1")
        result = await client.chat(model="gpt-test", messages=[{"role": "user", "content": "hi"}])

    assert result == "openai result"


@requires_openai
async def test_openai_chat_json_mode():
    with patch("openai.AsyncOpenAI") as MockCls:
        mock_instance = AsyncMock()
        MockCls.return_value = mock_instance

        mock_message = MagicMock()
        mock_message.content = '{"key": "val"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        client = OpenAIInferenceClient(base_url="http://test:8000/v1")
        result = await client.chat(
            model="m", messages=[{"role": "user", "content": "q"}], json_mode=True
        )

    assert result == '{"key": "val"}'
    call_kwargs = mock_instance.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}


# --- Factory ---


def test_create_client_ollama():
    with patch("ollama.AsyncClient"):
        client = create_client("ollama", "http://localhost:11434")
    assert isinstance(client, OllamaInferenceClient)


@requires_openai
def test_create_client_openai():
    with patch("openai.AsyncOpenAI"):
        client = create_client("openai", "http://localhost:8000/v1", "key123")
    assert isinstance(client, OpenAIInferenceClient)


def test_create_client_default_is_ollama():
    with patch("ollama.AsyncClient"):
        client = create_client()
    assert isinstance(client, OllamaInferenceClient)
