"""InferenceClient protocol and adapters for Ollama and OpenAI-compatible servers."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class InferenceClient(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str: ...


class OllamaInferenceClient:
    """Wraps ollama.AsyncClient behind the InferenceClient protocol."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        import ollama

        self._client = ollama.AsyncClient(host=host)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {"model": model, "messages": messages}
        if json_mode:
            kwargs["format"] = "json"
        response = await self._client.chat(**kwargs)
        return response["message"]["content"]


class OpenAIInferenceClient:
    """Wraps openai.AsyncOpenAI behind the InferenceClient protocol.

    Works with any OpenAI-compatible server (vLLM, Semantic Router, LiteLLM).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "unused",
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def create_client(
    backend: str = "ollama",
    base_url: str = "",
    api_key: str = "",
) -> InferenceClient:
    """Factory function to create an InferenceClient from config."""
    if backend == "openai":
        return OpenAIInferenceClient(
            base_url=base_url or "http://localhost:8000/v1",
            api_key=api_key or "unused",
        )
    # Default: ollama
    return OllamaInferenceClient(host=base_url or "http://localhost:11434")
