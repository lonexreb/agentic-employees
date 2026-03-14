"""Tests for VLLMLoRAManager — mocked httpx calls."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.inference.vllm_lora import VLLMLoRAManager


@pytest.fixture
def mock_http():
    client = AsyncMock()
    return client


@pytest.fixture
def manager(mock_http):
    mgr = VLLMLoRAManager(base_url="http://fake-vllm:8000")
    mgr._http = mock_http
    return mgr


async def test_load_adapter(manager, mock_http):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_http.post.return_value = mock_response

    await manager.load_adapter("my-lora", "/path/to/lora")

    mock_http.post.assert_awaited_once_with(
        "/v1/load_lora_adapter",
        json={"lora_name": "my-lora", "lora_path": "/path/to/lora", "load_inplace": True},
    )


async def test_load_adapter_no_inplace(manager, mock_http):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_http.post.return_value = mock_response

    await manager.load_adapter("my-lora", "/path/to/lora", inplace=False)

    call_kwargs = mock_http.post.call_args
    assert call_kwargs.kwargs["json"]["load_inplace"] is False


async def test_unload_adapter(manager, mock_http):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_http.post.return_value = mock_response

    await manager.unload_adapter("my-lora")

    mock_http.post.assert_awaited_once_with(
        "/v1/unload_lora_adapter",
        json={"lora_name": "my-lora"},
    )


async def test_health_check(manager, mock_http):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"id": "base-model"}, {"id": "lora-v1"}]
    }
    mock_http.get.return_value = mock_response

    models = await manager.health_check()

    assert models == ["base-model", "lora-v1"]
    mock_http.get.assert_awaited_once_with("/v1/models")


async def test_health_check_empty(manager, mock_http):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_http.get.return_value = mock_response

    models = await manager.health_check()
    assert models == []
