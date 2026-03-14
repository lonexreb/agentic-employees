"""Unit tests for Bridge HTTP API endpoints (mock NATS bus)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from src.bridge.http_api import create_app
from src.events.types import ResultEvent, TaskStatus


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def results_store():
    return {}


@pytest.fixture
def waiters_store():
    return {}


@pytest.fixture
async def client(mock_bus, results_store, waiters_store, aiohttp_client):
    app = create_app(mock_bus, results_store, waiters_store)
    return await aiohttp_client(app)


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"


async def test_assign_task(client, mock_bus):
    resp = await client.post("/tasks/assign", json={
        "manager_id": "manager-01",
        "task_type": "coding",
        "prompt": "Write hello world",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["topic"] == "tasks.coding"
    assert data["status"] == "published"
    assert "task_id" in data
    mock_bus.publish.assert_called_once()


async def test_assign_task_invalid_json(client):
    resp = await client.post(
        "/tasks/assign",
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400


async def test_assign_task_validation_error(client):
    resp = await client.post("/tasks/assign", json={
        "task_type": "coding",
        # missing manager_id
    })
    assert resp.status == 422


async def test_submit_result(client, mock_bus):
    resp = await client.post("/tasks/result", json={
        "task_id": "test-123",
        "worker_id": "worker-01",
        "prompt": "Write hello world",
        "result": "print('hello')",
        "status": "success",
        "steps": ["Understood the task", "Wrote the code"],
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["topic"] == "results.coding"
    mock_bus.publish.assert_called_once()


async def test_submit_result_failed_status(client, mock_bus):
    resp = await client.post("/tasks/result", json={
        "task_id": "test-456",
        "worker_id": "worker-01",
        "result": "Could not solve",
        "status": "failed",
    })
    assert resp.status == 201


async def test_submit_feedback(client, mock_bus):
    resp = await client.post("/feedback", json={
        "task_id": "test-123",
        "manager_id": "manager-01",
        "worker_id": "worker-01",
        "score": 0.85,
        "textual_feedback": "Good work",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["status"] == "published"
    mock_bus.publish.assert_called_once()


async def test_submit_feedback_invalid_score(client):
    resp = await client.post("/feedback", json={
        "task_id": "test-123",
        "manager_id": "manager-01",
        "worker_id": "worker-01",
        "score": 1.5,  # out of range
    })
    assert resp.status == 422


async def test_task_status_pending(client):
    resp = await client.get("/tasks/unknown-id/status")
    data = await resp.json()
    assert data["status"] == "pending"


async def test_task_status_completed(client, results_store):
    result = ResultEvent(
        task_id="done-123",
        worker_id="worker-01",
        result="print('hi')",
        status=TaskStatus.SUCCESS,
        steps=["step1"],
    )
    results_store["done-123"] = result

    resp = await client.get("/tasks/done-123/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "completed"
    assert data["result"]["task_id"] == "done-123"
