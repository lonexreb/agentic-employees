"""Integration test: HTTP request → NATS event published.

Requires NATS running at localhost:4222.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.conftest import requires_nats


@requires_nats
async def test_bridge_publishes_task_to_nats():
    """Verify that posting to /tasks/assign publishes a TaskEvent on NATS."""
    import nats as nats_lib
    from aiohttp.test_utils import TestClient, TestServer

    from src.bridge.http_api import create_app
    from src.events.bus import EventBus
    from src.events.types import ResultEvent, TaskEvent

    # Set up real NATS connection
    bus = EventBus()
    await bus.connect("nats://localhost:4222")

    results: dict[str, ResultEvent] = {}
    waiters: dict[str, asyncio.Event] = {}
    app = create_app(bus, results, waiters)

    # Subscribe to NATS directly to verify the event arrives
    nc = await nats_lib.connect("nats://localhost:4222")
    received: list[TaskEvent] = []
    done = asyncio.Event()

    async def on_msg(msg):
        event = TaskEvent.model_validate_json(msg.data)
        received.append(event)
        done.set()

    await nc.subscribe("tasks.coding", cb=on_msg)

    # Send HTTP request
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/tasks/assign", json={
            "manager_id": "manager-01",
            "task_type": "coding",
            "prompt": "Write fibonacci",
        })
        assert resp.status == 201
        data = await resp.json()
        task_id = data["task_id"]

    # Wait for NATS message
    await asyncio.wait_for(done.wait(), timeout=5.0)
    assert len(received) == 1
    assert received[0].task_id == task_id
    assert received[0].prompt == "Write fibonacci"

    await nc.drain()
    await bus.close()
