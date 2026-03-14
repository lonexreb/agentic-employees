"""Tests for EventBus pub/sub — requires NATS server running."""

import asyncio

import pytest

from src.events.bus import EventBus
from src.events.types import TaskEvent
from tests.conftest import requires_nats


@pytest.fixture
async def bus():
    b = EventBus()
    await b.connect()
    yield b
    await b.close()


@requires_nats
async def test_publish_subscribe_roundtrip(bus: EventBus):
    received: list[TaskEvent] = []
    done = asyncio.Event()

    async def handler(event: TaskEvent) -> None:
        received.append(event)
        done.set()

    await bus.subscribe("test.roundtrip", TaskEvent, handler)
    task = TaskEvent(manager_id="m1", task_type="coding", prompt="test prompt")
    await bus.publish("test.roundtrip", task)

    await asyncio.wait_for(done.wait(), timeout=5.0)
    assert len(received) == 1
    assert received[0].task_id == task.task_id
    assert received[0].prompt == "test prompt"


@requires_nats
async def test_multiple_subscribers(bus: EventBus):
    received_a: list[TaskEvent] = []
    received_b: list[TaskEvent] = []
    done = asyncio.Event()
    count = 0

    async def handler_a(event: TaskEvent) -> None:
        nonlocal count
        received_a.append(event)
        count += 1
        if count >= 2:
            done.set()

    async def handler_b(event: TaskEvent) -> None:
        nonlocal count
        received_b.append(event)
        count += 1
        if count >= 2:
            done.set()

    await bus.subscribe("test.multi", TaskEvent, handler_a)
    await bus.subscribe("test.multi", TaskEvent, handler_b)

    task = TaskEvent(manager_id="m1", task_type="coding", prompt="fan out")
    await bus.publish("test.multi", task)

    await asyncio.wait_for(done.wait(), timeout=5.0)
    assert len(received_a) == 1
    assert len(received_b) == 1
