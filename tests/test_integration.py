"""Integration test: full manager → worker → feedback loop. Requires NATS server."""

import pytest

from src.events.bus import EventBus
from src.events.types import TaskStatus
from src.manager.manager import Manager
from src.workers.echo_worker import EchoWorker


@pytest.fixture
async def bus():
    b = EventBus()
    await b.connect()
    yield b
    await b.close()


async def test_full_loop(bus: EventBus):
    manager = Manager("test-manager", bus)
    worker = EchoWorker("test-worker", bus)

    await manager.start()
    await worker.start()

    task = await manager.assign_task("coding", "Write a fibonacci function")
    result = await manager.wait_for_result(task.task_id, timeout=10.0)

    assert result.status == TaskStatus.SUCCESS
    assert result.result == "Write a fibonacci function"
    assert result.worker_id == "test-worker"
    assert len(result.steps) == 3
    assert result.elapsed_seconds > 0

    await manager.publish_feedback(result, score=0.9, text="Echoed correctly")
