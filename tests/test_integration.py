"""Integration test: full manager -> worker -> feedback loop. Requires NATS server."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.events.bus import EventBus
from src.events.topics import TRAINING_ROLLOUTS
from src.events.types import TaskStatus, TrainingRolloutEvent
from src.manager.manager import Manager
from src.rewards.prm_evaluator import PRMEvaluator
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
    assert result.prompt == "Write a fibonacci function"
    assert result.worker_id == "test-worker"
    assert len(result.steps) == 3
    assert result.elapsed_seconds > 0

    await manager.publish_feedback(result, score=0.9, text="Echoed correctly")


async def test_full_loop_with_prm(bus: EventBus):
    """Full loop: manager -> echo worker -> PRM evaluator -> training rollout."""
    mock_scorer = AsyncMock()
    mock_scorer.score_steps.return_value = [0.9, 0.8, 0.7]

    manager = Manager("test-manager", bus)
    worker = EchoWorker("test-worker", bus)
    evaluator = PRMEvaluator(bus, mock_scorer)

    await manager.start()
    await worker.start()
    await evaluator.start(["coding"])

    # Capture rollout
    rollout_received = asyncio.Event()
    captured_rollout: list[TrainingRolloutEvent] = []

    async def _capture(rollout: TrainingRolloutEvent) -> None:
        captured_rollout.append(rollout)
        rollout_received.set()

    await bus.subscribe(TRAINING_ROLLOUTS, TrainingRolloutEvent, _capture)

    task = await manager.assign_task("coding", "Write a fibonacci function")
    result = await manager.wait_for_result(task.task_id, timeout=10.0)
    assert result.status == TaskStatus.SUCCESS

    # Wait for PRM to process and publish rollout
    await asyncio.wait_for(rollout_received.wait(), timeout=5.0)

    assert len(captured_rollout) == 1
    rollout = captured_rollout[0]
    assert rollout.task_id == task.task_id
    assert rollout.prompt == "Write a fibonacci function"
    assert rollout.step_scores == [0.9, 0.8, 0.7]
    assert rollout.outcome_score == 0.7
