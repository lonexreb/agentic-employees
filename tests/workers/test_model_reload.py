"""Tests for worker model update subscription and reload behavior."""

from unittest.mock import AsyncMock

import pytest

from src.events.types import ModelUpdateEvent, ResultEvent, TaskEvent, TaskStatus
from src.workers.base import BaseWorker
from src.workers.echo_worker import EchoWorker


class FakeReloadWorker(BaseWorker):
    """Test worker that tracks reload calls."""

    def __init__(self, worker_id: str, bus: object) -> None:
        super().__init__(worker_id, bus, ["coding"])
        self.reload_calls: list[ModelUpdateEvent] = []

    async def process(self, task: TaskEvent) -> ResultEvent:
        return ResultEvent(
            task_id=task.task_id,
            worker_id=self.worker_id,
            result="fake",
            status=TaskStatus.SUCCESS,
        )

    async def reload_model(self, event: ModelUpdateEvent) -> None:
        self.reload_calls.append(event)


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    return bus


async def test_start_subscribes_to_model_updates(mock_bus):
    worker = EchoWorker("w-1", mock_bus)
    await worker.start()

    # Should subscribe to task topic AND model.updates
    topics = [call.args[0] for call in mock_bus.subscribe.call_args_list]
    assert "tasks.coding" in topics
    assert "model.updates" in topics


async def test_reload_model_called_on_update(mock_bus):
    worker = FakeReloadWorker("w-1", mock_bus)
    event = ModelUpdateEvent(
        model_version="v0005",
        checkpoint_path="/tmp/ckpts/v0005",
        metrics={"loss": 0.3},
    )

    await worker._handle_model_update(event)

    assert len(worker.reload_calls) == 1
    assert worker.reload_calls[0].model_version == "v0005"


async def test_base_worker_reload_is_noop(mock_bus):
    worker = EchoWorker("w-1", mock_bus)
    event = ModelUpdateEvent(
        model_version="v0005",
        checkpoint_path="/tmp/ckpts/v0005",
    )
    # Should not raise
    await worker.reload_model(event)


async def test_llm_worker_reload_updates_model(mock_bus):
    from src.workers.llm_worker import LLMWorker

    # Use a mock InferenceClient
    mock_client = AsyncMock()
    worker = LLMWorker("w-1", mock_bus, model="original-model", client=mock_client)

    assert worker.model == "original-model"
    assert worker._active_version is None

    event = ModelUpdateEvent(
        model_version="grpo-lora-v0005",
        checkpoint_path="/tmp/ckpts/v0005",
    )
    await worker.reload_model(event)

    assert worker.model == "grpo-lora-v0005"
    assert worker._active_version == "grpo-lora-v0005"


async def test_model_version_set_in_result(mock_bus):
    worker = FakeReloadWorker("w-1", mock_bus)
    worker._active_version = "v0003"

    task = TaskEvent(
        manager_id="m-1",
        task_type="coding",
        prompt="test prompt",
    )

    await worker._handle_task(task)

    # The result published should have model_version set
    published_result = mock_bus.publish.call_args.args[1]
    assert published_result.model_version == "v0003"


async def test_model_version_none_by_default(mock_bus):
    worker = FakeReloadWorker("w-1", mock_bus)

    task = TaskEvent(
        manager_id="m-1",
        task_type="coding",
        prompt="test prompt",
    )

    await worker._handle_task(task)

    published_result = mock_bus.publish.call_args.args[1]
    assert published_result.model_version is None
