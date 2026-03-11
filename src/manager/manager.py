from __future__ import annotations

import asyncio
import logging

from src.events.bus import EventBus
from src.events.topics import FEEDBACK_SCORED, result_topic
from src.events.types import FeedbackEvent, ResultEvent, TaskEvent

logger = logging.getLogger(__name__)


class Manager:
    def __init__(self, manager_id: str, bus: EventBus, task_types: list[str] | None = None) -> None:
        self.manager_id = manager_id
        self.bus = bus
        self.task_types = task_types or ["coding"]
        self._pending: dict[str, asyncio.Event] = {}
        self._results: dict[str, ResultEvent] = {}

    async def start(self) -> None:
        for tt in self.task_types:
            await self.bus.subscribe(result_topic(tt), ResultEvent, self._handle_result)
        logger.info("Manager %s started, listening for results on %s", self.manager_id, self.task_types)

    async def assign_task(self, task_type: str, prompt: str, **metadata: object) -> TaskEvent:
        from src.events.topics import task_topic

        task = TaskEvent(
            manager_id=self.manager_id,
            task_type=task_type,
            prompt=prompt,
            metadata=metadata,
        )
        waiter = asyncio.Event()
        self._pending[task.task_id] = waiter
        await self.bus.publish(task_topic(task_type), task)
        logger.info("Manager %s assigned task %s", self.manager_id, task.task_id)
        return task

    async def wait_for_result(self, task_id: str, timeout: float = 30.0) -> ResultEvent:
        waiter = self._pending.get(task_id)
        if waiter is None:
            raise KeyError(f"No pending task {task_id}")
        await asyncio.wait_for(waiter.wait(), timeout=timeout)
        return self._results[task_id]

    async def _handle_result(self, result: ResultEvent) -> None:
        self._results[result.task_id] = result
        waiter = self._pending.pop(result.task_id, None)
        if waiter:
            waiter.set()
        logger.info("Manager received result for task %s: %s", result.task_id, result.status)

    async def publish_feedback(self, result: ResultEvent, score: float, text: str = "") -> None:
        feedback = FeedbackEvent(
            task_id=result.task_id,
            manager_id=self.manager_id,
            worker_id=result.worker_id,
            score=score,
            textual_feedback=text,
        )
        await self.bus.publish(FEEDBACK_SCORED, feedback)
        logger.info("Manager published feedback for task %s: score=%.2f", result.task_id, score)
