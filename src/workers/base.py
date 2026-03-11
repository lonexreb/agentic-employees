from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from src.events.bus import EventBus
from src.events.topics import result_topic, task_topic
from src.events.types import ResultEvent, TaskEvent, TaskStatus

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    def __init__(self, worker_id: str, bus: EventBus, task_types: list[str]) -> None:
        self.worker_id = worker_id
        self.bus = bus
        self.task_types = task_types

    async def start(self) -> None:
        for tt in self.task_types:
            await self.bus.subscribe(task_topic(tt), TaskEvent, self._handle_task)
        logger.info("Worker %s started, listening for %s", self.worker_id, self.task_types)

    async def _handle_task(self, task: TaskEvent) -> None:
        logger.info("Worker %s received task %s", self.worker_id, task.task_id)
        t0 = time.monotonic()
        try:
            result = await self.process(task)
            result.elapsed_seconds = time.monotonic() - t0
        except Exception as exc:
            logger.exception("Worker %s failed task %s", self.worker_id, task.task_id)
            result = ResultEvent(
                task_id=task.task_id,
                worker_id=self.worker_id,
                result=str(exc),
                status=TaskStatus.FAILED,
                elapsed_seconds=time.monotonic() - t0,
            )
        await self.bus.publish(result_topic(task.task_type), result)

    @abstractmethod
    async def process(self, task: TaskEvent) -> ResultEvent: ...
