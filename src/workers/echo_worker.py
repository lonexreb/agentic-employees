from __future__ import annotations

from src.events.bus import EventBus
from src.events.types import ResultEvent, TaskEvent, TaskStatus
from src.workers.base import BaseWorker


class EchoWorker(BaseWorker):
    def __init__(self, worker_id: str, bus: EventBus, task_types: list[str] | None = None) -> None:
        super().__init__(worker_id, bus, task_types or ["coding"])

    async def process(self, task: TaskEvent) -> ResultEvent:
        steps = [
            "Received prompt",
            "Processing (echo)",
            "Returning result",
        ]
        return ResultEvent(
            task_id=task.task_id,
            worker_id=self.worker_id,
            result=task.prompt,
            status=TaskStatus.SUCCESS,
            steps=steps,
        )
