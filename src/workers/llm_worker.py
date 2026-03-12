from __future__ import annotations

import logging
import re

import ollama

from src.events.types import ResultEvent, TaskEvent, TaskStatus
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful assistant that solves tasks step by step.

Format your response as numbered steps wrapped in <step> tags, followed by a final answer in an <answer> tag.

Example:
<step>1. Understand the problem requirements.</step>
<step>2. Implement the solution.</step>
<answer>Here is the final answer.</answer>

Always use this format. Think carefully through each step."""

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _parse_steps(text: str) -> tuple[list[str], str]:
    """Extract steps and answer from LLM response.

    Returns (steps, answer). Falls back to entire response as one step if
    the model doesn't follow the format.
    """
    steps = [m.strip() for m in _STEP_RE.findall(text)]
    answer_match = _ANSWER_RE.search(text)
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    if not steps:
        steps = [text.strip()]
    return steps, answer


class LLMWorker(BaseWorker):
    def __init__(
        self,
        worker_id: str,
        bus: object,
        task_types: list[str] | None = None,
        *,
        model: str = "qwen2.5:1.5b",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        super().__init__(worker_id, bus, task_types or ["coding"])
        self.model = model
        self._client = ollama.AsyncClient(host=ollama_host)

    async def process(self, task: TaskEvent) -> ResultEvent:
        logger.info("LLMWorker %s calling model %s", self.worker_id, self.model)
        response = await self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
        )
        raw_text = response["message"]["content"]
        steps, answer = _parse_steps(raw_text)
        return ResultEvent(
            task_id=task.task_id,
            worker_id=self.worker_id,
            result=answer,
            status=TaskStatus.SUCCESS,
            steps=steps,
        )
