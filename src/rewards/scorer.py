from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

from src.inference.client import InferenceClient
from src.rewards.prompts import STEP_JUDGE_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_SCORE = 0.5


@runtime_checkable
class StepScorer(Protocol):
    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]: ...


class LLMJudgeScorer:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:1.5b",
        client: InferenceClient,
    ) -> None:
        self.model = model
        self._client = client

    async def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        scores: list[float] = []
        for i, step in enumerate(steps, 1):
            formatted_steps = "\n".join(
                f"  {j}. {s}" for j, s in enumerate(steps[:i], 1)
            )
            judge_prompt = STEP_JUDGE_PROMPT.format(
                step_num=i,
                task_description=prompt,
                formatted_steps=formatted_steps,
                current_step=step,
            )
            score = await self._judge_single_step(judge_prompt)
            scores.append(score)
        return scores

    async def _judge_single_step(self, judge_prompt: str) -> float:
        try:
            raw = await self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                json_mode=True,
            )
            parsed = json.loads(raw)
            progress = float(parsed.get("progress", DEFAULT_FALLBACK_SCORE))
            correctness = float(parsed.get("correctness", DEFAULT_FALLBACK_SCORE))
            return (progress + correctness) / 2.0
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse judge response: %s", exc)
            return DEFAULT_FALLBACK_SCORE
