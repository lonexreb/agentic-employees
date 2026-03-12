"""Tests for LLMJudgeScorer — mocked Ollama, no network required."""

from unittest.mock import AsyncMock

import pytest

from src.rewards.scorer import DEFAULT_FALLBACK_SCORE, LLMJudgeScorer


def _make_judge_response(progress: float, correctness: float) -> dict:
    import json

    return {
        "message": {
            "content": json.dumps(
                {
                    "progress": progress,
                    "correctness": correctness,
                    "reasoning": "test",
                }
            )
        }
    }


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def scorer(mock_client):
    return LLMJudgeScorer(model="test-model", client=mock_client)


async def test_score_steps_calls_per_step(scorer, mock_client):
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    steps = ["step one", "step two", "step three"]
    scores = await scorer.score_steps("solve x+1=2", steps)
    assert len(scores) == 3
    assert mock_client.chat.call_count == 3


async def test_score_computation(scorer, mock_client):
    mock_client.chat.return_value = _make_judge_response(0.8, 0.6)
    scores = await scorer.score_steps("task", ["only step"])
    assert scores == [pytest.approx(0.7)]  # (0.8 + 0.6) / 2


async def test_json_parse_error_fallback(scorer, mock_client):
    mock_client.chat.return_value = {"message": {"content": "not json at all"}}
    scores = await scorer.score_steps("task", ["step"])
    assert scores == [DEFAULT_FALLBACK_SCORE]


async def test_missing_keys_fallback(scorer, mock_client):
    mock_client.chat.return_value = {"message": {"content": "{}"}}
    scores = await scorer.score_steps("task", ["step"])
    # Both default to 0.5, average = 0.5
    assert scores == [DEFAULT_FALLBACK_SCORE]
