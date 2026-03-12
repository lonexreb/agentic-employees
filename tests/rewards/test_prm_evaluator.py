"""Tests for PRMEvaluator — mocked scorer, no NATS/Ollama required."""

from unittest.mock import AsyncMock

import pytest

from src.events.types import ResultEvent, TaskStatus, TrainingRolloutEvent
from src.rewards.prm_evaluator import PRMEvaluator


@pytest.fixture
def mock_scorer():
    scorer = AsyncMock()
    scorer.score_steps.return_value = [0.9, 0.8, 0.7]
    return scorer


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    return bus


@pytest.fixture
def evaluator(mock_bus, mock_scorer):
    return PRMEvaluator(mock_bus, mock_scorer)


def _make_result(status=TaskStatus.SUCCESS, steps=None):
    return ResultEvent(
        task_id="t-1",
        worker_id="w-1",
        prompt="solve x+1=2",
        result="x=1",
        status=status,
        steps=["parse", "solve", "verify"] if steps is None else steps,
    )


async def test_handle_result_publishes_rollout(evaluator, mock_bus, mock_scorer):
    result = _make_result()
    await evaluator._handle_result(result)

    mock_scorer.score_steps.assert_awaited_once_with("solve x+1=2", result.steps)
    mock_bus.publish.assert_awaited_once()
    topic, rollout = mock_bus.publish.call_args.args
    assert topic == "training.rollouts"
    assert isinstance(rollout, TrainingRolloutEvent)
    assert rollout.step_scores == [0.9, 0.8, 0.7]
    assert rollout.outcome_score == 0.7  # last step score


async def test_skips_failed_result(evaluator, mock_scorer):
    result = _make_result(status=TaskStatus.FAILED)
    await evaluator._handle_result(result)
    mock_scorer.score_steps.assert_not_awaited()


async def test_skips_empty_steps(evaluator, mock_scorer):
    result = _make_result(steps=[])
    await evaluator._handle_result(result)
    mock_scorer.score_steps.assert_not_awaited()
