"""Tests for RolloutBuffer — no NATS required."""

from src.events.types import TrainingRolloutEvent
from src.training.bridge import RolloutBuffer


def _make_rollout(prompt: str = "task", task_id: str = "t-1") -> TrainingRolloutEvent:
    return TrainingRolloutEvent(
        task_id=task_id,
        worker_id="w-1",
        prompt=prompt,
        response="answer",
        steps=["s1"],
        step_scores=[0.8],
        outcome_score=0.8,
    )


def test_add_and_has_batch():
    buf = RolloutBuffer(batch_size=3, group_size=1)
    assert not buf.has_batch()
    buf.add(_make_rollout(prompt="a"))
    buf.add(_make_rollout(prompt="b"))
    assert not buf.has_batch()
    buf.add(_make_rollout(prompt="c"))
    assert buf.has_batch()


def test_get_batch_returns_correct_size():
    buf = RolloutBuffer(batch_size=2, group_size=1)
    for i in range(5):
        buf.add(_make_rollout(prompt=f"p{i}"))
    batch = buf.get_batch()
    assert len(batch) == 2
    # Should still have 3 remaining
    assert buf.has_batch()
    batch2 = buf.get_batch()
    assert len(batch2) == 2


def test_group_size_delays_completion():
    buf = RolloutBuffer(batch_size=1, group_size=3)
    buf.add(_make_rollout(prompt="same"))
    assert not buf.has_batch()
    buf.add(_make_rollout(prompt="same"))
    assert not buf.has_batch()
    buf.add(_make_rollout(prompt="same"))
    assert buf.has_batch()


def test_different_prompts_group_independently():
    buf = RolloutBuffer(batch_size=2, group_size=2)
    buf.add(_make_rollout(prompt="a"))
    buf.add(_make_rollout(prompt="b"))
    assert not buf.has_batch()  # Neither group complete
    buf.add(_make_rollout(prompt="a"))
    # Group "a" complete (2 items moved to _complete), batch_size=2 → ready
    assert buf.has_batch()
