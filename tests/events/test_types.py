"""Tests for event type serialization — no NATS required."""

from src.events.types import (
    FeedbackEvent,
    ModelUpdateEvent,
    ResultEvent,
    TaskEvent,
    TaskStatus,
    TrainingRolloutEvent,
)


def test_task_event_roundtrip():
    event = TaskEvent(manager_id="mgr-1", task_type="coding", prompt="Write hello world")
    data = event.model_dump_json()
    restored = TaskEvent.model_validate_json(data)
    assert restored.task_id == event.task_id
    assert restored.prompt == "Write hello world"
    assert restored.created_at == event.created_at


def test_result_event_roundtrip():
    event = ResultEvent(
        task_id="abc-123",
        worker_id="w-1",
        result="done",
        status=TaskStatus.SUCCESS,
        steps=["step1", "step2"],
        elapsed_seconds=1.5,
    )
    data = event.model_dump_json()
    restored = ResultEvent.model_validate_json(data)
    assert restored.status == TaskStatus.SUCCESS
    assert restored.steps == ["step1", "step2"]
    assert restored.elapsed_seconds == 1.5


def test_feedback_event_roundtrip():
    event = FeedbackEvent(
        task_id="abc-123",
        manager_id="mgr-1",
        worker_id="w-1",
        score=0.85,
        textual_feedback="Good work",
    )
    data = event.model_dump_json()
    restored = FeedbackEvent.model_validate_json(data)
    assert restored.score == 0.85
    assert restored.textual_feedback == "Good work"


def test_training_rollout_event_roundtrip():
    event = TrainingRolloutEvent(
        task_id="abc-123",
        worker_id="w-1",
        prompt="Write code",
        response="print('hello')",
        steps=["parse", "generate"],
        step_scores=[0.9, 0.8],
        outcome_score=0.85,
    )
    data = event.model_dump_json()
    restored = TrainingRolloutEvent.model_validate_json(data)
    assert restored.step_scores == [0.9, 0.8]
    assert restored.outcome_score == 0.85


def test_model_update_event_roundtrip():
    event = ModelUpdateEvent(
        model_version="v0.1",
        checkpoint_path="/tmp/ckpt",
        metrics={"loss": 0.5},
    )
    data = event.model_dump_json()
    restored = ModelUpdateEvent.model_validate_json(data)
    assert restored.model_version == "v0.1"
    assert restored.metrics == {"loss": 0.5}


def test_task_event_auto_generates_id():
    e1 = TaskEvent(manager_id="m", task_type="t", prompt="p")
    e2 = TaskEvent(manager_id="m", task_type="t", prompt="p")
    assert e1.task_id != e2.task_id


def test_feedback_score_bounds():
    import pytest

    with pytest.raises(Exception):
        FeedbackEvent(task_id="x", manager_id="m", worker_id="w", score=1.5)
    with pytest.raises(Exception):
        FeedbackEvent(task_id="x", manager_id="m", worker_id="w", score=-0.1)
