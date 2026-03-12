from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class TaskStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"


class TaskEvent(BaseModel):
    task_id: str = Field(default_factory=_new_id)
    manager_id: str
    task_type: str
    prompt: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ResultEvent(BaseModel):
    task_id: str
    worker_id: str
    prompt: str = ""
    result: str
    status: TaskStatus
    steps: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


class FeedbackEvent(BaseModel):
    task_id: str
    manager_id: str
    worker_id: str
    score: float = Field(ge=0.0, le=1.0)
    textual_feedback: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class TrainingRolloutEvent(BaseModel):
    task_id: str
    worker_id: str
    prompt: str
    response: str
    steps: list[str] = Field(default_factory=list)
    step_scores: list[float] = Field(default_factory=list)
    outcome_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class ModelUpdateEvent(BaseModel):
    model_version: str
    checkpoint_path: str
    metrics: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
