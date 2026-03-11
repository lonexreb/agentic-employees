from __future__ import annotations

TASKS_CODING = "tasks.coding"
RESULTS_CODING = "results.coding"
FEEDBACK_SCORED = "feedback.scored"
TRAINING_ROLLOUTS = "training.rollouts"
MODEL_UPDATES = "model.updates"


def task_topic(task_type: str) -> str:
    return f"tasks.{task_type}"


def result_topic(task_type: str) -> str:
    return f"results.{task_type}"
