from __future__ import annotations

STEP_JUDGE_PROMPT = """\
You are a step-level evaluator for an AI assistant solving a task.

Task: {task_description}

The assistant's reasoning so far:
{formatted_steps}

Evaluate step {step_num}: "{current_step}"

Score this step on two dimensions:
- **progress**: How much does this step advance toward solving the task? (0.0 = no progress, 1.0 = completes the task)
- **correctness**: Is this step logically sound and factually correct? (0.0 = completely wrong, 1.0 = perfectly correct)

Respond with ONLY a JSON object, no other text:
{{"progress": <float 0-1>, "correctness": <float 0-1>, "reasoning": "<brief explanation>"}}"""
