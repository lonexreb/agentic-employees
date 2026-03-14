# Submit Feedback Skill

Submits scored feedback for a completed task via the Bridge API.

## Usage

Call this skill after evaluating a worker's result to submit your feedback.

## Execution

```bash
curl -s -X POST http://bridge:8100/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "{{task_id}}",
    "manager_id": "{{manager_id}}",
    "worker_id": "{{worker_id}}",
    "score": {{score}},
    "textual_feedback": "{{feedback_text}}"
  }'
```

## Parameters

- `task_id`: The task ID being evaluated
- `manager_id`: Your agent ID (e.g., "manager-01")
- `worker_id`: The worker who completed the task (e.g., "worker-01")
- `score`: Float from 0.0 (terrible) to 1.0 (perfect)
- `textual_feedback`: Constructive text explaining the score

## Response

```json
{"task_id": "uuid-here", "status": "published"}
```
