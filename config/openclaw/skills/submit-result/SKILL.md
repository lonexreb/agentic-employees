# Submit Result Skill

Submits a completed task result via the Bridge API.

## Usage

Call this skill after solving a task to submit your result.

## Execution

```bash
curl -s -X POST http://bridge:8100/tasks/result \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "{{task_id}}",
    "worker_id": "{{worker_id}}",
    "prompt": "{{original_prompt}}",
    "result": "{{final_answer}}",
    "status": "success",
    "steps": {{steps_json_array}},
    "task_type": "{{task_type}}"
  }'
```

## Parameters

- `task_id`: The task ID from the original assignment
- `worker_id`: Your agent ID (e.g., "worker-01")
- `prompt`: The original task prompt
- `result`: Your final answer/solution
- `status`: "success" or "failed"
- `steps`: JSON array of step strings (e.g., `["Step 1: ...", "Step 2: ..."]`)
- `task_type`: Type of task (e.g., "coding") — used for NATS topic routing

## Response

```json
{"task_id": "uuid-here", "topic": "results.coding", "status": "published"}
```
