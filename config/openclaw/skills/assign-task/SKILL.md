# Assign Task Skill

Assigns a new task to a worker agent via the Bridge API.

## Usage

Call this skill when you need to assign a coding task to a worker.

## Execution

```bash
curl -s -X POST http://bridge:8100/tasks/assign \
  -H "Content-Type: application/json" \
  -d '{
    "manager_id": "{{manager_id}}",
    "task_type": "{{task_type}}",
    "prompt": "{{prompt}}"
  }'
```

## Parameters

- `manager_id`: Your agent ID (e.g., "manager-01")
- `task_type`: Type of task (e.g., "coding")
- `prompt`: Clear description of the task to solve

## Response

Returns JSON with `task_id` that you can use to poll for results:
```json
{"task_id": "uuid-here", "topic": "tasks.coding", "status": "published"}
```

After assigning, poll `GET http://bridge:8100/tasks/{task_id}/status` for the result.
