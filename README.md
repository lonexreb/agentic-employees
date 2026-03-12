# agentic-employees

A system where **managers manage agentic employees** that learn from feedback — like performance appraisals that actually improve performance. Built on reinforcement learning from human feedback (RLHF), agents continuously improve through manager evaluations using GRPO/DAPO algorithms.

## Architecture

```
┌──────────────────────────────────┐
│   OpenClaw (Control Plane)       │
│   Identity · Memory · Channels   │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│   NATS Event Broker              │
│   tasks.* · results.* · feedback.*│
│   training.rollouts · model.updates│
└──┬───────┬───────┬───────┬───────┘
   │       │       │       │
 Manager  Wkr A  PRM      Training
 Agent    Agent  Evaluator  Bridge
                  (LLM-     (Rollout
                  as-Judge)  Buffer)
```

**Manager** assigns tasks → **Workers** execute (LLM via Ollama) → **PRM Evaluator** scores each step → **Training Bridge** batches rollouts → **RL Trainer** improves workers from feedback → repeat.

## Current Status

- **Phase 1** (DONE): Scaffolding, docs, git, GitHub
- **Phase 2** (DONE): Event loop — manager publishes task, EchoWorker processes, manager scores
- **Phase 3** (DONE): LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge (rollout buffer)
- **Phase 4** (NEXT): Connect RL trainer (OpenRLHF GRPO), weight hot-swap

## Getting Started

```bash
# Prerequisites: Python 3.10+, NATS server
pip install -e ".[dev]"

# Run standalone tests (no NATS or Ollama needed)
pytest tests/events/test_types.py tests/rewards/ tests/training/ -v

# Run full integration tests (requires nats-server running)
nats-server &
pytest tests/ -v

# Run demo loop (requires NATS; uses Ollama if available, else EchoWorker)
python -m src

# For full LLM demo (requires Ollama + model)
ollama pull qwen2.5:1.5b
python -m src
```

## Configuration

Environment variables (all optional with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS broker URL |
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model for LLM worker |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MANAGER_ID` | `manager-01` | Manager agent ID |
| `WORKER_ID` | `worker-01` | Worker agent ID |
| `TASK_TIMEOUT_SECONDS` | `30` | Task completion timeout |

## Documentation

- **[PLAN.md](./PLAN.md)** — Full technical research & architecture bible (papers, analysis, decisions)
- **[CLAUDE.md](./CLAUDE.md)** — Project conventions for Claude Code
- **[LEARNING.md](./LEARNING.md)** — Mistake/lesson tracking log
