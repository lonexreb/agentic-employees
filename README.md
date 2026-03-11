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
└──┬───────┬───────┬───────┬───────┘
   │       │       │       │
 Manager  Wkr A  Wkr B  RL Trainer
 Agent    Agent  Agent   (OpenRLHF)
```

**Manager** assigns tasks → **Workers** execute → **Manager** reviews & scores → **RL Trainer** improves workers from feedback → repeat.

## Getting Started

```bash
# Prerequisites: Python 3.10+, NATS server
pip install -e ".[dev]"

# Run type tests (no NATS needed)
pytest tests/events/test_types.py -v

# Run full integration tests (requires nats-server running)
nats-server &
pytest tests/ -v

# Run demo loop
python -m src
```

## Documentation

- **[PLAN.md](./PLAN.md)** — Full technical research & architecture bible (papers, analysis, decisions)
- **[CLAUDE.md](./CLAUDE.md)** — Project conventions for Claude Code
- **[LEARNING.md](./LEARNING.md)** — Mistake/lesson tracking log
