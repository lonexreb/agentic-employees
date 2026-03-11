# agentic-employees

Managers managing AI agents with appraisal-style feedback loops. Agents learn continuously from manager feedback using RLHF/GRPO — like performance reviews that actually improve performance.

## Architecture

Event-Driven Architecture (EDA):
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Event Broker**: NATS (pub/sub for agent coordination)
- **Training Plane**: OpenRLHF + OpenClaw-RL (GRPO/DAPO/AgentPRM)
- **Inference**: vLLM + Ray for distributed serving

## Language

**Python only.** Requires Python >= 3.10. All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Current Dependencies (pyproject.toml)

- [nats-py](https://github.com/nats-io/nats.py) — NATS client for event bus
- [pydantic](https://docs.pydantic.dev/) — event type serialization (v2)

Dev: pytest, pytest-asyncio, ruff

## Planned Dependencies (future phases)

- [OpenClaw](https://github.com/openclaw/openclaw) — control plane (identity, memory, channels)
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) — continuous learning from feedback (async GRPO)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — heavy RL training (PPO/GRPO/DAPO/REINFORCE++)
- [PicoClaw](https://github.com/sipeed/picoclaw) — future edge deployment (<10MB RAM)
- [M³HF](https://github.com/cooperativex/M3HF) — multi-phase feedback from mixed-quality humans
- Ray, vLLM, DeepSpeed — distributed compute and inference

## Directory Structure

```
src/
├── __main__.py    # Demo entry point (python -m src)
├── config.py      # Frozen dataclass with env var defaults
├── manager/
│   └── manager.py # Manager agent (assign tasks, wait for results, publish feedback)
├── workers/
│   ├── base.py         # BaseWorker ABC (subscribe, handle, process pattern)
│   └── echo_worker.py  # EchoWorker — echoes prompt back with PRM steps
├── events/
│   ├── types.py   # Pydantic v2 event models (TaskEvent, ResultEvent, FeedbackEvent, etc.)
│   ├── topics.py  # Topic constants and helpers
│   └── bus.py     # EventBus wrapping nats-py (connect, publish, subscribe, drain)
├── training/      # RL training loops (future — GRPO, DAPO, OpenRLHF integration)
└── rewards/       # PRM evaluator, reward functions, scoring (future)
config/
└── openclaw/      # SOUL.md, IDENTITY.md templates per agent (future)
tests/
├── events/
│   ├── test_types.py  # Serialization roundtrip tests (standalone)
│   └── test_bus.py    # EventBus pub/sub tests (requires NATS)
└── test_integration.py # Full manager→worker→feedback loop (requires NATS)
docs/
└── architecture/  # Diagrams and ADRs
```

## Code Style

- PEP 8
- Type hints on all function signatures
- Docstrings on public APIs only (not internal helpers)
- No unnecessary abstractions — keep it simple

## Testing

- Framework: pytest + pytest-asyncio (asyncio_mode = "auto")
- `tests/` mirrors `src/` structure (e.g., `tests/events/` tests `src/events/`)
- `tests/events/test_types.py` — standalone (no NATS)
- `tests/events/test_bus.py` and `tests/test_integration.py` — require `nats-server` running
- Run standalone: `pytest tests/events/test_types.py -v`
- Run all: `pytest tests/ -v` (with NATS running)

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Key Documents

- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
