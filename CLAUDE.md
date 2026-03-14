# agentic-employees

Managers managing AI agents with appraisal-style feedback loops. Agents learn continuously from manager feedback using RLHF/GRPO — like performance reviews that actually improve performance.

## Architecture

Event-Driven Architecture (EDA):
- **Control Plane**: OpenClaw (identity, memory, channels, UI)
- **Event Broker**: NATS (pub/sub for agent coordination)
- **Training Plane**: Standalone GRPO (lightweight) + OpenRLHF (production GPU)
- **Inference**: InferenceClient protocol — Ollama (dev) or OpenAI-compatible (vLLM/Semantic Router)

## Language

**Python only.** Requires Python >= 3.10. All components — RL training, agent logic, NATS clients, PRM evaluator — are Python.

## Current Dependencies (pyproject.toml)

- [nats-py](https://github.com/nats-io/nats.py) — NATS client for event bus
- [pydantic](https://docs.pydantic.dev/) — event type serialization (v2)
- [ollama](https://github.com/ollama/ollama-python) — LLM inference via Ollama (async client)

Dev: pytest, pytest-asyncio, pytest-aiohttp, ruff

Optional extras:
- `pip install -e ".[training]"` — torch, transformers, peft
- `pip install -e ".[inference]"` — openai, httpx (for OpenAI-compatible servers / vLLM)
- `pip install -e ".[bridge]"` — aiohttp (Bridge HTTP API for OpenClaw integration)
- `pip install -e ".[vllm]"` — vLLM (GPU inference server)
- `pip install -e ".[openrlhf]"` — OpenRLHF (production GRPO training)

## Directory Structure

```
src/
├── __main__.py    # Demo entry point (python -m src) — uses InferenceClient factory
├── config.py      # Frozen dataclass with env var defaults (NATS, inference, training)
├── manager/
│   └── manager.py # Manager agent (assign tasks, wait for results, publish feedback)
├── workers/
│   ├── base.py         # BaseWorker ABC (subscribe, handle, process, model update hook)
│   ├── echo_worker.py  # EchoWorker — echoes prompt back with PRM steps (testing)
│   └── llm_worker.py   # LLMWorker — LLM inference via InferenceClient with step parsing
├── events/
│   ├── types.py   # Pydantic v2 event models (TaskEvent, ResultEvent, FeedbackEvent, etc.)
│   ├── topics.py  # Topic constants and helpers
│   └── bus.py     # EventBus wrapping nats-py (connect, publish, subscribe, drain)
├── inference/
│   ├── client.py       # InferenceClient protocol + OllamaInferenceClient + OpenAIInferenceClient
│   └── vllm_lora.py    # VLLMLoRAManager — dynamic LoRA hot-swap via vLLM admin endpoints
├── rewards/
│   ├── scorer.py        # StepScorer protocol + LLMJudgeScorer (LLM-as-judge PRM)
│   ├── prompts.py       # STEP_JUDGE_PROMPT template for step-level evaluation
│   └── prm_evaluator.py # PRMEvaluator — subscribes to results, scores steps, publishes rollouts
├── training/
│   ├── bridge.py              # RolloutBuffer + NATSTrainingBridge (batch rollouts for RL trainer)
│   ├── grpo.py                # GRPO math: compute_group_advantages, clipped_surrogate_loss, kl_penalty
│   ├── trainer.py             # Trainer protocol, TrainStepResult, MockTrainer, GRPOTrainer (LoRA)
│   ├── loop.py                # TrainingLoop orchestrator (bridge → trainer → ModelUpdateEvent)
│   └── openrlhf_launcher.py   # OpenRLHFLauncher — subprocess launcher for production GRPO training
├── bridge/
│   ├── __main__.py    # Entrypoint: python -m src.bridge
│   ├── service.py     # BridgeService — connects HTTP API to NATS event bus
│   └── http_api.py    # HTTP endpoints for OpenClaw agents (assign, result, feedback, status, health)
├── services/
│   ├── __main__.py    # Entrypoint: python -m src.services.training
│   └── training.py    # Standalone training service (PRM Evaluator + Training Loop)
config/
└── openclaw/
    ├── AGENTS.md              # Agent registry (manager-01 + worker-01)
    ├── manager/
    │   ├── SOUL.md            # Manager behavior: decompose, assign, evaluate, score
    │   └── IDENTITY.md        # Manager identity (name, role, bio)
    ├── worker/
    │   ├── SOUL.md            # Worker behavior: step-by-step solving with <step>/<answer>
    │   └── IDENTITY.md        # Worker identity
    └── skills/
        ├── assign-task/SKILL.md     # exec: curl POST bridge:8100/tasks/assign
        ├── submit-result/SKILL.md   # exec: curl POST bridge:8100/tasks/result
        └── submit-feedback/SKILL.md # exec: curl POST bridge:8100/feedback
tests/
├── bridge/
│   ├── test_http_api.py # Bridge HTTP endpoint unit tests (mocked NATS)
│   └── test_service.py  # Bridge integration test (requires NATS)
├── events/
│   ├── test_types.py  # Serialization roundtrip tests (standalone)
│   └── test_bus.py    # EventBus pub/sub tests (requires NATS)
├── inference/
│   ├── test_client.py       # InferenceClient protocol + adapter tests (mocked)
│   └── test_vllm_lora.py    # VLLMLoRAManager tests (mocked httpx)
├── rewards/
│   ├── test_scorer.py        # LLMJudgeScorer tests (mocked InferenceClient)
│   └── test_prm_evaluator.py # PRMEvaluator tests (mocked scorer)
├── training/
│   ├── test_bridge.py        # RolloutBuffer unit tests
│   ├── test_grpo.py          # GRPO advantage math + torch loss/KL tests
│   └── test_trainer.py       # MockTrainer + GRPOTrainer protocol/integration tests
├── workers/
│   └── test_model_reload.py  # Worker model update subscription + reload tests
└── test_integration.py # Full manager→worker→PRM→rollout loop (requires NATS)
docker-compose.yml     # 5 services: NATS, Ollama, OpenClaw, Bridge, Training
Dockerfile             # Python 3.10 base for training service
Dockerfile.bridge      # Python 3.10 base for bridge service
scripts/
└── demo.sh            # One-command Docker Compose demo startup
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
- Standalone (no NATS/Ollama): `tests/events/test_types.py`, `tests/rewards/`, `tests/training/`, `tests/inference/`, `tests/workers/`, `tests/bridge/test_http_api.py`
- Requires NATS: `tests/events/test_bus.py`, `tests/test_integration.py`, `tests/bridge/test_service.py`
- Mock strategy: scorer/evaluator tests mock InferenceClient; bridge tests mock EventBus; integration tests use EchoWorker + mock scorer
- Run standalone: `pytest tests/ -v` (72 pass, 5 skip without NATS)
- Run standalone (skip slow torch tests): `pytest tests/training/ -v -k "not slow"`
- Run all: `pytest tests/ -v` (with NATS running — 77 pass)

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| INFERENCE_BACKEND | ollama | `"ollama"` or `"openai"` (for vLLM/Semantic Router) |
| INFERENCE_BASE_URL | (auto) | Base URL for inference server |
| INFERENCE_API_KEY | (empty) | API key for inference server |
| VLLM_LORA_NAME | default | LoRA adapter name for vLLM |
| TRAINER_BACKEND | standalone | `"standalone"` (GRPOTrainer) or `"openrlhf"` (OpenRLHFLauncher) |
| BRIDGE_PORT | 8100 | Bridge HTTP API port |
| OPENCLAW_GATEWAY_URL | ws://localhost:18789 | OpenClaw gateway WebSocket URL |

## Commit Format

Conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding/updating tests
- `chore:` maintenance tasks

## Current Phase

**Phase 6 complete** — OpenClaw full agent runtime integration with Docker Compose demo.

- Bridge Service (`src/bridge/`) — HTTP API ←→ NATS pub/sub translation for OpenClaw agents
- OpenClaw agent configs (`config/openclaw/`) — SOUL.md, IDENTITY.md, SKILL.md for manager + worker
- Standalone training service (`src/services/training.py`) — PRM Evaluator + Training Loop as Docker container
- Docker Compose (`docker-compose.yml`) — 5 services: NATS, Ollama, OpenClaw, Bridge, Training
- Demo script (`scripts/demo.sh`) — one-command startup with health checks
- Bridge tests (`tests/bridge/`) — 10 unit tests + 1 integration test
- Zero changes to existing `src/events/`, `src/rewards/`, `src/training/`, `src/workers/`, `src/manager/`

Previous phases:
- Phase 5: InferenceClient protocol, weight hot-swap, OpenRLHF, Semantic Router readiness
- Phase 4: Standalone GRPO trainer, LoRA fine-tuning, TrainingLoop
- Phase 3: LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge
- Phase 2: Event loop, Manager/Worker agents, EchoWorker
- Phase 1: Scaffolding, docs, git

**Phase 7 (next):** Trained PRM model, DAPO graduation, multi-model routing in Semantic Router, HaluGate scorer.

## Key Documents

- [PLAN.md](./PLAN.md) — Full technical research & architecture bible (papers, analysis, decisions)
- [LEARNING.md](./LEARNING.md) — Mistake/lesson tracking for autonomous decisions
- [RESEARCH-EXPERIMENT.md](./RESEARCH-EXPERIMENT.md) — Phase experiment records and findings
