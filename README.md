# agentic-employees

A system where **managers manage agentic employees** that learn from feedback — like performance appraisals that actually improve performance. Built on reinforcement learning from human feedback (RLHF), agents continuously improve through manager evaluations using GRPO/DAPO algorithms.

## Architecture

```
User → OpenClaw Web UI (:3000)
           │
           ▼
┌──────────────────────┐    ┌──────────┐
│  OpenClaw Gateway    │    │   NATS   │
│  (:18789)            │    │  (:4222) │
│  Manager + Worker    │    │          │
│  agents              │    │          │
└──────────┬───────────┘    └────┬─────┘
           │ exec/curl           │
     ┌─────▼─────────────────────▼──────┐
     │       Bridge Service (:8100)      │
     │  HTTP API ←→ NATS pub/sub         │
     │  (Python, aiohttp + nats-py)      │
     └──────────────────┬───────────────┘
                        │ NATS (unchanged)
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   PRM Evaluator  Training Loop   Ollama
   (unchanged)    (unchanged)     (:11434)
```

**OpenClaw** runs Manager + Worker agents (identity, memory, web UI) → agents call the **Bridge Service** via HTTP → Bridge translates to **NATS** pub/sub → **PRM Evaluator** scores steps → **Training Loop** runs GRPO → model improves → repeat.

## Current Status

- **Phase 1** (DONE): Scaffolding, docs, git, GitHub
- **Phase 2** (DONE): Event loop — manager publishes task, EchoWorker processes, manager scores
- **Phase 3** (DONE): LLM workers (Ollama), PRM scoring (LLM-as-judge), training bridge (rollout buffer)
- **Phase 4** (DONE): Standalone GRPO trainer, LoRA fine-tuning, TrainingLoop orchestrator
- **Phase 5** (DONE): Inference abstraction, weight hot-swap, OpenRLHF integration
- **Phase 6** (DONE): OpenClaw integration — Bridge Service, Docker Compose, agent configs

## Quick Start (Docker Compose)

```bash
# One command to start everything
./scripts/demo.sh

# Or manually:
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b

# Open http://localhost:3000 — message the Manager agent
```

## Getting Started (Local Development)

```bash
# Prerequisites: Python 3.10+, NATS server
pip install -e ".[dev]"

# Run standalone tests (no NATS or Ollama needed — 72 tests)
pytest tests/ -v

# Run full integration tests (requires nats-server running)
nats-server &
pytest tests/ -v

# Run demo loop (requires NATS; uses Ollama if available, else EchoWorker)
python -m src

# Run Bridge service standalone
pip install -e ".[bridge]"
python -m src.bridge

# Run Training service standalone
pip install -e ".[training]"
python -m src.services.training
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
| `INFERENCE_BACKEND` | `ollama` | `"ollama"` or `"openai"` (for vLLM/Semantic Router) |
| `INFERENCE_BASE_URL` | (auto) | Base URL for inference server |
| `INFERENCE_API_KEY` | (empty) | API key for inference server |
| `TRAINER_BACKEND` | `standalone` | `"standalone"` (GRPOTrainer) or `"openrlhf"` (OpenRLHFLauncher) |
| `BRIDGE_PORT` | `8100` | Bridge HTTP API port |
| `OPENCLAW_GATEWAY_URL` | `ws://localhost:18789` | OpenClaw gateway WebSocket URL |

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| NATS | 4222, 8222 | Event broker (pub/sub) |
| Ollama | 11434 | LLM inference |
| OpenClaw | 3000, 18789 | Agent runtime + Web UI |
| Bridge | 8100 | HTTP ←→ NATS translation |
| Training | — | PRM Evaluator + Training Loop |

## Documentation

- **[PLAN.md](./PLAN.md)** — Full technical research & architecture bible (papers, analysis, decisions)
- **[CLAUDE.md](./CLAUDE.md)** — Project conventions for Claude Code
- **[LEARNING.md](./LEARNING.md)** — Mistake/lesson tracking log
- **[RESEARCH-EXPERIMENT.md](./RESEARCH-EXPERIMENT.md)** — Phase experiment records and findings
