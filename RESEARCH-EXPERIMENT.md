# Research & Experiment Records

Living document tracking experiments, findings, and technical decisions across phases. Each phase records what was tried, what worked, what didn't, and quantitative results where available.

---

## Phase 1: Scaffolding

**Date:** 2026-03-09

**Scope:** Project setup, docs, git, GitHub.

**Outcome:** Established CLAUDE.md + PLAN.md + LEARNING.md documentation pattern. Python >=3.10, pyproject.toml with setuptools backend.

**Finding:** Dev machine runs Python 3.10.0 — initial pyproject.toml incorrectly set `>=3.11`. Caught during first install. Logged in LEARNING.md.

---

## Phase 2: Event Loop

**Date:** 2026-03-09

**Scope:** NATS event bus, Manager/Worker agents, EchoWorker, pub/sub task assignment.

**Outcome:** Manager publishes TaskEvent, EchoWorker subscribes and echoes, Manager receives ResultEvent and publishes FeedbackEvent. Full loop verified with NATS.

**Key files created:** `src/events/`, `src/manager/`, `src/workers/`, `src/config.py`, `src/__main__.py`

---

## Phase 3: LLM Workers, PRM Scoring & Training Bridge

**Date:** 2026-03-10

**Scope:** Real LLM inference (Ollama), step-level scoring (LLM-as-judge PRM), training bridge (RolloutBuffer + JSONL export).

### Experiments

#### 3.1 LLM Worker (Ollama)

- **Hypothesis:** Ollama AsyncClient can provide structured step-by-step output suitable for PRM scoring.
- **Approach:** System prompt with `<step>...</step>` and `<answer>...</answer>` markers. Regex parsing in LLMWorker.
- **Result:** Works reliably with qwen2.5:1.5b. Step parsing robust enough for prototype. ~5 req/s on CPU.
- **Graduation path:** Ollama → vLLM + LoRA (Phase 5).

#### 3.2 PRM Evaluator (LLM-as-Judge)

- **Hypothesis:** An LLM can score each reasoning step on progress + correctness (0-1) via JSON output.
- **Approach:** StepScorer protocol with LLMJudgeScorer implementation. Evaluator subscribes to results, scores steps, publishes TrainingRolloutEvent.
- **Result:** Scores are noisy but directionally correct. Good enough to bootstrap training data for future trained PRM.
- **Graduation trigger:** When judge latency bottlenecks training throughput, or when 10K+ scored trajectories accumulated.

#### 3.3 Training Bridge

- **Hypothesis:** RolloutBuffer can batch async scored rollouts for synchronous GRPO consumption.
- **Approach:** Group rollouts by prompt (group_size), emit batch when batch_size reached.
- **Result:** Clean separation — bridge handles async→batch conversion, trainer protocol consumes batches. JSONL export works for offline analysis.

### Findings

- `ResultEvent` needed a `prompt` field added — evaluator needs original prompt for context. Small duplication but keeps evaluator stateless.
- Group-based buffering is essential for GRPO — advantage calculation requires multiple responses to the same prompt.
- LLM-as-judge scores have high variance but low bias — averaging across group mitigates noise.

---

## Phase 4: GRPO Trainer Integration

**Date:** 2026-03-12

**Scope:** Standalone GRPO trainer with LoRA fine-tuning, TrainingLoop orchestrator, MockTrainer fallback, ModelUpdateEvent publishing.

### Key Decision: Standalone GRPO over OpenRLHF

- **Problem:** OpenRLHF requires torch+CUDA+ray+deepspeed+vllm — cannot install or test on CPU-only dev machine (macOS, Python 3.10).
- **Decision:** Build standalone GRPO trainer (~150 lines) using torch + transformers + peft. Same Trainer protocol — OpenRLHF becomes a drop-in swap in Phase 5.
- **Validation:** CPU-testable with distilgpt2 (~82M params). Slow (~minutes per batch) but functionally validates the pipeline.

### Experiments

#### 4.1 GRPO Math (`src/training/grpo.py`)

- **Functions:** `compute_group_advantages()`, `clipped_surrogate_loss()`, `kl_penalty()`
- **Advantage formula:** `(reward_i - mean) / std` within a prompt group
- **Finding:** Floating point precision matters — `[0.7, 0.7, 0.7]` produces variance ~1e-32 (not exactly 0.0) due to IEEE 754 representation. Fix: tolerance check `variance < 1e-12` instead of `== 0.0`.
- **Test results:** 7 list-based advantage tests pass (no torch). 4 tensor-based loss/KL tests pass with torch.

#### 4.2 Trainer Protocol + MockTrainer

- **Pattern:** Follows `StepScorer` protocol pattern from `src/rewards/scorer.py` — `@runtime_checkable` Protocol.
- **MockTrainer:** Zero-dep trainer for testing. Logs batch info, returns fake TrainStepResult with decreasing loss (1/step_count).
- **TrainStepResult:** Pydantic model with loss, mean_advantage, std_advantage, checkpoint_path, step_count.
- **Test results:** Protocol conformance verified via `isinstance(MockTrainer(), Trainer)`. 6 tests pass.

#### 4.3 GRPOTrainer (LoRA + distilgpt2)

- **Architecture:** Lazy-loaded model (import inside `_ensure_loaded()`, not at module top). Base model frozen as reference, LoRA adapter on trainable copy.
- **LoRA config:** rank=8, alpha=16, target_modules=["c_attn"], dropout=0.05, task_type=CAUSAL_LM.
- **Training loop:** Group by prompt → compute advantages → tokenize → forward pass (policy + reference) → clipped surrogate loss + KL penalty → backward → AdamW step → periodic LoRA checkpoint.
- **Checkpointing:** Saves LoRA adapter + tokenizer to `checkpoint_dir/v{step:04d}/` every N steps.
- **Finding:** distilgpt2 loads in ~2s on CPU. Full train_step with batch of 4 rollouts takes ~10-30s on CPU. Acceptable for functional validation.

#### 4.4 TrainingLoop Orchestrator

- **Wiring:** Creates NATSTrainingBridge internally, registers `_on_batch` callback.
- **Flow:** Bridge receives rollouts → buffer groups and batches → callback calls `trainer.train_step()` → publishes `ModelUpdateEvent` if checkpoint saved.
- **ModelUpdateEvent:** Published to `model.updates` topic with version, checkpoint_path, metrics (loss, advantages). Workers ignore this in Phase 4 — Phase 5 enables hot-swap.

#### 4.5 Optional Dependencies

- **Approach:** `pip install -e ".[training]"` adds torch>=2.0, transformers>=4.40, peft>=0.11.
- **Base install** (`pip install -e ".[dev]"`) works without torch — 32 tests pass, torch-dependent tests skip cleanly.
- **Finding:** pip 21.2.3 (bundled with Python 3.10) doesn't support editable installs from pyproject.toml without setup.py. Fix: upgrade pip+setuptools first.

#### 4.6 Entry Point Fallback

- **Pattern:** Try importing GRPOTrainer → if ImportError (no torch), fall back to MockTrainer. Same pattern as LLMWorker → EchoWorker fallback.
- **Config fields added:** training_model, training_lr, training_clip_epsilon, training_kl_beta, training_checkpoint_dir, training_lora_rank, training_batch_size, training_group_size, training_device.

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_grpo.py` (list-based) | 7 | None | <1s | PASS |
| `test_grpo.py` (tensor-based) | 4 | torch | <1s | SKIP (no torch) / PASS (with torch) |
| `test_trainer.py` (MockTrainer) | 6 | None | <1s | PASS |
| `test_trainer.py` (GRPOTrainer) | 3 | torch, transformers, peft | ~30s | SKIP (no torch) / PASS (with torch) |
| `test_bridge.py` (existing) | 4 | None | <1s | PASS |
| **Total standalone** | **32 pass, 5 skip** | | **<1s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/training/grpo.py` — GRPO math utilities |
| CREATE | `src/training/trainer.py` — Trainer protocol, MockTrainer, GRPOTrainer |
| CREATE | `src/training/loop.py` — TrainingLoop orchestrator |
| CREATE | `tests/training/test_grpo.py` — Advantage math + loss/KL tests |
| CREATE | `tests/training/test_trainer.py` — Protocol conformance + integration tests |
| MODIFY | `pyproject.toml` — training optional deps, slow marker |
| MODIFY | `src/config.py` — 8 training config fields |
| MODIFY | `src/training/__init__.py` — new exports |
| MODIFY | `src/__main__.py` — TrainingLoop wiring with fallback |
| MODIFY | `CLAUDE.md` — directory structure, phase status, testing docs |

### Deferred to Phase 5

- Weight hot-swap in workers (needs vLLM, Ollama has no native LoRA hot-swap)
- OpenRLHF integration (drop-in via Trainer protocol)
- Trained PRM model (still using LLM-as-judge)
- Multi-GPU / Ray / DeepSpeed
- Semantic Router as inference gateway

---

## Phase 5: Inference Abstraction, Weight Hot-Swap, OpenRLHF & Semantic Router

**Date:** 2026-03-12

**Scope:** Decouple inference from Ollama, make workers react to model updates, integrate OpenRLHF for production training, prepare Semantic Router as inference gateway.

### Key Decision: Clean Break from Ollama

- **Problem:** Both `LLMWorker` and `LLMJudgeScorer` were hardcoded to `ollama.AsyncClient` with Ollama-specific response extraction (`response["message"]["content"]`).
- **Decision:** Introduce `InferenceClient` protocol with `chat()` returning `str` directly. Adapters handle response extraction internally. Old `ollama_host` constructor params removed entirely (clean break, not backward compatible).
- **Validation:** Both Ollama and OpenAI adapters pass protocol conformance tests. Scorer and worker tests updated to mock `InferenceClient` instead of `ollama.AsyncClient`.

### Experiments

#### 5.1 InferenceClient Protocol (`src/inference/client.py`)

- **Design:** `@runtime_checkable` Protocol with single `chat()` method. Returns `str` — adapters handle response format differences internally.
- **Adapters:** `OllamaInferenceClient` (wraps `ollama.AsyncClient`, maps `json_mode=True` → `format="json"`) and `OpenAIInferenceClient` (wraps `openai.AsyncOpenAI`, maps `json_mode=True` → `response_format={"type":"json_object"}`).
- **Factory:** `create_client(backend, base_url, api_key)` — selects adapter based on config.
- **Finding:** Lazy import of `openai` inside `OpenAIInferenceClient.__init__` means base install doesn't require openai package. Same pattern as torch in GRPOTrainer.

#### 5.2 LLMWorker Refactor

- **Change:** Constructor takes `client: InferenceClient` (required) instead of `ollama_host: str`. `process()` calls `self._client.chat()` which returns `str` directly — no more `response["message"]["content"]` extraction.
- **Model reload:** Added `reload_model(event)` override — updates `self.model` and `self._active_version`. Version tracked in `ResultEvent.model_version`.

#### 5.3 LLMJudgeScorer Refactor

- **Change:** Constructor takes `client: InferenceClient` (required) instead of `ollama.AsyncClient`. `_judge_single_step()` uses `json_mode=True` instead of `format="json"`.
- **Finding:** Test simplification — mock returns `str` instead of nested dict `{"message": {"content": ...}}`.

#### 5.4 Weight Hot-Swap Infrastructure

- **BaseWorker:** Subscribes to `model.updates` topic in `start()`. `_handle_model_update()` calls `reload_model(event)` — default no-op, overridden by LLMWorker.
- **ResultEvent:** Added `model_version: str | None = None` field. `_handle_task()` sets it from `self._active_version`.
- **Finding:** EchoWorker inherits subscription but no-ops on model updates — clean separation.

#### 5.5 VLLMLoRAManager

- **Design:** Uses `httpx.AsyncClient` to call vLLM admin endpoints: `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`, `GET /v1/models`.
- **Requires:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var on vLLM server.
- **Finding:** All tests pass with mocked httpx. Real integration requires GPU with vLLM server running.

#### 5.6 OpenRLHFLauncher

- **Design:** NOT a Trainer protocol implementation — separate class that exports rollouts to JSONL and launches `python -m openrlhf.cli.train_grpo_ray` as `asyncio.create_subprocess_exec`.
- **Finding:** OpenRLHF has no simple Python API — it's a Ray-distributed CLI tool. Subprocess approach avoids coupling to OpenRLHF internals.

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_client.py` (protocol + Ollama) | 6 | ollama | <1s | PASS |
| `test_client.py` (OpenAI) | 4 | openai | <1s | SKIP (no openai) / PASS (with openai) |
| `test_vllm_lora.py` | 5 | httpx | <1s | PASS |
| `test_model_reload.py` | 6 | None | <1s | PASS |
| `test_scorer.py` (updated) | 4 | None | <1s | PASS |
| `test_types.py` (model_version) | 2 new | None | <1s | PASS |
| **Total standalone** | **51 pass, 11 skip** | | **<1s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/inference/__init__.py` — exports InferenceClient, adapters, factory |
| CREATE | `src/inference/client.py` — InferenceClient protocol + adapters |
| CREATE | `src/inference/vllm_lora.py` — VLLMLoRAManager |
| CREATE | `src/training/openrlhf_launcher.py` — OpenRLHFLauncher |
| CREATE | `tests/inference/__init__.py` |
| CREATE | `tests/inference/test_client.py` — protocol + adapter tests |
| CREATE | `tests/inference/test_vllm_lora.py` — VLLMLoRAManager tests |
| CREATE | `tests/workers/__init__.py` |
| CREATE | `tests/workers/test_model_reload.py` — worker reload tests |
| MODIFY | `src/workers/base.py` — model.updates subscription + reload_model hook |
| MODIFY | `src/workers/llm_worker.py` — InferenceClient, _active_version, reload_model |
| MODIFY | `src/rewards/scorer.py` — InferenceClient, json_mode=True |
| MODIFY | `src/events/types.py` — model_version field on ResultEvent |
| MODIFY | `src/config.py` — inference_backend, inference_base_url, inference_api_key, vllm_lora_name, trainer_backend |
| MODIFY | `src/__main__.py` — create_client factory, shared client, VLLMLoRAManager wiring |
| MODIFY | `pyproject.toml` — inference, vllm, openrlhf optional dep groups |
| MODIFY | `tests/rewards/test_scorer.py` — mock InferenceClient instead of ollama.AsyncClient |
| MODIFY | `tests/events/test_types.py` — model_version roundtrip tests |
| MODIFY | `CLAUDE.md` — directory structure, deps, phase status, config table |

### Deferred to Phase 6

- Trained PRM model (still using LLM-as-judge — need 10K+ scored trajectories first)
- DAPO graduation (requires OpenRLHF working + tuning)
- Multi-model routing rules in Semantic Router (deploy SR first, then configure)
- HaluGate as complementary StepScorer implementation
- PicoClaw edge deployment

---

## Phase 6: OpenClaw Full Agent Runtime Integration + Docker Compose Demo

**Date:** 2026-03-13

**Scope:** Make OpenClaw the full agent runtime (Manager + Worker agents run inside OpenClaw), Bridge Service translates HTTP ←→ NATS, Docker Compose provides all infrastructure. Zero changes to PRM/Training pipeline.

### Key Decision: Python Bridge Service Pattern

- **Problem:** OpenClaw is Node.js (session-based agents). Existing pipeline is Python (async NATS pub/sub). The openclaw-sdk requires Python >=3.11 but project uses 3.10.
- **Decision:** Bridge Service — a Python aiohttp server that receives HTTP requests from OpenClaw agents (via exec/curl) and publishes identical Pydantic v2 JSON to the same NATS topics. Skip the SDK entirely.
- **Validation:** All 5 Pydantic event models reused directly. PRM Evaluator and Training Loop see identical events — zero code changes needed.

### Experiments

#### 6.1 Bridge Service (`src/bridge/`)

- **Design:** `BridgeService` connects to NATS, starts aiohttp server on `:8100`. 5 HTTP endpoints: `POST /tasks/assign`, `POST /tasks/result`, `POST /feedback`, `GET /tasks/{id}/status`, `GET /health`.
- **Reuse:** All Pydantic models from `src/events/types.py`, topic helpers from `src/events/topics.py`, EventBus from `src/events/bus.py`.
- **Result polling:** Bridge subscribes to `results.*` via NATS, caches results in-memory. `GET /tasks/{id}/status` long-polls (30s timeout) or returns cached result immediately.
- **Finding:** aiohttp test client (`pytest-aiohttp`) integrates cleanly with pytest-asyncio. All 10 unit tests pass without NATS.

#### 6.2 OpenClaw Agent Configuration (`config/openclaw/`)

- **Manager SOUL.md:** Decompose requests → call `assign-task` skill → poll for result → evaluate → call `submit-feedback` skill → report to user.
- **Worker SOUL.md:** Replicates `SYSTEM_PROMPT` from `src/workers/llm_worker.py` — step-by-step solving with `<step>`/`<answer>` format for PRM compatibility.
- **Skills:** Three SKILL.md files with `exec: curl` templates targeting `bridge:8100`. Parameters templated for OpenClaw substitution.
- **Finding:** SKILL.md `exec` format is straightforward — just a bash command with curl. JSON escaping handled by OpenClaw's template engine.

#### 6.3 Standalone Training Service (`src/services/training.py`)

- **Design:** Extracted PRM Evaluator + Training Loop from `src/__main__.py` (lines 49-117) into standalone entrypoint. Connects to NATS, starts evaluator + loop, runs indefinitely.
- **Reuse:** Identical logic to `__main__.py` — same fallback chains (GRPOTrainer → MockTrainer), same config.
- **Finding:** Clean separation — training pipeline runs as its own Docker container, completely independent of Bridge/OpenClaw.

#### 6.4 Docker Compose

- **Services:** NATS (2.10), Ollama, OpenClaw, Bridge, Training — 5 containers.
- **Health checks:** NATS uses `nats-server --help`, Ollama uses `curl /api/tags`. `depends_on` with `condition: service_healthy` ensures startup order.
- **Dockerfiles:** Two — `Dockerfile` (training, installs `[training,inference]`) and `Dockerfile.bridge` (bridge, installs `[bridge]`).
- **Finding:** Pinning NATS to `2.10` avoids breaking changes. OpenClaw uses `latest` for now — should pin once stable.

### Event Flow Preservation

| Event | Old Publisher | New Publisher | Same NATS Topic | Same JSON Schema |
|-------|-------------|-------------|----------------|-----------------|
| TaskEvent | Manager (Python) | Bridge HTTP API | `tasks.{type}` | Yes (same Pydantic model) |
| ResultEvent | Worker (Python) | Bridge HTTP API | `results.{type}` | Yes |
| FeedbackEvent | Manager (Python) | Bridge HTTP API | `feedback.scored` | Yes |
| TrainingRolloutEvent | PRM Evaluator | PRM Evaluator | `training.rollouts` | **Unchanged** |
| ModelUpdateEvent | TrainingLoop | TrainingLoop | `model.updates` | **Unchanged** |

### Test Summary

| Test Suite | Tests | Deps | Speed | Status |
|------------|-------|------|-------|--------|
| `test_http_api.py` | 10 | aiohttp | <1s | PASS |
| `test_service.py` | 1 | aiohttp, NATS | <2s | SKIP (no NATS) / PASS (with NATS) |
| All existing tests | 67 | various | ~5s | PASS (unchanged) |
| **Total** | **77 collected, 72 pass, 5 skip** | | **~5s** | |

### Files

| Action | Path |
|--------|------|
| CREATE | `src/bridge/__init__.py` |
| CREATE | `src/bridge/__main__.py` — entrypoint for `python -m src.bridge` |
| CREATE | `src/bridge/service.py` — BridgeService (NATS + aiohttp) |
| CREATE | `src/bridge/http_api.py` — 5 HTTP endpoints |
| CREATE | `src/services/__init__.py` |
| CREATE | `src/services/__main__.py` — entrypoint for `python -m src.services.training` |
| CREATE | `src/services/training.py` — standalone PRM + Training service |
| CREATE | `config/openclaw/AGENTS.md` — agent registry |
| CREATE | `config/openclaw/manager/SOUL.md` — manager behavior |
| CREATE | `config/openclaw/manager/IDENTITY.md` — manager identity |
| CREATE | `config/openclaw/worker/SOUL.md` — worker behavior |
| CREATE | `config/openclaw/worker/IDENTITY.md` — worker identity |
| CREATE | `config/openclaw/skills/assign-task/SKILL.md` |
| CREATE | `config/openclaw/skills/submit-result/SKILL.md` |
| CREATE | `config/openclaw/skills/submit-feedback/SKILL.md` |
| CREATE | `docker-compose.yml` — 5 services |
| CREATE | `Dockerfile` — training service image |
| CREATE | `Dockerfile.bridge` — bridge service image |
| CREATE | `scripts/demo.sh` — one-command demo |
| CREATE | `tests/bridge/__init__.py` |
| CREATE | `tests/bridge/test_http_api.py` — 10 unit tests |
| CREATE | `tests/bridge/test_service.py` — 1 integration test |
| MODIFY | `src/config.py` — `bridge_port`, `openclaw_gateway_url` |
| MODIFY | `pyproject.toml` — `[bridge]` optional extra |

### Deferred to Phase 7

- Trained PRM model (still using LLM-as-judge — need 10K+ scored trajectories first)
- DAPO graduation (requires OpenRLHF working + tuning)
- Multi-model routing rules in Semantic Router
- HaluGate as complementary StepScorer
- OpenClaw WebSocket relay (Bridge → OpenClaw session for bidirectional communication)
- Pin OpenClaw Docker image version
