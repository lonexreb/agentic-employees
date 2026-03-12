from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    nats_url: str = os.environ.get("NATS_URL", "nats://localhost:4222")
    manager_id: str = os.environ.get("MANAGER_ID", "manager-01")
    worker_id: str = os.environ.get("WORKER_ID", "worker-01")
    task_timeout_seconds: float = float(os.environ.get("TASK_TIMEOUT_SECONDS", "30"))
    llm_model: str = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
