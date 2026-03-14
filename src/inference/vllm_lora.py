"""VLLMLoRAManager — dynamic LoRA adapter loading via vLLM admin endpoints."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class VLLMLoRAManager:
    """Manages LoRA adapter hot-swap on a vLLM server.

    Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the vLLM server.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def load_adapter(
        self, name: str, path: str, *, inplace: bool = True
    ) -> None:
        """Load a LoRA adapter into the running vLLM server."""
        payload = {
            "lora_name": name,
            "lora_path": path,
            "load_inplace": inplace,
        }
        response = await self._http.post("/v1/load_lora_adapter", json=payload)
        response.raise_for_status()
        logger.info("Loaded LoRA adapter %s from %s", name, path)

    async def unload_adapter(self, name: str) -> None:
        """Unload a LoRA adapter from the running vLLM server."""
        payload = {"lora_name": name}
        response = await self._http.post("/v1/unload_lora_adapter", json=payload)
        response.raise_for_status()
        logger.info("Unloaded LoRA adapter %s", name)

    async def health_check(self) -> list[str]:
        """List available models on the vLLM server."""
        response = await self._http.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        await self._http.aclose()
