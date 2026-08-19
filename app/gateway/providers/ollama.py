"""Local Ollama provider — zero-cost dev/online-judge model."""

from __future__ import annotations

import time

import httpx

from app.core.config import Settings, get_settings
from app.core.schemas import LLMRequest, LLMResponse


class OllamaProvider:
    """LLMProvider backed by a local Ollama daemon."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client

    async def complete(self, req: LLMRequest) -> LLMResponse:
        payload = {
            "model": self._settings.ollama_model,
            "prompt": req.prompt,
            "system": req.system or "",
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/generate"
        started = time.perf_counter()
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if owns_client:
                await client.aclose()
        latency_ms = (time.perf_counter() - started) * 1000

        return LLMResponse(
            text=data.get("response", ""),
            provider=self.name,
            model=self._settings.ollama_model,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            cost_usd=0.0,
            latency_ms=latency_ms,
        )
