"""Google Vertex AI cloud provider.

`google.cloud.aiplatform` / `vertexai` are imported lazily so the package can
boot without the Vertex SDK present. Only activates when
`settings.cloud_provider == "vertex"`.
"""

from __future__ import annotations

import time

from app.core.config import Settings, get_settings
from app.core.schemas import LLMRequest, LLMResponse
from app.gateway.providers.rates import VERTEX_DEFAULT_RATE, VERTEX_RATES_PER_1K, estimate_cost


class VertexUnavailableError(RuntimeError):
    """Raised when the Vertex provider cannot be used (missing SDK/creds/config)."""


class VertexProvider:
    """LLMProvider backed by Google Vertex AI's generative models."""

    name = "vertex"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._settings.cloud_provider != "vertex":
            raise VertexUnavailableError("cloud_provider is not set to 'vertex'")
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise VertexUnavailableError("google-cloud-aiplatform is not installed") from exc
        if not self._settings.vertex_project_id or not self._settings.vertex_model_id:
            raise VertexUnavailableError("vertex_project_id/vertex_model_id not configured")
        vertexai.init(project=self._settings.vertex_project_id, location=self._settings.vertex_location or "us-central1")
        self._model = GenerativeModel(self._settings.vertex_model_id)
        return self._model

    async def complete(self, req: LLMRequest) -> LLMResponse:
        import asyncio

        return await asyncio.to_thread(self._complete_sync, req)

    def _complete_sync(self, req: LLMRequest) -> LLMResponse:
        model = self._get_model()
        model_id = self._settings.vertex_model_id
        prompt = f"{req.system}\n\n{req.prompt}" if req.system else req.prompt

        started = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": req.temperature,
                "max_output_tokens": req.max_tokens,
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000

        text = response.text if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cost = estimate_cost(VERTEX_RATES_PER_1K, VERTEX_DEFAULT_RATE, model_id, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
