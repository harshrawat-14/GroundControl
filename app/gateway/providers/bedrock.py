"""AWS Bedrock cloud provider.

`boto3` is imported lazily so the package can be installed and the gateway
can boot without the AWS SDK present. The provider only activates when
`settings.cloud_provider == "bedrock"`; instantiating/calling it without
valid AWS credentials raises rather than faking a network response.
"""

from __future__ import annotations

import json
import time

from app.core.config import Settings, get_settings
from app.core.schemas import LLMRequest, LLMResponse
from app.gateway.providers.rates import BEDROCK_DEFAULT_RATE, BEDROCK_RATES_PER_1K, estimate_cost


class BedrockUnavailableError(RuntimeError):
    """Raised when the Bedrock provider cannot be used (missing SDK/creds/config)."""


class BedrockProvider:
    """LLMProvider backed by AWS Bedrock's InvokeModel API."""

    name = "bedrock"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._settings.cloud_provider != "bedrock":
            raise BedrockUnavailableError("cloud_provider is not set to 'bedrock'")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise BedrockUnavailableError("boto3 is not installed") from exc
        if not self._settings.aws_region or not self._settings.bedrock_model_id:
            raise BedrockUnavailableError("aws_region/bedrock_model_id not configured")
        self._client = boto3.client("bedrock-runtime", region_name=self._settings.aws_region)
        return self._client

    async def complete(self, req: LLMRequest) -> LLMResponse:
        import asyncio

        return await asyncio.to_thread(self._complete_sync, req)

    def _complete_sync(self, req: LLMRequest) -> LLMResponse:
        client = self._get_client()
        model_id = self._settings.bedrock_model_id
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            body["system"] = req.system

        started = time.perf_counter()
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - started) * 1000
        payload = json.loads(response["body"].read())

        text = "".join(
            block.get("text", "") for block in payload.get("content", []) if isinstance(block, dict)
        )
        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        cost = estimate_cost(BEDROCK_RATES_PER_1K, BEDROCK_DEFAULT_RATE, model_id, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
