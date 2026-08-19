"""AIGateway: the single choke point for all model calls.

Public entrypoint for both the agent orchestrator's draft/judge calls and
any other caller that needs an LLM completion:

    from app.gateway.gateway import AIGateway
    response: LLMResponse = await gateway.complete(LLMRequest(task="draft", prompt=...))

Checks the semantic cache first; tries providers in `RoutingPolicy` order;
on error/timeout retries with backoff (tenacity) then advances to the next
provider; trips a per-provider circuit breaker after repeated consecutive
failures (skipping it for a cooldown window); records every call via
`TokenAccountant`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.interfaces import LLMProvider
from app.core.schemas import LLMRequest, LLMResponse
from app.gateway.routing import RoutingPolicy
from app.gateway.semantic_cache import InMemorySemanticCache, SemanticCache
from app.gateway.token_accountant import TokenAccountant

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:  # pragma: no cover - exercised indirectly if httpx is present
    import httpx

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
except ImportError:  # pragma: no cover
    pass


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None


@dataclass
class CircuitBreaker:
    """Simple per-provider circuit breaker.

    Opens after `failure_threshold` consecutive failures and stays open for
    `cooldown_seconds`, during which the provider is skipped.
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    _states: dict[str, _CircuitState] = field(default_factory=dict)

    def _state(self, provider_name: str) -> _CircuitState:
        return self._states.setdefault(provider_name, _CircuitState())

    def is_open(self, provider_name: str) -> bool:
        state = self._state(provider_name)
        if state.opened_at is None:
            return False
        if time.monotonic() - state.opened_at >= self.cooldown_seconds:
            # Cooldown elapsed: half-open, allow a fresh attempt.
            state.opened_at = None
            state.consecutive_failures = 0
            return False
        return True

    def record_success(self, provider_name: str) -> None:
        state = self._state(provider_name)
        state.consecutive_failures = 0
        state.opened_at = None

    def record_failure(self, provider_name: str) -> None:
        state = self._state(provider_name)
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold and state.opened_at is None:
            state.opened_at = time.monotonic()


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in the routing chain fails or is circuit-open."""


class AIGateway:
    """The single public entrypoint for all model calls."""

    def __init__(
        self,
        routing_policy: RoutingPolicy,
        token_accountant: TokenAccountant | None = None,
        semantic_cache: SemanticCache | InMemorySemanticCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_attempts_per_provider: int = 2,
        namespace: str = "default",
    ) -> None:
        self._routing_policy = routing_policy
        self._accountant = token_accountant or TokenAccountant()
        self._cache = semantic_cache
        self._breaker = circuit_breaker or CircuitBreaker()
        self._max_attempts_per_provider = max_attempts_per_provider
        self._namespace = namespace

    @property
    def token_accountant(self) -> TokenAccountant:
        return self._accountant

    async def complete(self, req: LLMRequest) -> LLMResponse:
        cache_key_model = None
        if self._cache is not None:
            # Cache is keyed per-model; probe with the first candidate
            # provider's model name isn't known ahead of routing, so we use
            # the task as a coarse namespace segment instead.
            cache_key_model = req.task
            cached = await self._cache.get(req.prompt, cache_key_model, namespace=self._namespace)
            if cached is not None:
                self._accountant.record(
                    provider=cached.provider,
                    model=cached.model,
                    task=req.task,
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    cost_usd=0.0,
                    latency_ms=cached.latency_ms,
                    cache_hit=True,
                )
                return cached

        constraints: dict = {}
        if req.cost_ceiling_usd is not None:
            constraints["cost_ceiling_usd"] = req.cost_ceiling_usd

        providers = self._routing_policy.select(req.task, constraints)
        if not providers:
            raise AllProvidersFailedError(f"no providers configured for task={req.task!r}")

        last_error: Exception | None = None
        for provider in providers:
            if self._breaker.is_open(provider.name):
                continue
            try:
                response = await self._call_with_retry(provider, req)
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider error triggers fallback
                self._breaker.record_failure(provider.name)
                last_error = exc
                continue

            self._breaker.record_success(provider.name)
            self._accountant.record(
                provider=response.provider,
                model=response.model,
                task=req.task,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                cache_hit=False,
            )
            if self._cache is not None and cache_key_model is not None:
                await self._cache.set(req.prompt, cache_key_model, response, namespace=self._namespace)
            return response

        raise AllProvidersFailedError(
            f"all providers failed or circuit-open for task={req.task!r}"
        ) from last_error

    async def _call_with_retry(self, provider: LLMProvider, req: LLMRequest) -> LLMResponse:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts_per_provider),
            wait=wait_exponential(multiplier=0.1, max=2.0),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                return await provider.complete(req)
        raise AllProvidersFailedError(f"provider {provider.name} exhausted retries")  # pragma: no cover
