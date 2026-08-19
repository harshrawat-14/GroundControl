from __future__ import annotations

import pytest

from app.core.schemas import LLMRequest, LLMResponse
from app.gateway.gateway import AIGateway, AllProvidersFailedError, CircuitBreaker
from app.gateway.routing import ProviderTier, RoutingPolicy
from app.gateway.semantic_cache import InMemorySemanticCache
from app.gateway.token_accountant import TokenAccountant


class AlwaysFailsProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        raise ConnectionError("boom")


class AlwaysSucceedsProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="hi",
            provider=self.name,
            model="m",
            prompt_tokens=3,
            completion_tokens=3,
            cost_usd=0.001,
            latency_ms=5.0,
        )


def _policy(*providers) -> RoutingPolicy:
    return RoutingPolicy(
        tiers=[ProviderTier(provider=p, tasks=("draft", "judge", "embed")) for p in providers]
    )


async def test_fallback_advances_to_next_provider_on_failure() -> None:
    bad = AlwaysFailsProvider("bad")
    good = AlwaysSucceedsProvider("good")
    gateway = AIGateway(routing_policy=_policy(bad, good), max_attempts_per_provider=1)

    resp = await gateway.complete(LLMRequest(task="draft", prompt="hello"))
    assert resp.provider == "good"
    assert bad.calls == 1
    assert good.calls == 1


async def test_all_providers_failing_raises() -> None:
    bad1 = AlwaysFailsProvider("bad1")
    bad2 = AlwaysFailsProvider("bad2")
    gateway = AIGateway(routing_policy=_policy(bad1, bad2), max_attempts_per_provider=1)

    with pytest.raises(AllProvidersFailedError):
        await gateway.complete(LLMRequest(task="draft", prompt="hello"))


async def test_circuit_breaker_opens_after_threshold_and_skips_provider() -> None:
    bad = AlwaysFailsProvider("bad")
    good = AlwaysSucceedsProvider("good")
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=999)
    gateway = AIGateway(
        routing_policy=_policy(bad, good),
        circuit_breaker=breaker,
        max_attempts_per_provider=1,
    )

    # First two calls: "bad" fails twice, tripping the breaker; "good" serves both.
    await gateway.complete(LLMRequest(task="draft", prompt="one"))
    await gateway.complete(LLMRequest(task="draft", prompt="two"))
    assert bad.calls == 2
    assert breaker.is_open("bad") is True

    # Third call: breaker open, "bad" should be skipped entirely.
    await gateway.complete(LLMRequest(task="draft", prompt="three"))
    assert bad.calls == 2  # unchanged — skipped
    assert good.calls == 3


async def test_token_accountant_records_every_call() -> None:
    good = AlwaysSucceedsProvider("good")
    accountant = TokenAccountant()
    gateway = AIGateway(routing_policy=_policy(good), token_accountant=accountant)

    await gateway.complete(LLMRequest(task="draft", prompt="hello"))
    await gateway.complete(LLMRequest(task="draft", prompt="world"))

    summary = accountant.summary()
    assert summary["good"].calls == 2
    assert summary["good"].cost_usd == pytest.approx(0.002)


async def test_semantic_cache_short_circuits_provider_call() -> None:
    good = AlwaysSucceedsProvider("good")

    def embed(text: str) -> list[float]:
        return [1.0, 0.0]

    cache = InMemorySemanticCache(embed, similarity_threshold=0.9)
    gateway = AIGateway(routing_policy=_policy(good), semantic_cache=cache)

    first = await gateway.complete(LLMRequest(task="draft", prompt="hello"))
    assert first.cache_hit is False
    assert good.calls == 1

    second = await gateway.complete(LLMRequest(task="draft", prompt="hello"))
    assert second.cache_hit is True
    assert good.calls == 1  # provider not called again
