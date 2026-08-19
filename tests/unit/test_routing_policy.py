from __future__ import annotations

from app.core.schemas import LLMRequest, LLMResponse
from app.gateway.routing import ProviderTier, RoutingPolicy, default_policy


class FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    async def complete(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="ok",
            provider=self.name,
            model="fake",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )


def test_default_policy_local_first_for_all_tasks() -> None:
    ollama = FakeProvider("ollama")
    bedrock = FakeProvider("bedrock")
    policy = default_policy(ollama, bedrock_provider=bedrock)

    for task in ("embed", "draft", "judge"):
        providers = policy.select(task, {})
        assert providers[0].name == "ollama"

    # embed is local-only
    assert [p.name for p in policy.select("embed", {})] == ["ollama"]
    # draft/judge escalate to cloud after local
    assert [p.name for p in policy.select("draft", {})] == ["ollama", "bedrock"]
    assert [p.name for p in policy.select("judge", {})] == ["ollama", "bedrock"]


def test_default_policy_without_cloud_provider() -> None:
    ollama = FakeProvider("ollama")
    policy = default_policy(ollama)
    assert [p.name for p in policy.select("draft", {})] == ["ollama"]


def test_cost_ceiling_filters_tiers_above_budget() -> None:
    cheap = FakeProvider("cheap")
    expensive = FakeProvider("expensive")
    policy = RoutingPolicy(
        tiers=[
            ProviderTier(provider=cheap, tasks=("draft",), max_cost_usd=0.0),
            ProviderTier(provider=expensive, tasks=("draft",), max_cost_usd=0.05),
        ]
    )

    # No ceiling: both tiers available, cheap first.
    assert [p.name for p in policy.select("draft", {})] == ["cheap", "expensive"]

    # Ceiling below the expensive tier's cost excludes it.
    assert [p.name for p in policy.select("draft", {"cost_ceiling_usd": 0.01})] == ["cheap"]

    # Ceiling that covers both keeps both.
    assert [p.name for p in policy.select("draft", {"cost_ceiling_usd": 1.0})] == ["cheap", "expensive"]


def test_latency_budget_filters_tiers() -> None:
    fast = FakeProvider("fast")
    slow = FakeProvider("slow")
    policy = RoutingPolicy(
        tiers=[
            ProviderTier(provider=fast, tasks=("judge",), max_latency_ms=200),
            ProviderTier(provider=slow, tasks=("judge",), max_latency_ms=5000),
        ]
    )
    assert [p.name for p in policy.select("judge", {"max_latency_ms": 300})] == ["fast"]
    assert [p.name for p in policy.select("judge", {"max_latency_ms": 10000})] == ["fast", "slow"]


def test_task_filtering_excludes_irrelevant_tiers() -> None:
    embedder = FakeProvider("embedder")
    drafter = FakeProvider("drafter")
    policy = RoutingPolicy(
        tiers=[
            ProviderTier(provider=embedder, tasks=("embed",)),
            ProviderTier(provider=drafter, tasks=("draft", "judge")),
        ]
    )
    assert [p.name for p in policy.select("embed", {})] == ["embedder"]
    assert [p.name for p in policy.select("draft", {})] == ["drafter"]
    assert [p.name for p in policy.select("judge", {})] == ["drafter"]
