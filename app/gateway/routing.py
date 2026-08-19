"""RoutingPolicy: Strategy for choosing an ordered provider escalation ladder.

Per LLD 2.3: routes by task (embed vs draft vs judge), cost ceiling, and
latency budget; supports a cost/latency-tiered escalation ladder
(local -> cloud-small -> cloud-large).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.core.interfaces import LLMProvider

Task = Literal["embed", "draft", "judge"]


@dataclass
class ProviderTier:
    """One rung of the escalation ladder."""

    provider: LLMProvider
    tasks: tuple[Task, ...]
    max_cost_usd: float | None = None  # approx per-call ceiling this tier respects
    max_latency_ms: float | None = None


@dataclass
class RoutingPolicy:
    """Ordered, configurable cost/latency-tiered escalation ladder.

    Tiers are tried in the order given by `tiers`. `select()` filters tiers by
    task applicability and by the caller's constraints (cost ceiling / latency
    budget), preserving relative order (cheapest/local first).
    """

    tiers: list[ProviderTier] = field(default_factory=list)

    def select(self, task: Task, constraints: dict | None = None) -> list[LLMProvider]:
        constraints = constraints or {}
        cost_ceiling = constraints.get("cost_ceiling_usd")
        latency_budget_ms = constraints.get("max_latency_ms")

        chosen: list[LLMProvider] = []
        for tier in self.tiers:
            if task not in tier.tasks:
                continue
            if cost_ceiling is not None and tier.max_cost_usd is not None and tier.max_cost_usd > cost_ceiling:
                continue
            if (
                latency_budget_ms is not None
                and tier.max_latency_ms is not None
                and tier.max_latency_ms > latency_budget_ms
            ):
                continue
            chosen.append(tier.provider)
        return chosen


def default_policy(
    ollama_provider: LLMProvider,
    bedrock_provider: LLMProvider | None = None,
    vertex_provider: LLMProvider | None = None,
) -> RoutingPolicy:
    """Build the standard local-first escalation ladder.

    Ollama (free, local) is tried first for every task; a configured cloud
    provider follows as cloud-small/cloud-large depending on what's wired up.
    """

    tiers = [
        ProviderTier(
            provider=ollama_provider,
            tasks=("embed", "draft", "judge"),
            max_cost_usd=0.0,
            max_latency_ms=None,
        )
    ]
    if bedrock_provider is not None:
        tiers.append(
            ProviderTier(
                provider=bedrock_provider,
                tasks=("draft", "judge"),
                max_cost_usd=None,
                max_latency_ms=None,
            )
        )
    if vertex_provider is not None:
        tiers.append(
            ProviderTier(
                provider=vertex_provider,
                tasks=("draft", "judge"),
                max_cost_usd=None,
                max_latency_ms=None,
            )
        )
    return RoutingPolicy(tiers=tiers)
