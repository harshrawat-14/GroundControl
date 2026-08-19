"""TokenAccountant: per-call token/cost/latency bookkeeping.

Deliberately decoupled from the `requests` table (owned by the M1/M2
integration code) — this exposes a clean in-memory `record()` + `summary()`
API that callers can persist however they like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class CallRecord:
    provider: str
    model: str
    task: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ProviderSummary:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    cache_hits: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0


class TokenAccountant:
    """Accumulates in-memory counters for every gateway call."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> CallRecord:
        rec = CallRecord(
            provider=provider,
            model=model,
            task=task,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
        self._records.append(rec)
        return rec

    def summary(self) -> dict[str, ProviderSummary]:
        """Return a per-provider rollup of every call recorded so far."""
        rollup: dict[str, ProviderSummary] = {}
        for rec in self._records:
            s = rollup.setdefault(rec.provider, ProviderSummary())
            s.calls += 1
            s.prompt_tokens += rec.prompt_tokens
            s.completion_tokens += rec.completion_tokens
            s.cost_usd += rec.cost_usd
            s.latency_ms_total += rec.latency_ms
            if rec.cache_hit:
                s.cache_hits += 1
        return rollup

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def cache_hit_rate(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.cache_hit) / len(self._records)

    @property
    def records(self) -> list[CallRecord]:
        return list(self._records)
