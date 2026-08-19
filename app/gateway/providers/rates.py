"""Published per-1K-token USD rate tables for cloud providers.

Rates are approximate published list prices (2025) and intentionally kept in
one place so `TokenAccountant` produces meaningful cost estimates. Update as
provider pricing changes.
"""

from __future__ import annotations

# {model_id: (prompt_usd_per_1k, completion_usd_per_1k)}
BEDROCK_RATES_PER_1K: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (0.003, 0.015),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "meta.llama3-1-8b-instruct-v1:0": (0.00022, 0.00022),
    "meta.llama3-1-70b-instruct-v1:0": (0.00099, 0.00099),
}
BEDROCK_DEFAULT_RATE: tuple[float, float] = (0.003, 0.015)

VERTEX_RATES_PER_1K: dict[str, tuple[float, float]] = {
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash": (0.0001, 0.0004),
}
VERTEX_DEFAULT_RATE: tuple[float, float] = (0.00125, 0.005)


def estimate_cost(rates: dict[str, tuple[float, float]], default: tuple[float, float], model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = rates.get(model, default)
    return (prompt_tokens / 1000.0) * prompt_rate + (completion_tokens / 1000.0) * completion_rate
