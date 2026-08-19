"""Observability: structured logging + Prometheus metrics + optional
Langfuse tracing (LLD/HLD "Observability: Langfuse/Phoenix + Prometheus").

Every served query and every gateway call feeds the same metrics registry
exposed at GET /metrics; Langfuse tracing is best-effort and self-disables
when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set, so the service
runs identically with or without a Langfuse instance up.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

from app.core.config import get_settings

# -- structured logging ---------------------------------------------------


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# -- Prometheus metrics ----------------------------------------------------

QUERY_LATENCY_SECONDS = Histogram(
    "groundcontrol_query_latency_seconds",
    "End-to-end /query request latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)
QUERY_ITERATIONS = Histogram(
    "groundcontrol_query_iterations",
    "Agent loop iterations per served query",
    buckets=(1, 2, 3, 4, 5),
)
QUERY_COST_USD = Counter(
    "groundcontrol_query_cost_usd_total",
    "Cumulative LLM cost attributed to served queries",
)
QUERY_CACHE_HITS = Counter(
    "groundcontrol_query_cache_hits_total",
    "Semantic cache hits on the gateway's draft path",
)
QUERY_LOW_CONFIDENCE = Counter(
    "groundcontrol_query_low_confidence_total",
    "Queries finalized without clearing the groundedness/faithfulness thresholds",
)
GATEWAY_PROVIDER_CALLS = Counter(
    "groundcontrol_gateway_provider_calls_total",
    "LLM calls per provider and task, gateway-level",
    labelnames=("provider", "task", "outcome"),
)
EVAL_SCORE = Histogram(
    "groundcontrol_eval_score",
    "Per-request online judge scores",
    labelnames=("metric",),
    buckets=(0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0),
)


def record_query_metrics(
    *, latency_ms: float, iterations: int, cost_usd: float, cache_hit: bool, low_confidence: bool
) -> None:
    QUERY_LATENCY_SECONDS.observe(latency_ms / 1000)
    QUERY_ITERATIONS.observe(iterations)
    QUERY_COST_USD.inc(cost_usd)
    if cache_hit:
        QUERY_CACHE_HITS.inc()
    if low_confidence:
        QUERY_LOW_CONFIDENCE.inc()


def record_eval_scores(*, faithfulness: float, groundedness: float) -> None:
    EVAL_SCORE.labels(metric="faithfulness").observe(faithfulness)
    EVAL_SCORE.labels(metric="groundedness").observe(groundedness)


def record_provider_call(provider: str, task: str, success: bool) -> None:
    GATEWAY_PROVIDER_CALLS.labels(provider=provider, task=task, outcome="success" if success else "failure").inc()


# -- optional Langfuse tracing ----------------------------------------------


@contextmanager
def trace_span(name: str, **attrs: Any) -> Iterator[None]:
    """Best-effort tracing span. If `langfuse` isn't installed or no keys
    are configured, this is a no-op context manager — callers never need to
    branch on whether tracing is enabled.
    """
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        yield
        return
    try:
        from langfuse.decorators import langfuse_context  # type: ignore[import-not-found]

        langfuse_context.update_current_observation(name=name, metadata=attrs)
        yield
    except ImportError:
        yield
