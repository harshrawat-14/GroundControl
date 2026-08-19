"""Unit tests for CompositeEvaluator and the offline benchmark runner."""

from __future__ import annotations

import asyncio

from app.core.schemas import EvalSample
from app.evaluation.composite import CompositeEvaluator
from app.evaluation.golden_dataset import aggregate_scores, load_golden_dataset, run_benchmark

GROUNDED_SAMPLE = EvalSample(
    question="What is the uptime SLA?",
    contexts=[
        (
            "Section 5.1 Availability: Vendor commits to a monthly uptime of 99.9% "
            "for the Production Environment, excluding scheduled maintenance windows."
        ),
    ],
    answer="The uptime SLA commitment is 99.9% monthly uptime, excluding scheduled maintenance.",
    ground_truth="99.9% monthly uptime, excluding scheduled maintenance windows.",
)

HALLUCINATED_SAMPLE = EvalSample(
    question="What is the uptime SLA?",
    contexts=[
        (
            "Section 5.1 Availability: Vendor commits to a monthly uptime of 99.9% "
            "for the Production Environment, excluding scheduled maintenance windows."
        ),
    ],
    answer="Bananas are an excellent source of potassium and grow best in tropical climates.",
    ground_truth="99.9% monthly uptime, excluding scheduled maintenance windows.",
)


def test_composite_merges_all_four_metrics() -> None:
    result = asyncio.run(CompositeEvaluator().evaluate(GROUNDED_SAMPLE))
    assert result.faithfulness > 0.0
    assert result.groundedness > 0.0
    assert result.relevance > 0.0
    assert result.correctness is not None and result.correctness > 0.0
    assert len(result.details["sub_results"]) == 4


def test_composite_grounded_beats_hallucinated_on_every_metric() -> None:
    grounded = asyncio.run(CompositeEvaluator().evaluate(GROUNDED_SAMPLE))
    hallucinated = asyncio.run(CompositeEvaluator().evaluate(HALLUCINATED_SAMPLE))

    assert grounded.faithfulness > hallucinated.faithfulness
    assert grounded.groundedness > hallucinated.groundedness
    assert (grounded.correctness or 0.0) > (hallucinated.correctness or 0.0)


def test_run_benchmark_over_golden_dataset_produces_aggregate() -> None:
    samples = load_golden_dataset()
    per_sample, aggregate = asyncio.run(run_benchmark(CompositeEvaluator(), samples))

    assert aggregate["sample_count"] == len(samples)
    assert 0.0 <= aggregate["faithfulness_mean"] <= 1.0
    assert 0.0 <= aggregate["groundedness_mean"] <= 1.0
    assert 0.0 <= aggregate["relevance_mean"] <= 1.0
    assert aggregate["correctness_mean"] is not None
    assert len(per_sample) == len(samples)


def test_aggregate_scores_empty_dataset() -> None:
    assert aggregate_scores([]) == {"sample_count": 0}
