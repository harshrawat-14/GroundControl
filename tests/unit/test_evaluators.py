"""Unit tests for the M3 individual evaluators, the OnlineJudge, and the
golden-dataset loader. All tests run entirely on local TF-IDF heuristics —
no live LLM or network access required (see app/evaluation/similarity.py).
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.schemas import EvalSample
from app.evaluation.correctness import CorrectnessEvaluator
from app.evaluation.faithfulness import FaithfulnessEvaluator
from app.evaluation.golden_dataset import load_golden_dataset
from app.evaluation.groundedness import GroundednessEvaluator
from app.evaluation.online_judge import OnlineJudge
from app.evaluation.relevance import RelevanceEvaluator

GROUNDED_SAMPLE = EvalSample(
    question="What is the notice period for terminating the contract?",
    contexts=[
        (
            "Section 8.2 Termination for Convenience: Either party may terminate this "
            "Agreement for convenience upon ninety (90) days prior written notice."
        ),
    ],
    answer="The notice period for terminating the contract for convenience is ninety (90) days prior written notice.",
    ground_truth="Ninety (90) days prior written notice.",
)

HALLUCINATED_SAMPLE = EvalSample(
    question="What is the notice period for terminating the contract?",
    contexts=[
        (
            "Section 8.2 Termination for Convenience: Either party may terminate this "
            "Agreement for convenience upon ninety (90) days prior written notice."
        ),
    ],
    answer="The moon landing occurred in 1969 and pineapple is a tropical fruit grown in Hawaii.",
    ground_truth="Ninety (90) days prior written notice.",
)


class TestFaithfulnessEvaluator:
    def test_grounded_answer_scores_high(self) -> None:
        result = asyncio.run(FaithfulnessEvaluator().evaluate(GROUNDED_SAMPLE))
        assert result.faithfulness >= 0.5

    def test_hallucinated_answer_scores_low(self) -> None:
        result = asyncio.run(FaithfulnessEvaluator().evaluate(HALLUCINATED_SAMPLE))
        assert result.faithfulness < 0.3

    def test_empty_answer_scores_zero(self) -> None:
        sample = GROUNDED_SAMPLE.model_copy(update={"answer": ""})
        result = asyncio.run(FaithfulnessEvaluator().evaluate(sample))
        assert result.faithfulness == 0.0


class TestGroundednessEvaluator:
    def test_grounded_answer_scores_high(self) -> None:
        result = asyncio.run(GroundednessEvaluator().evaluate(GROUNDED_SAMPLE))
        assert result.groundedness > 0.4

    def test_hallucinated_answer_scores_low(self) -> None:
        result = asyncio.run(GroundednessEvaluator().evaluate(HALLUCINATED_SAMPLE))
        assert result.groundedness < 0.1

    def test_no_context_scores_zero(self) -> None:
        sample = GROUNDED_SAMPLE.model_copy(update={"contexts": []})
        result = asyncio.run(GroundednessEvaluator().evaluate(sample))
        assert result.groundedness == 0.0


class TestRelevanceEvaluator:
    def test_on_topic_answer_scores_higher_than_off_topic(self) -> None:
        on_topic = asyncio.run(RelevanceEvaluator().evaluate(GROUNDED_SAMPLE))
        off_topic = asyncio.run(RelevanceEvaluator().evaluate(HALLUCINATED_SAMPLE))
        assert on_topic.relevance > off_topic.relevance


class TestCorrectnessEvaluator:
    def test_matches_ground_truth_scores_high(self) -> None:
        result = asyncio.run(CorrectnessEvaluator().evaluate(GROUNDED_SAMPLE))
        assert result.correctness is not None
        assert result.correctness > 0.5

    def test_mismatched_ground_truth_scores_low(self) -> None:
        result = asyncio.run(CorrectnessEvaluator().evaluate(HALLUCINATED_SAMPLE))
        assert result.correctness is not None
        assert result.correctness < 0.2

    def test_missing_ground_truth_is_none(self) -> None:
        sample = GROUNDED_SAMPLE.model_copy(update={"ground_truth": None})
        result = asyncio.run(CorrectnessEvaluator().evaluate(sample))
        assert result.correctness is None


class TestOnlineJudge:
    def test_judge_returns_all_three_online_metrics(self) -> None:
        judge = OnlineJudge()
        result = asyncio.run(judge.judge(GROUNDED_SAMPLE.question, GROUNDED_SAMPLE.answer, GROUNDED_SAMPLE.contexts))
        assert result.faithfulness > 0.0
        assert result.groundedness > 0.0
        assert result.relevance >= 0.0
        assert result.correctness is None

    def test_judge_degrades_gracefully_on_timeout(self) -> None:
        judge = OnlineJudge(max_latency_s=0.0)
        result = asyncio.run(
            judge.judge(GROUNDED_SAMPLE.question, GROUNDED_SAMPLE.answer, GROUNDED_SAMPLE.contexts)
        )
        assert result.details.get("degraded") is True

    def test_evaluate_protocol_adapter_matches_judge(self) -> None:
        judge = OnlineJudge()
        via_evaluate = asyncio.run(judge.evaluate(GROUNDED_SAMPLE))
        via_judge = asyncio.run(
            judge.judge(GROUNDED_SAMPLE.question, GROUNDED_SAMPLE.answer, GROUNDED_SAMPLE.contexts)
        )
        assert via_evaluate.faithfulness == via_judge.faithfulness


class TestGoldenDataset:
    def test_loads_shipped_golden_dataset(self) -> None:
        samples = load_golden_dataset()
        assert len(samples) >= 5
        for sample in samples:
            assert sample.question
            assert sample.answer
            assert sample.contexts
            assert sample.ground_truth

    @pytest.mark.parametrize("evaluator_cls", [FaithfulnessEvaluator, GroundednessEvaluator])
    def test_golden_samples_are_well_grounded(self, evaluator_cls) -> None:
        # Regression guard: every shipped golden-set answer should be
        # constructed to be faithful/grounded to its own contexts.
        samples = load_golden_dataset()
        evaluator = evaluator_cls()
        scores = [asyncio.run(evaluator.evaluate(s)) for s in samples]
        metric = "faithfulness" if isinstance(evaluator, FaithfulnessEvaluator) else "groundedness"
        values = [getattr(s, metric) for s in scores]
        assert sum(values) / len(values) > 0.3
