"""CompositeEvaluator: runs Faithfulness/Groundedness/Relevance/Correctness
concurrently and merges them into a single EvalResult, per LLD 2.2
("... composed by a CompositeEvaluator").

This is the evaluator used by the offline benchmark runner
(`app.evaluation.golden_dataset`) and by `POST /eval/run`. It is *not* what
the online agent loop should call at request time — for that, use
`app.evaluation.online_judge.OnlineJudge`, which is tuned for bounded
latency instead of running every sub-evaluator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from app.core.interfaces import Evaluator
from app.core.schemas import EvalResult, EvalSample
from app.evaluation.correctness import CorrectnessEvaluator
from app.evaluation.faithfulness import FaithfulnessEvaluator
from app.evaluation.groundedness import GroundednessEvaluator
from app.evaluation.relevance import RelevanceEvaluator


class CompositeEvaluator:
    """Implements the `Evaluator` protocol by fanning out to sub-evaluators."""

    def __init__(self, evaluators: list[Evaluator] | None = None) -> None:
        self._evaluators: list[Evaluator] = evaluators or [
            FaithfulnessEvaluator(),
            GroundednessEvaluator(),
            RelevanceEvaluator(),
            CorrectnessEvaluator(),
        ]

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        results = await asyncio.gather(*(e.evaluate(sample) for e in self._evaluators))

        faithfulness = _first_nonzero(r.faithfulness for r in results)
        groundedness = _first_nonzero(r.groundedness for r in results)
        relevance = _first_nonzero(r.relevance for r in results)
        correctness_values = [r.correctness for r in results if r.correctness is not None]
        correctness = correctness_values[0] if correctness_values else None

        details: dict[str, object] = {"sub_results": [r.details for r in results]}
        return EvalResult(
            faithfulness=faithfulness,
            groundedness=groundedness,
            relevance=relevance,
            correctness=correctness,
            details=details,
        )


def _first_nonzero(values: Iterable[float]) -> float:
    for v in values:
        if v:
            return v
    return 0.0
