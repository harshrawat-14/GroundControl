"""CorrectnessEvaluator: does the answer match a known ground-truth answer?

Offline-only metric (LLD 2.5: "correctness (offline, against a golden
set)") — requires `EvalSample.ground_truth`, which is only populated for
golden-dataset samples, not for live production queries. Ragas equivalent:
`answer_correctness`, which blends factual-overlap (via claim extraction)
and semantic similarity. Here we approximate both with a single TF-IDF
cosine similarity between the answer and the ground truth.

Upgrade path: swap for ragas' `answer_correctness`, which requires an LLM
judge to extract and compare atomic factual statements, once
OLLAMA_BASE_URL is reachable.
"""

from __future__ import annotations

from app.core.schemas import EvalResult, EvalSample
from app.evaluation.similarity import tfidf_cosine


class CorrectnessEvaluator:
    """Implements the `Evaluator` protocol (app.core.interfaces.Evaluator)."""

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        score: float | None
        if not sample.ground_truth:
            score = None
        else:
            score = tfidf_cosine(sample.answer, sample.ground_truth)
        return EvalResult(
            faithfulness=0.0,
            groundedness=0.0,
            relevance=0.0,
            correctness=score,
            details={"metric": "correctness", "method": "tfidf_answer_vs_ground_truth"},
        )
