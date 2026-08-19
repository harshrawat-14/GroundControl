"""FaithfulnessEvaluator: are the answer's claims supported by the retrieved
context (as opposed to hallucinated)?

Ragas equivalent: `faithfulness` (claim decomposition + NLI entailment
against context, judged by an LLM). Here we approximate claim decomposition
by splitting the answer into sentences and checking each sentence's max
TF-IDF cosine similarity against the context chunks — see
`app.evaluation.similarity.mean_sentence_support` for the full rationale.

Upgrade to a live LLM judge (ragas.metrics.faithfulness backed by an
OllamaProvider) once OLLAMA_BASE_URL is reachable; the public interface
below does not need to change.
"""

from __future__ import annotations

from app.core.schemas import EvalResult, EvalSample
from app.evaluation.similarity import mean_sentence_support


class FaithfulnessEvaluator:
    """Implements the `Evaluator` protocol (app.core.interfaces.Evaluator)."""

    def __init__(self, support_threshold: float = 0.15) -> None:
        self._threshold = support_threshold

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        score = mean_sentence_support(sample.answer, sample.contexts, self._threshold)
        return EvalResult(
            faithfulness=score,
            groundedness=0.0,
            relevance=0.0,
            details={"metric": "faithfulness", "method": "tfidf_sentence_support"},
        )
