"""RelevanceEvaluator: does the answer actually address the question asked?

Ragas equivalent: `answer_relevancy`, which normally reverse-generates
candidate questions from the answer via an LLM and compares them to the
original question via embedding similarity. Here we directly compute the
TF-IDF cosine similarity between the answer and the question, which is a
correct, local, questionless approximation of the same signal (answers that
are off-topic or generic score low; answers that echo the question's
vocabulary in a substantive way score high).

Upgrade path: reintroduce the reverse-question-generation step through an
LLMProvider (Ollama) and swap TF-IDF for sentence embeddings once live
infra is reachable.
"""

from __future__ import annotations

from app.core.schemas import EvalResult, EvalSample
from app.evaluation.similarity import tfidf_cosine


class RelevanceEvaluator:
    """Implements the `Evaluator` protocol (app.core.interfaces.Evaluator)."""

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        score = tfidf_cosine(sample.answer, sample.question)
        return EvalResult(
            faithfulness=0.0,
            groundedness=0.0,
            relevance=score,
            details={"metric": "relevance", "method": "tfidf_answer_vs_question"},
        )
