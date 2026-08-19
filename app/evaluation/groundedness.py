"""GroundednessEvaluator: how much of the answer, taken as a whole, overlaps
with the concatenated retrieved context.

This is deliberately a coarser, document-level counterpart to
`FaithfulnessEvaluator` (which scores per-sentence claim support): it is the
"embedding-overlap heuristic" the design doc names explicitly as the
online-judge signal (LLD 2.5). Ragas equivalent: closest to
`context_precision`/`context_recall` collapsed to a single answer-vs-context
overlap number when no separate reference context set is available.

Upgrade path: replace the TF-IDF cosine with real sentence embeddings
(`sentence-transformers`, already a dependency) once local embedding models
are downloaded, or with ragas' `context_precision`/`context_recall` metrics
against a golden reference context set for the offline benchmark.
"""

from __future__ import annotations

from app.core.schemas import EvalResult, EvalSample
from app.evaluation.similarity import tfidf_cosine


class GroundednessEvaluator:
    """Implements the `Evaluator` protocol (app.core.interfaces.Evaluator)."""

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        joined_context = "\n".join(sample.contexts)
        score = tfidf_cosine(sample.answer, joined_context)
        return EvalResult(
            faithfulness=0.0,
            groundedness=score,
            relevance=0.0,
            details={"metric": "groundedness", "method": "tfidf_answer_vs_context"},
        )
