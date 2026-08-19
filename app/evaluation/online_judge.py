"""OnlineJudge: the cheap, latency-bounded evaluator used by the agent
orchestrator's self-correction loop at request time (M2), per LLD 2.5:

    "The online gate uses a cheap judge (local model plus embedding-overlap
    heuristics) to keep per-request latency and cost bounded."

Public interface (stable — M2 depends on this signature):

    from app.evaluation.online_judge import OnlineJudge

    judge = OnlineJudge()
    result: EvalResult = await judge.judge(question, answer, contexts)

`judge()` returns an `EvalResult` (app.core.schemas.EvalResult) populated
with `faithfulness`, `groundedness`, and `relevance` (no `correctness` —
that metric requires a golden-set ground truth, which is not available at
request time). The call is wrapped in a bounded timeout
(`max_latency_s`, default 2.0s); on timeout it degrades gracefully by
returning a zeroed, `details={"degraded": True, "reason": "timeout"}`
result rather than raising, so the orchestrator's decision logic can treat
a judge timeout as "insufficient evidence -> remediate" without a crash.

This currently runs entirely on local TF-IDF cosine similarity (see
`app.evaluation.similarity`) so it has zero external dependencies and
sub-millisecond latency in this sandbox. The design doc's target
implementation additionally consults a local Ollama model
(`OllamaProvider`, task="judge") for a fast LLM opinion, blended with the
heuristic score. That LLM leg is stubbed via `_llm_provider` (optional
constructor arg, defaults to None / heuristic-only) so the M2 orchestrator
or M4 gateway integrator can inject a real `LLMProvider` later without
changing this class's public interface.
"""

from __future__ import annotations

import asyncio

from app.core.interfaces import LLMProvider
from app.core.schemas import EvalResult, EvalSample
from app.evaluation.similarity import mean_sentence_support, tfidf_cosine

_DEGRADED_RESULT = EvalResult(
    faithfulness=0.0,
    groundedness=0.0,
    relevance=0.0,
    details={"degraded": True, "reason": "timeout"},
)


class OnlineJudge:
    """Cheap judge for the runtime eval gate. See module docstring for the
    stable public interface (`judge`) that M2's orchestrator imports.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        max_latency_s: float = 2.0,
        support_threshold: float = 0.15,
    ) -> None:
        self._llm_provider = llm_provider  # reserved for a future Ollama-backed leg
        self._max_latency_s = max_latency_s
        self._threshold = support_threshold

    async def judge(self, question: str, answer: str, contexts: list[str]) -> EvalResult:
        """Score a single draft answer against its question and retrieved
        contexts. Bounded to `max_latency_s`; degrades to a zeroed result
        with `details.degraded = True` on timeout instead of raising, so
        the caller's decision logic always gets an EvalResult back.
        """
        try:
            return await asyncio.wait_for(
                self._score(question, answer, contexts), timeout=self._max_latency_s
            )
        except TimeoutError:
            return _DEGRADED_RESULT

    async def _score(self, question: str, answer: str, contexts: list[str]) -> EvalResult:
        faithfulness = mean_sentence_support(answer, contexts, self._threshold)
        groundedness = tfidf_cosine(answer, "\n".join(contexts))
        relevance = tfidf_cosine(answer, question)
        return EvalResult(
            faithfulness=faithfulness,
            groundedness=groundedness,
            relevance=relevance,
            details={"judge": "online", "method": "tfidf_heuristic"},
        )

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        """Adapter so OnlineJudge also satisfies the `Evaluator` protocol
        (app.core.interfaces.Evaluator) for callers that want to treat it
        polymorphically alongside the offline evaluators.
        """
        return await self.judge(sample.question, sample.answer, sample.contexts)
