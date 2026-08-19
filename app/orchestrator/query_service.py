"""Binds the AgentGraph's injectable collaborators to real infra: the
HybridRetriever, the AIGateway (draft task), and the OnlineJudge. This is
the thing `app/api/routes/query.py` actually calls.
"""

from __future__ import annotations

import time
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.interfaces import Embedder
from app.core.schemas import Filter, LLMRequest, QueryResponse, ScoredChunk
from app.db.models import RequestRow
from app.evaluation.online_judge import OnlineJudge
from app.gateway.gateway import AIGateway, AllProvidersFailedError
from app.orchestrator.graph import AgentGraph, to_query_response
from app.retrieval.hybrid import HybridRetriever

_SYSTEM_PROMPT = (
    "You are GroundControl, a grounded question-answering assistant. Answer the "
    "question using ONLY the numbered context passages below. Cite passages "
    "inline as [n]. If the context does not contain the answer, say you don't "
    "have enough information rather than guessing."
)


def _build_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{numbered}\n\nQuestion: {question}\n\nAnswer:"


class AgenticQueryService:
    """Runs the LangGraph self-correction loop for a single /query request."""

    def __init__(self, session: AsyncSession, embedder: Embedder, gateway: AIGateway) -> None:
        self._session = session
        self._embedder = embedder
        self._gateway = gateway
        self._judge = OnlineJudge()

    async def _retrieve(self, query: str, flt: Filter | None, k: int) -> list[ScoredChunk]:
        retriever = HybridRetriever(session=self._session, embedder=self._embedder, k=k)
        context = await retriever.retrieve(query, flt)
        return context.chunks

    async def _draft(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return ""
        prompt = _build_prompt(query, contexts)
        try:
            response = await self._gateway.complete(
                LLMRequest(task="draft", prompt=prompt, system=_SYSTEM_PROMPT)
            )
            self._last_cost = response.cost_usd
            self._last_cache_hit = response.cache_hit
            return response.text.strip()
        except AllProvidersFailedError:
            return ""

    async def run(
        self,
        question: str,
        filters: Filter | None = None,
        max_iterations: int | None = None,
    ) -> QueryResponse:
        start = time.perf_counter()
        trace_id = str(uuid4())
        settings = get_settings()

        self._last_cost = 0.0
        self._last_cache_hit = False

        graph = AgentGraph(retrieve=self._retrieve, draft=self._draft, judge=self._judge.judge)
        state = await graph.run(
            question, filters=filters, max_iterations=max_iterations or settings.max_iterations
        )
        if not state.get("draft_answer"):
            state["draft_answer"] = (
                "I don't have enough information in the indexed corpus to answer this question."
            )

        latency_ms = (time.perf_counter() - start) * 1000
        response = to_query_response(
            state,
            trace_id=trace_id,
            cost_usd=self._last_cost,
            latency_ms=latency_ms,
            cache_hit=self._last_cache_hit,
        )

        self._session.add(
            RequestRow(
                trace_id=trace_id,
                query=question,
                answer=response.answer,
                model_path={"provider": "gateway", "model_tier": state.get("model_tier", "local")},
                iterations=response.iterations,
                faithfulness=response.faithfulness,
                groundedness=response.groundedness,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                cache_hit=response.cache_hit,
            )
        )
        await self._session.commit()
        return response
