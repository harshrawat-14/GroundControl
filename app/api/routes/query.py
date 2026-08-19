from __future__ import annotations

import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.schemas import Citation, LLMRequest, QueryRequest, QueryResponse
from app.db.models import RequestRow
from app.db.session import get_session
from app.gateway.gateway import AIGateway, AllProvidersFailedError
from app.gateway.providers.bedrock import BedrockProvider
from app.gateway.providers.ollama import OllamaProvider
from app.gateway.providers.vertex import VertexProvider
from app.gateway.routing import default_policy
from app.gateway.semantic_cache import InMemorySemanticCache
from app.ingestion.embedder import LocalEmbedder
from app.retrieval.hybrid import HybridRetriever

logger = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])

_SYSTEM_PROMPT = (
    "You are GroundControl, a grounded question-answering assistant. Answer the "
    "question using ONLY the numbered context passages below. Cite passages "
    "inline as [n]. If the context does not contain the answer, say you don't "
    "have enough information rather than guessing."
)

# Reused across requests to avoid reloading the embedding model per call.
_embedder = LocalEmbedder()


def _build_gateway() -> AIGateway:
    settings = get_settings()
    ollama = OllamaProvider(settings=settings)
    bedrock = BedrockProvider(settings=settings) if settings.cloud_provider == "bedrock" else None
    vertex = VertexProvider(settings=settings) if settings.cloud_provider == "vertex" else None
    policy = default_policy(ollama, bedrock, vertex)
    cache = InMemorySemanticCache(embed_fn=lambda text: _embedder.embed([text])[0])
    return AIGateway(routing_policy=policy, semantic_cache=cache)


# One gateway per process: providers, routing policy, and the semantic cache
# all need to persist across requests to be useful.
_gateway = _build_gateway()


def _build_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{numbered}\n\nQuestion: {question}\n\nAnswer:"


def _heuristic_confidence(num_chunks: int, top_score: float, answer: str) -> float:
    if not answer or num_chunks == 0:
        return 0.0
    coverage = min(num_chunks / 4, 1.0)
    score_component = max(0.0, min(top_score, 1.0))
    return round(0.5 * coverage + 0.5 * score_component, 4)


@router.post("", response_model=QueryResponse)
async def run_query(
    req: QueryRequest, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> QueryResponse:
    start = time.perf_counter()
    trace_id = str(uuid4())

    retriever = HybridRetriever(session=session, embedder=_embedder)
    flt = (req.options.filters if req.options else None) or req.filters
    context = await retriever.retrieve(req.question, flt)

    citations = [
        Citation(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
        )
        for c in context.chunks
    ]

    cost_usd = 0.0
    cache_hit = False
    model_name = get_settings().ollama_model

    if context.chunks:
        prompt = _build_prompt(req.question, [c.text for c in context.chunks])
        try:
            llm_response = await _gateway.complete(
                LLMRequest(task="draft", prompt=prompt, system=_SYSTEM_PROMPT)
            )
            answer = llm_response.text.strip()
            cost_usd = llm_response.cost_usd
            cache_hit = llm_response.cache_hit
            model_name = llm_response.model
        except AllProvidersFailedError as exc:
            logger.warning("llm_draft_failed", error=str(exc))
            answer = ""
        if not answer:
            answer = "I don't have enough information to answer confidently right now."
    else:
        answer = "I don't have enough information in the indexed corpus to answer this question."

    top_score = context.chunks[0].score if context.chunks else 0.0
    confidence = _heuristic_confidence(len(context.chunks), top_score, answer)
    latency_ms = (time.perf_counter() - start) * 1000

    response = QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        faithfulness=0.0,  # wired to a real evaluator in M2's self-correction loop
        groundedness=0.0,  # wired to a real evaluator in M2's self-correction loop
        iterations=1,  # single-pass until the M2 self-correction loop
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        trace_id=trace_id,
        low_confidence=confidence < 0.4,
    )

    session.add(
        RequestRow(
            trace_id=trace_id,
            query=req.question,
            answer=answer,
            model_path={"provider": "gateway", "model": model_name},
            iterations=response.iterations,
            faithfulness=response.faithfulness,
            groundedness=response.groundedness,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            cache_hit=response.cache_hit,
        )
    )
    await session.commit()

    return response
