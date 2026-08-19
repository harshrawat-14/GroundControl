from __future__ import annotations

import time
from uuid import uuid4

import httpx
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.schemas import Citation, QueryRequest, QueryResponse
from app.db.models import RequestRow
from app.db.session import get_session
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


def _build_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{numbered}\n\nQuestion: {question}\n\nAnswer:"


async def _draft_answer(prompt: str) -> str:
    settings = get_settings()
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "system": _SYSTEM_PROMPT,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("llm_draft_failed", error=str(exc))
        return ""


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

    if context.chunks:
        prompt = _build_prompt(req.question, [c.text for c in context.chunks])
        answer = await _draft_answer(prompt)
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
        faithfulness=0.0,  # wired to a real evaluator in M3
        groundedness=0.0,  # wired to a real evaluator in M3
        iterations=1,  # single-pass until the M2 self-correction loop
        cost_usd=0.0,  # populated once the M4 gateway tracks token cost
        latency_ms=latency_ms,
        cache_hit=False,  # semantic cache lands in M4
        trace_id=trace_id,
        low_confidence=confidence < 0.4,
    )

    session.add(
        RequestRow(
            trace_id=trace_id,
            query=req.question,
            answer=answer,
            model_path={"provider": "ollama", "model": get_settings().ollama_model},
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
