from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.schemas import QueryRequest, QueryResponse
from app.db.session import get_session
from app.gateway.gateway import AIGateway
from app.gateway.providers.bedrock import BedrockProvider
from app.gateway.providers.ollama import OllamaProvider
from app.gateway.providers.vertex import VertexProvider
from app.gateway.routing import default_policy
from app.gateway.semantic_cache import InMemorySemanticCache
from app.ingestion.embedder import LocalEmbedder
from app.orchestrator.query_service import AgenticQueryService

router = APIRouter(prefix="/query", tags=["query"])

# Reused across requests: reloading the embedding model or rebuilding the
# gateway's routing policy / semantic cache per call would defeat the point
# of both (model load cost, cache hit rate).
_embedder = LocalEmbedder()


def _build_gateway() -> AIGateway:
    settings = get_settings()
    ollama = OllamaProvider(settings=settings)
    bedrock = BedrockProvider(settings=settings) if settings.cloud_provider == "bedrock" else None
    vertex = VertexProvider(settings=settings) if settings.cloud_provider == "vertex" else None
    policy = default_policy(ollama, bedrock, vertex)
    cache = InMemorySemanticCache(embed_fn=lambda text: _embedder.embed([text])[0])
    return AIGateway(routing_policy=policy, semantic_cache=cache)


_gateway = _build_gateway()


@router.post("", response_model=QueryResponse)
async def run_query(
    req: QueryRequest, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> QueryResponse:
    service = AgenticQueryService(session=session, embedder=_embedder, gateway=_gateway)
    filters = (req.options.filters if req.options else None) or req.filters
    max_iterations = req.options.max_iterations if req.options else None
    return await service.run(req.question, filters=filters, max_iterations=max_iterations)
