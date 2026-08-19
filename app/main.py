from __future__ import annotations

import structlog
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import ingest, query as query_routes, eval as eval_routes, health
from app.core.config import get_settings

logger = structlog.get_logger()

settings = get_settings()

app = FastAPI(
    title="GroundControl",
    description="Self-correcting agentic RAG platform where evaluation metrics gate answers at request time.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query_routes.router)
app.include_router(eval_routes.router)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
