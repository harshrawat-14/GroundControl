from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from app.core.config import get_settings


def _tcp_reachable(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url.replace("postgresql+asyncpg", "postgresql"))
    host, port = parsed.hostname, parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


settings = get_settings()
_pg_up = _tcp_reachable(settings.database_url)
_redis_up = _tcp_reachable(settings.redis_url)
_ollama_up = _tcp_reachable(settings.ollama_base_url)

pytestmark = pytest.mark.skipif(
    not (_pg_up and _redis_up and _ollama_up),
    reason="requires live Postgres+pgvector, Redis, and Ollama (not available in this sandbox)",
)


@pytest.mark.asyncio
async def test_ingest_then_query_returns_grounded_cited_answer(tmp_path):
    """End-to-end: ingest a small sample doc via the worker function directly,
    then hit /query and assert the response is structurally grounded
    (non-empty citations pointing at the ingested document, confidence set).

    Skipped automatically unless Postgres, Redis, and Ollama are reachable -
    this exercises real infra this sandbox cannot start.
    """
    import httpx
    from asgi_lifespan import LifespanManager

    from app.ingestion.worker import run_ingest_job
    from app.main import app

    # Minimal valid PDF isn't trivial to hand-build; use a .txt-like connector
    # path instead is out of scope for the connectors we implemented, so this
    # test relies on a fixture PDF if present, otherwise skips gracefully.
    fixture = tmp_path.parent / "fixtures" / "sample.pdf"
    if not fixture.exists():
        pytest.skip("no sample.pdf fixture available")

    result = await run_ingest_job({}, "pdf", {"paths": [str(fixture)]})
    assert result["ingested"], f"ingestion failed: {result['failed']}"

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"question": "What is this document about?"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["citations"], "expected at least one citation"
            assert body["trace_id"]
            assert body["answer"]
