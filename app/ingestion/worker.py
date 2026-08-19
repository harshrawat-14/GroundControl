from __future__ import annotations

import uuid
from typing import Any, ClassVar

import structlog
from arq.connections import RedisSettings

from app.core.config import get_settings
from app.core.schemas import NormalizedDoc, RawDocument
from app.db.models import Document
from app.db.session import SessionLocal
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.connectors.pdf import PDFConnector
from app.ingestion.connectors.web import WebConnector
from app.ingestion.embedder import LocalEmbedder
from app.ingestion.normalizer import Normalizer
from app.retrieval.pgvector_store import PgVectorStore

logger = structlog.get_logger()

_CONNECTORS = {
    "pdf": lambda config: PDFConnector(paths=config["paths"]),
    "web": lambda config: WebConnector(urls=config["urls"]),
}


def build_connector(source_type: str, config: dict[str, Any]):
    try:
        factory = _CONNECTORS[source_type]
    except KeyError as exc:
        raise ValueError(f"unsupported source_type: {source_type}") from exc
    return factory(config)


async def run_ingest_job(ctx: dict[str, Any], source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """ARQ task: fetch -> normalize -> chunk -> embed -> upsert.

    Each source document is committed in its own transaction (per LLD 1.6 /
    2.6 data-integrity requirement); a failure on one document does not roll
    back documents already ingested in this job.
    """
    connector = build_connector(source_type, config)
    normalizer = Normalizer()
    chunker = RecursiveChunker()
    embedder: LocalEmbedder = ctx.get("embedder") or LocalEmbedder()

    ingested: list[str] = []
    failed: list[dict[str, str]] = []

    for raw in connector.fetch():
        try:
            document_id = await _ingest_one(raw, normalizer, chunker, embedder)
            ingested.append(document_id)
        except Exception as exc:  # noqa: BLE001 - keep the job going for other docs
            logger.error("ingest_document_failed", source_uri=raw.source_uri, error=str(exc))
            failed.append({"source_uri": raw.source_uri, "error": str(exc)})

    return {"ingested": ingested, "failed": failed}


async def _ingest_one(
    raw: RawDocument,
    normalizer: Normalizer,
    chunker,
    embedder: LocalEmbedder,
) -> str:
    normalized: NormalizedDoc = normalizer.normalize(raw)
    chunks = chunker.chunk(normalized)
    if not chunks:
        raise ValueError(f"no chunks produced for {raw.source_uri}")

    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    async with SessionLocal() as session:
        async with session.begin():
            existing = await session.get(Document, uuid.UUID(normalized.document_id))
            if existing is None:
                doc_row = Document(
                    id=uuid.UUID(normalized.document_id),
                    source_type=normalized.source_type,
                    source_uri=normalized.source_uri,
                    content_hash=normalized.content_hash,
                    metadata_=normalized.metadata,
                )
                session.add(doc_row)
            else:
                existing.version += 1
                existing.metadata_ = normalized.metadata

        store = PgVectorStore(session)
        await store.upsert(chunks)

    return normalized.document_id


class WorkerSettings:
    functions: ClassVar = [run_ingest_job]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
