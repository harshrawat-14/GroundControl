from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import Chunk, Filter, ScoredChunk, Vector
from app.db.models import ChunkRow, Document


class PgVectorStore:
    """VectorStore implementation backed by Postgres + pgvector.

    Upserts are transactional per document and idempotent: chunks belonging
    to a document are keyed by (document_id, ordinal), so re-ingesting the
    same content_hash replaces prior chunks for that document rather than
    duplicating them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        document_id = uuid.UUID(chunks[0].document_id)
        ctx = self._session.begin_nested() if self._session.in_transaction() else self._session.begin()
        async with ctx:
            await self._session.execute(
                delete(ChunkRow).where(ChunkRow.document_id == document_id)
            )
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError("chunk.embedding must be set before upsert")
                row = ChunkRow(
                    document_id=document_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    metadata_=chunk.metadata,
                    embedding=chunk.embedding,
                    tsv=func.to_tsvector("english", chunk.text),
                )
                self._session.add(row)
        await self._session.commit()

    async def search(self, qv: Vector, flt: Filter, k: int) -> list[ScoredChunk]:
        stmt = (
            select(
                ChunkRow,
                (1 - ChunkRow.embedding.cosine_distance(qv)).label("score"),
            )
            .join(Document, Document.id == ChunkRow.document_id)
            .order_by(ChunkRow.embedding.cosine_distance(qv))
            .limit(k)
        )
        stmt = self._apply_filter(stmt, flt)

        result = await self._session.execute(stmt)
        rows = result.all()
        return [
            ScoredChunk(
                chunk_id=str(row.ChunkRow.id),
                document_id=str(row.ChunkRow.document_id),
                text=row.ChunkRow.text,
                score=float(row.score),
                metadata=row.ChunkRow.metadata_,
            )
            for row in rows
        ]

    async def delete(self, document_id: str) -> None:
        await self._session.execute(
            delete(ChunkRow).where(ChunkRow.document_id == uuid.UUID(document_id))
        )
        await self._session.commit()

    @staticmethod
    def _apply_filter(stmt, flt: Filter):
        if flt is None:
            return stmt
        if flt.source_types:
            stmt = stmt.where(Document.source_type.in_(flt.source_types))
        if flt.document_ids:
            stmt = stmt.where(
                ChunkRow.document_id.in_([uuid.UUID(d) for d in flt.document_ids])
            )
        for key, value in (flt.metadata_eq or {}).items():
            stmt = stmt.where(ChunkRow.metadata_[key].astext == str(value))
        return stmt
