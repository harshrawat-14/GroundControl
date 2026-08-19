from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.interfaces import Embedder
from app.core.schemas import Filter, RetrievedContext, ScoredChunk
from app.db.models import ChunkRow, Document


class HybridRetriever:
    """Combines pgvector cosine search with Postgres full-text (tsvector)
    search, fused with Reciprocal Rank Fusion (RRF).

    Implements the Retriever protocol.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._k = k
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, flt: Filter | None = None) -> RetrievedContext:
        flt = flt or Filter()
        [qv] = self._embedder.embed([query])

        vector_hits = await self._vector_search(qv, flt)
        text_hits = await self._text_search(query, flt)

        fused = self._reciprocal_rank_fusion(vector_hits, text_hits)
        top = fused[: self._k]

        return RetrievedContext(query=query, chunks=top, strategy="hybrid")

    async def _vector_search(self, qv: list[float], flt: Filter) -> list[ScoredChunk]:
        stmt = (
            select(
                ChunkRow,
                (1 - ChunkRow.embedding.cosine_distance(qv)).label("score"),
            )
            .join(Document, Document.id == ChunkRow.document_id)
            .order_by(ChunkRow.embedding.cosine_distance(qv))
            .limit(self._k * 4)
        )
        stmt = self._apply_filter(stmt, flt)
        result = await self._session.execute(stmt)
        return [
            ScoredChunk(
                chunk_id=str(row.ChunkRow.id),
                document_id=str(row.ChunkRow.document_id),
                text=row.ChunkRow.text,
                score=float(row.score),
                metadata=row.ChunkRow.metadata_,
            )
            for row in result.all()
        ]

    async def _text_search(self, query: str, flt: Filter) -> list[ScoredChunk]:
        ts_query = func.plainto_tsquery("english", query)
        rank = func.ts_rank(ChunkRow.tsv, ts_query).label("score")
        stmt = (
            select(ChunkRow, rank)
            .join(Document, Document.id == ChunkRow.document_id)
            .where(ChunkRow.tsv.op("@@")(ts_query))
            .order_by(rank.desc())
            .limit(self._k * 4)
        )
        stmt = self._apply_filter(stmt, flt)
        result = await self._session.execute(stmt)
        return [
            ScoredChunk(
                chunk_id=str(row.ChunkRow.id),
                document_id=str(row.ChunkRow.document_id),
                text=row.ChunkRow.text,
                score=float(row.score),
                metadata=row.ChunkRow.metadata_,
            )
            for row in result.all()
        ]

    def _reciprocal_rank_fusion(
        self, vector_hits: list[ScoredChunk], text_hits: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, ScoredChunk] = {}

        for rank_list in (vector_hits, text_hits):
            for rank, chunk in enumerate(rank_list):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + rank + 1
                )
                chunks_by_id[chunk.chunk_id] = chunk

        ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        fused = []
        for cid in ordered_ids:
            base = chunks_by_id[cid]
            fused.append(
                ScoredChunk(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    text=base.text,
                    score=scores[cid],
                    metadata=base.metadata,
                )
            )
        return fused

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
