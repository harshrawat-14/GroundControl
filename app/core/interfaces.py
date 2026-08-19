"""Protocol interfaces for every cross-cutting capability.

Concrete implementations are injected, never imported directly by callers.
This keeps the pipeline open for extension (new source, embedder, provider,
metric) and closed for modification, and lets tests substitute fakes.
Interfaces are split by concern: a Chunker never sees provider credentials;
an Evaluator never touches the vector store.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.core.schemas import (
    Chunk,
    EvalResult,
    EvalSample,
    Filter,
    LLMRequest,
    LLMResponse,
    NormalizedDoc,
    RawDocument,
    RetrievedContext,
    ScoredChunk,
    Vector,
)


class SourceConnector(Protocol):
    def fetch(self) -> Iterable[RawDocument]: ...


class Normalizer(Protocol):
    def normalize(self, doc: RawDocument) -> NormalizedDoc: ...


class Chunker(Protocol):
    def chunk(self, doc: NormalizedDoc) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]: ...


class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk]) -> None: ...
    async def search(self, qv: Vector, flt: Filter, k: int) -> list[ScoredChunk]: ...
    async def delete(self, document_id: str) -> None: ...


class Retriever(Protocol):
    async def retrieve(self, query: str, flt: Filter) -> RetrievedContext: ...


class LLMProvider(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...


class Evaluator(Protocol):
    async def evaluate(self, sample: EvalSample) -> EvalResult: ...
