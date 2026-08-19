"""Shared Pydantic data contracts used across ingestion, retrieval, the
agent orchestrator, the gateway, and the evaluation engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Vector = list[float]


class RawDocument(BaseModel):
    source_type: str
    source_uri: str
    content: bytes
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDoc(BaseModel):
    document_id: str
    source_type: str
    source_uri: str
    content_hash: str
    text: str
    structure: dict[str, Any] = Field(default_factory=dict)  # headings, tables, page map
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    document_id: str
    ordinal: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Vector | None = None


class ScoredChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Filter(BaseModel):
    source_types: list[str] | None = None
    document_ids: list[str] | None = None
    metadata_eq: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    query: str
    chunks: list[ScoredChunk]
    strategy: str = "hybrid"


class LLMRequest(BaseModel):
    task: Literal["embed", "draft", "judge"]
    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    cost_ceiling_usd: float | None = None


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool = False


class EvalSample(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class EvalResult(BaseModel):
    faithfulness: float
    groundedness: float
    relevance: float
    correctness: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(str):
    FINALIZE = "finalize"
    REFORMULATE = "reformulate"
    RETRIEVE_BROADEN = "retrieve_broaden"
    ESCALATE = "escalate"


class QueryOptions(BaseModel):
    max_iterations: int | None = None
    model_tier: str | None = None
    filters: Filter | None = None


class QueryRequest(BaseModel):
    question: str
    filters: Filter | None = None
    options: QueryOptions | None = None


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    faithfulness: float
    groundedness: float
    iterations: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    trace_id: str
    low_confidence: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
