"""AgentState: the state LangGraph threads through plan -> retrieve ->
draft -> evaluate -> decide, per LLD 2.4.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from app.core.schemas import EvalResult, Filter, ScoredChunk

Decision = Literal["finalize", "reformulate", "retrieve_broaden", "escalate"]


class AgentState(TypedDict, total=False):
    query: str
    plan: str
    filters: Filter | None
    k: int
    retrieved_chunks: list[ScoredChunk]
    draft_answer: str
    eval_result: EvalResult | None
    iteration: int
    max_iterations: int
    model_tier: Literal["local", "cloud"]
    cost_accum: float
    trace_id: str
    decision: Decision | None
    low_confidence: bool
