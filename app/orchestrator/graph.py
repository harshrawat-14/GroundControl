"""LangGraph agent state machine — the self-correction loop (LLD 2.4).

    Plan -> Retrieve -> Draft -> Evaluate -> Decide
      Decide -> Finalize        (grounded and faithful >= thresholds)
      Decide -> Finalize        (iteration >= max: low-confidence / abstain)
      Decide -> Reformulate     (retrieval quality low) -> Retrieve
      Decide -> Retrieve        (insufficient context: broaden filters / raise k)
      Decide -> Escalate        (context good but draft unfaithful) -> Draft

Every collaborator (retriever, drafting LLM call, judge) is injected so the
graph itself has no hard dependency on Postgres, Ollama, or any specific
evaluator implementation — it only needs objects satisfying the narrow
callables declared below. This is what lets `tests/unit/test_agent_graph.py`
run the full loop against fakes with zero infra.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from langgraph.graph import END, StateGraph

from app.core.schemas import Citation, EvalResult, Filter, QueryResponse, ScoredChunk
from app.orchestrator.state import AgentState

RetrieveFn = Callable[[str, Filter | None, int], Awaitable[list[ScoredChunk]]]
DraftFn = Callable[[str, list[str]], Awaitable[str]]
JudgeFn = Callable[[str, str, list[str]], Awaitable[EvalResult]]
ReformulateFn = Callable[[str, int], str]


class SupportsAgentIO(Protocol):
    """Documents the three collaborators AgentGraph needs; concrete callers
    typically bind these to HybridRetriever.retrieve, AIGateway.complete
    (task="draft"), and OnlineJudge.judge respectively."""


def _default_reformulate(query: str, iteration: int) -> str:
    """Cheap, deterministic query broadening: strengthens recall by
    stripping the most restrictive punctuation/quoting and appending a
    generic recall hint. A real deployment could route this through the
    gateway's "draft" task instead; kept dependency-free here so the loop
    has zero extra network cost on the reformulate path.
    """
    stripped = query.strip().strip("?").strip()
    return f"{stripped} (broaden: include related and background context)"


@dataclass
class AgentConfig:
    faithfulness_threshold: float = 0.75
    groundedness_threshold: float = 0.75
    min_chunks_for_sufficient_context: int = 2
    max_iterations: int = 3
    initial_k: int = 8
    k_step: int = 4


class AgentGraph:
    """Builds and runs the compiled LangGraph state machine."""

    def __init__(
        self,
        retrieve: RetrieveFn,
        draft: DraftFn,
        judge: JudgeFn,
        reformulate: ReformulateFn = _default_reformulate,
        config: AgentConfig | None = None,
    ) -> None:
        self._retrieve = retrieve
        self._draft = draft
        self._judge = judge
        self._reformulate = reformulate
        self._config = config or AgentConfig()
        self._compiled = self._build().compile()

    # -- nodes ---------------------------------------------------------

    async def _node_plan(self, state: AgentState) -> AgentState:
        return {**state, "plan": f"answer '{state['query']}' from indexed corpus", "iteration": 0}

    async def _node_retrieve(self, state: AgentState) -> AgentState:
        k = state.get("k", self._config.initial_k)
        chunks = await self._retrieve(state["query"], state.get("filters"), k)
        return {**state, "retrieved_chunks": chunks, "k": k}

    async def _node_draft(self, state: AgentState) -> AgentState:
        texts = [c.text for c in state.get("retrieved_chunks", [])]
        answer = await self._draft(state["query"], texts)
        return {**state, "draft_answer": answer}

    async def _node_evaluate(self, state: AgentState) -> AgentState:
        texts = [c.text for c in state.get("retrieved_chunks", [])]
        result = await self._judge(state["query"], state.get("draft_answer", ""), texts)
        return {**state, "eval_result": result}

    async def _node_decide(self, state: AgentState) -> AgentState:
        cfg = self._config
        iteration = state.get("iteration", 0) + 1
        result = state.get("eval_result")
        num_chunks = len(state.get("retrieved_chunks", []))

        grounded_and_faithful = bool(
            result
            and result.groundedness >= cfg.groundedness_threshold
            and result.faithfulness >= cfg.faithfulness_threshold
        )

        if grounded_and_faithful or iteration >= state.get("max_iterations", cfg.max_iterations):
            decision = "finalize"
        elif num_chunks < cfg.min_chunks_for_sufficient_context:
            decision = "retrieve_broaden"
        elif result and result.relevance < 0.2:
            decision = "reformulate"
        else:
            decision = "escalate"

        return {
            **state,
            "iteration": iteration,
            "decision": decision,
            "low_confidence": decision == "finalize" and not grounded_and_faithful,
        }

    async def _node_reformulate(self, state: AgentState) -> AgentState:
        new_query = self._reformulate(state["query"], state.get("iteration", 0))
        return {**state, "query": new_query}

    async def _node_retrieve_broaden(self, state: AgentState) -> AgentState:
        new_k = state.get("k", self._config.initial_k) + self._config.k_step
        return {**state, "k": new_k}

    async def _node_escalate(self, state: AgentState) -> AgentState:
        return {**state, "model_tier": "cloud"}

    # -- routing ---------------------------------------------------------

    def _route_after_decide(self, state: AgentState) -> str:
        decision = state.get("decision", "finalize")
        return {
            "finalize": "finalize",
            "reformulate": "reformulate",
            "retrieve_broaden": "retrieve_broaden",
            "escalate": "escalate",
        }[decision]

    async def _node_finalize(self, state: AgentState) -> AgentState:
        return state

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._node_plan)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("draft", self._node_draft)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("decide", self._node_decide)
        graph.add_node("reformulate", self._node_reformulate)
        graph.add_node("retrieve_broaden", self._node_retrieve_broaden)
        graph.add_node("escalate", self._node_escalate)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "draft")
        graph.add_edge("draft", "evaluate")
        graph.add_edge("evaluate", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "finalize": "finalize",
                "reformulate": "reformulate",
                "retrieve_broaden": "retrieve_broaden",
                "escalate": "escalate",
            },
        )
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("retrieve_broaden", "retrieve")
        graph.add_edge("escalate", "draft")
        graph.add_edge("finalize", END)
        return graph

    async def run(
        self,
        query: str,
        filters: Filter | None = None,
        max_iterations: int = 3,
    ) -> AgentState:
        initial: AgentState = {
            "query": query,
            "filters": filters,
            "max_iterations": max_iterations,
            "model_tier": "local",
            "cost_accum": 0.0,
        }
        final_state = await self._compiled.ainvoke(initial)
        return final_state


def to_query_response(state: AgentState, trace_id: str, cost_usd: float, latency_ms: float, cache_hit: bool) -> QueryResponse:
    """Adapts the terminal AgentState into the API's QueryResponse contract."""
    result = state.get("eval_result") or EvalResult(faithfulness=0.0, groundedness=0.0, relevance=0.0)
    citations = [
        Citation(document_id=c.document_id, chunk_id=c.chunk_id, text=c.text, score=c.score)
        for c in state.get("retrieved_chunks", [])
    ]
    confidence = min(result.faithfulness, result.groundedness) if result else 0.0
    return QueryResponse(
        answer=state.get("draft_answer", ""),
        citations=citations,
        confidence=confidence,
        faithfulness=result.faithfulness,
        groundedness=result.groundedness,
        iterations=state.get("iteration", 1),
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        trace_id=trace_id,
        low_confidence=state.get("low_confidence", False),
    )
