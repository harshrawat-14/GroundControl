from __future__ import annotations

import pytest

from app.core.schemas import EvalResult, ScoredChunk
from app.orchestrator.graph import AgentConfig, AgentGraph


def _chunk(i: int, text: str = "relevant passage") -> ScoredChunk:
    return ScoredChunk(chunk_id=f"c{i}", document_id="d1", text=text, score=0.9)


@pytest.mark.asyncio
async def test_single_pass_finalizes_when_grounded_and_faithful():
    async def retrieve(query, flt, k):
        return [_chunk(1), _chunk(2)]

    async def draft(query, contexts):
        return "a well grounded answer"

    async def judge(question, answer, contexts):
        return EvalResult(faithfulness=0.9, groundedness=0.9, relevance=0.9)

    graph = AgentGraph(retrieve=retrieve, draft=draft, judge=judge)
    state = await graph.run("what is the refund policy?")

    assert state["decision"] == "finalize"
    assert state["iteration"] == 1
    assert not state["low_confidence"]


@pytest.mark.asyncio
async def test_insufficient_context_triggers_broaden_then_finalizes():
    calls = {"retrieve": 0}

    async def retrieve(query, flt, k):
        calls["retrieve"] += 1
        # First call returns too few chunks; second call (after broaden) returns enough.
        return [_chunk(1)] if calls["retrieve"] == 1 else [_chunk(1), _chunk(2), _chunk(3)]

    async def draft(query, contexts):
        return "answer"

    async def judge(question, answer, contexts):
        # Too few chunks -> low-confidence scoring; enough chunks -> confident.
        grounded = len(contexts) >= 2
        return EvalResult(
            faithfulness=0.9 if grounded else 0.3,
            groundedness=0.9 if grounded else 0.3,
            relevance=0.9,
        )

    graph = AgentGraph(
        retrieve=retrieve,
        draft=draft,
        judge=judge,
        config=AgentConfig(min_chunks_for_sufficient_context=2),
    )
    state = await graph.run("underspecified question")

    assert calls["retrieve"] == 2
    assert state["decision"] == "finalize"
    assert state["k"] > AgentConfig().initial_k


@pytest.mark.asyncio
async def test_low_relevance_triggers_reformulate():
    async def retrieve(query, flt, k):
        return [_chunk(1), _chunk(2)]

    async def draft(query, contexts):
        return "answer"

    calls = {"judge": 0}

    async def judge(question, answer, contexts):
        calls["judge"] += 1
        if calls["judge"] == 1:
            return EvalResult(faithfulness=0.3, groundedness=0.3, relevance=0.1)
        return EvalResult(faithfulness=0.9, groundedness=0.9, relevance=0.9)

    graph = AgentGraph(retrieve=retrieve, draft=draft, judge=judge)
    state = await graph.run("vague query")

    assert calls["judge"] == 2
    assert state["decision"] == "finalize"
    assert state["query"] != "vague query"  # reformulated


@pytest.mark.asyncio
async def test_grounded_but_unfaithful_escalates_model_tier():
    async def retrieve(query, flt, k):
        return [_chunk(1), _chunk(2)]

    draft_calls = {"n": 0}

    async def draft(query, contexts):
        draft_calls["n"] += 1
        return f"answer-{draft_calls['n']}"

    judge_calls = {"n": 0}

    async def judge(question, answer, contexts):
        judge_calls["n"] += 1
        if judge_calls["n"] == 1:
            # grounded context (high relevance) but unfaithful draft
            return EvalResult(faithfulness=0.2, groundedness=0.9, relevance=0.9)
        return EvalResult(faithfulness=0.9, groundedness=0.9, relevance=0.9)

    graph = AgentGraph(retrieve=retrieve, draft=draft, judge=judge)
    state = await graph.run("tricky question")

    assert draft_calls["n"] == 2
    assert state["model_tier"] == "cloud"
    assert state["decision"] == "finalize"


@pytest.mark.asyncio
async def test_bounded_iterations_finalize_with_low_confidence():
    async def retrieve(query, flt, k):
        return [_chunk(1), _chunk(2)]

    async def draft(query, contexts):
        return "never good enough"

    async def judge(question, answer, contexts):
        return EvalResult(faithfulness=0.1, groundedness=0.9, relevance=0.9)

    graph = AgentGraph(retrieve=retrieve, draft=draft, judge=judge)
    state = await graph.run("hard question", max_iterations=2)

    assert state["iteration"] == 2
    assert state["decision"] == "finalize"
    assert state["low_confidence"] is True
