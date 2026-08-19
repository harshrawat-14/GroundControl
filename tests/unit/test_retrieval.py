from __future__ import annotations

from app.core.schemas import ScoredChunk
from app.retrieval.hybrid import HybridRetriever


def _chunk(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, document_id="doc-1", text="t", score=score)


def test_rrf_prefers_items_ranked_highly_in_both_lists():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._rrf_k = 60

    vector_hits = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.5)]
    text_hits = [_chunk("b", 5.0), _chunk("a", 1.0), _chunk("d", 0.5)]

    fused = retriever._reciprocal_rank_fusion(vector_hits, text_hits)
    fused_ids = [c.chunk_id for c in fused]

    # "a" and "b" both appear near the top of each list, so they should
    # outrank "c" and "d", which each only appear in one list.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert fused_ids[-1] in {"c", "d"}


def test_rrf_scores_are_monotonically_non_increasing():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._rrf_k = 60

    vector_hits = [_chunk(str(i), 1.0 - i * 0.1) for i in range(5)]
    text_hits = [_chunk(str(i), 5.0 - i) for i in range(5)]

    fused = retriever._reciprocal_rank_fusion(vector_hits, text_hits)
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_handles_disjoint_lists():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._rrf_k = 60

    fused = retriever._reciprocal_rank_fusion([_chunk("a", 1.0)], [_chunk("b", 1.0)])
    assert {c.chunk_id for c in fused} == {"a", "b"}
