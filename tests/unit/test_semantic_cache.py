from __future__ import annotations

from app.core.schemas import LLMResponse
from app.gateway.semantic_cache import InMemorySemanticCache

_VOCAB = {
    "hello": [1.0, 0.0, 0.0],
    "hi": [0.99, 0.05, 0.0],  # near-orthogonal-but-close to "hello"
    "goodbye": [0.0, 1.0, 0.0],
}


def fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding keyed on the first word of the prompt.

    Uses orthogonal basis vectors per distinct word so "similar" and
    "dissimilar" prompts produce predictably high/low cosine similarity.
    """
    first_word = text.split()[0].lower()
    return _VOCAB.get(first_word, [0.0, 0.0, 1.0])


def make_response(text: str = "answer") -> LLMResponse:
    return LLMResponse(
        text=text,
        provider="ollama",
        model="draft",
        prompt_tokens=5,
        completion_tokens=5,
        cost_usd=0.0,
        latency_ms=10.0,
    )


async def test_miss_when_cache_empty() -> None:
    cache = InMemorySemanticCache(fake_embed, similarity_threshold=0.9)
    result = await cache.get("hello world", "draft")
    assert result is None


async def test_hit_on_identical_prompt() -> None:
    cache = InMemorySemanticCache(fake_embed, similarity_threshold=0.9)
    resp = make_response("cached answer")
    await cache.set("hello world", "draft", resp)

    hit = await cache.get("hello world", "draft")
    assert hit is not None
    assert hit.cache_hit is True
    assert hit.text == "cached answer"


async def test_miss_on_dissimilar_prompt() -> None:
    cache = InMemorySemanticCache(fake_embed, similarity_threshold=0.9)
    await cache.set("hello world", "draft", make_response())

    # "goodbye" buckets to a different vector direction -> low cosine similarity.
    miss = await cache.get("goodbye friend", "draft")
    assert miss is None


async def test_namespace_bump_invalidates_old_entries() -> None:
    cache = InMemorySemanticCache(fake_embed, similarity_threshold=0.9)
    await cache.set("hello world", "draft", make_response(), namespace="docs")

    assert await cache.get("hello world", "draft", namespace="docs") is not None

    cache.bump_namespace("docs")
    assert await cache.get("hello world", "draft", namespace="docs") is None


async def test_cache_is_scoped_by_model() -> None:
    cache = InMemorySemanticCache(fake_embed, similarity_threshold=0.9)
    await cache.set("hello world", "draft", make_response())
    assert await cache.get("hello world", "judge") is None
