"""SemanticCache: Redis exact-match fast path + pgvector similarity search.

Per LLD 2.3 / 2.6: embed the prompt, search the `cache` table for cosine
similarity >= threshold; on hit, return the stored response and mark
`cache_hit`. TTL plus invalidation via a per-namespace `version` bumped on
corpus change. A Redis-backed exact-match (`prompt_hash`) lookup is checked
before the vector similarity search.

Deliberately depends only on `app.core.schemas` / `app.db.models` — not on
the M1 ingestion embedder module — via a minimal local `EmbedFn` type, to
avoid coupling to code another agent is actively building.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import LLMResponse
from app.db.models import CacheRow

# Minimal local embedder contract — avoids importing app.ingestion.*
EmbedFn = Callable[[str], list[float] | Awaitable[list[float]]]


class RedisLike(Protocol):
    """Minimal subset of redis.asyncio.Redis used by SemanticCache."""

    async def get(self, key: str) -> bytes | str | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...


def _prompt_hash(namespace: str, model: str, prompt: str) -> str:
    payload = f"{namespace}:{model}:{prompt}".encode()
    return hashlib.sha256(payload).hexdigest()


async def _maybe_await(value: list[float] | Awaitable[list[float]]) -> list[float]:
    if hasattr(value, "__await__"):
        return await value  # type: ignore[return-value]
    return value  # type: ignore[return-value]


class SemanticCache:
    """Two-tier cache: Redis exact match, then pgvector cosine similarity."""

    def __init__(
        self,
        session: AsyncSession,
        embed_fn: EmbedFn,
        redis_client: RedisLike | None = None,
        similarity_threshold: float = 0.95,
        default_ttl_seconds: int = 86400,
    ) -> None:
        self._session = session
        self._embed_fn = embed_fn
        self._redis = redis_client
        self._threshold = similarity_threshold
        self._default_ttl = default_ttl_seconds
        # In-memory namespace -> version map (CacheRow lacks a namespace
        # column, so version bumps are tracked here and stamped onto rows).
        self._namespace_versions: dict[str, int] = {}

    def bump_namespace(self, namespace: str) -> int:
        """Invalidate all cache entries for a namespace by bumping its version."""
        new_version = self._namespace_versions.get(namespace, 1) + 1
        self._namespace_versions[namespace] = new_version
        return new_version

    def _version_for(self, namespace: str) -> int:
        return self._namespace_versions.get(namespace, 1)

    async def get(
        self, prompt: str, model: str, namespace: str = "default"
    ) -> LLMResponse | None:
        phash = _prompt_hash(namespace, model, prompt)
        current_version = self._version_for(namespace)

        # Fast path: Redis exact match.
        if self._redis is not None:
            raw = await self._redis.get(f"gc:cache:{phash}")
            if raw is not None:
                data = json.loads(raw)
                if data.get("namespace_version") == current_version:
                    resp = LLMResponse(**data["response"])
                    resp.cache_hit = True
                    return resp

        # Slow path: pgvector cosine similarity search.
        query_vec = await _maybe_await(self._embed_fn(prompt))
        stmt = (
            select(CacheRow)
            .where(CacheRow.model == model, CacheRow.namespace_version == current_version)
            .order_by(CacheRow.prompt_embedding.cosine_distance(query_vec))
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None

        # cosine_distance = 1 - cosine_similarity
        similarity = 1.0 - self._cosine_distance(query_vec, row.prompt_embedding)
        if similarity < self._threshold:
            return None

        resp = LLMResponse(**row.response)
        resp.cache_hit = True

        if self._redis is not None:
            await self._redis.set(
                f"gc:cache:{phash}",
                json.dumps({"namespace_version": current_version, "response": resp.model_dump()}),
                ex=row.ttl or self._default_ttl,
            )
        return resp

    async def set(
        self,
        prompt: str,
        model: str,
        response: LLMResponse,
        namespace: str = "default",
        ttl_seconds: int | None = None,
    ) -> None:
        phash = _prompt_hash(namespace, model, prompt)
        current_version = self._version_for(namespace)
        embedding = await _maybe_await(self._embed_fn(prompt))
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        row = CacheRow(
            prompt_hash=phash,
            prompt_embedding=embedding,
            response=response.model_dump(),
            model=model,
            namespace_version=current_version,
            ttl=ttl,
        )
        self._session.add(row)
        await self._session.flush()

        if self._redis is not None:
            await self._redis.set(
                f"gc:cache:{phash}",
                json.dumps({"namespace_version": current_version, "response": response.model_dump()}),
                ex=ttl,
            )

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity


class InMemorySemanticCache:
    """A DB-free stand-in for SemanticCache, used by unit tests and any
    caller that wants semantic caching without live Postgres/Redis.

    Implements the same `get`/`set`/`bump_namespace` surface.
    """

    def __init__(self, embed_fn: EmbedFn, similarity_threshold: float = 0.95) -> None:
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._namespace_versions: dict[str, int] = {}

    def bump_namespace(self, namespace: str) -> int:
        new_version = self._namespace_versions.get(namespace, 1) + 1
        self._namespace_versions[namespace] = new_version
        return new_version

    def _version_for(self, namespace: str) -> int:
        return self._namespace_versions.get(namespace, 1)

    async def get(self, prompt: str, model: str, namespace: str = "default") -> LLMResponse | None:
        key = f"{namespace}:{model}"
        current_version = self._version_for(namespace)
        entries = self._store.get(key, [])

        query_vec = await _maybe_await(self._embed_fn(prompt))
        best_sim = -1.0
        best_entry = None
        for entry in entries:
            if entry["namespace_version"] != current_version:
                continue
            sim = self._cosine_similarity(query_vec, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry is None or best_sim < self._threshold:
            return None
        resp = LLMResponse(**best_entry["response"])
        resp.cache_hit = True
        return resp

    async def set(
        self,
        prompt: str,
        model: str,
        response: LLMResponse,
        namespace: str = "default",
        ttl_seconds: int | None = None,
    ) -> None:
        key = f"{namespace}:{model}"
        embedding = await _maybe_await(self._embed_fn(prompt))
        self._store.setdefault(key, []).append(
            {
                "embedding": embedding,
                "response": response.model_dump(),
                "namespace_version": self._version_for(namespace),
                "created_at": time.time(),
            }
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
