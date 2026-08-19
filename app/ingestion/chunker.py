from __future__ import annotations

import re
from itertools import pairwise
from typing import ClassVar

from app.core.schemas import Chunk, NormalizedDoc

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 \-/]{2,80})$")


class RecursiveChunker:
    """Metadata-aware recursive chunker.

    Splits text on a hierarchy of separators (double newline -> newline ->
    sentence -> word), keeping chunks within [chunk_size] characters with
    [chunk_overlap] carried between consecutive chunks. When a line looks
    like a heading, it is tracked and attached to subsequent chunks'
    metadata so downstream retrieval/citations keep some document structure.
    """

    _SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc: NormalizedDoc) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        current_heading = None
        raw_chunks = self._split(text, self._SEPARATORS)

        chunks: list[Chunk] = []
        ordinal = 0
        for piece in raw_chunks:
            piece = piece.strip()
            if not piece:
                continue
            first_line = piece.splitlines()[0].strip()
            if _HEADING_RE.match(first_line) and len(first_line) < 100:
                current_heading = first_line.lstrip("# ").strip()

            metadata = dict(doc.metadata)
            metadata["source_uri"] = doc.source_uri
            metadata["source_type"] = doc.source_type
            if current_heading:
                metadata["heading"] = current_heading

            chunks.append(
                Chunk(
                    document_id=doc.document_id,
                    ordinal=ordinal,
                    text=piece,
                    metadata=metadata,
                )
            )
            ordinal += 1

        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text into chunks <= chunk_size, with overlap."""
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) <= 1:
            return self._split(text, separators[1:])

        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) > self._chunk_size:
                    merged.extend(self._split(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            merged.append(current)

        return self._apply_overlap(merged)

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        if self._chunk_overlap == 0 or len(pieces) <= 1:
            return pieces
        result = [pieces[0]]
        for prev, piece in pairwise(pieces):
            tail = prev[-self._chunk_overlap :]
            combined = f"{tail} {piece}" if not tail.endswith(piece[:1]) else piece
            result.append(combined if len(combined) <= self._chunk_size * 1.2 else piece)
        return result

    def _hard_split(self, text: str) -> list[str]:
        step = self._chunk_size - self._chunk_overlap
        return [text[i : i + self._chunk_size] for i in range(0, len(text), step)]
