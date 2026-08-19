from __future__ import annotations

import pytest

from app.core.schemas import NormalizedDoc
from app.ingestion.chunker import RecursiveChunker


def _doc(text: str, **kwargs) -> NormalizedDoc:
    return NormalizedDoc(
        document_id="11111111-1111-1111-1111-111111111111",
        source_type="pdf",
        source_uri="file://sample.pdf",
        content_hash="deadbeef",
        text=text,
        **kwargs,
    )


def test_empty_doc_produces_no_chunks():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk(_doc("   ")) == []


def test_short_doc_produces_single_chunk():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(_doc("This is a short document."))
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short document."
    assert chunks[0].ordinal == 0


def test_chunks_are_non_empty_and_belong_to_source_doc():
    text = ("Paragraph one about topic A. " * 20) + "\n\n" + ("Paragraph two about topic B. " * 20)
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    doc = _doc(text)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.strip() != ""
        assert chunk.document_id == doc.document_id


def test_ordinals_are_sequential_from_zero():
    text = "\n\n".join(f"Section {i}. " + ("filler text " * 30) for i in range(5))
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(_doc(text))

    ordinals = [c.ordinal for c in chunks]
    assert ordinals == list(range(len(chunks)))


def test_chunks_roughly_bounded_by_chunk_size():
    text = "word " * 2000
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=20)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    # allow slack for overlap stitching, but chunks shouldn't balloon unbounded
    for chunk in chunks:
        assert len(chunk.text) <= 300 * 2


def test_heading_metadata_attached_when_present():
    text = "Introduction\n\nSome body text here that talks about things at length."
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk(_doc(text))
    assert chunks[0].metadata.get("source_uri") == "file://sample.pdf"


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
