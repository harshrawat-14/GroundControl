from __future__ import annotations

import hashlib

from app.core.schemas import RawDocument
from app.ingestion.normalizer import Normalizer


def test_content_hash_is_sha256_of_raw_bytes():
    content = b"hello world"
    doc = RawDocument(source_type="web", source_uri="https://example.com", content=content)
    normalized = Normalizer().normalize(doc)
    assert normalized.content_hash == hashlib.sha256(content).hexdigest()


def test_same_content_yields_same_document_id_idempotent():
    content = b"identical bytes"
    doc_a = RawDocument(source_type="web", source_uri="https://a.example.com", content=content)
    doc_b = RawDocument(source_type="web", source_uri="https://b.example.com", content=content)
    normalized_a = Normalizer().normalize(doc_a)
    normalized_b = Normalizer().normalize(doc_b)
    assert normalized_a.document_id == normalized_b.document_id
    assert normalized_a.content_hash == normalized_b.content_hash


def test_web_normalization_extracts_headings_and_strips_scripts():
    html = b"""
    <html><head><title>T</title></head>
    <body>
      <h1>Main Heading</h1>
      <script>alert('x')</script>
      <p>Some body paragraph.</p>
    </body></html>
    """
    doc = RawDocument(source_type="web", source_uri="https://example.com", content=html)
    normalized = Normalizer().normalize(doc)
    assert "alert" not in normalized.text
    assert "Some body paragraph." in normalized.text
    headings = normalized.structure.get("headings", [])
    assert any(h["text"] == "Main Heading" for h in headings)
