from __future__ import annotations

import hashlib
import uuid

from bs4 import BeautifulSoup

from app.core.schemas import NormalizedDoc, RawDocument

# Stable namespace so document_id is deterministic for identical content, which
# keeps re-ingestion idempotent alongside the content_hash unique constraint.
_DOC_NAMESPACE = uuid.UUID("6f6d6f8a-4d0a-4a6a-9c8e-2b6a2b6a2b6a")


class Normalizer:
    """Turns a RawDocument into a canonical NormalizedDoc.

    Produces a sha256 content_hash used for idempotent upserts, and extracts
    lightweight structure (headings / page map) so the chunker can preserve
    document layout where feasible.
    """

    def normalize(self, doc: RawDocument) -> NormalizedDoc:
        content_hash = hashlib.sha256(doc.content).hexdigest()
        document_id = str(uuid.uuid5(_DOC_NAMESPACE, content_hash))

        if doc.source_type == "pdf":
            text, structure = self._normalize_pdf(doc)
        elif doc.source_type == "web":
            text, structure = self._normalize_web(doc)
        else:
            text = doc.content.decode("utf-8", errors="ignore")
            structure = {}

        return NormalizedDoc(
            document_id=document_id,
            source_type=doc.source_type,
            source_uri=doc.source_uri,
            content_hash=content_hash,
            text=text,
            structure=structure,
            metadata=doc.metadata,
        )

    @staticmethod
    def _normalize_pdf(doc: RawDocument) -> tuple[str, dict]:
        page_texts: list[str] = doc.metadata.get("page_texts") or []
        text = "\n\n".join(page_texts)
        structure = {"pages": [{"index": i, "chars": len(t)} for i, t in enumerate(page_texts)]}
        return text, structure

    @staticmethod
    def _normalize_web(doc: RawDocument) -> tuple[str, dict]:
        soup = BeautifulSoup(doc.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        headings = []
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                headings.append({"level": level, "text": h.get_text(strip=True)})

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        structure = {"headings": headings}
        return text, structure
