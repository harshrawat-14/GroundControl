from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader

from app.core.schemas import RawDocument


class PDFConnector:
    """Reads one or more local PDF files and yields RawDocuments.

    Implements the SourceConnector protocol.
    """

    def __init__(self, paths: list[str] | list[Path]) -> None:
        self._paths = [Path(p) for p in paths]

    def fetch(self) -> Iterable[RawDocument]:
        for path in self._paths:
            content = path.read_bytes()
            reader = PdfReader(str(path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            metadata = {
                "filename": path.name,
                "num_pages": len(reader.pages),
                "page_texts": page_texts,
            }
            yield RawDocument(
                source_type="pdf",
                source_uri=str(path),
                content=content,
                metadata=metadata,
            )
