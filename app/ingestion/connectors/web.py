from __future__ import annotations

from collections.abc import Iterable

import httpx
from bs4 import BeautifulSoup

from app.core.schemas import RawDocument


class WebConnector:
    """Fetches one or more URLs over HTTP and yields RawDocuments.

    Implements the SourceConnector protocol.
    """

    def __init__(self, urls: list[str], timeout: float = 15.0) -> None:
        self._urls = urls
        self._timeout = timeout

    def fetch(self) -> Iterable[RawDocument]:
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            for url in self._urls:
                response = client.get(url)
                response.raise_for_status()
                content = response.content
                soup = BeautifulSoup(content, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                metadata = {"title": title, "url": url, "status_code": response.status_code}
                yield RawDocument(
                    source_type="web",
                    source_uri=url,
                    content=content,
                    metadata=metadata,
                )
