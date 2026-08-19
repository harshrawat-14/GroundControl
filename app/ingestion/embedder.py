from __future__ import annotations

from app.core.schemas import Vector


class LocalEmbedder:
    """Batched local embeddings via sentence-transformers.

    The model is loaded lazily on first use (and the import itself deferred)
    so importing this module never requires torch/sentence-transformers to be
    installed unless embeddings are actually needed (keeps unit tests for
    other pieces of the ingestion pipeline lightweight).
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        from app.core.config import get_settings

        self._model_name = model_name or get_settings().local_embed_model
        self._batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]
