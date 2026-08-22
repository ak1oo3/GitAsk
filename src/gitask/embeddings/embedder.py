"""Embedding backends and the factory that selects one.

Two implementations are provided:

* :class:`LocalEmbedder` -- sentence-transformers, default
  ``BAAI/bge-small-en-v1.5`` (384 dims). No API key required. For BGE models
  the *query* is prefixed with the model's recommended retrieval instruction so
  asymmetric query/document embeddings line up (documents are embedded as-is).
* :class:`OpenAIEmbedder` -- ``text-embedding-3-small`` (1536 dims) via the
  OpenAI API.

Heavy third-party imports (``sentence_transformers``, ``openai``) are deferred
to instantiation time, so importing this module never requires those packages
to be installed. Instantiating an implementation whose dependency is missing
raises a clear :class:`ImportError`.

Every embedder exposes ``.dim`` which MUST equal ``config.settings.embedding_dim``
so it matches the pgvector column width.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gitask.config import settings

# Recommended retrieval instruction for BGE v1.5 English models. The document
# side is embedded verbatim; only the query is prefixed.
_BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedding backend implements.

    Implementations return plain ``list[list[float]]`` / ``list[float]`` (not
    numpy arrays) so callers and the pgvector layer stay dependency-light.
    """

    #: Output vector dimension. Must equal ``config.settings.embedding_dim``.
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of *documents*. Returns one vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single *query* string. May apply an asymmetric prefix."""
        ...


class LocalEmbedder:
    """sentence-transformers embedder (default: ``BAAI/bge-small-en-v1.5``)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.local_embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise ImportError(
                "LocalEmbedder requires 'sentence-transformers'. "
                "Install it with `pip install sentence-transformers`, or set "
                "EMBEDDING_PROVIDER=openai."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        # Authoritative dimension comes from config so the DB column matches;
        # we sanity-check it against the loaded model.
        self.dim = settings.embedding_dim
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim != self.dim:
            raise ValueError(
                f"Model {self.model_name!r} produces {model_dim}-d vectors but "
                f"config.settings.embedding_dim is {self.dim}. Align "
                "LOCAL_EMBEDDING_MODEL / EMBEDDING_PROVIDER with the model."
            )
        self._is_bge = "bge" in self.model_name.lower()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        query = f"{_BGE_QUERY_INSTRUCTION}{text}" if self._is_bge else text
        vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vec.tolist()


class OpenAIEmbedder:
    """OpenAI embedder (default: ``text-embedding-3-small``, 1536 dims)."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.openai_embedding_model
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                "Install it with `pip install openai`."
            ) from exc

        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError(
                "OpenAIEmbedder requires OPENAI_API_KEY to be set."
            )
        self._client = OpenAI(api_key=key)
        self.dim = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self.model_name, input=texts
        )
        # API preserves input order.
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        # OpenAI models are symmetric: query and document use the same call.
        return self.embed([text])[0]


def get_embedder() -> Embedder:
    """Return the configured embedder based on ``settings.embedding_provider``.

    ``"local"`` (default) -> :class:`LocalEmbedder`;
    ``"openai"`` -> :class:`OpenAIEmbedder`.
    """
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(
        f"Unknown embedding_provider {settings.embedding_provider!r}; "
        "expected 'local' or 'openai'."
    )
