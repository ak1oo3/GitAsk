"""Tests for the embedding factory and backends.

These must NOT require a network download. We always test the factory wiring
and the ``.dim`` contract; if ``sentence-transformers`` is installed AND the
model loads (cached or downloadable), we additionally assert a real embed
returns vectors of the right shape. Otherwise the real-model path is skipped.
"""

from __future__ import annotations

import pytest

from gitask.config import settings
from gitask.embeddings import (
    Embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    get_embedder,
)


def test_factory_returns_configured_type():
    """get_embedder() honours settings.embedding_provider."""
    provider = settings.embedding_provider.lower()
    if provider == "local":
        try:
            emb = get_embedder()
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        except Exception as exc:  # model download/load failure
            pytest.skip(f"local model unavailable: {exc}")
        assert isinstance(emb, LocalEmbedder)
    elif provider == "openai":
        # Instantiation needs an API key; just assert the branch is selected
        # without performing a network call.
        if not settings.openai_api_key:
            pytest.skip("openai provider configured but no API key set")
        emb = get_embedder()
        assert isinstance(emb, OpenAIEmbedder)
    else:
        pytest.fail(f"unexpected provider {provider!r}")


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "bogus")
    with pytest.raises(ValueError):
        get_embedder()


def test_openai_embedder_requires_key(monkeypatch):
    """OpenAIEmbedder raises clearly when no key is available."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    pytest.importorskip("openai", reason="openai not installed")
    with pytest.raises(ValueError):
        OpenAIEmbedder()


def test_embedder_protocol_surface():
    """A stub satisfying the surface is a structural Embedder."""

    class StubEmbedder:
        dim = settings.embedding_dim

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

        def embed_query(self, text):
            return [0.0] * self.dim

    stub = StubEmbedder()
    assert isinstance(stub, Embedder)
    assert stub.dim == settings.embedding_dim
    vecs = stub.embed(["a", "b"])
    assert len(vecs) == 2
    assert all(len(v) == settings.embedding_dim for v in vecs)
    assert len(stub.embed_query("q")) == settings.embedding_dim


def test_local_embedder_real_model_if_available():
    """If sentence-transformers + the model are usable, check dims & shape.

    Skips (does not fail) when the package is missing or the model cannot be
    loaded (e.g. no network on a cold cache).
    """
    pytest.importorskip(
        "sentence_transformers", reason="sentence-transformers not installed"
    )
    try:
        emb = LocalEmbedder()
    except Exception as exc:  # model download/load failure -> skip, not fail
        pytest.skip(f"local model unavailable: {exc}")

    assert emb.dim == settings.embedding_dim

    docs = emb.embed(["def login_user(): pass", "hello world"])
    assert len(docs) == 2
    assert all(len(v) == emb.dim for v in docs)
    assert all(isinstance(x, float) for x in docs[0])

    q = emb.embed_query("how do users log in")
    assert len(q) == emb.dim
