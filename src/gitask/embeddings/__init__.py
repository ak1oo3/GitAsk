"""Embedding layer for GitAsk (Milestone 2).

Exposes the :class:`Embedder` protocol, the concrete
:class:`LocalEmbedder` / :class:`OpenAIEmbedder` implementations, and the
:func:`get_embedder` factory that selects one based on ``config.settings``.
"""

from gitask.embeddings.embedder import (
    Embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    get_embedder,
)

__all__ = ["Embedder", "LocalEmbedder", "OpenAIEmbedder", "get_embedder"]
