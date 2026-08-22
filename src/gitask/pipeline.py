"""End-to-end orchestration (Milestone 4b).

Ties together the layers built in parallel:

    index_repo:  clone -> AST chunk -> embed (batched) -> upsert into pgvector
    ask:         hybrid retrieve -> grounded LLM answer with citations
    list_repos:  what has been indexed

Every heavy dependency (vector store, embedder, LLM, cloner, chunker) is either
dependency-injected or imported lazily *inside* the functions, so this module
imports cleanly even when sibling layers' native deps (tree-sitter, psycopg,
sentence-transformers, anthropic) are not installed — which is exactly what the
unit tests rely on.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Iterator, Optional

from gitask.config import settings
from gitask.models import Answer, Chunk, IndexResult, RetrievedChunk

# Chunks embedded per batch. Keeps memory bounded and lets a progress hook fire.
_EMBED_BATCH = 64


# --------------------------------------------------------------------------- #
# Default component factories (lazy — only imported when actually used)        #
# --------------------------------------------------------------------------- #
def _default_store() -> Any:
    from gitask.db.store import VectorStore

    store = VectorStore()
    store.init_schema()
    return store


def _default_embedder() -> Any:
    from gitask.embeddings.embedder import get_embedder

    return get_embedder()


def _default_llm() -> Any:
    from gitask.llm.client import LLMClient

    return LLMClient()


def _default_retriever(store: Any, embedder: Any) -> Any:
    from gitask.retrieval.hybrid import HybridRetriever

    return HybridRetriever(store, embedder)


# --------------------------------------------------------------------------- #
# Indexing                                                                     #
# --------------------------------------------------------------------------- #
def index_repo(
    url: str,
    *,
    store: Any = None,
    embedder: Any = None,
    batch_size: int = _EMBED_BATCH,
) -> IndexResult:
    """Clone, chunk, embed and store a repository, returning a summary.

    Prior chunks for the same repo are deleted first so re-indexing is clean
    (no stale duplicates). The repo identity is derived from the URL via
    ``repo_name_from_url`` (``"owner/name"``).
    """
    from gitask.ingestion.cloner import clone_repo, repo_name_from_url
    from gitask.ingestion.chunker import chunk_repo

    store = store if store is not None else _default_store()
    embedder = embedder if embedder is not None else _default_embedder()

    repo = repo_name_from_url(url)
    repo_path = clone_repo(url)

    # Clean re-index: drop anything we had for this repo.
    store.delete_repo(repo)

    files: set[str] = set()
    languages: Counter[str] = Counter()
    total_chunks = 0

    for batch in _batched(chunk_repo(repo_path, repo), batch_size):
        vectors = embedder.embed([c.embedding_text() for c in batch])
        store.upsert_chunks(batch, vectors)
        for c in batch:
            files.add(c.file_path)
            languages[c.language] += 1
            total_chunks += 1

    return IndexResult(
        repo=repo,
        files_indexed=len(files),
        chunks_indexed=total_chunks,
        languages=dict(languages),
    )


# --------------------------------------------------------------------------- #
# Asking                                                                       #
# --------------------------------------------------------------------------- #
def ask(
    question: str,
    repo: str,
    *,
    store: Any = None,
    embedder: Any = None,
    llm: Any = None,
    retriever: Any = None,
    k: Optional[int] = None,
) -> Answer:
    """Answer ``question`` about an already-indexed ``repo``.

    Runs hybrid retrieval, then grounded LLM synthesis. Retriever and LLM are
    injectable; by default they are built from the (also injectable) store and
    embedder.
    """
    if retriever is None:
        store = store if store is not None else _default_store()
        embedder = embedder if embedder is not None else _default_embedder()
        retriever = _default_retriever(store, embedder)

    llm = llm if llm is not None else _default_llm()

    retrieved: list[RetrievedChunk] = retriever.search(question, repo, k=k)
    return llm.answer(question, retrieved)


# --------------------------------------------------------------------------- #
# Repo listing                                                                 #
# --------------------------------------------------------------------------- #
def list_repos(*, store: Any = None) -> list[str]:
    """Return the repos currently indexed in the vector store."""
    store = store if store is not None else _default_store()
    return list(store.list_repos())


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _batched(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield successive lists of at most ``size`` items from an iterable."""
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
