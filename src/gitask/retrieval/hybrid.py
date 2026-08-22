"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF).

Dense (vector) search is great at semantic paraphrase but routinely misses
*exact code identifiers* -- a query of ``login_user`` may not surface the
``login_user`` function if the surrounding prose doesn't match. Sparse BM25
keyword search is the opposite: it nails exact tokens but is blind to meaning.
Fusing the two gives the best of both.

Pipeline (:meth:`HybridRetriever.search`):

1. **Dense:** pull ``settings.top_k_vector`` candidates from the vector store.
2. **Sparse:** build a BM25 index over the *whole repo's* chunks
   (``store.get_repo_chunks``) using code-aware tokenization, and score the
   query. Running over the full corpus -- not just the dense candidates -- is
   what lets BM25 *rescue* an exact-identifier chunk that dense search dropped
   entirely.
3. **Fuse:** combine the two rankings with Reciprocal Rank Fusion. RRF is
   scale-free (it uses ranks, not raw scores) so it needs no normalization and
   is robust to the very different score distributions of cosine similarity vs.
   BM25. Raw ``vector_score`` / ``keyword_score`` are retained on each result
   for display and eval; ``score`` holds the fused RRF value.

Returns the top ``k`` (default ``settings.top_k_final``) sorted by ``score``.
"""

from __future__ import annotations

from typing import Protocol

from gitask.config import settings
from gitask.models import Chunk, RetrievedChunk
from gitask.retrieval.tokenize import tokenize_code

# RRF dampening constant (Cormack et al. 2009). Larger => flatter contribution
# from top ranks; 60 is the widely used default.
RRF_K = 60


class _StoreLike(Protocol):
    """The slice of :class:`~gitask.db.store.VectorStore` this retriever uses.

    Declared as a Protocol so tests can pass a lightweight in-memory fake.
    """

    def vector_search(
        self, query_vector: list[float], repo: str, k: int
    ) -> list[RetrievedChunk]: ...

    def get_repo_chunks(self, repo: str) -> list[Chunk]: ...


class _EmbedderLike(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class HybridRetriever:
    """Combine vector search and BM25 keyword search via RRF."""

    def __init__(self, store: _StoreLike, embedder: _EmbedderLike) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self, question: str, repo: str, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return the top-``k`` fused results for ``question`` within ``repo``."""
        k = k or settings.top_k_final

        # --- (a) dense candidates -------------------------------------- #
        query_vector = self.embedder.embed_query(question)
        vector_hits = self.store.vector_search(
            query_vector, repo, settings.top_k_vector
        )

        # --- (b) sparse scores over the full repo corpus --------------- #
        corpus = self.store.get_repo_chunks(repo)
        keyword_scores = self._bm25_scores(question, corpus)

        # --- assemble a chunk registry & raw score maps ---------------- #
        chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in corpus}
        vector_by_id: dict[str, float] = {}
        for hit in vector_hits:
            cid = hit.chunk.chunk_id
            chunks_by_id.setdefault(cid, hit.chunk)
            vector_by_id[cid] = hit.vector_score

        # --- (c) Reciprocal Rank Fusion -------------------------------- #
        vector_rank = _ranks([(cid, s) for cid, s in vector_by_id.items()])
        keyword_rank = _ranks(
            [(cid, s) for cid, s in keyword_scores.items() if s > 0.0]
        )

        candidate_ids = set(vector_rank) | set(keyword_rank)
        fused: list[RetrievedChunk] = []
        for cid in candidate_ids:
            rrf = 0.0
            if cid in vector_rank:
                rrf += 1.0 / (RRF_K + vector_rank[cid])
            if cid in keyword_rank:
                rrf += 1.0 / (RRF_K + keyword_rank[cid])
            fused.append(
                RetrievedChunk(
                    chunk=chunks_by_id[cid],
                    score=rrf,
                    vector_score=vector_by_id.get(cid, 0.0),
                    keyword_score=keyword_scores.get(cid, 0.0),
                )
            )

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:k]

    # ------------------------------------------------------------------ #
    # BM25                                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bm25_scores(question: str, corpus: list[Chunk]) -> dict[str, float]:
        """Map ``chunk_id -> BM25 score`` for ``question`` over ``corpus``."""
        if not corpus:
            return {}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HybridRetriever's keyword stage requires 'rank-bm25'. "
                "Install it with `pip install rank-bm25`."
            ) from exc

        tokenized_corpus = [
            tokenize_code(c.embedding_text()) for c in corpus
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize_code(question)
        scores = bm25.get_scores(query_tokens)
        return {c.chunk_id: float(s) for c, s in zip(corpus, scores)}


def _ranks(scored: list[tuple[str, float]]) -> dict[str, int]:
    """Assign 1-based ranks (rank 1 = highest score) by descending score."""
    ordered = sorted(scored, key=lambda t: t[1], reverse=True)
    return {cid: i + 1 for i, (cid, _score) in enumerate(ordered)}
