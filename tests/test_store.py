"""VectorStore tests -- require a live Postgres+pgvector.

Skip gracefully (never fail) when psycopg/pgvector aren't installed or no DB is
reachable at ``settings.database_url``, so the suite stays green in
environments without a database.
"""

from __future__ import annotations

import random
import uuid

import pytest

from gitask.config import settings
from gitask.models import Chunk

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")
pytest.importorskip("pgvector", reason="pgvector not installed")

from gitask.db.store import VectorStore  # noqa: E402


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason=f"no Postgres reachable at {settings.database_url}",
)


@pytest.fixture()
def store():
    s = VectorStore()
    s.init_schema()
    return s


def _rand_vec(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def _mk(repo, path, symbol, content):
    return Chunk(
        repo=repo,
        file_path=path,
        language="python",
        kind="function",
        content=content,
        start_line=1,
        end_line=3,
        symbol_name=symbol,
    )


def test_init_upsert_search_count_delete(store):
    repo = f"test/{uuid.uuid4().hex[:8]}"
    dim = settings.embedding_dim
    try:
        chunks = [
            _mk(repo, "a.py", "alpha", "def alpha(): return 1"),
            _mk(repo, "b.py", "beta", "def beta(): return 2"),
            _mk(repo, "c.py", "gamma", "def gamma(): return 3"),
        ]
        # Make one vector clearly closest to the query vector.
        target = [1.0] + [0.0] * (dim - 1)
        far1 = [0.0, 1.0] + [0.0] * (dim - 2)
        far2 = [0.0, 0.0, 1.0] + [0.0] * (dim - 3)
        vectors = [target, far1, far2]

        store.upsert_chunks(chunks, vectors)
        assert store.count(repo) == 3
        assert repo in store.list_repos()

        # Idempotent upsert (same ids) keeps the count stable.
        store.upsert_chunks(chunks, vectors)
        assert store.count(repo) == 3

        query = [0.99, 0.01] + [0.0] * (dim - 2)
        results = store.vector_search(query, repo, k=3)
        assert len(results) == 3
        # Closest chunk ranks first, with a high similarity score.
        assert results[0].chunk.symbol_name == "alpha"
        assert results[0].vector_score == pytest.approx(results[0].score)
        assert results[0].vector_score >= results[1].vector_score

        # Repo isolation: searching another repo yields nothing.
        assert store.vector_search(query, "test/nonexistent", k=3) == []

        # get_repo_chunks returns the corpus.
        corpus = store.get_repo_chunks(repo)
        assert {c.symbol_name for c in corpus} == {"alpha", "beta", "gamma"}
    finally:
        store.delete_repo(repo)
        assert store.count(repo) == 0
