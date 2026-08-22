"""pgvector-backed vector store.

Wraps a single ``chunks`` table (one row per :class:`~gitask.models.Chunk`)
carrying a ``vector(dim)`` embedding column. Uses psycopg v3 with the
``pgvector.psycopg`` adapter so Python ``list[float]`` values round-trip to the
``vector`` type transparently.

Cosine distance (``<=>``) is used for search; the returned
``RetrievedChunk.vector_score`` is ``1 - distance`` so that higher means more
similar, which is what the fusion layer in Milestone 3 expects.
"""

from __future__ import annotations

from typing import Any

from gitask.config import settings
from gitask.models import Chunk, RetrievedChunk

# Columns in insert/select order (excluding the embedding, handled separately).
_CHUNK_COLUMNS = (
    "chunk_id",
    "repo",
    "file_path",
    "language",
    "kind",
    "content",
    "start_line",
    "end_line",
    "symbol_name",
    "parent_symbol",
)


class VectorStore:
    """A thin data-access object over the pgvector ``chunks`` table."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.database_url
        self.dim = settings.embedding_dim
        try:
            import psycopg  # noqa: F401
            from pgvector.psycopg import register_vector  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "VectorStore requires 'psycopg[binary]' and 'pgvector'. "
                "Install them with `pip install 'psycopg[binary]' pgvector`."
            ) from exc

    # ------------------------------------------------------------------ #
    # Connection helper                                                   #
    # ------------------------------------------------------------------ #
    def _connect(self):
        """Open a connection with the pgvector type adapter registered."""
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn)
        # The vector extension must exist before the adapter can register the
        # type OID; init_schema() creates it. register_vector is a no-op-safe
        # call once the extension is present.
        try:
            register_vector(conn)
        except Exception:
            # Extension not yet created (e.g. very first init_schema call).
            # init_schema handles its own registration after CREATE EXTENSION.
            pass
        return conn

    # ------------------------------------------------------------------ #
    # Schema                                                              #
    # ------------------------------------------------------------------ #
    def init_schema(self) -> None:
        """Create the extension, table and indexes if they don't exist."""
        import psycopg
        from pgvector.psycopg import register_vector

        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id      TEXT PRIMARY KEY,
            repo          TEXT NOT NULL,
            file_path     TEXT NOT NULL,
            language      TEXT NOT NULL,
            kind          TEXT NOT NULL,
            content       TEXT NOT NULL,
            start_line    INTEGER NOT NULL,
            end_line      INTEGER NOT NULL,
            symbol_name   TEXT,
            parent_symbol TEXT,
            embedding     vector({self.dim})
        );

        CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);

        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            # Now that the extension exists, register the adapter (harmless).
            try:
                register_vector(conn)
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #
    def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Insert or update chunks + their embeddings, batched.

        Rows are matched on ``chunk_id``; on conflict every column (including
        the embedding) is overwritten. ``chunks`` and ``vectors`` must be the
        same length and aligned by index.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) "
                "must have equal length"
            )
        if not chunks:
            return

        cols = ", ".join((*_CHUNK_COLUMNS, "embedding"))
        placeholders = ", ".join(["%s"] * (len(_CHUNK_COLUMNS) + 1))
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in (*_CHUNK_COLUMNS[1:], "embedding")  # skip PK chunk_id
        )
        sql = (
            f"INSERT INTO chunks ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
        )

        rows: list[tuple[Any, ...]] = []
        for chunk, vec in zip(chunks, vectors):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.repo,
                    chunk.file_path,
                    chunk.language,
                    chunk.kind,
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.symbol_name,
                    chunk.parent_symbol,
                    _to_vector(vec),
                )
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #
    def vector_search(
        self, query_vector: list[float], repo: str, k: int
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks in ``repo`` by cosine similarity to the query.

        ``vector_score`` and ``score`` are both set to ``1 - cosine_distance``
        (higher = more similar).
        """
        select_cols = ", ".join(_CHUNK_COLUMNS)
        sql = (
            f"SELECT {select_cols}, 1 - (embedding <=> %s) AS similarity "
            "FROM chunks WHERE repo = %s "
            "ORDER BY embedding <=> %s ASC LIMIT %s"
        )
        qv = _to_vector(query_vector)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (qv, repo, qv, k))
                fetched = cur.fetchall()

        results: list[RetrievedChunk] = []
        for row in fetched:
            *chunk_vals, similarity = row
            chunk = _row_to_chunk(chunk_vals)
            sim = float(similarity)
            results.append(
                RetrievedChunk(chunk=chunk, score=sim, vector_score=sim)
            )
        return results

    def get_repo_chunks(self, repo: str) -> list[Chunk]:
        """Return every chunk for ``repo`` (used as the BM25 corpus)."""
        select_cols = ", ".join(_CHUNK_COLUMNS)
        sql = f"SELECT {select_cols} FROM chunks WHERE repo = %s"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repo,))
                fetched = cur.fetchall()
        return [_row_to_chunk(list(row)) for row in fetched]

    def count(self, repo: str) -> int:
        """Number of chunks stored for ``repo``."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE repo = %s", (repo,)
                )
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_repos(self) -> list[str]:
        """Distinct repo identifiers present in the store, sorted."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT repo FROM chunks ORDER BY repo")
                fetched = cur.fetchall()
        return [row[0] for row in fetched]

    # ------------------------------------------------------------------ #
    # Deletes                                                             #
    # ------------------------------------------------------------------ #
    def delete_repo(self, repo: str) -> None:
        """Remove all chunks for ``repo``."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
            conn.commit()


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
def _to_vector(vec: list[float]):
    """Wrap the values in pgvector's ``Vector`` so they serialize to the
    ``vector`` type (a plain Python list would otherwise be sent as
    ``double precision[]``, which the ``<=>`` operator rejects)."""
    from pgvector import Vector

    try:
        import numpy as np

        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
    except ImportError:  # pragma: no cover
        pass
    return Vector(list(vec))


def _row_to_chunk(vals: list[Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from a row selected in ``_CHUNK_COLUMNS`` order."""
    data = dict(zip(_CHUNK_COLUMNS, vals))
    return Chunk.from_dict(data)
