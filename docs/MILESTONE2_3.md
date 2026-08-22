# Milestones 2 & 3 — Embeddings, pgvector Store, Hybrid Retrieval

This document covers the retrieval half of GitAsk: turning `Chunk`s into
vectors, persisting them in Postgres/pgvector, and answering questions by
fusing dense vector search with sparse BM25 keyword search into a grounded,
citation-bearing prompt.

All code consumes the shared `Chunk` / `RetrievedChunk` contract from
`gitask.models` and reads configuration from `gitask.config.settings`.

---

## 1. Embeddings (`gitask.embeddings`)

### Factory & contract
- `get_embedder() -> Embedder` picks a backend from
  `settings.embedding_provider` (`"local"` | `"openai"`).
- `Embedder` is a `runtime_checkable` `Protocol` with:
  - `.embed(texts: list[str]) -> list[list[float]]` — batched **document** embedding.
  - `.embed_query(text: str) -> list[float]` — single **query** embedding.
  - `.dim: int` — output dimension, which **equals `settings.embedding_dim`**.

Every backend returns plain Python floats (not numpy arrays) so the rest of the
stack stays dependency-light.

### Backends
| Backend | Model | Dims | Notes |
|---|---|---|---|
| `LocalEmbedder` (default) | `BAAI/bge-small-en-v1.5` (`settings.local_embedding_model`) | **384** | sentence-transformers, no API key. Vectors L2-normalized. |
| `OpenAIEmbedder` | `text-embedding-3-small` | **1536** | Requires `OPENAI_API_KEY`. |

`settings.embedding_dim` is authoritative (384 for bge-small, 768 for a `*base*`
model, 1536 for OpenAI). The pgvector column width is derived from it, so the
model and the DB schema can never drift.

### Choices worth noting
- **Asymmetric query prefix for BGE.** bge-v1.5 retrieval models are trained
  with an instruction on the *query* side only. `embed_query` prepends
  `"Represent this sentence for searching relevant passages: "` for bge models;
  documents are embedded verbatim. OpenAI models are symmetric, so no prefix.
- **Lazy heavy imports.** `sentence_transformers` / `openai` are imported inside
  the constructors, so `import gitask.embeddings` works even when those packages
  are absent. Instantiating a backend whose dependency is missing raises a clear
  `ImportError`; a missing OpenAI key raises `ValueError`.
- **Dim guard.** `LocalEmbedder` cross-checks the loaded model's dimension
  against `settings.embedding_dim` and refuses a mismatch.

---

## 2. pgvector store (`gitask.db.store.VectorStore`)

Built on **psycopg v3** + **`pgvector.psycopg`**. One table, one row per chunk.

### Schema (`gitask/db/schema.sql`, mirrored by `init_schema()`)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id      TEXT PRIMARY KEY,      -- Chunk.chunk_id, stable => upsert key
    repo          TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    language      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    symbol_name   TEXT,
    parent_symbol TEXT,
    embedding     vector(<embedding_dim>)  -- width from settings.embedding_dim
);

CREATE INDEX chunks_repo_idx ON chunks (repo);
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

The columns are a 1:1 map of the `Chunk` dataclass fields plus the embedding.
`init_schema()` substitutes the live `embedding_dim` into the `vector(...)`
width; `schema.sql` shows the default 384-d form for manual bootstrapping.

### Index choice — HNSW with cosine ops
- **Cosine** (`vector_cosine_ops`, the `<=>` operator) matches the normalized
  embeddings the local model produces and OpenAI's cosine-oriented vectors.
- **HNSW over IVFFlat:** HNSW needs no training pass and gives strong
  recall/latency on a table that grows incrementally as repos are indexed.
  IVFFlat requires a populated table to build good lists, which is awkward for a
  fresh index. The `chunks_repo_idx` btree keeps the per-repo `WHERE repo = …`
  filter cheap.

### API
- `init_schema()` — extension + table + indexes, idempotent.
- `upsert_chunks(chunks, vectors)` — batched `executemany` with
  `INSERT … ON CONFLICT (chunk_id) DO UPDATE`, so re-indexing a repo overwrites
  in place instead of duplicating.
- `vector_search(query_vector, repo, k)` — `ORDER BY embedding <=> query`,
  filtered by repo; returns `RetrievedChunk`s with `vector_score = score =
  1 - cosine_distance` (**higher = more similar**, as the fusion layer expects).
- `get_repo_chunks(repo)` — full corpus for a repo (the BM25 source).
- `delete_repo(repo)`, `count(repo)`, `list_repos()`.

**Vector typing gotcha:** query parameters are wrapped in pgvector's `Vector`
type. A bare Python list is sent as `double precision[]`, which the `<=>`
operator rejects (`operator does not exist: vector <=> double precision[]`).

---

## 3. Hybrid retrieval (`gitask.retrieval.hybrid.HybridRetriever`)

`HybridRetriever(store, embedder).search(question, repo, k=None)` runs three
stages:

1. **Dense.** Embed the query and pull `settings.top_k_vector` (default 20)
   candidates from `vector_search`.
2. **Sparse (BM25).** Build a `rank_bm25.BM25Okapi` index over
   `store.get_repo_chunks(repo)` — the **whole repo**, not just the dense
   candidates — using code-aware tokenization, and score the query. Running BM25
   over the full corpus is what lets it *rescue* an exact-identifier chunk that
   dense search dropped entirely.
3. **Fuse (RRF).** Combine the two rankings with **Reciprocal Rank Fusion**.

### Code-aware tokenization (`gitask.retrieval.tokenize.tokenize_code`)
BM25 only helps if identifiers survive tokenization. The tokenizer emits **both**
the whole identifier and its sub-words:
- extracts `[A-Za-z0-9_]+` runs, so `login_user` is kept **whole**;
- splits each on `_` (snake_case) and on camelCase / acronym boundaries
  (`getUserByID` → `get`, `user`, `by`, `id`); dotted paths split on `.`.

So a query typed as the exact name `login_user`, or as `login`, or as
`getUserById`, all reach the right chunk.

### Why RRF (the fusion method)
`score = Σ_lists 1 / (RRF_K + rank_in_list)`, with `RRF_K = 60`.

- **Scale-free.** Cosine similarity (≈0–1) and BM25 (unbounded, corpus-dependent)
  live on wildly different scales. RRF uses *ranks*, so no fragile min-max
  normalization or hand-tuned weights are needed.
- **Robust rescue.** A chunk missed by dense search but ranked #1 by BM25 gets a
  large `1/(60+1)` contribution and floats to the top — exactly the
  exact-identifier case embeddings miss.
- Raw `vector_score` and `keyword_score` are retained on every `RetrievedChunk`
  for display/eval; the fused RRF value is stored in `.score`. Results are
  sorted by `.score` desc and truncated to `k` (default `settings.top_k_final`,
  8).

**Proof it works:** `test_exact_identifier_rescued_by_bm25_despite_low_vector_score`
gives the `login_user` chunk the *worst* vector score in the set, queries the
exact name `login_user`, and asserts it comes back ranked **#1** via BM25.

---

## 4. Grounded prompt (`gitask.retrieval.prompt`)

- **`SYSTEM_PROMPT`** instructs the model to (1) answer **only** from the
  provided context, (2) never invent files/functions, (3) cite every claim
  inline with `file_path:start-end` labels, and (4) say it cannot find the
  answer when the context is insufficient rather than guessing.
- **`build_user_prompt(question, retrieved)`** renders each chunk as a numbered,
  citable header followed by its (optionally truncated) body:

  ```
  [1] [src/auth.py:12-40  function login_user]
  <chunk source, fenced>
  ```

  then the question with a reminder to cite `file_path:line-range`. A per-chunk
  cap (2000 chars) and total budget (24000 chars) keep the prompt within a sane
  context window; empty retrieval yields an explicit "no code context" notice.

The header label is derived from `Chunk.citation_label`, so citations the model
emits map straight back to real source locations.

---

## 5. Tests & how to run

```bash
cd /home/user/GitAsk
pip install -q "psycopg[binary]" pgvector rank-bm25 numpy
python -m pytest tests/test_embeddings.py tests/test_retrieval.py tests/test_store.py -q
```

- `tests/test_embeddings.py` — factory wiring, `.dim == settings.embedding_dim`,
  a stub embedder that structurally satisfies the `Embedder` protocol, and a
  real-model shape check that **skips** (never fails) when
  sentence-transformers/the model can't be loaded.
- `tests/test_retrieval.py` — **no DB required**; a fake in-memory store + fake
  deterministic embedder drive the fusion tests, including the exact-identifier
  BM25 rescue, RRF sanity, tokenizer behavior, and prompt/grounding assertions.
- `tests/test_store.py` — **skips gracefully** if psycopg/pgvector are missing or
  no Postgres is reachable at `settings.database_url`; otherwise exercises
  `init_schema`, upsert (incl. idempotency), ranked `vector_search`, repo
  isolation, `count`, `get_repo_chunks`, and `delete_repo`.

### Database availability in this build
A Postgres+pgvector DB **was available**. The Docker image
(`pgvector/pgvector:pg16`) could not be pulled through the sandbox proxy, so a
local PostgreSQL 16 cluster was initialized with the `postgresql-16-pgvector`
extension and started on `localhost:5432` matching the default DSN
(`postgresql://gitask:gitask@localhost:5432/gitask`). `test_store.py` therefore
ran for real rather than skipping.

### Passing test output
```
$ python -m pytest tests/test_embeddings.py tests/test_retrieval.py tests/test_store.py -v -rs

tests/test_embeddings.py::test_factory_returns_configured_type SKIPPED
tests/test_embeddings.py::test_factory_rejects_unknown_provider PASSED
tests/test_embeddings.py::test_openai_embedder_requires_key SKIPPED
tests/test_embeddings.py::test_embedder_protocol_surface PASSED
tests/test_embeddings.py::test_local_embedder_real_model_if_available SKIPPED
tests/test_retrieval.py::test_search_returns_top_k_final_sorted PASSED
tests/test_retrieval.py::test_defaults_to_top_k_final PASSED
tests/test_retrieval.py::test_exact_identifier_rescued_by_bm25_despite_low_vector_score PASSED
tests/test_retrieval.py::test_both_signals_beats_single_signal PASSED
tests/test_retrieval.py::test_scores_populated PASSED
tests/test_retrieval.py::test_tokenize_splits_snake_and_camel_and_keeps_whole PASSED
tests/test_retrieval.py::test_system_prompt_mentions_grounding_and_citations PASSED
tests/test_retrieval.py::test_build_user_prompt_has_headers_and_question PASSED
tests/test_retrieval.py::test_build_user_prompt_handles_empty PASSED
tests/test_store.py::test_init_upsert_search_count_delete PASSED

SKIPPED [1] tests/test_embeddings.py:29: sentence-transformers not installed
SKIPPED [1] tests/test_embeddings.py:53: openai not installed
SKIPPED [1] tests/test_embeddings.py:85: sentence-transformers not installed
======================== 12 passed, 3 skipped ========================
```

The 3 skips are the network/heavy-dependency paths (sentence-transformers and
openai were not installed here); all pure-logic and DB-backed tests pass.
