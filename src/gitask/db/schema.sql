-- GitAsk pgvector schema (Milestone 2).
--
-- The embedding column width MUST match config.settings.embedding_dim. The
-- application substitutes the active dimension into the {dim} placeholder when
-- running init_schema(); this file mirrors the DDL with the default 384-d
-- (bge-small-en-v1.5) width for reference / manual bootstrapping.

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
    embedding     vector(384)
);

-- Repo filter is applied on every search; keep it cheap.
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);

-- Approximate-nearest-neighbour index for cosine similarity. HNSW gives good
-- recall/latency without needing a populated table to train (unlike ivfflat).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
