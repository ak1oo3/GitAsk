# GitAsk — Metrics

Retrieval-precision evaluation of GitAsk on a hand-labeled question set.

## Evaluation setup

| Item | Value |
|---|---|
| Target repo | **`pallets/flask`** (Flask web framework) |
| Question set | `eval/questions.jsonl` — **20** hand-labeled questions |
| Labeling | Each question maps to the Flask source file(s) that actually answer it, verified against the cloned repo tree |
| Hit definition | A retrieved chunk whose `file_path` ∈ the question's `relevant_files` |
| Metrics | Top-1 / Top-3 / Top-5 accuracy, Precision@5, MRR |
| k (retrieved chunks scored) | 5 |
| Harness | `eval/evaluate.py` (supports `vector` / `bm25` / `hybrid` ablation) |

## Corpus statistics (measured)

Produced by the real `tree-sitter` AST chunker on `pallets/flask` (default clone):

| Stat | Value |
|---|---|
| Files chunked | **222** |
| Chunks produced | **1,268** |
| Chunks by language | python: 872, restructuredtext: 323, html: 21, text: 16, yaml: 10, toml: 10, markdown: 6, css: 4, json: 2, sql: 2, bash: 1, make: 1 |
| Embedding model | `BAAI/bge-small-en-v1.5` (**384** dims) — `config.settings.embedding_dim` |
| Vector store | Postgres + **pgvector** (cosine) |
| Chunking time | ~0.2 s (chunking only; full embed + upsert depends on hardware/model) |
| Full index time | `<FILL FROM eval run>` |

## Retrieval accuracy — ablation

`bm25` is a **real measured baseline**, produced offline with only the chunker +
`rank-bm25` (no DB, no model), reproducible via:

```bash
python eval/evaluate.py --offline-bm25 /path/to/flask
```

`vector` and `hybrid` require the full stack (pgvector + embedding model) and are
filled in from a maintainer's live run — see `eval/run_eval.md`.

| mode | Top-1 | Top-3 | Top-5 | P@5 | MRR |
|---|---|---|---|---|---|
| vector-only | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL>` | `<FILL>` |
| **bm25-only (measured)** | **25.0%** | **55.0%** | **80.0%** | **0.300** | **0.438** |
| **hybrid (production)** | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL>` | `<FILL>` |

> The `bm25-only` row is a real, reproducible number. Hybrid is expected to
> exceed it: the embedded chunks carry file-path + symbol metadata
> (`Chunk.embedding_text()`), so semantic queries like *"how are sessions
> signed?"* match `sessions.py` even when the exact keywords are absent, while
> BM25 alone over-weights prose in the `.rst` docs. Do **not** treat the hybrid
> row as known until a full run fills it in.

## How to reproduce

See `eval/run_eval.md`. In short:

```bash
# real BM25 baseline (no infra)
python eval/evaluate.py --offline-bm25 /path/to/flask --per-question

# full ablation (needs docker compose up -d db + pip install -r requirements.txt)
python eval/evaluate.py --repo-url https://github.com/pallets/flask --per-question
```

## Honesty note

Numbers labeled **measured** come from a real run of the committed harness.
Everything marked `<FILL FROM eval run>` is a placeholder for a live run and has
**not** been fabricated or estimated.

A live full-ablation run was attempted in the build/CI environment (pgvector was
brought up and the retrieval stack installed), but the `vector` and `hybrid`
rows could not be produced there: the network egress policy blocks the embedding
model download (`huggingface.co`) and the OpenAI API, and no weights were cached.
On any machine where those hosts are reachable the numbers fill in with the
single `--repo-url` command above.
