# Running the GitAsk retrieval evaluation

The eval measures **retrieval precision** — does the retriever surface the files
that actually answer a question? — on a hand-labeled question set
(`eval/questions.jsonl`, 20 questions) against **`pallets/flask`**.

A **hit** is a retrieved chunk whose `file_path` is in the question's
`relevant_files`. We report **Top-1 / Top-3 / Top-5 accuracy**, **Precision@5**,
and **MRR**, and an **ablation** across three retrieval modes:

| mode     | what it uses                                              |
|----------|----------------------------------------------------------|
| `vector` | pgvector cosine search over embedded chunks              |
| `bm25`   | BM25 keyword ranking over chunk text                     |
| `hybrid` | fused vector + BM25 re-ranking (the production retriever) |

## Option A — offline BM25 baseline (no DB, no model, no network)

Only needs the ingestion layer (tree-sitter). This is exactly how the
**real BM25 baseline** in `docs/METRICS.md` was produced.

```bash
pip install rank-bm25 tree-sitter tree-sitter-languages GitPython
git clone --depth 1 https://github.com/pallets/flask /tmp/flask
python eval/evaluate.py --offline-bm25 /tmp/flask --per-question
```

## Option B — full evaluation (vector + BM25 + hybrid)

Needs the whole stack live: a pgvector database, the embedding model, and the
ingestion + retrieval layers.

```bash
# 1. start pgvector
docker compose up -d db

# 2. install everything
pip install -r requirements.txt

# 3. index the target repo and evaluate all three modes
python eval/evaluate.py --repo-url https://github.com/pallets/flask --per-question

# re-run without re-indexing (reuses what's already in the store)
python eval/evaluate.py --repo pallets/flask --no-index
```

The harness prints a per-question table and the ablation summary. Copy the
summary table into `docs/METRICS.md` and the README, replacing the
`<FILL FROM eval run>` placeholders for the `vector` and `hybrid` rows.

## Guardrails

If the DB, embedding model, or a sibling layer is missing, the harness prints
setup instructions and exits with code 2 — it never crashes. That makes it safe
to wire into CI as a smoke test.
