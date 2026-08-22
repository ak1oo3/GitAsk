# GitHub repo description

## About (short — the repo "About" field, ≤ 350 chars)

> RAG-powered codebase Q&A: paste a GitHub URL, ask in plain English, get
> grounded answers with `file.py:line` citations. AST-aware chunking
> (tree-sitter) + pgvector hybrid (vector + BM25) retrieval + cited Claude
> answers. Evaluated for retrieval precision on a hand-labeled question set.

## Topics / tags

`rag` · `llm` · `claude` · `anthropic` · `pgvector` · `vector-search` ·
`hybrid-search` · `bm25` · `tree-sitter` · `code-search` · `fastapi` ·
`semantic-search` · `embeddings` · `question-answering`

## Results blurb (résumé / portfolio line)

Target sentence, with the metric filled from the evaluation. The hybrid figure
requires a full pipeline run (pgvector + embedding model) — it is clearly marked
to fill; the BM25 baseline below it is a **real measured number**.

> Built a RAG-based codebase Q&A tool using AST-aware code chunking
> (tree-sitter), pgvector for hybrid vector + keyword retrieval, and grounded LLM
> prompting with source citations; evaluated retrieval precision against a
> hand-labeled 20-question set, achieving **`<FILL FROM full hybrid eval run>`%
> top-3 retrieval accuracy**.

**Measured baseline (real, reproducible):** the BM25-only retriever already
reaches **55% Top-3** and **80% Top-5** accuracy (P@5 = 0.30, MRR = 0.44) on the
Flask question set — see `docs/METRICS.md`. Hybrid vector + BM25 retrieval is
expected to exceed this; run `python eval/evaluate.py` to fill in the hybrid
number.

> Honesty note: only replace `<FILL …>` after a real full run. Do not paste an
> estimate — the BM25 numbers above are the only measured figures so far.
