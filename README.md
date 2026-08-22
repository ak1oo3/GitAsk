# GitAsk — RAG-Powered Codebase Q&A Assistant

Paste any public GitHub repo URL, wait for indexing, then ask natural-language
questions about the codebase — *"where is auth handled?"* — and get **cited,
grounded answers** with `file.py:42`-style citations instead of hallucinated
generalities.

> Built a RAG-based codebase Q&A tool using AST-aware code chunking
> (tree-sitter), pgvector for hybrid vector + keyword retrieval, and grounded
> LLM prompting with source citations; evaluated retrieval precision against a
> hand-labeled question set.

## Architecture

```
GitHub URL
   │
   ▼
Clone + Walk ──► AST chunk (tree-sitter: function/class boundaries)
                      │
                      ▼
              Embed each chunk + metadata ──► pgvector (vector + file + line range)

Question ─► Embed query ─► Vector top-k ─┐
                          + BM25 keyword ─┤─► Hybrid re-rank ─► Grounded prompt ─► LLM ─► Answer + citations
```

See `docs/` and the module docstrings for details. This README is completed in
Milestone 5 with the evaluation results.

## Status

Under active construction across build milestones. See the milestone tracker in
the repository description / commit history.
