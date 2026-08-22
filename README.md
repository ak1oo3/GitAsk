# GitAsk — RAG-Powered Codebase Q&A Assistant

Paste any public GitHub repo URL, wait for indexing, then ask natural-language
questions about the codebase — *"where is auth handled?"*, *"how are sessions
signed?"* — and get **grounded, cited answers** with `file.py:42`-style
citations instead of hallucinated generalities.

> Built a RAG-based codebase Q&A tool using AST-aware code chunking
> (tree-sitter), pgvector for hybrid vector + keyword retrieval, and grounded
> LLM prompting with source citations; evaluated retrieval precision against a
> hand-labeled question set.

---

## Why it exists

Ask a plain LLM "where is X in this repo?" and it will confidently invent a file
that doesn't exist. GitAsk grounds every answer in **actual retrieved source
code** and forces the model to cite the `file:line` ranges it used — so answers
are checkable, not vibes. It is a compact, end-to-end **retrieval-augmented
generation (RAG)** system built around three ideas that matter for *code*
specifically: semantic chunking on AST boundaries, hybrid retrieval that
respects exact identifiers, and citation-constrained prompting.

---

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

### Layered design (one contract, swappable parts)

Every layer speaks the shared dataclasses in [`src/gitask/models.py`](src/gitask/models.py)
(`Chunk`, `RetrievedChunk`, `Citation`, `Answer`, `IndexResult`) and reads config
from [`src/gitask/config.py`](src/gitask/config.py). That makes each layer
independently testable and replaceable:

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `gitask.ingestion.{cloner,walker,chunker}` | clone a repo, walk files, AST-chunk into semantic units |
| Embeddings | `gitask.embeddings.embedder` | `bge-small` locally (384-dim) or OpenAI |
| Store | `gitask.db.store` | pgvector upsert + cosine search + repo bookkeeping |
| Retrieval | `gitask.retrieval.{hybrid,prompt}` | fuse vector + BM25, build the grounded prompt |
| LLM | `gitask.llm.client` | call Claude, parse and attach citations |
| Pipeline | `gitask.pipeline` | orchestrate index / ask / list |
| API | `gitask.api.main` | FastAPI endpoints + serve the frontend |
| Frontend | `frontend/index.html` | single-page chat UI |

Every component is **dependency-injected**, so the whole system runs under test
with fakes — no DB, model, network, or API key required.

---

## Tech stack & rationale

**AST-aware chunking (tree-sitter).** Naive fixed-size line windows split
functions in half and glue unrelated code together, which wrecks retrieval. GitAsk
chunks on **syntactic boundaries** — a function, method, or class is one chunk —
so a retrieved unit is a coherent, self-contained answer with real `start_line` /
`end_line` ranges to cite.

**pgvector for the vector store.** Vectors live next to everything else in
Postgres: no extra vector-DB service to run, transactional upserts, trivial
per-repo filtering with a `WHERE repo = …`, and it deploys as a single managed
database on Railway/Render. Plenty fast at this corpus size.

**Hybrid retrieval (vector + BM25).** Pure embeddings miss **exact identifiers** —
ask for `get_signing_serializer` and dense similarity may drift to
"session-ish" code. Pure BM25 misses **paraphrase** — *"how are sessions
signed?"* shares few tokens with the code that signs them. Fusing them gets both:
semantic recall *and* exact-symbol precision. The ablation below quantifies why.

**Metadata-enriched embeddings.** Each chunk is embedded as
`# path :: kind symbol_name\n<code>` (`Chunk.embedding_text()`), so the file path
and symbol name contribute to the vector — a big lift for "where is X" queries.

**Grounded, citation-constrained prompting + Claude.** The system prompt forbids
answering beyond the retrieved context and requires `file:line` citations; the
LLM client parses those labels back to the retrieved chunks so the UI can show
exactly what grounded the answer.

**FastAPI + vanilla-JS frontend.** Typed pydantic request/response models, auto
`/docs`, CORS for a split deploy, and a zero-build single HTML page so the whole
thing is easy to run and read.

---

## Quickstart

### 1. Infrastructure + install

```bash
# pgvector Postgres (matches src/gitask/config.py defaults)
docker compose up -d db

# install everything
pip install -r requirements.txt

# configure (copy and edit)
cp .env.example .env
#   set ANTHROPIC_API_KEY=... ; local embeddings need no key
```

### 2. Run the API (serves the frontend too)

```bash
uvicorn gitask.api.main:app --reload --app-dir src
# open http://localhost:8000  → the chat UI
# API docs at http://localhost:8000/docs
```

### 3. Index and ask (from the UI, or via curl)

```bash
# index a repo (synchronous — clones, chunks, embeds, upserts)
curl -X POST localhost:8000/index \
  -H 'content-type: application/json' \
  -d '{"url":"https://github.com/pallets/flask"}'

# ask a grounded question
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"repo":"pallets/flask","question":"How are sessions signed?"}'
```

`POST /ask` returns `Answer.to_dict()` — the answer `text`, a `citations` list of
`file:line` pointers, and the full `retrieved` chunk set with scores.

---

## How it works

### Indexing — `pipeline.index_repo(url)`

1. `repo_name_from_url` → `"owner/name"`; `clone_repo` fetches the source.
2. Prior chunks for that repo are **deleted first** (clean re-index, no dupes).
3. `chunk_repo` AST-chunks every file into `Chunk`s.
4. Chunks are embedded **in batches** and `upsert_chunks`'d into pgvector.
5. Returns an `IndexResult` (files, chunks, per-language counts).

### Asking — `pipeline.ask(question, repo)`

1. `HybridRetriever.search` embeds the query, runs vector + BM25, fuses, and
   returns the top `RetrievedChunk`s.
2. `LLMClient.answer` builds the grounded prompt (`SYSTEM_PROMPT` +
   `build_user_prompt`), calls Claude, and derives `Citation`s by parsing the
   `file:line` labels the model emitted and matching them back to retrieved
   chunks (falling back to all retrieved chunks so an answer is never uncited).

### API surface

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/` | — | the chat frontend |
| GET | `/health` | — | `{"status":"ok"}` |
| GET | `/repos` | — | `{"repos":[…]}` |
| POST | `/index` | `{"url":…}` | `IndexResult` |
| POST | `/ask` | `{"repo":…,"question":…}` | `Answer` (text + citations + retrieved) |

---

## Evaluation — retrieval precision

Retrieval is the part of a RAG system that actually determines answer quality, so
it is measured directly. The eval asks: **does the retriever surface the files
that truly answer a question?**

- **Target repo:** [`pallets/flask`](https://github.com/pallets/flask).
- **Question set:** [`eval/questions.jsonl`](eval/questions.jsonl) — **20**
  hand-labeled questions (realistic — *"where is routing handled?"*, *"how are
  sessions signed?"*, exact-identifier lookups like *"where is
  `get_signing_serializer`?"*), each mapped to the Flask source file(s) that
  answer it, verified against the real repo tree.
- **Hit:** a retrieved chunk whose `file_path` is in the question's
  `relevant_files`.
- **Metrics:** Top-1 / Top-3 / Top-5 accuracy, Precision@5, MRR.
- **Ablation:** vector-only vs BM25-only vs hybrid, via
  [`eval/evaluate.py`](eval/evaluate.py).

### Corpus (measured)

`pallets/flask` chunked by the real tree-sitter chunker: **1,268 chunks across
222 files** (872 Python, 323 reStructuredText, plus html/yaml/toml/…), embedded
with `BAAI/bge-small-en-v1.5` (**384** dims) into pgvector.

### Results

| mode | Top-1 | Top-3 | Top-5 | P@5 | MRR |
|---|---|---|---|---|---|
| vector-only | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL>` | `<FILL>` |
| **bm25-only (measured)** | **25.0%** | **55.0%** | **80.0%** | **0.300** | **0.438** |
| **hybrid (production)** | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL FROM eval run>` | `<FILL>` | `<FILL>` |

The **bm25-only** row is a **real, reproducible** measurement (no DB or model
needed — it chunks the repo and ranks with `rank-bm25`):

```bash
git clone --depth 1 https://github.com/pallets/flask /tmp/flask
python eval/evaluate.py --offline-bm25 /tmp/flask --per-question
```

The **vector** and **hybrid** rows require the full stack (pgvector + embedding
model); a maintainer fills them in with:

```bash
docker compose up -d db && pip install -r requirements.txt
python eval/evaluate.py --repo-url https://github.com/pallets/flask --per-question
```

**Honesty:** only the BM25 row is measured so far — the rest are placeholders,
not estimates. Hybrid is *expected* to beat BM25 (metadata-enriched embeddings
recover paraphrased queries that keyword search misses, and BM25's noise from the
`.rst` docs is down-weighted by fusion), but that claim is unproven until the
full run fills the table. Full methodology + how to reproduce:
[`eval/run_eval.md`](eval/run_eval.md) and [`docs/METRICS.md`](docs/METRICS.md).

---

## Testing

Everything runs green with **no network, DB, model, or API key** — sibling layers
are stubbed and components are faked through the DI seams:

```bash
pip install -q fastapi "uvicorn[standard]" pydantic pydantic-settings jinja2 \
  anthropic tabulate httpx numpy
python -m pytest tests/test_pipeline.py tests/test_api.py -q
```

`tests/test_pipeline.py` covers `index_repo` counts + clean re-index, batched
embedding, `ask` wiring retriever→LLM, and real citation parsing.
`tests/test_api.py` drives the FastAPI `TestClient` for `/health`, `/repos`,
`/index`, `/ask`, and the served frontend.

---

## Deployment

- **Backend** (FastAPI + pgvector) → **Railway** (`railway.json`) or **Render**
  (`render.yaml` Blueprint with a managed Postgres); a `Procfile` covers any
  Procfile host.
- **Frontend** → **Vercel** (`vercel.json`) as a static page, or just let the
  backend serve it at `/`.

CORS is enabled, and the frontend can be pointed at any backend URL at runtime.
Full instructions: [`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## Limitations & future work

- **Synchronous indexing.** `POST /index` blocks until done; large repos want a
  `BackgroundTasks` job with a `GET /index/status` poll (designed for, not yet
  built).
- **Answer-quality eval.** Only *retrieval* precision is measured; end-to-end
  answer correctness (faithfulness, citation accuracy) is future work.
- **Full ablation numbers.** Vector/hybrid rows await a live run (see above).
- **Chunking coverage.** Languages without a tree-sitter grammar fall back to
  coarser chunking; symbol-graph awareness (call/import edges) would improve
  retrieval further.
- **Re-ranking.** A cross-encoder re-rank over the fused candidates would likely
  lift Top-1.
- **Scale.** pgvector is ideal here; very large corpora would want an ANN index
  (IVFFlat/HNSW) and sharding.

---

## Repository layout

```
src/gitask/
  models.py            shared dataclasses (the contract)
  config.py            env-driven settings
  ingestion/           cloner, walker, tree-sitter chunker
  embeddings/          bge/openai embedder
  db/                  pgvector store
  retrieval/           hybrid retriever + grounded prompt
  llm/                 Claude client + citation parsing
  pipeline.py          index / ask / list orchestration
  api/                 FastAPI app
frontend/index.html    single-page chat UI
eval/                  questions.jsonl + evaluate.py + run_eval.md
docs/                  METRICS.md, DEPLOY.md, GITHUB_DESCRIPTION.md
tests/                 pipeline + API tests (no infra needed)
```
