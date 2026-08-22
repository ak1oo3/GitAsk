# Deploying GitAsk

GitAsk splits into two deployables:

- **Backend** — the FastAPI app (`gitask.api.main:app`) + a pgvector Postgres.
  Deploy to **Railway** or **Render**.
- **Frontend** — the static `frontend/index.html` chat page. Deploy to
  **Vercel** (or just let the backend serve it at `/`).

The backend can also serve the frontend itself (`GET /` returns
`frontend/index.html`), so a single-service deploy is fine — Vercel is optional.

## Backend

The app reads config from environment variables (see `.env.example`):

| Var | Purpose |
|---|---|
| `DATABASE_URL` | pgvector Postgres connection string |
| `EMBEDDING_PROVIDER` | `local` (bge, no key) or `openai` |
| `LOCAL_EMBEDDING_MODEL` | e.g. `BAAI/bge-small-en-v1.5` (384 dims) |
| `ANTHROPIC_API_KEY` | Claude key (secret — never commit) |
| `ANTHROPIC_MODEL` | e.g. `claude-sonnet-5` |

Start command (used by every platform below):

```bash
uvicorn gitask.api.main:app --host 0.0.0.0 --port $PORT --app-dir src
```

`--app-dir src` puts the `src/` layout on the import path so `gitask` resolves.

### Railway

`railway.json` is committed. Steps:

1. Create a project from the repo. Railway reads `railway.json` (NIXPACKS build,
   the start command above, `/health` healthcheck).
2. Add a **Postgres** plugin; enable pgvector (`CREATE EXTENSION vector;` — the
   app also runs this in `init_schema()`).
3. Set `DATABASE_URL` (from the plugin), `ANTHROPIC_API_KEY`, and the embedding
   vars in the service **Variables**.

### Render

`render.yaml` is committed as a Blueprint: one web service + one managed
Postgres, wired together. Steps:

1. New → Blueprint → point at the repo. Render provisions both.
2. `DATABASE_URL` is injected from the database automatically.
3. Add `ANTHROPIC_API_KEY` in the dashboard (it is marked `sync: false`).

### Heroku-style (Procfile)

`Procfile` is committed for any Procfile-based host:

```
web: uvicorn gitask.api.main:app --host 0.0.0.0 --port $PORT --app-dir src
```

Attach a Postgres with pgvector and set the same env vars.

## Frontend (Vercel, optional)

`vercel.json` deploys `frontend/` as a static site.

1. Import the repo in Vercel; it serves `frontend/index.html`.
2. Point the page at your backend: open it and paste the backend URL into the
   **Backend** field (stored in `localStorage`), or hardcode it near the top of
   the `<script>` in `frontend/index.html`.
3. CORS is already enabled in the FastAPI app (`allow_origins=["*"]`), so a
   cross-origin frontend works out of the box. Tighten `allow_origins` to your
   Vercel domain for production.

## Notes on indexing cost

`POST /index` runs synchronously: it clones, AST-chunks, embeds and upserts the
whole repo before responding, which can take from seconds to minutes depending
on repo size and whether the embedding model is warm. On platforms with short
request timeouts, index large repos via a one-off worker/CLI run of
`gitask.pipeline.index_repo`, or evolve `/index` into a `BackgroundTasks` job
with a `GET /index/status?repo=` poll (a documented, deliberate next step).
