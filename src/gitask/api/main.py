"""FastAPI backend for GitAsk (Milestone 4c).

Routes
------
GET  /                     -> serves the single-page chat frontend
GET  /health               -> liveness probe
GET  /repos                -> list indexed repos
POST /index                -> index a GitHub repo (synchronous) -> IndexResult
POST /ask                  -> grounded, cited answer for a question -> Answer

Components (vector store, embedder, LLM) are provided through a FastAPI
dependency (`get_components`) so tests can override them with fakes via
``app.dependency_overrides`` — no DB, model, network, or API key required.

Indexing is run **synchronously**: cloning + embedding a repo can take a while,
and the request simply blocks until the ``IndexResult`` is ready. This keeps the
API and the frontend simple; for very large repos a background-task variant
would be the natural next step (see docs/DEPLOY.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from gitask import pipeline

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


# --------------------------------------------------------------------------- #
# Injectable components                                                        #
# --------------------------------------------------------------------------- #
class Components:
    """Holds the pipeline's heavy components.

    In production the store/embedder/llm are built lazily and cached on first
    use. Tests construct this with ready-made fakes (``lazy=False``) and inject
    it through ``app.dependency_overrides[get_components]``.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        embedder: Any = None,
        llm: Any = None,
        retriever: Any = None,
        lazy: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._retriever = retriever
        self._lazy = lazy

    @property
    def store(self) -> Any:
        if self._store is None and self._lazy:
            self._store = pipeline._default_store()
        return self._store

    @property
    def embedder(self) -> Any:
        if self._embedder is None and self._lazy:
            self._embedder = pipeline._default_embedder()
        return self._embedder

    @property
    def llm(self) -> Any:
        if self._llm is None and self._lazy:
            self._llm = pipeline._default_llm()
        return self._llm

    @property
    def retriever(self) -> Any:
        if self._retriever is None and self._lazy:
            self._retriever = pipeline._default_retriever(self.store, self.embedder)
        return self._retriever


# Module-level singleton used in production. Tests override the dependency.
_components = Components(lazy=True)


def get_components() -> Components:
    """FastAPI dependency returning the active components bundle."""
    return _components


# --------------------------------------------------------------------------- #
# Request / response models                                                    #
# --------------------------------------------------------------------------- #
class IndexRequest(BaseModel):
    url: str = Field(..., description="Public GitHub repository URL to index.")


class IndexResponse(BaseModel):
    repo: str
    files_indexed: int
    chunks_indexed: int
    languages: dict[str, int]


class AskRequest(BaseModel):
    repo: str = Field(..., description='Repo identity, e.g. "pallets/flask".')
    question: str = Field(..., description="Natural-language question.")
    k: Optional[int] = Field(None, description="Override number of chunks retrieved.")


class ReposResponse(BaseModel):
    repos: list[str]


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="GitAsk",
    description="RAG-powered codebase Q&A with grounded, cited answers.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index_page() -> Any:
    """Serve the chat frontend, or a hint if it is missing."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse(
        {"detail": "Frontend not found. See /docs for the API."},
        status_code=200,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/repos", response_model=ReposResponse)
def repos(components: Components = Depends(get_components)) -> ReposResponse:
    return ReposResponse(repos=pipeline.list_repos(store=components.store))


@app.post("/index", response_model=IndexResponse)
def index(
    req: IndexRequest, components: Components = Depends(get_components)
) -> IndexResponse:
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="url is required")
    try:
        result = pipeline.index_repo(
            req.url, store=components.store, embedder=components.embedder
        )
    except Exception as exc:  # surface indexing failures cleanly to the client
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc
    return IndexResponse(**result.to_dict())


@app.post("/ask")
def ask(
    req: AskRequest, components: Components = Depends(get_components)
) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    if not req.repo.strip():
        raise HTTPException(status_code=400, detail="repo is required")
    try:
        answer = pipeline.ask(
            req.question,
            req.repo,
            retriever=components.retriever,
            llm=components.llm,
            k=req.k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc
    return answer.to_dict()
