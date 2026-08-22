"""FastAPI backend tests using TestClient with fake components.

No DB, model, network, or API key: ``app.dependency_overrides`` swaps the real
component bundle for one carrying fakes, and the pipeline's clone/chunk imports
are stubbed into ``sys.modules`` for the /index test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gitask.models import Answer, Chunk, Citation, RetrievedChunk


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
CHUNKS = [
    Chunk(repo="acme/demo", file_path="src/app.py", language="python", kind="function",
          content="def route(): pass", start_line=1, end_line=2, symbol_name="route"),
    Chunk(repo="acme/demo", file_path="src/auth.py", language="python", kind="function",
          content="def sign(): pass", start_line=10, end_line=20, symbol_name="sign"),
]


class FakeEmbedder:
    dim = 8

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


class FakeStore:
    def __init__(self, repos=None):
        self.repos = list(repos or [])
        self.upserts = 0
        self.deleted = []

    def init_schema(self):
        pass

    def delete_repo(self, repo):
        self.deleted.append(repo)

    def upsert_chunks(self, chunks, vectors):
        self.upserts += 1

    def list_repos(self):
        return list(self.repos)


class FakeRetriever:
    def search(self, question, repo, k=None):
        return [
            RetrievedChunk(chunk=c, score=0.9, vector_score=0.6, keyword_score=0.3)
            for c in CHUNKS
        ]


class FakeLLM:
    def answer(self, question, retrieved):
        return Answer(
            text="Routing lives in src/app.py:1.",
            citations=[Citation.from_chunk(retrieved[0].chunk)],
            retrieved=retrieved,
        )


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def client_with(monkeypatch):
    """Return a factory building a TestClient with an injected components bundle."""
    from gitask.api import main as api_main

    def _make(store=None, retriever=None, llm=None, embedder=None):
        comp = api_main.Components(
            store=store if store is not None else FakeStore(),
            embedder=embedder if embedder is not None else FakeEmbedder(),
            retriever=retriever if retriever is not None else FakeRetriever(),
            llm=llm if llm is not None else FakeLLM(),
            lazy=False,
        )
        api_main.app.dependency_overrides[api_main.get_components] = lambda: comp
        return TestClient(api_main.app), comp

    yield _make
    api_main.app.dependency_overrides.clear()


@pytest.fixture
def stub_ingestion(monkeypatch):
    cloner = types.ModuleType("gitask.ingestion.cloner")
    cloner.clone_repo = lambda url, dest_dir=None: Path("/tmp/fake-clone")
    cloner.repo_name_from_url = lambda url: "acme/demo"
    chunker = types.ModuleType("gitask.ingestion.chunker")
    chunker.chunk_repo = lambda repo_path, repo: iter(CHUNKS)

    pkg = sys.modules.get("gitask.ingestion") or types.ModuleType("gitask.ingestion")
    monkeypatch.setitem(sys.modules, "gitask.ingestion", pkg)
    monkeypatch.setitem(sys.modules, "gitask.ingestion.cloner", cloner)
    monkeypatch.setitem(sys.modules, "gitask.ingestion.chunker", chunker)
    yield


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_health(client_with):
    client, _ = client_with()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_repos(client_with):
    client, _ = client_with(store=FakeStore(repos=["acme/demo", "pallets/flask"]))
    r = client.get("/repos")
    assert r.status_code == 200
    assert r.json() == {"repos": ["acme/demo", "pallets/flask"]}


def test_index(client_with, stub_ingestion):
    store = FakeStore()
    client, comp = client_with(store=store)
    r = client.post("/index", json={"url": "https://github.com/acme/demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["repo"] == "acme/demo"
    assert body["chunks_indexed"] == 2
    assert body["files_indexed"] == 2
    assert body["languages"] == {"python": 2}
    assert "acme/demo" in store.deleted


def test_index_requires_url(client_with):
    client, _ = client_with()
    r = client.post("/index", json={"url": "   "})
    assert r.status_code == 400


def test_ask_shape(client_with):
    client, _ = client_with()
    r = client.post("/ask", json={"repo": "acme/demo", "question": "where is routing?"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Routing lives in src/app.py:1."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["file_path"] == "src/app.py"
    # retrieved metadata surfaced with scores
    assert len(body["retrieved"]) == 2
    assert body["retrieved"][0]["score"] == 0.9
    assert body["retrieved"][0]["chunk"]["file_path"] == "src/app.py"


def test_ask_requires_question(client_with):
    client, _ = client_with()
    r = client.post("/ask", json={"repo": "acme/demo", "question": ""})
    assert r.status_code == 400


def test_root_serves_frontend(client_with):
    client, _ = client_with()
    r = client.get("/")
    assert r.status_code == 200
    # Frontend HTML is served (index.html exists in the repo).
    assert "GitAsk" in r.text
