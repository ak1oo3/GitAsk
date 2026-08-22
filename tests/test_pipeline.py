"""Unit tests for the orchestration pipeline and the LLM client.

Everything runs with **no network, no DB, no model, and no API key**. Sibling
layers that aren't installed in the test env (ingestion, retrieval.prompt) are
stubbed into ``sys.modules``; the store / embedder / retriever / llm are fakes
injected through the pipeline's dependency-injection seams.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from gitask.models import Answer, Chunk, RetrievedChunk


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = abs(hash(text))
        return [((h >> (i * 3)) & 7) / 7.0 for i in range(FakeEmbedder.dim)]


class FakeStore:
    def __init__(self) -> None:
        self.by_repo: dict[str, list[Chunk]] = {}
        self.vectors: dict[str, list[float]] = {}
        self.schema_ready = False
        self.deleted: list[str] = []

    def init_schema(self) -> None:
        self.schema_ready = True

    def upsert_chunks(self, chunks, vectors) -> None:
        for c, v in zip(chunks, vectors):
            self.by_repo.setdefault(c.repo, []).append(c)
            self.vectors[c.chunk_id] = v

    def delete_repo(self, repo: str) -> None:
        self.deleted.append(repo)
        self.by_repo.pop(repo, None)

    def get_repo_chunks(self, repo: str) -> list[Chunk]:
        return list(self.by_repo.get(repo, []))

    def count(self, repo: str) -> int:
        return len(self.by_repo.get(repo, []))

    def list_repos(self) -> list[str]:
        return list(self.by_repo.keys())

    def vector_search(self, qvec, repo, k):  # not exercised here
        return []


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, str]] = []

    def search(self, question: str, repo: str, k=None) -> list[RetrievedChunk]:
        self.calls.append((question, repo))
        return [
            RetrievedChunk(chunk=c, score=1.0 - i * 0.1, vector_score=0.5, keyword_score=0.5)
            for i, c in enumerate(self._chunks)
        ]


class FakeLLM:
    def __init__(self) -> None:
        self.received: list[tuple[str, list[RetrievedChunk]]] = []

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        self.received.append((question, retrieved))
        from gitask.models import Citation

        return Answer(
            text="fake answer",
            citations=[Citation.from_chunk(r.chunk) for r in retrieved],
            retrieved=retrieved,
        )


# --------------------------------------------------------------------------- #
# Stub the sibling modules the pipeline imports lazily                         #
# --------------------------------------------------------------------------- #
SYNTH_CHUNKS = [
    Chunk(repo="acme/demo", file_path="src/app.py", language="python", kind="function",
          content="def route(): pass", start_line=1, end_line=2, symbol_name="route"),
    Chunk(repo="acme/demo", file_path="src/app.py", language="python", kind="function",
          content="def login(): pass", start_line=4, end_line=5, symbol_name="login"),
    Chunk(repo="acme/demo", file_path="web/ui.js", language="javascript", kind="function",
          content="function render(){}", start_line=1, end_line=3, symbol_name="render"),
]


@pytest.fixture(autouse=True)
def stub_ingestion(monkeypatch):
    cloner = types.ModuleType("gitask.ingestion.cloner")
    cloner.clone_repo = lambda url, dest_dir=None: Path("/tmp/fake-clone")
    cloner.repo_name_from_url = lambda url: "acme/demo"

    chunker = types.ModuleType("gitask.ingestion.chunker")
    chunker.chunk_repo = lambda repo_path, repo: iter(SYNTH_CHUNKS)

    ingestion_pkg = sys.modules.get("gitask.ingestion") or types.ModuleType("gitask.ingestion")
    monkeypatch.setitem(sys.modules, "gitask.ingestion", ingestion_pkg)
    monkeypatch.setitem(sys.modules, "gitask.ingestion.cloner", cloner)
    monkeypatch.setitem(sys.modules, "gitask.ingestion.chunker", chunker)
    yield


@pytest.fixture
def stub_prompt(monkeypatch):
    prompt = types.ModuleType("gitask.retrieval.prompt")
    prompt.SYSTEM_PROMPT = "You are GitAsk."
    prompt.build_user_prompt = lambda q, retrieved: (
        q + "\n" + "\n".join(r.chunk.citation_label for r in retrieved)
    )
    retrieval_pkg = sys.modules.get("gitask.retrieval") or types.ModuleType("gitask.retrieval")
    monkeypatch.setitem(sys.modules, "gitask.retrieval", retrieval_pkg)
    monkeypatch.setitem(sys.modules, "gitask.retrieval.prompt", prompt)
    yield


# --------------------------------------------------------------------------- #
# index_repo                                                                   #
# --------------------------------------------------------------------------- #
def test_index_repo_counts_and_clean_reindex():
    from gitask import pipeline

    store = FakeStore()
    embedder = FakeEmbedder()

    result = pipeline.index_repo("https://github.com/acme/demo", store=store, embedder=embedder)

    assert result.repo == "acme/demo"
    assert result.chunks_indexed == 3
    assert result.files_indexed == 2  # src/app.py, web/ui.js
    assert result.languages == {"python": 2, "javascript": 1}

    # A clean re-index deletes prior chunks for the repo first.
    assert "acme/demo" in store.deleted
    assert store.count("acme/demo") == 3


def test_index_repo_batches_embeddings():
    from gitask import pipeline

    store = FakeStore()
    embedder = FakeEmbedder()

    # Force batch size 1 -> 3 upsert batches, each vector aligned to its chunk.
    result = pipeline.index_repo(
        "https://github.com/acme/demo", store=store, embedder=embedder, batch_size=1
    )
    assert result.chunks_indexed == 3
    assert len(store.vectors) == 3
    for v in store.vectors.values():
        assert len(v) == embedder.dim


# --------------------------------------------------------------------------- #
# ask                                                                          #
# --------------------------------------------------------------------------- #
def test_ask_wires_retriever_to_llm():
    from gitask import pipeline

    retriever = FakeRetriever(SYNTH_CHUNKS)
    llm = FakeLLM()

    answer = pipeline.ask("where is login?", "acme/demo", retriever=retriever, llm=llm)

    assert isinstance(answer, Answer)
    assert answer.text == "fake answer"
    assert len(answer.citations) == 3
    assert retriever.calls == [("where is login?", "acme/demo")]
    # LLM received exactly what the retriever produced.
    assert llm.received[0][0] == "where is login?"
    assert len(llm.received[0][1]) == 3


def test_ask_with_real_llmclient_and_injected_completion(stub_prompt):
    """Exercise the real LLMClient citation-parsing with a fake completion fn."""
    from gitask import pipeline
    from gitask.llm.client import LLMClient

    retriever = FakeRetriever(SYNTH_CHUNKS)

    def fake_complete(model, system, user):
        # Model cites src/app.py:4 -> should match the login chunk (lines 4-5).
        return "Login is handled in src/app.py:4."

    llm = LLMClient(_complete=fake_complete)
    answer = pipeline.ask("where is login?", "acme/demo", retriever=retriever, llm=llm)

    assert answer.text == "Login is handled in src/app.py:4."
    assert len(answer.citations) == 1
    cite = answer.citations[0]
    assert cite.file_path == "src/app.py"
    assert cite.start_line == 4 and cite.end_line == 5
    assert cite.symbol_name == "login"


def test_llmclient_falls_back_to_all_retrieved_when_no_labels(stub_prompt):
    from gitask.llm.client import LLMClient

    retrieved = [
        RetrievedChunk(chunk=c, score=1.0, vector_score=0.5, keyword_score=0.5)
        for c in SYNTH_CHUNKS
    ]
    llm = LLMClient(_complete=lambda m, s, u: "No specific citation here.")
    answer = llm.answer("q", retrieved)
    assert len(answer.citations) == 3  # fell back to all retrieved


# --------------------------------------------------------------------------- #
# list_repos                                                                   #
# --------------------------------------------------------------------------- #
def test_list_repos():
    from gitask import pipeline

    store = FakeStore()
    store.upsert_chunks(SYNTH_CHUNKS, [[0.0] * 8] * 3)
    assert pipeline.list_repos(store=store) == ["acme/demo"]
