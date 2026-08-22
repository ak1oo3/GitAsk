"""Retrieval tests -- NO database required.

Uses an in-memory fake implementing the ``VectorStore`` method surface plus a
deterministic fake embedder, so we can control vector scores exactly and prove
the fusion behaviour:

* fusion returns ``top_k_final`` results, sorted by score;
* an exact code-identifier query surfaces the matching chunk via BM25 even when
  its vector score is deliberately the worst of the set;
* a chunk strong in both signals outranks one strong in only one (RRF sanity).

Also covers ``build_user_prompt`` / ``SYSTEM_PROMPT`` grounding + citations and
the code-aware tokenizer.
"""

from __future__ import annotations

from gitask.config import settings
from gitask.models import Chunk, RetrievedChunk
from gitask.retrieval.hybrid import HybridRetriever
from gitask.retrieval.prompt import SYSTEM_PROMPT, build_user_prompt
from gitask.retrieval.tokenize import tokenize_code


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic embedder; the fake store ignores the vector, so any
    fixed-length output is fine."""

    dim = settings.embedding_dim

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


class FakeStore:
    """In-memory stand-in for VectorStore with caller-controlled vector scores.

    ``vector_scores`` maps chunk_id -> similarity; vector_search returns chunks
    ranked by that value (desc), independent of the query vector, so tests can
    force a chunk to have a low/high dense score.
    """

    def __init__(self, chunks: list[Chunk], vector_scores: dict[str, float]):
        self._chunks = chunks
        self._scores = vector_scores

    def vector_search(self, query_vector, repo, k):
        ranked = sorted(
            (c for c in self._chunks if c.repo == repo),
            key=lambda c: self._scores.get(c.chunk_id, 0.0),
            reverse=True,
        )
        out = []
        for c in ranked[:k]:
            s = self._scores.get(c.chunk_id, 0.0)
            out.append(RetrievedChunk(chunk=c, score=s, vector_score=s))
        return out

    def get_repo_chunks(self, repo):
        return [c for c in self._chunks if c.repo == repo]


REPO = "acme/app"


def _mk(file_path, symbol, content, kind="function"):
    return Chunk(
        repo=REPO,
        file_path=file_path,
        language="python",
        kind=kind,
        content=content,
        start_line=1,
        end_line=1 + content.count("\n"),
        symbol_name=symbol,
    )


def _corpus():
    return [
        _mk(
            "src/auth.py",
            "login_user",
            "def login_user(username, password):\n"
            "    session = create_session(username)\n"
            "    return session",
        ),
        _mk(
            "src/math_utils.py",
            "add",
            "def add(a, b):\n    return a + b",
        ),
        _mk(
            "src/report.py",
            "generate_report",
            "def generate_report(data):\n"
            "    # summarize the quarterly numbers\n"
            "    return summary(data)",
        ),
        _mk(
            "src/cache.py",
            "get_value",
            "def get_value(key):\n    return store.get(key)",
        ),
    ]


# --------------------------------------------------------------------------- #
# Fusion behaviour                                                             #
# --------------------------------------------------------------------------- #
def test_search_returns_top_k_final_sorted():
    chunks = _corpus()
    scores = {c.chunk_id: 0.5 for c in chunks}
    retriever = HybridRetriever(FakeStore(chunks, scores), FakeEmbedder())

    results = retriever.search("how to add two numbers", REPO, k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)
    # sorted by score desc
    assert results[0].score >= results[1].score


def test_defaults_to_top_k_final():
    chunks = _corpus()
    scores = {c.chunk_id: 0.5 for c in chunks}
    retriever = HybridRetriever(FakeStore(chunks, scores), FakeEmbedder())
    results = retriever.search("anything", REPO)
    assert len(results) == min(settings.top_k_final, len(chunks))


def test_exact_identifier_rescued_by_bm25_despite_low_vector_score():
    """The killer test: query is the exact function name, but that chunk has
    the WORST vector score. BM25 must still float it to the top."""
    chunks = _corpus()
    login = next(c for c in chunks if c.symbol_name == "login_user")
    # Everyone else scores high on vectors; login_user scores lowest.
    scores = {c.chunk_id: 0.9 for c in chunks}
    scores[login.chunk_id] = 0.01

    retriever = HybridRetriever(FakeStore(chunks, scores), FakeEmbedder())
    results = retriever.search("login_user", REPO, k=4)

    ids = [r.chunk.chunk_id for r in results]
    assert login.chunk_id in ids
    assert results[0].chunk.chunk_id == login.chunk_id, (
        "exact identifier match should rank #1 via BM25 rescue"
    )
    # The winning chunk carries a keyword signal.
    assert results[0].keyword_score > 0.0


def test_both_signals_beats_single_signal():
    """A chunk strong in BOTH vector and keyword outranks one strong in only
    the vector signal (RRF sanity)."""
    chunks = _corpus()
    login = next(c for c in chunks if c.symbol_name == "login_user")
    add = next(c for c in chunks if c.symbol_name == "add")

    # login_user: high vector AND matches the keyword query.
    # add: highest vector but no keyword overlap with the query.
    scores = {c.chunk_id: 0.1 for c in chunks}
    scores[login.chunk_id] = 0.8
    scores[add.chunk_id] = 0.95

    retriever = HybridRetriever(FakeStore(chunks, scores), FakeEmbedder())
    results = retriever.search("login_user session", REPO, k=4)
    assert results[0].chunk.chunk_id == login.chunk_id


def test_scores_populated():
    chunks = _corpus()
    scores = {c.chunk_id: 0.7 for c in chunks}
    retriever = HybridRetriever(FakeStore(chunks, scores), FakeEmbedder())
    results = retriever.search("login_user", REPO, k=4)
    for r in results:
        # final fused score is set; component scores are present.
        assert r.score > 0.0
        assert hasattr(r, "vector_score")
        assert hasattr(r, "keyword_score")


# --------------------------------------------------------------------------- #
# Tokenizer                                                                    #
# --------------------------------------------------------------------------- #
def test_tokenize_splits_snake_and_camel_and_keeps_whole():
    toks = tokenize_code("login_user getUserByID foo.bar_baz")
    # whole snake identifier kept
    assert "login_user" in toks
    # sub-words present
    assert "login" in toks and "user" in toks
    # camelCase split
    assert "get" in toks and "user" in toks and "id" in toks
    # dotted split
    assert "foo" in toks and "bar_baz" in toks


# --------------------------------------------------------------------------- #
# Prompt building                                                             #
# --------------------------------------------------------------------------- #
def test_system_prompt_mentions_grounding_and_citations():
    low = SYSTEM_PROMPT.lower()
    assert "only" in low  # answer only from context
    assert "cit" in low  # citations
    assert "file_path" in low
    # refusal / no-invention guidance
    assert "can't find" in low or "cannot find" in low or "find" in low
    assert "invent" in low or "do not use outside" in low


def test_build_user_prompt_has_headers_and_question():
    chunks = _corpus()
    retrieved = [
        RetrievedChunk(chunk=c, score=1.0 - i * 0.1)
        for i, c in enumerate(chunks[:2])
    ]
    prompt = build_user_prompt("Where do users log in?", retrieved)

    assert "Where do users log in?" in prompt
    # numbered, citable headers containing the citation label + kind + symbol
    assert "[1]" in prompt and "[2]" in prompt
    assert "src/auth.py:1-3" in prompt
    assert "login_user" in prompt
    # chunk body included
    assert "create_session" in prompt


def test_build_user_prompt_handles_empty():
    prompt = build_user_prompt("q?", [])
    assert "q?" in prompt
    assert "no code context" in prompt.lower()
