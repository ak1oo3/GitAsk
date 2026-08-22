"""Retrieval-precision evaluation for GitAsk (Milestones 5 & 6).

Runs a hand-labeled question set (``eval/questions.jsonl``) against a target
repo and reports **Top-1 / Top-3 / Top-5 accuracy, Precision@k and MRR**, plus a
**vector-only vs BM25-only vs hybrid ablation**.

A "hit" is a retrieved chunk whose ``file_path`` is in the question's
``relevant_files``.

The harness needs the full stack live (pgvector DB + embedding model + the
ingestion/retrieval layers). When any of that is missing it prints clear setup
instructions and exits cleanly instead of crashing — so it is safe to run in CI
or on a laptop without a database.

Usage
-----
    # index the target repo, then evaluate all three retrieval modes
    python eval/evaluate.py --repo-url https://github.com/pallets/flask

    # skip indexing if the repo is already in the store
    python eval/evaluate.py --repo pallets/flask --no-index

    # single mode only
    python eval/evaluate.py --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
DEFAULT_REPO_URL = "https://github.com/pallets/flask"
DEFAULT_REPO = "pallets/flask"
K = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    question: str
    relevant_files: list[str]
    relevant_symbols: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ModeResult:
    name: str
    top1: float
    top3: float
    top5: float
    precision_at_k: float
    mrr: float
    rows: list[list[object]]


def load_questions(path: Path) -> list[Question]:
    qs: list[Question] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qs.append(
                Question(
                    question=d["question"],
                    relevant_files=d["relevant_files"],
                    relevant_symbols=d.get("relevant_symbols", []),
                    notes=d.get("notes", ""),
                )
            )
    return qs


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --------------------------------------------------------------------------- #
# Retrieval modes (return an ordered list of file_paths)                       #
# --------------------------------------------------------------------------- #
def make_vector_search(store, embedder) -> Callable[[str, str, int], list[str]]:
    def search(question: str, repo: str, k: int) -> list[str]:
        qvec = embedder.embed_query(question)
        results = store.vector_search(qvec, repo, k)
        return [r.chunk.file_path for r in results]

    return search


def make_bm25_search(store) -> Callable[[str, str, int], list[str]]:
    from rank_bm25 import BM25Okapi

    cache: dict[str, tuple] = {}

    def search(question: str, repo: str, k: int) -> list[str]:
        if repo not in cache:
            chunks = store.get_repo_chunks(repo)
            corpus = [tokenize(c.embedding_text()) for c in chunks]
            cache[repo] = (BM25Okapi(corpus), chunks)
        bm25, chunks = cache[repo]
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i].file_path for i in order]

    return search


def make_offline_bm25_search(repo_path: str, repo: str) -> Callable[[str, str, int], list[str]]:
    """BM25 over chunks produced *directly* by the chunker — no DB, no model.

    This lets a maintainer produce a real BM25 retrieval number with only the
    ingestion layer installed (tree-sitter), which is what the committed BM25
    baseline in docs/METRICS.md was generated with.
    """
    from rank_bm25 import BM25Okapi
    from gitask.ingestion.chunker import chunk_repo

    chunks = list(chunk_repo(Path(repo_path), repo))
    corpus = [tokenize(c.embedding_text()) for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"(offline BM25: chunked {len(chunks)} chunks from {len(set(c.file_path for c in chunks))} files)")

    def search(question: str, repo_: str, k: int) -> list[str]:
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i].file_path for i in order]

    return search


def make_hybrid_search(retriever) -> Callable[[str, str, int], list[str]]:
    def search(question: str, repo: str, k: int) -> list[str]:
        results = retriever.search(question, repo, k=k)
        return [r.chunk.file_path for r in results[:k]]

    return search


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def first_hit_rank(files: list[str], relevant: set[str]) -> Optional[int]:
    for i, fp in enumerate(files, start=1):
        if fp in relevant:
            return i
    return None


def evaluate_mode(
    name: str,
    search: Callable[[str, str, int], list[str]],
    questions: list[Question],
    repo: str,
    k: int = K,
) -> ModeResult:
    n = len(questions)
    top1 = top3 = top5 = 0
    precision_sum = 0.0
    rr_sum = 0.0
    rows: list[list[object]] = []

    for q in questions:
        relevant = set(q.relevant_files)
        files = search(q.question, repo, k)
        rank = first_hit_rank(files, relevant)
        hits_at_k = sum(1 for fp in files[:k] if fp in relevant)

        top1 += int(rank is not None and rank <= 1)
        top3 += int(rank is not None and rank <= 3)
        top5 += int(rank is not None and rank <= 5)
        precision_sum += hits_at_k / k
        rr_sum += (1.0 / rank) if rank else 0.0

        rows.append(
            [
                q.question[:48] + ("…" if len(q.question) > 48 else ""),
                rank if rank else "—",
                f"{hits_at_k}/{k}",
                (files[0] if files else "—"),
            ]
        )

    return ModeResult(
        name=name,
        top1=top1 / n,
        top3=top3 / n,
        top5=top5 / n,
        precision_at_k=precision_sum / n,
        mrr=rr_sum / n,
        rows=rows,
    )


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def print_per_question(result: ModeResult) -> None:
    from tabulate import tabulate

    print(f"\n=== Per-question ({result.name}) ===")
    print(
        tabulate(
            result.rows,
            headers=["question", "first-hit rank", f"hits@{K}", "top-1 file"],
            tablefmt="github",
        )
    )


def print_summary(results: list[ModeResult]) -> None:
    from tabulate import tabulate

    rows = [
        [
            r.name,
            f"{r.top1*100:.1f}%",
            f"{r.top3*100:.1f}%",
            f"{r.top5*100:.1f}%",
            f"{r.precision_at_k:.3f}",
            f"{r.mrr:.3f}",
        ]
        for r in results
    ]
    print("\n=== Retrieval metrics (ablation) ===")
    print(
        tabulate(
            rows,
            headers=["mode", "Top-1", "Top-3", "Top-5", f"P@{K}", "MRR"],
            tablefmt="github",
        )
    )
    print(
        "\nHit = a retrieved chunk whose file_path is in the question's "
        "relevant_files. Paste this table into docs/METRICS.md and the README."
    )


# --------------------------------------------------------------------------- #
# Setup guard                                                                  #
# --------------------------------------------------------------------------- #
SETUP_HELP = """\
------------------------------------------------------------------------------
Cannot run the live evaluation — a required piece of the stack is unavailable.

The eval needs the FULL pipeline online:
  1. A pgvector Postgres database (docker compose up -d db)
  2. The embedding model + native deps:  pip install -r requirements.txt
  3. The ingestion + retrieval layers importable (Milestones 1-3)

Quickstart:
  docker compose up -d db
  pip install -r requirements.txt
  python eval/evaluate.py --repo-url https://github.com/pallets/flask

The harness will then index the target repo and fill in real numbers. See
eval/run_eval.md for the full walkthrough.

Underlying error:
  {error}
------------------------------------------------------------------------------"""


def _ensure_src_on_path() -> None:
    """Make `gitask` importable from the src/ layout when run as a script."""
    src = Path(__file__).resolve().parents[1] / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_components():
    """Construct store, embedder, retriever. Raises on any missing dependency."""
    _ensure_src_on_path()

    from gitask import pipeline
    from gitask.retrieval.hybrid import HybridRetriever

    store = pipeline._default_store()
    embedder = pipeline._default_embedder()
    retriever = HybridRetriever(store, embedder)
    return pipeline, store, embedder, retriever


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="GitAsk retrieval evaluation")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="repo to index")
    ap.add_argument("--repo", default=None, help='indexed repo id, e.g. "pallets/flask"')
    ap.add_argument("--no-index", action="store_true", help="skip indexing (reuse store)")
    ap.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid", "all"],
        default="all",
        help="which retrieval mode(s) to evaluate",
    )
    ap.add_argument("--per-question", action="store_true", help="print per-question tables")
    ap.add_argument(
        "--offline-bm25",
        metavar="REPO_PATH",
        default=None,
        help="run a DB/model-free BM25 baseline by chunking REPO_PATH directly",
    )
    args = ap.parse_args()

    repo = args.repo or _repo_from_url(args.repo_url)
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} hand-labeled questions for {repo}.")

    # Offline BM25 baseline needs only the chunker — no DB, no embedding model.
    if args.offline_bm25:
        _ensure_src_on_path()
        try:
            search = make_offline_bm25_search(args.offline_bm25, repo)
        except Exception as exc:
            print(SETUP_HELP.format(error=repr(exc)))
            return 2
        result = evaluate_mode("bm25 (offline)", search, questions, repo)
        if args.per_question:
            print_per_question(result)
        print_summary([result])
        return 0

    try:
        pipeline, store, embedder, retriever = build_components()
    except Exception as exc:  # missing DB / model / layer
        print(SETUP_HELP.format(error=repr(exc)))
        return 2

    # Index unless told not to (or already present).
    try:
        already = store.count(repo) if not args.no_index else store.count(repo)
    except Exception:
        already = 0

    if not args.no_index and already == 0:
        print(f"Indexing {args.repo_url} …")
        t0 = time.time()
        try:
            res = pipeline.index_repo(args.repo_url, store=store, embedder=embedder)
        except Exception as exc:
            print(SETUP_HELP.format(error=repr(exc)))
            return 2
        dt = time.time() - t0
        print(
            f"Indexed {res.repo}: {res.files_indexed} files, {res.chunks_indexed} "
            f"chunks, languages={res.languages}, in {dt:.1f}s."
        )
    else:
        print(f"Using already-indexed repo ({already} chunks).")

    if store.count(repo) == 0:
        print(SETUP_HELP.format(error=f"No chunks indexed for {repo}."))
        return 2

    modes: dict[str, Callable[[str, str, int], list[str]]] = {}
    if args.mode in ("vector", "all"):
        modes["vector"] = make_vector_search(store, embedder)
    if args.mode in ("bm25", "all"):
        try:
            modes["bm25"] = make_bm25_search(store)
        except Exception as exc:
            print(f"(bm25 mode unavailable: {exc})")
    if args.mode in ("hybrid", "all"):
        modes["hybrid"] = make_hybrid_search(retriever)

    results: list[ModeResult] = []
    for name, search in modes.items():
        result = evaluate_mode(name, search, questions, repo)
        results.append(result)
        if args.per_question:
            print_per_question(result)

    print_summary(results)
    return 0


def _repo_from_url(url: str) -> str:
    try:
        # Reuse the canonical parser when the layer is importable.
        src = Path(__file__).resolve().parents[1] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from gitask.ingestion.cloner import repo_name_from_url

        return repo_name_from_url(url)
    except Exception:
        parts = url.rstrip("/").removesuffix(".git").split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else url


if __name__ == "__main__":
    raise SystemExit(main())
