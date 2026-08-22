"""Shared data-model contract for GitAsk.

This module is the **interface glue** between every layer of the pipeline:

    ingestion  ->  produces `Chunk`s
    embeddings ->  produces vectors for `Chunk.content`
    db / store ->  persists `Chunk`s + vectors, returns `RetrievedChunk`s
    retrieval  ->  ranks `RetrievedChunk`s, builds prompts
    llm        ->  produces `Answer` (text + `Citation`s)
    api        ->  serializes these to/from JSON

Every layer MUST import these types rather than inventing its own. Keep this
module dependency-free (stdlib only) so all layers can import it cheaply.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Ingestion / chunking (Milestone 1)                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A single AST-aware unit of code produced by the chunker.

    A chunk is ideally one function/method/class (semantic boundary), never a
    naive fixed-size line slice. `symbol_name` + `kind` capture what it is;
    `file_path`/`start_line`/`end_line` are what citations point at.
    """

    repo: str                      # e.g. "owner/name" or the clone URL
    file_path: str                 # path relative to repo root, e.g. "src/auth.py"
    language: str                  # tree-sitter language id, e.g. "python"
    kind: str                      # "function" | "method" | "class" | "module" | "block"
    content: str                   # the raw source text of the chunk
    start_line: int                # 1-indexed, inclusive
    end_line: int                  # 1-indexed, inclusive
    symbol_name: Optional[str] = None      # e.g. "login_user"; None for module/block
    parent_symbol: Optional[str] = None    # enclosing class/function, if any
    chunk_id: str = ""             # stable content+location hash (filled by __post_init__)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = self.compute_id()

    def compute_id(self) -> str:
        """Deterministic id from repo + location + content (safe for upsert)."""
        h = hashlib.sha256()
        h.update(
            f"{self.repo}\0{self.file_path}\0{self.start_line}\0{self.end_line}\0".encode()
        )
        h.update(self.content.encode("utf-8", errors="replace"))
        return h.hexdigest()[:32]

    @property
    def citation_label(self) -> str:
        """Human-facing citation, e.g. 'src/auth.py:12-40'."""
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """Text handed to the embedder. Prepends light metadata so identifiers
        and file context contribute to the vector."""
        header = f"# {self.file_path}"
        if self.symbol_name:
            header += f" :: {self.kind} {self.symbol_name}"
        return f"{header}\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


# --------------------------------------------------------------------------- #
# Retrieval (Milestones 2 & 3)                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    """A chunk returned by search, with its scores.

    `score` is the final fused/re-ranked score used for ordering.
    `vector_score` and `keyword_score` are kept for debugging / eval / display.
    """

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetrievedChunk":
        return cls(
            chunk=Chunk.from_dict(d["chunk"]),
            score=d.get("score", 0.0),
            vector_score=d.get("vector_score", 0.0),
            keyword_score=d.get("keyword_score", 0.0),
        )


# --------------------------------------------------------------------------- #
# Answer synthesis (Milestone 3/4)                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Citation:
    """A source pointer surfaced with an answer, e.g. 'src/auth.py:12-40'."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None

    @property
    def label(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Citation":
        return cls(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol_name=chunk.symbol_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """Final grounded answer returned to the user."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [r.to_dict() for r in self.retrieved],
        }


# --------------------------------------------------------------------------- #
# Indexing bookkeeping                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class IndexResult:
    """Summary returned after indexing a repository."""

    repo: str
    files_indexed: int
    chunks_indexed: int
    languages: dict[str, int] = field(default_factory=dict)  # language -> chunk count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
