"""Tests for the AST-aware chunker (Milestone 1).

These build tiny source files on disk and assert the chunker recovers the
expected functions, classes, methods, kinds, symbol names and line ranges --
plus the line-window fallback for unsupported file types.
"""

from __future__ import annotations

import pathlib

import pytest

# Ensure ``src`` layout is importable when tests run from the repo root.
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gitask.ingestion.chunker import chunk_file  # noqa: E402
from gitask.models import Chunk  # noqa: E402


def _write(tmp_path: pathlib.Path, name: str, content: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _by_symbol(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.symbol_name: c for c in chunks if c.symbol_name}


# --------------------------------------------------------------------------- #
# Python                                                                       #
# --------------------------------------------------------------------------- #
PY_SOURCE = '''\
import os
import sys

CONST = 42


def top_level(a, b):
    """A top-level function."""
    return a + b


class Widget:
    """A widget."""

    def __init__(self, name):
        self.name = name

    def render(self):
        return f"<{self.name}>"


TRAILER = top_level(1, 2)
'''


def test_python_functions_classes_methods(tmp_path):
    path = _write(tmp_path, "sample.py", PY_SOURCE)
    chunks = chunk_file(path, repo="acme/app", rel_path="sample.py")

    kinds = {c.kind for c in chunks}
    assert {"function", "class", "method", "module"} <= kinds

    by_sym = _by_symbol(chunks)

    # Top-level function.
    fn = by_sym["top_level"]
    assert fn.kind == "function"
    assert fn.parent_symbol is None
    assert fn.start_line == 7
    assert fn.end_line == 9
    assert "return a + b" in fn.content

    # Class.
    cls = by_sym["Widget"]
    assert cls.kind == "class"
    assert cls.start_line == 12

    # Nested methods carry parent_symbol.
    init = by_sym["__init__"]
    assert init.kind == "method"
    assert init.parent_symbol == "Widget"
    render = by_sym["render"]
    assert render.kind == "method"
    assert render.parent_symbol == "Widget"

    # Module-level code (imports + CONST + TRAILER) is captured, not dropped.
    module_chunks = [c for c in chunks if c.kind == "module"]
    assert module_chunks
    joined = "\n".join(c.content for c in module_chunks)
    assert "import os" in joined
    assert "CONST = 42" in joined
    assert "TRAILER =" in joined


def test_python_metadata_fields(tmp_path):
    path = _write(tmp_path, "sample.py", PY_SOURCE)
    chunks = chunk_file(path, repo="acme/app", rel_path="pkg/sample.py")
    for c in chunks:
        assert c.repo == "acme/app"
        assert c.file_path == "pkg/sample.py"
        assert c.language == "python"
        assert c.chunk_id  # populated by __post_init__
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


# --------------------------------------------------------------------------- #
# A second language: Go (top-level methods attributed to their receiver)       #
# --------------------------------------------------------------------------- #
GO_SOURCE = '''\
package main

import "fmt"

func main() {
\tfmt.Println("hello")
}

type Server struct {
\tport int
}

func (s Server) Start() error {
\treturn nil
}
'''


def test_go_functions_and_methods(tmp_path):
    path = _write(tmp_path, "main.go", GO_SOURCE)
    chunks = chunk_file(path, repo="acme/app", rel_path="main.go")
    by_sym = _by_symbol(chunks)

    assert by_sym["main"].kind == "function"

    assert by_sym["Server"].kind == "class"

    start = by_sym["Start"]
    assert start.kind == "method"
    assert start.parent_symbol == "Server"


# --------------------------------------------------------------------------- #
# A third language: JavaScript (class methods)                                 #
# --------------------------------------------------------------------------- #
JS_SOURCE = '''\
import x from "y";

function greet(name) {
  return `hi ${name}`;
}

class Counter {
  constructor() {
    this.n = 0;
  }
  inc() {
    this.n += 1;
  }
}
'''


def test_javascript_class_methods(tmp_path):
    path = _write(tmp_path, "app.js", JS_SOURCE)
    chunks = chunk_file(path, repo="acme/app", rel_path="app.js")
    by_sym = _by_symbol(chunks)

    assert by_sym["greet"].kind == "function"
    assert by_sym["Counter"].kind == "class"
    assert by_sym["inc"].kind == "method"
    assert by_sym["inc"].parent_symbol == "Counter"


# --------------------------------------------------------------------------- #
# Line-window fallback for unsupported extensions                              #
# --------------------------------------------------------------------------- #
def test_line_window_fallback_for_unsupported(tmp_path):
    body = "\n".join(f"bullet point number {i}" for i in range(150))
    path = _write(tmp_path, "notes.md", f"# Notes\n\n{body}\n")
    chunks = chunk_file(path, repo="acme/app", rel_path="notes.md")

    assert chunks, "fallback must produce chunks, never drop the file"
    assert all(c.kind == "block" for c in chunks)
    assert all(c.symbol_name is None for c in chunks)
    # More than one window for a 150+ line file.
    assert len(chunks) > 1
    # Windows overlap: a later window starts before the previous one ends.
    assert chunks[1].start_line <= chunks[0].end_line
    # Nothing lost: last window reaches the final line.
    assert chunks[-1].end_line >= 150


def test_fallback_when_language_unknown(tmp_path):
    path = _write(tmp_path, "data.xyz", "alpha\nbeta\ngamma\n")
    chunks = chunk_file(path, repo="acme/app", rel_path="data.xyz", language="text")
    assert len(chunks) == 1
    assert chunks[0].kind == "block"
    assert "alpha" in chunks[0].content
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


# --------------------------------------------------------------------------- #
# Oversized definitions are block-split on AST boundaries                      #
# --------------------------------------------------------------------------- #
def test_large_function_is_block_split(tmp_path):
    lines = ["def huge():"]
    for i in range(400):
        lines.append(f"    x{i} = {i}")
    path = _write(tmp_path, "big.py", "\n".join(lines) + "\n")
    chunks = chunk_file(path, repo="acme/app", rel_path="big.py")

    block_chunks = [c for c in chunks if c.kind == "block"]
    assert len(block_chunks) >= 2, "a 400-line function should be split"
    # Every block stays within the configured cap.
    for c in block_chunks:
        assert (c.end_line - c.start_line + 1) <= 200
    # Coverage is contiguous and complete over the function body.
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, 402))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
