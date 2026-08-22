"""AST-aware chunking of source files into :class:`~gitask.models.Chunk`s.

Why AST over fixed-size windows
-------------------------------
A fixed ``N``-line window slices straight through the middle of functions and
statements, so a retrieved chunk rarely lines up with a coherent unit of code
and citations point at arbitrary line ranges. Chunking on *syntax* boundaries
instead means every chunk is a whole function, method or class -- a unit a
reader (and an LLM) can reason about, and a citation that points at something
meaningful.

Strategy
--------
For each file we parse the source with tree-sitter and walk the top-level
definitions:

* a top-level function/def becomes a ``"function"`` chunk;
* a class/struct/impl/namespace becomes a ``"class"`` chunk, and each
  function defined inside it becomes a ``"method"`` chunk carrying
  ``parent_symbol`` (nesting recurses, so a method inside a nested class is
  attributed to the nearest enclosing named container);
* code that lives outside any definition (imports, module-level statements) is
  captured as ``"module"`` chunks so nothing is silently dropped;
* a definition longer than ``MAX_CHUNK_LINES`` is split into ``"block"``
  sub-chunks along AST child boundaries (never mid-statement).

When no grammar is available for a file (markdown, yaml, plain text, ...), or
when tree-sitter is missing or a parse fails, we fall back to an overlapping
line-window chunker producing ``"block"`` chunks -- the file is always chunked,
never skipped.
"""

from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Iterator, Optional

from gitask.config import settings
from gitask.models import Chunk
from gitask.ingestion.walker import (
    TREE_SITTER_LANGUAGES,
    language_for,
    walk_files,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional tree-sitter import (guarded so the module always imports)           #
# --------------------------------------------------------------------------- #
try:
    from tree_sitter_languages import get_parser  # type: ignore

    _TS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without grammars
    get_parser = None  # type: ignore
    _TS_AVAILABLE = False
    logger.warning(
        "tree-sitter-languages unavailable (%s); using line-window fallback "
        "for all files",
        exc,
    )

# Cache parsers per language (constructing them is not free).
_PARSER_CACHE: dict[str, object] = {}

# --------------------------------------------------------------------------- #
# Tuning                                                                       #
# --------------------------------------------------------------------------- #
MAX_CHUNK_LINES = 200          # definitions longer than this are block-split
WINDOW_LINES = 60              # line-window fallback size
WINDOW_OVERLAP = 10            # overlap between fallback windows

# --------------------------------------------------------------------------- #
# Per-language node-type configuration                                         #
# --------------------------------------------------------------------------- #
# ``func``: node types that are functions/methods.
# ``cls`` : container node types whose inner functions become methods and which
#           themselves become "class" chunks.
_FUNC = "func"
_CLS = "cls"

LANG_NODES: dict[str, dict[str, frozenset[str]]] = {
    "python": {
        _FUNC: frozenset({"function_definition"}),
        _CLS: frozenset({"class_definition"}),
    },
    "javascript": {
        _FUNC: frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
            }
        ),
        _CLS: frozenset({"class_declaration", "class"}),
    },
    "typescript": {
        _FUNC: frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
                "function_signature",
                "method_signature",
            }
        ),
        _CLS: frozenset(
            {"class_declaration", "class", "interface_declaration", "module"}
        ),
    },
    "java": {
        _FUNC: frozenset({"method_declaration", "constructor_declaration"}),
        _CLS: frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            }
        ),
    },
    "go": {
        _FUNC: frozenset({"function_declaration", "method_declaration"}),
        _CLS: frozenset({"type_declaration"}),
    },
    "ruby": {
        _FUNC: frozenset({"method", "singleton_method"}),
        _CLS: frozenset({"class", "module"}),
    },
    "rust": {
        _FUNC: frozenset({"function_item"}),
        _CLS: frozenset(
            {"impl_item", "struct_item", "enum_item", "trait_item", "mod_item"}
        ),
    },
    "c": {
        _FUNC: frozenset({"function_definition"}),
        _CLS: frozenset({"struct_specifier", "union_specifier", "enum_specifier"}),
    },
    "cpp": {
        _FUNC: frozenset({"function_definition"}),
        _CLS: frozenset(
            {
                "class_specifier",
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
                "namespace_definition",
            }
        ),
    },
}
# tsx shares TypeScript's node types.
LANG_NODES["tsx"] = LANG_NODES["typescript"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _get_parser(language: str):
    parser = _PARSER_CACHE.get(language)
    if parser is None:
        # tree-sitter-languages triggers a deprecated-API FutureWarning from the
        # tree-sitter core; silence it -- it is not actionable from here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            parser = get_parser(language)  # type: ignore[misc]
        _PARSER_CACHE[language] = parser
    return parser


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _node_name(node, source: bytes) -> Optional[str]:
    """Best-effort symbol name for a definition node across languages."""
    # Most grammars expose a ``name`` field.
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, source)

    # C / C++ functions: name lives inside nested declarators.
    declarator = node.child_by_field_name("declarator")
    seen = 0
    while declarator is not None and seen < 8:
        seen += 1
        if declarator.type in ("identifier", "field_identifier", "type_identifier"):
            return _node_text(declarator, source)
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            for child in declarator.children:
                if child.type in (
                    "identifier",
                    "field_identifier",
                    "type_identifier",
                    "qualified_identifier",
                ):
                    return _node_text(child, source)
            break
        declarator = inner

    # Rust impl blocks / Go type declarations: dig for a (type_)identifier.
    for child in node.children:
        if child.type in ("type_identifier", "identifier", "constant"):
            return _node_text(child, source)
        if child.type == "type_spec":  # go: type X struct {...}
            n = child.child_by_field_name("name")
            if n is not None:
                return _node_text(n, source)
    return None


def _go_receiver_type(node, source: bytes) -> Optional[str]:
    """Extract the receiver type name of a Go ``method_declaration``."""
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    # Look for the (possibly pointer) type identifier inside the receiver.
    stack = list(receiver.children)
    while stack:
        child = stack.pop(0)
        if child.type == "type_identifier":
            return _node_text(child, source)
        stack.extend(child.children)
    return None


def _body_node(node):
    """Return the node holding a container/function's inner definitions."""
    for field in ("body", "declaration_list"):
        b = node.child_by_field_name(field)
        if b is not None:
            return b
    # Fall back to the first block-ish child.
    for child in node.children:
        if child.type in (
            "block",
            "class_body",
            "declaration_list",
            "field_declaration_list",
            "statement_block",
            "enum_body",
        ):
            return child
    return None


def _lines(node) -> tuple[int, int]:
    """1-indexed inclusive (start_line, end_line) for a node."""
    return node.start_point[0] + 1, node.end_point[0] + 1


# --------------------------------------------------------------------------- #
# Block splitting for oversized definitions                                    #
# --------------------------------------------------------------------------- #
def _split_large_node(
    node,
    source: bytes,
    *,
    repo: str,
    rel_path: str,
    language: str,
    symbol_name: Optional[str],
    parent_symbol: Optional[str],
) -> list[Chunk]:
    """Split a too-large definition into ``block`` chunks on child boundaries."""
    body = _body_node(node)
    start_line, end_line = _lines(node)
    if body is None or not body.named_children:
        # Nothing to split along; keep as a single (large) chunk.
        return [
            _make_chunk(
                repo=repo,
                rel_path=rel_path,
                language=language,
                kind="block",
                content=_node_text(node, source),
                start_line=start_line,
                end_line=end_line,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
            )
        ]

    children = list(body.named_children)
    chunks: list[Chunk] = []
    src_lines = source.decode("utf-8", errors="replace").split("\n")

    # Signature/header block: from the definition start to just before the body.
    header_end = children[0].start_point[0]  # 0-indexed, exclusive of first child
    if header_end >= node.start_point[0]:
        header_text = "\n".join(src_lines[node.start_point[0] : header_end])
        if header_text.strip():
            chunks.append(
                _make_chunk(
                    repo=repo,
                    rel_path=rel_path,
                    language=language,
                    kind="block",
                    content=header_text,
                    start_line=node.start_point[0] + 1,
                    end_line=header_end,
                    symbol_name=symbol_name,
                    parent_symbol=parent_symbol,
                )
            )

    # Group consecutive body statements up to the line budget.
    group: list = []
    group_start = children[0].start_point[0]

    def flush(last_line_excl: int) -> None:
        if not group:
            return
        g_start = group[0].start_point[0]
        g_end = group[-1].end_point[0]
        text = "\n".join(src_lines[g_start : g_end + 1])
        chunks.append(
            _make_chunk(
                repo=repo,
                rel_path=rel_path,
                language=language,
                kind="block",
                content=text,
                start_line=g_start + 1,
                end_line=g_end + 1,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
            )
        )

    for child in children:
        c_start = child.start_point[0]
        c_end = child.end_point[0]
        if group and (c_end - group_start + 1) > MAX_CHUNK_LINES:
            flush(c_start)
            group = []
            group_start = c_start
        if not group:
            group_start = c_start
        group.append(child)
    flush(end_line)
    return chunks


# --------------------------------------------------------------------------- #
# Chunk construction                                                           #
# --------------------------------------------------------------------------- #
def _make_chunk(
    *,
    repo: str,
    rel_path: str,
    language: str,
    kind: str,
    content: str,
    start_line: int,
    end_line: int,
    symbol_name: Optional[str] = None,
    parent_symbol: Optional[str] = None,
) -> Chunk:
    return Chunk(
        repo=repo,
        file_path=rel_path,
        language=language,
        kind=kind,
        content=content,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        parent_symbol=parent_symbol,
    )


def _emit_definition(
    node,
    source: bytes,
    *,
    repo: str,
    rel_path: str,
    language: str,
    parent_symbol: Optional[str],
    func_types: frozenset[str],
    cls_types: frozenset[str],
) -> list[Chunk]:
    """Emit chunk(s) for one definition node and recurse into containers."""
    start_line, end_line = _lines(node)
    span = end_line - start_line + 1
    name = _node_name(node, source)
    chunks: list[Chunk] = []

    if node.type in cls_types:
        # The container itself is a "class" chunk.
        if span > MAX_CHUNK_LINES:
            chunks.extend(
                _split_large_node(
                    node,
                    source,
                    repo=repo,
                    rel_path=rel_path,
                    language=language,
                    symbol_name=name,
                    parent_symbol=parent_symbol,
                )
            )
        else:
            chunks.append(
                _make_chunk(
                    repo=repo,
                    rel_path=rel_path,
                    language=language,
                    kind="class",
                    content=_node_text(node, source),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=name,
                    parent_symbol=parent_symbol,
                )
            )
        # Recurse for methods / nested classes.
        body = _body_node(node)
        if body is not None:
            child_parent = name or parent_symbol
            for child in body.named_children:
                if child.type in cls_types or child.type in func_types:
                    chunks.extend(
                        _emit_definition(
                            child,
                            source,
                            repo=repo,
                            rel_path=rel_path,
                            language=language,
                            parent_symbol=child_parent,
                            func_types=func_types,
                            cls_types=cls_types,
                        )
                    )
        return chunks

    # Function / method node.
    kind = "method" if parent_symbol else "function"
    # Go attributes methods to their receiver type even though they are top-level.
    if language == "go" and node.type == "method_declaration":
        recv = _go_receiver_type(node, source)
        if recv:
            kind = "method"
            parent_symbol = parent_symbol or recv

    if span > MAX_CHUNK_LINES:
        blocks = _split_large_node(
            node,
            source,
            repo=repo,
            rel_path=rel_path,
            language=language,
            symbol_name=name,
            parent_symbol=parent_symbol,
        )
        chunks.extend(blocks)
    else:
        chunks.append(
            _make_chunk(
                repo=repo,
                rel_path=rel_path,
                language=language,
                kind=kind,
                content=_node_text(node, source),
                start_line=start_line,
                end_line=end_line,
                symbol_name=name,
                parent_symbol=parent_symbol,
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Module-level (leftover) chunking                                             #
# --------------------------------------------------------------------------- #
def _module_chunks(
    covered: set[int],
    src_lines: list[str],
    *,
    repo: str,
    rel_path: str,
    language: str,
) -> list[Chunk]:
    """Capture top-level lines not covered by any definition as module chunks."""
    chunks: list[Chunk] = []
    total = len(src_lines)
    run_start: Optional[int] = None  # 0-indexed

    def flush(run_end: int) -> None:  # run_end is exclusive, 0-indexed
        nonlocal run_start
        if run_start is None:
            return
        block = src_lines[run_start:run_end]
        if any(line.strip() for line in block):
            # Split very large module runs along the line window.
            i = 0
            n = len(block)
            while i < n:
                sub = block[i : i + MAX_CHUNK_LINES]
                chunks.append(
                    _make_chunk(
                        repo=repo,
                        rel_path=rel_path,
                        language=language,
                        kind="module",
                        content="\n".join(sub),
                        start_line=run_start + i + 1,
                        end_line=run_start + i + len(sub),
                    )
                )
                i += MAX_CHUNK_LINES
        run_start = None

    for idx in range(total):
        if idx in covered:
            flush(idx)
        else:
            if run_start is None:
                run_start = idx
    flush(total)
    return chunks


# --------------------------------------------------------------------------- #
# Line-window fallback                                                         #
# --------------------------------------------------------------------------- #
def _line_window_chunks(
    text: str,
    *,
    repo: str,
    rel_path: str,
    language: str,
    window: int = WINDOW_LINES,
    overlap: int = WINDOW_OVERLAP,
) -> list[Chunk]:
    """Overlapping fixed-size windows -> ``block`` chunks. Never drops content."""
    lines = text.split("\n")
    # Drop a trailing empty line produced by a final newline.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n = len(lines)
    if n == 0:
        return []
    step = max(1, window - overlap)
    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + window, n)
        sub = lines[start:end]
        if any(line.strip() for line in sub):
            chunks.append(
                _make_chunk(
                    repo=repo,
                    rel_path=rel_path,
                    language=language,
                    kind="block",
                    content="\n".join(sub),
                    start_line=start + 1,
                    end_line=end,
                )
            )
        if end >= n:
            break
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def chunk_file(
    path: pathlib.Path | str,
    repo: str,
    rel_path: str,
    language: str | None = None,
) -> list[Chunk]:
    """Chunk a single file into :class:`Chunk`s.

    Args:
        path: Absolute path to the file on disk.
        repo: Repository label stored on each chunk (e.g. ``"owner/name"``).
        rel_path: Path relative to the repo root, stored as ``Chunk.file_path``.
        language: Optional language id. Inferred from the extension if omitted.

    Falls back to the line-window chunker when the language has no grammar, when
    tree-sitter is unavailable, or when parsing fails -- the file is always
    chunked.
    """
    path = pathlib.Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read %s (%s); skipping", path, exc)
        return []
    text = raw.decode("utf-8", errors="replace")

    if language is None:
        language = language_for(path) or "text"

    use_ast = _TS_AVAILABLE and language in TREE_SITTER_LANGUAGES and language in LANG_NODES
    if not use_ast:
        return _line_window_chunks(
            text, repo=repo, rel_path=rel_path, language=language
        )

    try:
        parser = _get_parser(language)
        tree = parser.parse(raw)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning(
            "tree-sitter parse failed for %s (%s); using line-window fallback",
            rel_path,
            exc,
        )
        return _line_window_chunks(
            text, repo=repo, rel_path=rel_path, language=language
        )

    node_cfg = LANG_NODES[language]
    func_types = node_cfg[_FUNC]
    cls_types = node_cfg[_CLS]

    src_lines = text.split("\n")
    if src_lines and src_lines[-1] == "":
        src_lines = src_lines[:-1]

    chunks: list[Chunk] = []
    covered: set[int] = set()  # 0-indexed lines claimed by a top-level definition

    for child in tree.root_node.named_children:
        if child.type in cls_types or child.type in func_types:
            for line in range(child.start_point[0], child.end_point[0] + 1):
                covered.add(line)
            chunks.extend(
                _emit_definition(
                    child,
                    raw,
                    repo=repo,
                    rel_path=rel_path,
                    language=language,
                    parent_symbol=None,
                    func_types=func_types,
                    cls_types=cls_types,
                )
            )

    # Everything not inside a top-level definition -> module chunks.
    chunks.extend(
        _module_chunks(
            covered,
            src_lines,
            repo=repo,
            rel_path=rel_path,
            language=language,
        )
    )

    # If the file had no recognizable structure at all, fall back so nothing is
    # lost with an unhelpful single giant module chunk.
    if not chunks and text.strip():
        return _line_window_chunks(
            text, repo=repo, rel_path=rel_path, language=language
        )

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def chunk_repo(repo_path: pathlib.Path | str, repo: str) -> Iterator[Chunk]:
    """Walk ``repo_path`` and yield chunks for every indexable file.

    Args:
        repo_path: Local path to the (cloned) repository root.
        repo: Repository label stored on each chunk (e.g. ``"owner/name"``).
    """
    root = pathlib.Path(repo_path).resolve()
    for abs_path, language in walk_files(root):
        rel_path = str(abs_path.relative_to(root))
        yield from chunk_file(abs_path, repo=repo, rel_path=rel_path, language=language)
