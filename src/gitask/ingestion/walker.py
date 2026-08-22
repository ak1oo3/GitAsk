"""Walk a repository tree and yield indexable source files.

The walker is deliberately conservative: it skips version-control internals,
vendored/generated directories, binaries, lockfiles and minified bundles so the
chunker only ever sees real, human-authored source. Each yielded item is a
``(absolute_path, language_id)`` pair; the ``language_id`` is a tree-sitter
grammar name where one exists, otherwise a coarse text label (``"text"``) that
tells the chunker to use its line-window fallback.
"""

from __future__ import annotations

import pathlib
from typing import Iterator, Optional

from gitask.config import settings

# --------------------------------------------------------------------------- #
# Extension -> language id mapping                                             #
# --------------------------------------------------------------------------- #
# Languages for which tree-sitter-languages ships a prebuilt grammar. The
# chunker uses these ids directly with ``get_parser``.
EXT_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    # JVM
    ".java": "java",
    # Go
    ".go": "go",
    # Ruby
    ".rb": "ruby",
    # Rust
    ".rs": "rust",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}

# Grammars actually supported by the AST chunker. Everything else routes to the
# line-window fallback.
TREE_SITTER_LANGUAGES: frozenset[str] = frozenset(EXT_TO_LANGUAGE.values())

# Text-ish files we still index (via the fallback) even without a grammar. The
# value is the coarse ``language_id`` reported to the chunker.
EXT_TO_TEXT: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".xml": "xml",
    ".env": "text",
    ".dockerfile": "dockerfile",
}

# Filenames (no extension, or special) worth indexing as text.
FILENAME_TO_TEXT: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "README": "text",
}

# --------------------------------------------------------------------------- #
# Exclusions                                                                   #
# --------------------------------------------------------------------------- #
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "out",
        "vendor",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env.d",
        ".repos",
        "target",  # rust / java
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".next",
        ".nuxt",
        "coverage",
        "site-packages",
        ".eggs",
        ".gradle",
    }
)

SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        "go.sum",
        "Gemfile.lock",
    }
)

# Extensions that are always binary / never useful to chunk.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
        ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".rar",
        ".mp3", ".mp4", ".wav", ".mov", ".avi", ".mkv", ".flac", ".ogg",
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class",
        ".pyc", ".pyo", ".pyd", ".whl", ".egg", ".jar", ".war",
        ".db", ".sqlite", ".sqlite3", ".parquet", ".npy", ".npz",
        ".lock",  # generic lock artifacts
    }
)


def _is_minified(name: str) -> bool:
    """Heuristic for generated/minified bundles we should not chunk."""
    lowered = name.lower()
    return (
        ".min." in lowered
        or lowered.endswith(".min.js")
        or lowered.endswith(".min.css")
        or ".bundle." in lowered
    )


def language_for(path: pathlib.Path) -> Optional[str]:
    """Return the language id for a path, or ``None`` if it should be skipped.

    A tree-sitter grammar id is returned when available; otherwise a coarse
    text label for files the chunker can still line-window. ``None`` means the
    file is not text we want to index.
    """
    name = path.name
    suffix = path.suffix.lower()

    if suffix in BINARY_EXTENSIONS:
        return None
    if suffix in EXT_TO_LANGUAGE:
        return EXT_TO_LANGUAGE[suffix]
    if suffix in EXT_TO_TEXT:
        return EXT_TO_TEXT[suffix]
    if name in FILENAME_TO_TEXT:
        return FILENAME_TO_TEXT[name]
    # Dotfiles like ``.gitignore`` / ``.editorconfig`` -> plain text.
    if suffix == "" and name.startswith("."):
        return "text"
    return None


def _looks_binary(path: pathlib.Path) -> bool:
    """Sniff the first bytes for a NUL, a reliable binary signal."""
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return True


def walk_files(
    repo_path: pathlib.Path | str,
    max_bytes: int | None = None,
) -> Iterator[tuple[pathlib.Path, str]]:
    """Yield ``(absolute_path, language_id)`` for every indexable source file.

    Args:
        repo_path: Root of the repository to walk.
        max_bytes: Skip files larger than this. Defaults to
            ``settings.max_file_bytes``.

    Skips: version-control internals, vendored/generated directories, binaries,
    lockfiles, minified bundles and oversized files.
    """
    root = pathlib.Path(repo_path).resolve()
    limit = settings.max_file_bytes if max_bytes is None else max_bytes

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        # Skip anything inside an excluded directory.
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        name = path.name
        if name in SKIP_FILENAMES:
            continue
        if _is_minified(name):
            continue

        language = language_for(path)
        if language is None:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > limit:
            continue

        # Grammar-backed extensions are trusted as text; only sniff the rest.
        if language not in TREE_SITTER_LANGUAGES and _looks_binary(path):
            continue

        yield path.resolve(), language
