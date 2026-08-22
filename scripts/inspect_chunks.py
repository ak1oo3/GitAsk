#!/usr/bin/env python3
"""Manually inspect chunk quality for a repo (Milestone 1 verification tool).

Give it a GitHub URL, an ``owner/name`` shorthand, or a local path. It clones
(if needed), chunks the whole repo, and prints a readable summary: total files,
total chunks, per-language counts and a sample of chunks showing symbol name,
kind, ``file:line-range`` and the first two lines of content.

Usage::

    python scripts/inspect_chunks.py https://github.com/psf/requests
    python scripts/inspect_chunks.py owner/name
    python scripts/inspect_chunks.py .                 # a local checkout
    python scripts/inspect_chunks.py . --sample 40 --lang python
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

# Make the ``src/`` layout importable when run directly.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gitask.ingestion.chunker import chunk_repo  # noqa: E402
from gitask.ingestion.cloner import clone_repo, repo_name_from_url  # noqa: E402


def _resolve_source(target: str) -> tuple[pathlib.Path, str]:
    """Return (local_repo_path, repo_label) for a URL / shorthand / local path."""
    local = pathlib.Path(target).expanduser()
    if local.exists() and local.is_dir():
        label = repo_name_from_url(str(local.resolve()))
        return local.resolve(), label
    # Treat as a remote repo (URL or owner/name shorthand).
    label = repo_name_from_url(target)
    path = clone_repo(target)
    return path, label


def _first_lines(content: str, n: int = 2) -> str:
    lines = [ln.rstrip() for ln in content.split("\n") if ln.strip()]
    snippet = " ↵ ".join(lines[:n])
    return snippet[:110]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="GitHub URL, owner/name, or local path")
    parser.add_argument(
        "--sample", type=int, default=25, help="how many sample chunks to print"
    )
    parser.add_argument(
        "--lang", default=None, help="only sample chunks of this language id"
    )
    args = parser.parse_args(argv)

    repo_path, label = _resolve_source(args.target)
    print(f"Repo:        {label}")
    print(f"Local path:  {repo_path}")
    print("Chunking...\n")

    chunks = list(chunk_repo(repo_path, repo=label))

    files = {c.file_path for c in chunks}
    by_lang = Counter(c.language for c in chunks)
    by_kind = Counter(c.kind for c in chunks)

    print(f"Files chunked: {len(files)}")
    print(f"Total chunks:  {len(chunks)}")

    print("\nPer-language chunk counts:")
    for lang, count in by_lang.most_common():
        print(f"  {lang:<14} {count}")

    print("\nPer-kind chunk counts:")
    for kind, count in by_kind.most_common():
        print(f"  {kind:<14} {count}")

    sample = chunks
    if args.lang:
        sample = [c for c in chunks if c.language == args.lang]
    sample = sample[: args.sample]

    print(f"\nSample of {len(sample)} chunks:")
    print("-" * 100)
    for c in sample:
        sym = c.symbol_name or "-"
        parent = f" (in {c.parent_symbol})" if c.parent_symbol else ""
        loc = f"{c.file_path}:{c.start_line}-{c.end_line}"
        print(f"[{c.kind:<8}] {sym}{parent}")
        print(f"           {loc}")
        print(f"           {_first_lines(c.content)}")
        print("-" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
