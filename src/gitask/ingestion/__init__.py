"""Ingestion layer for GitAsk (Milestone 1).

Pipeline: clone a repo -> walk its source files -> chunk each file along
function/class boundaries with tree-sitter (with a line-window fallback).

Public surface::

    from gitask.ingestion.cloner import clone_repo, repo_name_from_url
    from gitask.ingestion.walker import walk_files
    from gitask.ingestion.chunker import chunk_file, chunk_repo
"""

from __future__ import annotations

from gitask.ingestion.chunker import chunk_file, chunk_repo
from gitask.ingestion.cloner import clone_repo, repo_name_from_url
from gitask.ingestion.walker import walk_files

__all__ = [
    "clone_repo",
    "repo_name_from_url",
    "walk_files",
    "chunk_file",
    "chunk_repo",
]
