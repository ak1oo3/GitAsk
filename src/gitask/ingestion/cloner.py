"""Clone (or refresh) a GitHub repository into a local cache.

Repositories are shallow-cloned (``depth=1``) into ``settings.clone_dir`` under
an ``owner/name`` layout so re-indexing the same repo reuses the working copy.
"""

from __future__ import annotations

import logging
import pathlib
import re

from gitask.config import settings

logger = logging.getLogger(__name__)

try:  # GitPython is optional at import time so the module always imports.
    import git  # type: ignore

    _GIT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without GitPython
    git = None  # type: ignore
    _GIT_AVAILABLE = False
    logger.warning("GitPython unavailable (%s); clone_repo will raise if called", exc)


_URL_RE = re.compile(
    r"""
    (?:git@|https?://|ssh://git@)?   # optional scheme / scp-like prefix
    (?:[^/:]+[:/])?                  # host, e.g. github.com/ or github.com:
    (?P<owner>[^/]+)                 # owner
    /
    (?P<name>[^/]+?)                 # repo name
    (?:\.git)?                       # optional .git suffix
    /?$
    """,
    re.VERBOSE,
)


def repo_name_from_url(url: str) -> str:
    """Return ``"owner/name"`` for a git URL or ``owner/name`` shorthand.

    Handles ``https://github.com/owner/name(.git)``, ``git@github.com:owner/name``
    and bare ``owner/name`` forms. Falls back to the last path segment when the
    URL is unusual so callers always get a usable, filesystem-safe label.
    """
    cleaned = url.strip().rstrip("/")
    match = _URL_RE.search(cleaned)
    if match:
        owner = match.group("owner")
        name = match.group("name")
        # Guard against the host being captured as the owner for host-only URLs.
        if owner and name and "." not in name:
            return f"{owner}/{name}"
    # Fallback: derive from the trailing path components.
    parts = [p for p in re.split(r"[/:]", cleaned) if p]
    if len(parts) >= 2:
        name = parts[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return f"{parts[-2]}/{name}"
    return cleaned


def _refresh(repo_dir: pathlib.Path) -> bool:
    """Best-effort fast refresh of an existing shallow clone. Returns success."""
    try:
        repo = git.Repo(repo_dir)  # type: ignore[union-attr]
        origin = repo.remotes.origin
        origin.fetch(depth=1, prune=True)
        default_ref = origin.refs[0]
        repo.git.reset("--hard", default_ref.name)
        logger.info("Refreshed existing clone at %s", repo_dir)
        return True
    except Exception as exc:  # network / detached state / etc.
        logger.warning("Could not refresh %s (%s); reusing existing copy", repo_dir, exc)
        return False


def clone_repo(url: str, dest_dir: str | None = None) -> pathlib.Path:
    """Shallow-clone ``url`` into the cache and return the local path.

    Args:
        url: Repository URL (or ``owner/name`` for GitHub).
        dest_dir: Base cache directory. Defaults to ``settings.clone_dir``.

    If the repo is already present it is refreshed (best effort) and reused.
    """
    if not _GIT_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "GitPython is not installed; cannot clone. Install with "
            "`pip install GitPython`, or pass a local path instead of a URL."
        )

    base = pathlib.Path(dest_dir or settings.clone_dir).expanduser().resolve()
    name = repo_name_from_url(url)
    target = base / name
    target.parent.mkdir(parents=True, exist_ok=True)

    if (target / ".git").exists():
        _refresh(target)
        return target

    # Normalize GitHub shorthand to a clone URL.
    clone_url = url
    if "://" not in url and "@" not in url:
        clone_url = f"https://github.com/{name}.git"

    logger.info("Cloning %s -> %s (depth=1)", clone_url, target)
    if target.exists():  # stale, non-git directory
        import shutil

        shutil.rmtree(target)
    git.Repo.clone_from(clone_url, target, depth=1)  # type: ignore[union-attr]
    return target
