"""LLM answer synthesis (Milestone 4a).

`LLMClient` turns a question plus a set of retrieved code chunks into a grounded
`Answer` with `file:line` citations. The Claude call is injectable (`_complete`)
so the whole class is unit-testable with no network, no API key, and no
`anthropic` package installed.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from gitask.config import settings
from gitask.models import Answer, Citation, Chunk, RetrievedChunk

# A "file:line" or "file:start-end" citation label emitted by the model.
# Matches paths like  src/flask/app.py:120-168  or  README.md:12
_CITATION_RE = re.compile(
    r"(?P<path>[\w./\-]+\.[A-Za-z0-9_]+):(?P<start>\d+)(?:-(?P<end>\d+))?"
)

# Signature of the injectable completion function used by tests.
CompleteFn = Callable[[str, str, str], str]  # (model, system, user) -> answer text


class LLMClient:
    """Grounded answer generator backed by the Anthropic Messages API.

    Parameters
    ----------
    model:
        Overrides ``settings.anthropic_model`` when given.
    api_key:
        Overrides ``settings.anthropic_api_key`` when given.
    _complete:
        Optional callable ``(model, system, user) -> str`` used *instead* of the
        real Anthropic client. Tests inject a fake here; production leaves it
        ``None`` and a lazily-constructed Anthropic client is used.
    max_tokens:
        Upper bound on generated tokens for the answer.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        _complete: Optional[CompleteFn] = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model or settings.anthropic_model
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.max_tokens = max_tokens
        self._complete = _complete
        self._client = None  # lazily constructed Anthropic client

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        """Generate a grounded answer for ``question`` from ``retrieved`` chunks."""
        # Import lazily so this module loads even before the retrieval layer's
        # heavy deps are installed.
        from gitask.retrieval import prompt as prompt_mod

        system = prompt_mod.SYSTEM_PROMPT
        user = prompt_mod.build_user_prompt(question, retrieved)

        text = self._run_completion(system, user)
        citations = self._derive_citations(text, retrieved)
        return Answer(text=text, citations=citations, retrieved=list(retrieved))

    # ------------------------------------------------------------------ #
    # Completion (network) — injectable                                  #
    # ------------------------------------------------------------------ #
    def _run_completion(self, system: str, user: str) -> str:
        if self._complete is not None:
            return self._complete(self.model, system, user)
        return self._anthropic_complete(system, user)

    def _anthropic_complete(self, system: str, user: str) -> str:
        """Real Anthropic Messages API call. Only reached in production."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(parts).strip()

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily; not needed for tests

            if not self.api_key:
                raise RuntimeError(
                    "No Anthropic API key configured. Set ANTHROPIC_API_KEY in the "
                    "environment / .env, or inject a `_complete` callable."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------ #
    # Citation extraction                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _derive_citations(
        text: str, retrieved: list[RetrievedChunk]
    ) -> list[Citation]:
        """Match ``file:line`` labels the model emitted back to retrieved chunks.

        A label matches a chunk when the file paths agree (exact, suffix, or
        basename) and, when line numbers are present, the cited line overlaps the
        chunk's line range. Falls back to *all* retrieved chunks as citations when
        no label could be matched (so an answer is never left uncited).
        """
        chunks = [rc.chunk for rc in retrieved]
        matched: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()

        for m in _CITATION_RE.finditer(text):
            path = m.group("path")
            start = int(m.group("start"))
            end = int(m.group("end")) if m.group("end") else start
            chunk = LLMClient._match_chunk(path, start, end, chunks)
            if chunk is None:
                continue
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            if key in seen:
                continue
            seen.add(key)
            matched.append(Citation.from_chunk(chunk))

        if matched:
            return matched
        # Fallback: cite everything we retrieved.
        return [Citation.from_chunk(c) for c in chunks]

    @staticmethod
    def _match_chunk(
        path: str, start: int, end: int, chunks: list[Chunk]
    ) -> Optional[Chunk]:
        """Find the best retrieved chunk for a cited ``path:start-end`` label."""

        def path_matches(chunk_path: str) -> bool:
            if chunk_path == path:
                return True
            if chunk_path.endswith("/" + path) or path.endswith("/" + chunk_path):
                return True
            return chunk_path.rsplit("/", 1)[-1] == path.rsplit("/", 1)[-1]

        candidates = [c for c in chunks if path_matches(c.file_path)]
        if not candidates:
            return None

        # Prefer a chunk whose line range overlaps the cited lines.
        for c in candidates:
            if c.start_line <= end and start <= c.end_line:
                return c
        # Otherwise the nearest chunk in the same file.
        return min(candidates, key=lambda c: abs(c.start_line - start))
