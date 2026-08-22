"""Grounded-answer prompt construction.

The whole point of RAG here is *grounding*: the model must answer only from the
retrieved code and cite exactly where each claim comes from, so a user can
verify it and so the model refuses rather than hallucinating when the context
is insufficient. :data:`SYSTEM_PROMPT` states those rules; :func:`build_user_prompt`
lays out the retrieved chunks with citable headers followed by the question.
"""

from __future__ import annotations

from gitask.models import RetrievedChunk

SYSTEM_PROMPT = """\
You are GitAsk, a precise code question-answering assistant.

You will be given a QUESTION about a codebase and a set of CODE CONTEXT chunks
retrieved from that codebase. Follow these rules strictly:

1. GROUNDING: Answer ONLY using the provided CODE CONTEXT. Do not use outside
   knowledge or assumptions about how the code "probably" works. If the context
   does not contain enough information to answer, say so plainly (e.g. "I can't
   find that in the provided code.") and stop. Do not guess.

2. NO INVENTION: Never invent file paths, function names, classes, or behavior
   that do not appear in the context. Only refer to symbols and files that are
   actually present in the chunks below.

3. CITATIONS: Support every concrete claim with an inline citation using the
   `file_path:start-end` label shown in each chunk's header, for example
   `src/auth.py:12-40`. Place the citation right after the statement it backs.
   If several chunks support a point, cite each.

4. STYLE: Be concise and technical. Quote or reference specific identifiers
   from the code. Prefer explaining what the code actually does over general
   commentary.
"""

# Keep prompts within a sane budget so we don't blow the context window. Bodies
# longer than this are truncated with a marker; headers/citations are preserved.
_MAX_CHARS_PER_CHUNK = 2000
_MAX_TOTAL_CHARS = 24000


def _chunk_header(index: int, rc: RetrievedChunk) -> str:
    """Build the citable header line for a retrieved chunk."""
    c = rc.chunk
    label = c.citation_label  # e.g. src/auth.py:12-40
    symbol = c.symbol_name or ""
    kind_symbol = f"{c.kind} {symbol}".strip()
    return f"[{index}] [{label}  {kind_symbol}]".rstrip()


def _truncate(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "\n... (truncated)"


def build_user_prompt(
    question: str, retrieved: list[RetrievedChunk]
) -> str:
    """Format retrieved chunks + the question into the user message.

    Each chunk is numbered and prefixed with a header of the form
    ``[N] [file_path:start-end  kind symbol_name]`` so the model can cite it,
    followed by the (optionally truncated) source. A running character budget
    caps total size.
    """
    parts: list[str] = ["CODE CONTEXT:\n"]

    if not retrieved:
        parts.append("(no code context was retrieved)\n")
    else:
        total = 0
        for i, rc in enumerate(retrieved, start=1):
            header = _chunk_header(i, rc)
            body = _truncate(rc.chunk.content, _MAX_CHARS_PER_CHUNK)
            block = f"{header}\n```\n{body}\n```\n"
            if total + len(block) > _MAX_TOTAL_CHARS and i > 1:
                parts.append("(additional chunks omitted for length)\n")
                break
            parts.append(block)
            total += len(block)

    parts.append(
        "\nQUESTION:\n"
        f"{question}\n\n"
        "Answer using only the code context above, and cite sources with "
        "`file_path:line-range` labels. If the answer is not in the context, "
        "say you cannot find it."
    )
    return "\n".join(parts)
