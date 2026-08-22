"""Code-aware tokenization for BM25.

BM25 only rescues exact identifier matches if the identifier survives
tokenization. A naive whitespace split leaves ``login_user`` and
``getUserById`` as single opaque tokens, so a query typed as ``login_user``
would match but ``login`` alone would not, and camelCase queries would miss.

:func:`tokenize_code` therefore emits **both** the whole identifier and its
sub-words: it splits on non-alphanumeric characters (``.``, ``_``, ``/``,
parens, etc.) and on camelCase / PascalCase boundaries, lowercases everything,
and keeps the original joined identifier too so an exact ``login_user`` query
still scores a direct hit.
"""

from __future__ import annotations

import re

# Whole identifiers, underscores included, so ``login_user`` survives intact;
# other punctuation (``. / ( ) : ,`` ...) is a separator.
_IDENT_RE = re.compile(r"[A-Za-z0-9_]+")
# camelCase / PascalCase / digit boundaries, e.g. "getUserByID2" ->
# get | User | By | ID | 2
_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"  # acronym boundary: "IDToken" -> "ID"
    r"|[A-Z]?[a-z]+"          # normal word: "User", "get"
    r"|[A-Z]+"                # trailing all-caps: "ID"
    r"|[0-9]+"                # digit runs
)


def _split_camel(word: str) -> list[str]:
    parts = _CAMEL_RE.findall(word)
    return parts or [word]


def tokenize_code(text: str) -> list[str]:
    """Tokenize source text / a query into BM25 terms.

    Returns lowercased tokens including both whole identifiers and their
    camelCase/snake_case sub-words, so exact-name queries and partial-word
    queries both find their target.
    """
    tokens: list[str] = []
    for ident in _IDENT_RE.findall(text):
        low = ident.lower()
        tokens.append(low)  # keep the whole identifier (snake_case joined)
        # snake_case parts, then camelCase within each part.
        snake_parts = [p for p in ident.split("_") if p]
        for part in snake_parts:
            plow = part.lower()
            if plow != low:
                tokens.append(plow)
            camel = _split_camel(part)
            if len(camel) > 1:
                tokens.extend(p.lower() for p in camel)
    return tokens
