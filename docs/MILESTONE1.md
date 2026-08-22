# Milestone 1 — Repo cloning + AST-aware chunking

This milestone turns a GitHub repository into a stream of semantically coherent
`Chunk` objects (defined in `src/gitask/models.py`), ready for embedding and
indexing in later milestones.

Pipeline: **clone → walk → chunk**.

```
gitask.ingestion.cloner.clone_repo(url)      -> local checkout (shallow)
gitask.ingestion.walker.walk_files(path)     -> (abs_path, language_id) pairs
gitask.ingestion.chunker.chunk_file(...)     -> list[Chunk] for one file
gitask.ingestion.chunker.chunk_repo(path)    -> Iterator[Chunk] for the repo
```

## Chunking strategy

Each file is parsed with **tree-sitter** and chunked along syntax boundaries:

- a top-level function/def → a `"function"` chunk;
- a class / struct / impl / interface / namespace → a `"class"` chunk, and
  every function defined inside it → a `"method"` chunk that carries
  `parent_symbol` (the enclosing container). Nesting recurses, so a method in a
  nested class is attributed to the nearest named container;
- code outside any definition (imports, module-level statements, constants) is
  captured as one or more `"module"` chunks — **nothing is dropped**;
- a definition longer than `MAX_CHUNK_LINES` (200) is split into `"block"`
  sub-chunks along **AST child boundaries** (whole statements), with a
  signature/header block first — never a mid-statement cut.

Every chunk records `repo`, `file_path` (relative), `language`, `kind`,
`symbol_name`, `parent_symbol`, `content`, and 1-indexed inclusive
`start_line`/`end_line`. `chunk_id` is filled by `Chunk.__post_init__`.

### Why AST over fixed-size windows

A fixed *N*-line window slices through the middle of functions and statements,
so a retrieved chunk seldom corresponds to a coherent unit of code and its
citation (`file.py:120-180`) points at an arbitrary range. Chunking on syntax
boundaries means every chunk is a whole function, method or class — a unit a
reader and an LLM can reason about, and a citation that lands on something
meaningful. `symbol_name`/`kind`/`parent_symbol` also give retrieval and the UI
real structure to filter and display.

## Languages supported (tree-sitter)

`python`, `javascript`, `typescript`, `tsx`, `java`, `go`, `ruby`, `rust`,
`c`, `cpp` — via the prebuilt grammars in `tree-sitter-languages`.

Language-specific handling worth noting:

- **Go** methods are top-level `method_declaration`s; the chunker reads the
  receiver type and attributes them as `"method"` with `parent_symbol` = the
  receiver (e.g. `Server`).
- **C/C++** function names live inside nested declarators; the name extractor
  walks the `declarator` chain to recover them.
- **Rust** `impl` blocks and Go `type` declarations have no `name` field; names
  are recovered from their `type_identifier` / `type_spec` children.

## The fallback

For files with **no grammar** (markdown, yaml, json, plain text, shell, …) or
when **tree-sitter is unavailable** or **a parse fails**, the file is chunked by
an overlapping **line-window** chunker (60-line windows, 10-line overlap)
producing `"block"` chunks. A file is *always* chunked, never silently skipped.

The tree-sitter import is guarded: if the grammar package is missing the module
still imports and every file routes through the line-window fallback (logged).

## The walker

`walk_files` skips: `.git`, `node_modules`, `dist`, `build`, `vendor`,
`__pycache__`, `.venv` (and more), lockfiles (`package-lock.json`, `yarn.lock`,
`poetry.lock`, `Cargo.lock`, `go.sum`, …), minified/bundled files
(`*.min.js`, `*.bundle.*`), binaries (by extension and by NUL-byte sniff), and
files larger than `settings.max_file_bytes`. Extensions map to tree-sitter
language ids; text-ish files without a grammar get a coarse label for the
fallback.

## Verification

- **Tests** — `tests/test_chunker.py` builds small Python, Go and JavaScript
  sources and asserts the recovered functions/classes/methods, their `kind`s,
  `symbol_name`s, `parent_symbol`s and line ranges; asserts module-level code is
  captured; asserts the line-window fallback for an unsupported extension; and
  asserts oversized functions are block-split within the cap with complete
  coverage.

  ```
  $ python -m pytest tests/test_chunker.py -q
  .......                                                                  [100%]
  7 passed in 0.34s
  ```

- **Manual chunk-quality check** — `scripts/inspect_chunks.py <url|owner/name|path>`
  clones (if needed), chunks the whole repo, and prints totals, per-language and
  per-kind counts, and a sample of chunks. Cloning was verified end-to-end
  against a real GitHub repo (`octocat/Hello-World`); the sample below is the
  tool run against the GitAsk repo itself.

### Sample `inspect_chunks.py` output

```
$ python scripts/inspect_chunks.py . --sample 8 --lang python

Repo:        user/GitAsk
Local path:  /home/user/GitAsk
Chunking...

Files chunked: 33
Total chunks:  224

Per-language chunk counts:
  python         212
  html           7
  text           2
  markdown       1
  yaml           1
  sql            1

Per-kind chunk counts:
  function       79
  module         56
  method         54
  class          23
  block          12

Sample of chunks:
----------------------------------------------------------------------------------
[function] _resolve_source
           scripts/inspect_chunks.py:33-42
           def _resolve_source(target: str) -> tuple[pathlib.Path, str]:
----------------------------------------------------------------------------------
[class   ] Components
           src/gitask/api/main.py:40-85
           class Components:  ↵  """Holds the pipeline's heavy components.
----------------------------------------------------------------------------------
[method  ] __init__ (in Components)
           src/gitask/api/main.py:48-61
           def __init__(  ↵      self,
----------------------------------------------------------------------------------
[function] get_components
           src/gitask/api/main.py:92-94
           def get_components() -> Components:
----------------------------------------------------------------------------------
```

The sample shows the key property: chunks land on real symbols
(`function`/`class`/`method` with names and parents) with tight, citable line
ranges, and module-level code is preserved rather than dropped.

## Files in this milestone

- `src/gitask/ingestion/__init__.py` — public surface re-exports
- `src/gitask/ingestion/cloner.py` — `clone_repo`, `repo_name_from_url`
- `src/gitask/ingestion/walker.py` — `walk_files` + extension/skip rules
- `src/gitask/ingestion/chunker.py` — `chunk_file`, `chunk_repo` (AST + fallback)
- `tests/test_chunker.py` — chunker tests
- `scripts/inspect_chunks.py` — manual chunk-quality inspector
- `docs/MILESTONE1.md` — this document

## Dependencies / notes

Requires `tree-sitter` **0.21.x** (pinned `>=0.21,<0.22` in `requirements.txt`)
paired with `tree-sitter-languages>=1.10.2`, plus `GitPython`. Note:
`tree-sitter-languages` 1.10.2 is built against the 0.21/0.22 grammar ABI — a
plain `pip install tree-sitter` pulls 0.26, whose changed `Language`
constructor breaks `get_parser`. Install the pinned range from
`requirements.txt`. (The one benign side effect at 0.21.x is a
`Language(path, name) is deprecated` `FutureWarning`, which the chunker
suppresses around parser construction.)
