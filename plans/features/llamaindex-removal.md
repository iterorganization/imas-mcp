# LlamaIndex Dependency Removal

> **Goal**: Remove all 4 `llama-index-*` packages from imas-codex, replacing them with direct tree-sitter and Cypher integrations. Clean up deprecated OpenRouter/session code along the way.

## Strategic Context

### Why Remove LlamaIndex

LlamaIndex was adopted early in the project as a rapid-prototyping framework. As imas-codex matured, direct patterns emerged that are simpler, more transparent, and already proven in production:

| LlamaIndex Provides | imas-codex Already Has |
|---------------------|------------------------|
| `CodeSplitter` — wraps `tree_sitter_language_pack.get_parser()` | Direct access to same library (`tree-sitter-language-pack ≥0.13.0`) |
| `SentenceSplitter` — sliding-window text chunker | Trivially replaceable (~20 LOC) |
| `IngestionPipeline` — sequential transform chain | Wiki pipeline already bypasses it: loop + direct Cypher |
| `Neo4jVectorStore` — writes/reads embeddings | Wiki pipeline uses `UNWIND` Cypher directly; signal discovery uses `db.index.vector.queryNodes()` directly |
| `VectorStoreIndex` + `MetadataFilters` | `db.index.vector.queryNodes()` + `WHERE` clauses (proven in `domain_queries.py`, `signals/parallel.py`) |
| `Document` / `BaseNode` — data containers | Plain dicts or Pydantic models |
| `TransformComponent` — pipeline stage interface | `Callable[[list[dict]], list[dict]]` |
| `BaseEmbedding` — embedding interface | `Encoder` class (framework-independent, already exists) |
| `OpenRouter` LLM | `create_litellm_model()` via LiteLLM (already migrated) |

**LlamaIndex features NOT used**: RAG orchestration, query engines, agent framework, streaming, callbacks, caching, deduplication, prompt templates, response synthesis.

### Impact on tree-sitter-gdl

Removing LlamaIndex makes tree-sitter-gdl **more valuable, not less**. Here's why:

- tree-sitter-gdl is a **tree-sitter plugin**, not a LlamaIndex plugin. It produces a `.so` grammar library that `tree-sitter-language-pack.get_parser()` or `tree_sitter.Language()` loads directly.
- LlamaIndex's `CodeSplitter` was a thin wrapper around tree-sitter. Post-removal, imas-codex will call tree-sitter directly, giving us full control over AST walking, chunk boundary selection, and custom language registration.
- With direct tree-sitter integration, adding tree-sitter-gdl is a one-line parser registration + removing `"idl"` from `TEXT_SPLITTER_LANGUAGES`. No LlamaIndex plumbing required.
- The LlamaIndex wrapper was actually an obstacle: `CodeSplitter` only accepts languages from `tree-sitter-language-pack`'s `SupportedLanguage` literal type. Custom grammars required a `parser` kwarg workaround.

## Dependencies to Remove

```toml
# pyproject.toml — all 4 lines deleted
"llama-index-core>=0.14",
"llama-index-vector-stores-neo4jvector>=0.4",
"llama-index-llms-openrouter>=0.4",      # Only used by deprecated get_llm()
"llama-index-readers-file>=0.5.6",        # Completely unused — zero imports
```

## Implementation Phases

### Phase 0: Dead Code Removal (no functional changes)

Remove code that is already dead or deprecated, before touching live integrations. Every step is independently committable.

#### 0a. Remove `llama-index-readers-file` dependency
- **Delete** from `pyproject.toml`: `"llama-index-readers-file>=0.5.6"`
- Zero imports exist. Pure dead dependency.

#### 0b. Remove `llama-index-llms-openrouter` dependency + deprecated files
- **Delete** `imas_codex/llm/llm.py` — deprecated `get_llm()` wrapping OpenRouter
- **Delete** `imas_codex/llm/session.py` — dead `LLMSession` / `CostTracker` (no production consumers; budget tracking reimplemented in `discovery/base/llm.py` and `embeddings/openrouter_embed.py`)
- **Delete** `tests/test_agentic_session.py` — tests dead code
- **Update** `imas_codex/llm/__init__.py`:
  - Remove `from imas_codex.llm.llm import get_llm`
  - Remove `from imas_codex.llm.session import CostTracker, LLMSession, create_session`
  - Remove from `__all__`: `"get_llm"`, `"CostTracker"`, `"LLMSession"`, `"create_session"`
- **Delete** from `pyproject.toml`: `"llama-index-llms-openrouter>=0.4"`

#### 0c. Remove deprecated `_get_scannable_paths()` from scanner.py
- **Delete** the function at `imas_codex/discovery/code/scanner.py` ~L252-280
- Private function, zero callers outside the file.

#### 0d. Migrate `get_schema_context()` caller in server.py
- At `imas_codex/llm/server.py` ~L1766, replace call to deprecated `get_schema_context()` with `get_schema_for_prompt()`
- Then remove `get_schema_context()` from `imas_codex/llm/prompt_loader.py`

---

### Phase 1: Replace Text Chunking (drop `SentenceSplitter`)

Replace LlamaIndex's `SentenceSplitter` with a minimal text chunker. This is used in 3 places:
1. `ingestion/pipeline.py` — fallback for languages without tree-sitter
2. `discovery/wiki/pipeline.py` — wiki page chunking (2 instances)

#### 1a. Create `imas_codex/ingestion/chunkers.py`

A single module with two functions:

```python
"""Text and code chunking via tree-sitter and sliding window.

Replaces LlamaIndex CodeSplitter and SentenceSplitter with direct
tree-sitter-language-pack integration and a simple text chunker.
"""

from dataclasses import dataclass

from tree_sitter_language_pack import get_parser


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    start_line: int
    end_line: int


def chunk_code(
    text: str,
    language: str,
    max_chars: int = 10000,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
) -> list[Chunk]:
    """Chunk source code using tree-sitter AST boundaries.

    Parses the source with tree-sitter, walks top-level AST nodes,
    and accumulates them into chunks that respect function/class
    boundaries. Falls back to text chunking if parsing fails.
    """
    parser = get_parser(language)
    tree = parser.parse(text.encode())
    root = tree.root_node

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0
    current_chars = 0

    for child in root.children:
        child_text = child.text.decode()
        child_chars = len(child_text)

        if current_chars + child_chars > max_chars and current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(Chunk(
                text=chunk_text,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
            ))
            # Overlap: keep last N lines
            overlap = current_lines[-chunk_lines_overlap:]
            current_lines = overlap
            current_start = current_start + len(current_lines) - len(overlap)
            current_chars = sum(len(l) for l in current_lines)

        current_lines.extend(child_text.split("\n"))
        current_chars += child_chars

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunks.append(Chunk(
            text=chunk_text,
            start_line=current_start,
            end_line=current_start + len(current_lines) - 1,
        ))

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 10000,
    chunk_overlap: int = 200,
    separator: str = "\n",
) -> list[Chunk]:
    """Chunk text using a sliding window on separator boundaries.

    Used for languages without tree-sitter support (IDL, TDI)
    and for document content (wiki pages, markdown).
    """
    parts = text.split(separator)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_chars = 0
    current_start = 0

    for i, part in enumerate(parts):
        part_len = len(part) + len(separator)
        if current_chars + part_len > chunk_size and current_parts:
            chunk_text = separator.join(current_parts)
            chunks.append(Chunk(
                text=chunk_text,
                start_line=current_start,
                end_line=current_start + len(current_parts) - 1,
            ))
            # Calculate overlap in parts
            overlap_chars = 0
            overlap_start = len(current_parts)
            for j in range(len(current_parts) - 1, -1, -1):
                overlap_chars += len(current_parts[j]) + len(separator)
                if overlap_chars >= chunk_overlap:
                    overlap_start = j
                    break
            overlap = current_parts[overlap_start:]
            current_start = current_start + overlap_start
            current_parts = list(overlap)
            current_chars = sum(len(p) + len(separator) for p in current_parts)

        current_parts.append(part)
        current_chars += part_len

    if current_parts:
        chunk_text = separator.join(current_parts)
        chunks.append(Chunk(
            text=chunk_text,
            start_line=current_start,
            end_line=current_start + len(current_parts) - 1,
        ))

    return chunks
```

This is approximately 100 lines, replacing `CodeSplitter` (150 LOC wrapper) and `SentenceSplitter` (complex, hundreds of LOC in llama-index-core).

#### 1b. Update `ingestion/pipeline.py` to use `chunkers.py`

Replace:
```python
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
```

With:
```python
from imas_codex.ingestion.chunkers import chunk_code, chunk_text
```

Rewrite `create_pipeline()` to return a simple callable instead of an `IngestionPipeline`.

#### 1c. Update `discovery/wiki/pipeline.py` to use `chunk_text()`

Replace `SentenceSplitter` usage (lines ~810 and ~1637) with `chunk_text()`. The wiki pipeline already does everything else directly — this is the last LlamaIndex dependency in wiki code.

#### 1d. Remove `Document` import

Replace `llama_index.core.Document` with plain dicts:
```python
# Before
doc = Document(text=content, metadata={...})
# After
doc = {"text": content, "metadata": {...}}
```

---

### Phase 2: Replace Vector Store Integration (drop `Neo4jVectorStore`)

#### 2a. Replace vector writes in `ingestion/pipeline.py`

Replace `Neo4jVectorStore` write path with direct Cypher UNWIND — copy the pattern from `discovery/wiki/pipeline.py` lines 980-1005:

```python
with GraphClient() as gc:
    gc.query("""
        UNWIND $chunks AS chunk
        MERGE (c:CodeChunk {id: chunk.id})
        SET c += chunk
        WITH c, chunk
        MATCH (f:Facility {id: chunk.facility_id})
        MERGE (c)-[:AT_FACILITY]->(f)
    """, chunks=chunk_batch)
```

#### 2b. Replace `ChunkSearch` in `ingestion/search.py`

Replace `VectorStoreIndex` + `MetadataFilters` with direct vector search Cypher — the pattern already proven in `graph/domain_queries.py` and `discovery/signals/parallel.py`:

```python
from imas_codex.embeddings import Encoder
from imas_codex.graph import GraphClient

def search_code_chunks(
    query: str,
    top_k: int = 10,
    facility: str | None = None,
    ids_filter: list[str] | None = None,
    min_score: float = 0.5,
) -> list[ChunkSearchResult]:
    encoder = Encoder()
    embedding = encoder.embed_text(query)

    where_clauses = []
    params: dict = {"embedding": embedding, "k": top_k}

    if facility:
        where_clauses.append("node.facility_id = $facility")
        params["facility"] = facility
    if ids_filter:
        where_clauses.append(
            "any(ids IN node.related_ids WHERE ids IN $ids_filter)"
        )
        params["ids_filter"] = ids_filter

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with GraphClient() as gc:
        results = gc.query(f"""
            CALL db.index.vector.queryNodes(
                'code_chunk_embedding', $k, $embedding
            ) YIELD node, score
            {where}
            RETURN node.id AS chunk_id,
                   node.text AS content,
                   node.function_name AS function_name,
                   node.source_file AS source_file,
                   node.facility_id AS facility_id,
                   node.related_ids AS related_ids,
                   node.start_line AS start_line,
                   node.end_line AS end_line,
                   score
            ORDER BY score DESC
        """, **params)

    return [
        ChunkSearchResult(**r)
        for r in results
        if r["score"] >= min_score
    ]
```

This is ~40 lines, replacing ~80 lines of LlamaIndex vector store wrappers. The direct Cypher approach is already used by 3 other modules in the codebase.

#### 2c. Update ChunkSearch consumers

- `llm/tools.py` ~L111: Update import to new function
- `llm/server.py` ~L612: Update import to new function

#### 2d. Remove `Neo4jVectorStore` dependency
- **Delete** from `pyproject.toml`: `"llama-index-vector-stores-neo4jvector>=0.4"`

---

### Phase 3: Remove Embedding Adapter (drop `BaseEmbedding`)

#### 3a. Remove `EncoderEmbedding` wrapper

`imas_codex/embeddings/embed.py` exists solely to wrap the framework-independent `Encoder` class in LlamaIndex's `BaseEmbedding` interface. After Phases 1-2, nothing requires `BaseEmbedding`.

- Refactor `embed.py` to export `Encoder` directly (or a thin `get_encoder()` singleton)
- Update all callers of `get_embed_model()`:
  - `ingestion/pipeline.py` (now using direct chunkers)
  - `discovery/wiki/pipeline.py` (calls `embed_model.get_text_embedding_batch()`)
  - Any other consumers

The `Encoder` class already has `embed_texts()` / `embed_text()` methods that don't depend on LlamaIndex.

#### 3b. Remove `TransformComponent` subclasses

- `ingestion/extractors/ids.py`: Replace `class IDSExtractor(TransformComponent)` with a plain function `extract_ids(nodes: list[dict]) -> list[dict]`
- `ingestion/extractors/mdsplus.py`: Same pattern
- Update tests accordingly

#### 3c. Remove `llama-index-core` dependency
- **Delete** from `pyproject.toml`: `"llama-index-core>=0.14"`
- This is the final LlamaIndex package removal.

---

### Phase 4: Cleanup and Validation

#### 4a. Update `imas_codex/ingestion/__init__.py` exports
- Remove any LlamaIndex re-exports
- Export new chunker functions

#### 4b. Update `readers/remote.py` comments
- Update the comment at L27: "Languages with tree-sitter support" to reflect direct integration

#### 4c. Update `pyproject.toml` tree-sitter deps
- Keep: `"tree-sitter>=0.25.2"`, `"tree-sitter-language-pack>=0.13.0"`
- Evaluate removing: `"tree-sitter-languages>=1.10.2"` (legacy, may still be needed)

#### 4d. Update tree-sitter-gdl plan
- Remove all LlamaIndex integration references from `TREE_SITTER_GDL.md`
- Update integration section: direct `get_parser()` call, not CodeSplitter injection

#### 4e. Run full test suite
```bash
uv run pytest --cov=imas_codex
```

#### 4f. Update AGENTS.md
- Remove LlamaIndex references from ingestion documentation
- Update quick reference for code search

---

## File Impact Summary

### Files to Delete (5)
| File | Reason |
|------|--------|
| `imas_codex/llm/llm.py` | Deprecated `get_llm()` using LlamaIndex OpenRouter |
| `imas_codex/llm/session.py` | Dead `LLMSession` / `CostTracker` — no production consumers |
| `tests/test_agentic_session.py` | Tests for dead code |
| *(none created)* | |

### Files to Create (1)
| File | Purpose |
|------|---------|
| `imas_codex/ingestion/chunkers.py` | Direct tree-sitter + text chunking (~100 LOC) |

### Files to Modify (12)
| File | Changes |
|------|---------|
| `pyproject.toml` | Remove 4 `llama-index-*` deps |
| `imas_codex/llm/__init__.py` | Remove dead re-exports |
| `imas_codex/llm/server.py` | Migrate `get_schema_context()` → `get_schema_for_prompt()` |
| `imas_codex/llm/tools.py` | Update ChunkSearch → direct search |
| `imas_codex/llm/prompt_loader.py` | Remove deprecated `get_schema_context()` |
| `imas_codex/ingestion/pipeline.py` | Replace LlamaIndex pipeline with direct chunking + Cypher |
| `imas_codex/ingestion/search.py` | Replace VectorStoreIndex with direct Cypher vector search |
| `imas_codex/ingestion/__init__.py` | Update exports |
| `imas_codex/ingestion/extractors/ids.py` | Remove `TransformComponent`, use plain function |
| `imas_codex/ingestion/extractors/mdsplus.py` | Remove `TransformComponent`, use plain function |
| `imas_codex/embeddings/embed.py` | Remove `BaseEmbedding` wrapper, expose `Encoder` directly |
| `imas_codex/discovery/wiki/pipeline.py` | Replace `SentenceSplitter` + `Document` with `chunk_text()` |
| `imas_codex/discovery/code/scanner.py` | Remove dead `_get_scannable_paths()` |

### Unchanged (key files)
| File | Why Unchanged |
|------|---------------|
| `imas_codex/embeddings/encoder.py` | Framework-independent — already the real implementation |
| `imas_codex/ingestion/readers/remote.py` | Only `TEXT_SPLITTER_LANGUAGES` set changes (when tree-sitter-gdl lands) |
| `imas_codex/graph/client.py` | Already used by wiki pipeline's direct-Cypher pattern |

## Dependency Comparison

### Before (4 LlamaIndex packages)
```
llama-index-core>=0.14               # ~50MB installed, 200+ transitive deps
llama-index-vector-stores-neo4jvector # neo4j driver (already have via GraphClient)
llama-index-llms-openrouter           # openrouter SDK (replaced by LiteLLM)
llama-index-readers-file              # Unused
tree-sitter>=0.25.2
tree-sitter-language-pack>=0.13.0
```

### After (tree-sitter only)
```
tree-sitter>=0.25.2
tree-sitter-language-pack>=0.13.0
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| CodeSplitter's chunking heuristics are subtly valuable | Our `chunk_code()` uses the same approach (walk children, accumulate by size). Can compare chunk quality on a sample of files before/after. |
| Neo4jVectorStore handles edge cases we'd miss | Vector writes are simple UNWIND SETs — wiki pipeline has been doing this in production. Vector reads are `db.index.vector.queryNodes()` — 3 modules already do this directly. |
| Breaking ChunkSearch API for MCP consumers | `search_code()` MCP tool and `search_code_examples()` agent tool are internal — we control all consumers. |
| Regression in wiki pipeline chunking | Wiki pipeline already barely uses LlamaIndex — only `SentenceSplitter`. Text chunking is trivial to validate. |

## Commit Sequence

Each commit is independently deployable:

1. `chore: remove unused llama-index-readers-file dependency`
2. `refactor: remove deprecated get_llm and LLMSession`
3. `refactor: remove deprecated _get_scannable_paths`
4. `refactor: migrate get_schema_context to get_schema_for_prompt`
5. `feat: add direct tree-sitter and text chunkers`
6. `refactor: replace LlamaIndex pipeline with direct chunking`
7. `refactor: replace wiki SentenceSplitter with chunk_text`
8. `refactor: replace ChunkSearch with direct vector search`
9. `refactor: remove EncoderEmbedding BaseEmbedding wrapper`
10. `refactor: remove TransformComponent from extractors`
11. `chore: remove llama-index-core and neo4jvector dependencies`
12. `docs: update AGENTS.md and tree-sitter-gdl plan`
