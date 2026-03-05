# Document Unification: WikiDocument + Document → Document

Merge `WikiDocument` (40,126 nodes, battle-tested) and `Document` (0 nodes, unused) into a single source-agnostic `Document` node, following the `Image` pattern with a `source_type` discriminator.

## Motivation

- `Document` has zero graph nodes, zero tests, and has never been run in production
- `WikiDocument` has all capabilities Document would need (ContentScoring, DiscoveryProvenance, chunk/image linkage)
- `Image` already proves the source-agnostic pattern with `ImageSourceType`
- Having two node types for the same concept (a non-code document) creates unnecessary schema/code divergence

## Phase 1: Schema Changes

### 1a. Add `DocumentSourceType` enum

```yaml
DocumentSourceType:
  description: >-
    How a document was discovered. Determines fetch strategy
    and parent linkage.
  permissible_values:
    wiki:
      description: Discovered via wiki site scan (MediaWiki, Confluence, etc.)
    filesystem:
      description: Discovered via facility filesystem scan
    repository:
      description: From a code repository (future)
```

### 1b. Rename `WikiDocumentStatus` → `DocumentStatus`

Update in `common.yaml`. Same values: discovered, scored, ingested, deferred, skipped, failed.

### 1c. Merge `DocumentType` into `ArtifactType`

Add missing values from `DocumentType` to `ArtifactType`:
- `markdown` — Markdown documentation
- `text` — Plain text file
- `html` — HTML document
- `config` — Configuration file

Delete `DocumentType` enum from `facility.yaml`.

### 1d. Rename `WikiDocument` → `Document` class

In `facility.yaml`:
- Rename class `WikiDocument` → `Document`
- Add `source_type: DocumentSourceType` attribute
- Add optional `path` attribute (filesystem source only)
- Add optional `in_directory → FacilityPath` relationship (filesystem source only)
- Keep all existing WikiDocument attributes (url becomes optional for filesystem sources)
- Update `class_uri` to `facility:Document`
- Keep mixins: `ContentScoring`, `DiscoveryProvenance`

### 1e. Delete old `Document` class

Remove the current bare `Document` class definition (~80 lines at facility.yaml:2702).

### 1f. Regenerate models

```bash
uv run build-models --force
```

## Phase 2: Code Rename (mechanical)

Global find-and-replace across the codebase:

| Old | New | Files |
|-----|-----|-------|
| `WikiDocument` | `Document` | ~12 Python files |
| `WikiDocumentStatus` | `DocumentStatus` | ~8 Python files |
| `DocumentType` | (remove usage, use `ArtifactType`) | ~4 Python files |

### Key modules:
- `imas_codex/discovery/wiki/graph_ops.py` (35 refs) — Cypher queries, MERGE/MATCH patterns
- `imas_codex/discovery/wiki/pipeline.py` (25 refs) — pipeline orchestration
- `imas_codex/discovery/wiki/workers.py` (14 refs) — worker functions
- `imas_codex/discovery/wiki/parallel.py` (11 refs) — parallel execution
- `imas_codex/agentic/` (3 files) — agent references
- `imas_codex/cli/discover/wiki.py` — CLI command
- `imas_codex/discovery/base/` (2 files) — base infrastructure

### Update Cypher queries:
All `(:WikiDocument {...})` patterns in graph_ops.py → `(:Document {...})`

## Phase 3: Adapt `discovery/documents/` module

Refactor the 1,063-LOC `discovery/documents/` module to create `Document` nodes with `source_type: filesystem`:

1. **scanner.py** — Update MERGE queries: create `Document` nodes instead of old Document, set `source_type='filesystem'`, `status='discovered'`
2. **workers.py** — Update claim/complete queries to use `DocumentStatus` enum
3. **pipeline.py** — Update imports and type references
4. Keep the filesystem-specific logic (SSH enumeration, path-based discovery) intact — it's the value of this module

## Phase 4: Graph Migration

```cypher
-- Relabel WikiDocument → Document (same pattern as WikiArtifact migration)
CALL apoc.periodic.iterate(
  "MATCH (n:WikiDocument) RETURN n",
  "SET n:Document REMOVE n:WikiDocument",
  {batchSize: 5000}
)

-- Verify
MATCH (n:Document) RETURN count(n)  -- expect 40,126
MATCH (n:WikiDocument) RETURN count(n)  -- expect 0
```

No relationship changes needed — `HAS_DOCUMENT`, `HAS_CHUNK`, `HAS_IMAGE`, `FROM_SOURCE`, `AT_FACILITY` all remain valid.

## Phase 5: Update References

- `agents/schema-reference.md` — auto-generated on `uv run build-models`
- Prompts in `imas_codex/agentic/prompts/` — update any WikiDocument references
- `AGENTS.md` — update schema examples if any mention WikiDocument

## Phase 6: Tests

- Run full test suite: `uv run pytest`
- Verify model generation: `uv run build-models --force`
- Verify graph queries work with new label

## Enum Summary After Merge

| Enum | Status |
|------|--------|
| `DocumentSourceType` | **NEW** — wiki, filesystem, repository |
| `DocumentStatus` | **RENAMED** from WikiDocumentStatus |
| `ArtifactType` | **EXTENDED** — add markdown, text, html, config |
| `DocumentType` | **DELETED** — merged into ArtifactType |
| `WikiDocumentStatus` | **DELETED** — renamed to DocumentStatus |

## Risk Mitigation

- Document has 0 nodes and 0 tests — no production risk
- WikiDocument→Document is the same relabel pattern we used for WikiArtifact→WikiDocument
- Schema regeneration is automated and well-tested
- All changes are additive to the production WikiDocument data
