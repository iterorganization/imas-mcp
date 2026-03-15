# Error & Metadata Field Filtering in Search

**Status**: Plan  
**Created**: 2025-07-24  
**Scope**: `imas_codex/tools/graph_search.py`, `imas_codex/ids/tools.py`, `imas_codex/core/exclusions.py`, `imas_codex/llm/server.py`

## Problem Statement

Error fields (`_error_upper`, `_error_lower`, `_error_index`) and metadata
subtrees (`ids_properties/*`, `code/*`) dominate all search results, crowding
out the data fields that users and the mapping pipeline actually need.

### Key Metrics (from graph audit)

| Category | Nodes | % of Total | Embedded | % of Embedded |
|----------|-------|-----------|----------|---------------|
| Error    | 31,281 | 51.0%    | 30,082   | 50.2%         |
| Metadata | 7,745  | 12.6%   | 7,456    | 12.5%         |
| Data     | 22,340 | 36.4%   | 22,331   | 37.3%         |

**Half of all indexed embeddings are error fields.** This means:

- **Vector search pollution**: For "electron temperature profile", 14/20 top kNN
  results are error fields (70%). The actual target
  `core_profiles/profiles_1d/electrons/temperature` doesn't appear in the top-100
  despite having an embedding (cosine sim = 0.73 vs 0.92 for error_index fields).
- **Enrichment bias**: Error field embedding texts are *more descriptive* than
  their parent data fields because the enrichment pipeline generates explanatory
  text like "Integer index into the error description vector for temperature.
  Links this measurement to a specific error model..." while the data field
  gets a shorter functional description.
- **Consistent across queries**: Tested 5 representative physics queries:
  - "electron temperature profile": 70% error in top-20
  - "magnetic field components": 60% error in top-20
  - "safety factor q profile": 50% error in top-20
  - "plasma current": 45% error in top-20
  - "wall surface temperature": 35% error in top-20
- **Metadata noise**: `_is_generic_metadata_path()` only filters 5 tail
  segments (`description`, `name`, `comment`, `source`, `provider`), completely
  missing `ids_properties/*` (78 nodes/IDS) and `code/*` (13 nodes/IDS).

### Existing Infrastructure (unused at query time)

- `ExclusionChecker` in `core/exclusions.py`: Has `_is_error_field()` and
  `_is_ggd_path()` — but only wired into build-time parsers, never into search.
- `settings.py`: `get_include_error_fields()` and `get_include_ggd()` exist
  but are only consumed by ExclusionChecker at build time.
- `HAS_ERROR` relationships: 31,281 in graph, linking data→error with
  `error_type` property. These can be traversed programmatically without
  search.

## Design Constraints

1. **Error fields must remain searchable** — 311 facility signals contain
   "error" or "err" in their names. Some are legitimate measurement errors
   (e.g., "HRTS Electron Density Error", "Ion Temperature Error") that should
   map to `_error_upper`/`_error_lower` fields. Others are "error field"
   physics concepts (magnetic error fields, feedback error signals).
2. **ids_properties and code are per-IDS, not per-node** — These fields
   describe the IDS *occurrence* (who wrote it, what version, what software),
   not individual signal paths. They should never be search targets for
   signal mapping but may be useful for metadata queries.
3. **Backward compatibility** — The MCP tools (`search_imas`, `list_imas`,
   etc.) must not break existing agent workflows.

## Proposed Solution: Exclude-by-Default with Opt-In

### Phase 1: Cypher-Level Filtering (Core Fix)

Add WHERE clauses to all vector and text search queries that exclude error
and metadata nodes by default. This is the cheapest, most effective fix.

**Affected queries** (all in `graph_search.py`):

1. **`GraphSearchTool.search_imas_paths()`** (line ~143): Vector search
   `db.index.vector.queryNodes('imas_node_embedding', ...)` — add
   `AND NOT n.id CONTAINS '_error_'` and metadata path exclusion.

2. **`_text_search_imas_paths()`** (line ~1694): BM25 fulltext search —
   add same exclusion to both fulltext and CONTAINS fallback queries.

3. **`compute_semantic_matches()`** in `ids/tools.py` (line ~599): Direct
   vector query for mapping pipeline — add exclusion to the IMAS vector
   search block.

**Implementation**:

```python
# New helper in graph_search.py
def _exclusion_clause(alias: str = "n", *, include_errors: bool = False,
                      include_metadata: bool = False) -> str:
    """Return Cypher WHERE fragments to exclude error/metadata fields."""
    clauses = []
    if not include_errors:
        clauses.append(f"AND NOT {alias}.id CONTAINS '_error_'")
    if not include_metadata:
        clauses.append(
            f"AND NOT any(seg IN ['ids_properties', 'code'] "
            f"WHERE {alias}.id CONTAINS '/' + seg + '/')"
        )
    return " ".join(clauses)
```

**Search tool parameter additions**:

```python
async def search_imas_paths(
    self,
    query: str,
    ...,
    include_error_fields: bool = False,   # NEW
    include_metadata: bool = False,        # NEW
) -> SearchPathsResult:
```

**Cost**: Low — string concatenation in existing queries. No schema changes,
no re-embedding, no index rebuilds.

**Impact**: Immediately removes ~60% of noise from all vector and text
searches. The mapping pipeline gets 2-3× more relevant candidates per kNN
call without increasing k.

### Phase 2: Upgrade `_is_generic_metadata_path()`

Replace the narrow tail-segment check with a proper subtree check:

```python
def _is_generic_metadata_path(path_id: str) -> bool:
    """Check if an IMAS path is a generic metadata/descriptor field."""
    parts = path_id.split("/")
    if len(parts) < 3:
        return False
    # Full subtree exclusion
    if any(seg in ("ids_properties", "code") for seg in parts[1:]):
        return True
    # Existing tail checks
    tail = parts[-1]
    if tail in ("description", "name", "comment", "source", "provider"):
        return True
    if len(parts) >= 2 and parts[-2] == "identifier" and tail in ("description", "name"):
        return True
    return False
```

This catches the 7,745 metadata nodes that currently pass through post-filter.

### Phase 3: Wire ExclusionChecker into Search Path

Unify build-time and query-time exclusion logic:

1. Add `_is_metadata_path()` to `ExclusionChecker` for consistency.
2. Have `_is_generic_metadata_path()` delegate to the singleton checker.
3. Add "metadata" exclusion reason to `EXCLUSION_REASONS`.

This ensures the same logic governs what gets built and what gets searched.

### Phase 4: MCP Tool Surface

Add `include_error_fields` and `include_metadata` optional boolean parameters
to the MCP tool registrations in `llm/server.py`:

- `search_imas`: Defaults `include_error_fields=False`, `include_metadata=False`
- `list_imas_paths`: Defaults to including all (listing is explicit, not search)
- `check_imas_paths` / `fetch_imas_paths`: No filtering (exact lookup)

For list tools, add an optional `exclude_errors` parameter so agents can get
clean subtree listings when needed by the mapping pipeline.

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Legitimate error-field queries miss results | `include_error_fields=True` opt-in; error mapping stage uses this explicitly |
| Performance impact of WHERE clauses on vector queries | Neo4j post-filters vector results; CONTAINS on string is O(n) per result but n is bounded by k parameter |
| Breaking existing agent workflows | Default behavior improves (less noise); agents that need error fields can opt in |
| `ids_properties` queries for provenance | `include_metadata=True` opt-in; separate metadata population pipeline handles these |

## Testing Strategy

1. **Unit tests**: Verify `_exclusion_clause()` generates correct Cypher fragments
2. **Integration tests**: Confirm vector search for "electron temperature" returns
   `core_profiles/profiles_1d/electrons/temperature` in top-5 after filtering
3. **Regression tests**: Confirm `include_error_fields=True` still returns error
   fields when explicitly requested
4. **Pipeline smoke test**: Run `imas map run` for a known IDS and verify mapping
   quality improves (fewer forced low-confidence bindings)

## Dependencies

- None for Phase 1-2 (self-contained in existing query infrastructure)
- Phase 3 depends on verifying `ExclusionChecker` is importable from tools
- Phase 4 depends on MCP schema update (non-breaking addition)
