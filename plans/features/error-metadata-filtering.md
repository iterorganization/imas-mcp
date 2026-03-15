# Error & Metadata Field Filtering

**Status**: Plan
**Created**: 2025-07-24
**Updated**: 2026-03-15
**Scope**: Schema, graph build, enrichment, embedding, search, mapping, MCP tools, clustering, CLI

## Problem Statement

Error fields (`_error_upper`, `_error_lower`, `_error_index`) and metadata
subtrees (`ids_properties/*`, `code/*`) dominate the graph and pollute every
downstream process — search, mapping, clustering, and enrichment.

### Key Metrics (from graph audit)

| Category | Nodes | % of Total | Embedded | % of Embedded |
|----------|-------|-----------|----------|---------------|
| Error    | 31,281 | 51.0%    | 30,082   | 50.2%         |
| Metadata | 7,745  | 12.6%   | 7,456    | 12.5%         |
| Data     | 22,340 | 36.4%   | 22,331   | 37.3%         |

**Half of all indexed embeddings are error fields.** Consequences:

- **Vector search pollution**: "electron temperature profile" returns 14/20
  error fields in top kNN results. The actual target
  `core_profiles/profiles_1d/electrons/temperature` doesn't appear in the
  top-100 despite having an embedding (cosine sim 0.73 vs 0.92 for
  error_index fields).
- **Enrichment waste**: 31k error fields each consume an LLM call during
  enrichment. Their generated descriptions are *more descriptive* than
  parent data fields, amplifying search pollution.
- **Embedding waste**: 30k error field embeddings consume vector index
  memory and degrade kNN recall.
- **Cluster pollution**: Error fields get clustered alongside data fields,
  distorting semantic cluster quality.
- **Mapping pipeline noise**: `compute_semantic_matches()` returns error
  fields as mapping candidates, forcing low-confidence bindings.
- **Consistent across queries**: Tested 5 representative physics queries —
  35–70% of top-20 results are error fields.
- **Metadata noise**: `_is_generic_metadata_path()` only filters 5 tail
  segments, completely missing `ids_properties/*` (78 nodes/IDS) and
  `code/*` (13 nodes/IDS).

### Existing Infrastructure

- `ExclusionChecker` in `core/exclusions.py`: Has `_is_error_field()` and
  `_is_ggd_path()` — wired into build-time parsers only, never into search
  or downstream pipelines.
- `settings.py`: `get_include_error_fields()` defaults to `True`, so error
  fields are built, enriched, and embedded.
- `HAS_ERROR` relationships: 31,281 in graph, linking data→error with
  `error_type` property. These support programmatic traversal without search.
- `_is_generic_metadata_path()`: Python post-filter in `graph_search.py`,
  catches only 5 tail segments.

## Design Approach: Schema-First Indexed Filtering

The core fix is adding an **indexed `node_category` property** to `IMASNode`
that classifies every node at build time. All downstream filtering operates
on this indexed property — no string `CONTAINS` scans, no Python post-filters.

### Node Categories

| Category | Examples | Search Default | Enrichment | Embedding |
|----------|----------|---------------|------------|-----------|
| `data` | `equilibrium/time_slice/boundary/psi` | Included | LLM | Yes |
| `error` | `*/psi_error_upper`, `*/psi_error_lower` | **Excluded** | Template only | **No** |
| `metadata` | `*/ids_properties/*`, `*/code/*` | **Excluded** | Template only | **No** |

Error fields are **not searched, not enriched by LLM, and not embedded**.
They remain as graph nodes reachable via `HAS_ERROR` relationship traversal.
When a mapping pipeline binds a signal to a data field, the error fields
are discoverable via `(data)-[:HAS_ERROR]->(error)`.

## Implementation Plan

### Phase 1: Schema & Index

**Files**: `imas_codex/schemas/imas_dd.yaml`, `imas_codex/graph/build_dd.py`

1. Add `node_category` slot to `IMASNode` in `imas_dd.yaml`:

```yaml
node_category:
  description: >-
    Classification of this node for filtering: data (physics quantities),
    error (uncertainty bounds), or metadata (ids_properties, code subtrees).
  range: NodeCategory
  required: true
```

2. Add `NodeCategory` enum to `imas_dd.yaml`:

```yaml
NodeCategory:
  description: Classification of IMASNode fields.
  permissible_values:
    data:
      description: Physics data fields
    error:
      description: Uncertainty bound fields (_error_upper, _error_lower, _error_index)
    metadata:
      description: Bookkeeping subtrees (ids_properties, code, identifiers)
```

3. Add range index in `_ensure_indexes()` in `build_dd.py`:

```python
client.query("CREATE INDEX imas_node_category IF NOT EXISTS FOR (p:IMASNode) ON (p.node_category)")
```

4. Set `node_category` during `_batch_create_path_nodes()` in `build_dd.py`:

```python
def _classify_node(path_id: str, name: str) -> str:
    """Classify an IMASNode path into a category."""
    parts = path_id.split("/")
    # Error fields
    if (
        name.endswith("_error_upper")
        or name.endswith("_error_lower")
        or name.endswith("_error_index")
    ):
        return "error"
    # Metadata subtrees
    if any(seg in ("ids_properties", "code") for seg in parts[1:]):
        return "metadata"
    # Generic metadata leaf fields
    if len(parts) >= 3:
        tail = parts[-1]
        if tail in ("description", "name", "comment", "source", "provider"):
            return "metadata"
        if len(parts) >= 2 and parts[-2] == "identifier" and tail in ("description", "name"):
            return "metadata"
    return "data"
```

This replaces the fragile `_detect_error_relationships()` pattern matching
with a single classification function used at node creation time.

5. Rebuild models: `uv run build-models --force`

**Performance**: `WHERE n.node_category = 'data'` uses the range index —
O(log n) lookup, not O(n) string scan. Neo4j planner resolves this to an
index seek.

### Phase 2: Build Pipeline — Skip Enrichment & Embedding for Non-Data

**Files**: `imas_codex/graph/dd_graph_ops.py`, `imas_codex/graph/dd_workers.py`, `imas_codex/graph/build_dd.py`

1. **Enrichment claim query** — Add category filter to
   `claim_paths_for_enrichment()` in `dd_graph_ops.py`:

```python
MATCH (p:IMASNode)
WHERE p.status = $status
  AND p.node_category = 'data'
  AND (p.claimed_at IS NULL
       OR p.claimed_at < datetime() - duration($cutoff))
```

Error and metadata nodes skip LLM enrichment entirely. They get a
minimal template description at build time (already handled by
`is_boilerplate_path()` in `dd_workers.py` — extend this to cover all
non-data nodes).

2. **Embedding phase** — Add category filter to `phase_embed()` in
   `build_dd.py`. Only embed `node_category = 'data'` nodes:

```python
# In the query that fetches paths needing embeddings:
MATCH (p:IMASNode)
WHERE p.node_category = 'data'
  AND p.embedding IS NULL
```

This removes ~38k nodes from the embedding pipeline (31k error + 7.7k
metadata), cutting embedding compute by ~63% on rebuild.

3. **Strip existing embeddings** — One-time migration query:

```cypher
MATCH (p:IMASNode)
WHERE p.node_category IN ['error', 'metadata']
SET p.embedding = null, p.embedding_text = null, p.embedding_hash = null
```

This removes ~37.5k embeddings from the vector index, improving kNN
recall for all data field queries.

### Phase 3: Search — Replace All Post-Filters with Indexed WHERE

**Files**: `imas_codex/tools/graph_search.py`, `imas_codex/ids/tools.py`,
`imas_codex/graph/domain_queries.py`, `imas_codex/cli/imas_dd.py`

All vector and text search queries get a `WHERE node.node_category = 'data'`
clause. Since non-data nodes will have no embeddings (Phase 2), this is
belt-and-suspenders — the vector index won't return them anyway, but the
WHERE clause handles edge cases during migration.

1. **`search_imas_paths()`** in `graph_search.py` — Remove
   `_is_generic_metadata_path()` post-filter, add Cypher filter:

```python
CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
YIELD node AS path, score
WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
  AND path.node_category = 'data'
  {filter_clause}
  {dd_clause}
```

Remove the `include_error_fields` and `include_metadata` opt-in parameters
from the API — error fields are not in the embedding index, so they cannot
be searched. If a caller needs error fields for a data path, they traverse
`HAS_ERROR` relationships.

2. **`_text_search_imas_paths()`** in `graph_search.py` — Add filter to
   both BM25 fulltext and CONTAINS fallback queries:

```python
# BM25
CALL db.index.fulltext.queryNodes('imas_node_text', $query)
YIELD node AS p, score
WHERE p.node_category = 'data'
  ...

# CONTAINS fallback
MATCH (p:IMASNode)
WHERE p.node_category = 'data'
  AND ...
```

3. **Delete `_is_generic_metadata_path()`** — replaced entirely by
   `node_category` index filtering. Remove the function and all three
   call sites in `graph_search.py`.

4. **`compute_semantic_matches()`** in `ids/tools.py` — Add filter:

```python
CALL db.index.vector.queryNodes("imas_node_embedding", $k, $embedding)
YIELD node AS n, score
WHERE n.node_category = 'data'
  {imas_where}
```

5. **`find_imas()`** in `domain_queries.py` — Add filter:

```python
CALL db.index.vector.queryNodes("imas_node_embedding", $k, $embedding)
YIELD node AS p, score
WHERE p.node_category = 'data'
  {where_clause}
```

6. **CLI search** in `cli/imas_dd.py` — Add filter:

```python
CALL db.index.vector.queryNodes("imas_node_embedding", $limit * 2, $embedding)
YIELD node, score
WHERE node.node_category = 'data'
  {where_clause}
```

### Phase 4: ExclusionChecker Consolidation

**Files**: `imas_codex/core/exclusions.py`, `imas_codex/settings.py`

1. Add `_is_metadata_path()` to `ExclusionChecker`:

```python
def _is_metadata_path(self, path: str) -> bool:
    parts = path.split("/")
    if any(seg in ("ids_properties", "code") for seg in parts[1:]):
        return True
    if len(parts) >= 3:
        tail = parts[-1]
        if tail in ("description", "name", "comment", "source", "provider"):
            return True
    return False
```

2. Wire `_is_metadata_path()` into `get_exclusion_reason()` unconditionally
   (metadata is always excluded — no opt-in flag).

3. Change `get_include_error_fields()` default from `True` to `False` in
   `settings.py` and `pyproject.toml`:

```toml
[tool.imas-codex.data-dictionary]
include-error-fields = false
```

This means the XML parser / DocumentStore also excludes error fields,
keeping the build-time and graph-time behavior consistent.

4. `_classify_node()` from Phase 1 delegates to `ExclusionChecker` methods
   for consistency — single source of truth for what constitutes an error
   or metadata field.

### Phase 5: List & Cluster Cleanup

**Files**: `imas_codex/tools/list_tool.py`, `imas_codex/tools/graph_search.py`,
`imas_codex/clusters/`, `imas_codex/core/clusters.py`

1. **`list_imas_paths`** — Add `node_category` filter to graph queries.
   Default to `node_category = 'data'`. No opt-in parameter — error fields
   are discoverable via `HAS_ERROR` traversal, not listing.

2. **`GraphListTool.list_imas_paths()`** in `graph_search.py` — Add WHERE
   clause:

```python
MATCH (p:IMASNode)
WHERE p.id STARTS WITH $prefix
  AND p.node_category = 'data'
```

3. **Cluster coverage** in `core/clusters.py` — Replace phantom
   `p.is_error_field = true` query with `p.node_category = 'error'`:

```python
MATCH (p:IMASNode)
WITH count(p) AS total,
     count(CASE WHEN p.embedding IS NOT NULL THEN 1 END) AS with_emb,
     count(CASE WHEN p.node_category != 'data' THEN 1 END) AS non_data
WITH total, with_emb, total - non_data AS embeddable
RETURN total, with_emb, embeddable,
       CASE WHEN embeddable > 0
            THEN toFloat(with_emb) / embeddable
            ELSE 0.0 END AS coverage
```

4. **Cluster preprocessing** in `clusters/preprocessing.py` — Filter to
   `node_category = 'data'` when loading embeddings for clustering. Error
   nodes won't have embeddings after Phase 2, but add the filter for
   correctness.

### Phase 6: MCP Tool Surface

**Files**: `imas_codex/llm/server.py`

Update MCP tool signatures — no opt-in parameters needed:

- `search_imas`: Always excludes error/metadata (they have no embeddings).
  Remove any `include_error_fields` / `include_metadata` parameters.
- `list_imas_paths`: Always excludes error/metadata. Add tool description
  noting that error fields are accessible via `HAS_ERROR` relationships
  from their parent data paths.
- `check_imas_paths` / `fetch_imas_paths`: No change — exact lookups work
  on any node regardless of category.

Add a new MCP tool or extend `fetch_imas_paths` to support error field
retrieval by relationship traversal:

```python
def fetch_error_fields(path: str, dd_version: int | None = None) -> str:
    """Fetch error fields for a data path via HAS_ERROR relationships."""
    # MATCH (d:IMASNode {id: $path})-[:HAS_ERROR]->(e:IMASNode)
    # RETURN e.id, e.name, ...
```

### Phase 7: Graph Rebuild

Run a full DD rebuild to apply all changes:

1. `uv run build-dd --force` — Rebuilds all IMASNode with `node_category`
2. One-time migration strips embeddings from error/metadata nodes
3. Verify with audit query:

```cypher
MATCH (p:IMASNode)
RETURN p.node_category, count(*) AS cnt,
       count(p.embedding) AS with_emb
ORDER BY cnt DESC
```

Expected result:
| node_category | cnt | with_emb |
|---------------|-----|----------|
| error | ~31,281 | 0 |
| data | ~22,340 | ~22,340 |
| metadata | ~7,745 | 0 |

## Impact Summary

| Component | Change | Effect |
|-----------|--------|--------|
| `imas_dd.yaml` | Add `NodeCategory` enum, `node_category` slot | Schema source of truth |
| `build_dd.py` | Classify nodes, add index, skip error/metadata embedding | 63% fewer embeddings |
| `dd_graph_ops.py` | Filter enrichment claims to `data` only | 63% fewer LLM calls |
| `dd_workers.py` | Extend `is_boilerplate_path()` for non-data | Template-only enrichment |
| `graph_search.py` | Replace `_is_generic_metadata_path()` with indexed WHERE | Faster, complete filtering |
| `ids/tools.py` | Add `node_category = 'data'` to vector search | Clean mapping candidates |
| `domain_queries.py` | Add `node_category = 'data'` to vector search | Clean domain search |
| `cli/imas_dd.py` | Add `node_category = 'data'` to CLI search | Clean CLI results |
| `list_tool.py` | Add `node_category = 'data'` to list queries | Clean path listings |
| `core/clusters.py` | Replace phantom `is_error_field` with `node_category` | Correct coverage |
| `clusters/` | Filter to `data` nodes for clustering input | Clean clusters |
| `core/exclusions.py` | Add `_is_metadata_path()`, unify logic | Single source of truth |
| `settings.py` | Default `include-error-fields = false` | Consistent build-time |
| `pyproject.toml` | `include-error-fields = false` | Config change |
| `llm/server.py` | Remove opt-in params, add `fetch_error_fields` | Clean MCP API |

## Performance Analysis

| Operation | Before | After | Why |
|-----------|--------|-------|-----|
| Vector kNN post-filter | `CONTAINS '_error_'` — O(n) string scan per result | `node_category = 'data'` — index seek | Range index on `node_category` |
| Text search post-filter | Python `_is_generic_metadata_path()` — O(n) | Cypher `node_category = 'data'` — index seek | Filtering moves to DB engine |
| Vector index size | ~60k embeddings | ~22k embeddings | 63% smaller index, better recall |
| Enrichment pipeline | ~61k LLM calls | ~22k LLM calls | Skip non-data nodes |
| kNN recall@20 for data queries | ~30% (polluted by error fields) | ~95%+ (data only) | Cleaner index |

## Testing Strategy

1. **Schema compliance**: Verify `node_category` exists on all `IMASNode` nodes,
   values are in `{data, error, metadata}`
2. **Index verification**: `SHOW INDEXES` confirms `imas_node_category` exists
   and is populated
3. **Embedding absence**: Confirm zero embeddings on `error`/`metadata` nodes
4. **Search quality**: "electron temperature profile" returns
   `core_profiles/profiles_1d/electrons/temperature` in top-5
5. **Error traversal**: `MATCH (d)-[:HAS_ERROR]->(e)` still works for all
   data→error pairs
6. **Mapping quality**: `imas map run` for a known IDS produces higher-confidence
   bindings with fewer error-field candidates
7. **List filtering**: `list_imas_paths` for any IDS returns zero error/metadata
   paths
8. **Cluster coverage**: Coverage query returns correct count with `node_category`
   instead of phantom `is_error_field`

## Dependencies

- Phase 1 (Schema & Index) blocks all subsequent phases
- Phase 2 (Build Pipeline) blocks Phase 3 (embedding removal makes search
  filter mostly redundant, but belt-and-suspenders)
- Phase 4 (ExclusionChecker) is independent, can run in parallel with 2-3
- Phase 5-6 (List/Cluster/MCP) depend on Phase 1 only
- Phase 7 (Rebuild) runs after all code changes are merged
