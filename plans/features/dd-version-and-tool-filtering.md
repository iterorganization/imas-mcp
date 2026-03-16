# DD Version Validity & Tool Filtering Fixes

**Status:** Plan
**Created:** 2026-03-16
**Scope:** `imas_codex/tools/graph_search.py`, `imas_codex/tools/overview_tool.py`,
tests, CLI, legacy code paths

## Problem Statement

Live testing of Codex MCP IMAS tools against the graph revealed 5 defects
in query construction and filtering, ranging from critical (91% of valid
paths silently excluded) to low (minor query inefficiency). These were
discovered via paired comparison with the standalone IMAS MCP server and
validated with direct Cypher queries against the production graph.

### Validated Findings (Cypher-confirmed)

| # | Defect | Severity | Impact |
|---|--------|----------|--------|
| 1 | `_dd_version_clause` uses wrong semantics | **Critical** | `dd_version=4` returns 1,561 paths instead of 16,928 (91% excluded) |
| 2 | `_search_by_path` scope filter syntax error | **High** | `scope` parameter triggers Neo4j syntax error |
| 3 | `get_imas_overview` counts all node categories | **Medium** | Path counts 2-3× inflated (equilibrium: 2,019 vs 641 data) |
| 4 | `export_imas_ids` / `export_imas_domain` unfiltered | **Medium** | Exports include error (51%) and metadata (16%) nodes |
| 5 | `list_imas_paths` runs redundant overwriting query | **Low** | IDS-only queries run two queries, second overwrites first |

### Graph Data Model (validated)

```
DDVersion nodes:       35 (3.22.0 → 4.1.1, current = 4.1.0)
IMASNode total:        61,366
  data:                20,037
  error:               31,281
  metadata:            10,048

INTRODUCED_IN edges:   61,366 (100% coverage, always exactly 1 per node)
DEPRECATED_IN edges:   ~17,000 (2 paths have multiple, 0 paths have re-introduction)
```

Version model is strictly monotonic: every path has exactly one `INTRODUCED_IN`
edge to its first appearance. `DEPRECATED_IN` records the version where the
path was removed. There are no re-introductions (deprecated-then-reintroduced = 0).

---

## Bug 1: `_dd_version_clause` — Wrong Version Validity Semantics

**File:** `imas_codex/tools/graph_search.py` line 39

### Current Behavior

```python
def _dd_version_clause(alias, dd_version, params):
    """Return Cypher WHERE fragment for DD major version filtering."""
    prefix = f"{dd_version}."
    return (
        f"AND EXISTS {{ MATCH ({alias})-[:INTRODUCED_IN]->(iv:DDVersion) "
        f"  WHERE iv.id STARTS WITH $dd_version_prefix }} "
        f"AND NOT EXISTS {{ MATCH ({alias})-[:DEPRECATED_IN]->(dv:DDVersion) "
        f"  WHERE dv.id STARTS WITH $dd_version_prefix }}"
    )
```

This means "introduced in major N AND not deprecated in major N" — which
excludes paths introduced in any earlier major version. For `dd_version=4`,
only paths introduced in DD 4.x are returned (1,561), while 15,367 paths
introduced in DD 3.x and still valid are silently excluded.

### Correct Semantics

A path is **valid in DD major N** if:
1. Introduced in **any** version with major ≤ N
2. NOT deprecated in **any** version with major ≤ N

### Validated Correct Cypher

```cypher
-- Valid in DD4: 16,928 paths (confirmed)
MATCH (p:IMASNode)-[:INTRODUCED_IN]->(iv:DDVersion)
WHERE p.node_category = 'data'
  AND toInteger(split(iv.id, '.')[0]) <= 4
  AND NOT EXISTS {
    MATCH (p)-[:DEPRECATED_IN]->(dv:DDVersion)
    WHERE toInteger(split(dv.id, '.')[0]) <= 4
  }
RETURN count(p)

-- Valid in DD3: 17,391 paths (confirmed)
-- (same pattern with <= 3)
```

**Performance:** `toInteger(split())` approach benchmarked at 0.177s avg vs
0.166s avg for no-filter — negligible overhead.

### Fix

Replace `_dd_version_clause`:

```python
def _dd_version_clause(
    alias: str = "p",
    dd_version: int | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Return a Cypher WHERE fragment for DD major version validity filtering.

    A path is valid in DD major N if:
    - introduced in any version with major <= N
    - NOT deprecated in any version with major <= N
    """
    if dd_version is None:
        return ""
    if params is not None:
        params["dd_version"] = dd_version
    return (
        f"AND EXISTS {{ "
        f"  MATCH ({alias})-[:INTRODUCED_IN]->(iv:DDVersion) "
        f"  WHERE toInteger(split(iv.id, '.')[0]) <= $dd_version "
        f"}} "
        f"AND NOT EXISTS {{ "
        f"  MATCH ({alias})-[:DEPRECATED_IN]->(dv:DDVersion) "
        f"  WHERE toInteger(split(dv.id, '.')[0]) <= $dd_version "
        f"}}"
    )
```

### Test Cases (all validated against live graph)

| Path | Introduced | Deprecated | DD3 Valid | DD4 Valid |
|------|-----------|------------|----------|----------|
| `equilibrium/time_slice/profiles_1d/psi` | 3.22.0 | — | ✓ | ✓ |
| `magnetics/bpol_probe/position/r` | 3.22.0 | 4.0.0 | ✓ | ✗ |
| `gyrokinetics/species/velocity_tor_gradient_norm` | 3.22.0 | 3.40.0 | ✗ | ✗ |
| `plasma_profiles/profiles_1d/neutral/label` | 3.22.0 | 4.1.0 | ✓ | ✗ |

### Call Sites (all uses of `_dd_version_clause` in graph_search.py)

1. `GraphSearchTool.search_imas_paths` — vector + text search (line ~146)
2. `GraphPathTool.check_imas_paths` — path validation (line ~380)
3. `GraphPathTool.fetch_imas_paths` — path documentation (line ~470)
4. `GraphPathTool.fetch_error_fields` — error field lookup (line ~610)
5. `GraphListTool.list_imas_paths` — path listing (line ~730, 750)
6. `GraphOverviewTool.get_imas_overview` — IDS overview (line ~862)
7. `GraphClustersTool._list_by_ids` — cluster IDS listing (line ~1040)
8. `GraphClustersTool._search_by_path` — cluster path lookup (line ~1080)
9. `GraphClustersTool._search_by_text` — cluster semantic search (line ~1130, implicit in ft search)
10. `GraphPathContextTool.get_imas_path_context` — semantic/cluster/coordinate/unit sections (line ~1470+)
11. `GraphStructureTool.export_imas_ids` — IDS export (line ~1720)
12. `GraphStructureTool.export_imas_domain` — domain export (line ~1790)
13. `_text_search_imas_paths` — fulltext helper (line ~1870, 1900)

All 13 call sites use the same function signature. The fix is a single
function replacement — no call site changes needed. The parameter name
changes from `dd_version_prefix` (string) to `dd_version` (int), which
is already the parameter name used at all call sites.

### Legacy Code (separate fix)

Two older code paths use a property-based `dd_version` filter instead of
relationship-based version checking:

- `imas_codex/ids/tools.py` line 765: `AND n.dd_version = $dd_version`
- `imas_codex/cli/imas_dd.py` line 533: `node.dd_version = '{version_filter}'`

These should be migrated to use the same relationship-based helper or
removed if the code paths are unused. These are lower priority since the
MCP tools don't use them.

---

## Bug 2: Cluster `_search_by_path` Scope Filter Syntax Error

**File:** `imas_codex/tools/graph_search.py` line 1064

### Current Behavior

```python
def _search_by_path(self, path, scope, *, dd_version=None):
    scope_filter = "AND c.scope = $scope" if scope else ""
    # ...
    results = self._gc.query(f"""
        MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        {scope_filter}
        OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
        WHERE true {dd_clause}
        ...
    """)
```

`scope_filter` is `"AND c.scope = $scope"` placed directly after `MATCH`
without a `WHERE` keyword. Neo4j rejects this with:

```
Invalid input 'AND': expected 'WHERE', 'WITH', ...
```

**Confirmed by live test:** Querying with `scope='global'` produces the
exact error above.

### Fix

Add `WHERE true` before the scope filter:

```python
results = self._gc.query(f"""
    MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
    WHERE true {scope_filter}
    OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
    WHERE true {dd_clause}
    ...
""")
```

**Note:** `_list_by_ids` and `_search_by_text` are NOT affected — they
already have `WHERE` clauses before their respective scope filters.

---

## Bug 3: Overview Counts Include Error + Metadata Nodes

**File:** `imas_codex/tools/graph_search.py` line 828

### Current Behavior

```cypher
MATCH (i:IDS)
OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASNode)
WHERE true {dd_clause}
WITH i, count(p) AS path_count
```

No `node_category` filter. For `equilibrium`, this reports 2,019 paths
instead of 641 data paths (1,224 error + 154 metadata inflating the count).

### Fix

Add `p.node_category = 'data'` to the WHERE clause:

```cypher
MATCH (i:IDS)
OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASNode)
WHERE p.node_category = 'data' {dd_clause}
WITH i, count(p) AS path_count
```

**Validation:** With this filter, equilibrium reports 641 paths without
dd_version filter, and 455 paths with `dd_version=4` (correct — excludes
paths deprecated in 4.0.0 and 4.1.0).

---

## Bug 4: Export Tools Include All Node Categories

**File:** `imas_codex/tools/graph_search.py` lines 1697, 1745

### Current Behavior

`export_imas_ids` and `export_imas_domain` query:
```cypher
MATCH (p:IMASNode)
WHERE p.ids = $ids_name ...
```

No `node_category` filter. For `equilibrium`:
- 641 data paths
- 1,224 error paths
- 154 metadata paths

All 2,019 are returned. Error fields are 60% of the export.

### Fix

Default to `node_category = 'data'` with optional expansion:

```python
async def export_imas_ids(
    self,
    ids_name: str,
    leaf_only: bool = False,
    dd_version: int | None = None,
    include_errors: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
```

Cypher:
```cypher
MATCH (p:IMASNode)
WHERE p.ids = $ids_name
  AND p.node_category IN $categories
  {leaf_filter} {dd_clause}
```

Where `categories` defaults to `['data']` and expands to
`['data', 'error']` when `include_errors=True`.

Same pattern for `export_imas_domain`.

---

## Bug 5: List Tool Redundant Query Overwrite

**File:** `imas_codex/tools/graph_search.py` line 733

### Current Behavior

For IDS-only queries (e.g., `paths="equilibrium"`):

```python
# First query runs:
path_results = gc.query("MATCH ... WHERE p.id STARTS WITH $prefix ...")

# Then immediately overwrites:
if "/" not in prefix:
    path_results = gc.query("MATCH ... WHERE p.ids = $ids_name ...")
```

The first query's results are discarded. Both queries return the same data
since all paths with `ids='equilibrium'` also start with `'equilibrium/'`.

### Fix

Consolidate into a single query per intent:

```python
if "/" in query:
    # Subtree query
    path_results = gc.query("... WHERE p.id STARTS WITH $prefix ...")
else:
    # IDS query (skip the prefix query entirely)
    path_results = gc.query("... WHERE p.ids = $ids_name ...")
```

---

## Implementation Plan

### Phase 1: Critical Fixes (Bugs 1-2)

**Priority:** Immediate — these cause visible user-facing failures.
**Risk:** Low — localized changes to query construction.

1. **Replace `_dd_version_clause`** with corrected validity semantics.
   Single function change, all 13 call sites automatically fixed.
   - Change parameter from `dd_version_prefix` (str) to `dd_version` (int)
   - No call site changes needed (already pass `dd_version` as int)

2. **Fix `_search_by_path` scope syntax** by adding `WHERE true` before
   scope filter clause.

3. **Add regression tests:**
   - Test `_dd_version_clause` with DD3 path checked against DD4 (must be found)
   - Test `_dd_version_clause` with DD4-deprecated path against DD3 (must be found)
     and DD4 (must not be found)
   - Test `_dd_version_clause` with DD3-deprecated path against DD3 and DD4
     (must not be found in either)
   - Test `search_imas_clusters` with `scope='global'` (must not error)
   - Test `search_imas_clusters` with `scope='ids'` + path query (must not error)

### Phase 2: Consistency Fixes (Bugs 3-4)

**Priority:** Next — inflated counts and noisy exports degrade LLM tool use.
**Risk:** Low — additive WHERE clauses.

1. **Add `node_category='data'` to overview** path count query.

2. **Add default `node_category` filtering to exports** with optional
   `include_errors` parameter.

3. **Add tests:**
   - Overview path counts match data-only counts
   - Export default excludes error/metadata nodes
   - Export with `include_errors=True` includes error nodes

### Phase 3: Cleanup (Bug 5 + Legacy)

**Priority:** Low — no user-facing impact.
**Risk:** Very low — code simplification.

1. **Consolidate list tool query logic** — remove redundant first query
   for IDS-only mode.

2. **Audit legacy `dd_version` property filters** in `ids/tools.py` and
   `cli/imas_dd.py`. Either migrate to the shared helper or document as
   intentional legacy behavior.

### Phase 4: Graph-Native Quality Improvements

**Priority:** Enhancement — exploits existing graph structure for better results.
**Risk:** Medium — behavioral changes in search ranking.

These are not bug fixes but improvements that leverage the graph model.

#### 4.1 Attach Error Fields as Context

When a search returns a data path that has `HAS_ERROR` relationships,
include the error field paths as structured metadata in the result rather
than ranking them as independent search hits. The vector index already
excludes error nodes (only data nodes are embedded), so this is about
enriching results, not filtering.

```cypher
-- After fetching search hits:
OPTIONAL MATCH (path)-[:HAS_ERROR]->(err:IMASNode)
RETURN collect({path: err.id, error_type: rel.error_type}) AS error_fields
```

#### 4.2 Cluster-Aware Reranking

When multiple hits come from the same semantic cluster, boost the one
with highest vector score and suppress duplicates. Use `IN_CLUSTER`
relationships already in the graph:

```cypher
MATCH (path)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
RETURN cl.label AS cluster, collect(path.id) AS cluster_peers
```

This reduces result redundancy when a query like "electron temperature"
returns 10 paths from the same temperature measurement cluster.

#### 4.3 Version-Validity Materialization (Optional)

During graph build, compute `valid_from_major` and `valid_to_major` properties
on each `IMASNode` derived from `INTRODUCED_IN` and `DEPRECATED_IN` edges.
This would allow simple property-based filtering:

```cypher
WHERE p.valid_from_major <= $dd_version
  AND (p.valid_to_major IS NULL OR p.valid_to_major > $dd_version)
```

Benefits:
- Faster queries (property index vs. relationship traversal + string split)
- Simpler Cypher
- Consistent with the dual property+relationship schema design

Implementation:
- Add `valid_from_major` and `valid_to_major` slots to `IMASNode` in
  `imas_dd.yaml`
- Compute during `_batch_create_path_nodes` and `_batch_mark_paths_deprecated`
- Add range indexes
- Migrate `_dd_version_clause` to use properties instead of relationships

This is Phase 4 because the relationship-based fix (Phase 1) is correct
and sufficient. Materialization is a performance/clarity optimization.

---

## Files Changed

| Phase | File | Change |
|-------|------|--------|
| 1 | `imas_codex/tools/graph_search.py` | Fix `_dd_version_clause`, fix `_search_by_path` scope |
| 1 | `tests/tools/test_dd_version_filtering.py` | New — regression tests for version validity |
| 1 | `tests/tools/test_clusters.py` | Add scope parameter test cases |
| 2 | `imas_codex/tools/graph_search.py` | Add node_category to overview, exports |
| 2 | `tests/tools/test_overview.py` | New or extend — verify data-only counts |
| 3 | `imas_codex/tools/graph_search.py` | Consolidate list query |
| 3 | `imas_codex/ids/tools.py` | Migrate or document legacy dd_version filter |
| 3 | `imas_codex/cli/imas_dd.py` | Migrate or document legacy dd_version filter |
| 4 | `imas_codex/schemas/imas_dd.yaml` | Add valid_from_major, valid_to_major (optional) |
| 4 | `imas_codex/graph/build_dd.py` | Compute materialized validity (optional) |

## Relationship to Existing Plans

- **`error-metadata-filtering.md`**: Phase 1-3 of that plan (schema, build
  pipeline, search filtering) are implemented — `node_category` exists and
  is indexed, embeddings are data-only. This plan addresses the remaining
  gaps where that filtering was not applied to overview and export tools.

- **`imas-tool-quality.md`**: Phase 1 (cluster search), Phase 2 (domain
  resolution), and Phase 3.0 (path annotation stripping) are implemented.
  This plan is complementary — it covers bugs not identified in that plan
  (DD version semantics, cluster scope syntax, overview/export filtering).

- **`shared-tool-consolidation.md`**: This plan does not conflict. The
  `_dd_version_clause` fix is in the shared function that all tools use.
