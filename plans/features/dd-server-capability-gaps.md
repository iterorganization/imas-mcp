# DD Server Capability Gaps: Decision Guide & Implementation Plan

## Context

A/B testing of the `imas` (frozen static) and `imas-dd` (graph-backed) MCP servers
identified 5 capability gaps where REPL-only Cypher queries can extract valuable
information from the graph that no dedicated MCP tool exposes. Each gap represents
a class of query that frontier LLMs cannot answer from training data alone.

This plan serves two purposes:
1. **Decision guide** — weigh the benefit of each new tool against the cost of
   context window consumption. Each phase includes a cost/benefit scoring to guide
   go/no-go decisions independently.
2. **Implementation plan** — each phase is self-contained and assignable to a
   parallel agent. Phases are independent of each other unless noted.

### Tool Context Budget

The DD server currently exposes **17 tools** (15 graph-backed + 2 version tools).
Each tool definition consumes ~400-500 tokens of agent context. Industry best
practice recommends keeping tool definitions under ~40% of available context
(~13,000 tokens for a 32K window). At 17 tools we consume ~7,500-8,500 tokens —
well under budget but approaching the point where each new tool should justify
its existence.

**Budget rule:** A new tool must either (a) expose information unavailable through
existing tools, or (b) reduce multi-step tool chains to a single call, saving
net context in typical agent workflows. Tools that merely reorganize data already
accessible via `search_imas` + `fetch_imas_paths` should be absorbed as parameters
on existing tools instead.

### Current Tool Surface (17 tools)

| # | Tool | Class | Purpose |
|---|------|-------|---------|
| 1 | `search_imas` | GraphSearchTool | Hybrid vector+text search |
| 2 | `check_imas_paths` | GraphPathTool | Path validation |
| 3 | `fetch_imas_paths` | GraphPathTool | Path documentation |
| 4 | `fetch_error_fields` | GraphPathTool | Error field lookup |
| 5 | `list_imas_paths` | GraphListTool | Path enumeration |
| 6 | `get_imas_overview` | GraphOverviewTool | IDS summary |
| 7 | `search_imas_clusters` | GraphClustersTool | Semantic cluster search |
| 8 | `get_imas_identifiers` | GraphIdentifiersTool | Identifier schemas |
| 9 | `get_imas_path_context` | GraphPathContextTool | Cross-IDS relationships |
| 10 | `analyze_imas_structure` | GraphStructureTool | IDS structural analysis |
| 11 | `export_imas_ids` | GraphStructureTool | Full IDS export |
| 12 | `export_imas_domain` | GraphStructureTool | Physics domain export |
| 13 | `explain_concept` | GraphExplainTool | Concept explanation |
| 14 | `get_dd_versions` | VersionTool | DD version metadata |
| 15 | `get_dd_version_context` | VersionTool | Per-path version history |
| 16 | `get_dd_migration_guide` | MigrationGuideTool | Migration guide generation |
| 17 | `get_graph_schema` | (codex) | Graph schema introspection |

---

## Phase 1 — COCOS Field Inventory Tool

### Benefit

COCOS (COordinate COnventionS) is the #1 source of sign errors in fusion code.
When migrating between DD versions or between facilities using different COCOS
conventions, developers must know exactly which fields need sign flips and what
transformation type applies. This information is:

- **Beyond LLM knowledge**: The specific mapping of 554 COCOS-labeled fields
  across 12 transformation types is not in any LLM's training data. It lives
  exclusively in the DD XML metadata, ingested into our Neo4j graph.
- **Currently requires 2+ tool calls**: `explain_concept("COCOS")` returns
  cluster-level COCOS info but not a complete field inventory.
  `analyze_imas_structure` returns COCOS fields per-IDS but requires one call
  per IDS — 21+ calls for full coverage.
- **Critical for code migration**: The `get_dd_migration_guide` tool shows
  COCOS changes between versions but doesn't provide a standalone COCOS field
  reference independent of version transitions.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Beyond-LLM uniqueness | 10/10 | No LLM has this data |
| Reduces multi-call chains | 8/10 | Replaces 21 `analyze_imas_structure` calls |
| Frequency of use | 7/10 | Every COCOS migration, every new facility integration |
| Context cost | Low | ~450 tokens for tool definition |
| **Net benefit** | **High** | **Implement** |

### Design

**Recommendation: New tool `get_cocos_fields`** on `GraphStructureTool`.

This is a new tool rather than a parameter on `analyze_imas_structure` because
(a) it spans all IDSs, (b) it returns a fundamentally different grouping
(by transformation type, not by IDS subtree), and (c) its primary use case
(COCOS migration) is distinct from structural analysis.

```python
@mcp_tool(
    "Get all COCOS-dependent fields across the Data Dictionary, grouped by "
    "transformation type. Use when migrating code between COCOS conventions "
    "or verifying sign handling. "
    "transformation_type: Filter to specific type (e.g., 'psi_like', 'ip_like', 'b0_like'). "
    "ids_filter: Limit to specific IDS (e.g., 'equilibrium'). "
    "dd_version: Filter by DD major version (3 or 4)."
)
async def get_cocos_fields(
    self,
    transformation_type: str | None = None,
    ids_filter: str | None = None,
    dd_version: int | str | None = None,
) -> dict[str, Any]:
```

### Implementation

**File:** `imas_codex/tools/graph_search.py` — add method to `GraphStructureTool`

**Cypher query** (proven in REPL testing):
```cypher
MATCH (p:IMASNode)
WHERE p.cocos_label_transformation IS NOT NULL
  AND p.node_category = 'data'
  [$ids_clause]
  [$dd_clause]
RETURN p.cocos_label_transformation AS transformation_type,
       p.ids AS ids,
       collect(p.id) AS paths,
       count(p) AS field_count
ORDER BY field_count DESC
```

**Return structure:**
```python
{
    "total_fields": 554,
    "transformation_types": [
        {
            "type": "psi_like",
            "field_count": 120,
            "ids_affected": ["equilibrium", "core_profiles", ...],
            "sample_paths": ["equilibrium/time_slice/profiles_1d/psi", ...],
            "sign_factor_11_to_17": -1,  # computed from cocos_sign()
        },
        ...
    ],
    "cocos_convention": {"current_dd": 17, "label": "COCOS 17"},
}
```

**Tests:**
```python
# tests/tools/test_cocos_fields.py
async def test_get_cocos_fields_all():
    result = await tool.get_cocos_fields()
    assert result["total_fields"] > 0
    assert len(result["transformation_types"]) >= 5

async def test_get_cocos_fields_filter_type():
    result = await tool.get_cocos_fields(transformation_type="psi_like")
    assert all(t["type"] == "psi_like" for t in result["transformation_types"])

async def test_get_cocos_fields_filter_ids():
    result = await tool.get_cocos_fields(ids_filter="equilibrium")
    for group in result["transformation_types"]:
        assert all("equilibrium" in p for p in group["sample_paths"])
```

**Registration:** Add to `GraphStructureTool`, auto-registered via `@mcp_tool` decorator.

**Estimated effort:** Small (50-80 lines + tests). Single Cypher query + formatting.

---

## Phase 2 — Path Rename Discovery Tool

### Benefit

When DD versions remove or rename paths, codes that access old paths break
silently — the data access returns empty/null rather than an explicit error.
Path renames are tracked in the graph via `RENAMED_TO` relationships and
`IMASNodeChange` nodes with `change_type='path_renamed'`, but no dedicated tool
exposes them.

- **Beyond LLM knowledge**: The specific rename mappings (e.g.,
  `momentum_tor` → `momentum_phi`, `source` → `provenance`, `label` → `name`)
  are version-specific DD metadata not in training data.
- **Currently requires REPL**: The only way to find renames is raw Cypher:
  `MATCH (old)-[:RENAMED_TO]->(new) RETURN old.id, new.id`.
- **Overlaps with migration guide**: The `get_dd_migration_guide` already
  includes path renames for a specific version transition. The question is
  whether a standalone rename lookup tool adds sufficient value.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Beyond-LLM uniqueness | 9/10 | Version-specific rename chains unavailable to LLMs |
| Reduces multi-call chains | 4/10 | `check_imas_paths` already shows renames for specific paths |
| Frequency of use | 5/10 | Only during major version migrations |
| Context cost | Low | ~400 tokens |
| Overlap with existing | High | `check_imas_paths` handles individual lookups; migration guide handles version transitions |
| **Net benefit** | **Medium** | **Refactor existing tools instead** |

### Design

**Recommendation: Enhance `check_imas_paths` + `get_dd_migration_guide` rather
than adding a new tool.**

The rename use case splits into two patterns:
1. **"Is this specific path renamed?"** — already handled by `check_imas_paths`,
   which checks `RENAMED_TO` edges for not-found paths.
2. **"What paths were renamed between version X and Y?"** — already in
   `get_dd_migration_guide`'s path update section via `_get_renames()`.

Adding a third tool creates discovery ambiguity: an agent wouldn't know whether
to call `get_renamed_paths`, `check_imas_paths`, or `get_dd_migration_guide`.

**Instead, add a `renamed_paths_only` mode to `get_dd_version_context`:**

```python
# Existing tool: get_dd_version_context
# Add parameter: change_type_filter
async def get_dd_version_context(
    self,
    paths: str | list[str] | None = None,       # existing
    change_type_filter: str | None = None,       # NEW: "path_renamed", "units", etc.
    ids_filter: str | None = None,               # NEW: filter by IDS
    from_version: str | None = None,             # NEW: version range start
    to_version: str | None = None,               # NEW: version range end
) -> dict[str, Any]:
```

When `paths` is None but `change_type_filter="path_renamed"`, the tool returns
all renames (optionally filtered by IDS and version range). This avoids a new
tool while making the rename data accessible.

### Implementation

**File:** `imas_codex/tools/version_tool.py` — extend `get_dd_version_context`

**New query mode** (when `paths` is None):
```cypher
MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
WHERE c.change_type = $change_type
  [$version_range_clause]
MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
  [$ids_clause]
RETURN p.id AS path, p.ids AS ids,
       c.old_value AS old_value, c.new_value AS new_value,
       v.id AS version, c.breaking_level AS severity
ORDER BY v.id, p.ids, p.id
LIMIT 200
```

**Tool description update:**
```python
@mcp_tool(
    "Get version change history for IMAS paths, or list changes of a specific type. "
    "paths: One or more IMAS paths to check version history. "
    "change_type_filter: Filter to specific change type when paths is omitted "
    "(e.g., 'path_renamed', 'units', 'data_type', 'cocos_label_transformation'). "
    "ids_filter: Limit to specific IDS. "
    "from_version: Start version for range filter. "
    "to_version: End version for range filter."
)
```

**Tests:**
```python
async def test_version_context_rename_filter():
    result = await tool.get_dd_version_context(
        change_type_filter="path_renamed"
    )
    assert result["change_count"] > 0
    for change in result["changes"]:
        assert change["change_type"] == "path_renamed"

async def test_version_context_with_version_range():
    result = await tool.get_dd_version_context(
        change_type_filter="path_renamed",
        from_version="3.39.0",
        to_version="4.0.0"
    )
    assert all("3.39.0" < c["version"] <= "4.0.0" for c in result["changes"])
```

**Estimated effort:** Medium (60-100 lines refactor + tests).

---

## Phase 3 — Unit Distribution & Analysis

### Benefit

Understanding what units are used across the DD helps with:
- **Data validation**: Verifying that code produces values in expected units
- **Dimensional analysis**: Finding all quantities measured in the same units
- **Cross-IDS consistency**: Confirming equivalent quantities share units

The A/B test REPL query found 5,262 paths using `m` (meter), 3,179 using
`m.s^-1`, etc. — useful aggregated metadata no tool exposes.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Beyond-LLM uniqueness | 6/10 | Unit names are in LLM training data; distribution is not |
| Reduces multi-call chains | 3/10 | `get_imas_path_context` already finds unit companions |
| Frequency of use | 3/10 | Niche — mostly for data validation tooling |
| Context cost | Low | ~400 tokens |
| Overlap with existing | Moderate | `fetch_imas_paths` returns units per path |
| **Net benefit** | **Low** | **Do not implement as standalone tool** |

### Design

**Recommendation: Do not add a new tool. Instead, enhance `get_imas_overview`
to include unit statistics in its response.**

The unit distribution is aggregate metadata that naturally belongs in the
overview response. Adding it as an optional section avoids a new tool while
making the data available.

**Enhancement to `get_imas_overview`:**

Add a `include_unit_stats: bool = False` parameter. When true, append a
`unit_statistics` section to the response:

```python
# In GraphOverviewTool.get_imas_overview()
if include_unit_stats:
    unit_stats = self._gc.query("""
        MATCH (p:IMASNode)-[:HAS_UNIT]->(u:Unit)
        WHERE p.node_category = 'data'
        RETURN u.id AS unit, count(p) AS path_count
        ORDER BY path_count DESC
        LIMIT 25
    """)
    result["unit_statistics"] = {
        "top_units": [{"unit": r["unit"], "count": r["path_count"]}
                      for r in unit_stats],
        "total_units": len(unit_stats),
    }
```

**Estimated effort:** Small (20-30 lines). No new tool registration needed.

---

## Phase 4 — Lifecycle Status Filtering

### Benefit

IMAS DD paths have lifecycle statuses: `active`, `alpha`, `obsolescent`. Agents
writing code against the DD need to know which paths are stable vs experimental.

- **Beyond LLM knowledge**: The lifecycle status of specific paths changes between
  DD versions and is not in LLM training data.
- **Currently accessible**: `fetch_imas_paths` returns `lifecycle_status` per path,
  and `analyze_imas_structure` does not aggregate by lifecycle (but could).
- **Missing capability**: No way to ask "show me all alpha paths in equilibrium"
  without calling `list_imas_paths` + filtering manually.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Beyond-LLM uniqueness | 7/10 | Status per path is version-specific |
| Reduces multi-call chains | 6/10 | Replaces list + manual filter pattern |
| Frequency of use | 5/10 | Important for new code, less for migration |
| Context cost | 0 | No new tool — parameter addition |
| **Net benefit** | **Medium** | **Add as filter parameter to existing tools** |

### Design

**Recommendation: Add `lifecycle_filter` parameter to `list_imas_paths` and
`search_imas`.**

This is the zero-context-cost approach — no new tool definition, just an
additional parameter on tools that already filter paths.

**Enhancement to `list_imas_paths`:**
```python
async def list_imas_paths(
    self,
    paths: str,
    lifecycle_filter: str | None = None,  # NEW: "active", "alpha", "obsolescent"
    ...
) -> ListPathsResult:
```

**Cypher addition:**
```cypher
-- Add to existing path query WHERE clause:
AND ($lifecycle IS NULL OR p.lifecycle_status = $lifecycle)
```

**Enhancement to `search_imas` (search_imas_paths):**
```python
async def search_imas_paths(
    self,
    query: str,
    lifecycle_filter: str | None = None,  # NEW
    ...
) -> SearchPathsResult:
```

Add to vector search WHERE clauses:
```python
if lifecycle_filter:
    _search_where.append(f"path.lifecycle_status = $lifecycle")
    params["lifecycle"] = lifecycle_filter
```

**Also add lifecycle aggregation to `analyze_imas_structure`:**
```cypher
-- New query in analyze_imas_structure:
MATCH (p:IMASNode)
WHERE p.ids = $ids_name AND p.lifecycle_status IS NOT NULL
RETURN p.lifecycle_status AS status, count(p) AS count
ORDER BY count DESC
```

**Tests:**
```python
async def test_list_paths_lifecycle_filter():
    result = await tool.list_imas_paths("equilibrium", lifecycle_filter="alpha")
    # All returned paths should have alpha lifecycle
    for item in result.results:
        for path in item.paths:
            node = await fetch_tool.fetch_imas_paths(path)
            assert node.nodes[0].lifecycle_status == "alpha"

async def test_search_lifecycle_filter():
    result = await tool.search_imas_paths("temperature", lifecycle_filter="active")
    for hit in result.hits:
        assert hit.lifecycle_status == "active"
```

**Estimated effort:** Small (30-50 lines across 3 files + tests).

---

## Phase 5 — Breaking Change Summary Tool

### Benefit

The `get_dd_migration_guide` produces extremely detailed output — 295 KB for the
equilibrium IDS alone in a 3.39.0→4.0.0 migration. When an agent just needs to
understand "how many breaking changes are there and what categories?", this
verbosity is counter-productive and burns context window tokens.

- **Beyond LLM knowledge**: The specific counts (14,031 removals, 113 COCOS changes,
  111 type changes for DD 4.0.0) are absolutely beyond LLM knowledge.
- **Currently available**: The migration guide contains this data but embedded in
  295 KB of per-path detail. There's no summary-only mode.
- **Critical for decision-making**: Before diving into a full migration, agents
  need a quick triage: "Is this a 10-change migration or a 10,000-change one?"

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Beyond-LLM uniqueness | 9/10 | Version-specific change statistics |
| Reduces multi-call chains | 9/10 | Replaces full migration guide + manual counting |
| Frequency of use | 7/10 | Every migration triage conversation |
| Context cost | Low | ~450 tokens |
| Context *savings* | High | Avoids loading 295KB migration guide for triage |
| **Net benefit** | **High** | **Implement** |

### Design

**Recommendation: Add a `summary_only` parameter to `get_dd_migration_guide`
rather than a new tool.** This is the most context-efficient approach — agents
already know about the migration guide tool, and a `summary_only=True` parameter
is cheaper than learning a new tool exists.

**Enhancement to `get_dd_migration_guide`:**

```python
@mcp_tool(
    "Generate a migration guide between two DD versions. "
    "from_version (required): Source DD version (e.g., '3.39.0'). "
    "to_version (required): Target DD version (e.g., '4.0.0'). "
    "ids_filter: Restrict to specific IDS. "
    "summary_only: If true, return only statistics without per-path details. "
    "include_recipes: Include code search patterns (default true)."
)
async def get_dd_migration_guide(
    self,
    from_version: str,
    to_version: str,
    ids_filter: str | None = None,
    summary_only: bool = False,      # NEW
    include_recipes: bool = True,
) -> dict[str, Any]:
```

When `summary_only=True`, the tool calls only `_get_change_summary()` and
`_get_version_cocos()`, skipping the expensive per-path queries (`_get_renames()`,
`_get_cocos_table()`, `_get_removals()`, `_get_additions()`, etc.). This reduces
response from ~295KB to ~2KB.

**Summary response structure:**
```python
{
    "from_version": "3.39.0",
    "to_version": "4.0.0",
    "cocos_change": "11 → 17",
    "summary": {
        "total_changes": 15371,
        "by_type": {
            "path_removed": {"count": 14031, "breaking": 14031},
            "path_added": {"count": 663, "breaking": 0},
            "documentation": {"count": 560, "breaking": 12},
            "cocos_label_transformation": {"count": 113, "breaking": 113},
            "data_type": {"count": 111, "breaking": 111},
        },
        "breaking_total": 14267,
        "optional_total": 1104,
        "ids_affected": ["equilibrium", "core_profiles", ...],
    },
    "recommendation": "Major migration — 14,267 breaking changes. Use full guide with ids_filter for per-IDS migration.",
}
```

**Implementation:**

**File:** `imas_codex/tools/migration_guide.py` — add `summary_only` branch

```python
def build_migration_guide(
    gc: GraphClient,
    from_version: str,
    to_version: str,
    ids_filter: str | None = None,
    summary_only: bool = False,
    include_recipes: bool = True,
) -> CodeMigrationGuide:
    version_range = _resolve_version_range(gc, from_version, to_version)
    change_summary = _get_change_summary(gc, version_range, ids_filter)
    from_cocos = _get_version_cocos(gc, from_version)
    to_cocos = _get_version_cocos(gc, to_version)

    if summary_only:
        return CodeMigrationGuide(
            from_version=from_version,
            to_version=to_version,
            cocos_change=f"{from_cocos} → {to_cocos}" if from_cocos and to_cocos else None,
            change_summary=change_summary,
            # Skip all per-path details
            required_actions=[],
            optional_actions=[],
            ...
        )

    # ... existing full guide logic
```

**Tests:**
```python
async def test_migration_guide_summary_only():
    result = await tool.get_dd_migration_guide(
        "3.39.0", "4.0.0", summary_only=True
    )
    assert "summary" in result
    assert result["summary"]["total_changes"] > 0
    assert "path_removed" in result["summary"]["by_type"]
    # Should not have per-path details
    assert len(result.get("required_actions", [])) == 0

async def test_migration_guide_summary_vs_full_consistency():
    summary = await tool.get_dd_migration_guide("3.39.0", "4.0.0", summary_only=True)
    full = await tool.get_dd_migration_guide("3.39.0", "4.0.0", ids_filter="equilibrium")
    # Summary totals should be >= IDS-filtered totals
    assert summary["summary"]["total_changes"] >= full["total_actions"]
```

**Estimated effort:** Medium (40-60 lines + tests). The change summary query
already exists in `_get_change_summary()`.

---

## Phase 6 — Refactor `explain_concept` for Richer Output

### Benefit

The A/B test revealed that `explain_concept` is the weakest graph-backed tool:
- For "COCOS", it returns cluster data and sign convention paths but lacks the
  structured COCOS field inventory that Phase 1 would provide.
- It uses only text matching (`toLower(label) CONTAINS $concept`), missing
  concepts that have semantic but not lexical overlap.
- The IMAS server's LLM-generated explanations scored 9/10 for quality,
  while the graph-backed version scored 6/10 — the gap is in synthesis,
  not data availability.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Improves existing tool quality | 8/10 | Directly addresses A/B test weakness |
| Context cost | 0 | No new tool — existing tool improvement |
| User-facing impact | 7/10 | First tool agents call for concept questions |
| **Net benefit** | **High** | **Implement** |

### Design

**Recommendation: Refactor `explain_concept` to use vector search + cross-reference
the COCOS field inventory from Phase 1.**

**Changes:**

1. **Add vector search** for concept matching. The current text-only CONTAINS
   matching misses semantic synonyms. Use the existing `cluster_embedding` index:

```python
# Replace text-only cluster search with hybrid:
# 1. Vector search on cluster embeddings
embedding = self._embed_query(concept)
vector_clusters = self._gc.query(f"""
    {build_vector_search("cluster_embedding", "IMASSemanticCluster",
                          k="10", node_alias="c", score_alias="score")}
    WITH c, score WHERE score > 0.4
    OPTIONAL MATCH (m:IMASNode)-[:IN_CLUSTER]->(c)
    WITH c, score, collect(DISTINCT m.id)[..15] AS paths
    RETURN c.id AS id, c.label AS label, c.description AS description,
           c.scope AS scope, paths, score
    ORDER BY score DESC
    LIMIT 5
""", embedding=embedding)

# 2. Text match (existing, for exact keyword hits)
text_clusters = ... # existing CONTAINS query
```

2. **Cross-reference COCOS fields** when the concept is COCOS-related. If
   Phase 1 is implemented, delegate to `get_cocos_fields()`. Otherwise,
   use the existing Cypher query for COCOS-labeled paths but group by
   transformation type rather than flat listing.

3. **Add DD metadata context**: Include the current DD version, COCOS
   convention, and link to the migration guide tool for version-specific
   COCOS changes.

4. **Improve output structure** with clear section types for agent consumption:
   - `"clusters"` → semantic concept groups with example paths
   - `"cocos"` → COCOS metadata and field categories
   - `"identifiers"` → related enumeration schemas
   - `"ids"` → related IDS descriptions
   - `"paths"` → relevant data paths with documentation

**File:** `imas_codex/tools/graph_search.py` — refactor `GraphExplainTool.explain_concept()`

**Estimated effort:** Medium (80-120 lines refactor + tests). Requires embedding
access (already available in the module).

---

## Phase 7 — Search Parameter Unification (Existing Tool Improvements)

### Benefit

The A/B test identified several gaps in existing tools that don't require new
tools but do require parameter additions for feature parity. These are collected
here as a single phase since they share the same implementation pattern.

### Cost/Benefit Score

| Factor | Score | Rationale |
|--------|:-----:|-----------|
| Context cost | 0 | Parameter additions, not new tools |
| Reduces agent friction | 8/10 | Fewer multi-step workarounds |
| **Net benefit** | **High** | **Implement** |

### Changes

#### 7a. Add `physics_domain` filter to `search_imas` and `list_imas_paths`

The IMAS server's tools accept physics domain filtering. The DD server does not.

```python
# search_imas_paths — add parameter:
physics_domain: str | None = None  # e.g., "magnetics", "equilibrium"

# Cypher addition:
if physics_domain:
    _search_where.append("path.physics_domain = $domain")
    params["domain"] = physics_domain
```

#### 7b. Add `node_type` filter to `list_imas_paths`

Allow filtering by `node_type` (e.g., `'dynamic'`, `'static'`, `'constant'`).

```python
# list_imas_paths — add parameter:
node_type: str | None = None

# Cypher:
AND ($node_type IS NULL OR p.node_type = $node_type)
```

#### 7c. Add `include_children_preview` to `fetch_imas_paths`

When fetching a structure node, optionally include a preview of its children
(already implemented in `search_imas` via `_enrich_children()` but not available
in `fetch_imas_paths`).

```python
# fetch_imas_paths — add parameter:
include_children: bool = False

# When true, call _enrich_children() for structure nodes
```

#### 7d. Improve `get_imas_overview` response with tool count

Currently the overview includes a hardcoded `mcp_tools` list. Auto-generate
it from the registered tools to stay in sync as tools are added/removed.

**Estimated effort:** Small per item (10-20 lines each). Total: 40-80 lines + tests.

---

## Decision Summary

| Phase | Capability | Recommendation | Net New Tools | Context Impact |
|-------|-----------|----------------|:-------------:|:--------------:|
| **1** | COCOS field inventory | **New tool** `get_cocos_fields` | +1 | +450 tokens |
| **2** | Path rename discovery | **Enhance** `get_dd_version_context` | 0 | ~0 |
| **3** | Unit distribution | **Enhance** `get_imas_overview` | 0 | ~0 |
| **4** | Lifecycle filtering | **Add parameter** to 3 existing tools | 0 | ~0 |
| **5** | Breaking change summary | **Add parameter** to `get_dd_migration_guide` | 0 | ~0 |
| **6** | Explain concept quality | **Refactor** `explain_concept` | 0 | ~0 |
| **7** | Search parameter gaps | **Add parameters** to existing tools | 0 | ~0 |

**Net tool count change:** 17 → 18 (+1 new tool: `get_cocos_fields`)
**Net context impact:** +~450 tokens (one new tool definition)

This conservative approach maximizes capability while minimizing context window
consumption. Only Phase 1 adds a genuinely new tool — all other capabilities are
absorbed into existing tools via parameter additions and refactors.

---

## Implementation Order & Parallelism

All 7 phases are independent and can be implemented in parallel by separate agents.

```
Phase 1: get_cocos_fields ──────────────────── Agent A
Phase 2: Enhance get_dd_version_context ────── Agent B
Phase 3: Enhance get_imas_overview ─────────── Agent C (tiny)
Phase 4: Add lifecycle_filter parameters ───── Agent C (combines with 3)
Phase 5: Add summary_only to migration guide ─ Agent D
Phase 6: Refactor explain_concept ──────────── Agent E (after Phase 1 if cross-ref desired)
Phase 7: Search parameter unification ──────── Agent F
```

**Dependency:** Phase 6 can optionally cross-reference Phase 1's `get_cocos_fields`
for COCOS concept explanations. If implementing in parallel, Phase 6 should use
the existing COCOS Cypher query directly and a follow-up commit can add the
cross-reference once Phase 1 lands.

**Maximum parallelism:** 5-6 agents. Phases 3+4 are small enough for one agent.

## Documentation Updates

After all phases are implemented:

| Target | Update Needed |
|--------|--------------|
| `AGENTS.md` | Update "Quick Reference" tool table, add `get_cocos_fields` |
| `plans/README.md` | Add this plan, mark status |
| `agents/schema-reference.md` | Auto-generated — rebuild with `uv run build-models` |
| MCP tool descriptions | Updated inline via `@mcp_tool` decorator changes |
