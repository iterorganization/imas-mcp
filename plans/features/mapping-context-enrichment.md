# Mapping Pipeline IMAS Context Enrichment

**Depends on: `imas-mcp-gap-closure.md` Phases 1, 3, 6**

**Priority: HIGH — directly impacts LLM mapping output quality**

Enrich the IMAS context injected into each LLM call of the `discover imas map`
pipeline so the LLM makes better-informed mapping decisions using graph data
that already exists but is not currently surfaced.

**Dual-purpose principle:** Every enrichment that is also useful to MCP consumers
must be delivered as a shared tool enhancement in `imas_codex/tools/`, not as a
standalone function in `ids/tools.py`. The mapping pipeline consumes shared tools
via Phase 6 delegation. New standalone functions in `ids/tools.py` are only for
mapping-specific graph traversals that have no MCP consumer value.

---

## Current State: What Each LLM Step Receives

### Step 1 — `assign_sections` (section_assignment.md)

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_sources` | `query_signal_sources()` | SignalSource nodes: id, group_key, description, physics_domain, rep_unit, rep_cocos, sample_accessors, existing MAPS_TO_IMAS |
| `imas_subtree` | `fetch_imas_subtree()` | IMASNode flat list: id, name, data_type, node_type, documentation, units |
| `semantic_results` | `search_imas_semantic()` | Vector-searched IMASNode: id, documentation, data_type, node_type, units, score |

**What the LLM does NOT see:**
- Cluster membership — which paths are semantically grouped
- Physics domain labels on individual paths
- Identifier schemas on struct-array sections (e.g., geometry types, coil types)
- DD version scoping — gets all paths including deprecated ones

### Step 2 — `map_signals` (signal_mapping.md) — per section

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_source_detail` | (formatted from groups) | Source metadata: description, physics_domain, units, sign_convention, cocos, members, accessors, existing mappings |
| `imas_fields` | `fetch_imas_fields()` or `fetch_imas_subtree(leaf_only=True)` | Path, data_type, units, documentation, ndim, physics_domain, cluster_labels, coordinates |
| `cocos_paths` | `get_sign_flip_paths()` | COCOS sign-flip paths for this IDS |
| `source_cocos` | (built from rep_cocos) | Per-source COCOS context |
| `existing_mappings` | `search_existing_mappings()` | Existing IMASMapping + POPULATES + MAPS_TO_IMAS state |
| `code_references` | `fetch_source_code_refs()` | Code snippets showing how signal is read |
| `unit_analysis` | `analyze_units()` | Pint-based unit compatibility checks |

**What the LLM does NOT see:**
- Identifier schema valid values for typed fields (geometry_type, grid_type, etc.)
- Version change history (sign convention changes, unit changes, renames)
- Cross-facility mapping precedent (what other facilities mapped to these paths)
- DD version scoping

### Step 3 — `discover_assembly` (assembly.md) — per section

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_mappings` | (formatted from Step 2) | Mapped source→target pairs with transforms |
| `imas_section_structure` | `fetch_imas_subtree()` | Section subtree paths |
| `source_metadata` | (formatted from groups) | Basic source info |

**What the LLM does NOT see:**
- Coordinate spec details — dimensionality, axis semantics, shared coordinates
- Identifier schema options — enumeration values needed by assembly code
- Array sizing constraints from coordinate specs

### Step 4 — `validate_mappings` — programmatic, no LLM

Uses `check_imas_paths()` and `get_sign_flip_paths()`. Does not DD-version-scope
its path existence checks.

---

## Gap Assessment

### HIGH impact — LLM currently lacks information that exists in the graph

| # | Gap | Affects step | Graph data available | Expected quality improvement |
|---|-----|-------------|---------------------|------------------------------|
| 1 | **DD version scoping** | All steps | `DDVersion`, `INTRODUCED_IN`, `DEPRECATED_IN` | Eliminates ghost paths from old DD versions that confuse section assignment and produce mappings to deprecated paths. Reduces escalations. |
| 2 | **Identifier schemas for typed fields** | Steps 2, 3 | `IdentifierSchema` via `HAS_IDENTIFIER_SCHEMA` | Fields like `geometry_type`, `grid_type`, `coordinate_system_type` have enumerated valid values. Without them the LLM guesses or omits these fields entirely. Assembly code also needs valid values for initialization. |
| 3 | **Version change history** | Step 2 | `IMASNodeChange` via `FOR_IMAS_PATH` | Fields with historical `sign_convention`, `coordinate_convention`, or `units` changes need version-aware transforms. Currently the LLM only sees COCOS labels — not whether a field's convention changed between DD versions. |
| 4 | **Cluster context in section assignment** | Step 1 | `IMASSemanticCluster` via `IN_CLUSTER` on subtree paths | The LLM sees a flat path list and has to infer which paths form physics-coherent groups. Cluster labels (e.g., "boundary geometry", "MHD stability") directly indicate section purpose, improving assignment accuracy for ambiguous sources. |

### MEDIUM impact — additional context improves quality but is not critical

| # | Gap | Affects step | Graph data available | Expected quality improvement |
|---|-----|-------------|---------------------|------------------------------|
| 5 | **Coordinate specs in assembly** | Step 3 | `IMASCoordinateSpec` via `HAS_COORDINATE` | Assembly patterns depend on understanding dimensions (time, space, channel). Coordinate spec data (axis semantics, sizing) directly informs `array_per_node` vs `concatenate` vs `matrix_assembly` selection. |
| 6 | **Cross-facility mapping precedent** | Steps 1, 2 | `MAPS_TO_IMAS` from other facilities' SignalSources | When another facility has already mapped similar signals to an IMAS section, this provides strong precedent for correct assignment. Currently only same-facility existing mappings are shown. |
| 7 | **Cross-IDS path context** | Step 2 | `get_imas_path_context` shared tool (Phase 3 of gap closure plan) | Shows how a target field connects to related fields in other IDS via shared clusters/coordinates/units. Helps the LLM understand dimensional conventions. |

---

## MCP Tool Assessment — Existing Tools vs Enrichment Needs

Each enrichment was assessed against the existing DD MCP server tools
(`imas_codex/tools/`) and the Codex agents MCP server (`imas_codex/llm/server.py`).

### Existing DD MCP tool surface (9 tools via `Tools.register(mcp)`)

| Tool | Class | Relevant enrichment data |
|------|-------|-------------------------|
| `search_imas_paths` | `GraphSearchTool` | Returns `coordinates`, `has_identifier_schema`, `introduced_after_version`, `lifecycle_status` per hit. Does NOT return identifier schema *options* or version *change history*. |
| `check_imas_paths` | `GraphPathTool` | Validates paths, detects RENAMED_TO. No DD version scoping. |
| `fetch_imas_paths` | `GraphPathTool` | Returns documentation, units, data_type, cluster_labels, coordinates. Does NOT return identifier schemas or version changes. |
| `list_imas_paths` | `GraphListTool` | Lists IDS paths. No cluster grouping or DD version scoping. |
| `get_imas_overview` | `GraphOverviewTool` | IDS-level statistics. No section-cluster breakdowns. |
| `search_imas_clusters` | `GraphClustersTool` | Semantic or path-based cluster search. Requires a `query` string — cannot list all clusters for an IDS programmatically. |
| `get_imas_identifiers` | `GraphIdentifiersTool` | Global keyword search over IdentifierSchema nodes. Cannot query "which schemas apply to these specific paths?" |
| `query_imas_graph` | `CypherTool` | Raw read-only Cypher. Escape hatch for arbitrary queries. |
| `get_dd_versions` | `VersionTool` | Lists DDVersion nodes. No per-path version history. |
| `get_dd_graph_schema` | `SchemaTool` | Schema introspection. |

### Enrichment-to-tool mapping

| Enrichment | Existing tool overlap | Action |
|------------|----------------------|--------|
| **E1: DD version scoping** | All tools lack `dd_version` param | **Shared tool enhancement** — gap closure Phase 1.1. Already planned. |
| **E2: Cluster context** | `search_imas_clusters` partial — requires text query, not IDS-scoped listing | **Shared tool enhancement** — add `ids_name` mode to `search_imas_clusters` |
| **E3: Identifier schemas per path** | `get_imas_identifiers` global search, `fetch_imas_paths` lacks schema data | **Shared tool enhancement** — extend `fetch_imas_paths` to return identifier schema options |
| **E4: Version change history** | `_get_version_context()` in `search_tools.py` but not shared, gap closure Phase 1.3 puts it on `GraphSearchTool` only | **Shared tool enhancement** — also add to `fetch_imas_paths` |
| **E5: Coordinate specs** | `fetch_imas_paths` returns coordinate IDs but not grouped per-section | Covered by gap closure Phase 3 (`get_imas_path_context`) coordinate traversal |
| **E6: Cross-facility precedent** | No existing tool | **Pipeline-specific** — stays in `ids/tools.py`. IMASMapping traversal is mapping-specific. |

### Conclusion

Three existing shared tools should be enhanced to deliver enrichments E2–E4.
These enhancements benefit both MCP consumers and the mapping pipeline
simultaneously. Only E6 requires a mapping-specific function. E1, E5, and E7
are already addressed by the gap closure plan.

---

## Shared Tool Enhancements (amendments to gap closure plan)

These amendments extend `imas-mcp-gap-closure.md` Phase 1. They should be
implemented as part of that phase, not as separate work.

### T1: Extend `fetch_imas_paths` — identifier schemas + version history

**Location:** `imas_codex/tools/graph_search.py` → `GraphPathTool.fetch_imas_paths()`

**Current state:** The enrichment Cypher JOINs to `HAS_UNIT`, `IN_CLUSTER`, and
`HAS_COORDINATE` but **not** to `HAS_IDENTIFIER_SCHEMA` or `FOR_IMAS_PATH`
(via `IMASNodeChange`).

**Enhancement:** Add two additional OPTIONAL MATCHes to the existing query:

```python
async def fetch_imas_paths(
    self,
    paths: str | list[str],
    ids: str | None = None,
    include_version_history: bool = False,   # NEW
    dd_version: int | None = None,           # NEW (Phase 1.1)
    ctx: Context | None = None,
) -> FetchPathsResult:
```

Extend the Cypher:

```cypher
MATCH (p:IMASNode {id: $path})
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
-- NEW: identifier schema
OPTIONAL MATCH (p)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
-- NEW: version changes (when requested)
OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
  WHERE change.semantic_change_type IN
        ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
RETURN ...
       ident.name AS identifier_schema_name,
       ident.options AS identifier_schema_options,
       collect(DISTINCT {version: change.version,
                         type: change.semantic_change_type,
                         summary: change.summary})[..5] AS version_changes
```

Update `IdsNode` model to carry optional `identifier_schema` and `version_changes`.

**MCP consumer benefit:** Any MCP user calling `fetch_imas_paths` on a path like
`magnetics/bpol_probe/identifier` gets the valid enumeration options inline.
No separate `get_imas_identifiers` call needed for path-specific schemas.

**Mapping pipeline benefit:** After Phase 6 consolidation, `ids/tools.py`
`fetch_imas_fields()` delegates to `fetch_imas_paths` and the identifier
schema data flows into the `signal_mapping.md` prompt automatically.

### T2: Extend `search_imas_clusters` — IDS-scoped listing mode

**Location:** `imas_codex/tools/graph_search.py` → `GraphClustersTool.search_imas_clusters()`

**Current state:** Requires a `query` string (text or path). Cannot list
"all clusters for this IDS" without a search query.

**Enhancement:** Make `query` optional when `ids_filter` is provided:

```python
@mcp_tool(
    "Search semantic clusters of related IMAS data paths. "
    "query: Natural language description, exact IMAS path, or omit to list all clusters for an IDS. "
    "ids_filter: Filter by IDS (required when query is omitted). "
    "scope: Filter by cluster scope — 'global', 'domain', or 'ids'. "
    "section_only: If true, only return clusters containing structural sections."
)
async def search_imas_clusters(
    self,
    query: str | None = None,     # CHANGED: now optional
    scope: str | None = None,
    ids_filter: str | list[str] | None = None,
    section_only: bool = False,    # NEW
    dd_version: int | None = None, # NEW (Phase 1.1)
    ctx: Context | None = None,
) -> dict[str, Any]:
```

When `query` is None and `ids_filter` is provided, run a listing query:

```cypher
MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE p.ids IN $ids_filter
  AND ($section_only = false
       OR p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
WITH c, collect(DISTINCT p.id) AS section_paths,
     collect(DISTINCT p.ids) AS ids_covered
RETURN c.id AS id, c.label AS label, c.description AS description,
       c.scope AS scope, c.cross_ids AS cross_ids,
       ids_covered AS ids_names,
       section_paths AS paths
ORDER BY size(section_paths) DESC
```

**MCP consumer benefit:** An LLM agent can call
`search_imas_clusters(ids_filter="pf_active", section_only=true)` to understand
the cluster organization of an IDS without needing to formulate a search query.

**Mapping pipeline benefit:** Replaces the proposed standalone
`fetch_section_clusters()` in `ids/tools.py`. After Phase 6, the pipeline calls
the shared tool.

### T3: Add `get_dd_version_context` to `VersionTool`

**Location:** `imas_codex/tools/version_tool.py`

**Current state:** `get_dd_versions()` returns the version list. No per-path
version change history.

**Enhancement:** Add a second tool method:

```python
@mcp_tool(
    "Get version change history for specific IMAS paths. "
    "Shows sign convention, coordinate convention, unit, and definition changes "
    "across DD versions. Use to understand how a field's semantics evolved. "
    "paths (required): One or more IMAS paths to check."
)
async def get_dd_version_context(
    self,
    paths: str | list[str],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get version change context for IMAS paths."""
    path_list = [p.strip() for p in paths.replace(",", " ").split()
                 if p.strip()] if isinstance(paths, str) else list(paths)
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (p:IMASNode {id: pid})
        OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
        WHERE change.semantic_change_type IN
              ['sign_convention', 'coordinate_convention',
               'units', 'definition_clarification']
        RETURN p.id AS id,
               count(change) AS change_count,
               collect({version: change.version,
                        type: change.semantic_change_type,
                        summary: change.summary})[..5] AS notable_changes
    """
    results = self._gc.query(cypher, path_ids=path_list)
    paths_with_changes = [r for r in results if r["change_count"] > 0]
    return {
        "paths_queried": len(path_list),
        "paths_with_changes": len(paths_with_changes),
        "results": results,
    }
```

Register in `Tools.__init__` (already has `self.version_tool`).
Add delegation method to `Tools`.

**MCP consumer benefit:** Standalone version history query without needing
`fetch_imas_paths(include_version_history=True)`. Useful for targeted
investigation of specific paths.

**Mapping pipeline benefit:** Used directly by E4 implementation when the
mapping pipeline has a target `dd_version` and wants change history for
section fields.

---

## Codex Agents MCP Server Updates

**Location:** `imas_codex/llm/server.py`

The Codex agents MCP server currently only exposes `search_imas` as an IMAS DD
MCP tool. The gap closure plan (Phase 2) already proposes promoting 6 IMAS tools
from REPL to MCP surface. The enrichments in this plan require one additional
tool promotion:

### Promote `get_dd_version_context` to Codex MCP

```python
@self.mcp.tool()
def get_dd_version_context(
    paths: str,
) -> str:
    """Get version change history for IMAS paths.

    Shows sign convention, coordinate convention, unit, and definition
    changes across DD versions.

    Args:
        paths: Space-delimited IMAS paths
    """
    tools = _get_imas_tools()
    result = _run_async(tools.get_dd_version_context(paths=paths))
    return format_version_context_report(result)
```

The other enhanced tools (`fetch_imas_paths` with identifiers/version_history,
`search_imas_clusters` with IDS listing mode) are already promoted in the
gap closure plan's Phase 2. The enhancements from T1 and T2 above are
parameter additions to already-promoted tools — no additional wiring needed.

---

## Relationship to `imas-mcp-gap-closure.md`

The gap closure plan addresses the **infrastructure** for these enrichments.
This plan adds **three amendments** (T1, T2, T3) to the gap closure plan's
Phase 1 and one tool promotion to Phase 2:

| Gap closure phase | What it provides | What this plan amends/adds |
|-------------------|-----------------|---------------------------|
| Phase 1.1 — DD version filter | `_dd_version_clause()` + parameter on all shared tools | **E1** — Thread `dd_version` through `gather_context()` and all pipeline calls |
| Phase 1 — Shared tool enhancement | Enhanced `GraphSearchTool` | **T1** — Also enhance `GraphPathTool.fetch_imas_paths` with identifier schemas + version history |
| Phase 1 — Shared tool enhancement | Enhanced `GraphClustersTool` | **T2** — Add IDS-scoped listing mode to `search_imas_clusters` |
| Phase 1 — Shared tool enhancement | `VersionTool.get_dd_versions` | **T3** — Add `get_dd_version_context` for per-path change history |
| Phase 1.3 — Version context | `include_version_context` on search | Subsumed by T1 + T3 enrichement |
| Phase 2 — Wire Codex MCP | 6 tools promoted to MCP surface | **Add** `get_dd_version_context` promotion |
| Phase 3 — `get_imas_path_context` | Cross-IDS relationship traversal | **E5** — Coordinate spec groupings available via this tool |
| Phase 6 — Consolidation | `ids/tools.py` delegates to shared tools | **Prerequisite** — after delegation, T1/T2/T3 data flows to pipeline automatically |

**After both plans are implemented:**
- `fetch_imas_paths` returns identifier schemas + version history → flows into
  `signal_mapping.md` and `assembly.md` prompts via Phase 6 delegation
- `search_imas_clusters(ids_filter=..., section_only=True)` replaces standalone
  `fetch_section_clusters()` → flows into `section_assignment.md` prompt
- `get_dd_version_context` provides targeted change history → flows into
  `signal_mapping.md` prompt for version-aware transforms
- All three tools are simultaneously available on both MCP servers for external
  consumers

---

## Implementation

### E1: Thread DD version through the pipeline

**Files:** `ids/mapping.py`, `ids/tools.py`

Pass `dd_version` from `generate_mapping()` through `gather_context()` and into
every IMAS query:

```python
def gather_context(facility, ids_name, *, gc, dd_version=None):
    groups = query_signal_sources(facility, gc=gc)
    subtree = fetch_imas_subtree(ids_name, gc=gc, dd_version=dd_version)
    semantic = search_imas_semantic(
        f"{facility} {ids_name}", ids_name, gc=gc, k=10, dd_version=dd_version
    )
    existing = search_existing_mappings(facility, ids_name, gc=gc)
    cocos_paths = get_sign_flip_paths(ids_name)
    return {
        "groups": groups,
        "subtree": subtree,
        "semantic": semantic,
        "existing": existing,
        "cocos_paths": cocos_paths,
        "dd_version": dd_version,
    }
```

Similarly thread into `fetch_imas_fields()` and `check_imas_paths()` calls
in `map_signals()` and `validate_mappings()`.

After Phase 6, these functions delegate to shared tools that already have
`dd_version` parameters (Phase 1.1). The `dd_version` just needs to be passed
through.

### E2: Inject cluster context into section assignment

**Files:** `ids/mapping.py`, `llm/prompts/mapping/section_assignment.md`

After T2 is implemented, the pipeline calls the shared tool:

```python
# In gather_context():
from imas_codex.tools import Tools

tools = Tools(graph_client=gc)
cluster_result = _run_async(tools.search_imas_clusters(
    ids_filter=ids_name, section_only=True
))
clusters = cluster_result.get("clusters", [])
```

After Phase 6 consolidation, this becomes a simple call to the shared tool.

Format and inject into section assignment prompt:

```markdown
### Section Clusters

These semantic clusters group related IDS sections by physics concept:

{{ section_clusters }}
```

### E3: Inject identifier schemas into signal mapping

**Files:** `ids/mapping.py`, `llm/prompts/mapping/signal_mapping.md`, `llm/prompts/mapping/assembly.md`

After T1 is implemented, `fetch_imas_paths` returns identifier schema data
inline. After Phase 6, `fetch_imas_fields()` delegates to `fetch_imas_paths`
and the schema data is available automatically:

```python
# In map_signals(), after fetching fields:
# identifier_schemas is now part of each field record from fetch_imas_fields()
schemas = {
    f["id"]: f["identifier_schema"]
    for f in (subtree_fields or fields)
    if f.get("identifier_schema")
}
```

Add to `signal_mapping.md`:

```markdown
### Identifier Schemas

These target fields have enumerated valid values. Use these exact values
when populating identifier/type fields:

{{ identifier_schemas }}
```

Also inject into `assembly.md` template for assembly code generation.

### E4: Inject version change history into signal mapping

**Files:** `ids/mapping.py`, `llm/prompts/mapping/signal_mapping.md`

After T1 and T3 are implemented, version history is available via two paths:
- Inline from `fetch_imas_paths(include_version_history=True)` — comes with
  field details after Phase 6 delegation
- Standalone via `get_dd_version_context(paths)` — for targeted queries

```python
# In map_signals(), after fetching fields:
# Version history is now part of each field record from fetch_imas_fields()
# OR call standalone tool for targeted history:
field_paths = [f["id"] for f in (subtree_fields or fields)]
version_ctx = _run_async(tools.get_dd_version_context(paths=field_paths))
```

Add to `signal_mapping.md`:

```markdown
### Version History

Notable changes to target fields across DD versions. Check whether your
target DD version is before or after these changes:

{{ version_context }}
```

### E5: Inject coordinate specs into assembly prompt

**Files:** `ids/mapping.py`, `llm/prompts/mapping/assembly.md`

After gap closure Phase 3 (`get_imas_path_context`), coordinate spec
groupings are available via the shared tool. The pipeline calls:

```python
# In discover_assembly(), for each section:
for field_path in key_section_fields:
    ctx = _run_async(tools.get_imas_path_context(
        path=field_path, relationship_types="coordinate"
    ))
```

Alternatively, a lightweight section-scoped coordinate query can stay in
`ids/tools.py` since it groups coordinates by section prefix rather than
traversing cross-IDS relationships:

```python
def fetch_section_coordinate_specs(
    ids_name: str, section_path: str, *, gc: GraphClient
) -> list[dict]:
    """Return coordinate specs for fields in a section."""
    prefix = f"{ids_name}/{section_path}" if not section_path.startswith(ids_name) else section_path
    cypher = """
        MATCH (p:IMASNode)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        WHERE p.id STARTS WITH $prefix
        RETURN coord.id AS coordinate_path,
               collect(DISTINCT p.id) AS fields_using,
               count(DISTINCT p) AS field_count
        ORDER BY field_count DESC
    """
    return gc.query(cypher, prefix=prefix + "/")
```

Add to `assembly.md`:

```markdown
### Coordinate Specifications

These coordinate axes are used by fields in this section. Fields sharing
a coordinate should share that array dimension:

{{ coordinate_specs }}
```

### E6: Cross-facility mapping precedent (pipeline-specific, optional)

**Files:** `ids/tools.py` (new function), `ids/mapping.py`,
`llm/prompts/mapping/section_assignment.md`

This is the one enrichment that stays in `ids/tools.py` — traversing
`IMASMapping` nodes is mapping-pipeline-specific and has no MCP consumer value.

```python
def fetch_cross_facility_mappings(
    ids_name: str, exclude_facility: str, *, gc: GraphClient
) -> list[dict]:
    """Return mappings from other facilities to this IDS."""
    cypher = """
        MATCH (m:IMASMapping)-[:POPULATES]->(ip:IMASNode)
        WHERE m.ids_name = $ids_name
          AND m.facility_id <> $exclude
          AND m.status IN ['active', 'validated']
        RETURN m.facility_id AS facility, ip.id AS section_path,
               m.status AS status
        ORDER BY m.facility_id, ip.id
    """
    return gc.query(cypher, ids_name=ids_name, exclude=exclude_facility)
```

Inject into section assignment:

```markdown
### Cross-Facility Precedent

Other facilities have mapped signals to these IDS sections:

{{ cross_facility_mappings }}
```

---

## Prompt Template Changes Summary

| Template | New context variables | Source |
|----------|----------------------|--------|
| `section_assignment.md` | `section_clusters` | `search_imas_clusters` (T2 shared tool) |
| `section_assignment.md` | `cross_facility_mappings` (optional) | `fetch_cross_facility_mappings` (pipeline-specific) |
| `signal_mapping.md` | `identifier_schemas` | `fetch_imas_paths` (T1 shared tool) |
| `signal_mapping.md` | `version_context` | `get_dd_version_context` (T3 shared tool) |
| `assembly.md` | `coordinate_specs` | `get_imas_path_context` (gap closure Phase 3) or pipeline function |
| `assembly.md` | `identifier_schemas` | `fetch_imas_paths` (T1 shared tool) |
| `validation.md` | (no new variables) | `dd_version` already present |

---

## Cross-Plan Analysis

### Outstanding plans in `plans/features/`

| Plan | Status | Priority | Overlap with this plan |
|------|--------|----------|----------------------|
| `imas-mcp-gap-closure.md` | Planning | Critical | **Strong** — this plan amends Phases 1, 2, 3. T1/T2/T3 should be implemented as part of gap closure. |
| `check-mcp-remediation.md` | Planning (Plans G+H) | High | **None** — check pipeline fixes and `search_signals` enhancements address facility-specific signal data, not IMAS DD context. Plans G.1–G.5 already implemented. |
| `jet-magnetics-quality-remediation.md` | Planning | Critical | **Indirect** — magnetics signals need correct identifier schemas (probe type, grid type) for IMAS mapping. T1 (identifier schemas on `fetch_imas_paths`) directly improves the mapping output that this plan's corrected signals feed into. |
| `jet-machine-description-completion.md` | Planning | Medium | **None** — incremental JET data enrichment (historical sensors, calibration epochs, PF JPF addressing). No tool overlap. |
| `signal-scanner-diagnostics.md` | Planning | Medium | **None** — progress streaming, worker health, MCP logs tools. Phases 3 log tools (list_logs, get_logs, tail_logs) already implemented in server.py. |

### Combining opportunities

**`imas-mcp-gap-closure.md` is the primary integration target.** The three tool
enhancements (T1, T2, T3) should be folded into Phase 1 of the gap closure plan
as sub-items 1.5, 1.6, 1.7:

- **1.5** — `fetch_imas_paths`: add `HAS_IDENTIFIER_SCHEMA` + `IMASNodeChange`
  OPTIONAL MATCHes, and `include_version_history` + `dd_version` params
- **1.6** — `search_imas_clusters`: make `query` optional, add `section_only`
  and `dd_version` params
- **1.7** — `VersionTool.get_dd_version_context`: new method for per-path
  change history

Phase 2 (Codex MCP wiring) gains one additional tool promotion:
`get_dd_version_context`.

**No combining opportunities with the other four plans.** Their concerns
(signal check reliability, JET-specific data, diagnostics) are orthogonal
to IMAS DD context enrichment.

---

## Implementation Order

```
Gap closure Phase 1 (amended)
    │
    ├── 1.1  DD version filter       (existing)
    ├── 1.2  Facility cross-refs     (existing)
    ├── 1.3  Version context         (existing, subsumed by T1+T3)
    ├── 1.4  Raw dict return         (existing)
    ├── 1.5  T1: fetch_imas_paths    identifier schemas + version history
    ├── 1.6  T2: search_imas_clusters IDS-scoped listing mode
    └── 1.7  T3: get_dd_version_context new tool
    │
Gap closure Phase 2 (amended)
    └── Promote get_dd_version_context to Codex MCP
    │
Gap closure Phase 6 (ids/tools.py consolidation)
    │   After this, shared tool data flows to pipeline automatically.
    │
This plan (pipeline prompt injection)
    │
    ├── E1  DD version threading     Parameter pass-through
    ├── E2  Cluster context          Calls T2 → prompt update
    ├── E3  Identifier schemas       Available from T1 → prompt updates (2 templates)
    ├── E4  Version change history   Calls T3 → prompt update
    ├── E5  Coordinate specs         Pipeline function or Phase 3 tool → prompt update
    └── E6  Cross-facility precedent Pipeline function → prompt update (optional)
```

---

## Testing

Each enrichment item should include:

1. **Unit test** — verify the shared tool enhancement returns expected shape
   against the test graph (which has IMASSemanticCluster, IdentifierSchema,
   IMASCoordinateSpec, DDVersion, IMASNodeChange nodes).
2. **Integration test** — run `generate_mapping()` for a test facility/IDS and
   verify the new context appears in the rendered prompts (mock the LLM call,
   assert prompt contains the new sections).
3. **MCP test** — verify the enhanced MCP tools return the new data fields in
   their responses (test both DD MCP server and Codex agents server).
4. **Quality validation** — compare mapping output with and without each enrichment
   for at least one facility/IDS pair. Document quality delta.
