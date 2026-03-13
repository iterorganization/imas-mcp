# IMAS MCP Feature Gap Closure Plan

**Priority: CRITICAL — directly affects discovery and mapping pipeline quality**

Close all capability gaps between the partner IMAS MCP server and the Codex MCP server.
Both servers share the same Neo4j knowledge graph. The Codex server must match or exceed
the IMAS MCP on every IMAS-specific capability.

**Non-negotiable constraint: every tool must be graph-backed.**

---

## Problem Statement

Three separate implementations currently serve IMAS data dictionary queries:

| Consumer | Location | Pattern |
|----------|----------|---------|
| **IMAS MCP server** | `tools/graph_search.py` → `Graph*Tool` classes | Shared via `Tools.register(mcp)` |
| **Codex MCP server** | `llm/search_tools.py` → `_search_imas()` etc. | Standalone — duplicates shared tools |
| **Mapping pipeline** | `ids/tools.py` → `search_imas_semantic()` etc. | Standalone — duplicates shared tools |

This fragmentation means improvements to one implementation don't benefit the others.
The `_search_imas()` function in `search_tools.py` has valuable enrichments (facility
cross-refs, version context, cluster search) that the shared tools lack. Meanwhile,
`ids/tools.py` has its own `search_imas_semantic()`, `check_imas_paths()`, and
`fetch_imas_fields()` that duplicate the shared `Graph*Tool` classes.

### Dual-Purpose Goal

The **shared common functionality** in `imas_codex/tools/` is consumed both by the
MCP servers and by the IMAS mapping pipeline (`imas_codex/ids/`). Every improvement
to the shared tools improves:

1. **IMAS MCP server** — the existing `Tools.register(mcp)` surface
2. **Codex MCP server** — once wired to delegate to shared tools
3. **Mapping pipeline** — `ids/tools.py` can delegate to shared tools for DD context

Creating parallel private functions in `search_tools.py` **breaks** this model.

## Architecture: Current → Target

### Current State (fragmented)

```
Codex MCP ──→ _search_imas()  [search_tools.py — own Cypher]
IMAS MCP  ──→ Graph*Tool      [tools/graph_search.py]
Mapping   ──→ ids/tools.py    [own Cypher, own Encoder]
```

### Target State (unified)

```
Codex MCP ──┐                              search_formatters.py
            │                              ─────────────────────
            ├──→ Graph*Tool classes     ←── format_imas_report()
            │    [tools/graph_search.py]    format_check_report()
IMAS MCP  ──┤    (enhanced: dd_version,     etc.
            │     facility xrefs,
            │     version context)
Mapping   ──┘
            └──→ .as_dicts() for pipeline consumption
```

- **`tools/graph_search.py`**: The single source of truth. Enhanced with features
  currently only in `search_tools.py` (facility cross-refs, version context, dd_version
  filtering). Returns typed result models for MCP, raw dicts for pipeline.
- **`llm/server.py`**: Thin `@self.mcp.tool()` wrappers that delegate to `Tools`,
  then format via `search_formatters.py`. The REPL already does this pattern.
- **`llm/search_formatters.py`**: Pure text formatting — unchanged.
- **`ids/tools.py`**: Gradually delegates DD context queries to shared tools.

### What the REPL Already Gets Right

The REPL functions (`search_imas()`, `fetch_imas()`, `list_imas()`, `check_imas()`,
`get_imas_overview()`) already delegate to the shared `Tools` singleton via
`_get_imas_tools()`. This is the correct pattern. The gap is that only `search_imas`
is promoted to an MCP tool — and even that uses a separate `_search_imas()` instead
of the shared tools
pattern. The Codex MCP wrappers call `Tools` methods, then format the structured
results into text strings via `search_formatters.py`.

---

## Gap Inventory

| # | Gap | Shared Tool | Codex MCP State | Action |
|---|-----|------------|-----------------|--------|
| 1 | Path validation | `GraphPathTool.check_imas_paths()` | REPL only | Promote to MCP tool |
| 2 | Full path docs | `GraphPathTool.fetch_imas_paths()` | REPL only | Promote to MCP tool |
| 3 | Path listing | `GraphListTool.list_imas_paths()` | REPL only | Promote to MCP tool |
| 4 | IDS overview | `GraphOverviewTool.get_imas_overview()` | REPL only | Promote to MCP tool |
| 5 | Identifier schemas | `GraphIdentifiersTool.get_imas_identifiers()` | Not exposed | Promote to MCP tool |
| 6 | Cluster search | `GraphClustersTool.search_imas_clusters()` | Not exposed | Promote to MCP tool |
| 7 | DD version scoping | Not in any tool | Not implemented | Add `dd_version` param to all Graph*Tool methods |
| 8 | Facility cross-refs | Only in `_search_imas()` | `search_imas` only | Lift into `GraphSearchTool` |
| 9 | Version context | Only in `_search_imas()` | `search_imas` only | Lift into `GraphSearchTool` |
| 10 | `search_imas` divergence | `GraphSearchTool.search_imas_paths()` | Uses separate `_search_imas()` | Unify onto shared tool |
| 11 | Cross-IDS relationships | N/A | Not implemented | New shared tool: `get_imas_path_context` |
| 12 | IDS structural analysis | N/A | Not implemented | New shared tool |
| 13 | IDS export | N/A | Not implemented | New shared tool |
| 14 | Physics domain export | N/A | Not implemented | New shared tool |
| 15 | Search quality | Good in shared tool | Weaker in `_search_imas` | Converge on shared tool fixes |
| 16 | `ids/tools.py` duplication | 3 functions shadow shared tools | N/A | Consolidate onto shared tools (Phase 6) |

**Dropped**: `explain_concept` — frontier LLMs already know physics concepts.
Adding a tool that generates explanatory text from graph data is AI fluff.

**Renamed**: `explore_relationships` → `get_imas_path_context` — communicates
that this is a graph-structural context tool, not a relationship browser.
Alternative names to consider: `trace_imas_connections`, `get_imas_neighborhood`.

---

## Discovery Pipeline Audit

An exhaustive audit of all graph-based context injection into LLM/VLM calls
across the `discovery/` pipeline confirms there are **no additional splits**
for IMAS DD graph access. The fragmentation is confined to three consumers
already identified in this plan.

### What the discovery pipeline uses (facility-specific graph context only)

| Pipeline | Graph Context Injected into LLM | IMAS DD overlap? |
|----------|-------------------------------|-----------------|
| **Signal enrichment** (`discovery/signals/`) | Wiki chunks (path-match + vector), code chunks (vector), tree context (SignalNode traversal), TDI source code, epoch context, code refs (graph traversal) | **No** — all facility-specific nodes |
| **Path scoring** (`discovery/paths/`) | Calibration examples from `FacilityPath` nodes | **No** |
| **Code scoring** (`discovery/code/`) | Calibration examples from `CodeFile` nodes | **No** |
| **Wiki scoring** (`discovery/wiki/`) | Calibration examples from `WikiPage`/`Document` nodes | **No** |
| **Static enrichment** (`discovery/static/`) | `SignalNode` parent/sibling context | **No** |
| **Image captioning** (`discovery/base/`) | No graph context injected | **No** |
| **Wiki entity linking** (`discovery/wiki/pipeline.py`) | `MERGE (c)-[:MENTIONS_IMAS]->(ip:IMASNode)` — **write only**, no read context | Write-only |
| **MDSplus unit attachment** (`discovery/mdsplus/graph_ops.py`) | `MERGE (unit:Unit {id: ...})`, `MERGE (n)-[:HAS_UNIT]->(unit)` — **write only** | Write-only |

### Where the IMAS DD duplication actually lives

The only consumers that read `IMASNode`/`Unit`/`IMASSemanticCluster`/`IMASCoordinateSpec`
for LLM context injection are:

1. **Mapping pipeline** (`ids/tools.py` → `ids/mapping.py`) — 3 functions duplicate shared tools:

   | `ids/tools.py` function | Shadows shared tool | Cypher overlap |
   |------------------------|-------------------|----------------|
   | `fetch_imas_subtree()` | `GraphListTool.list_imas_paths()` | `MATCH (p:IMASNode) WHERE p.ids = $ids_name` + `HAS_UNIT` join |
   | `fetch_imas_fields()` | `GraphPathTool.fetch_imas_paths()` | `MATCH (p:IMASNode {id: pid})` + `HAS_UNIT` + `IN_CLUSTER` + `HAS_COORDINATE` |
   | `search_imas_semantic()` | `GraphSearchTool.search_imas_paths()` | `db.index.vector.queryNodes('imas_node_embedding', ...)` + `DEPRECATED_IN` filter |

2. **Codex MCP standalone** (`llm/search_tools.py`) — `_search_imas()` and helpers duplicate
   `GraphSearchTool.search_imas_paths()` plus add `_get_facility_crossrefs()` and
   `_get_version_context()` Cypher that should be lifted into the shared tool.

Functions in `ids/tools.py` that have **no shared tool equivalent** and should stay:
- `query_signal_sources()` — facility-specific `SignalSource` traversal
- `fetch_source_code_refs()` — `SignalSource→FacilitySignal→SignalNode→CodeChunk` traversal
- `search_existing_mappings()` — `IMASMapping` + `POPULATES` + `MAPS_TO_IMAS` traversal
- `get_sign_flip_paths()` — COCOS sign flip data from `imas_codex.cocos.transforms`
- `analyze_units()` — pint-based unit compatibility analysis
- `check_imas_paths()` — **could** delegate to `GraphPathTool.check_imas_paths()` but
  has a slightly different return shape (flat dicts vs `CheckPathsResult`)

---

## Phase 1: Enhance Shared Graph Tools

**Location:** `imas_codex/tools/graph_search.py`

**Goal:** Add missing capabilities to the shared `Graph*Tool` classes so that
every consumer (IMAS MCP, Codex MCP, mapping pipeline) benefits.

### 1.1 DD Version Filter

Add `dd_version: int | None = None` parameter to every `@mcp_tool`-decorated method
in all `Graph*Tool` classes. Generate a Cypher WHERE fragment:

```python
def _dd_version_clause(alias: str = "p", dd_version: int | None = None) -> str:
    """Returns a Cypher WHERE fragment for DD major version filtering.

    When dd_version is None, returns empty string (no filter).
    When specified, excludes paths deprecated before that major version
    and paths introduced after it.
    """
    if dd_version is None:
        return ""
    return (
        f"AND NOT EXISTS {{ ({alias})-[:DEPRECATED_IN]->(dep:DDVersion) "
        f"WHERE dep.major <= {dd_version} }} "
        f"AND ("
        f"  NOT EXISTS {{ ({alias})-[:INTRODUCED_IN]->(:DDVersion) }} "
        f"  OR EXISTS {{ ({alias})-[:INTRODUCED_IN]->(intro:DDVersion) "
        f"  WHERE intro.major <= {dd_version} }}"
        f")"
    )
```

Thread into: `GraphSearchTool.search_imas_paths`, `GraphPathTool.check_imas_paths`,
`GraphPathTool.fetch_imas_paths`, `GraphListTool.list_imas_paths`,
`GraphOverviewTool.get_imas_overview`, `GraphClustersTool.search_imas_clusters`,
`GraphIdentifiersTool.get_imas_identifiers`.

The IMAS MCP server's tool descriptions will expose this parameter to LLM consumers.
The Codex MCP will pass a default value. The mapping pipeline can pass `None` to get
all versions.

### 1.2 Facility Cross-Reference Enrichment in GraphSearchTool

Currently only in `_search_imas()` → `_get_facility_crossrefs()`. Move this
enrichment into `GraphSearchTool.search_imas_paths()` as an optional parameter:

```python
async def search_imas_paths(
    self,
    query: str,
    ids_filter: str | list[str] | None = None,
    max_results: int = 50,
    search_mode: str | SearchMode = "auto",
    response_profile: str = "standard",
    facility: str | None = None,          # NEW
    dd_version: int | None = None,        # NEW
    ctx: Context | None = None,
) -> SearchPathsResult:
```

When `facility` is provided, each `SearchHit` gains a `facility_xrefs` dict with
`facility_signals`, `wiki_mentions`, `code_files` from the same Cypher pattern
that `_get_facility_crossrefs()` uses. The `SearchPathsResult` model will need
a new optional field or the xrefs can be attached per-hit.

### 1.3 Version Context Enrichment in GraphSearchTool

Add `include_version_context: bool = False` parameter. When True, query
`IMASNodeChange` nodes for each result path and attach to hits:

```python
if include_version_context:
    version_data = self._gc.query("""
        UNWIND $path_ids AS pid
        MATCH (p:IMASNode {id: pid})
        OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
        WHERE change.semantic_change_type IN
              ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
        RETURN p.id AS id, count(change) AS change_count,
               collect({version: change.version,
                        type: change.semantic_change_type,
                        summary: change.summary})[..5] AS notable_changes
    """, path_ids=sorted_ids)
```

### 1.4 Raw Dict Return Mode for Pipeline Consumption

The mapping pipeline (`ids/tools.py`) needs raw dicts, not Pydantic result models.
Add `as_dicts()` convenience methods to result models, or provide a `raw=True`
parameter that returns `list[dict]` instead of typed results.

This allows `ids/tools.py` functions like `search_imas_semantic()` to delegate:

```python
# ids/tools.py — AFTER consolidation
def search_imas_semantic(query, ids_name=None, *, gc=None, k=20):
    tool = GraphSearchTool(gc or GraphClient())
    result = _run_async(tool.search_imas_paths(
        query=query, ids_filter=ids_name, max_results=k
    ))
    return [{"id": h.path, "documentation": h.documentation, ...} for h in result.hits]
```

---

## Phase 2: Wire Codex MCP to Shared Tools

**Location:** `imas_codex/llm/server.py`

**Goal:** Promote REPL-only IMAS tools to first-class MCP tools. All delegate to
the shared `Tools` singleton via `_get_imas_tools()`, then format results via
`search_formatters.py`.

### 2.1 Promote 6 IMAS Tools to MCP Surface

The REPL already has `fetch_imas`, `list_imas`, `check_imas`, `get_imas_overview`
that correctly delegate to `Tools`. Add these as `@self.mcp.tool()` wrappers:

```python
@self.mcp.tool()
def check_imas(
    paths: str,
    ids: str | None = None,
    dd_version: int | None = None,
) -> str:
    """Validate IMAS paths against the Data Dictionary graph.

    Checks existence, returns data type + units, detects renamed/deprecated paths.
    Use this before mapping signals to verify target paths exist.

    Args:
        paths: Space-delimited IMAS paths (e.g. "equilibrium/time_slice/profiles_1d/psi")
        ids: Optional IDS prefix to prepend
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.check_imas_paths(paths=paths, ids=ids, dd_version=dd_version))
    return format_check_report(result)

@self.mcp.tool()
def fetch_imas(
    paths: str,
    ids: str | None = None,
    dd_version: int | None = None,
) -> str:
    """Get full documentation for IMAS paths including units, coordinates, cluster membership.

    Args:
        paths: Space-delimited IMAS paths
        ids: Optional IDS prefix to prepend
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.fetch_imas_paths(paths=paths, ids=ids, dd_version=dd_version))
    return format_fetch_report(result)

@self.mcp.tool()
def list_imas(
    paths: str,
    leaf_only: bool = True,
    max_paths: int = 100,
    dd_version: int | None = None,
) -> str:
    """List data paths within an IMAS IDS or subtree.

    Args:
        paths: Space-separated IDS names or path prefixes (e.g. "equilibrium" or "equilibrium/time_slice")
        leaf_only: Only return leaf data fields (default True)
        max_paths: Max paths to return (default 100)
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.list_imas_paths(
        paths=paths, leaf_only=leaf_only, max_paths=max_paths, dd_version=dd_version
    ))
    return format_list_report(result)

@self.mcp.tool()
def get_imas_overview(
    query: str | None = None,
    dd_version: int | None = None,
) -> str:
    """Get overview of available IMAS IDS with statistics and physics domains.

    Args:
        query: Optional keyword filter (e.g. "magnetics" or "equilibrium")
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.get_imas_overview(query=query, dd_version=dd_version))
    return format_overview_report(result)

@self.mcp.tool()
def get_imas_identifiers(
    query: str | None = None,
    dd_version: int | None = None,
) -> str:
    """Browse IMAS identifier/enumeration schemas (coordinate systems, grid types, probe types).

    Args:
        query: Optional filter keyword
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.get_imas_identifiers(query=query, dd_version=dd_version))
    return format_identifiers_report(result)

@self.mcp.tool()
def search_imas_clusters(
    query: str,
    scope: str | None = None,
    ids_filter: str | None = None,
    dd_version: int | None = None,
) -> str:
    """Search semantic clusters of related IMAS data paths.

    Finds groups of physics-related paths across IDS boundaries.
    Can search by path (find its clusters) or by text (semantic search).

    Args:
        query: Natural language description or exact IMAS path
        scope: Cluster scope filter — "global", "domain", or "ids"
        ids_filter: Limit to clusters containing paths from specific IDS
        dd_version: DD major version scope (None=current)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.search_imas_clusters(
        query=query, scope=scope, ids_filter=ids_filter, dd_version=dd_version
    ))
    return format_cluster_report(result)
```

### 2.2 Refactor `search_imas` to Delegate to Shared Tool

The existing `search_imas` MCP tool calls `_search_imas()` from `search_tools.py` —
a parallel implementation of what `GraphSearchTool.search_imas_paths()` already does.

Refactor to delegate to the shared tool (now enhanced with facility xrefs and
version context from Phase 1):

```python
@self.mcp.tool()
def search_imas(
    query: str,
    ids_filter: str | None = None,
    facility: str | None = None,
    include_version_context: bool = False,
    dd_version: int | None = None,
    k: int = 20,
) -> str:
    """Search IMAS Data Dictionary with cross-domain enrichment.

    Performs hybrid search (vector + keyword) across IMAS path and cluster
    embeddings, enriched with cluster membership, units, coordinates,
    and optional facility cross-references and version history.

    Args:
        query: Natural language search text (e.g. "electron temperature")
        ids_filter: Optional IDS name filter (e.g. "core_profiles")
        facility: Optional facility for cross-references (e.g. "tcv")
        include_version_context: Include DD version change history
        dd_version: DD major version scope (None=current)
        k: Number of results (default 20)
    """
    tools = _get_imas_tools()
    result = _run_async(tools.search_imas_paths(
        query=query,
        ids_filter=ids_filter,
        max_results=k,
        facility=facility,
        include_version_context=include_version_context,
        dd_version=dd_version,
    ))
    # Also get cluster results for the combined report
    cluster_result = _run_async(tools.search_imas_clusters(query=query, dd_version=dd_version))
    return format_imas_report(result, cluster_result)
```

This replaces the ~120 lines of `_search_imas()` + `_vector_search_imas_paths()` +
`_text_search_imas_paths_by_query()` + `_enrich_imas_paths()` in `search_tools.py`.
Those functions can be removed once the shared tool handles all their cases.

### 2.3 Formatter Functions

**File: `imas_codex/llm/search_formatters.py`**

Add new formatters that convert typed result models into text strings:

- `format_check_report(CheckPathsResult) → str`
- `format_fetch_report(FetchPathsResult) → str`
- `format_list_report(ListPathsResult) → str`
- `format_overview_report(GetOverviewResult) → str`
- `format_identifiers_report(GetIdentifiersResult) → str`
- `format_cluster_report(dict) → str`

Update the existing `format_imas_report()` to accept `SearchPathsResult` and
cluster results (typed) instead of raw dicts.

---

## Phase 3: New Shared Tool — `get_imas_path_context`

**Location:** `imas_codex/tools/graph_search.py` — new `GraphPathContextTool` class

**Goal:** Given an IMAS path, traverse graph relationships to discover structural
context across IDS boundaries. This is a Codex-exclusive capability — no equivalent
in the old file-backed IMAS MCP server.

### Why This Tool Is Useful

The mapping pipeline needs to understand how a target IMAS path relates to other
paths in the data dictionary. When mapping signals to `pf_active/coil/element/geometry/outline/r`,
the pipeline benefits from knowing:
- Sibling paths in the same semantic cluster (other geometry paths)
- Paths sharing the same coordinate spec (same dimensionality)
- Paths with the same unit (metres → other length measurements)
- Paths using the same identifier schema (same enumeration)

### Graph Edges to Traverse

| Relationship | Meaning | Cross-IDS? |
|-------------|---------|------------|
| `IN_CLUSTER` → `IMASSemanticCluster` | Shared physics concept | Yes |
| `HAS_COORDINATE` → `IMASCoordinateSpec` | Shared coordinate axis | Yes |
| `HAS_UNIT` → `Unit` | Same measurement unit | Possible |
| `HAS_IDENTIFIER_SCHEMA` → `IdentifierSchema` | Shared enumeration | Yes |
| `MAPS_TO_IMAS` via `SignalSource` | Facility signal mappings | N/A |

### Implementation

```python
class GraphPathContextTool:
    """Graph-backed path context for cross-IDS relationship discovery."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @mcp_tool(
        "Get structural context for an IMAS path via graph traversal. "
        "Discovers sibling paths via shared clusters, coordinates, units, "
        "and identifier schemas across IDS boundaries. "
        "path (required): Exact IMAS path (e.g. 'equilibrium/time_slice/profiles_1d/psi'). "
        "relationship_types: Filter to specific types — 'cluster', 'coordinate', 'unit', 'identifier', or 'all' (default)."
    )
    async def get_imas_path_context(
        self,
        path: str,
        relationship_types: str = "all",
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Discover cross-IDS relationships for an IMAS path."""
        dd_where = _dd_version_clause("sibling", dd_version)
        sections = {}

        # Cluster siblings — paths in same cluster but different IDS
        if relationship_types in ("all", "cluster"):
            cluster_siblings = self._gc.query(f"""
                MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
                      <-[:IN_CLUSTER]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_where}
                RETURN cl.label AS cluster, sibling.id AS path,
                       sibling.ids AS ids, sibling.documentation AS doc
                ORDER BY cl.label, sibling.ids
            """, path=path)
            if cluster_siblings:
                sections["cluster_siblings"] = cluster_siblings

        # Coordinate partners — paths sharing coordinate spec
        if relationship_types in ("all", "coordinate"):
            coord_partners = self._gc.query(f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
                      <-[:HAS_COORDINATE]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_where}
                RETURN coord.id AS coordinate, sibling.id AS path,
                       sibling.ids AS ids, sibling.data_type AS data_type
                ORDER BY coord.id, sibling.ids
            """, path=path)
            if coord_partners:
                sections["coordinate_partners"] = coord_partners

        # Unit companions — paths with same unit in same physics domain
        if relationship_types in ("all", "unit"):
            unit_companions = self._gc.query(f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_UNIT]->(u:Unit)
                      <-[:HAS_UNIT]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids
                  AND sibling.physics_domain = p.physics_domain {dd_where}
                RETURN u.id AS unit, sibling.id AS path,
                       sibling.ids AS ids, sibling.documentation AS doc
                ORDER BY u.id, sibling.ids
                LIMIT 30
            """, path=path)
            if unit_companions:
                sections["unit_companions"] = unit_companions

        # Identifier schema links
        if relationship_types in ("all", "identifier"):
            ident_links = self._gc.query(f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
                      <-[:HAS_IDENTIFIER_SCHEMA]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_where}
                RETURN s.name AS schema, sibling.id AS path,
                       sibling.ids AS ids
                ORDER BY s.name
            """, path=path)
            if ident_links:
                sections["identifier_links"] = ident_links

        return {
            "path": path,
            "relationship_types": relationship_types,
            "sections": sections,
            "total_connections": sum(len(v) for v in sections.values()),
        }
```

Register in `Tools.__init__`:
```python
self.path_context_tool = GraphPathContextTool(graph_client)
```

Add to `_tool_instances` list, add delegation method to `Tools`.

Codex MCP wrapper in `server.py`:
```python
@self.mcp.tool()
def get_imas_path_context(path: str, relationship_types: str = "all",
                           dd_version: int | None = None) -> str:
    tools = _get_imas_tools()
    result = _run_async(tools.get_imas_path_context(
        path=path, relationship_types=relationship_types, dd_version=dd_version
    ))
    return format_path_context_report(result)
```

---

## Phase 4: Additional Shared Tools

### 4.1 `analyze_imas_structure` — IDS Structural Analysis

**Location:** new method on a `GraphStructureTool` class in `graph_search.py`

Pure-Cypher structural analysis: depth metrics, leaf/structure ratio, array patterns,
physics domain distribution, coordinate usage, COCOS-labeled fields.

```python
@mcp_tool("Analyze the hierarchical structure of an IMAS IDS. ...")
async def analyze_imas_structure(self, ids_name: str, dd_version=None, ctx=None):
    # Basic metrics: total nodes, structures, leaves, max depth
    # Physics domain distribution
    # Array structures with coordinates
    # Data type distribution
    # COCOS-labeled fields
    ...
```

### 4.2 `export_imas_ids` — Full IDS Export

Single enriched Cypher query returning all paths with documentation, units, types:

```python
@mcp_tool("Export full IDS structure with documentation, units, and types. ...")
async def export_imas_ids(self, ids_name: str, leaf_only=False, dd_version=None, ctx=None):
    ...
```

### 4.3 `export_imas_domain` — Physics Domain Export

All IMAS paths in a physics domain, grouped by IDS:

```python
@mcp_tool("Export all IMAS paths in a physics domain, grouped by IDS. ...")
async def export_imas_domain(self, domain: str, ids_filter=None, dd_version=None, ctx=None):
    ...
```

All three tools follow the same pattern: implemented as methods on shared tool
classes, registered via `Tools`, exposed in both IMAS MCP and Codex MCP.

---

## Phase 5: Search Quality Improvement

**Goal:** Bring hybrid search relevance up to match focused physics queries.

### Problem

For "equilibrium boundary separatrix":
- **IMAS MCP** returned: `boundary/psi_norm`, `boundary/outline/r`, `boundary/outline/z`
- **Codex MCP** returned: `profiles_1d/chi_squared`, `profiles_1d/q`

### Root Causes and Fixes

All fixes go into `GraphSearchTool.search_imas_paths()` in `graph_search.py`:

**5.1 Path segment boost** — after merging vector + text scores, boost paths
whose ID segments match query words:

```python
query_words = [w.lower() for w in query.split() if len(w) > 2]
for pid in sorted_ids:
    segments = pid.lower().split("/")
    match_count = sum(1 for w in query_words if any(w in seg for seg in segments))
    if match_count > 0:
        scores[pid] += 0.03 * match_count
```

**5.2 DD version filter** — `dd_version` parameter from Phase 1 filters out
DD 3.x duplicates that dilute top-k results.

**5.3 Generic metadata path filter** — already exists as `_is_generic_metadata_path()`.
Verify it's applied consistently in both vector and text search branches.

### Regression Tests

| Query | Expected top-5 must contain |
|-------|---------------------------|
| "equilibrium boundary separatrix" | `boundary/outline/r` or `boundary/psi_norm` |
| "electron temperature" | `core_profiles/profiles_1d/electrons/temperature` |
| "plasma current" | `equilibrium/time_slice/global_quantities/ip` |
| "safety factor q profile" | `equilibrium/time_slice/profiles_1d/q` |

---

## Phase 6: Consolidate Mapping Pipeline

**Location:** `imas_codex/ids/tools.py`

**Goal:** Replace duplicate implementations with delegation to shared tools.
This is the primary "split pipeline" identified during the discovery pipeline audit.

### Current Duplication (confirmed by audit)

The mapping pipeline (`ids/mapping.py`) calls `ids/tools.py` functions that contain
their own Cypher queries shadowing the shared `Graph*Tool` classes:

| `ids/tools.py` function | Cypher pattern | Shadows shared tool |
|------------------------|---------------|-------------------|
| `fetch_imas_subtree()` L24 | `MATCH (p:IMASNode) WHERE p.ids = $ids_name` + `HAS_UNIT` | `GraphListTool.list_imas_paths()` |
| `fetch_imas_fields()` L72 | `MATCH (p:IMASNode {id: pid})` + `HAS_UNIT` + `IN_CLUSTER` + `HAS_COORDINATE` | `GraphPathTool.fetch_imas_paths()` |
| `search_imas_semantic()` L108 | `db.index.vector.queryNodes('imas_node_embedding', ...)` + `DEPRECATED_IN` filter | `GraphSearchTool.search_imas_paths()` |
| `check_imas_paths()` L213 | `MATCH (p:IMASNode {id: $path})` + `RENAMED_TO` detection | `GraphPathTool.check_imas_paths()` |

These functions are consumed by `gather_context()` → LLM prompts (section_assignment,
signal_mapping, assembly). When the shared tools improve (e.g., dd_version filtering,
better enrichment), the mapping pipeline won't benefit unless consolidated.

### Approach

Refactor each `ids/tools.py` function to delegate to the shared tool and
extract raw dicts from the result. The shared tools gain `as_dicts()` support
from Phase 1.4:

```python
# ids/tools.py — AFTER consolidation
from imas_codex.tools.graph_search import GraphSearchTool

def search_imas_semantic(query, ids_name=None, *, gc=None, k=20):
    gc = gc or GraphClient()
    tool = GraphSearchTool(gc)
    result = _run_async(tool.search_imas_paths(query=query, ids_filter=ids_name, max_results=k))
    return [
        {"id": h.path, "documentation": h.documentation, "data_type": h.data_type,
         "units": h.units, "score": h.score}
        for h in result.hits
    ]
```

Functions that have NO shared equivalent and stay in `ids/tools.py`:
- `query_signal_sources()` — facility-specific `SignalSource` + `MAPS_TO_IMAS` traversal
- `fetch_source_code_refs()` — `SignalSource→FacilitySignal→SignalNode→CodeChunk`
- `search_existing_mappings()` — `IMASMapping` + `POPULATES` + `USES_SIGNAL_SOURCE`
- `get_sign_flip_paths()` — COCOS-specific (delegates to `cocos.transforms`)
- `analyze_units()` — pint-based unit compatibility analysis
- `analyze_units()` — pint-based unit analysis

---

## Phase 7: REPL & Registry Update

### REPL Functions

The REPL functions already delegate to `Tools` — they just need the new
`dd_version` parameter threaded through:

```python
def search_imas(query_text, ids_filter=None, max_results=10, dd_version=None):
    tools = _get_imas_tools()
    result = _run_async(tools.search_imas_paths(
        query=query_text, ids_filter=ids_filter,
        max_results=max_results, dd_version=dd_version
    ))
    ...
```

Add new REPL functions for the new tools:
- `get_imas_path_context(path, relationship_types="all")`
- `analyze_imas_structure(ids_name)`
- `export_imas_ids(ids_name, leaf_only=False)`
- `export_imas_domain(domain)`

Update `_registry` dict under `IMAS_DD`.

### Cleanup

Once `search_imas` MCP tool delegates to shared `Tools` (Phase 2.2), remove
the standalone `_search_imas()` implementation and its helper functions from
`search_tools.py`:
- `_search_imas()`
- `_vector_search_imas_paths()`
- `_text_search_imas_paths_by_query()`
- `_enrich_imas_paths()`
- `_vector_search_clusters()`
- `_get_facility_crossrefs()`
- `_get_version_context()`
- `_is_generic_metadata_path()` (move to shared utils if needed)

---

## Implementation Order

```
Phase 1  (4 tasks)   Enhance shared Graph*Tool classes
   │                 1.1 DD version filter helper + parameter threading
   │                 1.2 Facility cross-ref enrichment in GraphSearchTool
   │                 1.3 Version context enrichment in GraphSearchTool
   │                 1.4 Raw dict return mode for pipeline
   ▼
Phase 2  (3 tasks)   Wire Codex MCP to shared tools
   │                 2.1 Promote 6 IMAS tools to MCP surface
   │                 2.2 Refactor search_imas to delegate to shared tool
   │                 2.3 Add formatter functions
   ▼
Phase 3  (1 task)    New shared tool: get_imas_path_context
   ▼
Phase 4  (3 tasks)   Additional shared tools
   │                 4.1 analyze_imas_structure
   │                 4.2 export_imas_ids
   │                 4.3 export_imas_domain
   ▼
Phase 5  (3 tasks)   Search quality
   │                 5.1 Path segment boost
   │                 5.2 DD version filter (benefits from Phase 1.1)
   │                 5.3 Generic metadata filter audit
   ▼
Phase 6  (1 task)    Consolidate mapping pipeline
   ▼
Phase 7  (2 tasks)   REPL update + cleanup
                     7.1 Thread dd_version, add new REPL functions
                     7.2 Remove standalone _search_imas() and helpers
```

### Dependencies

- Phase 2 depends on Phase 1 (shared tools must be enhanced before Codex can delegate)
- Phase 3 and 4 are independent of Phase 2
- Phase 5 depends on Phase 1.1 (dd_version)
- Phase 6 depends on Phase 1.4 (raw dict mode)
- Phase 7 depends on Phase 2 (delegation must be in place before removing old code)

---

## Files Modified

| File | Phases | Changes |
|------|--------|---------|
| `imas_codex/tools/graph_search.py` | 1, 3, 4, 5 | Add dd_version, facility xrefs, version context, path context tool, structure/export tools, search quality |
| `imas_codex/tools/__init__.py` | 1, 3, 4 | Wire new tool instances into Tools class |
| `imas_codex/llm/server.py` | 2, 7 | Register 7 new MCP tools, refactor search_imas, update REPL |
| `imas_codex/llm/search_formatters.py` | 2 | Add 7 new formatter functions |
| `imas_codex/llm/search_tools.py` | 7 | Remove standalone IMAS functions (cleanup) |
| `imas_codex/ids/tools.py` | 6 | Delegate DD context queries to shared tools |
| `imas_codex/models/result_models.py` | 1 | Add facility_xrefs, version_context fields to SearchPathsResult |
| `tests/tools/test_graph_search.py` | 1, 3, 4, 5 | Test shared tool enhancements |
| `tests/llm/test_imas_tools.py` | 2 | Test Codex MCP tool wrappers + formatters |

---

## Codex MCP Tool Inventory (Post-Implementation)

| Tool | Category | Phase | Backed by |
|------|----------|-------|-----------|
| `search_signals` | Facility | Existing | search_tools.py |
| `signal_analytics` | Facility | Existing | search_tools.py |
| `search_docs` | Facility | Existing | search_tools.py |
| `search_code` | Facility | Existing | search_tools.py |
| `fetch` | Retrieval | Existing | search_tools.py |
| `search_imas` | IMAS DD | Refactored (Phase 2) | **Tools → GraphSearchTool** |
| `check_imas` | IMAS DD | Phase 2 | **Tools → GraphPathTool** |
| `fetch_imas` | IMAS DD | Phase 2 | **Tools → GraphPathTool** |
| `list_imas` | IMAS DD | Phase 2 | **Tools → GraphListTool** |
| `get_imas_overview` | IMAS DD | Phase 2 | **Tools → GraphOverviewTool** |
| `get_imas_identifiers` | IMAS DD | Phase 2 | **Tools → GraphIdentifiersTool** |
| `search_imas_clusters` | IMAS DD | Phase 2 | **Tools → GraphClustersTool** |
| `get_imas_path_context` | IMAS DD | Phase 3 | **Tools → GraphPathContextTool** |
| `analyze_imas_structure` | IMAS DD | Phase 4 | **Tools → GraphStructureTool** |
| `export_imas_ids` | IMAS DD | Phase 4 | **Tools → GraphExportTool** |
| `export_imas_domain` | IMAS DD | Phase 4 | **Tools → GraphExportTool** |
| `get_graph_schema` | Graph | Existing | graph/schema.py |
| `add_to_graph` | Graph | Existing | server.py |
| `update_facility_infrastructure` | Infra | Existing | discovery/ |
| `get_facility_infrastructure` | Infra | Existing | discovery/ |
| `add_exploration_note` | Infra | Existing | discovery/ |
| `update_facility_config` | Infra | Existing | discovery/ |
| `get_discovery_context` | Infra | Existing | discovery/ |
| `list_logs` | Logs | Existing | server.py |
| `get_logs` | Logs | Existing | server.py |
| `tail_logs` | Logs | Existing | server.py |
| `python` | REPL | Existing | server.py |

**Total: 26 tools (10 new/refactored IMAS DD tools)**

All IMAS DD tools backed by shared `Graph*Tool` classes in `imas_codex/tools/graph_search.py`.

---

## Validation Criteria

1. **Shared-first:** Every IMAS tool in the Codex MCP delegates to `Tools` → `Graph*Tool`
2. **Dual benefit:** Improvements to shared tools are visible in IMAS MCP server too
3. **DD version scoping:** `dd_version` parameter available on all IMAS tools
4. **No duplication:** `_search_imas()` and helpers removed from `search_tools.py`
5. **Pipeline benefit:** `ids/tools.py` delegates DD queries to shared tools
6. **Search quality:** "equilibrium boundary separatrix" returns boundary paths in top 5
7. **No AI fluff:** No `explain_concept` tool
8. **Practical naming:** `get_imas_path_context` communicates graph-structural purpose
9. **All tests pass:** No regressions in existing test suite
