# IMAS MCP Feature Gap Closure Plan

**Priority: CRITICAL — directly affects discovery and mapping pipeline quality**

Close all capability gaps between the partner IMAS MCP server and the Codex MCP server.
Both servers share the same Neo4j knowledge graph. The Codex MCP must match or exceed
the IMAS MCP on every IMAS-specific capability.

**Non-negotiable constraint: every tool must be graph-backed.** No delegation to the
old `Tools` class or its `Graph*Tool` wrappers. All new IMAS tools are implemented as
`_` prefixed functions in `search_tools.py` using `GraphClient.query()` directly,
following the established Codex MCP pattern.

---

## Architecture Context

### Codex MCP Pattern (the target pattern — all new tools follow this)

```
server.py                          search_tools.py
─────────────────                  ──────────────────────────────
@self.mcp.tool()                   def _check_imas(
def check_imas(paths, ...) -> str:     paths, *, gc=None, encoder=None,
    return _check_imas(paths, ...)  ) -> str:
                                       gc = gc or GraphClient()
                                       # Direct Cypher queries
                                       results = gc.query("MATCH ...")
                                       # Format into text report
                                       return format_check_report(results)
```

- **`search_tools.py`**: Backend functions prefixed `_`, take `gc: GraphClient | None` and `encoder: Encoder | None` kwargs (lazy-instantiated), return formatted `str`.
- **`server.py`**: Thin `@self.mcp.tool()` wrappers with docstrings/type hints — delegates to `_` functions.
- **`search_formatters.py`**: Pure formatting functions (`format_*_report`) that convert structured dicts into text reports.
- **Graph access**: Direct `gc.query(cypher, **params)` — no ORM, no intermediate class hierarchy.

### IMAS MCP Pattern (the source — extract good Cypher, do NOT re-use the classes)

```
server.py (imas mcp)               tools/graph_search.py
─────────────────                   ──────────────────────────────
Tools.register(mcp)                 class GraphPathTool:
  → auto-registers @mcp_tool           async def check_imas_paths(...):
    decorated methods                       row = self._gc.query("MATCH ...")
                                            return CheckPathsResult(...)
```

The IMAS MCP's `Graph*Tool` classes contain **excellent Cypher queries** for enrichment,
validation, and traversal. These queries should be copied and adapted for the Codex MCP
pattern — but the class wrappers, result models, and `@mcp_tool` decorator are left behind.
The Codex MCP returns formatted text strings, not Pydantic result objects.

### What NOT to Do

- ❌ `_get_imas_tools()` → `Tools` → `GraphPathTool.check_imas_paths()` — class delegation
- ❌ `_run_async(tools.fetch_imas_paths(...))` — async bridge to old tool classes
- ❌ Import from `imas_codex.tools` in `search_tools.py` — wrong dependency direction
- ✅ Copy the Cypher from `GraphPathTool.check_imas_paths()` into a new `_check_imas()` function
- ✅ Direct `gc.query()` in `search_tools.py`, return `str` via formatter

---

## Gap Inventory

| # | Gap | IMAS MCP Source | Codex MCP State | Approach |
|---|-----|----------------|-----------------|----------|
| 1 | Path validation | `GraphPathTool.check_imas_paths()` — MATCH + RENAMED_TO detection | Not exposed | New `_check_imas()` with same Cypher pattern |
| 2 | Full path docs | `GraphPathTool.fetch_imas_paths()` — HAS_UNIT, IN_CLUSTER, HAS_COORDINATE joins | Not exposed | New `_fetch_imas()` with enrichment Cypher |
| 3 | Path listing | `GraphListTool.list_imas_paths()` — IDS subtree enumeration | Not exposed | New `_list_imas()` with IMASNode prefix queries |
| 4 | IDS overview | `GraphOverviewTool.get_imas_overview()` — IDS stats, DDVersion, domains | Not exposed | New `_get_imas_overview()` with aggregation Cypher |
| 5 | Identifier schemas | `GraphIdentifiersTool.get_imas_identifiers()` — IdentifierSchema queries | Not exposed | New `_get_imas_identifiers()` with IdentifierSchema queries |
| 6 | Cluster search | `GraphClustersTool.search_imas_clusters()` — cluster_description_embedding vector search | Not exposed | New `_search_imas_clusters()` with vector + path lookup |
| 7 | Cross-IDS relationships | N/A in old tools | Not implemented | New `_explore_imas_relationships()` — Codex advantage via multi-hop graph traversal |
| 8 | Physics concept explanation | N/A in old tools | Not implemented | New `_explain_imas_concept()` — multi-index vector search synthesis |
| 9 | IDS structural analysis | Old `graph_analyzer.py` (NetworkX) | Not implemented | New `_analyze_imas_structure()` — pure Cypher aggregation |
| 10 | IDS export | Composition | Not implemented | New `_export_imas_ids()` — single enriched Cypher |
| 11 | Physics domain export | N/A | Not implemented | New `_export_imas_domain()` — domain-grouped Cypher |
| 12 | DD version scoping | Not implemented anywhere | Not implemented | `dd_version` kwarg on all IMAS tools |
| 13 | Search quality | IMAS MCP returns better results for focused queries | search_imas exists but weaker ranking | Path segment boost, cluster cap, version filter |

---

## Phase 1: Core IMAS Tools — Graph-Backed Implementations

**Goal:** Implement `check_imas`, `fetch_imas`, `list_imas`, `get_imas_overview`, `get_imas_identifiers`, `search_imas_clusters` as graph-backed functions in `search_tools.py`, register in `server.py`.

All implementations include `dd_version: int | None = 4` from Day 1.

### 1.1 DD Version Filter Helper

**File: `imas_codex/llm/search_tools.py`** — add near other helpers

```python
def _dd_version_clause(
    alias: str = "p",
    dd_version: int | None = 4,
) -> tuple[str, dict[str, Any]]:
    """Generate Cypher WHERE fragment for DD major version scoping.

    Returns (clause_str, params_dict). The clause should be AND-ed into
    an existing WHERE. Returns ("", {}) when dd_version is None.
    """
    if dd_version is None:
        return "", {}

    # Exclude paths deprecated before this major version,
    # include paths with no INTRODUCED_IN or introduced at/before this major version.
    clause = (
        f"NOT EXISTS {{ ({alias})-[:DEPRECATED_IN]->(dep:DDVersion) "
        f"WHERE dep.major < $dd_major }} "
        f"AND ("
        f"NOT EXISTS {{ ({alias})-[:INTRODUCED_IN]->(:DDVersion) }} "
        f"OR EXISTS {{ ({alias})-[:INTRODUCED_IN]->(intro:DDVersion) "
        f"WHERE intro.major <= $dd_major }}"
        f")"
    )
    return clause, {"dd_major": dd_version}
```

### 1.2 `_check_imas` — Path Validation

**Source Cypher from:** `GraphPathTool.check_imas_paths()` (graph_search.py:267–320)

```python
def _check_imas(
    paths: str,
    *,
    ids: str | None = None,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Validate IMAS paths against the graph.

    For each path: checks existence, returns data_type + units,
    detects RENAMED_TO migrations for deprecated paths.
    """
    gc = gc or GraphClient()
    path_list = [p.strip() for p in paths.replace(",", " ").split() if p.strip()]
    if ids:
        path_list = [
            f"{ids}/{p}" if not p.startswith(ids) else p for p in path_list
        ]

    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    results = []
    found = 0
    for path in path_list:
        row = gc.query(
            f"""
            MATCH (p:IMASNode {{id: $path}})
            WHERE true {dd_where}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS id, p.ids AS ids, p.data_type AS data_type,
                   p.documentation AS documentation, u.id AS units
            """,
            path=path, **dd_params,
        )
        if row:
            results.append({**row[0], "exists": True})
            found += 1
        else:
            # Check for renamed/deprecated path
            renamed = gc.query(
                """
                MATCH (old:IMASNode {id: $path})-[:RENAMED_TO]->(new:IMASNode)
                RETURN old.id AS old_path, new.id AS new_path
                """,
                path=path,
            )
            if renamed:
                results.append({
                    "id": path, "exists": False,
                    "renamed_to": renamed[0]["new_path"],
                    "suggestion": f"Use {renamed[0]['new_path']} instead",
                })
            else:
                results.append({"id": path, "exists": False})

    return _format_check_report(results, found, len(path_list))
```

### 1.3 `_fetch_imas` — Full Path Documentation

**Source Cypher from:** `GraphPathTool.fetch_imas_paths()` (graph_search.py:327–390)

```python
def _fetch_imas(
    paths: str,
    *,
    ids: str | None = None,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Fetch detailed IMAS path metadata via graph traversal.

    Enriches each path with: documentation, data_type, units, cluster
    labels (IN_CLUSTER), coordinates (HAS_COORDINATE), physics domain,
    introduced version, lifecycle status, coordinate_same_as.
    """
    gc = gc or GraphClient()
    path_list = [p.strip() for p in paths.replace(",", " ").split() if p.strip()]
    if ids:
        path_list = [
            f"{ids}/{p}" if not p.startswith(ids) else p for p in path_list
        ]

    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    nodes = []
    not_found = []
    for path in path_list:
        row = gc.query(
            f"""
            MATCH (p:IMASNode {{id: $path}})
            WHERE true {dd_where}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
            OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
            OPTIONAL MATCH (p)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
            RETURN p.id AS id, p.name AS name, p.ids AS ids,
                   p.documentation AS documentation, p.data_type AS data_type,
                   p.node_type AS node_type, p.physics_domain AS physics_domain,
                   p.ndim AS ndim,
                   p.lifecycle_status AS lifecycle_status,
                   p.lifecycle_version AS lifecycle_version,
                   p.path_doc AS structure_reference,
                   p.coordinate1_same_as AS coordinate1,
                   p.coordinate2_same_as AS coordinate2,
                   p.cocos_label_transformation AS cocos,
                   u.id AS units, u.symbol AS unit_symbol,
                   collect(DISTINCT c.label) AS cluster_labels,
                   collect(DISTINCT coord.id) AS coordinates,
                   intro.id AS introduced_in,
                   ident.name AS identifier_schema
            """,
            path=path, **dd_params,
        )
        if row and row[0]["id"]:
            nodes.append(row[0])
        else:
            not_found.append(path)

    return _format_fetch_imas_report(nodes, not_found)
```

### 1.4 `_list_imas` — IDS Path Listing

**Source Cypher from:** `GraphListTool.list_imas_paths()` (graph_search.py:423–510)

```python
def _list_imas(
    paths: str,
    *,
    leaf_only: bool = True,
    max_paths: int = 100,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """List IMAS paths within an IDS or subtree.

    Queries IMASNode nodes by ids name or path prefix.
    Optionally filters to leaf-only (excludes STRUCTURE/STRUCT_ARRAY).
    """
    gc = gc or GraphClient()
    queries = paths.strip().split()

    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    leaf_filter = (
        "AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']" if leaf_only else ""
    )

    parts = []
    for query in queries:
        if "/" in query:
            ids_name = query.split("/")[0]
            prefix = query
        else:
            ids_name = query
            prefix = None

        # Verify IDS exists
        ids_exists = gc.query("MATCH (i:IDS {id: $name}) RETURN i.name", name=ids_name)
        if not ids_exists:
            parts.append(f"IDS '{ids_name}' not found.\n")
            continue

        if prefix and "/" in prefix:
            path_results = gc.query(
                f"""
                MATCH (p:IMASNode)
                WHERE p.id STARTS WITH $prefix
                {leaf_filter} {dd_where}
                RETURN p.id AS id
                ORDER BY p.id
                LIMIT $limit
                """,
                prefix=prefix, limit=max_paths, **dd_params,
            )
        else:
            path_results = gc.query(
                f"""
                MATCH (p:IMASNode)
                WHERE p.ids = $ids_name
                {leaf_filter} {dd_where}
                RETURN p.id AS id
                ORDER BY p.id
                LIMIT $limit
                """,
                ids_name=ids_name, limit=max_paths, **dd_params,
            )

        path_ids = [r["id"] for r in (path_results or [])]
        parts.append(f"## {query} ({len(path_ids)} paths)\n")
        for pid in path_ids:
            parts.append(f"  {pid}")
        if len(path_ids) >= max_paths:
            parts.append(f"  ... truncated to {max_paths}")
        parts.append("")

    return "\n".join(parts) or "No paths found."
```

### 1.5 `_get_imas_overview` — IDS Catalog Overview

**Source Cypher from:** `GraphOverviewTool.get_imas_overview()` (graph_search.py:540–630)

```python
def _get_imas_overview(
    query: str | None = None,
    *,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Get overview of available IMAS Interface Data Structures.

    Queries IDS nodes for names, descriptions, path counts,
    physics domains. Includes DDVersion metadata.
    """
    gc = gc or GraphClient()

    # IDS statistics
    ids_results = gc.query("""
        MATCH (i:IDS)
        OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASNode)
        WITH i, count(p) AS path_count
        RETURN i.name AS name, i.description AS description,
               i.physics_domain AS physics_domain,
               i.lifecycle_status AS lifecycle_status,
               path_count
        ORDER BY path_count DESC
    """)

    # Current DD version
    version_results = gc.query(
        "MATCH (v:DDVersion {is_current: true}) RETURN v.id AS version"
    )
    current_version = version_results[0]["version"] if version_results else "unknown"

    # Filter by query if provided
    filtered = []
    for r in (ids_results or []):
        if query:
            q = query.lower()
            if not (q in r["name"].lower()
                    or q in (r["description"] or "").lower()
                    or q in (r["physics_domain"] or "").lower()):
                continue
        filtered.append(r)

    return _format_overview_report(filtered, current_version, query)
```

### 1.6 `_get_imas_identifiers` — Identifier Schema Browser

**Source Cypher from:** `GraphIdentifiersTool.get_imas_identifiers()` (graph_search.py:825–900)

```python
def _get_imas_identifiers(
    query: str | None = None,
    *,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Browse IMAS identifier/enumeration schemas.

    Queries IdentifierSchema nodes for valid values of typed fields
    (coordinate systems, grid types, probe types, etc.).
    """
    gc = gc or GraphClient()

    results = gc.query("""
        MATCH (s:IdentifierSchema)
        RETURN s.name AS name, s.description AS description,
               s.option_count AS option_count, s.options AS options,
               s.field_count AS field_count, s.source AS source
        ORDER BY s.name
    """)

    schemas = []
    for r in (results or []):
        if query:
            q = query.lower()
            if not (q in (r["name"] or "").lower()
                    or q in (r["description"] or "").lower()
                    or q in (r["options"] or "").lower()):
                continue
        schemas.append(r)

    return _format_identifiers_report(schemas)
```

### 1.7 `_search_imas_clusters` — Semantic Cluster Search

**Source Cypher from:** `GraphClustersTool.search_imas_clusters()` (graph_search.py:640–800)

```python
def _search_imas_clusters(
    query: str,
    *,
    scope: str | None = None,
    ids_filter: str | None = None,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search semantic clusters of related IMAS paths.

    Supports two query types:
    - Path query (contains '/'): find clusters containing that path
    - Text query: vector search on cluster_description_embedding
    """
    gc = gc or GraphClient()

    # Path-based lookup
    if "/" in query and " " not in query:
        scope_filter = "AND c.scope = $scope" if scope else ""
        params = {"path": query}
        if scope:
            params["scope"] = scope

        results = gc.query(f"""
            MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            {scope_filter}
            OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
            WITH c, collect(DISTINCT member.id) AS paths
            RETURN c.id AS id, c.label AS label, c.description AS description,
                   c.scope AS scope, c.cross_ids AS cross_ids,
                   c.ids_names AS ids_names, paths
            ORDER BY c.scope
        """, **params)

        return _format_cluster_report(results or [], query, "path")

    # Semantic vector search
    if encoder is None:
        encoder = Encoder(EncoderConfig())
    embedding = _embed(encoder, query)

    scope_filter = "AND cluster.scope = $scope" if scope else ""
    ids_clause = ""
    params = {"embedding": embedding, "k": 10}
    if scope:
        params["scope"] = scope
    if ids_filter:
        filter_list = ids_filter.split() if isinstance(ids_filter, str) else ids_filter
        ids_clause = "AND any(n IN cluster.ids_names WHERE n IN $ids_filter)"
        params["ids_filter"] = filter_list

    results = gc.query(f"""
        CALL db.index.vector.queryNodes(
            'cluster_description_embedding', $k, $embedding
        )
        YIELD node AS cluster, score
        WHERE score > 0.3
        {scope_filter} {ids_clause}
        OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(cluster)
        WITH cluster, score, collect(DISTINCT member.id) AS paths
        RETURN cluster.id AS id, cluster.label AS label,
               cluster.description AS description,
               cluster.scope AS scope, cluster.cross_ids AS cross_ids,
               cluster.ids_names AS ids_names,
               score AS relevance_score, paths
        ORDER BY relevance_score DESC
    """, **params)

    return _format_cluster_report(results or [], query, "semantic")
```

### Formatters

**File: `imas_codex/llm/search_formatters.py`** — add new formatter functions:

- `_format_check_report(results, found, total)` → validation table
- `_format_fetch_imas_report(nodes, not_found)` → detailed documentation
- `_format_overview_report(ids_list, dd_version, query)` → IDS catalog
- `_format_identifiers_report(schemas)` → identifier options table
- `_format_cluster_report(clusters, query, mode)` → cluster membership

Each follows the existing pattern: takes structured dicts, returns markdown `str`.

### MCP Registration

**File: `imas_codex/llm/server.py`** — add 6 tools to `_register_tools()`:

```python
@self.mcp.tool()
def check_imas(paths: str, ids: str | None = None, dd_version: int | None = 4) -> str:
    """Validate IMAS paths against the Data Dictionary graph.
    Checks existence, returns data type + units, detects renamed/deprecated paths.
    Args:
        paths: Space-delimited IMAS paths
        ids: Optional IDS prefix to prepend
        dd_version: DD major version scope (default 4, None=all)
    """
    return _check_imas(paths, ids=ids, dd_version=dd_version)

@self.mcp.tool()
def fetch_imas(paths: str, ids: str | None = None, dd_version: int | None = 4) -> str:
    """Get full documentation for IMAS paths with graph enrichment.
    Returns documentation, units, coordinates, cluster membership, lifecycle.
    Args:
        paths: Space-delimited IMAS paths
        ids: Optional IDS prefix to prepend
        dd_version: DD major version scope (default 4, None=all)
    """
    return _fetch_imas(paths, ids=ids, dd_version=dd_version)

@self.mcp.tool()
def list_imas(paths: str, leaf_only: bool = True, max_paths: int = 100,
              dd_version: int | None = 4) -> str:
    """List data paths within an IMAS IDS or subtree.
    Args:
        paths: Space-separated IDS names or path prefixes
        leaf_only: Only return leaf data fields (default True)
        max_paths: Max paths to return (default 100)
        dd_version: DD major version scope (default 4, None=all)
    """
    return _list_imas(paths, leaf_only=leaf_only, max_paths=max_paths, dd_version=dd_version)

@self.mcp.tool()
def get_imas_overview(query: str | None = None, dd_version: int | None = 4) -> str:
    """Get overview of available IMAS IDS with statistics and physics domains.
    Args:
        query: Optional keyword filter
        dd_version: DD major version scope (default 4, None=all)
    """
    return _get_imas_overview(query, dd_version=dd_version)

@self.mcp.tool()
def get_imas_identifiers(query: str | None = None, dd_version: int | None = 4) -> str:
    """Browse IMAS identifier/enumeration schemas (coordinate systems, grid types, etc).
    Args:
        query: Optional filter keyword
        dd_version: DD major version scope (default 4, None=all)
    """
    return _get_imas_identifiers(query, dd_version=dd_version)

@self.mcp.tool()
def search_imas_clusters(query: str, scope: str | None = None,
                         ids_filter: str | None = None, dd_version: int | None = 4) -> str:
    """Search semantic clusters of related IMAS data paths.
    Args:
        query: Natural language description or exact IMAS path
        scope: Cluster scope filter — "global", "domain", or "ids"
        ids_filter: Limit to clusters containing paths from specific IDS
        dd_version: DD major version scope (default 4, None=all)
    """
    return _search_imas_clusters(query, scope=scope, ids_filter=ids_filter, dd_version=dd_version)
```

### Update `search_imas`

Add `dd_version: int | None = 4` to existing `search_imas` tool signature.
Thread through to `_search_imas()` → `_vector_search_imas_paths()` → `_text_search_imas_paths_by_query()` → `_enrich_imas_paths()`.

---

## Phase 2: Advanced IMAS Tools — New Graph Capabilities

These tools have no direct ancestor in the IMAS MCP server. They exploit the Codex MCP's
graph advantage — multi-hop traversals, cross-index vector search, and Cypher aggregation
that exceed what the old file-backed tools could provide.

### 2.1 `_explore_imas_relationships` — Cross-IDS Relationship Explorer

**Goal:** Given an IMAS path, traverse graph relationships to discover connections
across IDS boundaries. This is a **Codex-exclusive capability** — the IMAS MCP server
has `explore_relationships` but it only does basic parent/child navigation. Graph
traversal enables discovering non-obvious cross-IDS connections.

**Graph edges to traverse:**

| Relationship | Meaning | Cross-IDS? |
|-------------|---------|------------|
| `IN_CLUSTER` → `IMASSemanticCluster` | Shared physics concept | Yes — clusters span IDS |
| `HAS_COORDINATE` → `IMASCoordinateSpec` | Shared coordinate axis | Yes — same coord across IDS |
| `HAS_UNIT` → `Unit` | Same measurement unit | Possible — same unit, same domain |
| `HAS_IDENTIFIER_SCHEMA` → `IdentifierSchema` | Shared enumeration | Yes — same schema across IDS |
| `INTRODUCED_IN` → `DDVersion` | Version context | No — but useful metadata |
| Facility-scoped: `MAPS_TO_IMAS` via `SignalSource` | Signal mappings | N/A — facility cross-ref |

```python
def _explore_imas_relationships(
    path: str,
    *,
    relationship_types: str = "all",
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Explore cross-IDS relationships for an IMAS path via graph traversal.

    Discovers sibling paths via shared clusters, coordinates, units,
    and identifier schemas. Groups results by relationship type.
    """
    gc = gc or GraphClient()
    dd_clause, dd_params = _dd_version_clause("sibling", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    sections = {}

    # 1. Cluster siblings — paths in same semantic cluster but different IDS
    if relationship_types in ("all", "cluster"):
        cluster_siblings = gc.query(f"""
            MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
                  <-[:IN_CLUSTER]-(sibling:IMASNode)
            WHERE sibling.ids <> p.ids {dd_where}
            RETURN cl.label AS cluster, sibling.id AS sibling_path,
                   sibling.ids AS sibling_ids, sibling.documentation AS doc
            ORDER BY cl.label, sibling.ids
        """, path=path, **dd_params)
        if cluster_siblings:
            sections["cluster"] = cluster_siblings

    # 2. Coordinate partners — paths sharing same coordinate spec
    if relationship_types in ("all", "coordinate"):
        coord_partners = gc.query(f"""
            MATCH (p:IMASNode {{id: $path}})-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
                  <-[:HAS_COORDINATE]-(sibling:IMASNode)
            WHERE sibling.ids <> p.ids {dd_where}
            RETURN coord.id AS coordinate, sibling.id AS sibling_path,
                   sibling.ids AS sibling_ids, sibling.data_type AS data_type
            ORDER BY coord.id, sibling.ids
        """, path=path, **dd_params)
        if coord_partners:
            sections["coordinate"] = coord_partners

    # 3. Unit companions — paths with same unit in same physics domain
    if relationship_types in ("all", "unit"):
        unit_companions = gc.query(f"""
            MATCH (p:IMASNode {{id: $path}})-[:HAS_UNIT]->(u:Unit)
                  <-[:HAS_UNIT]-(sibling:IMASNode)
            WHERE sibling.ids <> p.ids
              AND sibling.physics_domain = p.physics_domain
              {dd_where}
            RETURN u.id AS unit, sibling.id AS sibling_path,
                   sibling.ids AS sibling_ids, sibling.documentation AS doc
            ORDER BY u.id, sibling.ids
            LIMIT 30
        """, path=path, **dd_params)
        if unit_companions:
            sections["unit"] = unit_companions

    # 4. Identifier schema links — paths sharing same identifier enumeration
    if relationship_types in ("all", "identifier"):
        ident_links = gc.query(f"""
            MATCH (p:IMASNode {{id: $path}})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
                  <-[:HAS_IDENTIFIER_SCHEMA]-(sibling:IMASNode)
            WHERE sibling.ids <> p.ids {dd_where}
            RETURN s.name AS schema_name, sibling.id AS sibling_path,
                   sibling.ids AS sibling_ids
            ORDER BY s.name
        """, path=path, **dd_params)
        if ident_links:
            sections["identifier"] = ident_links

    return _format_relationships_report(path, sections)
```

### 2.2 `_explain_imas_concept` — Physics Concept Explainer

**Goal:** Given a physics concept, synthesize an explanation from multiple graph indexes.
This exceeds the IMAS MCP's `explain_concept` by drawing from IMAS paths, clusters,
identifiers, AND facility wiki documentation.

```python
def _explain_imas_concept(
    concept: str,
    *,
    depth: str = "standard",
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Explain an IMAS physics concept using multi-index graph synthesis.

    Gathers context from:
    1. IMAS path embeddings — relevant DD paths + documentation
    2. Cluster embeddings — physics groupings containing the concept
    3. IdentifierSchema nodes — enumeration values related to concept
    4. Wiki chunk embeddings — expert documentation from facility wikis
    5. Coordinate specs — relevant coordinate systems
    """
    gc = gc or GraphClient()
    if encoder is None:
        encoder = Encoder(EncoderConfig())
    embedding = _embed(encoder, concept)
    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    # 1. Relevant IMAS paths (top 10)
    imas_paths = gc.query(f"""
        CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
        YIELD node AS p, score
        WHERE score > 0.5
        AND NOT (p)-[:DEPRECATED_IN]->(:DDVersion)
        {dd_where}
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        RETURN p.id AS path, p.ids AS ids, p.documentation AS doc,
               p.data_type AS data_type, u.symbol AS unit, score
        ORDER BY score DESC
        LIMIT 10
    """, k=50, embedding=embedding, **dd_params)

    # 2. Related clusters (top 5)
    clusters = gc.query("""
        CALL db.index.vector.queryNodes('cluster_description_embedding', $k, $embedding)
        YIELD node AS cl, score
        WHERE score > 0.4
        RETURN cl.label AS label, cl.description AS description,
               cl.scope AS scope, cl.path_count AS path_count, score
        ORDER BY score DESC
        LIMIT 5
    """, k=20, embedding=embedding)

    # 3. Matching identifiers
    identifiers = gc.query("""
        MATCH (s:IdentifierSchema)
        WHERE toLower(s.name) CONTAINS $concept_lower
           OR toLower(s.description) CONTAINS $concept_lower
        RETURN s.name AS name, s.description AS description,
               s.option_count AS options
        LIMIT 5
    """, concept_lower=concept.lower())

    # 4. Wiki context (top 5 chunks, if available)
    wiki_chunks = []
    try:
        wiki_chunks = gc.query("""
            CALL db.index.vector.queryNodes('wiki_chunk_embedding', $k, $embedding)
            YIELD node AS wc, score
            WHERE score > 0.5
            OPTIONAL MATCH (page:WikiPage)-[:HAS_CHUNK]->(wc)
            RETURN wc.section AS section, wc.text AS text,
                   page.title AS page_title, score
            ORDER BY score DESC
            LIMIT 5
        """, k=20, embedding=embedding)
    except Exception:
        pass  # wiki index may not exist

    # 5. Coordinate specs
    coord_specs = gc.query("""
        MATCH (c:IMASCoordinateSpec)
        WHERE toLower(c.id) CONTAINS $concept_lower
        RETURN c.id AS id, c.is_bounded AS bounded
        LIMIT 5
    """, concept_lower=concept.lower())

    return _format_concept_report(
        concept, imas_paths, clusters, identifiers, wiki_chunks, coord_specs, depth
    )
```

### 2.3 `_analyze_imas_structure` — IDS Structural Analysis

**Goal:** Pure-Cypher structural analysis of an IDS. Replaces the NetworkX-based
`graph_analyzer.py` with zero-copy graph aggregation.

```python
def _analyze_imas_structure(
    ids_name: str,
    *,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Analyze the hierarchical structure of an IMAS IDS via Cypher.

    Computes: depth metrics, leaf/structure ratio, branching factors,
    array patterns, physics domain distribution, coordinate usage.
    """
    gc = gc or GraphClient()
    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""

    # Basic metrics
    metrics = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE true {dd_where}
        WITH count(p) AS total,
             count(CASE WHEN p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'] THEN 1 END) AS structures,
             count(CASE WHEN NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'] THEN 1 END) AS leaves,
             max(size(split(p.id, '/'))) AS max_depth,
             avg(size(split(p.id, '/'))) AS avg_depth
        RETURN total, structures, leaves, max_depth, round(avg_depth, 2) AS avg_depth
    """, ids_name=ids_name, **dd_params)

    if not metrics or metrics[0]["total"] == 0:
        return f"IDS '{ids_name}' not found or has no paths."

    # Physics domain distribution
    domains = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE p.physics_domain IS NOT NULL {dd_where}
        RETURN p.physics_domain AS domain, count(*) AS count
        ORDER BY count DESC
    """, ids_name=ids_name, **dd_params)

    # Array structures with coordinates
    arrays = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE p.data_type = 'STRUCT_ARRAY' {dd_where}
        OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        RETURN p.id AS path, p.maxoccur AS maxoccur,
               collect(DISTINCT coord.id) AS coordinates
        ORDER BY p.id
    """, ids_name=ids_name, **dd_params)

    # Data type distribution
    type_dist = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE true {dd_where}
        RETURN p.data_type AS data_type, count(*) AS count
        ORDER BY count DESC
    """, ids_name=ids_name, **dd_params)

    # COCOS-labeled fields
    cocos = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE p.cocos_label_transformation IS NOT NULL {dd_where}
        RETURN p.id AS path, p.cocos_label_transformation AS transform
        ORDER BY p.id
    """, ids_name=ids_name, **dd_params)

    return _format_structure_report(ids_name, metrics[0], domains, arrays, type_dist, cocos)
```

### 2.4 `_export_imas_ids` — Full IDS Export

```python
def _export_imas_ids(
    ids_name: str,
    *,
    leaf_only: bool = False,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Export full IDS structure with documentation, units, and types.

    Single enriched Cypher query — not a composition of list + fetch.
    """
    gc = gc or GraphClient()
    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""
    leaf_filter = (
        "AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']" if leaf_only else ""
    )

    results = gc.query(f"""
        MATCH (p:IMASNode {{ids: $ids_name}})
        WHERE true {dd_where} {leaf_filter}
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        RETURN p.id AS path, p.name AS name, p.documentation AS doc,
               p.data_type AS data_type, p.ndim AS ndim,
               u.symbol AS unit, p.lifecycle_status AS lifecycle
        ORDER BY p.id
    """, ids_name=ids_name, **dd_params)

    if not results:
        return f"IDS '{ids_name}' not found."

    return _format_export_report(ids_name, results, leaf_only)
```

### 2.5 `_export_imas_domain` — Physics Domain Export

```python
def _export_imas_domain(
    domain: str,
    *,
    ids_filter: str | None = None,
    dd_version: int | None = 4,
    gc: GraphClient | None = None,
) -> str:
    """Export all IMAS paths in a physics domain, grouped by IDS.

    Shows how a physics concept is represented across the Data Dictionary.
    """
    gc = gc or GraphClient()
    dd_clause, dd_params = _dd_version_clause("p", dd_version)
    dd_where = f"AND {dd_clause}" if dd_clause else ""
    ids_clause = ""
    if ids_filter:
        ids_list = ids_filter.split()
        ids_clause = "AND p.ids IN $ids_list"
        dd_params["ids_list"] = ids_list

    results = gc.query(f"""
        MATCH (p:IMASNode)
        WHERE p.physics_domain = $domain {dd_where} {ids_clause}
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        RETURN p.ids AS ids, p.id AS path, p.documentation AS doc,
               p.data_type AS data_type, u.symbol AS unit
        ORDER BY p.ids, p.id
    """, domain=domain, **dd_params)

    if not results:
        return f"No paths found for physics domain '{domain}'."

    return _format_domain_export_report(domain, results)
```

### MCP Registration for Phase 2

```python
@self.mcp.tool()
def explore_imas_relationships(path: str, relationship_types: str = "all",
                                dd_version: int | None = 4) -> str: ...

@self.mcp.tool()
def explain_imas_concept(concept: str, depth: str = "standard",
                          dd_version: int | None = 4) -> str: ...

@self.mcp.tool()
def analyze_imas_structure(ids_name: str, dd_version: int | None = 4) -> str: ...

@self.mcp.tool()
def export_imas_ids(ids_name: str, leaf_only: bool = False,
                     dd_version: int | None = 4) -> str: ...

@self.mcp.tool()
def export_imas_domain(domain: str, ids_filter: str | None = None,
                        dd_version: int | None = 4) -> str: ...
```

---

## Phase 3: Search Quality Improvement

**Goal:** Bring `search_imas` relevance to match or exceed the IMAS MCP for focused physics queries.

### Problem

For "equilibrium boundary separatrix":
- **IMAS MCP** returned: `boundary/psi_norm`, `boundary/outline/r`, `boundary/outline/z` — directly relevant
- **Codex MCP** returned: `profiles_1d/chi_squared`, `profiles_1d/q` — tangentially related

### Root Causes

1. **No DD version filter** — DD 3.x duplicates dilute top-k (fixed by Phase 1 `dd_version=4` default)
2. **No path segment boost** — path IDs containing query words aren't boosted
3. **Cluster overlay inflation** — cluster member paths added to scored list broaden results

### Fixes (all in `search_tools.py`)

**3.1 Thread `dd_version` into `_search_imas`**

Add `dd_version: int | None = 4` parameter, pass to `_vector_search_imas_paths()` and `_text_search_imas_paths_by_query()`. Use `_dd_version_clause()` in both Cypher queries.

**3.2 Path segment boost**

After merging vector + text scores, boost paths whose ID segments match query words:

```python
query_words = [w.lower() for w in query.split() if len(w) > 2]
for pid in sorted_ids:
    segments = pid.lower().split("/")
    match_count = sum(1 for w in query_words if any(w in seg for seg in segments))
    if match_count > 0:
        scores[pid] += 0.03 * match_count
```

**3.3 Cluster overlay cap**

Currently `_vector_search_clusters()` returns cluster metadata for the report.
Ensure cluster member paths are NOT injected into the scored `path_ids` list —
clusters are display-only context in the formatted report.

### Regression Tests

| Query | Expected top-5 must contain |
|-------|---------------------------|
| "equilibrium boundary separatrix" | `boundary/outline/r` or `boundary/psi_norm` |
| "electron temperature" | `core_profiles/profiles_1d/electrons/temperature` |
| "plasma current" | `equilibrium/time_slice/global_quantities/ip` |
| "safety factor q profile" | `equilibrium/time_slice/profiles_1d/q` |

---

## Phase 4: REPL Function & Registry Update

### REPL Functions

Update existing REPL closures (`search_imas`, `fetch_imas`, `list_imas`, `check_imas`)
to call the new `_` functions from `search_tools.py` instead of `_get_imas_tools()`.
This gives REPL users the same graph-backed implementations:

```python
# Before (delegation to old Tools class):
def fetch_imas(paths: str) -> str:
    tools = _get_imas_tools()
    result = _run_async(tools.fetch_imas_paths(paths=paths))
    return str(result)

# After (direct call to graph-backed function):
def fetch_imas(paths: str, dd_version: int | None = 4) -> str:
    return _fetch_imas(paths, dd_version=dd_version)
```

### REPL Registry

Add all new tools to the `_registry` dict under `IMAS_DD`:

```python
"IMAS_DD": {
    "search_imas": ...,
    "fetch_imas": ...,
    "list_imas": ...,
    "check_imas": ...,
    "get_imas_overview": ...,
    "get_imas_identifiers": ...,
    "search_imas_clusters": ...,
    "explore_imas_relationships": ...,
    "explain_imas_concept": ...,
    "analyze_imas_structure": ...,
    "export_imas_ids": ...,
    "export_imas_domain": ...,
},
```

### Remove `_get_imas_tools()` Dependency

Once all REPL functions are converted to use `_` functions directly, the
`_get_imas_tools()` / `_imas_tools_instance` singleton can be removed from
`server.py`. The `imas_codex.tools` module remains for the standalone
IMAS MCP server — it is not touched.

---

## Implementation Order

```
Phase 1  (8 tasks)   Core IMAS tools + DD version scoping
   │                 1.1 _dd_version_clause helper
   │                 1.2 _check_imas
   │                 1.3 _fetch_imas
   │                 1.4 _list_imas
   │                 1.5 _get_imas_overview
   │                 1.6 _get_imas_identifiers
   │                 1.7 _search_imas_clusters
   │                 1.8 Formatters + MCP registration + tests
   ▼
Phase 2  (6 tasks)   Advanced tools — Codex graph advantage
   │                 2.1 _explore_imas_relationships
   │                 2.2 _explain_imas_concept
   │                 2.3 _analyze_imas_structure
   │                 2.4 _export_imas_ids
   │                 2.5 _export_imas_domain
   │                 2.6 Formatters + MCP registration + tests
   ▼
Phase 3  (3 tasks)   Search quality
   │                 3.1 Thread dd_version into _search_imas
   │                 3.2 Path segment boost
   │                 3.3 Cluster overlay cap
   ▼
Phase 4  (2 tasks)   REPL & registry cleanup
                     4.1 Convert REPL functions to _functions
                     4.2 Remove _get_imas_tools() dependency
```

### Dependencies

- Phase 1.2–1.7 all depend on 1.1 (dd_version helper)
- Phase 2 is independent of Phase 1 (uses same helper, different functions)
- Phase 3 depends on Phase 1.1 (dd_version threading)
- Phase 4 depends on Phase 1 (needs _functions to exist before REPL can use them)

---

## Files Modified

| File | Phases | Changes |
|------|--------|---------|
| `imas_codex/llm/search_tools.py` | 1, 2, 3 | Add 12 new `_` functions + dd_version helper + search quality fixes |
| `imas_codex/llm/search_formatters.py` | 1, 2 | Add 8 new formatter functions |
| `imas_codex/llm/server.py` | 1, 2, 4 | Register 11 new MCP tools, update REPL functions, remove `_get_imas_tools()` |
| `tests/llm/test_imas_tools.py` | 1, 2, 3 | New test file: all IMAS tool coverage with mock GraphClient |

### Files NOT Modified

| File | Reason |
|------|--------|
| `imas_codex/tools/graph_search.py` | Belongs to IMAS MCP server — not touched |
| `imas_codex/tools/__init__.py` | Belongs to IMAS MCP server — not touched |
| `imas_codex/server.py` | IMAS MCP server entry point — not touched |
| `imas_codex/graph_analyzer.py` | Old NetworkX analyzer — superseded, not touched |

---

## Codex MCP Tool Inventory (Post-Implementation)

| Tool | Category | Phase |
|------|----------|-------|
| `search_signals` | Facility | Existing |
| `signal_analytics` | Facility | Existing |
| `search_docs` | Facility | Existing |
| `search_code` | Facility | Existing |
| `fetch` | Retrieval | Existing |
| `search_imas` | IMAS DD | Existing (improved Phase 3) |
| `check_imas` | IMAS DD | Phase 1 |
| `fetch_imas` | IMAS DD | Phase 1 |
| `list_imas` | IMAS DD | Phase 1 |
| `get_imas_overview` | IMAS DD | Phase 1 |
| `get_imas_identifiers` | IMAS DD | Phase 1 |
| `search_imas_clusters` | IMAS DD | Phase 1 |
| `explore_imas_relationships` | IMAS DD | Phase 2 |
| `explain_imas_concept` | IMAS DD | Phase 2 |
| `analyze_imas_structure` | IMAS DD | Phase 2 |
| `export_imas_ids` | IMAS DD | Phase 2 |
| `export_imas_domain` | IMAS DD | Phase 2 |
| `get_graph_schema` | Graph | Existing |
| `add_to_graph` | Graph | Existing |
| `update_facility_infrastructure` | Infrastructure | Existing |
| `get_facility_infrastructure` | Infrastructure | Existing |
| `add_exploration_note` | Infrastructure | Existing |
| `update_facility_config` | Infrastructure | Existing |
| `get_discovery_context` | Infrastructure | Existing |
| `list_logs` | Logs | Existing |
| `get_logs` | Logs | Existing |
| `tail_logs` | Logs | Existing |
| `python` | REPL | Existing |

**Total: 27 tools (11 new IMAS DD tools)**

---

## Validation Criteria

1. **All graph-backed:** Every `_` function in search_tools.py uses `gc.query()` directly — no imports from `imas_codex.tools`
2. **DD version scoping:** `dd_version=4` (default) filters to DD 4.x; `dd_version=None` returns all
3. **Search quality:** "equilibrium boundary separatrix" returns boundary paths in top 5
4. **Tool parity:** `list_tools` returns all 11 new IMAS tools
5. **No regressions:** All existing tests pass
6. **REPL parity:** REPL functions use same graph-backed implementations as MCP tools
7. **No `_get_imas_tools()`:** Codex MCP server has no dependency on `imas_codex.tools` at runtime
