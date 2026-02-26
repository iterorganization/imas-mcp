# Graph-Native MCP Server

## Motivation

The IMAS MCP server currently loads all data from flat JSON files at startup, runs semantic search via in-process numpy dot products, and builds embeddings and clusters inside the Docker image. This creates duplicate storage (graph + files), a brittle build chain (`.pkl`, `.npz`, `clusters.json`, JSON schemas), and no ability to traverse graph relationships at query time.

The graph already contains every data point the server needs: `IMASPath` nodes with documentation, units, coordinates, physics domains, embeddings, and cluster memberships; `IMASSemanticCluster` nodes with label/description embeddings; `IDS`, `Unit`, `IMASCoordinateSpec`, and `IdentifierSchema` nodes. The `DocumentStore` and its JSON files are a redundant read-only projection of this graph.

The graph also contains the complete version history of the IMAS Data Dictionary — 34 versions from 3.22.0 to 4.1.0 — with `IMASPathChange` nodes tracking every metadata mutation, `INTRODUCED_IN`/`DEPRECATED_IN` version lifecycle on every path, `RENAMED_TO` migration chains, and `SemanticChangeType` classification of physics-significant changes (sign conventions, coordinate systems). None of this version intelligence is currently exposed to MCP clients. The file-based server is locked to a single DD version with no ability to query evolution.

This plan replaces all file-based data with a single Neo4j instance embedded in the Docker image, loaded from a pre-built IMAS-only graph pulled from GHCR. JSON files, pickle caches, and numpy arrays are eliminated entirely. The graph-native architecture also enables a new class of version-aware and schema-introspecting tools that support flexible codegen discovery. This is a clean break — no backward compatibility with file-based deployments.

## Phase Dependencies

```
Phase 0: Test Infrastructure ──────────────────────────────────┐
                                                                │
Phase 1: IMAS Graph on GHCR ───┐                               │
                                ├── Phase 3: Embedded Neo4j ──── Phase 5: Graph-Native Search
Phase 2: Tool Architecture ────┘                               │
                                                                │
Phase 4: Incremental Clusters ─────────────────────────────────┘
                                                                │
                                                    Phase 6: Cleanup
```

**Parallel work:**
- Phase 0 (test infrastructure) can start immediately and run throughout
- Phase 1 (GHCR) and Phase 2 (tool architecture) are independent — work in parallel
- Phase 4 (incremental clusters) is independent of the main pipeline — can be done at any time
- Phase 3 (embedded Neo4j) depends on Phase 1 completing
- Phase 5 (graph-native search) depends on Phases 2 and 3
- Phase 6 (cleanup) is the final step after everything else is validated

## Architecture

### Current Flow

```
Docker build:
  build-models → generated Python models
  build-schemas → JSON files (~15k paths)
  build-path-map → version upgrade mappings
  build-embeddings → .pkl cache
  clusters build → clusters.json + .npz

Server startup:
  DocumentStore ← JSON files (keyword search, path lookup)
  Embeddings ← .pkl cache → numpy matrix
  SemanticSearch ← numpy dot products
  ClusterSearcher ← clusters.json + .npz → numpy dot products
  dd_version ← hardcoded from imas-python package
```

### Target Flow

```
Docker build:
  graph pull --imas → download IMAS-only graph dump from GHCR (private, requires GHCR_TOKEN)
  neo4j-admin load → load dump into embedded Neo4j data directory
  FAIL if graph pull fails (no fallback — the graph IS the data)

Server startup:
  Neo4j ← starts from pre-loaded data directory
  GraphClient ← connects to local bolt://localhost:7687
  All search ← Neo4j vector indexes + Cypher traversal
  Encoder ← loads model for query-time embedding only
  DD versions ← queried from DDVersion nodes (version range + current)
  Schema ← introspected from LinkML at startup, exposed to clients
```

### What Changes

| Component | Current | Target |
|-----------|---------|--------|
| Path semantic search | numpy over .pkl | `db.index.vector.queryNodes('imas_path_embedding', k, $vec)` |
| Cluster search | numpy over .npz | `db.index.vector.queryNodes('cluster_description_embedding', k, $vec)` |
| Keyword / full-text search | SQLite FTS5 over JSON | Neo4j full-text index |
| Path lookup | `DocumentStore.get_document(path_id)` | `MATCH (p:IMASPath {id: $path})` |
| IDS listing | JSON catalog file | `MATCH (i:IDS) RETURN i.name` |
| Cluster path membership | in-memory dict from clusters.json | `MATCH (p)-[:IN_CLUSTER]->(c)` |
| Identifier schemas | identifier_catalog.json | `MATCH (s:IdentifierSchema)` |
| Version info | `imas_codex.dd_version` (single hardcoded) | DDVersion nodes from graph (range + current) |
| Graph exploration | Not exposed to MCP clients | Read-only Cypher tool + schema introspection |
| Version evolution | Not available | `IMASPathChange` traversal via Cypher |
| Docker image data | JSON + pkl + npz (~200MB) | Neo4j data directory (~500MB) |
| Build steps in Dockerfile | 5 (models, schemas, path-map, embeddings, clusters) | 1 (graph pull + load) |

### What Gets Removed

- `DocumentStore` class and all JSON schema files
- `build-schemas`, `build-path-map`, `build-embeddings` Dockerfile steps
- `clusters build` Dockerfile step
- `.pkl` embedding cache files
- `.npz` cluster embedding files
- `clusters.json` file (in Docker context only — still used for graph population)
- `labels.json` cached LLM labels (labels live in graph, not version-controlled files)
- SQLite FTS index
- `SearchIndex` in-memory indices
- `SemanticSearch` numpy implementation
- `ClusterSearcher` numpy implementation
- `imas_codex.dd_version` module constant as the source of truth for version info

### What Stays

- **`Encoder` class**: needed to embed query text at runtime
- **`build-models`**: generated Python models for graph node types
- **All graph build CLI** (`imas build`, `clusters build/label/sync`): used on ITER cluster to populate the graph, not in Docker
- **`graph push/pull`**: used to move graph dumps to/from GHCR

## Version Management

The graph contains all 34 DD versions (3.22.0 → 4.1.0) with full version tracking infrastructure: `DDVersion` nodes chained by `PREDECESSOR`, every `IMASPath` linked to its introduction and deprecation versions, `IMASPathChange` nodes recording every metadata mutation between versions with semantic classification. This version intelligence must be the single source of truth — not the `imas-python` package version, not JSON metadata, not hardcoded constants.

### Version Reporting

The server's `/health` endpoint and `server_name` currently embed the `imas_codex.dd_version` constant. Replace this with graph-derived version metadata:

- **DD version range**: Query `DDVersion` nodes for min and max version IDs (e.g., `3.22.0 - 4.1.0`), showing the full scope of data available in the graph
- **Current version**: The `DDVersion` node with `is_current: true` identifies the latest version — use this as the primary version for default queries
- **Package version**: Keep the `imas-codex` package version as a separate field for debugging, but it should not be conflated with DD version

The overview tool should report both the version range and current version so clients understand that the server contains multi-version data.

### Version-Aware Queries

Currently no tools accept a version parameter — they operate against a single implicit version. The graph-native architecture enables version-scoped queries:

- **Path at version**: Fetch a path's documentation, units, and coordinates as they existed at a specific DD version. The graph's `INTRODUCED_IN`/`DEPRECATED_IN` relationships and `IMASPathChange` history make this a traversal, not a separate data store.
- **Evolution tracking**: Query the full change history of a path across versions — when it was introduced, what changed, whether sign conventions or coordinate systems shifted. `IMASPathChange` nodes with `SemanticChangeType` classification (sign_convention, coordinate_convention, definition_clarification) are already in the graph.
- **Version comparison**: Compare a path between two versions to surface what changed. This is a direct graph query pattern, not something that requires a specialized tool.
- **Deprecation/migration**: Follow `RENAMED_TO` chains to trace a path's lineage. The `PathMap` currently does this from JSON — the graph already has the same data as relationships.

These capabilities are best exposed through the read-only Cypher tool (below) rather than adding version parameters to every existing tool. The structured tools (search, path, list) should default to the current version. Version-specific queries are graph traversal — let agents compose them.

## Tool Architecture

The current MCP server exposes 6 rigid tools (`search_imas_paths`, `check_imas_paths`, `fetch_imas_paths`, `list_imas_paths`, `get_imas_overview`, `search_imas_clusters`, `get_imas_identifiers`) backed by `DocumentStore` and in-process search. The graph-native server should expose flexible access patterns that support codegen discovery, version evolution, and schema-driven query composition — not just a 1:1 port of existing tools with a different backend.

### Read-Only Cypher Tool

Add a read-only Cypher tool that lets agents compose arbitrary graph queries. The agentic server already has `query_neo4j` (smolagents) and `query()` in the python REPL — this brings the same capability to MCP clients, with safety constraints:

- Accept only read-only Cypher (`MATCH`, `RETURN`, `CALL`, `WITH`, `WHERE`, `ORDER BY`, `LIMIT`)
- Reject mutations (`CREATE`, `DELETE`, `SET`, `MERGE`, `REMOVE`, `DROP`)
- Return results as structured data with row limits to prevent context overflow
- No Cypher injection risk — Neo4j parameterized queries with agent-provided Cypher are read-only by design when mutations are blocked

This is the primary enabler of flexible codegen discovery. Instead of building specialized tools for every query pattern, agents use the schema tool to understand the graph structure, then compose Cypher queries for their specific needs. Example access patterns the Cypher tool enables without additional tooling:

- "What sign convention changes happened between DD v3 and v4?"
- "Show me all paths in equilibrium that were renamed since 3.42.0"
- "Which IDS gained the most new paths in 4.0.0?"
- "Find all paths with units that changed between versions"
- "What coordinate specifications exist for 2D profile data?"
- "List all paths that cross-reference psi in their coordinates"

### DD Graph Schema Tool

Add a tool that returns the IMAS DD graph schema — node types, properties, relationships, enums, and vector indexes. The agentic server already has `get_graph_schema()` which exposes the full schema (facility + DD). The MCP server version should expose only the DD portion.

Agents call this before composing Cypher queries to understand what's available. The schema is derived from the LinkML definitions (`imas_codex/schemas/imas_dd.yaml`) and loaded once at startup. This replaces agents needing to guess at node properties or relationship types.

The schema should include:
- All DD node types with their properties and types
- All DD relationship types with source → target directionality
- Enum values for `ChangeType`, `SemanticChangeType`, `DDDataType`, `DDNodeType`, `PhysicsDomain`
- Vector indexes available for semantic search
- Notes on version lifecycle relationships (`INTRODUCED_IN`, `DEPRECATED_IN`, `PREDECESSOR`, `RENAMED_TO`)

### Existing Tools — Graph Backend Swap

The existing structured tools (`search_imas_paths`, `check_imas_paths`, `fetch_imas_paths`, `list_imas_paths`, `get_imas_overview`, `search_imas_clusters`, `get_imas_identifiers`) are preserved with their current interfaces. Their backends change from `DocumentStore` to `GraphClient`, but the MCP contract stays the same. These tools default to the current DD version and serve the common case.

For version-specific or evolution queries, agents use the Cypher tool. This avoids polluting every existing tool with version parameters that most callers don't need.

## Data Preparation: IMAS-Only Graph on GHCR

The Docker image needs only the IMAS Data Dictionary portion of the graph — no facility data. The existing per-facility graph federation already supports filtering via `get_package_name()` and dump-and-clean. Extend this to treat `imas` as a pseudo-facility.

### IMAS-Only Graph Push/Pull

The `graph push` and `graph pull` commands already support `--facility` flags which produce packages like `imas-codex-graph-tcv`. Add an `--imas-only` flag (or treat `--facility imas` as special) that exports only DD nodes: `DDVersion`, `IDS`, `IMASPath`, `Unit`, `IMASCoordinateSpec`, `IdentifierSchema`, `IMASSemanticCluster`, `IMASPathChange`, `EmbeddingChange`.

The IMAS-only graph is ~500MB. The full graph with all facility data is significantly larger and growing. Separate IMAS-only packages on GHCR avoid pulling unnecessary facility data into every MCP server instance. The GHCR packages are private — the Docker build requires a `GHCR_TOKEN` (passed as a build secret) with `read:packages` scope to download the graph.

### Versioning

The server's health endpoint reports DD version metadata derived from the graph:
- **`dd_versions`**: The full version range present in the graph (e.g., `"3.22.0 - 4.1.0"`)
- **`dd_version_current`**: The version marked `is_current: true` (e.g., `"4.1.0"`)
- **`dd_version_count`**: Number of DD versions in the graph (e.g., `34`)
- **`imas_codex_version`**: The package version, for debugging

The `server_name` includes the current DD version for identification.

### Graph Build Workflow

The graph is built on the ITER cluster (or any environment with `imas-python` installed), then pushed to GHCR:

```
ITER cluster:
  imas-codex imas build              → populates Neo4j with DD structure + embeddings
  imas-codex imas clusters build     → HDBSCAN clustering from graph embeddings
  imas-codex imas clusters label     → LLM labels for clusters
  imas-codex imas clusters sync      → sync clusters + labels into graph
  imas-codex graph push --imas-only  → push IMAS-only dump to GHCR

Docker build:
  GHCR_TOKEN secret → authenticate
  graph pull --imas-only             → download IMAS-only dump
  neo4j-admin load                   → load into Neo4j data directory
  → FAIL build if pull fails
```

## Cluster Management: Incremental Updates

The current `_import_clusters` does a full `DETACH DELETE` of all `IMASSemanticCluster` nodes followed by a complete recreate from `clusters.json`. With persisted graphs, this is unnecessarily destructive.

Replace with incremental sync:
- Compare cluster IDs in `clusters.json` with existing graph nodes
- Create new clusters, update changed clusters (label, description, membership)
- Remove clusters that no longer exist
- Preserve cluster embeddings that haven't changed
- Remove the dependency on `labels.json` as a version-controlled file — labels are persisted in the graph on `IMASSemanticCluster.label` and `.description` properties

## Test-Driven Development

This migration benefits from a TDD approach because the boundaries are sharply defined: every tool has a clear input/output contract, every graph query has deterministic results given known data, and the graph schema is machine-readable.

### Why TDD Fits

- **Query contracts are testable**: Each graph query (semantic search, path lookup, version traversal) produces deterministic output from known graph state. Write the expected query results first, then implement the Cypher.
- **Tool interfaces are stable**: The MCP tool signatures are the contract. Write tests against the current tool responses, then swap the backend from `DocumentStore` to `GraphClient` while keeping tests green.
- **Schema is the spec**: The LinkML schema defines exactly what nodes, properties, and relationships exist. Tests validate graph content against the schema — if the schema changes, tests and models update together via `build-models`.
- **Version queries are pure functions**: Given a path and a version, the graph returns deterministic metadata. These are ideal candidates for parameterized tests across version ranges.
- **Agentic validation**: The Cypher tool accepts arbitrary queries — fuzz tests and safety tests (mutation rejection, result truncation) are critical and easy to write upfront.

### Test Infrastructure

Tests need a Neo4j instance loaded with known DD graph data. Options:
- **Graph service container in CI**: Pull the IMAS graph from GHCR into a Neo4j testcontainer. This enables data quality checks against real graph content and makes the existing `tests/graph/` test suite (structural, referential integrity, DD build) run in CI.
- **Fixture graphs**: For unit tests, use small hand-crafted graph fixtures with a few DDVersions and paths to test query logic without needing the full graph.

Write tests first for each phase:
- Phase 1: Tests that the IMAS-only graph contains exactly the expected DD node types
- Phase 2: Tests for Cypher tool safety (mutation rejection), schema tool completeness, version reporting
- Phase 3: Tests for embedded Neo4j lifecycle (startup, health, shutdown)
- Phase 5: Tests that each graph-backed tool produces equivalent results to the current file-backed tool for the same queries

## Key Steps

### Phase 0: Test Infrastructure ✅

**Status: Complete.** Mock GraphClient with query routing, graph-only test fixtures.

**What was built:**
- `conftest.py` mock `GraphClient` with Cypher pattern-based query routing (IDS, DDVersion, IMASPath count, connectivity)
- Integration tests rewritten for graph-only: `test_mcp_server.py`, `test_server_extended.py`, `test_health_endpoint.py`, `test_workflows.py`
- Tool tests updated: `test_tools.py` checks graph-backed tool properties
- Resource tests updated: old file-backed resource tests removed, `TestResourcesGraphOnly` added
- Health endpoint tests use mock GraphClient, verify graph-only fields
- 2286 tests passing (17 incremental cluster tests included)

### Phase 1: IMAS-Only Graph on GHCR ✅

**Status: Complete.** IMAS-only graph push/pull/switch fully operational.

**What was built:**
- `--imas-only` flag on `graph push`, `graph pull`, `graph fetch`, `graph export`
- `get_package_name(imas_only=True)` → `imas-codex-graph-imas` GHCR package
- Temp Neo4j filtering: loads full dump, deletes non-DD nodes, re-dumps via shared helpers (`_write_temp_neo4j_conf`, `_start_temp_neo4j`, `_stop_temp_neo4j`, `_dump_temp_neo4j`)
- Remote push delegates to `imas-codex graph export --imas-only` on iter (avoids 500MB SCP)
- `remote_check_imas_codex()` pre-flight check discovers CLI path on remote host
- `resolve_remote_service_name()` 3-step resolution: exact match → any `imas-codex-neo4j-*` → legacy
- `graph switch` works across named graphs with a single service (all bind `neo4j/` symlink)

**E2E validated:**
- Push: 477MB imas-only graph to `ghcr.io/simon-mcintosh/imas-codex-graph-imas`
- Pull: downloaded and loaded on iter
- Switch: `codex` ↔ `imas` bidirectional, auto-restarts Neo4j via any available service
- Content: 61,366 IMASPath + 9,287 IMASPathChange + 7,133 IMASSemanticCluster + 132 Unit + 87 IDS + 35 DDVersion + 8 IMASCoordinateSpec (zero facility nodes)

**DD node labels retained:** DDVersion, IDS, IMASPath, IMASCoordinateSpec, IMASSemanticCluster, IdentifierSchema, IMASPathChange, CoordinateRelationship, ClusterMembership, EmbeddingChange, Unit, PhysicsDomain, SignConvention

### Phase 2: Tool Architecture ✅

**Status: Complete.** All three tools operational and tested.

**What was built:**
- `CypherTool` (`imas_codex/tools/cypher_tool.py`): read-only Cypher with mutation keyword blocking (regex strips comments/strings before checking), result truncation to 200 rows, embedding vector stripping from results
- `SchemaTool` (`imas_codex/tools/schema_tool.py`): exposes DD-only LinkML schema (node types, properties, relationships, enums, vector indexes, version lifecycle notes) via `lru_cache`
- `VersionTool` (`imas_codex/tools/version_tool.py`): queries `DDVersion` nodes for current version, version range, count, and full version list
- Overview tool reports version range and lists all 10 MCP tools

### Phase 3: Embedded Neo4j in Docker ✅

**Status: Complete.** Multi-stage Dockerfile with embedded Neo4j fully operational.

**What was built:**
- 5-stage Dockerfile: uv binary → Neo4j source → builder (Python + graph pull) → graph-loader (neo4j-admin load) → final runtime
- ORAS CLI v1.2.0 pulls `imas-codex-graph-imas:latest` from private GHCR (requires `GHCR_TOKEN` build secret)
- Build FAILS if graph cannot be downloaded — no fallback
- `docker-entrypoint.sh`: starts Neo4j in background, waits for readiness, then starts MCP server; SIGTERM/SIGINT cleanup for both processes
- Neo4j configured for embedded use: internal bolt only (`127.0.0.1:7687`), auth disabled, 256-512MB heap
- HEALTHCHECK verifies both Neo4j (`7474`) and MCP server (`8000/health`)
- `docker-compose.yml` with GHCR_TOKEN secret, optional standalone Neo4j service, nginx reverse proxy

### Phase 4: Incremental Cluster Sync ✅

**Status: Complete.** `_import_clusters` uses MERGE-based incremental sync.

**What was built:**
- `_import_clusters` reads embeddings directly from graph (no file dependencies)
- HDBSCAN clustering produces content-hash IDs via `_compute_cluster_content_hash`
- `MERGE (n:IMASSemanticCluster {id: ...})` preserves existing labels/descriptions on unchanged clusters
- Stale cluster detection: `existing_ids - new_cluster_ids` → targeted `DETACH DELETE` only
- IN_CLUSTER relationships refreshed per-cluster (delete old → recreate)
- Cluster centroid embeddings computed in Neo4j via Cypher aggregation
- Cluster text embeddings (labels/descriptions) with per-cluster hash caching
- `_sync_labels_to_graph` in CLI writes labels via batched `UNWIND`
- No dependency on `labels.json` — labels persist in graph nodes

### Phase 5: Graph-Native Search ✅

**Status: Complete.** All 7 structured tools backed by GraphClient.

**What was built:**
- `GraphSearchTool`: vector index search via `db.index.vector.queryNodes('imas_path_embedding', k, $embedding)` with optional IDS filter, unit/physics_domain traversal
- `GraphPathTool`: check/fetch with `RENAMED_TO` migration chain detection, cluster membership, coordinate specs
- `GraphListTool`: path listing with `leaf_only` filter (excludes STRUCTURE/STRUCT_ARRAY types), IDS existence verification
- `GraphOverviewTool`: IDS listing from graph with physics domain aggregation, path count per IDS, version from DDVersion nodes
- `GraphClustersTool`: dual-mode search — path lookup via `IN_CLUSTER` traversal, semantic search via `cluster_description_embedding` vector index
- `GraphIdentifiersTool`: identifier schemas from graph
- `Encoder` class retained for query-time embedding only

### Phase 6: Cleanup ✅

**Status: Complete.** Clean break — server is graph-only.

**What was removed:**
- `DocumentStore`, `SearchIndex`, `SemanticSearch`, `ClusterSearcher` removed from server initialization
- All file-backed tool implementations removed from `Tools.__init__` and tool registration
- `build-schemas`, `build-path-map`, `build-embeddings`, `clusters build` removed from Dockerfile
- `IMAS_CODEX_GRAPH_NATIVE` env var toggle removed — no dual-mode
- `imas_codex.dd_version` no longer used as version source — comes from DDVersion graph nodes
- `ids://catalog`, `ids://structure/{ids_name}`, `ids://identifiers`, `ids://clusters` resources removed
- Only `examples://resource-usage` resource remains
- Health endpoint queries graph for stats (path count, IDS count)
- Server name derived from `DDVersion {is_current: true}` node

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Neo4j adds ~500MB to Docker image | Acceptable for a data-serving container. No hard Docker image size limit (GHCR/Docker Hub support multi-GB images) |
| Neo4j startup latency | Data loaded at build time; Neo4j starts in ~5s with warm page cache |
| GHCR pull fails during Docker build | Build fails — this is intentional. No silent fallback to degraded mode |
| GHCR_TOKEN not available | Build secret must be provided. Document required token scopes (`read:packages`). Use `gh auth token` or PAT |
| Private GHCR package access | Configure package visibility and token permissions. GitHub Actions can use `GITHUB_TOKEN` with appropriate scopes |
| JRE dependency adds image size | Multi-stage build: copy only JRE + Neo4j runtime, not build tools |
| Cypher tool injection/mutation | Blocklist mutation keywords before execution. Read-only Neo4j user as defense in depth |
| Version queries return too much data | Default to current version in structured tools. Cypher tool has result row limits |

## Non-Goals

- Running Neo4j as a separate service (embedded is simpler for MCP)
- Supporting graph writes from the MCP server (read-only)
- Backward compatibility with file-based deployments (clean break)
- Adding version parameters to every existing tool (use Cypher tool for version-specific queries)

## Remaining Gaps

All six phases are complete. The following gaps remain between the plan's aspirations and the current implementation:

### Minor Gaps (polish, not blocking)

| Gap | Plan Expectation | Current State | Priority |
|-----|------------------|---------------|----------|
| Full-text index | Neo4j FTS index for keyword search (replaces SQLite FTS5) | `search_imas_paths` uses vector search only — no full-text index fallback for exact keyword matches | Low — vector search handles most cases; agents use `query_imas_graph` for exact STARTS WITH queries |
| Health endpoint version detail | Plan specifies `dd_versions` (range), `dd_version_current`, `dd_version_count` as separate fields | Health returns `imas_dd_version` (derived from server name) + `ids_count` + `path_count` | Low — `get_dd_versions` tool provides full version metadata |
| Read-only Neo4j user | Defense in depth: dedicated read-only Neo4j user | Auth disabled (`dbms.security.auth_enabled=false`) in embedded Docker config | Low — internal bolt only, Cypher mutation blocking is enforced at tool level |
| CI graph service container | Graph testcontainer in CI for integration tests | Tests use mock GraphClient — no real Neo4j in CI | Medium — would enable data quality tests against real graph content |
| `search_mode` parameter | Plan mentions `auto`, `semantic`, `lexical`, `hybrid` modes | Graph search always uses vector (semantic) — `lexical` and `hybrid` modes don't route to a full-text index | Low — vector search is the primary mode; `search_mode` parameter accepted but not differentiated |

### Not Gaps (deliberate deviations)

- **No `labels.json` in Docker**: Plan said to remove from Docker context. Done — labels persist in graph.
- **No equivalence test harness**: Plan suggested running both file-backed and graph-backed tools side-by-side. Unnecessary after clean break — file-backed code is gone.
- **No version parameters on structured tools**: Plan explicitly said to use Cypher tool for version-specific queries. Correct — `query_imas_graph` handles all version traversal.

## Phase 7: Richer Content via Multi-Index Search + Graph Traversal

With the graph-native foundation complete, the MCP tools can now exploit the full richness of the knowledge graph. The current tools perform single-index queries — one vector search or one Cypher traversal per request. The graph enables **compositional queries** that combine semantic similarity across multiple indexes with relationship traversal for context-rich responses.

### Multi-Index Semantic Search

The graph contains multiple vector indexes, each embedding different aspects of the data:

| Index | Node Type | Property | Content Embedded |
|-------|-----------|----------|-----------------|
| `imas_path_embedding` | IMASPath | embedding | Path documentation (physics meaning) |
| `cluster_description_embedding` | IMASSemanticCluster | description_embedding | Cluster description (thematic grouping) |
| `imas_path_name_embedding` | IMASPath | name_embedding | Path name (structural/naming pattern) |

**Proposed: `search_imas_deep`** — a new tool that fans out across multiple indexes and merges results:

```python
@mcp_tool("Deep semantic search across IMAS paths, clusters, and related data. ...")
async def search_imas_deep(
    self,
    query: str,
    include_clusters: bool = True,
    include_version_context: bool = False,
    max_results: int = 20,
) -> DeepSearchResult:
    """Fan-out search across path + cluster indexes, then enrich via traversal."""
    embedding = self._embed_query(query)

    # 1. Vector search on path embeddings (primary)
    path_hits = self._gc.query("""
        CALL db.index.vector.queryNodes('imas_path_embedding', $k, $embedding)
        YIELD node AS path, score
        OPTIONAL MATCH (path)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (path)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        OPTIONAL MATCH (path)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        RETURN path.id, path.documentation, path.physics_domain,
               u.id AS units, collect(DISTINCT c.label) AS clusters,
               collect(DISTINCT coord.id) AS coordinates, score
        ORDER BY score DESC LIMIT $k
    """, embedding=embedding, k=max_results)

    # 2. Vector search on cluster descriptions (discover related groups)
    if include_clusters:
        cluster_hits = self._gc.query("""
            CALL db.index.vector.queryNodes('cluster_description_embedding', $k, $embedding)
            YIELD node AS cluster, score
            WHERE score > 0.7
            OPTIONAL MATCH (p:IMASPath)-[:IN_CLUSTER]->(cluster)
            RETURN cluster.label, cluster.description, cluster.scope,
                   collect(p.id)[..10] AS sample_paths, score
            ORDER BY score DESC LIMIT 5
        """, embedding=embedding, k=10)

    # 3. Version context enrichment (optional)
    if include_version_context:
        # For top hits, check version evolution
        top_path_ids = [h['path.id'] for h in path_hits[:5]]
        version_info = self._gc.query("""
            UNWIND $paths AS pid
            MATCH (p:IMASPath {id: pid})
            OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
            OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dep:DDVersion)
            OPTIONAL MATCH (change:IMASPathChange)-[:FOR_IMAS_PATH]->(p)
            RETURN p.id, intro.id AS introduced_in, dep.id AS deprecated_in,
                   count(change) AS change_count
        """, paths=top_path_ids)
```

### Graph-Augmented Search Patterns

Beyond multi-index search, the graph enables **traversal-based enrichment** that file-backed tools couldn't do:

#### 1. Coordinate Cross-Referencing
When a user asks about a measurement, show what coordinate systems it expects:
```cypher
CALL db.index.vector.queryNodes('imas_path_embedding', 5, $embedding)
YIELD node AS path, score
MATCH (path)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
MATCH (coord)<-[:HAS_COORDINATE]-(sibling:IMASPath)
WHERE sibling <> path AND sibling.ids = path.ids
RETURN path.id, coord.id, collect(DISTINCT sibling.id) AS shared_coordinate_paths, score
```

#### 2. Cross-IDS Discovery
Find paths in other IDS that share the same physics domain and coordinate space:
```cypher
CALL db.index.vector.queryNodes('imas_path_embedding', 5, $embedding)
YIELD node AS path, score
MATCH (path)-[:IN_CLUSTER]->(c:IMASSemanticCluster {cross_ids: true})
MATCH (sibling)-[:IN_CLUSTER]->(c)
WHERE sibling.ids <> path.ids
RETURN path.id AS source, path.ids AS source_ids,
       collect(DISTINCT {path: sibling.id, ids: sibling.ids})[..10] AS cross_ids_matches, score
```

#### 3. Version Evolution Enrichment
For codegen workflows, show whether a path was recently introduced or has had sign convention changes:
```cypher
MATCH (p:IMASPath {id: $path})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(v:DDVersion)
OPTIONAL MATCH (change:IMASPathChange)-[:FOR_IMAS_PATH]->(p)
WHERE change.semantic_change_type IN ['sign_convention', 'coordinate_convention']
RETURN p.id, v.id AS introduced_version,
       collect({version: change.version, type: change.semantic_change_type, summary: change.summary}) AS breaking_changes
```

#### 4. Unit-Aware Search
Group results by physical unit, useful for checking dimensional consistency:
```cypher
CALL db.index.vector.queryNodes('imas_path_embedding', 20, $embedding)
YIELD node AS path, score
MATCH (path)-[:HAS_UNIT]->(u:Unit)
RETURN u.id AS unit, collect({path: path.id, score: score})[..5] AS paths
ORDER BY size(paths) DESC
```

### Proposed New Tools

| Tool | Purpose | Key Capability |
|------|---------|---------------|
| `search_imas_deep` | Multi-index fan-out search | Combines path + cluster vector search with traversal enrichment |
| `compare_dd_versions` | Version diff between two DD versions | Surfaces new, removed, renamed, and semantically changed paths |
| `get_imas_path_context` | Full context for a single path | Unit, coordinates, cluster membership, version history, related paths |
| `search_imas_by_unit` | Unit-based path discovery | Find all paths with a given physical unit or dimensionality |

### Implementation Notes

- All new tools use the existing `CypherTool` patterns — read-only, graph-backed, result-limited
- Multi-index queries should run in parallel where possible (fan out to multiple vector indexes, merge results)
- `search_imas_deep` replaces `search_imas_paths` as the primary search entry point for agents that want rich context
- `search_imas_paths` remains for lightweight, fast searches that return minimal metadata
- The Cypher tool already enables all of these queries ad-hoc — the new tools **package common patterns** into ergonomic interfaces
