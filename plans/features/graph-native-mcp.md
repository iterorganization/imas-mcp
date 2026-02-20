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

### Phase 0: Test Infrastructure

Set up graph test fixtures and CI integration before starting the migration. This phase runs throughout and supports all subsequent phases.

- Graph service container configuration for CI
- Small fixture graphs for unit testing version queries and tool contracts
- Equivalence test harness: run the same tool request against both file-backed and graph-backed implementations, assert matching results
- Baseline coverage of existing tool responses to use as regression targets

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

### Phase 2: Tool Architecture

Add the read-only Cypher tool, DD graph schema tool, and version reporting — these can be built and tested against the current local Neo4j graph before the Docker embedding work.

- Read-only Cypher tool with mutation blocking and result truncation
- DD graph schema tool exposing LinkML-derived schema for the IMAS DD portion of the graph
- Version metadata reporting: query DDVersion nodes for range, current, and count
- Update overview tool to report version range and available graph capabilities

### Phase 3: Embedded Neo4j in Docker

Add Neo4j Community to the Docker image. Pull the IMAS-only graph during Docker build. The image must fail to build if the graph cannot be downloaded (no fallback to file-based data).

- Multi-stage build: copy Neo4j + JRE from official image
- `oras pull` the IMAS-only graph archive from private GHCR (requires `GHCR_TOKEN` build secret)
- `neo4j-admin database load` at build time
- Process supervisor (shell entrypoint) to start Neo4j + MCP server at runtime
- Internal bolt port only — not exposed externally
- Health check verifies both Neo4j and MCP server

### Phase 4: Incremental Cluster Sync

Replace the delete-all-recreate pattern in `_import_clusters` with a diff-based merge. Labels and descriptions persist in graph — no more `labels.json`. This phase is independent and can be done at any point.

### Phase 5: Graph-Native Search

Replace `DocumentStore`, `SemanticSearch`, and `ClusterSearcher` with `GraphClient` queries. Run the equivalence test harness to validate.

- `GraphClient` singleton in `Server.__post_init__`
- Vector index queries for semantic search (paths + clusters)
- Neo4j full-text index for keyword search (replaces SQLite FTS5)
- Cypher for path lookup, IDS listing, identifier schemas
- Version-aware search defaults (current version for structured tools)
- Enriched queries combining vector similarity with relationship traversal

### Phase 6: Cleanup

Remove all file-based data paths from the server. Remove `build-schemas`, `build-embeddings`, `build-path-map`, and `clusters build` steps from the Dockerfile. Remove `DocumentStore`, pickle/numpy loaders, `labels.json`, and the `definitions/clusters/` directory from the Docker context. Remove `imas_codex.dd_version` as a module-level constant — version comes from the graph.

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
