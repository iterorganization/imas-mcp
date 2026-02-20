# Graph-Native MCP Server

## Motivation

The IMAS MCP server currently loads all data from flat JSON files at startup, runs semantic search via in-process numpy dot products, and builds embeddings and clusters inside the Docker image. This creates duplicate storage (graph + files), a brittle build chain (`.pkl`, `.npz`, `clusters.json`, JSON schemas), and no ability to traverse graph relationships at query time.

The graph already contains every data point the server needs: `IMASPath` nodes with documentation, units, coordinates, physics domains, embeddings, and cluster memberships; `IMASSemanticCluster` nodes with label/description embeddings; `IDS`, `Unit`, `IMASCoordinateSpec`, and `IdentifierSchema` nodes. The `DocumentStore` and its JSON files are a redundant read-only projection of this graph.

This plan replaces all file-based data with a single Neo4j instance embedded in the Docker image, loaded from a pre-built IMAS-only graph pulled from GHCR. JSON files, pickle caches, and numpy arrays are eliminated entirely. This is a clean break — no backward compatibility with file-based deployments.

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
| Version info | JSON metadata + package version | `MATCH (v:DDVersion {is_current: true})` |
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

### What Stays

- **`Encoder` class**: needed to embed query text at runtime
- **`build-models`**: generated Python models for graph node types
- **All graph build CLI** (`imas build`, `clusters build/label/sync`): used on ITER cluster to populate the graph, not in Docker
- **`graph push/pull`**: used to move graph dumps to/from GHCR

## Data Preparation: IMAS-Only Graph on GHCR

The Docker image needs only the IMAS Data Dictionary portion of the graph — no facility data. The existing per-facility graph federation already supports filtering via `get_package_name()` and dump-and-clean. Extend this to treat `imas` as a pseudo-facility.

### IMAS-Only Graph Push/Pull

The `graph push` and `graph pull` commands already support `--facility` flags which produce packages like `imas-codex-graph-tcv`. Add an `--imas-only` flag (or treat `--facility imas` as special) that exports only DD nodes: `DDVersion`, `IDS`, `IMASPath`, `Unit`, `IMASCoordinateSpec`, `IdentifierSchema`, `IMASSemanticCluster`, `IMASPathChange`, `EmbeddingChange`.

The IMAS-only graph is ~500MB. The full graph with all facility data is significantly larger and growing. Separate IMAS-only packages on GHCR avoid pulling unnecessary facility data into every MCP server instance. The GHCR packages are private — the Docker build requires a `GHCR_TOKEN` (passed as a build secret) with `read:packages` scope to download the graph.

### Versioning

The server version displayed by `/health` and `server_name` should be the IMAS DD version of the backing graph data, not the `imas-codex` package version. Read from `DDVersion {is_current: true}` at startup. The package version can be included as a separate field for debugging.

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

## Key Steps

### Phase 1: IMAS-Only Graph on GHCR

Extend `graph push/pull` to support IMAS-only exports. The filtering logic already exists for per-facility graphs — apply the same dump-and-clean approach but keep only DD node types.

### Phase 2: Embedded Neo4j in Docker

Add Neo4j Community to the Docker image. Pull the IMAS-only graph during Docker build. The image must fail to build if the graph cannot be downloaded (no fallback to file-based data).

- Multi-stage build: copy Neo4j + JRE from official image
- `oras pull` the IMAS-only graph archive from private GHCR (requires `GHCR_TOKEN` build secret)
- `neo4j-admin database load` at build time
- Process supervisor (shell entrypoint) to start Neo4j + MCP server at runtime
- Internal bolt port only — not exposed externally
- Health check verifies both Neo4j and MCP server

### Phase 3: GraphClient in Server + Graph-Native Search

Replace `DocumentStore`, `SemanticSearch`, and `ClusterSearcher` with `GraphClient` queries.

- `GraphClient` singleton in `Server.__post_init__`
- Vector index queries for semantic search (paths + clusters)
- Neo4j full-text index for keyword search (replaces SQLite FTS5)
- Cypher for path lookup, IDS listing, identifier schemas
- `DDVersion.is_current` for server version
- Enriched queries combining vector similarity with relationship traversal

### Phase 4: Incremental Cluster Sync

Replace the delete-all-recreate pattern in `_import_clusters` with a diff-based merge. Labels and descriptions persist in graph — no more `labels.json`.

### Phase 5: Cleanup

Remove all file-based data paths from the server. Remove `build-schemas`, `build-embeddings`, `build-path-map`, and `clusters build` steps from the Dockerfile. Remove `DocumentStore`, pickle/numpy loaders, `labels.json`, and the `definitions/clusters/` directory from the Docker context.

### Future: Graph-Based CI Testing

Use a Neo4j service container in CI that pulls the IMAS graph from GHCR. This enables data quality checks against real graph content (embedding coverage, cluster integrity, relationship completeness). Not critical for the current implementation.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Neo4j adds ~500MB to Docker image | Acceptable for a data-serving container. No hard Docker image size limit (GHCR/Docker Hub support multi-GB images) |
| Neo4j startup latency | Data loaded at build time; Neo4j starts in ~5s with warm page cache |
| GHCR pull fails during Docker build | Build fails — this is intentional. No silent fallback to degraded mode |
| GHCR_TOKEN not available | Build secret must be provided. Document required token scopes (`read:packages`). Use `gh auth token` or PAT |
| Private GHCR package access | Configure package visibility and token permissions. GitHub Actions can use `GITHUB_TOKEN` with appropriate scopes |
| JRE dependency adds image size | Multi-stage build: copy only JRE + Neo4j runtime, not build tools |

## Non-Goals

- Running Neo4j as a separate service (embedded is simpler for MCP)
- Supporting graph writes from the MCP server (read-only)
- Backward compatibility with file-based deployments (clean break)
- Graph-based CI testing (future work, not blocking)
