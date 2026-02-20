# Graph-Native MCP Server

## Motivation

The IMAS MCP server currently loads all data from flat files at startup: JSON schemas, pickle embedding caches, and `.npz` cluster embedding files. Semantic search runs in-process using numpy dot products against in-memory matrices. This works but has three problems:

1. **Duplicate storage.** Embeddings exist in the Neo4j graph (on `IMASPath.embedding`, `IMASSemanticCluster.embedding`, `.label_embedding`, `.description_embedding`) AND as local `.npz`/`.pkl` files. Every build must export from graph to file, and every server instance must reload files back into memory.

2. **No link traversal.** The current search returns isolated results — paths or clusters ranked by cosine similarity. It cannot follow graph relationships (`HAS_PARENT`, `IN_CLUSTER`, `HAS_COORDINATE`, `MAPS_TO_IMAS`, `HAS_ERROR`) to provide richer context. A graph-native approach would let a single query combine vector similarity with structure traversal (e.g., "find clusters about electron density and return their member paths with coordinate specs and unit relationships").

3. **Brittle CI fallback chain.** The server needs `.pkl`, `.npz`, and `clusters.json` files to exist at specific paths. Docker builds generate these during image construction. If any step fails or files drift out of sync, the server silently degrades or crashes. A graph-native approach replaces this with a single data source: a populated Neo4j instance.

## Architecture

### Current Flow

```
Docker build:
  build-schemas → JSON files
  build-embeddings → .pkl cache + in-memory numpy
  clusters build → clusters.json + .npz

Server startup:
  DocumentStore ← JSON files
  Embeddings ← .pkl cache → numpy matrix
  SemanticSearch ← numpy dot products
  ClusterSearcher ← clusters.json + .npz → numpy dot products
```

### Target Flow

```
Docker build:
  build-schemas → JSON files (kept for DocumentStore)
  graph pull → load Neo4j dump from GHCR OCI artifact

Server startup:
  DocumentStore ← JSON files (unchanged — used for full-text + path lookup)
  SemanticSearch ← Neo4j vector index queries
  ClusterSearcher ← Neo4j vector index queries + link traversal
```

### What Changes

| Component | Current | Target |
|-----------|---------|--------|
| Path semantic search | numpy dot product over .pkl matrix | `db.index.vector.queryNodes('imas_path_embedding', k, $vec)` |
| Cluster semantic search | numpy dot product over .npz centroids | `db.index.vector.queryNodes('cluster_description_embedding', k, $vec)` |
| Cluster label search | numpy dot product over .npz label embeddings | `db.index.vector.queryNodes('cluster_label_embedding', k, $vec)` |
| Cluster path lookup | in-memory dict index from clusters.json | `MATCH (p:IMASPath)-[:IN_CLUSTER]->(c:IMASSemanticCluster)` |
| Search + context | separate tool calls with no linking | single Cypher query: vector search → traverse relationships |
| Docker image | no Neo4j, files only | embedded Neo4j with pre-loaded graph dump |
| `.pkl` / `.npz` files | required at runtime | eliminated from server path |

### What Stays

- **DocumentStore + JSON schemas**: retained for full-text search, path validation, and IDS catalog. These are small, fast, and have no graph equivalent yet.
- **`build-schemas` + `build-path-map`**: still needed for DocumentStore and version upgrade mappings.
- **Encoder class**: still needed to encode query text into embedding vectors at query time.
- **`build-embeddings` + `clusters build`**: still needed for graph population on ITER cluster, but no longer needed in Docker image.

## Key Steps

### Phase 1: Embedded Neo4j in Docker

Add a Neo4j Community instance to the MCP server Docker image. The Dockerfile pulls a pre-built graph dump from GHCR and loads it at image build time. At runtime, Neo4j starts alongside the MCP server as a supervised process.

- Add Neo4j to the Docker image (JRE + neo4j-community tarball, or a multi-stage copy from the official image)
- Add a process supervisor (e.g., a shell entrypoint or `supervisord`) to start Neo4j and the MCP server
- Pull the graph dump during Docker build via `graph pull` or `oras pull`
- Load the dump via `neo4j-admin database load` at build time
- Expose an internal bolt port (not published externally) for the MCP server
- Add health check that verifies both Neo4j and the MCP server are ready

### Phase 2: GraphClient in the MCP Server

Make `GraphClient` available inside the MCP server process. Currently `GraphClient` is only used by CLI commands and the agentic REPL — the server tools have no graph dependency.

- Add a `GraphClient` singleton to `Server.__post_init__` that connects to the local Neo4j
- Pass the client to `Tools` so individual tool classes can use it
- Add a graph readiness check to the `/health` endpoint
- Handle graceful degradation if Neo4j is unavailable (fall back to file-based search)

### Phase 3: Graph-Native Semantic Search

Replace the numpy-based `SemanticSearch` with Neo4j vector index queries. This eliminates the `.pkl` embedding cache from the server path.

- Add a `GraphSemanticSearch` class that wraps `db.index.vector.queryNodes()`
- Use the existing `imas_path_embedding` vector index
- Encode query text with `Encoder.embed_texts()` (same as today)
- Return the same `SemanticSearchResult` interface so callers don't change
- Remove `.pkl` loading from `Embeddings` when graph is available
- Combine vector results with relationship traversal for enriched context

### Phase 4: Graph-Native Cluster Search

Replace `ClusterSearcher` (which loads `.npz` + `clusters.json`) with graph queries. This eliminates the `.npz` file and the in-memory cluster index.

- Query `cluster_description_embedding` vector index for NL cluster search
- Query `cluster_label_embedding` vector index as an alternative ranking signal
- Replace path→cluster lookup with `MATCH (p:IMASPath)-[:IN_CLUSTER]->(c)`
- Replace IDS→cluster lookup with `MATCH (c:IMASSemanticCluster) WHERE $ids IN c.ids_names`
- Return the same `ClusterSearchResult` interface
- Add link traversal to enrich results (member paths, coordinates, units)

### Phase 5: Cleanup

Remove dead code paths and file-based fallbacks that are no longer needed.

- Remove `.npz` export from `_import_clusters` and `_export_cluster_embeddings_npz`
- Remove `ClusterSearcher._load_embeddings()` .npz loading path
- Remove embedding `.pkl` cache generation from Docker build
- Remove `build-embeddings` step from Dockerfile
- Remove `clusters build` step from Dockerfile (graph dump already has clusters)
- Simplify `Embeddings` class — in graph mode it only needs the encoder for query embedding
- Update CI to use graph-based tests where appropriate

### Phase 6: Enriched Queries (Future Extension)

With the graph available, compose richer queries that combine vector similarity with structure traversal.

- "Find paths similar to X and show their coordinates and units"
- "Find clusters about Y and return member paths with their IDS descriptions"
- "Find paths that changed between DD versions and are semantically similar to Z"
- Cross-facility queries when facility graphs are also loaded

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Neo4j increases Docker image size (~500MB) | Use community edition, strip docs/samples; compare against current .pkl+.npz size |
| Neo4j startup latency delays server readiness | Load dump at build time; Neo4j starts in ~5s with warm data |
| Breaking change for existing Docker deployments | Phased rollout; keep file-based fallback in Phase 2-3 until Phase 5 |
| JRE dependency in slim Python image | Multi-stage build: copy Neo4j + JRE from official image |
| CI tests currently don't have Neo4j | CI can pull the graph dump in a service container, or keep file-based tests as unit tests |

## Non-Goals

- Running Neo4j as a separate service in production (the embedded approach is simpler for the MCP use case)
- Replacing DocumentStore with graph queries (full-text search on JSON schemas is fast and sufficient)
- Supporting graph writes from the MCP server (read-only access to a pre-built graph)
