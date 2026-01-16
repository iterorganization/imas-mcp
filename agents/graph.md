# Graph Operations

Knowledge graph management with Neo4j - queries, schema evolution, and releases.

## MCP Tools (Primary Interface)

When the Codex MCP server is running, use `python()` REPL:

```python
# Query the graph (always project properties, never RETURN n)
python("result = query('MATCH (f:Facility) RETURN f.id, f.name'); print(result)")

# Get schema for Cypher generation
python("schema = get_graph_schema(); print(schema['node_labels'])")

# Semantic search
python("hits = semantic_search('plasma current', 'code_chunk_embedding', 5); print(hits)")

# Ingest nodes with validation
python("""
ingest_nodes("Facility", [
    {"id": "epfl", "name": "EPFL-SPC", "machine": "TCV"}
])
""")
```

## Neo4j Service Management

Neo4j runs as a systemd user service (auto-starts on login):

```bash
# Check status
uv run imas-codex neo4j status

# Service management
uv run imas-codex neo4j service status
systemctl --user restart imas-codex-neo4j
journalctl --user -u imas-codex-neo4j -f

# Interactive Cypher shell
uv run imas-codex neo4j shell

# Neo4j Browser
open http://localhost:7474
```

## Graph Backup and Restore

Always dump before destructive operations:

```bash
# Dump current graph
uv run imas-codex neo4j dump

# Load a dump
uv run imas-codex neo4j load graph.dump

# Pull latest from GHCR
uv run imas-codex neo4j pull

# Push to GHCR (requires GHCR_TOKEN)
uv run imas-codex neo4j push v1.0.0
```

## Schema Evolution

Neo4j is schema-optional. Additive changes only:

| Change | Safe? | Notes |
|--------|-------|-------|
| New class (node label) | ✅ | Old dumps load fine |
| New property | ✅ | Missing = `null` |
| Rename property | ❌ | Requires migration |
| Remove class | ⚠️ | Old data orphaned |

Policy: Never rename/remove, just add.

## LLM-First Cypher Queries

Generate Cypher directly. Use `UNWIND` for batch operations:

```python
python("""
tools = [{"id": "epfl:gcc", "name": "gcc", "available": True, "version": "11.5.0"}]
query('''
    UNWIND $tools AS tool
    MERGE (t:Tool {id: tool.id})
    SET t += tool
    WITH t
    MATCH (f:Facility {id: 'epfl'})
    MERGE (t)-[:FACILITY_ID]->(f)
''', tools=tools)
""")
```

## _GraphMeta Node

Every graph has a metadata node:

```cypher
MATCH (m:_GraphMeta) RETURN m
-- Returns: {version: "1.0.0", message: "Add EPFL", facilities: ["epfl"]}
```

## Release Workflow

```bash
# 1. Prepare PR (pushes tag to fork)
uv run imas-codex release v4.0.0 -m 'Release message' --remote origin

# 2. Create PR on GitHub, merge to upstream

# 3. Sync with upstream
git pull upstream main

# 4. Finalize release (graph to GHCR, tag to upstream)
uv run imas-codex release v4.0.0 -m 'Release message'
```

Options:

```bash
# Preview changes
uv run imas-codex release v4.0.0 -m 'Test' --dry-run

# Skip graph operations
uv run imas-codex release v4.0.0 -m 'Code only' --skip-graph

# Skip git tag
uv run imas-codex release v4.0.0 -m 'Graph only' --skip-git
```

## Schema Principles

1. **Schema-first**: All data models defined in LinkML YAML
2. **Auto-generate**: Pydantic models via `uv run build-models --force`
3. **Runtime introspection**: `GraphSchema` derives structure from LinkML
4. **No hard-coded duplication**: All graph structure comes from schema

## Key Files

| File | Purpose |
|------|---------|
| `schemas/facility.yaml` | LinkML schema (source of truth) |
| `imas_codex/graph/models.py` | Generated Pydantic models |
| `imas_codex/graph/schema.py` | Runtime schema introspection |
| `imas_codex/graph/client.py` | Neo4j operations |

## Vector Indexes

Available for semantic search:

| Index | Content |
|-------|---------|
| `imas_path_embedding` | IMAS Data Dictionary paths (61k) |
| `code_chunk_embedding` | Code examples (8.5k chunks) |
| `wiki_chunk_embedding` | Wiki documentation (25k chunks) |
| `cluster_centroid` | Semantic clusters |

## Token Cost Optimization

Never return full nodes with `RETURN n` - always project specific properties.

Nodes like `IMASPath` and `CodeChunk` contain 384-dimension embedding vectors (~2k tokens each). Returning full nodes wastes tokens and increases API costs.

```python
# BAD - returns embeddings (~2k tokens per node)
query("MATCH (n:IMASPath) WHERE n.name CONTAINS 'temperature' RETURN n LIMIT 10")

# GOOD - project only needed properties (~50 tokens per node)
query("MATCH (n:IMASPath) WHERE n.name CONTAINS 'temperature' RETURN n.id, n.name, n.documentation LIMIT 10")

# GOOD - use labels() and properties selectively
query("MATCH (n) WHERE n.path CONTAINS '/sauter/' RETURN labels(n) as type, n.id, n.path")
```

Cost impact: returning 10 nodes with embeddings costs ~$0.30 extra per query.

## Prefer Cypher Aggregations Over Python Post-Processing

Each tool call has overhead (~$0.01-0.05). Use Cypher's built-in aggregations instead of fetching raw data and processing in Python.

```python
# BAD - multiple calls, Python aggregation
files = query("MATCH (f:SourceFile) WHERE f.id CONTAINS 'CHEASE' RETURN f.status, f.language")
# Then Counter(f['status'] for f in files) in another call

# GOOD - single call, Cypher aggregation
query("""
MATCH (f:SourceFile)
WHERE f.id CONTAINS 'CHEASE'
RETURN f.status AS status, f.language AS lang, count(*) AS count
ORDER BY count DESC
""")

# GOOD - combine related queries with collect()
query("""
MATCH (f:SourceFile)
WHERE f.id CONTAINS 'CHEASE'
WITH f.status AS status, collect(f.path)[..5] AS sample_paths
RETURN status, size(sample_paths) AS count, sample_paths
""")
```

Useful Cypher aggregations: `count()`, `collect()`, `sum()`, `avg()`, `min()`, `max()`, `size()`.
