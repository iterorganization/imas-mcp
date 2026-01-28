# Graph Operations

Knowledge graph management with Neo4j - queries, schema evolution, and releases.

This agent builds on the core rules in [AGENTS.md](../AGENTS.md). The main file covers graph backup, token optimization, and schema enforcement. This file covers graph-specific details.

## MCP Tools (Primary Interface)

When the Codex MCP server is running, use `python()` REPL:

```python
# Query the graph (always project properties, never RETURN n)
python("result = query('MATCH (f:Facility) RETURN f.id, f.name'); print(result)")

# Get schema for Cypher generation
python("schema = get_graph_schema(); print(schema['node_labels'])")

# Semantic search
python("hits = semantic_search('plasma current', 'code_chunk_embedding', 5); print(hits)")

# Add nodes with validation
python("""
add_to_graph("Facility", [
    {"id": "tcv", "name": "EPFL-SPC", "machine": "TCV"}
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

- New class (node label): Safe - old dumps load fine
- New property: Safe - missing = `null`
- Rename property: Unsafe - requires migration
- Remove class: Risky - old data orphaned

Policy: Never rename/remove, just add.

## LLM-First Cypher Queries

Generate Cypher directly. Use `UNWIND` for batch operations:

```python
python("""
tools = [{"id": "tcv:gcc", "name": "gcc", "available": True, "version": "11.5.0"}]
query('''
    UNWIND $tools AS tool
    MERGE (t:Tool {id: tool.id})
    SET t += tool
    WITH t
    MATCH (f:Facility {id: 'tcv'})
    MERGE (t)-[:FACILITY_ID]->(f)
''', tools=tools)
""")
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

## Vector Indexes

Available for semantic search:

| Index | Content |
|-------|---------|
| `imas_path_embedding` | IMAS Data Dictionary paths (61k) |
| `code_chunk_embedding` | Code examples (8.5k chunks) |
| `wiki_chunk_embedding` | Wiki documentation (25k chunks) |
| `cluster_centroid` | Semantic clusters |

## Useful Cypher Aggregations

Use Cypher aggregations instead of Python post-processing: `count()`, `collect()`, `sum()`, `avg()`, `min()`, `max()`, `size()`.

```python
# Combine related queries with collect()
query("""
MATCH (f:SourceFile)
WHERE f.id CONTAINS 'CHEASE'
WITH f.status AS status, collect(f.path)[..5] AS sample_paths
RETURN status, size(sample_paths) AS count, sample_paths
""")
```
