# Knowledge Graph Architecture

> **Module**: `imas_codex.graph`

## Overview

The imas-codex knowledge graph is a Neo4j-based store that unifies:
- **Facility knowledge**: TreeNodes, CodeChunks, Diagnostics, Analysis Codes
- **IMAS Data Dictionary**: DDPath nodes with version tracking and embeddings

All schema definitions live in **LinkML** (`schemas/*.yaml`) as the single source of truth.

## Architecture

```
BUILD TIME (slow, run once)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐
│  LinkML      │───▶│  Pydantic    │───▶│  Generate    │───▶│  Populate │
│  Schemas     │    │  Models      │    │  Embeddings  │    │   Neo4j   │
└──────────────┘    └──────────────┘    └──────────────┘    └───────────┘

QUERY TIME (fast, every request)
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐
│   Agent /    │───▶│    Graph     │───▶│           Neo4j              │
│  MCP Tool    │    │    Client    │    │  - Facility nodes            │
│              │    │   (neo4j     │    │  - IMAS DD nodes             │
└──────────────┘    │    driver)   │    │  - Vector indexes            │
                    └──────────────┘    └──────────────────────────────┘
```

## Schema Management

### LinkML as Single Source of Truth

```
imas_codex/schemas/
├── common.yaml      # Shared enums, base types
├── facility.yaml    # Facility-side nodes (TreeNode, CodeChunk, etc.)
└── imas_dd.yaml     # IMAS DD nodes (DDPath, DDVersion, etc.)
```

### Schema → Graph Workflow

```python
from imas_codex.graph import get_schema

schema = get_schema()
print(schema.node_labels)  # ['Facility', 'MDSplusTree', 'TreeNode', ...]
```

## Graph Client

```python
from imas_codex.graph import GraphClient

with GraphClient() as client:
    result = client.query("MATCH (n:Facility) RETURN n.id")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PASSWORD` | `imas-codex` | Database password |

## Node Types

### Facility Data (facility.yaml)

| Node | Purpose | Key Properties |
|------|---------|----------------|
| `Facility` | Fusion facility | id, name, machine |
| `MDSplusTree` | MDSplus tree | name, facility_id |
| `TreeNode` | Tree node | path, units, description |
| `CodeChunk` | Code snippet | content, embedding |
| `TDIFunction` | TDI function | name, signature |
| `AnalysisCode` | Analysis code | name, code_type |

### IMAS DD Data (imas_dd.yaml)

| Node | Purpose | Key Properties |
|------|---------|----------------|
| `DDVersion` | DD version | version, release_date |
| `DDPath` | Data path | full_path, documentation, units |
| `PathChange` | Version change | change_type, semantic_type |

## CLI Commands

```bash
# Neo4j management
imas-codex neo4j start
imas-codex neo4j stop
imas-codex neo4j shell

# Graph artifacts
imas-codex neo4j dump
imas-codex neo4j push v1.0.0
imas-codex neo4j pull
```

## Build Scripts

| Script | Purpose |
|--------|---------|
| `build-models` | Generate Pydantic from LinkML |
| `build-dd-graph` | Populate IMAS DD nodes |
| `discover-mdsplus` | Ingest MDSplus tree structure |
