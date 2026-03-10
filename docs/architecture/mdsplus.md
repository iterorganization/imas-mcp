# MDSplus Tree Architecture

> **Schema**: `imas_codex/schemas/facility.yaml`

## Overview

MDSplus trees are hierarchical data containers used at fusion facilities. We model a **single unified tree** where nodes have **applicability ranges** (shot ranges) rather than storing snapshots.

## The "Super Tree" Concept

```
StructuralEpoch v1 (shots 3000-50000): 6,895 nodes
StructuralEpoch v2 (shots 50000-70000): 9,980 nodes  [+3,085 nodes]
StructuralEpoch v3 (shots 70000-82000): 11,403 nodes [+1,423 nodes]
StructuralEpoch v4 (shots 82000-current): 11,463 nodes [+60 nodes]
```

Each SignalNode has:
- `first_shot`: When this node first appeared
- `last_shot`: When it was removed (null = still active)
- `introduced_version`: Link to StructuralEpoch that added it

## Schema

### DataSource

```cypher
(:DataSource {
    name: "results",
    facility_id: "tcv"
})-[:AT_FACILITY]->(:Facility {id: "tcv"})
```

### SignalNode

```cypher
(:SignalNode {
    id: "tcv:results:\\RESULTS::PSI",
    path: "\\RESULTS::PSI",
    data_source_name: "results",
    facility_id: "tcv",
    parent_path: "\\RESULTS",
    units: "Wb",
    first_shot: 3000,
    last_shot: null,
    description: "Poloidal flux profile from LIUQE",
    physics_domain: "equilibrium",
    accessor_function: "tcv_eq"
})
```

## Ingestion Workflow

```bash
# Discover tree structure
uv run discover-mdsplus tcv results -v

# Batch ingest
uv run ingest-mdsplus tcv results tcv_shot magnetics
```

## Enrichment

DataNodes are enriched via `agent enrich`:

```bash
uv run imas-codex agent enrich --tree results --batch-size 50
```

| Field | Source | Description |
|-------|--------|-------------|
| `description` | LLM | Physics description |
| `physics_domain` | LLM | Classification |
| `enrichment_status` | Workflow | pending, enriched, failed |

## TDI Function Linking

```cypher
// Find TDI function for a node
MATCH (n:SignalNode {path: "\\RESULTS::PSI"})
MATCH (t:TDIFunction {name: n.accessor_function})
RETURN t.signature

// Find all nodes accessible via tcv_eq
MATCH (t:TDIFunction {name: "tcv_eq"})-[:ACCESSES]->(n:SignalNode)
RETURN n.path, n.units
```

## Queries

```cypher
-- What nodes existed at shot 50000?
MATCH (n:SignalNode {data_source_name: "results", facility_id: "tcv"})
WHERE n.first_shot <= 50000 
  AND (n.last_shot IS NULL OR n.last_shot > 50000)
RETURN n.path, n.description

-- Physics domain query
MATCH (n:SignalNode {physics_domain: "equilibrium", facility_id: "tcv"})
RETURN n.path, n.description, n.units
```
