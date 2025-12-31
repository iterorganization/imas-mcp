# MDSplus Tree Exploration Plan

> Strategy and progress tracking for systematic MDSplus TreeNode ingestion.

## Overview

This document describes the approach for populating the knowledge graph with MDSplus tree structure from remote facilities. The system supports both:

1. **Script-driven ingestion** - For high-volume batch operations (>1000 nodes)
2. **LLM-driven exploration** - For discovery, triage, and enrichment tasks

Neo4j easily handles 100k+ nodes, so capacity is not a constraint.

## Reference Shots

For dynamic trees (shot-dependent structure), we use reference shots that have rich data across diagnostics. These are documented per-tree in the graph.

### EPFL/TCV Reference Shots

| Tree | Reference Shot | Reason | Node Count |
|------|---------------|--------|------------|
| results | 84469 | Recent, 62% data occupancy | ~11,500 |
| tcv_shot | 84469 | Recent, full diagnostic set | ~82,500 |
| magnetics | - | Static tree | ~500 |
| base | - | Static tree | ~1,300 |
| thomson | 84469 | Rich Thomson data | ~1,000 |

**Selection criteria**:
- Recent shot (within last 1000 shots)
- High data occupancy (sample nodes with data)
- Representative diagnostic coverage
- Successful plasma discharge

## Ingestion Workflow

### Phase 1: Script-Driven Bulk Ingestion

Use `ingest-mdsplus` for initial tree population:

```bash
# Ingest a single tree
uv run ingest-mdsplus epfl results --shot 84469

# Ingest multiple trees
uv run ingest-mdsplus epfl results tcv_shot magnetics --shot 84469

# Dry run to preview
uv run ingest-mdsplus epfl results --shot 84469 --dry-run -v

# Limit for testing
uv run ingest-mdsplus epfl results --shot 84469 --limit 100
```

The script:
1. Connects to facility via SSH
2. Introspects tree structure using Python MDSplus
3. Computes `parent_path` for hierarchy traversal
4. Batch-inserts TreeNodes with UNWIND
5. Creates `TREE_NAME` and `FACILITY_ID` relationships
6. Updates MDSplusTree stats (node_count_ingested, reference_shot)

### Phase 2: LLM-Driven Enrichment

After bulk ingestion, agents use `get_exploration_progress` to identify gaps:

```python
# Agent workflow
progress = get_exploration_progress("epfl")

# Check next_targets for prioritized work
for target in progress["next_targets"]:
    print(f"{target['priority']}: {target['action']}")
    # 1: Ingest thomson tree structure
    # 2: Continue results ingestion (2.5% complete)
    # 3: Expand equilibrium domain coverage
```

Agents then enrich nodes with:
- `physics_domain` classification
- `accessor_function` links
- `description` from documentation
- `shape` and `units` refinement

## Progress Tracking

### get_exploration_progress Output

The `get_exploration_progress` tool returns comprehensive metrics:

```python
{
    "facility": "epfl",
    "paths": {...},  # FacilityPath status
    "mdsplus_coverage": {
        "results": {
            "status": "partial",
            "total_nodes": 11500,
            "ingested_nodes": 272,
            "coverage_pct": 2.4,
            "population_type": "dynamic"
        },
        ...
    },
    "tree_node_coverage": {
        "total": 493,
        "by_tree": {"results": 272, "magnetics": 42, ...},
        "by_domain": {"equilibrium": 165, "profiles": 100, ...},
        "with_accessor": 66,
        "accessor_pct": 13.4,
        "top_subtrees": {"THOMSON": 23, ...}
    },
    "next_targets": [
        {
            "priority": 1,
            "type": "mdsplus_tree",
            "target": "thomson",
            "action": "Ingest thomson tree structure",
            "expected_nodes": 1000,
            "effort": "medium"
        },
        ...
    ],
    "recommendation": "Ingest thomson tree structure"
}
```

### Priority Algorithm

`next_targets` uses this priority order:

| Priority | Type | Condition |
|----------|------|-----------|
| 1 | mdsplus_tree | 0% coverage (breadth-first) |
| 2 | mdsplus_tree | <10% coverage (continue partial) |
| 3 | physics_domain | High-value domain with <50 nodes |
| 4 | results_subtree | High-value subtree not yet explored |

High-value domains: equilibrium, profiles, magnetics, heating
High-value subtrees: THOMSON, LANGMUIR, CXRS, ECE, FIR, BOLOMETER, TORAY, LIUQE, PSITBX, ECRH, NBI

## Schema Design

### TreeNode Properties

| Property | Required | Description |
|----------|----------|-------------|
| path | ✓ | Full MDSplus path (identifier) |
| tree_name | ✓ | Parent tree name |
| facility_id | ✓ | Parent facility |
| units | ✓ | Physical units (SI) |
| parent_path | | Computed parent for hierarchy |
| node_type | | STRUCTURE, SIGNAL, NUMERIC, etc. |
| physics_domain | | For categorization |
| accessor_function | | TDI function for access |
| example_shot | | Reference shot number |
| description | | Node description |

### Auto-Created Relationships

When ingesting TreeNodes, `ingest_nodes` automatically creates:

1. `FACILITY_ID` → Facility (always)
2. `TREE_NAME` → MDSplusTree (if tree_name provided)
3. `ACCESSOR_FUNCTION` → TDIFunction (if accessor_function provided)

### Hierarchy Traversal

Use `parent_path` property for hierarchy queries:

```cypher
-- Find all children of a node
MATCH (n:TreeNode)
WHERE n.parent_path = '\\RESULTS::LIUQE'
RETURN n.path

-- Find ancestors
MATCH (n:TreeNode {path: '\\RESULTS::LIUQE:PSI'})
WITH n.parent_path AS parent
MATCH (p:TreeNode {path: parent})
RETURN p.path
```

## Current Coverage (2025-12-31)

### TreeNode Statistics

| Tree | Expected | Ingested | Coverage | Population |
|------|----------|----------|----------|------------|
| results | 11,500 | 272 | 2.4% | dynamic |
| tcv_shot | 82,500 | 49 | 0.1% | dynamic |
| magnetics | 500 | 42 | 8.4% | static |
| base | 1,300 | 9 | 0.7% | static |
| thomson | 1,000 | 0 | 0% | hybrid |
| **Total** | ~96k | **493** | <1% | - |

### Physics Domain Distribution

| Domain | Nodes | Pct |
|--------|-------|-----|
| equilibrium | 165 | 33% |
| profiles | 100 | 20% |
| magnetics | 63 | 13% |
| heating | 33 | 7% |
| edge | 18 | 4% |
| (others) | 114 | 23% |

## Action Items

### Completed ✓

- [x] Add `parent_path` property to TreeNode schema
- [x] Enhance `get_exploration_progress` with `next_targets` section
- [x] Auto-create `TREE_NAME` and `ACCESSOR_FUNCTION` relationships in `ingest_nodes`
- [x] Create `ingest-mdsplus` script for batch ingestion

### Next Steps

1. [ ] Run `ingest-mdsplus epfl results --shot 84469` to complete results tree
2. [ ] Run `ingest-mdsplus epfl tcv_shot --shot 84469` for tcv_shot tree
3. [ ] Run `ingest-mdsplus epfl thomson --shot 84469` for thomson tree
4. [ ] Regenerate models: `uv run build-models --force`
5. [ ] LLM pass to enrich nodes with physics_domain classification

## Query Reference

### Check exploration progress
```cypher
MATCH (t:MDSplusTree)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:FACILITY_ID]->(f)
WITH t.name AS tree, count(n) AS ingested, t.node_count_total AS expected
RETURN tree, ingested, expected,
       round(100.0 * ingested / CASE WHEN expected > 0 THEN expected ELSE 1 END, 1) AS pct
ORDER BY pct ASC
```

### Coverage by physics domain
```cypher
MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
RETURN n.physics_domain AS domain, count(*) AS nodes
ORDER BY nodes DESC
```

### Find nodes needing enrichment
```cypher
MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
WHERE n.physics_domain IS NULL
RETURN n.tree_name AS tree, count(*) AS needs_domain
ORDER BY needs_domain DESC
```

### Hierarchy traversal example
```cypher
-- Get subtree rooted at LIUQE
MATCH (n:TreeNode)
WHERE n.path STARTS WITH '\\RESULTS::LIUQE'
RETURN n.path, n.node_type, n.physics_domain
ORDER BY n.path
```

## See Also

- [MDSPLUS_INGESTION.md](MDSPLUS_INGESTION.md) - Original ingestion workflow
- [FACILITY_KNOWLEDGE.md](FACILITY_KNOWLEDGE.md) - Facility exploration guide
- [schemas/facility.yaml](../imas_codex/schemas/facility.yaml) - Schema definitions
