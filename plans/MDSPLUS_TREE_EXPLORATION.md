# MDSplus Tree Exploration Plan

> Status tracking and strategy for systematic MDSplus TreeNode ingestion.

## Current Coverage (2025-12-31)

### TreeNode Statistics

| Tree | Expected | Ingested | Coverage | Population |
|------|----------|----------|----------|------------|
| results | 11,000 | 272 | 2.5% | dynamic |
| tcv_shot | 84,000 | 49 | 0.1% | dynamic |
| magnetics | 500 | 42 | 8.4% | static |
| base | 1,300 | 9 | 0.7% | static |
| thomson | 1,000 | 0 | 0% | hybrid |
| atlas | - | 31 | - | - |
| diagz | - | 29 | - | - |
| ecrh | - | 17 | - | - |
| pcs | - | 15 | - | - |
| vsystem | - | 14 | - | - |
| **Total** | ~98k | **493** | <1% | - |

### Physics Domain Distribution

| Domain | Nodes | Pct |
|--------|-------|-----|
| equilibrium | 165 | 33% |
| profiles | 100 | 20% |
| magnetics | 63 | 13% |
| heating | 33 | 7% |
| edge | 18 | 4% |
| diagnostics | 15 | 3% |
| radiation | 14 | 3% |
| control | 14 | 3% |
| (16 other domains) | 71 | 14% |

### Accessor Function Links

| Function | Nodes Linked |
|----------|-------------|
| tcv_eq | 40 |
| tcv_get | 17 |
| tcv_psitbx | 6 |
| tcv_ip | 3 |
| **Total with accessor** | 66 (13%) |

### Top Subtrees in RESULTS Tree

| Subtree | Nodes |
|---------|-------|
| THOMSON | 23 |
| THOMSON.PROFILES.AUTO | 17 |
| LANGMUIR | 15 |
| CXRS | 14 |
| PSITBX | 11 |
| BOLOMETER | 9 |
| ECE | 8 |
| FIR | 7 |
| TORAY | 6 |

## Strategy Questions

### 1. Duplicate Handling in ingest_nodes

**Current behavior**: Uses `MERGE` which updates existing nodes if path matches.

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| A. MERGE (current) | Idempotent, safe re-runs | Silent overwrites |
| B. Check before insert | Explicit control | Extra query overhead |
| C. Return which were new vs updated | Best visibility | More complex return type |

**Recommendation**: Keep MERGE but enhance return value to indicate `{"created": N, "updated": M, "errors": []}`. This provides visibility without breaking idempotency.

**Implementation** (in `create_nodes`):
```cypher
UNWIND $batch AS item
MERGE (n:{label} {path: item.path})
ON CREATE SET n = item, n._created = true
ON MATCH SET n += item
WITH n, exists(n._created) AS was_created
REMOVE n._created
RETURN count(CASE WHEN was_created THEN 1 END) AS created,
       count(CASE WHEN NOT was_created THEN 1 END) AS updated
```

### 2. Hierarchy Storage Strategy

**Current state**: Hierarchy is embedded in path string `\\RESULTS::LIUQE:PSI`.

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| A. Path string only (current) | Simple, path prefix queries work | No parent traversal |
| B. Add `parent_path` property | Explicit parent link, easy tree walks | Denormalized, must maintain |
| C. `PARENT_OF` relationships | Full graph traversal | Many relationships, complex queries |
| D. Hybrid: `parent_path` + lazy relationships | Best of both | Two maintenance burdens |

**Recommendation**: Add `parent_path` property. Benefits:
- Enables `MATCH (n:TreeNode)-[:PARENT]->(p:TreeNode)` style queries
- Cheap to compute at ingestion time
- Works even when parent node not yet ingested

**Schema change** (in `facility.yaml`):
```yaml
parent_path:
  description: >-
    Path to parent node (computed from path). Enables hierarchy traversal
    without explicit relationships. Null for root nodes (tree level).
  range: string
```

**Compute at ingestion**:
```python
def compute_parent_path(path: str) -> str | None:
    """Extract parent path from MDSplus node path.
    
    \\RESULTS::LIUQE:PSI -> \\RESULTS::LIUQE
    \\RESULTS::LIUQE -> \\RESULTS
    \\RESULTS -> None (root)
    """
    if ':' in path:
        return path.rsplit(':', 1)[0]
    return None
```

### 3. Progress Tool Integration

**Current `get_exploration_progress`** returns:
- `mdsplus_coverage`: Per-tree stats from `MDSplusTree` nodes
- `tdi_coverage`: TDI functions by physics domain
- `code_coverage`: Analysis codes by type

**Problem**: Current query uses `MDSPLUS_TREE` relationship that doesn't exist.

**Fix**: Query should use `tree_name` property match instead:

```cypher
-- Current (broken)
OPTIONAL MATCH (n:TreeNode)-[:MDSPLUS_TREE]->(t)

-- Fixed (uses property)
OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:FACILITY_ID]->(f)
```

**Enhanced coverage section** for progress tool:

```python
# Add TreeNode coverage detail
tree_node_rows = client.query("""
    MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: $fid})
    WITH n.tree_name AS tree,
         n.physics_domain AS domain,
         count(*) AS nodes,
         count(n.accessor_function) AS with_accessor
    RETURN tree, domain, nodes, with_accessor
    ORDER BY nodes DESC
""", fid=facility)

# Add subtree coverage
subtree_rows = client.query("""
    MATCH (n:TreeNode {tree_name: 'results'})-[:FACILITY_ID]->(f:Facility {id: $fid})
    WITH split(replace(n.path, '\\\\RESULTS::', ''), ':')[0] AS subtree,
         count(*) AS nodes
    RETURN subtree, nodes
    ORDER BY nodes DESC
    LIMIT 20
""", fid=facility)
```

**New return structure**:
```python
{
    # ... existing fields ...
    "tree_node_coverage": {
        "total": 493,
        "by_tree": {"results": 272, "magnetics": 42, ...},
        "by_domain": {"equilibrium": 165, "profiles": 100, ...},
        "with_accessor": 66,
        "accessor_pct": 13.4,
        "top_subtrees": {"THOMSON": 23, "LANGMUIR": 15, ...}
    }
}
```

### 4. Relationship Materialization

**Current state**: Only `FACILITY_ID` relationships exist (493).

**Defined in schema but not created**:
- `TREE_NAME` (TreeNode → MDSplusTree)
- `ACCESSOR_FUNCTION` (TreeNode → TDIFunction)
- `VARIANT_SOURCES` (TreeNode → AnalysisCode)

**Strategy**: Materialize lazily via property values.

For `TREE_NAME`:
```cypher
MATCH (n:TreeNode)
WHERE n.tree_name IS NOT NULL
MATCH (t:MDSplusTree {name: n.tree_name})
MERGE (n)-[:TREE_NAME]->(t)
```

For `ACCESSOR_FUNCTION`:
```cypher
MATCH (n:TreeNode)
WHERE n.accessor_function IS NOT NULL
MATCH (tdi:TDIFunction {name: n.accessor_function})
MERGE (n)-[:ACCESSOR_FUNCTION]->(tdi)
```

**Decision**: Add relationship creation to ingest_nodes for TreeNode type. When ingesting TreeNodes:
1. Create FACILITY_ID (existing)
2. Create TREE_NAME if tree_name provided
3. Create ACCESSOR_FUNCTION if accessor_function provided

### 5. Agent Coordination for Exploration

**Problem**: Multiple agents may discover same paths.

**Solution**: Before ingesting, check existence:

```cypher
-- Batch check for existing paths
UNWIND $paths AS p
OPTIONAL MATCH (n:TreeNode {path: p})
RETURN p AS path, n IS NOT NULL AS exists
```

**Coverage query for unexplored areas**:
```cypher
-- Find subtrees with low coverage
MATCH (t:MDSplusTree {name: 'results'})-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
OPTIONAL MATCH (n:TreeNode {tree_name: 'results'})-[:FACILITY_ID]->(f)
WITH t, count(n) AS ingested, t.node_count_total AS total
RETURN t.name, ingested, total, 
       round(100.0 * ingested / total, 1) AS pct
```

**Next exploration priorities**:
1. Thomson tree (0% coverage, 1000 nodes)
2. results subtrees not yet touched
3. Nodes with accessor functions but no TreeNode entry

## Action Items

### Immediate (Schema/Tool Fixes)

1. [ ] Fix `get_exploration_progress` to use property match instead of `MDSPLUS_TREE` relationship
2. [ ] Add `tree_node_coverage` section to progress output
3. [ ] Add `parent_path` property to TreeNode schema

### Near-term (Relationship Materialization)

4. [ ] Create `TREE_NAME` relationships for existing TreeNodes
5. [ ] Create `ACCESSOR_FUNCTION` relationships for nodes with accessor_function property
6. [ ] Update ingest_nodes to auto-create these relationships

### Exploration (Ongoing)

7. [ ] Ingest Thomson tree structure via MDSplus introspection
8. [ ] Continue results tree expansion focusing on:
   - Diagnostics with accessor functions
   - Physics domains not yet covered
   - High-value subtrees (LANGMUIR, CXRS, ECE)

## Query Reference

### Check before ingest
```cypher
MATCH (n:TreeNode {path: $path})
RETURN count(n) > 0 AS exists
```

### Find gaps in coverage
```cypher
MATCH (t:MDSplusTree)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:FACILITY_ID]->(f)
WITH t.name AS tree, count(n) AS ingested, t.node_count_total AS expected
WHERE expected IS NOT NULL
RETURN tree, ingested, expected,
       round(100.0 * ingested / expected, 1) AS pct
ORDER BY pct ASC
```

### Unexplored subtrees
```cypher
MATCH (n:TreeNode {tree_name: 'results'})-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
WITH split(replace(n.path, '\\\\RESULTS::', ''), ':')[0] AS subtree
RETURN subtree, count(*) AS nodes
ORDER BY nodes ASC
LIMIT 20
```

### Nodes missing accessor function
```cypher
MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
WHERE n.accessor_function IS NULL
AND n.physics_domain IN ['equilibrium', 'profiles']
RETURN n.path, n.physics_domain
ORDER BY n.path
```

### Coverage by physics domain
```cypher
MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
RETURN n.physics_domain AS domain, count(*) AS nodes
ORDER BY nodes DESC
```

## See Also

- [MDSPLUS_INGESTION.md](MDSPLUS_INGESTION.md) - Original ingestion workflow
- [FACILITY_KNOWLEDGE.md](FACILITY_KNOWLEDGE.md) - Facility exploration guide
- [schemas/facility.yaml](../imas_codex/schemas/facility.yaml) - Schema definitions
