# IMAS Data Dictionary Knowledge Graph

> Implementation plan for graph-based IMAS DD with version tracking and cross-linking.

**Status**: Implementation in progress  
**Date**: 2026-01-03  
**Related**: [CODEX_PLAN.md](CODEX_PLAN.md), [KNOWLEDGE_GRAPH.md](KNOWLEDGE_GRAPH.md)

---

## 1. Executive Summary

Build a Neo4j-based representation of the IMAS Data Dictionary that:
1. Captures hierarchical IDS/path structure with parent-child relationships
2. Tracks version evolution across all DD versions (currently 34: 3.22.0 → 4.1.0)
3. Links data paths to units and coordinate definitions
4. Supports compliance checking via Cypher queries
5. Integrates semantic clusters as graph nodes
6. Connects to facility data via `IMASMapping` relationships

**Design Philosophy**: Follow the MDSplus "Super Tree" pattern - store a single unified
graph where nodes have applicability ranges (introduced_version, deprecated_version)
rather than separate snapshots per version.

---

## 2. Key Design Decisions

### 2.1 Version Tracking Strategy

**Decision**: Track ALL DD versions with incremental updates.

**Rationale**:
- The `imas-data-dictionaries` package provides access to all 34 versions
- Users need compliance checking against specific versions
- New versions should be addable without full rebuild

**Implementation**:
- `DDVersion` nodes with `PREDECESSOR` chain
- `DDPath` nodes with `INTRODUCED_IN` and `DEPRECATED_IN` relationships
- `RENAMED_TO` relationships for path migrations
- `PathChange` nodes for metadata changes (units, documentation, type)

### 2.2 Metadata Change Tracking

**Decision**: Track changes to path metadata beyond rename/deprecation.

**Rationale**:
- A path's units, documentation, or type can change between versions
- Important for data provenance and compliance

**Implementation**:
```cypher
(:PathChange {
    change_type: "units" | "documentation" | "type" | "coordinates",
    old_value: "T",
    new_value: "Wb"
})-[:PATH]->(:DDPath)
  -[:VERSION]->(:DDVersion)
```

### 2.3 Semantic Cluster Integration

**Decision**: Store clusters as separate `SemanticCluster` nodes linked via `:IN_CLUSTER`.

**Rationale**:
- Cluster centroids represent semantic concepts
- Paths link to clusters with distance for ranking
- Enables cluster-based navigation without semantic search
- Cross-IDS clusters naturally represented

**Implementation**:
```cypher
(:DDPath)-[:IN_CLUSTER {distance: 0.15}]->(:SemanticCluster {
    label: "boundary geometry",
    cross_ids: true
})
```

### 2.4 Coordinate Handling

**Decision**: Follow `imas-python` coordinate patterns.

**Analysis** (from `imas/ids_coordinates.py`):
- Index coordinates: `1...N`, `1...3` → `CoordinateSpec` nodes
- Path references: `time`, `profiles_1d(itime)/time` → relationships to `DDPath`
- Alternatives: `r OR rho_tor_norm` → multiple relationships with `alternative: true`
- External IDS: `IDS:equilibrium/time` → stored as property (rare)

**Implementation**:
```cypher
(:DDPath)-[:COORDINATE {dimension: 1, alternative: false}]->(:DDPath)
(:DDPath)-[:COORDINATE {dimension: 2}]->(:CoordinateSpec {id: "1...N"})
```

### 2.5 Build vs Augment Strategy

**Decision**: Augment existing graph, never rebuild.

**Rationale**:
- IMAS DD graph shares Neo4j with facility knowledge
- Facility data (TreeNodes, IMASMappings) links to DDPath nodes
- Rebuilding would break cross-references

**Implementation**:
- `build dd-graph --incremental` for new versions
- Version comparison to detect additions/removals
- Update `is_current` flags appropriately

### 2.6 Hierarchy Representation

**Decision**: Store both full path as ID and explicit PARENT relationships.

**Rationale**:
- Full path enables O(1) lookup by path string
- PARENT relationships enable hierarchy traversal
- Matches MDSplus TreeNode pattern

**Implementation**:
```cypher
(:DDPath {id: "equilibrium/time_slice/boundary/psi"})
  -[:PARENT]->(:DDPath {id: "equilibrium/time_slice/boundary"})
```

---

## 3. Schema Design

### Node Types

| Node | Count (est.) | Identifier | Purpose |
|------|--------------|------------|---------|
| `DDVersion` | 34+ | `id` (e.g., "3.42.2") | DD version metadata |
| `IDS` | ~55 | `name` (e.g., "equilibrium") | Top-level IDS definitions |
| `DDPath` | ~60k | `id` (full path) | Data dictionary paths |
| `Unit` | ~50 | `symbol` (e.g., "Wb") | Physical units |
| `CoordinateSpec` | ~10 | `id` (e.g., "1...N") | Index-based coordinates |
| `SemanticCluster` | ~500 | `id` | Cluster centroids |
| `IdentifierSchema` | ~30 | `name` | Enumeration definitions |
| `PathChange` | variable | composite | Metadata change tracking |

### Relationships

| Relationship | From | To | Properties |
|--------------|------|----|----|
| `PREDECESSOR` | DDVersion | DDVersion | - |
| `INTRODUCED_IN` | IDS, DDPath | DDVersion | - |
| `DEPRECATED_IN` | IDS, DDPath | DDVersion | - |
| `RENAMED_TO` | DDPath | DDPath | `version` |
| `IDS` | DDPath | IDS | - |
| `PARENT` | DDPath | DDPath | - |
| `HAS_UNIT` | DDPath | Unit | - |
| `COORDINATE` | DDPath | DDPath/CoordinateSpec | `dimension`, `alternative`, `same_as` |
| `USES_IDENTIFIER` | DDPath | IdentifierSchema | - |
| `IN_CLUSTER` | DDPath | SemanticCluster | `distance` |
| `PATH` | PathChange | DDPath | - |
| `VERSION` | PathChange | DDVersion | - |

---

## 4. Integration Points

### 4.1 Facility Graph Integration

The IMAS DD graph connects to facility data via `IMASMapping`:

```cypher
-- Facility TreeNode → IMAS DDPath mapping
(:TreeNode {path: "\\RESULTS::LIUQE:PSI"})
  <-[:SOURCE]-(:IMASMapping)-[:TARGET]->
(:DDPath {id: "equilibrium/time_slice/profiles_2d/psi"})
```

### 4.2 Existing Infrastructure Reuse

| Component | Current Use | Graph Use |
|-----------|-------------|-----------|
| `build_path_map.py` | JSON mappings | Version diff logic |
| `DataDictionaryTransformer` | Extract XML metadata | Path extraction |
| `Clusters` | Pickle storage | SemanticCluster nodes |
| `PathMap` | Check/fetch tools | (Replaced by Cypher) |
| `DocumentStore` | Semantic search | (Parallel, then replace) |

---

## 5. Query Examples

### Check path exists in specific version
```cypher
MATCH (p:DDPath {id: $path})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dv:DDVersion)
WITH p, iv, dv
WHERE (iv.id IS NULL OR iv.id <= $version)
  AND (dv.id IS NULL OR dv.id > $version)
RETURN p.id, p.data_type, p.documentation
```

### Get all paths valid in a version
```cypher
MATCH (ids:IDS {name: $ids_name})<-[:IDS]-(p:DDPath)
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv)
OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dv)
WHERE (iv.id IS NULL OR iv.id <= $version)
  AND (dv.id IS NULL OR dv.id > $version)
RETURN p.id, p.data_type
ORDER BY p.id
```

### Find paths by cluster
```cypher
MATCH (c:SemanticCluster {label: $label})<-[r:IN_CLUSTER]-(p:DDPath)
RETURN p.id, p.documentation, r.distance
ORDER BY r.distance
LIMIT 20
```

### Get path rename history
```cypher
MATCH path = (old:DDPath)-[:RENAMED_TO*]->(current:DDPath {id: $path})
RETURN [n IN nodes(path) | n.id] AS history
```

### Check compliance
```cypher
UNWIND $paths AS path_id
OPTIONAL MATCH (p:DDPath {id: path_id})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv)
OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dv)
WITH path_id, p, iv, dv, $version AS target
RETURN path_id,
       CASE 
         WHEN p IS NULL THEN 'not_found'
         WHEN iv.id IS NOT NULL AND iv.id > target THEN 'not_yet_introduced'
         WHEN dv.id IS NOT NULL AND dv.id <= target THEN 'deprecated'
         ELSE 'valid'
       END AS status
```

---

## 6. Implementation Roadmap

### Phase 1: Schema ✅
- [x] Create `schemas/imas_dd.yaml` LinkML schema
- [x] Generate Pydantic models (`graph/dd_models.py`)

### Phase 2: Build Script ✅
- [x] Create `scripts/build_dd_graph.py`
- [x] DDVersion node creation with predecessor chain
- [x] IDS node extraction
- [x] DDPath extraction with hierarchy (PARENT relationships)
- [x] Unit node deduplication and HAS_UNIT relationships
- [x] CoordinateSpec node creation
- [x] IDS filter option for testing
- [x] INTRODUCED_IN relationships
- [x] RENAMED_TO from path mappings
- [ ] DEPRECATED_IN for removed paths (multi-version)
- [ ] PathChange detection for metadata changes

### Phase 3: Cluster Migration
- [ ] Load existing cluster data
- [ ] Create SemanticCluster nodes
- [ ] Create IN_CLUSTER relationships

### Phase 4: Incremental Updates
- [x] `--from-version` flag for starting from specific version
- [x] `--all-versions` flag for full build
- [ ] Proper version comparison for incremental updates
- [ ] is_current flag management

### Phase 5: MCP Tool Migration (Future)
- [ ] Graph-based check_imas_paths
- [ ] Graph-based fetch_imas_paths
- [ ] Graph-based search (cluster navigation)

---

## 7. CLI Interface

```bash
# Build full DD graph (all versions)
uv run imas-codex build dd-graph --all-versions

# Build current version only
uv run imas-codex build dd-graph

# Incremental update for new version
uv run imas-codex build dd-graph --from-version 4.0.0

# Include cluster migration
uv run imas-codex build dd-graph --include-clusters

# Dry run
uv run imas-codex build dd-graph --dry-run -v
```

---

## 8. References

- [imas-python ids_coordinates.py](../.venv/lib/python3.12/site-packages/imas/ids_coordinates.py) - Coordinate parsing logic
- [imas-python nc_metadata.py](../.venv/lib/python3.12/site-packages/imas/backends/netcdf/nc_metadata.py) - Dimension tensorization
- [build_path_map.py](../scripts/build_path_map.py) - Version mapping logic
- [MDSPLUS_TREE_EXPLORATION.md](MDSPLUS_TREE_EXPLORATION.md) - Super tree pattern

