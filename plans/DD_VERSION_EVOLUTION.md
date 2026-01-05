# IMAS DD Version Evolution Tracking

> Plan for multi-version ingestion with semantic change detection.

**Status**: Implementation complete (schema + build script)  
**Date**: 2026-01-05  
**Related**: [IMAS_DD_GRAPH.md](IMAS_DD_GRAPH.md), [MDSPLUS_INGESTION.md](MDSPLUS_INGESTION.md)

---

## 1. Current State

### Graph State (after test with equilibrium IDS, 3.42.0 → 4.1.0)
- **DDVersion nodes**: 5 (with PREDECESSOR chain)
- **DDPath nodes**: 2,018
- **PathChange nodes**: 274 (with semantic classification)
- **Sign convention changes detected**: 2 (equilibrium psi paths)

### Implementation Status
- ✅ `SemanticChangeType` enum added to schema
- ✅ `PathChange` enhanced with `semantic_type`, `keywords_detected`
- ✅ `classify_doc_change()` implemented in build script
- ✅ Multi-version ingestion tested and working
- ⏳ Full ingestion (all 34 versions, all IDS) pending

---

## 2. Key Changes: v3 → v4 Poloidal Flux Convention

The transition from DD v3.42.2 to v4.0.0 introduced a **critical sign convention change** for poloidal flux:

### Documentation Change
- **Old (v3.x)**: "Poloidal flux" (minimal definition)
- **New (v4.0)**: "Poloidal flux. Integral of magnetic field passing through a contour defined by the intersection of a flux surface passing through the point of interest and a Z=constant plane. **If the integration surface is flat, the surface normal vector is in the increasing vertical coordinate direction, Z, namely upwards.**"

### Affected Paths (24 total)
| IDS | Path | Change Type |
|-----|------|-------------|
| equilibrium | time_slice/profiles_1d/psi | sign_convention |
| equilibrium | time_slice/profiles_2d/psi | sign_convention |
| core_profiles | profiles_1d/grid/psi | sign_convention |
| core_transport | model/profiles_1d/grid_d/psi | sign_convention |
| edge_profiles | profiles_1d/grid/psi | sign_convention |
| ... | (19 more paths) | sign_convention |

### Statistics (v3.42.2 → v4.0.0)
- **Added**: 663 paths
- **Removed**: 14,031 paths (major restructuring)
- **Documentation changes**: 586 paths
- **Unit changes**: 6 paths
- **Type changes**: 111 paths

---

## 3. Proposed Schema Enhancement

### 3.1 SemanticChangeType Enum

Add a new enum to classify documentation changes by physics significance:

```yaml
enums:
  SemanticChangeType:
    description: Classification of physics-significant documentation changes
    permissible_values:
      sign_convention:
        description: Sign or orientation convention change (affects data interpretation)
      definition_clarification:
        description: More precise definition without semantic change
      coordinate_convention:
        description: Coordinate system or reference frame change
      unit_convention:
        description: Unit interpretation or normalization change
      boundary_condition:
        description: Boundary or edge case behavior change
      deprecated_behavior:
        description: Previously allowed behavior now deprecated
      none:
        description: Documentation change with no physics significance
```

### 3.2 PathChange Enhancement

Extend `PathChange` to include semantic classification:

```yaml
classes:
  PathChange:
    attributes:
      # ... existing attributes ...
      semantic_type:
        description: Physics significance classification (for documentation changes)
        range: SemanticChangeType
      keywords_added:
        description: Significant keywords in new documentation
        multivalued: true
      affects_sign:
        description: Whether change affects data sign interpretation
        range: boolean
      migration_note:
        description: Guidance for data migration
```

### 3.3 VersionTransition Node

Track major version transitions with summary metadata:

```yaml
classes:
  VersionTransition:
    description: Summary of changes between consecutive DD versions
    attributes:
      id:
        description: "{from_version}:{to_version}"
        identifier: true
      from_version:
        range: DDVersion
        required: true
      to_version:
        range: DDVersion
        required: true
      paths_added:
        range: integer
      paths_removed:
        range: integer
      paths_changed:
        range: integer
      semantic_changes:
        description: Count of physics-significant changes
        range: integer
      is_major:
        description: Whether this is a major version transition
        range: boolean
      breaking_changes:
        description: JSON summary of breaking changes
```

---

## 4. Semantic Change Detection

### 4.1 Keyword-Based Detection

Detect physics-significant changes by keyword analysis:

```python
SIGN_CONVENTION_KEYWORDS = [
    "upwards", "downwards", "increasing", "decreasing",
    "positive", "negative", "clockwise", "anti-clockwise",
    "normal vector", "surface normal", "sign convention"
]

COORDINATE_CONVENTION_KEYWORDS = [
    "right-handed", "left-handed", "coordinate system",
    "reference frame", "origin", "axis"
]

BOUNDARY_KEYWORDS = [
    "boundary", "LCFS", "separatrix", "edge", "core"
]

def classify_doc_change(old_doc: str, new_doc: str) -> SemanticChangeType:
    """Classify a documentation change by physics significance."""
    old_lower = old_doc.lower() if old_doc else ""
    new_lower = new_doc.lower() if new_doc else ""
    
    # Check for new sign convention text
    for kw in SIGN_CONVENTION_KEYWORDS:
        if kw in new_lower and kw not in old_lower:
            return SemanticChangeType.SIGN_CONVENTION
    
    # Check for coordinate changes
    for kw in COORDINATE_CONVENTION_KEYWORDS:
        if kw in new_lower and kw not in old_lower:
            return SemanticChangeType.COORDINATE_CONVENTION
    
    # ... more classification logic ...
    
    return SemanticChangeType.DEFINITION_CLARIFICATION
```

### 4.2 Version-Aware Queries

Enable queries like:

```cypher
-- Find all paths with sign convention changes in v4.0.0
MATCH (pc:PathChange)-[:VERSION]->(v:DDVersion {id: "4.0.0"})
WHERE pc.semantic_type = "sign_convention"
MATCH (pc)-[:PATH]->(p:DDPath)
RETURN p.id, pc.old_value, pc.new_value

-- Get migration guidance for equilibrium paths
MATCH (p:DDPath)-[:IDS]->(ids:IDS {name: "equilibrium"})
OPTIONAL MATCH (pc:PathChange {semantic_type: "sign_convention"})-[:PATH]->(p)
WHERE pc IS NOT NULL
RETURN p.id, pc.migration_note
```

---

## 5. Implementation Plan

### Phase 1: Schema + Build Script ✅ COMPLETE

**Implemented:**
1. Added `SemanticChangeType` enum to `schemas/imas_dd.yaml`
2. Added `semantic_type`, `keywords_detected` to `PathChange`
3. Regenerated Pydantic models
4. Implemented `classify_doc_change()` in `build_dd_graph.py`
5. Updated `_batch_create_path_changes()` with classification

### Phase 2: Full Ingestion (User Action Required)

**Command to run full ingestion:**
```bash
# Clear existing DD nodes (preserves facility data)
uv run python -c "
from imas_codex.graph import GraphClient
with GraphClient() as client:
    client.query('MATCH (p:DDPath) DETACH DELETE p')
    client.query('MATCH (v:DDVersion) DETACH DELETE v')
    client.query('MATCH (u:Unit) DETACH DELETE u')
    client.query('MATCH (c:CoordinateSpec) DETACH DELETE c')
    client.query('MATCH (pc:PathChange) DETACH DELETE pc')
    client.query('MATCH (ids:IDS) DETACH DELETE ids')
"

# Build all 34 versions (~5 minutes)
uv run build-dd-graph --all-versions -v
```

### Phase 3: Query Examples (No Tools Needed)

Use Cypher directly via `cypher()` MCP tool or Neo4j browser:

```cypher
-- Find all sign convention changes
MATCH (pc:PathChange {semantic_type: "sign_convention"})
MATCH (pc)-[:PATH]->(p:DDPath)
MATCH (pc)-[:VERSION]->(v:DDVersion)
RETURN v.id AS version, p.id AS path, pc.keywords_detected

-- Get path history across versions
MATCH (p:DDPath {id: "equilibrium/time_slice/profiles_1d/psi"})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv)
OPTIONAL MATCH (pc:PathChange)-[:PATH]->(p)
OPTIONAL MATCH (pc)-[:VERSION]->(cv)
RETURN iv.id AS introduced, collect({version: cv.id, type: pc.semantic_type})

-- Check if path valid in specific version
WITH "equilibrium/time_slice/profiles_1d/psi" AS path_id, "3.42.2" AS target
MATCH (p:DDPath {id: path_id})-[:INTRODUCED_IN]->(iv)
RETURN p.id, iv.id <= target AS valid_in_version
```

---

## 6. Comparison with MDSplus Epochs

The MDSplus ingestion uses **TreeModelVersion** nodes with shot-range validity:

| Concept | MDSplus | IMAS DD |
|---------|---------|---------|
| Version ID | `epfl:results:v3` | `4.0.0` |
| Validity | shot range | version string |
| Node tracking | TreeNode.first_shot/last_shot | DDPath INTRODUCED_IN/DEPRECATED_IN |
| Change detection | path added/removed per epoch | PathChange nodes |
| Semantic changes | (not tracked) | SemanticChangeType classification |

The DD version tracking is more sophisticated because:
1. DD has metadata changes (units, docs) not just path presence
2. Physics conventions require semantic classification
3. Version chain is linear (not shot-based)

---

## 7. Example Queries

### Get path validity across versions
```cypher
MATCH (p:DDPath {id: "equilibrium/time_slice/profiles_1d/psi"})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dv:DDVersion)
RETURN p.id, iv.id AS introduced, dv.id AS deprecated
```

### Get all sign convention changes
```cypher
MATCH (pc:PathChange)
WHERE pc.semantic_type = "sign_convention"
MATCH (pc)-[:PATH]->(p:DDPath)
MATCH (pc)-[:VERSION]->(v:DDVersion)
RETURN v.id AS version, p.id AS path, pc.keywords_added
ORDER BY v.id, p.id
```

### Get breaking changes between versions
```cypher
MATCH (t:VersionTransition)
WHERE t.from_version.id = "3.42.2" AND t.to_version.id = "4.0.0"
RETURN t.semantic_changes, t.breaking_changes
```

### Check path compliance with semantic warnings
```cypher
WITH "equilibrium/time_slice/profiles_1d/psi" AS path_id, "3.42.2" AS target
MATCH (p:DDPath {id: path_id})
OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv)
OPTIONAL MATCH (pc:PathChange {semantic_type: "sign_convention"})-[:PATH]->(p)
OPTIONAL MATCH (pc)-[:VERSION]->(cv)
WHERE cv.id > target
RETURN p.id,
  CASE WHEN iv.id > target THEN "not_available" ELSE "valid" END AS status,
  CASE WHEN cv IS NOT NULL THEN "⚠️ Sign convention changed in " + cv.id ELSE NULL END AS warning
```

---

## 8. References

- [IMAS DD v4.0.0 Release Notes](https://imas.iter.org/) - Official change documentation
- [imas-python coordinate handling](../.venv/lib/python3.12/site-packages/imas/ids_coordinates.py)
- [build_dd_graph.py](../scripts/build_dd_graph.py) - Current implementation
- [MDSPLUS_INGESTION.md](MDSPLUS_INGESTION.md) - Epoch pattern reference
