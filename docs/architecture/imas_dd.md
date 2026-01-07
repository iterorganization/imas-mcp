# IMAS Data Dictionary Architecture

> **Schema**: `imas_codex/schemas/imas_dd.yaml`

## Overview

The IMAS Data Dictionary (DD) is stored in Neo4j with full version tracking across all 34 versions (3.22.0 → 4.1.0). The design follows a "Super Path" pattern where paths have applicability ranges rather than storing separate snapshots per version.

## Schema

### DDVersion Node

```cypher
(:DDVersion {
    version: "4.1.0",
    release_date: "2024-01-15",
    path_count: 19136
})-[:PREDECESSOR]->(:DDVersion {version: "4.0.0"})
```

### DDPath Node

```cypher
(:DDPath {
    full_path: "equilibrium/time_slice/profiles_1d/psi",
    ids_name: "equilibrium",
    documentation: "Poloidal flux profile...",
    units: "Wb",
    data_type: "FLT_1D",
    embedding: [...]  // 384 floats for semantic search
})
```

### PathChange Node

Tracks metadata changes between versions:

```cypher
(:PathChange {
    path: "equilibrium/time_slice/profiles_1d/psi",
    from_version: "3.42.2",
    to_version: "4.0.0",
    change_type: "documentation",
    semantic_type: "sign_convention"
})
```

## Version Tracking

Instead of storing N copies of each path, we track:

1. **Introduction**: When path first appeared (`INTRODUCED_IN` relationship)
2. **Deprecation**: When path was removed (`DEPRECATED_IN` relationship)
3. **Changes**: Metadata changes between versions (`PathChange` nodes)
4. **Renames**: Path migrations (`RENAMED_TO` relationship)

## Semantic Change Detection

| Semantic Type | Detection Keywords | Impact |
|---------------|-------------------|--------|
| `sign_convention` | sign, positive, negative | **Critical** |
| `coordinate_system` | coordinate, reference frame | **Critical** |
| `normalization` | normalized, per unit | **High** |
| `clarification` | General doc improvements | Low |

## Build Process

```bash
# Build DD graph for specific version
uv run build-dd-graph --version 4.1.0

# Build all versions with change tracking
uv run build-dd-graph --all --track-changes
```

## Queries

```cypher
-- Version-specific lookup
MATCH (p:DDPath {full_path: "equilibrium/time_slice/profiles_1d/psi"})
MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
WHERE intro.version <= "3.42.0"
RETURN p.documentation, p.units

-- Sign convention changes
MATCH (c:PathChange {semantic_type: "sign_convention"})
WHERE c.from_version STARTS WITH "3." AND c.to_version STARTS WITH "4."
RETURN c.path, c.old_value, c.new_value
```
