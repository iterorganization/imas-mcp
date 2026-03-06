# Signal Structure Naming Generalization

**Status:** Proposal (v2 — revised)  
**Author:** Agent (on behalf of user)  
**Date:** 2025-06-03, revised 2026-03-06

## Problem Statement

The current graph schema uses MDSplus-specific terminology for concepts that are generic across all data systems:

| Current Name | Concept | MDSplus-Specific? |
|---|---|---|
| `MDSplusTree` | A hierarchical data container | Yes — JET uses XML, ITER uses IMAS IDS |
| `TreeModelVersion` | A structural epoch (schema revision) | Yes — "tree model" implies MDSplus tree |
| `TreeNode` | A data point within a hierarchy | Yes — "tree node" implies MDSplus tree structure |
| `TreeNodePattern` | Indexed repetition pattern | Yes — coupled to `TreeNode` |
| `TreeNodeType` | Node classification enum | Yes — STRUCTURE, SIGNAL are MDSplus terms |
| `TreeNodeSource` | How a node was discovered | Yes — `tree_introspection` value |
| `IN_TREE` | "belongs to this container" | Yes — implies tree |
| `SOURCE_NODE` | Signal provenance link | Neutral but undeclared in schema |
| `WRITES_TO_TREE` | Code output destination | Yes |
| `RESOLVES_TO_TREE_NODE` | TDI→data resolution | Yes |

JET's machine description lives in XML files versioned in git. ITER will use IMAS IDS. Future facilities may use HDF5 or SQL. The "tree" naming forces non-tree data into tree-shaped abstractions.

## Design Principles

1. **Generic names for generic concepts.** If a concept applies to all data systems (hierarchical data, structural epochs, data points), the name should not reference any specific system.
2. **Format-specific properties stay as properties, not labels.** The MDSplus tree name, XML file path, or IDS identifier is a *property* of a generic node, not a *type* of node.
3. **One rename, done right.** This is a greenfield project — rename everything once, update TCV's 34K nodes in a migration, and never look back.
4. **Preserve the proven architecture.** The epoch-based versioning, super-structure concept, and promote-to-signal pipeline are excellent. Only the names change.
5. **Use `data_source` consistently as a compound noun** — not bare `source` — to avoid ambiguity with code sources, discovery sources, evidence sources, and other "source" concepts in the codebase.

## Naming Analysis

### "Epoch" Usage Audit

"Epoch" appears throughout the codebase as a standalone concept:

| Location | Usage |
|---|---|
| `EpochConfig` (facility_config.yaml) | Config class for structural epoch detection |
| `detect_epochs` (facility_config.yaml) | Boolean flag on TreeConfig |
| `epoch_config` (facility_config.yaml) | Property referencing EpochConfig |
| `discovery/mdsplus/epochs.py` | Entire module — `detect_epochs_for_tree()`, `discover_epochs_optimized()` |
| `limiter_versions` (jet.yaml) | JET limiter contour version list (renamed from `limiter_epochs`) |
| `epoch_id` (graph_ops.py) | Variable constructing TreeModelVersion IDs |
| FacilitySignal description | "epoch when signal first appeared/disappeared" |

**IMAS does not use "epoch"** — the DD schema uses `DDVersion` with `HAS_PREDECESSOR` chains. The concept is analogous (versions of a schema that evolve over time) but IMAS calls them "versions", not "epochs".

**Verdict:** `StructuralEpoch` is a good name. It's precise, doesn't collide with IMAS terminology, and aligns with the existing `EpochConfig` patterns. The "structural" prefix distinguishes from other epoch uses (limiter epochs, shot epochs). No rename needed for this term.

### "DataSource" Collision Analysis

The name `DataSource` already exists heavily in the config schema layer:

| Existing Name | Layer | What It Is |
|---|---|---|
| `DataSourceType` | Graph enum | mdsplus, tdi, uda, hdf5, netcdf, imas, ppf, allas |
| `DataSourceBase` | Config mixin | Common fields (setup_commands, python_command) |
| `DataSourcesConfig` | Config class | Container: tdi, mdsplus, ppf, edas, hdf5, imas |
| `data_sources:` | YAML key | Top-level config section in facility YAMLs |

Using `DataSource` as both a **graph node label** and a **config mixin name** creates genuine ambiguity. An agent seeing `DataSource` won't know if we mean the Neo4j node or the config class.

**However**, `DataSourceBase` is a *config-layer mixin* (never appears in the graph), while the proposed `DataSource` would be a *graph node label* (never appears in config). The namespaces don't overlap at runtime, but they do overlap in developer mental models and grep results.

### Should We Use `data_source_*` Prefixes?

The question: could we have non-data "sources" in the future?

Current uses of "source" in the codebase:
- `discovery_source` — how a signal was found (tree_traversal, wiki_scrape, code_analysis)
- `enrichment_source` — who enriched a node (llm_agent, manual, wiki)
- `TreeNodeSource` — how a node was discovered (tree_introspection, code_extraction)
- `source_url` — evidence provenance
- `source_node_id` — evidence graph node reference
- `SourceFile` — a code file in the graph
- `source_format` — JET's XML format identifier

**"Source" alone is overloaded.** Five different meanings already exist. `source_name` without the `data_` prefix could mean any of these. `data_source_name` is unambiguous: it's the name of the data source this node belongs to.

**Verdict:** Use the `data_source_` prefix consistently. The slight verbosity buys permanent clarity. In Cypher queries, `n.data_source_name` is self-documenting; `n.source_name` is not.

### Will We Have Other Sources?

Yes. The `DataSourceType` enum already lists 8 types, and the `DataSourcesConfig` has slots for tdi, mdsplus, ppf, edas, hdf5, imas. Future additions are expected:
- **REST/API sources** (WEST, KSTAR have REST APIs)
- **SQL databases** (some facilities store metadata in SQL)
- **Object storage** (Allas for CSC, S3 for cloud)

All of these are *data* sources. The `data_` prefix groups them correctly and distinguishes from code sources (`SourceFile`), wiki sources (`WikiPage`), and evidence sources (`MappingEvidence`).

## Proposed Naming (v2)

### Tier 1: Graph Node Labels

| Current | Proposed | Rationale |
|---|---|---|
| `MDSplusTree` | `DataSource` | Despite config-layer collision, this is the correct domain term. The config classes are prefixed (`DataSourceBase`, `DataSourcesConfig`), and the graph label is used in a different context. `DataSource` is what agents and Cypher queries will see; `DataSourceBase` is what pipeline developers import. |
| `TreeModelVersion` | `StructuralEpoch` | Precise, aligns with existing `EpochConfig`. IMAS uses "version" not "epoch" — no collision. |
| `TreeNode` | `DataNode` | Format-agnostic hierarchical data point. |
| `TreeNodePattern` | `DataNodePattern` | Follows from DataNode. |
| `TreeNodeType` | `DataNodeType` | Enum values stay: STRUCTURE, SIGNAL, NUMERIC, TEXT (general enough). |
| `TreeNodeSource` | `DataNodeSource` | Values: tree_introspection → introspection, tdi_parameter → accessor_parameter. |

### Tier 2: Relationships

| Current | Proposed | Rationale |
|---|---|---|
| `IN_TREE` | `IN_DATA_SOURCE` | "belongs to this data source". Unambiguous — cannot be confused with "in source file" or "in source code". |
| `SOURCE_NODE` | `HAS_DATA_SOURCE_NODE` | FacilitySignal→DataNode provenance. The `data_source_` prefix distinguishes from `source_node_id` on MappingEvidence (which references WikiChunk/CodeChunk/TreeNode). Formalize in schema. |
| `WRITES_TO_TREE` | `WRITES_TO` | AnalysisCode→DataSource. Short and clear — no ambiguity about what an analysis code writes to. |
| `RESOLVES_TO_TREE_NODE` | `RESOLVES_TO_NODE` | DataReference→DataNode. Drop "tree". |
| `INTRODUCED_IN` | `INTRODUCED_IN` | No change — already generic. |
| `REMOVED_IN` | `REMOVED_IN` | No change — already generic. |
| `HAS_PREDECESSOR` | `HAS_PREDECESSOR` | No change — already generic. |
| `HAS_NODE` | `HAS_NODE` | No change — already generic (parent→child). |
| `HAS_ERROR` | `HAS_ERROR` | No change. |
| `FOLLOWS_PATTERN` | `FOLLOWS_PATTERN` | No change. |

### Tier 3: Properties

| Current | Proposed | Context | Rationale |
|---|---|---|---|
| `tree_name` (on DataNode) | `data_source_name` | Which DataSource this node belongs to | Self-documenting. `n.data_source_name` in Cypher is unambiguous. |
| `tree_name` (on StructuralEpoch) | `data_source_name` | Which DataSource this epoch belongs to | Consistent. |
| `tree_name` (on FacilitySignal) | `data_source_name` | MDSplus tree containing this signal | Consistent. |
| `tree_name` (on DataNodePattern) | `data_source_name` | Which DataSource this pattern belongs to | Consistent. |
| `node_path` (on FacilitySignal) | `data_source_path` | Full path within the data source | `s.data_source_path` is unambiguous vs `s.source_path` which could mean source file path. |
| `source_node` (ad-hoc on FacilitySignal) | `data_source_node` | ID of the DataNode backing this signal | Formalize in schema with `range: DataNode`, `relationship_type: HAS_DATA_SOURCE_NODE`. |

### Tier 4: Config Schema (facility_config.yaml)

| Current | Proposed | Rationale |
|---|---|---|
| `TreeConfig` | `SourceConfig` | Configuration for a data source (MDSplus tree, XML archive, HDF5 store). |
| `TreeVersion` | `SourceVersion` | Version definition within a data source config. |
| `TreeSystem` | `SourceSystem` | Named subsystem within a data source. |
| `tree_name` (property) | `source_name` | Config-layer property — shorter is fine here since context is unambiguous within `data_sources.mdsplus.trees[].source_name`. |

`MDSplusConfig`, `EpochConfig` stay unchanged — they are already correctly scoped as format-specific configuration.

### Tier 5: Vector Indexes

| Current | Proposed |
|---|---|
| `tree_node_desc_embedding` | `data_node_desc_embedding` |

Auto-generated from schema, no manual change needed — just rebuild after schema update.

## Static Value Properties on DataNode

The `DataNode` schema gains explicit typed properties for storing static/immutable values (machine geometry, hardware configuration). These are populated by scanners when the data is static and should be preserved in the graph — either for offline facilities (JET) or for static trees (TCV).

### Design Rationale

Why explicit typed properties instead of a JSON blob or ad-hoc properties:
1. **Schema-visible** — `get_graph_schema()` exposes them to agents automatically
2. **Pydantic-validated** — generated model catches type errors at ingest time
3. **Cypher-queryable** — `WHERE n.r > 4.0 AND n.z < 1.0` without JSON parsing
4. **Cross-facility** — same property names map the same physics across JET XML, TCV static trees, future facilities

### Field Inventory (SSH-verified)

**JET device XML sub-elements** (332 instances × 10 sections × 4 distinct XMLs):

| Section | Fields | Types |
|---------|--------|-------|
| magprobes (191) | r, z, angle, abs_error, rel_error | float scalars |
| flux (36) | r, z, dphi, abs_error, rel_error | float scalars |
| pfcoils (22) | r, z, dr, dz, turnsperelement, abs_error, rel_error | float/float arrays (multi-element coils have space-separated values) |
| pfcircuits (16) | coil_connect, supply_connect | int lists (topology) |
| pfsupplies (16) | abs_error, rel_error | float scalars |
| toroidalfield (1) | abs_error, rel_error | float scalars |
| plasmacurrent (1) | abs_error, rel_error | float scalars |
| diamagneticflux (1) | abs_error, rel_error | float scalars |
| pfpassive (48) | r, z, dr, dz, ang1, ang2, resistance, abs_error, rel_error | float scalars |

Instance-level attributes (metadata, not values): `id`, `archive`, `file`, `owner`, `seq`, `signal`, `status`, `factor`

**TCV static tree** (`\STATIC::` NUMERIC nodes in graph — 34,082 FacilitySignals via `tcv:mdsplus:static` DataAccess):

| Category | Leaf fields | Types |
|----------|-------------|-------|
| COIL (18 nodes) | R, Z, A, B, D, W, H, ANG, N, NA, NB, NT, RHO, XSECT, INOM, UNOM, TRANSI, TRANSU | float arrays (per-coil), shape varies |
| ACTIVE_COIL (5) | INOM, UNOM, N, TRANSI, TRANSU | float arrays |
| FLUX_LOOP (8) | R, Z, A, B, D, N, ANG | float arrays |
| FLUX_LOOP_3D (9) | R, Z, A, B, D, N, ANG, PHI | float arrays |
| VESSEL (24) | R, Z, A, B, D, N, ANG, RHO, W, S, PERIM, XSECT, R:IN, Z:IN, R:OUT, Z:OUT, PP:BREAKS, PP:COEFS | float arrays, nested structure |
| VESSEL_EXP (11) | R, Z, A, B, D, N, RHO, W, S, PERIM, XSECT | float arrays |
| TILES (8+) | R, Z, ANG, N, PERIM, S, PP:BREAKS, PP:COEFS | float arrays |
| SLOOP (9) | R, Z, A, B, D, N, ANG, PHI, TRANSU | float arrays |
| WINDING (9) | R, Z, A, B, D, N, ANG, RHO, XSECT | float arrays |
| MESH/COARSE_MESH/FAR_MESH/FBT_MESH (15 each) | R, Z, A, B, D, N, NA, NB, RHO, MESH | float arrays |
| PP_* probes (26 coil names × 8 each) | BREAKS, COEFS, IN, OUT, IN:BREAKS, IN:COEFS, OUT:BREAKS, OUT:COEFS | float arrays (magnetic probe transfer functions) |
| Greens matrix nodes | VAL, PRE | float matrices (require libjmmshr_gsl.so) |

### Proposed DataNode Schema Additions

Properties are grouped by physics category. All are optional (nullable) — only populated for nodes where the value is static/immutable.

```yaml
# === Geometry: Position ===
r:
  description: >-
    Major radius position [m]. Scalar for probes/loops, array for
    multi-element coils. JET: device XML r sub-element. TCV: COIL:VAL:R.
  range: float
z:
  description: >-
    Vertical position [m]. Same dimensionality as r.
  range: float
phi:
  description: >-
    Toroidal angle [rad]. For 3D flux loops, saddle coils.
    TCV: FLUX_LOOP_3D:VAL:PHI, SLOOP:VAL:PHI.
  range: float

# === Geometry: Extent ===
dr:
  description: >-
    Radial extent/width [m]. For rectangular cross-section elements
    (PF coils, passive structures). JET: pfcoils/pfpassive dr.
  range: float
dz:
  description: >-
    Vertical extent/height [m]. JET: pfcoils/pfpassive dz.
  range: float
width:
  description: >-
    Element width [m]. TCV: COIL:VAL:W, VESSEL:VAL:W.
  range: float
height:
  description: >-
    Element height [m]. TCV: COIL:VAL:H.
  range: float
cross_section:
  description: >-
    Cross-sectional area [m²]. TCV: COIL:VAL:XSECT, VESSEL:VAL:XSECT.
  range: float
perimeter:
  description: >-
    Perimeter length [m]. TCV: VESSEL:VAL:PERIM, TILES:VAL:PERIM.
  range: float
surface:
  description: >-
    Surface area [m²]. TCV: VESSEL:VAL:S, TILES:VAL:S.
  range: float

# === Geometry: Orientation ===
angle:
  description: >-
    Orientation angle [deg]. JET magprobes: probe tilt. TCV: COIL:VAL:ANG.
    For multiple angles, use angle, angle_2 convention or store as array.
  range: float
angle_2:
  description: >-
    Second orientation angle [deg]. JET pfpassive has ang1 + ang2.
  range: float
dphi:
  description: >-
    Toroidal angular extent. JET flux loops: dphi in units of pi.
  range: float

# === Electrical ===
turns:
  description: >-
    Number of turns per element. JET: pfcoils turnsperelement.
    TCV: COIL:VAL:N, ACTIVE_COIL:VAL:N (array per coil).
  range: float
resistance:
  description: >-
    Electrical resistance [Ohm]. JET: pfpassive resistance.
    TCV: via COIL:VAL:RHO (resistivity).
  range: float
resistivity:
  description: >-
    Electrical resistivity [Ohm·m]. TCV: COIL:VAL:RHO, VESSEL:VAL:RHO.
  range: float
nominal_current:
  description: >-
    Nominal operating current [A]. TCV: COIL:VAL:INOM, ACTIVE_COIL:VAL:INOM.
  range: float
nominal_voltage:
  description: >-
    Nominal operating voltage [V]. TCV: COIL:VAL:UNOM, ACTIVE_COIL:VAL:UNOM.
  range: float

# === Error/Uncertainty ===
abs_error:
  description: >-
    Absolute measurement uncertainty. JET: all sections have abs_error.
  range: float
rel_error:
  description: >-
    Relative measurement uncertainty [fraction]. JET: all sections have rel_error.
  range: float

# === Topology (connectivity) ===
coil_ids:
  description: >-
    Connected coil indices (circuit topology). JET: pfcircuits coil_connect.
    Stored as integer list.
  multivalued: true
  range: integer
supply_ids:
  description: >-
    Connected supply indices (circuit topology). JET: pfcircuits supply_connect.
  multivalued: true
  range: integer

# === Transfer Functions ===
transfer_current:
  description: >-
    Current transfer function coefficient. TCV: COIL/ACTIVE_COIL:VAL:TRANSI.
  range: float
transfer_voltage:
  description: >-
    Voltage transfer function coefficient. TCV: COIL/ACTIVE_COIL:VAL:TRANSU.
  range: float
transfer_inverse:
  description: >-
    Inverse transfer function. TCV: CRS_SEG:VAL:TRANSINV.
  range: float

# === Scaling/Calibration ===
factor:
  description: >-
    Scaling factor for measurement. JET: instance attribute 'factor'
    on pfsupplies, toroidalfield, diamagneticflux.
  range: float

# === Generic Scalar Fallback ===
scalar_value:
  description: >-
    Generic scalar value for static data that doesn't fit named properties.
    Use named properties (r, z, turns, etc.) whenever possible.
    This is a fallback for facility-specific fields that appear rarely.
  range: float
```

### Array-Valued Properties

Many TCV static tree values are arrays (e.g., `COIL:VAL:R` has shape `(32,)` — one R value per coil). Neo4j supports `multivalued: true` for list properties, but this creates complications:

- **Schema**: Each property would need both scalar and array variants, or always be `multivalued`
- **Queries**: `WHERE n.r > 4.0` doesn't work on arrays; need `ANY(x IN n.r WHERE x > 4.0)`
- **IMAS mapping**: Array indices map to IMAS array indices (e.g., `pf_active.coil[i].element[j].geometry.rectangle.r`)

**Recommendation**: Store scalars as scalar properties. For array data, store only when the node represents a single element (JET magprobe #1: `r=4.292`) not when it represents the whole collection. TCV `COIL:VAL:R` is a collection node — its children are individual coils that get scalar `r` values.

When a node genuinely represents an array (e.g., a limiter contour R,Z profile), use `multivalued: true`:

```yaml
r_contour:
  description: >-
    Array of R values defining a contour [m].
    Used for limiter outlines, vessel cross-sections.
  multivalued: true
  range: float
z_contour:
  description: >-
    Array of Z values defining a contour [m].
    Parallel array to r_contour.
  multivalued: true
  range: float
```

### Properties NOT Added to Schema

These exist in the source data but are stored as DataNode metadata or relationships, not as typed value properties:

| Field | Why Not a Value Property | Where It Goes |
|-------|------------------------|---------------|
| `archive`, `file`, `owner`, `seq`, `signal`, `status` | JET PPF routing metadata, not physics values | DataNode.description or custom metadata property |
| `BREAKS`, `COEFS` | Spline coefficients — large arrays, not queried by value | Store if needed as `multivalued` blob, but likely catalog-only (DataAccess) |
| Greens matrices (`VAL`, `PRE`) | Large float matrices, computed on-demand | Catalog-only — `DataAccess` template shows how to compute |
| `MESH` grid data | Computational mesh, not measurement geometry | Catalog-only |
| `DIM` labels | Text metadata describing array dimensions | DataNode.description |

## Format-Specific Properties

The question: should we have format-specific nodes (e.g., `MDSplusDataNode` vs `XMLDataNode`) or format-specific *properties* on generic nodes?

**Recommendation: Format-specific properties on generic nodes.**

A `DataNode` gains its format identity through:
- `source_type` property on its parent `DataSource` node
- Format-specific properties stored as regular node properties

```
DataNode {
  path: "\\RESULTS::LIUQE:PSI",       # MDSplus path
  data_source_name: "results",         # Which DataSource
  node_type: SIGNAL,                   # Classification

  # Format-specific (nullable)
  mdsplus_usage: "SIGNAL",             # MDSplus-specific
  xml_xpath: null,                     # XML-specific
  hdf5_dataset: null,                  # HDF5-specific
}
```

**Why not subtyped labels?**

1. Neo4j doesn't support schema inheritance — `(:MDSplusDataNode)` and `(:XMLDataNode)` would be independent labels with no shared query path.
2. Every Cypher query would need `MATCH (n:MDSplusDataNode) UNION MATCH (n:XMLDataNode)` instead of `MATCH (n:DataNode)`.
3. The vector index `data_node_desc_embedding` would need to exist per subtype.
4. The promote-to-signal pipeline would need per-type variants.

Instead, format-specific behavior lives in:
- **Config YAML**: `MDSplusConfig` (how to extract), `source_format: git_xml` (JET)
- **Scanner plugins**: `MDSplusScanner`, `XMLScanner`, `HDF5Scanner` (how to discover)
- **DataSource.source_type**: property on the DataSource graph node (MDSplus, XML, HDF5, IMAS)

The `DataNode` is always just a `DataNode`. The scanner that created it knows the format, but the graph queries don't need to.

## Scope Assessment

### Files Requiring Changes

Based on the comprehensive audit:

| Category | Files | Estimated Changes |
|---|---|---|
| **LinkML Schemas** | 3 files | ~8 class/enum renames, ~20 property renames |
| **Auto-generated** (rebuild only) | 4 files | `uv run build-models --force` |
| **Core pipeline** — graph_ops.py | 1 file | ~50 Cypher query updates |
| **Core pipeline** — extraction.py | 1 file | ~15 Cypher query updates |
| **Core pipeline** — batch_discovery.py | 1 file | ~8 Cypher query updates |
| **Discovery** — parallel.py (signals) | 1 file | ~40 Cypher + variable renames |
| **Discovery** — parallel.py (static) | 1 file | ~10 Cypher updates |
| **Discovery** — pipeline.py, workers.py, epochs.py | 3 files | ~15 label references |
| **Discovery** — tdi_linkage.py | 1 file | ~8 Cypher updates |
| **Discovery** — progress.py | 1 file | ~5 field renames |
| **Agentic** — search_tools.py, server.py, tools.py, enrich.py | 4 files | ~20 Cypher updates |
| **Graph** — domain_queries.py, schema_context.py, schema.py, __init__.py | 4 files | ~15 label/import changes |
| **CLI** — enrich.py, ingest.py, discover/*.py | 4 files | ~12 label references |
| **Tests** | 12 files | ~80 label/mock/assertion updates |
| **Scripts** | 3 files | ~10 Cypher updates |
| **Documentation** | 10 files | ~60 text references |
| **Config YAML** | facility_config.yaml | ~6 class renames |
| **Facility YAMLs** | tcv.yaml, jet.yaml | Property name updates |
| **Plans/agents** | 8 files | ~30 references |

**Total: ~55 source files, ~400 individual string replacements.**

### Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| TCV production graph needs migration | Medium | Migration script tested on archive first |
| ~400 string replacements across 55 files | Medium | Systematic find-replace with tests as validation |
| Vector index name change | Low | Auto-generated from schema, just rebuild |
| Other agents' code references old names | Medium | Schema reference doc auto-regenerates |
| Possible missed references | Low | Grep + test suite + vector index rebuild as safety net |
| `DataSource` label vs `DataSourceBase` config mixin | Low | Different layers — graph vs config. Docstrings clarify. |

## Implementation Plan

### Phase 1: Schema + Models

**Goal:** Single source of truth updated; all generated code rebuilt.

1. Rename classes in `facility.yaml`:
   - `MDSplusTree` → `DataSource`
   - `TreeModelVersion` → `StructuralEpoch`
   - `TreeNode` → `DataNode`
   - `TreeNodePattern` → `DataNodePattern`
2. Rename enums:
   - `TreeNodeType` → `DataNodeType`
   - `TreeNodeSource` → `DataNodeSource`
   - Update `tree_introspection` → `introspection`, `tdi_parameter` → `accessor_parameter`
3. Update relationship annotations:
   - `IN_TREE` → `IN_DATA_SOURCE`
   - `WRITES_TO_TREE` → `WRITES_TO`
   - `RESOLVES_TO_TREE_NODE` → `RESOLVES_TO_NODE`
4. Rename properties across all schema classes:
   - `tree_name` → `data_source_name` (on DataNode, StructuralEpoch, DataNodePattern, FacilitySignal)
   - `node_path` → `data_source_path` (on FacilitySignal)
5. Add `data_source_node` slot to FacilitySignal:
   - `range: DataNode`, `annotations: { relationship_type: HAS_DATA_SOURCE_NODE }`
6. Update config schema (`facility_config.yaml`):
   - `TreeConfig` → `SourceConfig`
   - `TreeVersion` → `SourceVersion`
   - `TreeSystem` → `SourceSystem`
   - `tree_name` → `source_name` (config property)
7. Update description references in `common.yaml`, `task_groups.yaml`
8. Rebuild: `uv run build-models --force`

### Phase 2: Core Pipeline

**Goal:** All discovery/ingestion code uses new labels.

1. `discovery/mdsplus/graph_ops.py` — ~50 Cypher label/property/relationship updates
2. `mdsplus/extraction.py` — ~15 updates
3. `mdsplus/batch_discovery.py` — ~8 updates
4. `discovery/mdsplus/pipeline.py`, `workers.py`, `epochs.py` — ~15 updates
5. `discovery/static/parallel.py` — ~10 updates
6. `discovery/signals/parallel.py` — ~40 updates (largest single file)
7. `discovery/signals/progress.py` — `signals_in_tree` field rename consideration
8. `discovery/mdsplus/tdi_linkage.py` — ~8 updates
9. `discovery/base/engine.py` — orphan labels list

### Phase 3: Agentic + Graph Layer

**Goal:** MCP tools, search, and REPL use new labels.

1. `agentic/search_tools.py` — Cypher queries, vector index name
2. `agentic/server.py` — MCP tool definitions, example queries
3. `agentic/tools.py` — Cypher queries
4. `agentic/enrich.py` — enrichment Cypher
5. `graph/domain_queries.py` — `find_tree_nodes()` → `find_data_nodes()`
6. `graph/schema_context.py` — schema examples
7. `graph/__init__.py` — imports and `__all__`
8. `graph/schema.py` — docstrings

### Phase 4: CLI + Tests

**Goal:** All user-facing commands and test suite green.

1. `cli/enrich.py` — label references
2. `cli/ingest.py` — output labels
3. `cli/discover/signals.py` — scanner references
4. All 12 test files — labels, mocks, assertions
5. Full test suite must pass

### Phase 5: Documentation, Configs, and Live Graph Migration

**Goal:** All references updated. Production graph migrated.

1. Update 10 documentation files (docs/architecture/*.md, docs/api/REPL_API.md)
2. Update 8 plans/agents files
3. Update facility YAMLs: `tree_name` → `source_name` in tcv.yaml, jet.yaml
4. **Write migration script** (see below)
5. **Execute migration sequence** (see below)

### Live Graph Migration Steps

The production graph contains TCV data (34,082 FacilitySignal, ~50K TreeNode, 8 TreeModelVersion, ~5 MDSplusTree nodes). Migration must be atomic and tested.

#### Pre-Migration

```bash
# 1. Backup current graph
uv run imas-codex graph backup

# 2. Export archive for testing
uv run imas-codex graph export -o pre-migration-backup.tar.gz

# 3. Load archive into test graph location for dry-run
uv run imas-codex graph load pre-migration-backup.tar.gz -g codex-test
```

#### Migration Script (execute via `graph shell` or Python)

```cypher
// ============================================================
// STEP 1: Add new labels (non-destructive, idempotent)
// ============================================================
// Each node gets the new label alongside the old one.
// This means queries using EITHER label will work during transition.

CALL apoc.periodic.iterate(
  'MATCH (n:MDSplusTree) RETURN n',
  'SET n:DataSource',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeModelVersion) RETURN n',
  'SET n:StructuralEpoch',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeNode) RETURN n',
  'SET n:DataNode',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeNodePattern) RETURN n',
  'SET n:DataNodePattern',
  {batchSize: 1000}
);

// ============================================================
// STEP 2: Rename properties (idempotent via COALESCE)
// ============================================================
// For nodes that have tree_name but not yet data_source_name.

CALL apoc.periodic.iterate(
  'MATCH (n:DataNode) WHERE n.tree_name IS NOT NULL AND n.data_source_name IS NULL RETURN n',
  'SET n.data_source_name = n.tree_name REMOVE n.tree_name',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:StructuralEpoch) WHERE n.tree_name IS NOT NULL AND n.data_source_name IS NULL RETURN n',
  'SET n.data_source_name = n.tree_name REMOVE n.tree_name',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (s:FacilitySignal) WHERE s.tree_name IS NOT NULL AND s.data_source_name IS NULL RETURN s',
  'SET s.data_source_name = s.tree_name, s.data_source_path = s.node_path REMOVE s.tree_name, s.node_path',
  {batchSize: 1000}
);

// Rename source_node → data_source_node on FacilitySignal
CALL apoc.periodic.iterate(
  'MATCH (s:FacilitySignal) WHERE s.source_node IS NOT NULL AND s.data_source_node IS NULL RETURN s',
  'SET s.data_source_node = s.source_node REMOVE s.source_node',
  {batchSize: 1000}
);

// ============================================================
// STEP 3: Recreate relationships with new types
// ============================================================
// Neo4j cannot rename relationship types in-place.
// Create new, delete old. Use APOC for batching.

// IN_TREE → IN_DATA_SOURCE
CALL apoc.periodic.iterate(
  'MATCH (n)-[r:IN_TREE]->(t) RETURN r, n, t',
  'CREATE (n)-[:IN_DATA_SOURCE]->(t) DELETE r',
  {batchSize: 1000}
);

// SOURCE_NODE → HAS_DATA_SOURCE_NODE
CALL apoc.periodic.iterate(
  'MATCH (s)-[r:SOURCE_NODE]->(n) RETURN r, s, n',
  'CREATE (s)-[:HAS_DATA_SOURCE_NODE]->(n) DELETE r',
  {batchSize: 1000}
);

// WRITES_TO_TREE → WRITES_TO
CALL apoc.periodic.iterate(
  'MATCH (a)-[r:WRITES_TO_TREE]->(t) RETURN r, a, t',
  'CREATE (a)-[:WRITES_TO]->(t) DELETE r',
  {batchSize: 500}
);

// RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE
CALL apoc.periodic.iterate(
  'MATCH (d)-[r:RESOLVES_TO_TREE_NODE]->(n) RETURN r, d, n',
  'CREATE (d)-[:RESOLVES_TO_NODE]->(n) DELETE r',
  {batchSize: 500}
);

// ============================================================
// STEP 4: Remove old labels
// ============================================================
CALL apoc.periodic.iterate(
  'MATCH (n:MDSplusTree) RETURN n',
  'REMOVE n:MDSplusTree',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeModelVersion) RETURN n',
  'REMOVE n:TreeModelVersion',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeNode) RETURN n',
  'REMOVE n:TreeNode',
  {batchSize: 1000}
);

CALL apoc.periodic.iterate(
  'MATCH (n:TreeNodePattern) RETURN n',
  'REMOVE n:TreeNodePattern',
  {batchSize: 1000}
);

// ============================================================
// STEP 5: Recreate vector index
// ============================================================
DROP INDEX tree_node_desc_embedding IF EXISTS;

CREATE VECTOR INDEX data_node_desc_embedding IF NOT EXISTS
FOR (n:DataNode) ON n.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};
```

#### Post-Migration Validation

```bash
# 4. Verify node counts match pre-migration
uv run imas-codex graph shell <<'EOF'
MATCH (n:DataSource) RETURN 'DataSource' AS label, count(n) AS count
UNION ALL
MATCH (n:StructuralEpoch) RETURN 'StructuralEpoch', count(n)
UNION ALL
MATCH (n:DataNode) RETURN 'DataNode', count(n)
UNION ALL
MATCH (n:DataNodePattern) RETURN 'DataNodePattern', count(n)
UNION ALL
MATCH (n:FacilitySignal) RETURN 'FacilitySignal', count(n);
EOF

# 5. Verify no old labels remain
uv run imas-codex graph shell <<'EOF'
MATCH (n:MDSplusTree) RETURN 'MDSplusTree STILL EXISTS' AS warning, count(n)
UNION ALL
MATCH (n:TreeModelVersion) RETURN 'TreeModelVersion STILL EXISTS', count(n)
UNION ALL
MATCH (n:TreeNode) RETURN 'TreeNode STILL EXISTS', count(n);
EOF

# 6. Verify no old relationships remain
uv run imas-codex graph shell <<'EOF'
MATCH ()-[r:IN_TREE]->() RETURN 'IN_TREE STILL EXISTS' AS warning, count(r)
UNION ALL
MATCH ()-[r:SOURCE_NODE]->() RETURN 'SOURCE_NODE STILL EXISTS', count(r)
UNION ALL
MATCH ()-[r:WRITES_TO_TREE]->() RETURN 'WRITES_TO_TREE STILL EXISTS', count(r)
UNION ALL
MATCH ()-[r:RESOLVES_TO_TREE_NODE]->() RETURN 'RESOLVES_TO_TREE_NODE STILL EXISTS', count(r);
EOF

# 7. Verify vector index is online
uv run imas-codex graph shell -c "SHOW INDEXES YIELD name, state WHERE name = 'data_node_desc_embedding' RETURN name, state"

# 8. Verify new relationships exist
uv run imas-codex graph shell <<'EOF'
MATCH ()-[r:IN_DATA_SOURCE]->() RETURN 'IN_DATA_SOURCE' AS rel, count(r) AS count
UNION ALL
MATCH ()-[r:HAS_DATA_SOURCE_NODE]->() RETURN 'HAS_DATA_SOURCE_NODE', count(r)
UNION ALL
MATCH ()-[r:WRITES_TO]->() RETURN 'WRITES_TO', count(r);
EOF

# 9. Spot-check a signal's full traversal
uv run imas-codex graph shell <<'EOF'
MATCH (s:FacilitySignal {facility_id: 'tcv'})
WHERE s.data_source_name IS NOT NULL
WITH s LIMIT 1
OPTIONAL MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(dn:DataNode)
OPTIONAL MATCH (dn)-[:IN_DATA_SOURCE]->(ds:DataSource)
RETURN s.id, s.data_source_name, s.data_source_path,
       dn.path, ds.name;
EOF

# 10. Push updated graph
uv run imas-codex graph push --dev
```

## Clarifications

### What Are `limiter_epochs`?

`limiter_epochs` in jet.yaml lists the 7 historical **first-wall contour geometries** used at JET. Each entry names a limiter design (e.g., `L91NEW`, `Mk1`, `Mk2A`, `Mk2GB`, `Mk2HD`, `Mk2ILW`) with the shot range during which it was installed. The limiter contour defines the R,Z boundary that plasmas touch — it changed independently of the device XML configurations (magnetic probes, flux loops, PF coils).

These are **not** structural epochs of a data source. They're hardware configuration eras for one component (the limiter). In EFIT, each limiter contour is a separate R,Z text file (`Limiters/limiter.mk2ilw_cc` etc.), and each device_xml version references one of these files.

**Why the name is unclear:** "epoch" is overloaded — it already means "structural version of a data source" in our naming system. Calling limiter geometry changes "epochs" conflates hardware installation history with data source schema versioning.

**Recommendation:** Rename `limiter_epochs` → `limiter_versions` in jet.yaml. A limiter version is a specific first-wall geometry that was physically installed. This distinguishes it from `StructuralEpoch` (a schema revision of a data source) and from `EpochConfig` (configuration for automated epoch detection). The `versions` list under `device_xml` already uses `version` terminology for the same concept at the whole-machine level — `limiter_versions` follows the same pattern for a single component.

## Dependency: Naming Generalization Before JET Cataloging

The JET graph cataloging plan — ingesting JET's machine description signals into the graph — requires implementing a new scanner plugin for git-versioned XML data sources. This plugin will create `DataSource`, `StructuralEpoch`, and `DataNode` graph nodes.

**The naming generalization must be completed first.** Here's why:

1. **New code should use final names.** Writing the JET XML scanner with `TreeNode`, `TreeModelVersion`, `MDSplusTree` and then immediately renaming everything doubles the work. The scanner should emit `DataNode`, `StructuralEpoch`, `DataSource` from day one.

2. **Schema slots needed for JET don't exist yet.** The current `MDSplusTree` class assumes MDSplus-specific properties (`server_hostname`, `shot_dependent`). The generic `DataSource` class will add `source_format` (git_xml, mdsplus_tree, imas_ids) and drop the MDSplus-specific fields to optional. JET's XML scanner needs these generic properties.

3. **Relationship types must be final.** The XML scanner will create `IN_DATA_SOURCE` relationships to link `DataNode` → `DataSource`. Writing `IN_TREE` now and migrating later is unnecessary churn.

4. **Config schema rename is a prerequisite.** The scanner will read from `SourceConfig` (currently `TreeConfig`). The JET YAML already uses the `device_xml` data source section, but the config schema needs its `TreeConfig` → `SourceConfig` rename before a non-MDSplus scanner can use it naturally.

### Execution Order

```
Phase 1: Naming Generalization (schema + code + migration)
   ├── Rename LinkML schemas
   ├── Update all Python code (~400 replacements)
   ├── Migrate TCV production graph data
   └── Verify: test suite green, graph validated

Phase 2: JET Graph Cataloging (new feature)
   ├── Implement XML scanner plugin (uses DataSource, DataNode, StructuralEpoch)
   ├── Parse device XMLs → DataNode (PF coils, probes, flux loops, circuits)
   ├── Parse device_ppfs namelists → DataNode (PPF DDA mappings)
   ├── Parse limiter contour files → DataNode (R,Z boundary points)
   ├── Create StructuralEpoch per device_xml version (p68613–p90626)
   ├── Promote leaf DataNodes to FacilitySignal (reuse existing pipeline)
   └── IMAS mapping for JET machine description signals
```

### JET Graph Cataloging Plan (Updated for New Naming)

The earlier JET cataloging plan used MDSplus-specific terminology. Here it is updated for the generalized naming:

#### Step 1: Create DataSource Nodes

```cypher
// JET machine description data source
MERGE (ds:DataSource {name: 'device_xml'})
SET ds.facility_id = 'jet',
    ds.source_format = 'git_xml',
    ds.description = 'JET machine description — geometry from device XMLs, namelists, and limiter contours',
    ds.shot_dependent = false
WITH ds
MATCH (f:Facility {id: 'jet'})
MERGE (ds)-[:AT_FACILITY]->(f);
```

#### Step 2: Create StructuralEpoch Nodes (10 Versions)

```cypher
// One StructuralEpoch per device_xml version
UNWIND $versions AS v
MERGE (e:StructuralEpoch {id: 'jet:device_xml:' + v.version})
SET e.data_source_name = 'device_xml',
    e.facility_id = 'jet',
    e.version = v.version,
    e.first_shot = v.first_shot,
    e.last_shot = v.last_shot,
    e.description = v.description,
    e.status = 'discovered'
WITH e
MATCH (ds:DataSource {name: 'device_xml', facility_id: 'jet'})
MERGE (e)-[:IN_DATA_SOURCE]->(ds);

// Build HAS_PREDECESSOR chain
// (p78359 → p68613, p79854 → p78359, ...)
```

#### Step 3: Parse & Ingest DataNodes

The XML scanner reads device XMLs and creates one `DataNode` per parameter:

```cypher
// Example: PF coil R coordinate
MERGE (n:DataNode {path: 'jet:device_xml:PF:P1:R'})
SET n.data_source_name = 'device_xml',
    n.facility_id = 'jet',
    n.node_type = 'NUMERIC',
    n.source = 'introspection',
    n.description = 'Major radius of PF coil P1',
    n.unit = 'm'
WITH n
MATCH (ds:DataSource {name: 'device_xml', facility_id: 'jet'})
MERGE (n)-[:IN_DATA_SOURCE]->(ds);

// Link to StructuralEpoch for applicability
MATCH (n:DataNode {path: 'jet:device_xml:PF:P1:R'})
MATCH (e:StructuralEpoch {id: 'jet:device_xml:p68613'})
MERGE (n)-[:INTRODUCED_IN]->(e);
```

#### Step 4: Promote to FacilitySignal

```cypher
// Reuse the existing promote pipeline — works identically
// because it now operates on DataNode → FacilitySignal
MATCH (n:DataNode {facility_id: 'jet', data_source_name: 'device_xml'})
WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
AND NOT EXISTS { (n)<-[:HAS_DATA_SOURCE_NODE]-(:FacilitySignal) }
// ... promote logic creates FacilitySignal with HAS_DATA_SOURCE_NODE edge
```

#### Step 5: IMAS Mapping

Map JET machine description signals to IMAS paths (e.g., `pf_active.coil[*].element[*].geometry.rectangle.r` ← JET PF coil R coordinate).

## Summary

The revised naming uses `data_source` as a consistent compound noun prefix across labels, relationships, and properties:

```
MDSplusTree      → DataSource
TreeModelVersion → StructuralEpoch
TreeNode         → DataNode
TreeNodePattern  → DataNodePattern
IN_TREE          → IN_DATA_SOURCE
SOURCE_NODE      → HAS_DATA_SOURCE_NODE (formalized in schema)
WRITES_TO_TREE   → WRITES_TO
RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE
tree_name        → data_source_name  (graph property)
tree_name        → source_name       (config property — shorter, unambiguous in context)
node_path        → data_source_path
source_node      → data_source_node
limiter_epochs   → limiter_versions  (jet.yaml — hardware versions, not data epochs)
```

~55 files, ~400 replacements, 5 phases. Architecture unchanged. Production migration is batched, idempotent, and validated.

**Naming generalization is a prerequisite for JET graph cataloging.** New scanner plugins should emit `DataSource`/`DataNode`/`StructuralEpoch` from day one, not `MDSplusTree`/`TreeNode`/`TreeModelVersion` that would need immediate re-migration.
