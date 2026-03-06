# Signal Structure Naming Generalization

**Status:** Proposal  
**Author:** Agent (on behalf of user)  
**Date:** 2025-06-03

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

## Proposed Naming

### Tier 1: Graph Node Labels (Schema Classes)

| Current | Proposed | Rationale |
|---|---|---|
| `MDSplusTree` | `DataSource` | A hierarchical data container — could be an MDSplus tree, HDF5 file group, XML document, or IMAS IDS. "DataSource" is the standard term for "where data lives." |
| `TreeModelVersion` | `StructuralEpoch` | A snapshot of the data structure at a point in time. "Epoch" is already used in the codebase (epoch_config, EpochConfig), and "structural" distinguishes it from shot/version epochs. |
| `TreeNode` | `DataNode` | A single data point or structure within a hierarchy. "DataNode" is format-agnostic and self-explanatory. |
| `TreeNodePattern` | `DataNodePattern` | Follows from DataNode. |
| `TreeNodeType` | `DataNodeType` | Follows from DataNode. Values: STRUCTURE → STRUCTURE, SIGNAL → SIGNAL, NUMERIC → NUMERIC (these are general enough). |
| `TreeNodeSource` | `DataNodeSource` | Follows from DataNode. Values: tree_introspection → introspection, code_extraction stays, tdi_parameter → accessor_parameter, manual stays. |

**Why `DataSource` over alternatives:**

| Candidate | Rejected because |
|---|---|
| `DataContainer` | Sounds like a runtime object, not a persistent store |
| `DataStore` | Too close to "data store" (database), implies persistence layer |
| `DataTree` | Still "tree" |
| `DataCatalog` | Implies a registry, not a single container |
| `DataSource` | ✅ Standard term. Clear. Used in facility configs already (`data_sources:` in YAML). |

**Why `StructuralEpoch` over alternatives:**

| Candidate | Rejected because |
|---|---|
| `SchemaVersion` | Conflicts with software schema versioning |
| `DataVersion` | Too vague — version of data vs version of structure? |
| `ConfigurationEpoch` | Too long, and "configuration" is overloaded |
| `ModelVersion` | "Model" is ambiguous (ML model? physics model?) |
| `StructuralEpoch` | ✅ Precise: structure of the data changed. "Epoch" already used in codebase. |

**Why `DataNode` over alternatives:**

| Candidate | Rejected because |
|---|---|
| `DataPoint` | Implies a single scalar, not a hierarchical element |
| `DataElement` | Not bad, but "node" matches the hierarchical structure and is already natural |
| `DataEntry` | Implies database row |
| `DataNode` | ✅ Generic. Hierarchical. Clear. |

### Tier 2: Relationships

| Current | Proposed | Rationale |
|---|---|---|
| `IN_TREE` | `IN_SOURCE` | "belongs to this data source". Applies to DataNode→DataSource and StructuralEpoch→DataSource. |
| `SOURCE_NODE` | `HAS_SOURCE_NODE` | FacilitySignal→DataNode provenance. Direction: signal HAS a source node. Formalize in schema. |
| `WRITES_TO_TREE` | `WRITES_TO` | AnalysisCode→DataSource. Drop "tree". |
| `RESOLVES_TO_TREE_NODE` | `RESOLVES_TO_NODE` | DataReference→DataNode. Drop "tree". |
| `INTRODUCED_IN` | `INTRODUCED_IN` | No change — already generic. |
| `REMOVED_IN` | `REMOVED_IN` | No change — already generic. |
| `HAS_PREDECESSOR` | `HAS_PREDECESSOR` | No change — already generic. |
| `HAS_NODE` | `HAS_NODE` | No change — already generic (parent→child). |
| `HAS_ERROR` | `HAS_ERROR` | No change. |
| `FOLLOWS_PATTERN` | `FOLLOWS_PATTERN` | No change. |

### Tier 3: Properties on FacilitySignal

| Current | Proposed | Rationale |
|---|---|---|
| `tree_name` | `source_name` | Generic: which DataSource this signal comes from. |
| `node_path` | `source_path` | Generic: path within the DataSource. |
| `source_node` (ad-hoc) | `source_node` | Keep — formalize in schema as slot with `range: DataNode`. |

### Tier 4: Config Schema (facility_config.yaml)

The config schema already uses reasonably generic names (`TreeConfig`, `TreeVersion`, `TreeSystem`). These map to MDSplus-specific *configuration* of the generic graph structure. The rename here is lighter:

| Current | Proposed | Rationale |
|---|---|---|
| `TreeConfig` | `SourceConfig` | Configuration for a data source (MDSplus tree, XML archive, HDF5 store). |
| `TreeVersion` | `SourceVersion` | Version definition within a data source config. |
| `TreeSystem` | `SourceSystem` | Named subsystem within a data source. |
| `tree_name` (property) | `source_name` | Consistent with graph property rename. |

`MDSplusConfig`, `EpochConfig` stay unchanged — they are already correctly scoped as MDSplus-specific configuration. `MDSplusConfig` is the MDSplus-specific data source configuration, not a generic label.

### Tier 5: Vector Indexes

| Current | Proposed |
|---|---|
| `tree_node_desc_embedding` | `data_node_desc_embedding` |

Auto-generated from schema, no manual change needed — just rebuild after schema update.

## Format-Specific Properties

The question: should we have format-specific nodes (e.g., `MDSplusDataNode` vs `XMLDataNode`) or format-specific *properties* on generic nodes?

**Recommendation: Format-specific properties on generic nodes.**

A `DataNode` gains its format identity through:
- `source_type` property (already exists on `DataSource` via the `source_format` we added to JET's config)
- Format-specific properties stored as regular node properties

```
DataNode {
  path: "\\RESULTS::LIUQE:PSI",       # MDSplus path
  source_name: "results",              # Which DataSource
  node_type: SIGNAL,                   # MDSplus usage type
  
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
- **DataSource.source_type**: property on the DataSource node (MDSplus, XML, HDF5, IMAS)

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

### Graph Migration (Production Data)

TCV has 34,082 signals and 8 TreeModelVersions in the production graph. Migration requires:

```cypher
-- Phase 1: Add new labels alongside old
MATCH (n:MDSplusTree) SET n:DataSource;
MATCH (n:TreeModelVersion) SET n:StructuralEpoch;
MATCH (n:TreeNode) SET n:DataNode;
MATCH (n:TreeNodePattern) SET n:DataNodePattern;

-- Phase 2: Rename properties
MATCH (n:DataNode) SET n.source_name = n.tree_name REMOVE n.tree_name;
MATCH (n:StructuralEpoch) SET n.source_name = n.tree_name REMOVE n.tree_name;
MATCH (s:FacilitySignal) SET s.source_name = s.tree_name, s.source_path = s.node_path
  REMOVE s.tree_name, s.node_path;

-- Phase 3: Recreate relationships with new types
// IN_TREE → IN_SOURCE
MATCH (n)-[r:IN_TREE]->(t) CREATE (n)-[:IN_SOURCE]->(t) DELETE r;
// SOURCE_NODE → HAS_SOURCE_NODE
MATCH (s)-[r:SOURCE_NODE]->(n) CREATE (s)-[:HAS_SOURCE_NODE]->(n) DELETE r;
// WRITES_TO_TREE → WRITES_TO
MATCH (a)-[r:WRITES_TO_TREE]->(t) CREATE (a)-[:WRITES_TO]->(t) DELETE r;
// RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE
MATCH (d)-[r:RESOLVES_TO_TREE_NODE]->(n) CREATE (d)-[:RESOLVES_TO_NODE]->(n) DELETE r;

-- Phase 4: Remove old labels
MATCH (n:MDSplusTree) REMOVE n:MDSplusTree;
MATCH (n:TreeModelVersion) REMOVE n:TreeModelVersion;
MATCH (n:TreeNode) REMOVE n:TreeNode;
MATCH (n:TreeNodePattern) REMOVE n:TreeNodePattern;

-- Phase 5: Recreate vector index
DROP INDEX tree_node_desc_embedding IF EXISTS;
CREATE VECTOR INDEX data_node_desc_embedding IF NOT EXISTS
FOR (n:DataNode) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
```

This migration is reversible and can be tested on a graph archive before production.

### Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| TCV production graph needs migration | Medium | Migration script tested on archive first |
| ~400 string replacements across 55 files | Medium | Systematic find-replace with tests as validation |
| Vector index name change | Low | Auto-generated from schema, just rebuild |
| Other agents' code references old names | Medium | Schema reference doc auto-regenerates |
| Possible missed references | Low | Grep + test suite as safety net |

## Implementation Plan

### Phase 1: Schema (1 PR)
1. Rename classes in `facility.yaml`: MDSplusTree→DataSource, TreeModelVersion→StructuralEpoch, TreeNode→DataNode, TreeNodePattern→DataNodePattern
2. Rename enums: TreeNodeType→DataNodeType, TreeNodeSource→DataNodeSource
3. Update relationship annotations: IN_TREE→IN_SOURCE, WRITES_TO_TREE→WRITES_TO, RESOLVES_TO_TREE_NODE→RESOLVES_TO_NODE
4. Add `source_node` slot to FacilitySignal with `range: DataNode`, `relationship_type: HAS_SOURCE_NODE`
5. Rename `tree_name` → `source_name`, `node_path` → `source_path` on FacilitySignal
6. Update config schema: TreeConfig→SourceConfig, TreeVersion→SourceVersion, TreeSystem→SourceSystem
7. Update `common.yaml` and `task_groups.yaml` descriptions
8. `uv run build-models --force` — regenerate all models
9. Run full test suite (expect failures — this is the baseline)

### Phase 2: Core Pipeline (1 PR)
1. Update all Cypher queries in `graph_ops.py`, `extraction.py`, `batch_discovery.py`
2. Update `tdi_linkage.py`
3. Update variable names where they reference `tree_name` → `source_name`
4. Update `discovery/mdsplus/pipeline.py`, `workers.py`, `epochs.py`
5. Update `discovery/static/parallel.py`
6. Update `discovery/signals/parallel.py`, `progress.py`
7. Update `discovery/base/engine.py` orphan labels

### Phase 3: Agentic + Graph Layer (1 PR)
1. Update `search_tools.py`, `server.py`, `tools.py`, `enrich.py`
2. Update `domain_queries.py`, `schema_context.py`
3. Update `graph/__init__.py` imports and `__all__`
4. Update `graph/schema.py` docstrings

### Phase 4: CLI + Tests (1 PR)
1. Update `cli/enrich.py`, `cli/ingest.py`, `cli/discover/signals.py`
2. Update all 12 test files
3. Run full test suite — all tests must pass

### Phase 5: Documentation + Migration (1 PR)
1. Update 10 documentation files
2. Update 8 plans/agents files
3. Update facility YAMLs (tcv.yaml, jet.yaml) property names
4. Write and test graph migration script
5. Execute migration on production graph
6. Verify with `imas-codex graph status`

## Open Questions

1. **`DataSource` collision?** The name `DataSource` is used in `data_sources:` config key. Is `DataSource` (graph label) vs `data_sources` (YAML config section) confusing? Alternative: `SignalSource`. But `DataSource` aligns better with the existing config vocabulary.

2. **`source_name` vs `data_source`?** For the property on DataNode that references which DataSource it belongs to: `source_name` (parallel to old `tree_name`) or `data_source` (matches the graph label)? The slot `range: DataSource` with relationship `IN_SOURCE` handles the graph edge; the string property just needs to be greppable.

3. **Should `MDSplusConfig` also rename?** It's correctly MDSplus-specific config. But if we add `XMLConfig`, `HDF5Config` later, the pattern is already established. Keep as-is.

4. **`StructuralEpoch` length?** At 15 characters it's longer than `TreeModelVersion` (16). Not a real concern — ID format changes from `tcv:results:v67` to the same `tcv:results:v67`. The label appears in Cypher which is already verbose.

## Summary

This rename touches ~55 files with ~400 replacements. The core architecture (epochs, super-structure, promote-to-signal pipeline) is unchanged. The benefit is permanent: every future data system (XML, HDF5, IMAS, SQL) fits naturally without forcing "tree" metaphors. The migration is straightforward Cypher. Five focused PRs over a few days of work.

The recommended names:

```
MDSplusTree     → DataSource
TreeModelVersion → StructuralEpoch
TreeNode        → DataNode
TreeNodePattern → DataNodePattern
IN_TREE         → IN_SOURCE
SOURCE_NODE     → HAS_SOURCE_NODE (formalized in schema)
WRITES_TO_TREE  → WRITES_TO
tree_name       → source_name
node_path       → source_path
```
