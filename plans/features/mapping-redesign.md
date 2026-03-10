# Mapping Architecture Redesign: Symmetric Levels with Relationship-Based Mappings

Status: **Draft**
Supersedes: `ids-mapping-architecture.md` (v1 node-based architecture)

## Summary

Redesign the IDS mapping architecture around three symmetric levels using the existing
IMAS hierarchy. Field-level mappings become `HAS_IMAS_MAPPING` relationships instead of
`IMASMapping` nodes. Unified signal grouping via `SignalGroup`. Assembly orchestration
via the renamed `IMASMapping` node (was `IDSRecipe`). No backward compatibility.

## Motivation

The v1 architecture (implemented in Phases 1–7) revealed several design tensions:

1. **IMASMapping as a node is overkill.** The multi-hop fan-out that justified a node
   (AgentSession, MappingEvidence, five-way relationships) collapsed when AgentSession
   was removed and the agent team workflow was dropped. A mapping IS a 1:1 link between
   a source and an IMAS target — a relationship, not a node.

2. **Two incompatible dedup systems.** `DataNodePattern` (MDSplus, `FOLLOWS_PATTERN`)
   and FacilitySignal patterns (`pattern_representative_id`, `pattern_template`) solve
   the same problem with different mechanics. Neither connects to mappings.

3. **SOURCE_PATH was never created.** Declared `required` in the schema, but
   `create_mappings()` never creates it. Queried in 4 places — all return empty results.

4. **Naming inconsistencies.** `DataNode` is a signal source, not an abstract "data node".
   `IMASPath` is an IMAS node in the hierarchy, not just a "path". `IDSRecipe` is the
   entity that orchestrates IMAS mappings, yet `IMASMapping` was taken by field-level nodes.

5. **No unified grouping for template mappings.** When 830 wall nodes all need the same
   mapping to `wall/.../limiter.unit[*].outline.r`, there's no way to express "this mapping
   applies to all members of this group" in the graph.

## Architecture

Three symmetric levels, using the existing IMAS hierarchy on the target side:

| Level | Source Side | Target Side (already exists) |
|-------|-----------|-------------|
| Individual | `SignalNode` (was DataNode) | `IMASNode` leaf (was IMASPath, e.g., FLT_0D) |
| Group | `SignalGroup` (new, unifies dedup) | `IMASNode` STRUCT_ARRAY (e.g., `pf_active/coil`, maxoccur=32) |
| Collection | `IMASMapping` (was IDSRecipe) | `IDS` (e.g., `pf_active`) |

The IMAS side already has all three levels. `HAS_PARENT` connects leaves to their
struct-array roots. `IN_IDS` connects roots to `IDS`. No new IMAS node types needed.

### Graph Structure

```
IMASMapping ──[:HAS_SIGNAL_GROUP]──▶ SignalGroup ──[:HAS_IMAS_MAPPING {field props}]──▶ IMASNode (any level)
    │                                     ▲
    │                                [:MEMBER_OF]
    │                                     │
    │                              SignalNode / FacilitySignal
    │
    └──[:HAS_IMAS_NODE {assembly props}]──▶ IMASNode (STRUCT_ARRAY root)
                                                │
                                           [:HAS_PARENT*]
                                                │
                                           IMASNode (leaf) ← same node as HAS_IMAS_MAPPING target
```

Full traversal from assembly to data:

```cypher
MATCH (m:IMASMapping {id: $mapping_id})
      -[:HAS_IMAS_NODE {structure: $structure}]->(root:IMASNode)
MATCH (root)<-[:HAS_PARENT*]-(leaf:IMASNode)
      <-[map:HAS_IMAS_MAPPING]-(sg:SignalGroup)
      <-[:MEMBER_OF]-(n:SignalNode)
WHERE (m)-[:HAS_SIGNAL_GROUP]->(sg)
RETURN leaf.id, map.source_property, map.transform_code, n.path
```

### Three Distinct Relationship Types

Each relationship has one purpose and one property set. No overloading.

#### HAS_IMAS_MAPPING (SignalGroup/SignalNode → IMASNode)

Field-level mapping: "this signal/group maps to this IMAS field with this transform."

| Property | Type | Purpose |
|----------|------|---------|
| `source_property` | string | Field on SignalNode to extract (e.g., "r", "z") |
| `transform_code` | string | Python expression evaluated with `value` as input |
| `units_in` | string | Source units |
| `units_out` | string | Target units |
| `driver` | string | Data source type (e.g., "device_xml") |
| `status` | string | "validated" or "proposed" |
| `confidence` | float | 0.0–1.0 |
| `cocos_source` | integer | Source COCOS convention (nullable) |
| `cocos_target` | integer | Target COCOS convention (nullable) |

Examples:

| Source | Target | transform_code | units |
|--------|--------|----------------|-------|
| SignalGroup "jet:PF:r" | `pf_active/coil/.../rectangle/r` | `value` | m → m |
| SignalGroup "jet:PF:dr" | `pf_active/coil/.../rectangle/width` | `value` | m → m |
| SignalGroup "jet:MP:angle" | `magnetics/.../poloidal_angle` | `math.radians(value)` | deg → rad |
| SignalNode (one-off sensor) | `magnetics/.../ip/data` | `-value` | A → A |

#### HAS_IMAS_NODE (IMASMapping → IMASNode STRUCT_ARRAY)

Assembly-level config: "this assembly targets this IMAS section with this structural pattern."
Only 1–2 per IDS (one per struct-array root).

| Property | Type | Purpose |
|----------|------|---------|
| `structure` | string | "array_per_node" or "nested_array" |
| `init_arrays` | string | JSON: `{"position": 1}` |
| `elements_config` | string | JSON: `{"geometry_type": 2}` |
| `enrichment_config` | string | JSON: secondary data source config |
| `nested_path` | string | For nested_array: "limiter.unit" |
| `parent_size` | integer | For nested_array: 1 |

#### HAS_SIGNAL_GROUP (IMASMapping → SignalGroup)

Link from assembly orchestrator to its source groups. No properties — pure composition.

### Four Mapping Combinations

All via `HAS_IMAS_MAPPING` — source/target types provide context:

| Source | Target | Use Case | Example |
|--------|--------|----------|---------|
| SignalNode → leaf IMASNode | Individual → field | One-off unique sensor | Ip sensor → `magnetics/.../ip/data` |
| SignalNode → non-leaf IMASNode | Individual → container | Single signal to struct-array | Unusual signal → struct-array root |
| SignalGroup → leaf IMASNode | Group → field (common) | Templated mapping | All PF coil R → `coil/.../rectangle/r` |
| SignalGroup → non-leaf IMASNode | Group → container | Group to struct-array | Group maps to struct-array root |

### Why Not IMASGroup?

The v3 plan proposed an `IMASGroup` node to mirror `SignalGroup` on the IMAS side.
This is unnecessary — the IMAS graph already provides grouping:

```
IDS ("pf_active")
  ▲ IN_IDS
  │
IMASNode ("pf_active/coil", data_type=STRUCT_ARRAY, maxoccur=32)     ← natural group
  ▲ HAS_PARENT
  │
IMASNode ("pf_active/coil/element", data_type=STRUCT_ARRAY, maxoccur=328)
  ▲ HAS_PARENT
  │
IMASNode ("pf_active/coil/element/geometry/rectangle/r", data_type=FLT_0D)  ← leaf
```

2,438 STRUCT_ARRAY nodes exist in the graph. Traversal from any leaf to its struct-array
root is `[:HAS_PARENT*]` filtering on `data_type = 'STRUCT_ARRAY'`. The assembly config
lives on `HAS_IMAS_NODE` relationships from `IMASMapping` to these existing nodes.

### Why Relationship-Based Field Mappings?

The v1 architecture used `IMASMapping` as a node with 15+ properties and relationships to
five types (DataNode, IMASPath, Facility, AgentSession, MappingEvidence). After removing
AgentSession (defunct) and agent teams (dropped), the remaining concerns resolve cleanly:

| Concern | Resolution |
|---------|-----------|
| Multi-target fan-out | Gone. A mapping connects exactly two endpoints. |
| Lifecycle status | Property on relationship: `m.status = 'validated'` |
| Evidence chain | `MappingEvidence` attaches to `SignalGroup` via `HAS_EVIDENCE` |
| Recipe composition | `IMASMapping → HAS_SIGNAL_GROUP → SignalGroup → HAS_IMAS_MAPPING → IMASNode` |
| Vector search | Not needed on mappings. Search signals or IMAS nodes instead. |

The codebase already uses rich relationship properties (`HAS_COORDINATE.dimension`,
`HAS_ERROR_FIELD.error_type`). Neo4j relationship properties are fully indexed and
queryable.

## Renames

No backward compatibility. All renames apply across schema, codebase, graph, tests, docs.

### Node Renames

| Current | New | Rationale |
|---------|-----|-----------|
| `DataNode` | `SignalNode` | It's a signal source, not generic "data" |
| `IMASPath` | `IMASNode` | It's a node in the IMAS hierarchy |
| `IDSRecipe` | `IMASMapping` | It orchestrates IMAS mappings (the old IMASMapping field-mapping node is gone) |
| `IDSRecipeStatus` | `IMASMappingStatus` | Follows node rename |
| `DataNodePattern` | *(deleted)* | Replaced by `SignalGroup` |

### Relationship Renames

| Current | New | Rationale |
|---------|-----|-----------|
| `FOLLOWS_PATTERN` | `MEMBER_OF` | SignalNode/FacilitySignal is a member of a SignalGroup |
| `TARGET_PATH` | *(deleted)* | Old IMASMapping node gone |
| `SOURCE_PATH` | *(deleted)* | Old IMASMapping node gone (never created anyway) |
| `INCLUDES_MAPPING` | *(deleted)* | IDSRecipe→IMASMapping composition gone |
| `USES_SOURCE` | *(deleted)* | IDSRecipe→DataSource gone |

### New Relationships

| Relationship | From → To | Purpose |
|---|---|---|
| `HAS_IMAS_MAPPING` | SignalGroup/SignalNode → IMASNode | Field-level mapping with transform |
| `HAS_IMAS_NODE` | IMASMapping → IMASNode (STRUCT_ARRAY) | Assembly section target with config |
| `HAS_SIGNAL_GROUP` | IMASMapping → SignalGroup | Assembly sources |
| `MEMBER_OF` | SignalNode/FacilitySignal → SignalGroup | Group membership |
| `HAS_EVIDENCE` | SignalGroup → MappingEvidence | Evidence for group's mapping |

### Property Removals

| Node | Property | Replaced By |
|------|----------|-------------|
| `FacilitySignal` | `pattern_representative_id` | `MEMBER_OF` → `SignalGroup` |
| `FacilitySignal` | `pattern_template` | `SignalGroup.group_key` |

### Deletions

| Entity | Type | Reason |
|--------|------|--------|
| `AgentSession` | Class + enum | Defunct — zero code usage |
| `IMASMapping` (old) | Class + enum | Field-level mapping → `HAS_IMAS_MAPPING` relationship |
| `MappingEvidence.created_by` | Slot | Referenced `AgentSession` |
| `DataNodePattern` | Class | Replaced by `SignalGroup` |
| `plans/features/agent-teams.md` | File | Agent teams dropped |
| `docs/agent-teams.md` | File | Agent teams dropped |

## Phase 1: Schema Cleanup + All Renames

### 1a. Remove Dead Weight from facility.yaml

- Delete `AgentSession` class + `AgentSessionStatus` enum
- Delete old `IMASMapping` class + `MappingStatus` enum (field-mapping node → relationship)
- On `MappingEvidence`: remove `created_by` slot (referenced AgentSession)
- Delete `plans/features/agent-teams.md`, `docs/agent-teams.md`
- Remove agent team references from `.claude/skills/`

### 1b. All Node Renames

| Schema | Current Label | New Label |
|--------|--------------|-----------|
| `facility.yaml` | `DataNode` | `SignalNode` |
| `facility.yaml` | `DataNodePattern` | *(delete — replaced by SignalGroup in Phase 2)* |
| `facility.yaml` | `IDSRecipe` | `IMASMapping` |
| `facility.yaml` | `IDSRecipeStatus` | `IMASMappingStatus` |
| `imas_dd.yaml` | `IMASPath` | `IMASNode` |

### 1c. Relationship Renames + Removals

| Current | Action |
|---------|--------|
| `FOLLOWS_PATTERN` | Rename to `MEMBER_OF` |
| `TARGET_PATH` | Delete (old IMASMapping node gone) |
| `SOURCE_PATH` | Delete (old IMASMapping node gone; never created) |
| `INCLUDES_MAPPING` | Delete (IDSRecipe→IMASMapping gone) |
| `USES_SOURCE` | Delete (IDSRecipe→DataSource gone) |

### 1d. Property Removals

- `FacilitySignal.pattern_representative_id` → replaced by `MEMBER_OF`
- `FacilitySignal.pattern_template` → replaced by `SignalGroup.group_key`

### 1e. Codebase Impact

~50 files, ~500+ occurrences. Key files:

| File | Primary Changes |
|------|----------------|
| `imas_codex/schemas/facility.yaml` | All class renames, deletions, new SignalGroup |
| `imas_codex/schemas/imas_dd.yaml` | IMASPath → IMASNode |
| `imas_codex/discovery/mdsplus/graph_ops.py` | DataNode→SignalNode, DataNodePattern→SignalGroup |
| `imas_codex/graph/build_dd.py` | IMASPath→IMASNode |
| `imas_codex/ids/graph_ops.py` | IDSRecipe→IMASMapping, mapping functions |
| `imas_codex/ids/assembler.py` | Assembly refactor |
| `imas_codex/agentic/search_tools.py` | Cross-domain query updates |
| `imas_codex/ingestion/graph.py` | SOURCE_PATH removal |
| `imas_codex/graph/domain_queries.py` | Signal→IMAS traversal |
| All `tests/ids/`, `tests/discovery/` | Label and relationship updates |
| `AGENTS.md`, `docs/architecture/ids-mapping.md` | Documentation |

### 1f. Graph Migration Script

`scripts/migrate_v5.py`:

```cypher
-- Label migrations
MATCH (n:DataNode) SET n:SignalNode REMOVE n:DataNode;
MATCH (n:IMASPath) SET n:IMASNode REMOVE n:IMASPath;
MATCH (n:IDSRecipe) SET n:IMASMapping REMOVE n:IDSRecipe;
MATCH (n:DataNodePattern) SET n:SignalGroup REMOVE n:DataNodePattern;

-- Relationship migrations
CALL { MATCH (a)-[r:FOLLOWS_PATTERN]->(b) CREATE (a)-[:MEMBER_OF]->(b) DELETE r } IN TRANSACTIONS;

-- Remove defunct nodes (0 exist today, but clean up schema artifacts)
MATCH (n:IMASMapping) WHERE n.status IN ['proposed', 'validated'] DETACH DELETE n;
MATCH (n:AgentSession) DETACH DELETE n;

-- Drop old vector indexes and recreate for new labels
DROP INDEX data_node_desc_embedding IF EXISTS;
DROP INDEX imas_path_desc_embedding IF EXISTS;
CREATE VECTOR INDEX signal_node_desc_embedding IF NOT EXISTS
  FOR (n:SignalNode) ON n.embedding
  OPTIONS { indexConfig: { `vector.dimensions`: 1024, `vector.similarity_function`: 'cosine' } };
CREATE VECTOR INDEX imas_node_desc_embedding IF NOT EXISTS
  FOR (n:IMASNode) ON n.embedding
  OPTIONS { indexConfig: { `vector.dimensions`: 1024, `vector.similarity_function`: 'cosine' } };
```

## Phase 2: SignalGroup — Unified Per-Property Grouping

### 2a. SignalGroup Schema

New class in `facility.yaml`, replacing `DataNodePattern`:

```yaml
SignalGroup:
  description: >-
    Groups signal instances that will map identically to the same IMAS field.
    Per-property: "jet:PF:r" groups all PF coil R values. One group = one
    HAS_IMAS_MAPPING relationship = one mapping definition.
    Unifies DataNodePattern and FacilitySignal pattern mechanics into a
    single grouping concept with MEMBER_OF relationships.
  class_uri: facility:SignalGroup
  attributes:
    id:
      identifier: true
      description: >-
        Composite key: "{facility}:{group_key}".
        Examples: "jet:PF:r", "tcv:magnetics:BPOL_NNN"
      required: true
    facility_id:
      description: Parent facility ID
      required: true
      range: Facility
      annotations:
        relationship_type: AT_FACILITY
    group_key:
      description: >-
        What members share — defines the mapping type.
        Examples: "PF:r", "MP:angle", "magnetics:BPOL_NNN"
      required: true
    member_count:
      description: Number of SignalNode/FacilitySignal members in this group
      range: integer
    representative_id:
      description: ID of one representative member (enriched first, propagated to others)
    description:
      description: Physics description of the group (enriched once, propagated)
    keywords:
      description: Physics keywords
      multivalued: true
    physics_domain:
      description: Physics domain classification
      range: PhysicsDomain
    status:
      description: Lifecycle status
      range: SignalGroupStatus
      required: true
    embedding:
      description: Vector embedding of description for semantic search
      multivalued: true
      range: float
    embedded_at:
      description: When embedding was last computed
      range: datetime
```

```yaml
SignalGroupStatus:
  permissible_values:
    discovered:
      description: Group created from pattern detection
    enriched:
      description: Description and keywords populated via LLM
```

### 2b. Relationships

| Relationship | From | To | Purpose |
|---|---|---|---|
| `MEMBER_OF` | SignalNode / FacilitySignal | SignalGroup | Group membership (replaces FOLLOWS_PATTERN) |
| `HAS_EVIDENCE` | SignalGroup | MappingEvidence | Evidence for the group's mapping claim |
| `HAS_IMAS_MAPPING` | SignalGroup | IMASNode | The mapping IS this relationship |
| `HAS_SIGNAL_GROUP` | IMASMapping | SignalGroup | Assembly uses this group |

### 2c. Per-Property Semantics

Groups signal instances that will map identically:

| Group ID | Members | Maps To |
|----------|---------|---------|
| `jet:PF:r` | All PF coil R-position signals | `pf_active/coil/.../rectangle/r` |
| `jet:PF:z` | All PF coil Z-position signals | `pf_active/coil/.../rectangle/z` |
| `jet:PF:dr` | All PF coil width signals | `pf_active/coil/.../rectangle/width` |
| `jet:MP:angle` | All magnetic probe angle signals | `magnetics/.../poloidal_angle` |
| `tcv:magnetics:BPOL_NNN` | All BPOL probe signals | `magnetics/bpol_probe/.../field/data` |

One SignalGroup = one `HAS_IMAS_MAPPING` = one mapping definition.

### 2d. Discovery CLI Refactor

#### MDSplus (`imas_codex/discovery/mdsplus/graph_ops.py`)

| Current | New |
|---------|-----|
| `detect_and_create_patterns()` | `detect_and_create_signal_groups()` |
| `detect_and_create_member_patterns()` | Same refactor |
| `claim_patterns_for_enrichment()` | `claim_signal_groups()` |
| `mark_patterns_enriched()` | `mark_signal_groups_enriched()` |

Creates SignalGroup nodes + `MEMBER_OF` edges. Detection logic unchanged —
finds (grandparent STRUCTURE → parent STRUCTURE → leaf) groups by (gp_path, leaf_name).

#### Signals (`imas_codex/discovery/signals/parallel.py`)

| Current | New |
|---------|-----|
| `detect_signal_patterns()` | `detect_signal_groups()` |
| `_accessor_to_pattern()` | Unchanged (regex \d{2,} → NNN) |
| `propagate_pattern_enrichment()` | `propagate_signal_group_enrichment()` |

Claim filter changes from `WHERE pattern_representative_id IS NULL` to
`WHERE NOT EXISTS { (s)-[:MEMBER_OF]->(:SignalGroup) }`.

### 2e. Removals

- `DataNodePattern` class (replaced by SignalGroup)
- `follows_pattern` slot on SignalNode
- `pattern_representative_id` property on FacilitySignal
- `pattern_template` property on FacilitySignal

## Phase 3: HAS_IMAS_MAPPING + IMASMapping + Assembly Refactor

### 3a. IMASMapping Node (was IDSRecipe)

The orchestration node that coordinates all mappings for a facility+IDS combination.
Renamed from `IDSRecipe` because the field-level mapping functionality moved to
`HAS_IMAS_MAPPING` relationships — this node IS the IMAS mapping coordinator.

```yaml
IMASMapping:
  description: >-
    Orchestrates the assembly of an IDS for a facility. Links to SignalGroups
    via HAS_SIGNAL_GROUP and to IMAS struct-array roots via HAS_IMAS_NODE
    with assembly configuration properties.
  class_uri: facility:IMASMapping
  attributes:
    id:
      identifier: true
      description: >-
        Composite key: "{facility}:{ids_name}".
        Examples: "jet:pf_active", "tcv:magnetics"
      required: true
    facility_id:
      description: Parent facility ID
      required: true
      range: Facility
      annotations:
        relationship_type: AT_FACILITY
    ids_name:
      description: IMAS IDS name (e.g., "pf_active", "magnetics")
      required: true
    dd_version:
      description: Data dictionary version (e.g., "4.1.1")
    status:
      description: Lifecycle status
      range: IMASMappingStatus
      required: true
    provider:
      description: Who created this mapping (e.g., "imas-codex")
    static_config:
      description: >-
        JSON: static IDS properties (ids_properties values).
        E.g., {"homogeneous_time": 1}
      range: string
    description:
      description: Human-readable description of this IDS mapping
    has_signal_group:
      description: SignalGroups used by this mapping
      range: SignalGroup
      multivalued: true
      annotations:
        relationship_type: HAS_SIGNAL_GROUP
    has_imas_node:
      description: IMAS struct-array roots targeted by this mapping
      range: IMASNode
      multivalued: true
      annotations:
        relationship_type: HAS_IMAS_NODE
```

```yaml
IMASMappingStatus:
  permissible_values:
    draft:
      description: Under construction
    active:
      description: Ready for assembly
    deprecated:
      description: Superseded by newer mapping
```

### 3b. Seed Function Rewrite

`seed_ids_mappings()` in `imas_codex/ids/graph_ops.py`:

1. Create SignalGroup per (facility, system, source_property)
2. Create `HAS_IMAS_MAPPING` from SignalGroup to target IMASNode with field properties
3. Create IMASMapping node with `HAS_SIGNAL_GROUP` → SignalGroup
4. Create `HAS_IMAS_NODE` from IMASMapping to STRUCT_ARRAY root with assembly config

```cypher
-- Step 1: Create SignalGroup
MERGE (sg:SignalGroup {id: $group_id})
SET sg.facility_id = $facility,
    sg.group_key = $group_key,
    sg.status = 'discovered'
WITH sg
MATCH (f:Facility {id: $facility})
MERGE (sg)-[:AT_FACILITY]->(f)

-- Step 2: Field-level mapping via HAS_IMAS_MAPPING
MATCH (sg:SignalGroup {id: $group_id})
MATCH (ip:IMASNode {id: $target_path})
MERGE (sg)-[m:HAS_IMAS_MAPPING]->(ip)
SET m.source_property = $source_property,
    m.transform_code = $transform_code,
    m.units_in = $units_in,
    m.units_out = $units_out,
    m.driver = $driver,
    m.status = 'validated',
    m.confidence = 1.0

-- Step 3: Create IMASMapping node
MERGE (r:IMASMapping {id: $mapping_id})
SET r.facility_id = $facility,
    r.ids_name = $ids_name,
    r.dd_version = $dd_version,
    r.status = 'active',
    r.provider = 'imas-codex'
WITH r
MATCH (f:Facility {id: $facility})
MERGE (r)-[:AT_FACILITY]->(f)

-- Step 4: Assembly-level HAS_IMAS_NODE to STRUCT_ARRAY root
MATCH (r:IMASMapping {id: $mapping_id})
MATCH (root:IMASNode {id: $struct_array_root})
MERGE (r)-[t:HAS_IMAS_NODE]->(root)
SET t.structure = $structure,
    t.init_arrays = $init_arrays_json,
    t.elements_config = $elements_json

-- Step 5: Link IMASMapping → SignalGroup
MATCH (r:IMASMapping {id: $mapping_id})
MATCH (sg:SignalGroup {id: $group_id})
MERGE (r)-[:HAS_SIGNAL_GROUP]->(sg)
```

### 3c. Assembly Refactor

Graph-led traversal replaces JSON config + property-based selection:

```python
class IDSAssembler:
    def _assemble_from_graph(self, facility: str, ids_name: str, shot: int):
        # 1. Load IMASMapping node
        mapping = self._load_imas_mapping(facility, ids_name)

        # 2. Get section roots with assembly config
        sections = self._query("""
            MATCH (m:IMASMapping {id: $id})
                  -[t:HAS_IMAS_NODE]->(root:IMASNode)
            RETURN root.id AS root_path,
                   t.structure AS structure,
                   t.init_arrays AS init_arrays,
                   t.elements_config AS elements_config,
                   t.nested_path AS nested_path,
                   t.parent_size AS parent_size
        """, id=mapping["id"])

        # 3. For each section, get field mappings via SignalGroups
        for section in sections:
            field_mappings = self._query("""
                MATCH (m:IMASMapping {id: $mapping_id})
                      -[:HAS_SIGNAL_GROUP]->(sg:SignalGroup)
                      -[map:HAS_IMAS_MAPPING]->(leaf:IMASNode)
                      -[:HAS_PARENT*]->(root:IMASNode {id: $root_id})
                MATCH (sg)<-[:MEMBER_OF]-(n:SignalNode)
                WHERE n.epoch_id = $epoch
                RETURN sg.id AS group_id,
                       map.source_property AS source_property,
                       map.transform_code AS transform_code,
                       map.units_in AS units_in,
                       map.units_out AS units_out,
                       leaf.id AS target_path,
                       collect(n.path) AS signal_node_paths
            """, mapping_id=mapping["id"],
                 root_id=section["root_path"],
                 epoch=epoch_id)

            # 4. Build IDS structure from field mappings
            self._build_section(ids, section, field_mappings)
```

### 3d. Downstream Query Updates

| File | Current | New |
|------|---------|-----|
| `search_tools.py` (line 242) | `OPTIONAL MATCH ... -[:SOURCE_PATH]->` | `OPTIONAL MATCH (sg:SignalGroup)-[:HAS_IMAS_MAPPING]->(ip:IMASNode)` (via SignalNode MEMBER_OF) |
| `search_tools.py` (line 1523) | `OPTIONAL MATCH ... -[:SOURCE_PATH]->` | Same pattern |
| `ingestion/graph.py` (line 211) | `MATCH ... -[:SOURCE_PATH]->` | Remove or update |
| `domain_queries.py` (line 622) | `OPTIONAL MATCH ... -[:SOURCE_PATH]->` | `OPTIONAL MATCH (n)<-[:MEMBER_OF]-(sg)-[:HAS_IMAS_MAPPING]->(ip:IMASNode)` |

## Verification

1. `uv run build-models --force` — schema compiles with new classes
2. `uv run pytest` — all tests pass after codebase updates
3. `ids seed jet pf_active` — creates SignalGroups + HAS_IMAS_MAPPING + IMASMapping + HAS_IMAS_NODE
4. Assembly produces identical IDS output as v1 architecture
5. Migration script runs on existing graph without errors
6. `uv run ruff check .` — no lint errors
7. `REFERENCES_IMAS` and `MENTIONS_IMAS` queries still work (just label rename IMASPath→IMASNode)
8. All 4 mapping combinations work (single/group → leaf/non-leaf)

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | No IMASGroup | IMAS hierarchy (IMASNode STRUCT_ARRAY + IDS + HAS_PARENT) provides grouping natively. 2,438 STRUCT_ARRAY nodes already exist. |
| 2 | Field mappings are HAS_IMAS_MAPPING relationships | A mapping IS a 1:1 link. AgentSession and agent teams removed — no need for multi-hop fan-out. Lifecycle status and transform properties fit on relationship properties. |
| 3 | Dedicated HAS_IMAS_MAPPING vs overloaded HAS_TARGET | Field mapping properties (transform_code, units, COCOS) are semantically different from assembly config (structure, init_arrays). Separate relationship types prevent ambiguity. |
| 4 | IDSRecipe → IMASMapping rename | The field-level IMASMapping node is gone (→ relationship). The orchestration node is what coordinates IMAS mappings — `IMASMapping` is the natural name. |
| 5 | HAS_IMAS_NODE, HAS_SIGNAL_GROUP (not HAS_TARGET, HAS_SOURCE) | Generic names would collide with future uses. Qualified names are self-documenting and match the schema's `HAS_X` containment grammar. |
| 6 | Per-property SignalGroups | Signals that map identically should be grouped by what they map (property), not just by structure. One group = one mapping. |
| 7 | MEMBER_OF (not FOLLOWS_PATTERN) | More general. Works for both MDSplus DataNodePattern successors and FacilitySignal pattern members. |
| 8 | Remove AgentSession | Zero code usage. Schema-only artifact of agent team workflow that was never built. |
| 9 | Keep MappingEvidence on SignalGroup | Evidence supports the group's mapping claim, not individual transforms. Better semantic fit. |
| 10 | No backward compatibility | Project philosophy: "breaking changes are expected — remove deprecated code decisively." 0 IMASMapping nodes exist in the live graph. |
| 11 | No agent teams | Templated LLM prompts for mapping proposals. No multi-agent coordination layer. |

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `imas_codex/schemas/facility.yaml` | Modify | SignalNode, SignalGroup, IMASMapping renames; delete old IMASMapping, AgentSession, DataNodePattern |
| `imas_codex/schemas/imas_dd.yaml` | Modify | IMASPath → IMASNode |
| `imas_codex/ids/graph_ops.py` | Rewrite | HAS_IMAS_MAPPING creation, SignalGroup creation, seeding |
| `imas_codex/ids/assembler.py` | Rewrite | Graph-led assembly via HAS_IMAS_MAPPING traversal |
| `imas_codex/ids/transforms.py` | No change | Transform execution engine unchanged |
| `imas_codex/cli/ids.py` | Modify | Terminology updates |
| `imas_codex/graph/build_dd.py` | Modify | IMASPath → IMASNode label |
| `imas_codex/discovery/mdsplus/graph_ops.py` | Modify | DataNodePattern → SignalGroup, FOLLOWS_PATTERN → MEMBER_OF |
| `imas_codex/discovery/signals/parallel.py` | Modify | Pattern properties → MEMBER_OF relationships |
| `imas_codex/agentic/search_tools.py` | Modify | SOURCE_PATH queries → HAS_IMAS_MAPPING traversal |
| `imas_codex/ingestion/graph.py` | Modify | SOURCE_PATH removal |
| `imas_codex/graph/domain_queries.py` | Modify | Signal→IMAS traversal |
| `scripts/migrate_v5.py` | Create | Graph migration script |
| `tests/ids/` | Modify | All test updates |
| `tests/discovery/` | Modify | DataNode→SignalNode, pattern→group |
| `docs/architecture/ids-mapping.md` | Rewrite | Updated architecture documentation |
| `AGENTS.md` | Modify | Relationship table, schema reference |
