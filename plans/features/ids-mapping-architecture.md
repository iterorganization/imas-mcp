# IDS Mapping Architecture: From Naive Recipes to Graph-Driven Assembly

Status: **Design Proposal**
Supersedes: `ids-assembly.md` (Phase 1 proof-of-concept)

## What We Learned from pf_active

The naive recipe approach (embedded Cypher in YAML, direct field assignment) proved
that the data is there and assembly works. But it has fundamental flaws:

1. **Mappings are invisible to the graph** — buried in YAML files, not queryable
2. **No transformation metadata** — units, COCOS, sign conventions are ignored
3. **No evidence or provenance** — why does `r → geometry.rectangle.r`? Because a human wrote it
4. **Structural logic is ad-hoc** — the concept of "array properties become element sub-arrays"
   is coded into the assembler, not declared
5. **No validation** — we write fields but never check types, units, or dimensions
6. **No composability** — can't answer "which IMAS paths are covered?" from the graph alone

The IMASMapping schema was designed for exactly this purpose — and we bypassed it entirely.

## The Two-Level Architecture

The mapping problem has two distinct layers. Conflating them was the root mistake.

### Level 1: IMASMapping — "What maps where, and how?"

Each `IMASMapping` node is an atomic, validated, evidenced assertion:

> "DataNode property X at facility F maps to IMASPath Y,
>  with unit conversion A→B, COCOS transform C→D, and scale factor S."

This is the **knowledge layer**. Agents discover these mappings from code, wiki,
documentation, and data validation. Each mapping has:
- **source**: DataNode (or property within it) 
- **target**: IMASPath (leaf node in the DD)
- **transformation**: units_in/out, cocos_source/target, scale, expression
- **evidence**: why we believe this mapping is correct
- **lifecycle**: proposed → endorsed → validated (or contested → rejected)

IMASMapping is a **graph-first** concept: queryable, traversable, auditable.
"Show me all unvalidated mappings for JET magnetics" is a Cypher query, 
not a grep through YAML files.

### Level 2: IDSAssemblyRule — "How do sources compose into IDS structure?"

The structural problem — how DataNodes group into array-of-structures entries,
how array properties fan out into sub-arrays, how multiple sources merge —
is a separate concern from field-level mapping.

An assembly rule declares:

> "For pf_active.coil: select DataNodes with system=PF from device_xml per epoch.
>  Each DataNode becomes one coil entry. Array-valued properties (r[], z[], dr[], dz[])
>  create one element per array position. Enrichment from jec2020 matches by coil index."

This is the **structural layer**. It lives in the graph as recipe metadata (IDSRecipe)
with relationships to IMASMappings that define the individual field transforms.

### Why Two Levels?

| Concern | IMASMapping | IDSAssemblyRule |
|---------|-------------|-----------------|
| Granularity | One property → one IDS leaf | Set of nodes → struct array |
| Knowledge source | Code analysis, wiki, experts | Data structure analysis |
| Changes when... | Physics understanding improves | Data organization changes |
| Lifecycle | proposed → validated | Per-recipe definition |
| Reusable across | Different IDSs using same data | N/A, IDS-specific |

A single DataNode field like `r` might map to different IMAS paths depending
on context (PF coil geometry vs. probe position). The IMASMapping captures 
the full context. The assembly rule captures how to apply it structurally.

## Current Graph State: What We Have vs. What We Need

### What exists (good foundation)

| Entity | Count | Status |
|--------|------:|--------|
| DataNode (JET) | 6,623 | Geometry data present, epoch-linked |
| FacilitySignal (device_xml) | 1,058 | Created with units, accessor, data_source_node property |
| StructuralEpoch | 21 | Full shot range coverage |
| IMASPath (pf_active) | 258 | Data-bearing paths with types, units, COCOS labels |
| IMASPath (magnetics) | 481 | Full target structure |
| IMASPath (pf_passive) | 220 | Full target structure |
| IMASPath (wall) | 1,458 | Full target structure |

### What's missing (data quality gaps)

| Gap | Impact | Fix |
|-----|--------|-----|
| 0 HAS_DATA_SOURCE_NODE relationships | Can't traverse FacilitySignal → DataNode | Create relationships from `data_source_node` property |
| FacilitySignal.system = None (all 1058) | Can't filter by system | Set from SECTION_METADATA during creation |
| DataNode.unit = None (geometry nodes) | Can't validate unit compatibility | Set from SECTION_METADATA |
| DataNode.sign_convention = None | Can't determine COCOS transforms | Set where applicable (PF coils, probes) |
| 0 IMASMapping nodes | No field-level mapping knowledge | Create from SECTION_METADATA + discovery |
| FacilitySignal → IMASPath (no link) | Can't answer "what IMAS paths does this signal map to?" | Via IMASMapping: Signal → DataNode ← IMASMapping → IMASPath |

### The relationship chain (designed but not instantiated)

```
FacilitySignal ──HAS_DATA_SOURCE_NODE──▶ DataNode
                                             ▲
                                        SOURCE_PATH
                                             │
                                        IMASMapping
                                             │
                                        TARGET_PATH
                                             ▼
                                         IMASPath
```

This chain already exists in the schema. None of the relationships are created.

## Concrete Plan

### Phase 0: Fix the Data Foundation

Before creating mappings, fix the signal and data node metadata.

**0a. Create HAS_DATA_SOURCE_NODE relationships**

Every device_xml FacilitySignal has a `data_source_node` property string
pointing to a DataNode path. The relationship was never MERGE'd.

```cypher
MATCH (fs:FacilitySignal)
WHERE fs.data_source_node IS NOT NULL
MATCH (dn:DataNode {path: fs.data_source_node})
MERGE (fs)-[:HAS_DATA_SOURCE_NODE]->(dn)
```

**0b. Set FacilitySignal.system from SECTION_METADATA**

The device_xml scanner knows the system per section (PF, CI, PS, MP, FL)
but doesn't set it on created FacilitySignals. Fix: update scanner + backfill.

**0c. Set DataNode.unit from SECTION_METADATA**

PF coil DataNodes have `r`, `z`, `dr`, `dz` in meters but `unit` is NULL.
Fix: update scanner to SET unit per field, or infer from SECTION_METADATA.

Note: DataNode is per-coil (one node with r[], z[] arrays), while FacilitySignal
is per-field (one signal for coil_3_r, another for coil_3_z). The unit belongs
on the FacilitySignal (which already has it from the scanner) AND on the IMASMapping
(as units_in). DataNode.unit is ambiguous for multi-property nodes — might not need fixing.

### Phase 1: Create IMASMapping Nodes

Create field-level mappings from what we already know. The device_xml scanner's
`SECTION_METADATA` encodes the domain knowledge:

```python
SECTION_METADATA = {
    "pfcoils": {
        "imas_ids": "pf_active.coil",
        "system": "PF",
        "fields": {
            "r": {"unit": "m", "desc": "Radial position"},
            "z": {"unit": "m", "desc": "Vertical position"},
            "dr": {"unit": "m", "desc": "Radial width"},
            "dz": {"unit": "m", "desc": "Vertical height"},
            "turnsperelement": {"unit": "", "desc": "Turns per element"},
        },
    },
}
```

This already tells us:
- `r` (m) → `pf_active/coil/element/geometry/rectangle/r` (m) — no conversion
- `z` (m) → `pf_active/coil/element/geometry/rectangle/z` (m) — no conversion
- `dr` (m) → `pf_active/coil/element/geometry/rectangle/width` (m) — name change only
- `dz` (m) → `pf_active/coil/element/geometry/rectangle/height` (m) — name change only
- `turnsperelement` → `pf_active/coil/element/turns_with_sign` — dimensionless

Each of these becomes an IMASMapping node:

```python
IMASMapping(
    id="jet:device_xml:PF:r→pf_active/coil/element/geometry/rectangle/r",
    facility_id="jet",
    source_path="jet:device_xml:*:pfcoils:*",  # Pattern, not one node
    target_path="pf_active/coil/element/geometry/rectangle/r",
    driver="device_xml",
    units_in="m",
    units_out="m",
    scale=1.0,
    status="validated",  # We proved this works in pf_active PoC
    confidence=1.0,
    notes="Validated via HDF5 round-trip of 22 coils, 1169 elements",
)
```

**Key design question: source_path granularity**

The IMASMapping schema has `source_path: DataNode` — one specific node. But our
mapping is a *pattern*: "for ANY PF DataNode, property `r` maps to this IDS path."

Options:
1. **One IMASMapping per DataNode per field** — 308 PF nodes × 5 fields = 1,540 mappings
   just for PF coils. Explosion. But each is precise and traversable.
2. **One IMASMapping per field pattern** — 5 mappings for all PF coils. Compact, but
   `source_path` can't point to a specific DataNode.
3. **IMASMapping references a DataNodePattern** — the existing `DataNodePattern` schema
   handles parametric patterns. Not yet connected to IMASMapping.

**Recommendation: Option 2 with a pattern DataNode.**

Create a sentinel `DataNode` per system+data_source representing the pattern:
`jet:device_xml:PF:*` with properties listing the common fields. Or better:
extend IMASMapping with `source_system`, `source_data_source`, and `source_property`
fields so it doesn't need to point to one specific DataNode.

Actually, the cleanest approach: **IMASMapping.source_path stays as-is** (pointing to a
DataNode), but we create it once per *representative* DataNode (e.g., the first epoch's
first coil), with metadata indicating it's a pattern mapping. The assembly engine
uses the mapping's `driver`, `units_in/out`, `cocos_*` for transformation, not the
specific source_path for data retrieval.

This inverts the flow: mappings define *how to transform*, assembly rules define
*what to select*.

### Phase 2: Redesign IDSRecipe as Structural Composition

Replace the YAML recipe with a graph-native IDSRecipe that composes IMASMappings.

**Current IDSRecipe schema** (from Phase 1 PoC):

```yaml
IDSRecipe:
  # Just tracks: facility, ids_name, dd_version, status, completeness
  # Relationships: USES_SOURCE → DataSource, COVERS → IMASPath
```

**Enhanced IDSRecipe**:

Add `assembly_config` — a JSON/YAML blob stored as a graph property that defines
the structural rules. This replaces the standalone YAML recipe file:

```yaml
IDSRecipe:
  attributes:
    # ... existing ...
    assembly_config:
      description: >-
        Structural assembly rules (JSON). Defines how DataNodes group into
        array-of-structures entries, how array properties fan out, and how
        multiple sources merge. Field transformations come from IMASMapping nodes.
      range: string  # JSON blob
    includes_mapping:
      range: IMASMapping
      multivalued: true
      annotations:
        relationship_type: INCLUDES_MAPPING
```

The `assembly_config` encodes only structural concerns:

```json
{
  "coil": {
    "source": {
      "system": "PF",
      "data_source": "device_xml",
      "epoch_field": "introduced_version"
    },
    "structure": "array_per_node",
    "element_source": "array_properties",
    "enrichment": [
      {
        "data_source": "jec2020_geometry",
        "system": "PF",
        "match_by": "coil_index"
      }
    ]
  },
  "circuit": {
    "source": {
      "system": "CI",
      "data_source": "device_xml",
      "epoch_field": "introduced_version"
    },
    "structure": "array_per_node"
  }
}
```

Field-level transformations come from traversing:
```
IDSRecipe -[:INCLUDES_MAPPING]-> IMASMapping -[:TARGET_PATH]-> IMASPath
```

### Phase 3: Graph-Driven Assembly Engine

Replace the current Cypher-in-YAML assembler with one that reads from the graph:

```python
class IDSAssembler:
    def assemble(self, facility: str, ids_name: str, epoch: str) -> IDSToplevel:
        # 1. Load IDSRecipe from graph
        recipe = self._load_recipe(facility, ids_name)
        assembly_config = json.loads(recipe["assembly_config"])
        
        # 2. Load IMASMappings for this recipe
        mappings = self._load_mappings(recipe["id"])
        
        # 3. For each struct array section...
        for section_name, section_config in assembly_config.items():
            # 3a. Query DataNodes per structural rules
            nodes = self._select_nodes(facility, section_config, epoch)
            
            # 3b. Group into struct-array entries
            entries = self._group_entries(nodes, section_config)
            
            # 3c. For each entry, apply field mappings with transforms
            for entry_data in entries:
                entry = struct_array[i]
                for mapping in mappings:
                    value = entry_data.get(mapping.source_property)
                    if value is not None:
                        value = self._transform(value, mapping)  # units, COCOS, scale
                        _set_nested(entry, mapping.target_path, value)
        
        return ids
    
    def _transform(self, value, mapping):
        """Apply unit conversion, COCOS transform, and scaling."""
        if mapping.units_in and mapping.units_out and mapping.units_in != mapping.units_out:
            value = convert_units(value, mapping.units_in, mapping.units_out)
        if mapping.cocos_source and mapping.cocos_target:
            value = apply_cocos_transform(value, mapping)
        if mapping.scale and mapping.scale != 1.0:
            value *= mapping.scale
        return value
```

### Phase 4: Unit and COCOS Integration

**Units**: `imas_codex/units/` has pint with DD aliases. Add `convert_value()`:

```python
def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units using pint."""
    q = unit_registry.Quantity(value, from_unit)
    return q.to(to_unit).magnitude
```

**COCOS**: `imas_codex/cocos/` has the full transformation infrastructure.
For machine description data (geometry), COCOS is mostly irrelevant — positions
and dimensions don't depend on sign conventions.

COCOS matters for:
- `pf_active/circuit/current` — has `cocos_label_transformation: ip_like`
- Any time-dependent data (not machine description)
- Sign conventions on probe orientations

The IMASPath schema already tells us which paths need COCOS transforms via
`cocos_label_transformation` and `cocos_transformation_expression`. The assembler
can check this automatically:

```python
if imas_path.cocos_label_transformation:
    # This path is COCOS-sensitive — ensure mapping has cocos_source/target
    if not mapping.cocos_source:
        logger.warning("COCOS-sensitive path %s has no source COCOS", imas_path.id)
```

### Phase 5: Expand to All Machine Description IDSs

With the graph-driven architecture, adding new IDSs becomes:

1. Create IMASMapping nodes for new field transformations
2. Create/update IDSRecipe with structural assembly rules
3. Link mappings to recipe via INCLUDES_MAPPING
4. Run `imas-codex ids export jet magnetics --epoch p68613`

No new Python code required for each IDS — the engine is generic.

**Target IDSs and their structural patterns**:

| IDS | Struct Array | Source System | Pattern |
|-----|-------------|---------------|---------|
| pf_active | coil, circuit | PF, CI | array_per_node + array_properties |
| pf_passive | loop | PS | array_per_node + array_properties |
| magnetics | bpol_probe, flux_loop, ip | MP, FL, IP | array_per_node (scalar fields) |
| wall | description_2d.limiter.unit | LIM | single_entry + contour_arrays |

Each pattern type is a known structural transform:

- **array_per_node**: Each DataNode → one struct-array entry. Field values are simple.
- **array_properties**: Node has `r[]`, `z[]` arrays → sub-array entries (elements).
- **single_entry**: One DataNode → one entry (wall limiter is just one contour).
- **contour_arrays**: `r[]` and `z[]` stay as arrays (not expanded to sub-entries).

### Phase 6: Mapping Discovery and Agent Workflows

The IMASMapping lifecycle (proposed → endorsed → validated) enables agent-driven
mapping discovery:

1. **Code analysis agent**: Scans source files for IDS read/write patterns.
   "This code writes `q_profile` to `equilibrium/profiles_1d/q`" → propose mapping.
2. **Wiki analysis agent**: Finds documentation about signal conventions.
   "JET uses COCOS 1 for plasma current" → evidence on mapping.
3. **Data validation agent**: Tests a mapping against real shot data.
   "Reading shot 99999, signal X returns Y, which matches expectations" → validate.
4. **Human review**: Contended or high-impact mappings get human sign-off.

This is the long-term value — the graph accumulates verified mapping knowledge,
not just one developer's YAML recipe.

## Migration Path

### Step 1: Keep current assembler working

Don't break the existing `imas-codex ids export` command. The current YAML-based
assembler continues to work while we build the graph-driven replacement alongside it.

### Step 2: Fix data foundation (Phase 0)

Quick wins that improve data quality regardless of architecture:
- Backfill HAS_DATA_SOURCE_NODE relationships
- Set system on FacilitySignals
- Takes minutes, no schema changes needed

### Step 3: Create IMASMapping nodes for pf_active (Phase 1)

Prove that graph-driven mappings work for the same data the YAML recipe handles.
Compare outputs — they must be identical.

### Step 4: Build graph-driven assembler (Phase 3)

Parallel implementation that reads from IMASMappings instead of YAML.
When outputs match, switch the CLI to use the new engine.

### Step 5: Expand (Phase 5)

Add magnetics, pf_passive, wall using only graph operations (IMASMapping creation +
IDSRecipe configuration). No new Python code per IDS.

## Open Design Questions

1. **IMASMapping source_path granularity**: One mapping per field-pattern, or per DataNode?
   Pattern mappings are compact but require the assembly rule to know which property
   to extract. Per-DataNode mappings are precise but explode in count.

2. **Assembly config storage**: JSON property on IDSRecipe node, or separate YAML file
   referenced by path? JSON-in-graph is self-contained but hard to review. YAML is
   reviewable but creates a file dependency.

3. **Multi-property DataNodes**: A PF coil DataNode has 5+ properties (r, z, dr, dz,
   turnsperelement). Should we split into one DataNode per property (matching the
   FacilitySignal granularity) or keep compound nodes? Current schema supports both.

4. **Enrichment as separate mappings**: When jec2020 provides coil names, is that a
   separate IMASMapping (jec2020 DataNode → pf_active/coil/name) or an enrichment
   concern at the recipe level? If it's a mapping, how do we handle the merge priority?

5. **Temporal data**: Machine description is static per epoch. But `pf_active/coil/current`
   is time-dependent. Do we need a different assembly pattern for time traces, or does
   the same architecture extend naturally?

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `imas_codex/schemas/facility.yaml` | Modify | Extend IDSRecipe, possibly extend IMASMapping |
| `imas_codex/ids/assembler.py` | Rewrite | Graph-driven assembly replacing YAML parsing |
| `imas_codex/ids/transforms.py` | Create | Unit conversion + COCOS transform helpers |
| `imas_codex/ids/graph_ops.py` | Create | IMASMapping + IDSRecipe graph queries |
| `imas_codex/discovery/signals/scanners/device_xml.py` | Modify | Fix system, create relationships |
| `tests/ids/test_assembler.py` | Update | Test graph-driven assembly |
| `tests/ids/test_transforms.py` | Create | Test unit + COCOS transforms |
