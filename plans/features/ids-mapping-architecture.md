# IDS Mapping Architecture: From Naive Recipes to Graph-Driven Assembly

Status: **Implemented** (Migration Path Steps 1–7 complete)
Supersedes: `ids-assembly.md` (Phase 1 proof-of-concept)

## What We Learned from pf_active

The naive recipe approach (embedded Cypher in YAML, direct field assignment) proved
that the data is there and assembly works. But it has fundamental flaws:

1. **Mappings are invisible to the graph** — buried in YAML files, not queryable
2. **No transformation metadata** — units, COCOS, sign conventions are ignored
3. **No evidence or provenance** — why does `r → geometry.rectangle.r`? Because a human wrote it
4. **Structural logic is ad-hoc** — the concept of "array properties become element sub-arrays"
   is coded into the assembler, not declared
5. **No executable transforms** — even where we document units and COCOS, there's no code
   to actually perform the conversion at assembly time
6. **No composability** — can't answer "which IMAS paths are covered?" from the graph alone

The IMASMapping schema was designed for exactly this purpose — and we bypassed it entirely.

## Decision: IMASMapping vs MAPS_TO_IMAS

Two competing patterns exist for connecting facility data to IMAS paths:

| | IMASMapping (schema-defined) | MAPS_TO_IMAS (ghost relationship) |
|---|---|---|
| **Defined in schema?** | Yes — full class in `facility.yaml` with 20+ fields | No — not in any LinkML schema |
| **Nodes in graph?** | 0 (never instantiated) | 0 (never created) |
| **Used in code?** | `find_shared_imas_mappings()` in client.py | `map_signals_to_imas()` in domain_queries.py, search_tools.py |
| **Relationship model** | Node: `(DataNode)←[:SOURCE_PATH]-(IMASMapping)-[:TARGET_PATH]→(IMASPath)` | Direct edge: `(DataAccess)-[:MAPS_TO_IMAS]->(IMASPath)` |
| **Transformation support** | Full: units_in/out, cocos_source/target, scale, transformation expression, transform_code | None — just a link |
| **Evidence/provenance** | MappingEvidence nodes, agent lifecycle | None |
| **When it connects** | Source=DataNode, Target=IMASPath | Source=DataAccess, Target=IMASPath |

**Decision: Keep IMASMapping. Remove MAPS_TO_IMAS.**

Rationale:
- MAPS_TO_IMAS is a ghost — it was never defined in the schema and never created in
  the graph. It only exists as aspirational Cypher in query functions.
- IMASMapping provides the full mapping model: transformations, evidence, lifecycle,
  code. A direct edge can't carry this metadata.
- The FacilitySignal schema docstring already says: "Use IMASMapping nodes
  (SOURCE_PATH from DataNode, TARGET_PATH to IMASPath) — not a direct MAPS_TO_IMAS
  relationship on FacilitySignal or DataAccess."

**Cleanup task**: Remove all `MAPS_TO_IMAS` references from:
- `imas_codex/graph/domain_queries.py` — `map_signals_to_imas()` query
- `imas_codex/agentic/search_tools.py` — IMAS path reverse-lookup query
- `imas_codex/agentic/server.py` — any references
- `AGENTS.md` — relationship table
- `docs/architecture/signals.md` — relationship documentation
- `plans/features/unified-mcp-tools.md` — query examples
- `tests/graph/test_domain_queries.py` — test assertions
- `tests/agentic/test_search_tools.py` — test assertions

Replace with queries that traverse IMASMapping nodes instead.

## Relationship Direction Consistency

All relationships flow **from the node that "has" or "belongs to"** towards the
node it references. The grammar is: **subject VERB object**.

```
FacilitySignal ──HAS_DATA_SOURCE_NODE──▶ DataNode     ✓ signal HAS node
IMASMapping    ──SOURCE_PATH──▶          DataNode      ✓ mapping's SOURCE is node  
IMASMapping    ──TARGET_PATH──▶          IMASPath      ✓ mapping's TARGET is path
IMASMapping    ──AT_FACILITY──▶          Facility      ✓ mapping is AT facility
IMASMapping    ──HAS_EVIDENCE──▶         MappingEvidence ✓ mapping HAS evidence
IDSRecipe      ──INCLUDES_MAPPING──▶     IMASMapping   ✓ recipe INCLUDES mapping
IDSRecipe      ──USES_SOURCE──▶          DataSource    ✓ recipe USES source
IDSRecipe      ──COVERS──▶              IMASPath      ✓ recipe COVERS path
```

The traversal chain to answer "What IMAS path does this signal map to?" reads
left-to-right following outgoing edges, with one direction reversal at the mapping:

```
(FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(DataNode)
                                                      ↑
(IMASMapping)-[:SOURCE_PATH]->(DataNode)    ← reverse traversal here
(IMASMapping)-[:TARGET_PATH]->(IMASPath)    → forward to answer
```

In Cypher:
```cypher
MATCH (fs:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(dn:DataNode)
      <-[:SOURCE_PATH]-(m:IMASMapping)-[:TARGET_PATH]->(ip:IMASPath)
RETURN fs.id, ip.id, m.units_in, m.units_out, m.transform_code
```

This is the correct directional model — the IMASMapping acts as a **named, rich edge**
between DataNode and IMASPath, with both outgoing relationships.

## The Two-Level Architecture

The mapping problem has two distinct layers. Conflating them was the root mistake.

### Level 1: IMASMapping — "What maps where, and how do we transform it?"

Each `IMASMapping` node is an atomic, validated, evidenced assertion:

> "DataNode property X at facility F maps to IMASPath Y,
>  with unit conversion A→B, COCOS transform C→D, and here is the
>  Python code to execute the transformation."

This is the **knowledge + execution layer**. Agents discover mappings from code,
wiki, documentation, and data validation. Each mapping has:
- **source**: DataNode (or property within it)
- **target**: IMASPath (leaf node in the DD)
- **metadata**: units_in/out, cocos_source/target, scale (documentation)
- **transform_code**: Executable Python code to convert source → target values
- **evidence**: why we believe this mapping is correct
- **lifecycle**: proposed → endorsed → validated (or contested → rejected)

#### Executable Transform Code

Following the established `DataAccess` template pattern (where `data_template`,
`connection_template`, `full_example` store executable Python as strings), 
IMASMapping stores the actual transform code as a `transform_code` property:

```python
IMASMapping(
    id="jet:device_xml:PF:r→pf_active/coil/element/geometry/rectangle/r",
    facility_id="jet",
    source_path="jet:device_xml:p68613:pfcoils:1",
    target_path="pf_active/coil/element/geometry/rectangle/r",
    driver="device_xml",
    # Metadata — documents what the transform does (human-readable, queryable)
    units_in="m",
    units_out="m",
    cocos_source=None,
    cocos_target=None,
    scale=1.0,
    # Code — the actual transform to execute (machine-executable)
    transform_code="value",  # identity: source value passes through unchanged
    # Evidence
    status="validated",
    confidence=1.0,
    notes="Validated via HDF5 round-trip of 22 coils, 1169 elements",
)
```

`transform_code` is a Python expression evaluated with `value` as the input.
The metadata fields (units, COCOS, scale) document and describe the transform;
the code actually performs it. Examples:

| Mapping | units_in | units_out | cocos | transform_code |
|---------|----------|-----------|-------|----------------|
| r → rectangle.r | m | m | — | `value` |
| I_p (COCOS 1→17) | A | A | 1→17 | `-value` |
| T_e (eV→keV) | eV | keV | — | `value * 1e-3` |
| angle (deg→rad) | deg | rad | — | `math.radians(value)` |
| B_tor (sign flip) | T | T | 2→17 | `value * sigma_b0` |
| Complex | — | — | — | `transform_equilibrium(value, cocos_in=2, cocos_out=17)` |

For simple mappings (identity, scale, unit conversion), the code is a one-liner.
For complex physics transforms (COCOS, coordinate changes), it calls functions
from `imas_codex.cocos` or `imas_codex.units`. The code string is evaluated
at assembly time using the transform engine (see Phase 3).

This dual approach means:
- **Metadata** enables graph queries: "show me all mappings that change COCOS"
- **Code** enables execution: the assembler runs the transform without
  reimplementing transform logic from metadata fields

### Level 2: IDSAssembly — "How do sources compose into IDS structure?"

The structural problem — how DataNodes group into array-of-structures entries,
how array properties fan out into sub-arrays, how multiple sources merge —
is a separate concern from field-level mapping.

An IDSAssembly declares:

> "For pf_active.coil: select DataNodes with system=PF from device_xml per epoch.
>  Each DataNode becomes one coil entry. Array-valued properties (r[], z[], dr[], dz[])
>  create one element per array position. Enrichment from jec2020 matches by coil index."

This is the **structural layer**. It lives in the graph as recipe metadata (IDSRecipe)
with relationships to IMASMappings that define the individual field transforms.

### Why Two Levels?

| Concern | IMASMapping | IDSAssembly |
|---------|-------------|-------------|
| Granularity | One property → one IDS leaf | Set of nodes → struct array |
| Knowledge source | Code analysis, wiki, experts | Data structure analysis |
| Changes when... | Physics understanding improves | Data organization changes |
| Contains | Transform code + metadata | Structural rules + selection criteria |
| Lifecycle | proposed → validated | Per-recipe definition |
| Reusable across | Different IDSs using same data | N/A, IDS-specific |

A single DataNode field like `r` might map to different IMAS paths depending
on context (PF coil geometry vs. probe position). The IMASMapping captures
the full context. The IDSAssembly captures how to apply it structurally.

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

| Gap | Root Cause | Fix Location |
|-----|-----------|--------------|
| 0 HAS_DATA_SOURCE_NODE relationships | `create_nodes()` bug: uses `{id: ...}` to match targets, but DataNode identifier is `path` | Fix `client.py` `create_nodes()` + backfill |
| FacilitySignal.system = None (all 1058) | device_xml scanner doesn't set `system` on FacilitySignal | Fix scanner + backfill |
| FacilitySignal units on device_xml signals | Already set correctly by scanner | ✓ No fix needed |
| 0 IMASMapping nodes | Never created | Create from SECTION_METADATA + discovery |
| MAPS_TO_IMAS referenced everywhere | Ghost relationship not in schema | Remove all references |

### Bug: `create_nodes()` relationship target resolution

`create_nodes()` at `client.py` line 603 hardcodes:
```python
MATCH (t:{rel.to_class} {{id: item.{rel.slot_name}}})
```

This always looks up targets by `{id: ...}`. But DataNode's `identifier: true` is
on `path`, not `id`. So `MATCH (t:DataNode {id: "jet:device_xml:..."})` matches
nothing — DataNode nodes have `id=None` and are keyed by `path`.

**Fix**: `create_nodes()` must use `get_identifier(rel.to_class)` to determine
the correct target identifier field:
```python
target_id_field = self.schema.get_identifier(rel.to_class) or "id"
MATCH (t:{rel.to_class} {{{target_id_field}: item.{rel.slot_name}}})
```

This is a general bug affecting any relationship targeting a class whose
identifier is not `id`. DataNode is the primary case.

## Concrete Plan

### Phase 0: Fix Data Foundation in Discovery Tools

These fixes belong in the scanners that create the data, not as afterthought
backfill scripts. Fix the root cause, then re-run discovery to populate correctly.

**0a. Fix `create_nodes()` target identifier resolution (client.py)**

The relationship creation in `create_nodes()` must respect the target class's
identifier field. This is the root cause of 0 HAS_DATA_SOURCE_NODE relationships.

```python
# In create_nodes(), replace hardcoded {id: ...} with schema-aware lookup
target_id_field = self.schema.get_identifier(rel.to_class) or "id"
rel_query = f"""
    UNWIND $batch AS item
    MATCH (n:{label} {{{id_field}: item.{id_field}}})
    MATCH (t:{rel.to_class} {{{target_id_field}: item.{rel.slot_name}}})
    MERGE (n)-[:{rel.cypher_type}]->(t)
"""
```

After this fix, re-running the device_xml scanner will automatically create
HAS_DATA_SOURCE_NODE relationships because `create_nodes("FacilitySignal", ...)`
will correctly match `(t:DataNode {path: item.data_source_node})`.

**0b. Set FacilitySignal.system in device_xml scanner**

The scanner's `SECTION_METADATA` knows the system (PF, CI, PS, MP, FL) for each
section. Add `system=meta.get("system")` to the FacilitySignal constructor:

```python
# In device_xml.py, _persist_graph_nodes(), around line 395
all_signals[sig_id] = FacilitySignal(
    id=sig_id,
    facility_id=facility,
    status=FacilitySignalStatus.discovered,
    physics_domain=meta["physics_domain"],
    system=meta.get("system"),  # ← ADD THIS
    name=f"{meta['label'].title()} {inst_id} {field_meta['desc']}",
    ...
)
```

**Backfill existing data** after fixing the scanner:
```cypher
-- Backfill system from accessor prefix
MATCH (fs:FacilitySignal)
WHERE fs.facility_id = 'jet' AND fs.data_source_name = 'device_xml'
  AND fs.system IS NULL
WITH fs,
  CASE
    WHEN fs.accessor STARTS WITH 'device_xml:magprobes' THEN 'MP'
    WHEN fs.accessor STARTS WITH 'device_xml:flux' THEN 'FL'
    WHEN fs.accessor STARTS WITH 'device_xml:pfcoils' THEN 'PF'
    WHEN fs.accessor STARTS WITH 'device_xml:pfcircuits' THEN 'CI'
    WHEN fs.accessor STARTS WITH 'device_xml:pfpassive' THEN 'PS'
  END AS sys
SET fs.system = sys
```

**0c. Add `transform_code` field to IMASMapping schema**

Extend `facility.yaml` to add the executable transform:

```yaml
IMASMapping:
  attributes:
    # ... existing fields ...
    transform_code:
      description: >-
        Python expression to transform source value to target value.
        Evaluated with 'value' as input variable. Access to math, numpy,
        and imas_codex.cocos/units functions.
        Simple: "value" (identity), "value * 1e-3" (scale), "-value" (sign flip)
        Complex: "transform_equilibrium(value, cocos_in=2, cocos_out=17)"
        The metadata fields (units_in/out, cocos_source/target, scale) document
        what the transform does; this field is the executable implementation.
    source_property:
      description: >-
        Name of the property on the source DataNode to extract.
        E.g., "r", "z", "dr", "dz", "turnsperelement", "description".
        When source DataNode has multiple properties, this identifies
        which one this mapping applies to.
```

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

Each field becomes an IMASMapping node with executable transform code:

```python
# r → pf_active/coil/element/geometry/rectangle/r (identity, meters)
IMASMapping(
    id="jet:PF:r→pf_active/coil/element/geometry/rectangle/r",
    facility_id="jet",
    source_path="jet:device_xml:p68613:pfcoils:1",  # representative DataNode
    target_path="pf_active/coil/element/geometry/rectangle/r",
    source_property="r",
    driver="device_xml",
    units_in="m",
    units_out="m",
    transform_code="value",  # identity transform
    status="validated",
    confidence=1.0,
)

# dr → width (name change, same units)
IMASMapping(
    id="jet:PF:dr→pf_active/coil/element/geometry/rectangle/width",
    facility_id="jet",
    source_path="jet:device_xml:p68613:pfcoils:1",
    target_path="pf_active/coil/element/geometry/rectangle/width",
    source_property="dr",
    driver="device_xml",
    units_in="m",
    units_out="m",
    transform_code="value",
    status="validated",
    confidence=1.0,
)
```

**Source_path granularity**: One IMASMapping per field pattern (not per DataNode).
The mapping connects to a *representative* DataNode. The `source_property` field
says which property to read. The IDSAssembly rules determine *which* DataNodes
to apply it to at assembly time.

### Phase 2: IDSRecipe as Structural Composition

The existing IDSRecipe schema holds structural assembly configuration.
Add `INCLUDES_MAPPING` relationship to compose IMASMappings:

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

### Phase 3: Graph-Driven Assembly Engine with Executable Transforms

Replace the current Cypher-in-YAML assembler with one that reads from the graph
and executes transform code:

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

            # 3c. For each entry, apply field mappings with executable transforms
            for entry_data in entries:
                entry = struct_array[i]
                for mapping in mappings:
                    value = entry_data.get(mapping.source_property)
                    if value is not None:
                        value = self._execute_transform(value, mapping)
                        _set_nested(entry, mapping.target_imas_path, value)

        return ids

    def _execute_transform(self, value, mapping):
        """Execute the mapping's transform_code with safety constraints."""
        if not mapping.transform_code or mapping.transform_code == "value":
            return value  # Fast path: identity transform

        # Build execution context with allowed modules/functions
        context = {
            "value": value,
            "math": math,
            "np": numpy,
            "convert_units": convert_units,
            "cocos_sign": cocos_sign_factor,
        }
        return eval(mapping.transform_code, {"__builtins__": {}}, context)
```

The `eval()` call uses a restricted builtins context. The transform_code is
authored by agents/humans through the mapping lifecycle, not from untrusted
input. This follows the same trust model as DataAccess templates (which also
store executable Python code strings).

### Phase 4: Unit and COCOS Integration

**Units**: `imas_codex/units/` has pint with DD aliases. Add `convert_units()`:

```python
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
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

A `cocos_sign_factor()` helper is available in `transform_code`:
```python
# transform_code for a COCOS-sensitive mapping:
"value * cocos_sign('ip_like', cocos_in=1, cocos_out=17)"
```

### Phase 5: Expand to All Machine Description IDSs

With the graph-driven architecture, adding new IDSs becomes:

1. Create IMASMapping nodes for new field transformations (with transform_code)
2. Create/update IDSRecipe with structural assembly config
3. Link mappings to recipe via INCLUDES_MAPPING
4. Run `imas-codex imas export jet magnetics --epoch p68613`

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

### Phase 6: Remove MAPS_TO_IMAS Ghost Relationship

Clean up all references to the never-defined, never-instantiated MAPS_TO_IMAS
relationship. Replace with IMASMapping traversal queries.

**Files to update:**

| File | Change |
|------|--------|
| `imas_codex/graph/domain_queries.py` | Rewrite `map_signals_to_imas()` to traverse IMASMapping |
| `imas_codex/agentic/search_tools.py` | Update reverse-lookup query to use IMASMapping |
| `imas_codex/agentic/server.py` | Remove any MAPS_TO_IMAS references |
| `AGENTS.md` | Update relationship table |
| `docs/architecture/signals.md` | Update documentation |
| `plans/features/unified-mcp-tools.md` | Update query examples |
| `tests/graph/test_domain_queries.py` | Update test assertions |
| `tests/agentic/test_search_tools.py` | Update test assertions |

### Phase 7: Mapping Discovery and Agent Workflows

The IMASMapping lifecycle (proposed → endorsed → validated) enables agent-driven
mapping discovery:

1. **Code analysis agent**: Scans source files for IDS read/write patterns.
   "This code writes `q_profile` to `equilibrium/profiles_1d/q`" → propose mapping
   with `transform_code` derived from the source code.
2. **Wiki analysis agent**: Finds documentation about signal conventions.
   "JET uses COCOS 1 for plasma current" → evidence on mapping.
3. **Data validation agent**: Tests a mapping against real shot data.
   "Reading shot 99999, signal X returns Y, which matches expectations" → validate.
4. **Human review**: Contended or high-impact mappings get human sign-off.

This is the long-term value — the graph accumulates verified, executable mapping
knowledge, not just one developer's YAML recipe.

## Migration Path

### Step 1: Fix `create_nodes()` target identifier bug (client.py)

This is a general infrastructure bug affecting any relationship targeting a class
with a non-`id` identifier. Fix it first — it unblocks everything.

### Step 2: Fix device_xml scanner (system, backfill)

Add `system` to FacilitySignal creation. Re-run scanner to populate correctly.
After the `create_nodes()` fix, HAS_DATA_SOURCE_NODE relationships will be
created automatically on re-scan.

### Step 3: Add `transform_code` and `source_property` to IMASMapping schema

Extend the schema with executable transform support. Run `build-models`.

### Step 4: Create IMASMapping nodes for pf_active

Prove that graph-driven mappings with executable transforms produce identical
output to the current YAML recipe. Compare HDF5 files byte-for-byte.

### Step 5: Build graph-driven assembler

New assembler that reads IMASMappings + IDSRecipe from graph, executes
transform_code per mapping. When outputs match, switch the CLI.

### Step 6: Remove MAPS_TO_IMAS

Clean up ghost references across codebase and tests.

### Step 7: Expand to magnetics, pf_passive, wall

Add new IDSs using only graph operations (create IMASMapping nodes,
configure IDSRecipe). No new Python code per IDS.

## Open Design Questions

1. **IMASMapping source_path granularity**: One mapping per field-pattern, or per DataNode?
   Pattern mappings are compact but require the IDSAssembly to know which property
   to extract. Per-DataNode mappings are precise but explode in count.
   Current recommendation: one per field-pattern with `source_property`.

2. **Assembly config storage**: JSON property on IDSRecipe node, or separate YAML file
   referenced by path? JSON-in-graph is self-contained but hard to review. YAML is
   reviewable but creates a file dependency.

3. **Enrichment as separate mappings**: When jec2020 provides coil names, is that a
   separate IMASMapping (jec2020 DataNode → pf_active/coil/name) or an enrichment
   concern at the IDSAssembly level? If it's a mapping, how do we handle merge priority?

4. **Temporal data**: Machine description is static per epoch. But `pf_active/coil/current`
   is time-dependent. Do we need a different assembly pattern for time traces, or does
   the same architecture extend naturally?

5. **Transform code safety**: Current design uses `eval()` with restricted builtins.
   This is consistent with DataAccess templates (which store Python code strings).
   Should we add a validation step (AST check for forbidden operations) or is the
   agent lifecycle review sufficient?

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `imas_codex/graph/client.py` | Fix | Target identifier resolution in `create_nodes()` |
| `imas_codex/schemas/facility.yaml` | Modify | Add `transform_code`, `source_property` to IMASMapping; extend IDSRecipe |
| `imas_codex/discovery/signals/scanners/device_xml.py` | Fix | Set `system` on FacilitySignal |
| `imas_codex/ids/assembler.py` | Rewrite | Graph-driven assembly with executable transforms |
| `imas_codex/ids/transforms.py` | Create | Transform execution engine + unit/COCOS helpers |
| `imas_codex/ids/graph_ops.py` | Create | IMASMapping + IDSRecipe graph queries |
| `imas_codex/graph/domain_queries.py` | Modify | Replace MAPS_TO_IMAS with IMASMapping traversal |
| `imas_codex/agentic/search_tools.py` | Modify | Replace MAPS_TO_IMAS with IMASMapping traversal |
| `tests/ids/test_assembler.py` | Update | Test graph-driven assembly |
| `tests/ids/test_transforms.py` | Create | Test transform execution + unit/COCOS |
| Multiple docs/plans/tests | Modify | Remove MAPS_TO_IMAS ghost references |
