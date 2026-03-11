# IDS Assembly: From Graph Data to IMAS IDS Files

Status: **Design Complete — Ready for Implementation**
Priority: High — enables machine description export for imas-ambix

## Problem Statement

The knowledge graph contains detailed machine description data for JET organized
as individual DataNodes — geometry, coil positions, probe locations, circuit
configurations. But there is no mechanism to:

1. **Declare** which IMAS IDSs can be produced for a facility
2. **Assemble** a complete IDS from multiple data sources and epochs
3. **Track** completeness — which required IDS fields are populated
4. **Export** valid IMAS-compliant HDF5 files

The fundamental gap: we have **individual data points** but no **IDS-level concept**
that aggregates them into exportable structures.

## Validated Proof of Concept

Successfully assembled and round-tripped a complete `pf_active` IDS for JET:

- **22 PF coils** with **1,169 elements** from device_xml epoch p68613
- Multi-element coil geometry (2 to 213 elements per coil)
- Cross-source enrichment: device_xml geometry + jec2020 naming
- Written to HDF5 (244 KB) and read back with full fidelity
- Used `imas-python 2.2.0` with DD version `4.1.1`

### Key API findings

```python
# Create and open in one call (mode='x')
entry = imas.DBEntry('imas:hdf5?path=/path/to/ids', 'x')

# ids_properties fields: comment, provider, homogeneous_time
pf.ids_properties.homogeneous_time = 0  # static machine description
pf.ids_properties.provider = 'imas-codex'

# IDSFloat0D/IDSString0D need explicit conversion for format strings
r = float(coil.element[0].geometry.rectangle.r)
name = str(coil.name)
```

## Data Inventory

JET machine description data in the graph (6,623 DataNodes):

| System | Source | Nodes | Target IDS |
|--------|--------|------:|------------|
| PF | device_xml | 308 | pf_active.coil |
| PF | jec2020_geometry | 20 | pf_active.coil (enrichment) |
| PF | pf_coil_turns | 12 | pf_active.coil.element.turns |
| PS | device_xml | 672 | pf_passive.loop |
| MP | device_xml | 2,674 | magnetics.bpol_probe |
| MP | jec2020_geometry | 95 | magnetics.bpol_probe (enrichment) |
| MP | magnetics_config | 1,838 | magnetics.bpol_probe (calibration) |
| FL | jec2020_geometry | 36 | magnetics.flux_loop |
| FL | magnetics_config | 650 | magnetics.flux_loop (calibration) |
| CI | device_xml | 224 | pf_active.circuit |
| CI | jec2020_geometry | 10 | pf_active.circuit (enrichment) |
| DV | magnetics_config | 40 | magnetics.diamagnetic_flux |
| IP | magnetics_config | 19 | magnetics.ip |
| LIM | jec2020_geometry | 1 | wall.description_2d |
| GR | greens_table | 5 | (pf_active mutual inductances) |

**Targetable IDSs from current data: `pf_active`, `pf_passive`, `magnetics`, `wall`**

## Architecture

Three-layer design separating concerns:

### Layer 1: Graph Schema — IDSRecipe (what can we produce?)

A new `IDSRecipe` node type answers graph-level questions:
- "Which IDSs can we produce for JET?"
- "How complete is each one?"
- "Which data sources contribute?"
- "Which epochs are covered?"

```yaml
IDSRecipe:
  id: "jet:pf_active"              # facility:ids_name
  facility_id: jet
  ids_name: pf_active
  dd_version: "4.1.1"
  status: draft | complete | validated
  completeness: 0.85               # fraction of required fields populated
  provider: "imas-codex"
  description: "JET PF active coil system from device_xml + jec2020"
  # Relationships:
  #   USES_SOURCE → DataSource (device_xml, jec2020, pf_coil_turns)
  #   COVERS → IMASPath (which DD paths are populated)
  #   FOR_EPOCH → StructuralEpoch (which epochs this recipe handles)
```

### Layer 2: Mapping Definitions — YAML (how to transform data)

Per-IDS YAML files declare field-level mappings. These are versionable,
reviewable, and don't require code changes to update.

Location: `imas_codex/mappings/ids_recipes/`

```yaml
# imas_codex/mappings/ids_recipes/jet_pf_active.yaml
ids_name: pf_active
facility_id: jet
dd_version: "4.1.1"

static:
  ids_properties.homogeneous_time: 0
  ids_properties.comment: "JET PF active coils assembled by imas-codex"

arrays:
  coil:
    # Primary source: device_xml PF coils per epoch
    primary:
      data_source: device_xml
      system: PF
      epoch_field: introduced_version
      order_by: coil_index  # extracted from path suffix
      fields:
        # DataNode property → IDS field within each coil
        element_source: array   # r[], z[], dr[], dz[] are arrays → creates elements
        element_fields:
          geometry.geometry_type: 2  # constant: rectangle
          geometry.rectangle.r: r
          geometry.rectangle.z: z
          geometry.rectangle.width: dr
          geometry.rectangle.height: dz
          turns_with_sign: turnsperelement

    # Enrichment: jec2020 provides coil names and total turn counts
    enrichment:
      - data_source: jec2020_geometry
        system: PF
        match_by: coil_index
        fields:
          name: description  # extract from "PF coil 1 (P1/ME)"
          description: description

    # Supplementary: pf_coil_turns provides precision turn data
    supplements:
      - data_source: pf_coil_turns
        fields:
          # Maps to specific coils by name matching

  circuit:
    primary:
      data_source: device_xml
      system: CI
      epoch_field: introduced_version
      order_by: circuit_index
      fields:
        name: description
        # Circuit→coil connections stored as relationships
```

### Layer 3: Assembly Engine — Python (generic builder)

A generic engine that:
1. Reads the YAML mapping definition
2. Queries the graph for matching DataNodes (epoch-aware)
3. Constructs the imas-python IDS object
4. Writes to HDF5

```python
# imas_codex/mappings/assembly.py

class IDSAssembler:
    """Assembles IMAS IDS instances from graph data using mapping definitions."""

    def __init__(self, recipe_path: Path):
        self.recipe = yaml.safe_load(recipe_path.read_text())

    def assemble(self, epoch: str | None = None) -> IDSToplevel:
        """Build an IDS instance for the given epoch."""
        factory = imas.IDSFactory(self.recipe['dd_version'])
        ids = factory.new(self.recipe['ids_name'])

        # Set static properties
        for path, value in self.recipe.get('static', {}).items():
            self._set_nested(ids, path, value)

        # Build array-of-structures
        for array_name, array_def in self.recipe.get('arrays', {}).items():
            self._build_array(ids, array_name, array_def, epoch)

        return ids

    def export(self, output_path: Path, epoch: str | None = None):
        """Assemble and write to HDF5."""
        ids = self.assemble(epoch)
        uri = f'imas:hdf5?path={output_path}'
        entry = imas.DBEntry(uri, 'x')
        entry.put(ids)
        entry.close()
```

### CLI Integration

> **CLI section superseded by `cli-unification.md`.** IDS commands are now under
> `imas-codex imas` (not `imas-codex ids`).

```bash
# List available IDS recipes for a facility
imas-codex imas list jet

# Show recipe details and completeness
imas-codex imas show jet pf_active

# Export IDS to HDF5
imas-codex imas export jet pf_active --epoch p68613 --output jet_pf_active.h5

# Export all machine description IDSs for an epoch
imas-codex imas export jet --epoch p68613 --output jet_machine_desc/
```

## Graph Schema Extensions

### New Node: IDSRecipe

```yaml
IDSRecipe:
  description: >-
    A recipe for assembling a complete IMAS IDS from facility graph data.
    Tracks which data sources contribute and which DD paths are covered.
    The assembly logic is defined in YAML recipe files, not in the graph.

    Query patterns:
      # What IDSs can we produce?
      MATCH (r:IDSRecipe {facility_id: 'jet'}) RETURN r.ids_name, r.completeness

      # Which sources contribute to pf_active?
      MATCH (r:IDSRecipe {id: 'jet:pf_active'})-[:USES_SOURCE]->(ds:DataSource)
      RETURN ds.name, ds.source_type

      # Which DD paths are covered?
      MATCH (r:IDSRecipe {id: 'jet:pf_active'})-[:COVERS]->(ip:IMASPath)
      RETURN ip.id, ip.documentation
  class_uri: facility:IDSRecipe
  attributes:
    id:
      identifier: true
      description: "Composite key: facility:ids_name (e.g., 'jet:pf_active')"
      required: true
    facility_id:
      description: Parent facility
      required: true
      range: Facility
      annotations:
        relationship_type: AT_FACILITY
    ids_name:
      description: "IMAS IDS name (e.g., 'pf_active', 'magnetics', 'wall')"
      required: true
    dd_version:
      description: "Target DD version for this recipe"
      required: true
    status:
      description: Recipe lifecycle status
      range: IDSRecipeStatus
      required: true
    completeness:
      description: >-
        Fraction of required IDS fields covered (0.0-1.0).
        Computed by comparing covered IMASPaths against total
        required paths in the DD for this IDS.
      range: float
    description:
      description: Human-readable description of what this recipe produces
    provider:
      description: "Provider string for ids_properties.provider"
    recipe_file:
      description: "Path to YAML recipe file (relative to mappings/ids_recipes/)"
    # Epoch coverage
    epoch_count:
      description: Number of structural epochs this recipe can produce for
      range: integer
    first_epoch_shot:
      description: First shot covered by this recipe
      range: integer
    last_epoch_shot:
      description: Last shot covered (null = current)
      range: integer
    # Relationships
    uses_source:
      description: DataSources contributing to this recipe
      range: DataSource
      multivalued: true
      inlined: false
      annotations:
        relationship_type: USES_SOURCE
    covers:
      description: IMASPaths populated by this recipe
      range: IMASPath
      multivalued: true
      inlined: false
      annotations:
        relationship_type: COVERS
    created_at:
      description: When this recipe was created
      range: datetime
    updated_at:
      description: When this recipe was last updated
      range: datetime
    last_exported_at:
      description: When this recipe was last exported to HDF5
      range: datetime

IDSRecipeStatus:
  description: Lifecycle status for IDS recipes
  permissible_values:
    draft:
      description: Recipe is being developed, may be incomplete
    complete:
      description: Recipe covers all available fields, ready for export
    validated:
      description: Exported IDS has been validated against reference data
    stale:
      description: Underlying data has changed, recipe needs update
```

## Implementation Plan

### Phase 1: Foundation (pf_active proof of concept)

1. **Add IDSRecipe to graph schema** (`facility.yaml`)
   - Add IDSRecipeStatus enum
   - Add IDSRecipe class with relationships
   - Run `uv run build-models --force`

2. **Create mapping recipe format**
   - Define YAML schema for recipe files
   - Create `imas_codex/mappings/ids_recipes/jet_pf_active.yaml`
   - Implement recipe loader with validation

3. **Build assembly engine**
   - `imas_codex/mappings/assembly.py` — generic assembler
   - Graph query functions for epoch-aware DataNode retrieval
   - Field-level mapping from DataNode properties to IDS fields
   - Multi-source merging (primary + enrichment + supplements)

4. **Create pf_active builder** as reference implementation
   - Load device_xml PF DataNodes for a given epoch
   - Merge jec2020 names and pf_coil_turns data
   - Handle multi-element coils (R[], Z[], dR[], dZ[] arrays)
   - Build circuit connections from CI system DataNodes

5. **Add CLI commands** *(superseded by cli-unification.md)*
   - `imas-codex imas list <facility>`
   - `imas-codex imas export <facility> <ids_name> [--epoch] [--output]`
   - `imas-codex imas show <facility> <ids_name>`

### Phase 2: Expand to related IDSs

6. **magnetics IDS recipe**
   - bpol_probe from MP DataNodes (2,674 device_xml + calibration)
   - flux_loop from FL DataNodes (650 magnetics_config)
   - ip from IP DataNodes
   - Cross-reference sensor_calibration data

7. **pf_passive IDS recipe**
   - loop geometry from PS DataNodes (672 device_xml)

8. **wall IDS recipe**
   - description_2d from LIM DataNodes (limiter contour)
   - First element from FE DataNode

### Phase 3: Completeness tracking

9. **Populate IDSRecipe graph nodes**
   - Create recipe nodes for each buildable IDS
   - Link to contributing DataSources via USES_SOURCE
   - Link to covered IMASPaths via COVERS
   - Compute completeness scores

10. **MCP integration**
    - `export_ids(facility, ids_name, epoch)` tool
    - Recipe completeness in facility overview
    - Integration with imas-ambix pipeline


## Design Decisions

### Why YAML recipes, not pure graph?

Field-level mappings (DataNode.rCentre → element.geometry.rectangle.r) are
transformation rules, not knowledge graph data. They are:
- **Versionable** — review changes in git
- **Testable** — unit tests can validate mappings
- **Portable** — same recipe format across facilities
- **Simple** — YAML is readable without graph queries

The graph provides the **data inventory** (what DataNodes exist) and
**metadata** (completeness, coverage). The YAML provides the **rules**.

### Why not extend IMASMapping?

IMASMapping is designed for individual signal-to-path mappings with
evidence and voting workflows. IDS assembly needs:
- **Array construction** — building array-of-structures from sets of DataNodes
- **Multi-source merging** — combining geometry, names, calibration
- **Epoch awareness** — different configurations per shot range
- **Sub-array nesting** — coil → element → geometry → rectangle

These are structural transformations, not individual mappings. IMASMapping
will still be used for temporal signal mappings (plasma current, etc.),
while IDSRecipe handles machine description assembly.

### Why epoch-aware, not shot-aware?

Machine description changes at epoch boundaries (structural epochs),
not per-shot. The recipe resolves the correct epoch for a given shot
number, then queries DataNodes for that epoch's introduced_version.

## Mapping: JET Coil Index ↔ IMAS Array Index

The device_xml coil numbering (1-22) defines the IMAS array ordering.
The JEC2020 coil indices (1, 2, 3, 4, 9, 10, ...) are a _subset_ 
used for enrichment matching.

```
device_xml coil 1  → pf_active.coil[0]  (P1/ME - central solenoid main+ext)
device_xml coil 2  → pf_active.coil[1]  (P1/MC - central solenoid main central)
device_xml coil 3  → pf_active.coil[2]  (P2/SUI - shaping upper inner, 142 elem)
...
device_xml coil 19 → pf_active.coil[18] (D1/T200 - divertor coil 1)
device_xml coil 22 → pf_active.coil[21] (D4/T200 - divertor coil 4)
```

JEC2020 indices map differently because they number individual filaments,
not the aggregate coils that device_xml uses. The matching uses the coil
index extracted from the DataNode path suffix.
