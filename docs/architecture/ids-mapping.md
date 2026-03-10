# IDS Mapping Architecture

This document describes the graph-driven IDS mapping and assembly system used to
produce complete IMAS IDS instances from facility knowledge graph data.

## Overview

The system converts facility-specific data (device descriptions, diagnostic geometry,
wall contours) into standardised IMAS Interface Data Structures using two abstraction
layers:

1. **IMASMapping nodes** — declarative field-level transformations stored in the graph
2. **IMASMapping nodes** — structural assembly rules that define how DataNodes group into
   IDS array-of-structures entries

Both are first-class graph nodes defined in the LinkML schema, queryable via Cypher,
and created through `seed_ids_mappings()` or the `ids seed` CLI command.

## Graph Model

```
(SignalNode) ←[:SOURCE_PATH]— (IMASMapping) —[:TARGET_PATH]→ (IMASNode)
                                  ↑
                          [:INCLUDES_MAPPING]
                                  |
                            (IMASMapping) —[:AT_FACILITY]→ (Facility)
```

### IMASMapping

Each mapping node encodes one field-level transformation:

| Property | Purpose |
|----------|---------|
| `source_property` | SignalNode property name (e.g., `r`, `z`, `angle`) |
| `transform_code` | Executable Python expression (e.g., `math.radians(value)`) |
| `units_in` / `units_out` | Automatic unit conversion via pint |
| `cocos_source` / `cocos_target` | COCOS convention for sign/scale transforms |
| `driver` | Data source type (`device_xml`) |
| `status` | Lifecycle state (`validated`, etc.) |

The `transform_code` expression is evaluated with a controlled context providing
`value` (the input), `math`, `numpy`, `cocos_sign()`, and `convert_units()`.

### IMASMapping

Each recipe node stores structural assembly configuration as a JSON blob in
`assembly_config`. This defines:

- **Static fields** — values set directly on the IDS (e.g., `ids_properties.homogeneous_time`)
- **Sections** — one per IDS struct-array, defining source queries, structure type,
  enrichment, and sub-array initialization

## Assembly Modes

`IDSAssembler` supports two modes, selected automatically:

### Graph-Driven (preferred)

Reads `IMASMapping` + `IMASMapping` nodes from the graph. The recipe's `assembly_config`
defines structural patterns; mapping nodes define field transforms with executable code.

```python
assembler = IDSAssembler("jet", "pf_active")
ids = assembler.assemble("p68613")
```

### YAML Fallback

When no graph recipe exists, loads a YAML recipe from `imas_codex/ids/recipes/`.
Used for prototyping or facilities without graph data.

## Structure Types

Each IDS section declares a `structure` type that controls how DataNodes map to IDS
array entries:

### `array_per_node`

One SignalNode produces one struct-array entry. Used for coils, probes, loops.

```
SignalNode[0] → coil[0]
SignalNode[1] → coil[1]
...
```

Supports `elements` config for sub-arrays (e.g., coil geometry elements with
`geometry_type`) and `init_arrays` for pre-sizing nested arrays (e.g.,
`flux_loop.position`).

### `nested_array`

DataNodes populate a nested array within a parent container. Used for wall limiter
contours where the IDS path is `description_2d[0].limiter.unit[i]`.

Configuration:

| Field | Purpose |
|-------|---------|
| `nested_path` | Dot-separated path to the nested array (e.g., `limiter.unit`) |
| `parent_size` | Number of parent entries to create (typically 1) |
| `select_via` | Relationship-based node selection (e.g., `USES_LIMITER`) |

```
StructuralEpoch —[:USES_LIMITER]→ SignalNode[0] → unit[0]
                                   SignalNode[1] → unit[1]
```

## Node Selection

DataNodes are selected per-section using two strategies:

### Property-based (default)

Matches on `system`, `data_source`, and epoch field:

```cypher
MATCH (d:SignalNode {facility_id: $f, data_source_name: $ds, system: $sys})
WHERE d.introduced_version = $epoch_id
```

### Relationship-based (`select_via`)

Traverses a relationship from the structural epoch. Used when DataNodes lack a
direct epoch property (e.g., limiter nodes selected via `USES_LIMITER`):

```cypher
MATCH (se:StructuralEpoch {id: $epoch_id})-[:USES_LIMITER]->(d:SignalNode)
```

## COCOS Handling

IMAS paths carry a `cocos_label_transformation` property (e.g., `ip_like`, `psi_like`)
from the Data Dictionary. During mapping load, the system warns if a COCOS-sensitive
path has no `cocos_source` set on its mapping.

The `cocos_sign()` function computes sign/scale factors using the COCOS parameter
decomposition from Sauter & Medvedev (2013). Supported labels:

| Label | Transformation |
|-------|---------------|
| `ip_like` | `(σ_RφZ · σ_Bp)_out / (σ_RφZ · σ_Bp)_in` |
| `b0_like` | `σ_RφZ_out / σ_RφZ_in` |
| `psi_like` | `σ_Bp · (2π)^(1-e_Bp)` ratio |
| `one_like` | Always 1 (no transform) |

COCOS transforms compose with unit conversion — both are applied automatically during
assembly when the mapping specifies them.

## Unit Conversion

When `units_in` and `units_out` differ on a mapping, conversion is automatic via pint
with the DD unit alias registry. Common patterns:

- `deg → rad` for probe angles
- `ohm → ohm` (identity, for documentation)
- `m → m` (identity, confirms dimensional consistency)

## Supported IDS

### pf_active

- **Sections**: `coil` (6 mappings, geometry + turns), `circuit` (1 mapping, name)
- **Structure**: `array_per_node` with `elements` sub-arrays
- **Enrichment**: JEC-2020 geometry data merged per coil index

### magnetics

- **Sections**: `b_field_pol_probe` (4 mappings), `flux_loop` (4 mappings)
- **Structure**: `array_per_node`
- **Special**: `init_arrays: {"position": 1}` for flux_loop position sub-struct;
  angle conversion from degrees to radians via `math.radians(value)`

### pf_passive

- **Sections**: `loop` (6 mappings, geometry + resistance)
- **Structure**: `array_per_node` with `elements` sub-arrays

### wall

- **Sections**: `description_2d` (3 mappings: `r_contour`, `z_contour`, `description`)
- **Structure**: `nested_array` — `description_2d[0].limiter.unit[i].outline.r/z`
- **Selection**: `select_via: USES_LIMITER` from structural epoch

## CLI

```bash
# Seed all mappings + recipe for an IDS
uv run imas-codex ids seed jet pf_active --dd-version 4.1.1

# List available recipes (checks graph first, falls back to YAML)
uv run imas-codex ids list

# Assemble and export
uv run imas-codex ids export jet pf_active p68613 -o output.json
```

## Design Decisions

1. **IMASMapping over MAPS_TO_IMAS** — Mappings are full graph nodes with
   transformation metadata, not direct edges. The ghost `MAPS_TO_IMAS` relationship
   was never defined in the schema and has been removed from all code.

2. **Executable transforms** — `transform_code` uses `eval()` with a restricted
   context, following the same trust model as `DataAccess` templates. Code is authored
   through the mapping lifecycle, not from external input.

3. **Dual-mode assembler** — Graph-driven mode is preferred; YAML fallback enables
   prototyping. The assembler auto-selects based on recipe availability.

4. **Declarative structure types** — Assembly patterns (`array_per_node`,
   `nested_array`) are declared in config, not hardcoded in the assembler.
   New patterns can be added without changing existing code.

5. **MappingSpec constants** — Canonical mappings are defined as typed tuples in
   `graph_ops.py`, seeded into the graph via `seed_ids_mappings()`. This keeps
   mapping definitions version-controlled alongside the code.

## Modules

| Module | Purpose |
|--------|---------|
| `imas_codex/ids/graph_ops.py` | Graph queries, mapping specs, seed function |
| `imas_codex/ids/assembler.py` | Dual-mode assembly engine |
| `imas_codex/ids/transforms.py` | Transform execution, COCOS, units, `set_nested()` |
| `imas_codex/cli/ids.py` | CLI commands (`seed`, `list`, `export`, `summary`) |
