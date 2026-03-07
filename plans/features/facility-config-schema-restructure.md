# Facility Config Schema Restructure

**Goal**: Replace ad-hoc, facility-specific config sections with generic, reusable schema types. Any facility should be able to describe its machine description data sources — files, calibration epochs, static signals — without schema changes. Exploration notes move out of structured config.

**Scope**: `facility_config.yaml` LinkML schema, all 4 facility configs, generated models.

**Priority**: Precedes the JET machine description ingestion plan. That plan's scanners need typed config inputs, and this plan provides them — generically.

---

## Problem Analysis

### The core problem

The previous proposal (v1) tried to fix untyped config by creating `JEC2020Config`, `MCFGConfig`, `PPFStaticGeometryConfig` — types named after JET-internal acronyms, tied to specific file names and directory structures. That's not a schema, it's a glossary. It promotes the clutter from YAML into LinkML without solving the underlying problem: **the schema models JET's filing system rather than the data patterns common to all fusion facilities.**

Every tokamak/stellarator has:
- **Files** containing machine geometry (vessel, limiters, coils, probes, iron core)
- **Calibration data** that changes at specific shots/pulses
- **Signals in existing data systems** (MDSplus, PPF, IMAS) that contain static geometry
- **Versioned configurations** that map shot ranges to hardware states

These are the generic patterns. JEC2020 is an instance of "XML geometry files." MCFG is an instance of "sensor calibration with epoch tracking." PPF static geometry is an instance of "signals in a data system that happen to contain static data." The schema should model the pattern, not the instance.

### What the current schema gets right

1. **`SourceConfig`** — already a good generic model: `source_name`, `versions[]` (shot ranges), `systems[]` (physical subsystems). Used for MDSplus trees but the structure is reusable.
2. **`SourceVersion`** — shot-range epoch indexing. Reusable as-is.
3. **`SourceSystem`** — physical subsystem grouping (symbol, name, parameters). Reusable as-is.
4. **`DataSystemBase` mixin** — `setup_commands`, `python_command`. Reusable for any data source.
5. **`DataSystemsConfig` → scanner dispatch** — keys map to scanner classes. Clean.

### What's wrong

| Problem | Example | Why it matters |
|---------|---------|----------------|
| Untyped file-based sources | `jec2020`, `mcfg` in jet.yaml are raw dicts | Scanners use `dict.get()` with no validation |
| Facility-specific type names | `DeviceXMLConfig` assumes git repo + EFIT format | Another facility's geometry files can't use it |
| `LimiterVersion` missing fields | `source`, `n_points` used in YAML but not in schema | Schema doesn't match reality |
| Notes as config | 43 prose entries in `data_access_patterns.notes` | Config unreadable, loaded on every `get_facility()` call |
| No way to annotate existing data system signals as "static" | PPF signals with geometry live outside `ppf` config | Scanner must hardcode signal list or read untyped config |

---

## Design: Generic before specific

### Principle

The schema models **data organization patterns**, not file names or facility acronyms. Three new generic types replace all facility-specific proposals:

1. **`StaticSourceConfig`** — a named collection of files providing machine description data
2. **`StaticFileConfig`** — a single file within a source
3. **`StaticSignalRef`** — a signal in an existing data system that contains static data

These compose with existing types (`SourceVersion`, `SourceSystem`) and work for any facility without schema modification.

### Why this works across facilities

| Facility | Data source | Previous approach | Generic approach |
|----------|-------------|-------------------|------------------|
| JET | JEC2020 XML geometry | `JEC2020Config` (JET-specific) | `StaticSourceConfig(name="jec2020", format=xml)` |
| JET | MCFG sensor positions | `MCFGConfig` (JET-specific) | `StaticSourceConfig(name="mcfg", format=text)` |
| JET | PPF limiter contour | `PPFStaticGeometryConfig` (JET-specific) | `StaticSignalRef` on existing `PPFConfig` |
| TCV | CAD vessel geometry | Would need `TCV_CADConfig` | `StaticSourceConfig(name="vessel_cad", format=csv)` |
| ITER | IMAS machine_description IDS | Would need `ITERMachDescConfig` | `StaticSignalRef` on existing `IMASConfig` |
| W7-X | VMEC equilibrium files | Would need `VMECConfig` | `StaticSourceConfig(name="vmec_geometry", format=netcdf)` |
| KSTAR | EFIT geometry files | Would need another `EFITConfig` | `StaticSourceConfig(name="efit_geometry", format=text)` |

One schema type handles all cases. Adding a new facility's machine description data is a YAML-only change.

---

## Proposed Schema Types

### StaticSourceConfig

A named collection of files providing static (non-shot-varying) machine description or calibration data. Generic container — no facility-specific fields.

```yaml
StaticSourceConfig:
  description: >-
    A named collection of files providing machine description, calibration,
    or other static technical data about the physical machine. "Static" means
    the data does not change per-shot, or changes only at major hardware
    modification boundaries (tracked via versions).

    Generic container for any file-based technical data: geometry XMLs,
    sensor position files, calibration tables, contour data, CAD exports,
    equilibrium inputs, response function matrices.

    Example Cypher (after ingestion):
      MATCH (ds:DataSource {name: $source_name})-[:AT_FACILITY]->(f:Facility {id: $facility})
      RETURN ds
  class_uri: fc:StaticSourceConfig
  mixins:
    - DataSystemBase
  attributes:
    name:
      description: >-
        Unique identifier within this facility. Use descriptive names that
        indicate what the source provides, not internal project codes.
        Examples: "efit_geometry", "sensor_positions", "vessel_cad",
        "iron_core", "response_functions".
      required: true
    description:
      description: What this data source provides
    format:
      description: Primary data format of files in this source
      range: StaticSourceFormat
    base_dir:
      description: >-
        Root filesystem directory containing the files. File paths
        are resolved relative to this directory.
    git_repo:
      description: >-
        Path to git repository containing the files (bare or working tree).
        Mutually exclusive with base_dir for file path resolution.
    input_prefix:
      description: >-
        Path prefix within git_repo for file resolution.
        Example: "JET/input" means files are at <git_repo>/JET/input/<path>.
    reference_shot:
      description: First shot where this source's data applies
      range: integer
    files:
      description: Individual files in this source
      multivalued: true
      range: StaticFileConfig
    versions:
      description: >-
        Shot-range versions for epoch-dependent data. Each version
        maps a shot range to a configuration state. Reuses the same
        SourceVersion type used for MDSplus tree versions.
      multivalued: true
      range: SourceVersion
    systems:
      description: >-
        Physical subsystems described by this source. Reuses the same
        SourceSystem type used for MDSplus static tree systems.
        Examples: magnetics, pf_coils, vessel, limiters, iron_core.
      multivalued: true
      range: SourceSystem
```

### StaticFileConfig

A single file within a static source. Describes role, format, and validation hints — not file content.

```yaml
StaticFileConfig:
  description: >-
    A single file within a StaticSourceConfig. Describes what the file
    provides (role), its format, and optional validation hints.
    Path is relative to the parent source's base_dir or git_repo.
  class_uri: fc:StaticFileConfig
  attributes:
    path:
      description: >-
        File path relative to parent source's base_dir, or
        input_prefix within git_repo.
      required: true
    role:
      description: >-
        What physical system or data category this file describes.
        Use consistent role names across facilities:
        limiter, vessel, magnetics, pf_coils, iron_core, sensors,
        calibration_index, calibration_data, circuits, diagnostics,
        response_functions.
      required: true
    description:
      description: What this file contains
    format:
      description: >-
        File format override. If omitted, inherits from parent source.
      range: StaticSourceFormat
    expected_count:
      description: >-
        Expected element count for validation. Scanner can verify
        the parsed file produces this many elements (probes, coils,
        contour points, etc.).
      range: integer
```

### StaticSourceFormat enum

```yaml
StaticSourceFormat:
  description: Data formats for static source files
  permissible_values:
    xml:
      description: XML-structured data (device XMLs, geometry definitions)
    text:
      description: Plain text with whitespace-delimited or fixed-width columns
    csv:
      description: Comma-separated values
    netcdf:
      description: NetCDF scientific data format
    hdf5:
      description: HDF5 hierarchical data format
    json:
      description: JSON structured data
    fortran_namelist:
      description: Fortran namelist format (&name ... /)
```

### StaticSignalRef

For signals in existing data systems (PPF, MDSplus, IMAS) that contain static machine description data. Added to the `DataSystemBase` mixin so any data system config can annotate its signals.

```yaml
StaticSignalRef:
  description: >-
    A reference to a signal within a data system that contains static
    (non-time-varying) machine description data. Used to annotate
    signals in PPF, MDSplus, IMAS, or any other data system as
    containing geometry, calibration, or other static technical data.

    This enables scanners to cross-reference file-based geometry with
    data-system-accessible signals — the same limiter contour may exist
    as both a text file and a PPF signal.
  class_uri: fc:StaticSignalRef
  attributes:
    name:
      description: >-
        Signal identifier in the data system's native format.
        PPF: "DDA/DTYPE" (e.g., "EFIT/RLIM").
        MDSplus: node path (e.g., "\static::top.v:r").
        IMAS: IDS path (e.g., "wall.description_2d[0].limiter.unit[0].outline.r").
      required: true
    description:
      description: What this signal contains
    static:
      description: >-
        Whether truly static (identical across all shots) or quasi-static
        (changes only at major hardware boundaries).
      range: boolean
    expected_shape:
      description: Expected array shape for validation
      multivalued: true
      range: integer
```

### DataSystemBase extension

Add `static_signals` to the existing mixin:

```yaml
DataSystemBase:
  attributes:
    # ... existing setup_commands, python_command ...
    static_signals:
      description: >-
        Signals from this data system that contain static machine
        description data (geometry, calibration, diagnostic positions).
        Enables scanners to discover which signals are non-time-varying
        without hardcoding signal lists.
      multivalued: true
      range: StaticSignalRef
```

### DataSystemsConfig extension

Add one generic list — no facility-specific keys:

```yaml
DataSystemsConfig:
  attributes:
    # ... existing tdi, mdsplus, ppf, edas, hdf5, imas, device_xml ...
    static_sources:
      description: >-
        File-based static data sources providing machine description,
        calibration, or geometry data. Each entry describes a collection
        of files — any format, any facility. Scanners iterate this list
        and dispatch based on source name and file format.
      multivalued: true
      range: StaticSourceConfig
```

### LimiterVersion extensions

Add only generic fields (not JET-specific `file_cc`/`source_cc`):

```yaml
LimiterVersion:
  attributes:
    # ... existing name, first_shot, last_shot, description, file ...
    source_dir:
      description: >-
        Filesystem directory containing the limiter contour file.
        Used when the file lives on the facility filesystem rather
        than in the device_xml git_repo.
    n_points:
      description: Expected number of R,Z contour points (for validation)
      range: integer
```

### DeviceXMLConfig extensions

Add generic filesystem fallback:

```yaml
DeviceXMLConfig:
  attributes:
    # ... existing git_repo, input_prefix, versions, systems, limiter_versions ...
    limiter_dir:
      description: >-
        Filesystem directory containing limiter contour files.
        Fallback source when files are not in the git repository.
```

---

## How JET config looks after restructure

```yaml
data_systems:
  device_xml:
    git_repo: /home/chain1/git/efit_f90.git
    input_prefix: JET/input
    limiter_dir: /home/chain1/input/efit/Limiters
    versions: [...]       # same 14 DeviceXMLVersion entries
    systems: [...]        # same SourceSystem entries
    limiter_versions:
      - name: Mk2A
        first_shot: 1
        last_shot: 44414
        file: limiter.mk2a
        source_dir: /home/chain1/input/efit/Limiters
        n_points: 108
      - name: Mk2ILW
        first_shot: 79854
        file: limiter.mk2ilw_cc
        n_points: 251
        # no source_dir → file is in git_repo

  ppf:
    # ... existing PPF config ...
    static_signals:
      - name: EFIT/RLIM
        description: Limiter R coordinates (251 points, static across ILW)
        static: true
        expected_shape: [1, 251]
      - name: EFIT/ZLIM
        description: Limiter Z coordinates
        static: true
        expected_shape: [1, 251]
      - name: VESL/CROS
        description: Vessel cross-section contour
        static: true

  static_sources:
    - name: jec2020_geometry
      description: EFIT++ equilibrium code geometry files (XML format)
      format: xml
      base_dir: /home/chain1/jec2020
      reference_shot: 79951
      files:
        - path: limiter.xml
          role: limiter
          description: ILW first wall contour at T=200°C
          expected_count: 248
        - path: magnetics.xml
          role: magnetics
          description: Magnetic probes and flux loops with dual PPF/JPF sources
          expected_count: 131
        - path: pfSystems.xml
          role: pf_coils
          description: PF coils with multi-element sub-coil geometry
          expected_count: 20
        - path: ironBoundaries3.xml
          role: iron_core
          description: Iron core boundary segments with permeabilities
          expected_count: 96

    - name: sensor_calibration
      description: Magnetics sensor positions and calibration epochs
      format: text
      base_dir: /home/MAGNW/chain1/input
      files:
        - path: PPFcfg/sensors_200c_2019-03-11.txt
          role: sensors
          description: Canonical sensor R,Z,angle positions from CATIA CAD
          expected_count: 238
        - path: magn_ep_2019-05-14/MCFG.ix
          role: calibration_index
          description: 77 calibration epoch entries (pulse 54283–93563)
          expected_count: 77
```

Compare with the previous v1 approach which had `JEC2020Config`, `MCFGConfig`, `PPFStaticGeometryConfig` — three types that only JET could ever use. Now there are zero facility-specific types and any facility can add entries to `static_sources` without touching the schema.

---

## How a hypothetical new facility would use it

```yaml
# hypothetical ASDEX-Upgrade config
data_systems:
  mdsplus:
    trees:
      - source_name: machine
        versions:
          - version: 1
            first_shot: 1
            description: Original vessel
        systems:
          - symbol: PF
            name: Poloidal field coils
            size: 16
  static_sources:
    - name: vessel_geometry
      description: Vessel cross-section from CAD export
      format: csv
      base_dir: /data/geometry
      files:
        - path: vessel_outline_2023.csv
          role: vessel
          description: Full vessel R,Z contour
          expected_count: 500
        - path: divertor_tiles.csv
          role: limiter
          description: Divertor tile positions
```

No schema changes required.

---

## What stays the same

- **`DeviceXMLConfig`** — keep as-is (working scanner depends on it). Add only `limiter_dir`.
- **`TDIConfig`, `MDSplusConfig`, `PPFConfig`, `EDASConfig`, `IMASConfig`** — unchanged.
- **`SourceConfig`, `SourceVersion`, `SourceSystem`** — unchanged, reused by `StaticSourceConfig`.
- **`DataSystemsConfig` → scanner dispatch** — preserved. `static_sources` adds a new dispatch target.
- **Private/public YAML split** — preserved.

---

## Notes curation

Orthogonal to the schema design but equally important for config readability.

### Pattern

- `data_access_patterns.notes`: ≤10 concise, one-line notes per facility
- `exploration_notes` (private YAML): timestamped prose from exploration sessions
- Nothing is deleted — long notes relocate to private YAML

### JET (43 → ≤10 notes)

Keep:
```yaml
notes:
  - "PPF via ppfget/ppfdata (libppf.so ctypes), getdat for JPF raw signals"
  - "MDSplus thin-client at mdsplus.jet.uk — TDI functions only, no trees"
  - "1448 TDI functions, 287 JET-specific across jpf/ppf/cmg/dgm/flush"
  - "Machine description in EFIT device XMLs + chain1 files, not MDSplus"
  - "SAL REST at sal.jet.uk — not responding as of exploration (2026)"
  - "IMAS modules exist but import fails on current Python (dead end)"
```

Move to `jet_private.yaml` `exploration_notes`: all 43 verbose entries with `[2026-03-07]` timestamps.

### Other facilities

Same pattern: curate notes to ≤10 concise entries, move verbose prose to private YAML.

---

## Implementation Plan

### Phase 1: Schema — add generic types

**File**: `imas_codex/schemas/facility_config.yaml`

1. Add `StaticSourceFormat` enum
2. Add `StaticFileConfig` class
3. Add `StaticSourceConfig` class (mixes in `DataSystemBase`, references `SourceVersion`, `SourceSystem`)
4. Add `StaticSignalRef` class
5. Add `static_signals` to `DataSystemBase` mixin
6. Add `static_sources` to `DataSystemsConfig`
7. Extend `LimiterVersion` with `source_dir`, `n_points`
8. Extend `DeviceXMLConfig` with `limiter_dir`

Regenerate: `uv run build-models --force`

### Phase 2: Restructure JET config

**Files**: `jet.yaml`, `jet_private.yaml`

1. Replace `data_systems.jec2020` (untyped dict) → `data_systems.static_sources[0]` (StaticSourceConfig)
2. Replace `data_systems.mcfg` (untyped dict) → `data_systems.static_sources[1]` (StaticSourceConfig)
3. Replace `data_systems.ppf_static_geometry` (untyped dict) → `data_systems.ppf.static_signals[]` (StaticSignalRef)
4. Add `limiter_dir` and `source_dir` to DeviceXMLConfig and LimiterVersion entries
5. Curate notes (43 → ≤10), move prose to `jet_private.yaml` `exploration_notes`

### Phase 3: Validate and test

1. `uv run build-models --force` (regenerate)
2. Validate all 4 configs against schema
3. Run existing test suite — no scanner code changes

### Phase 4: Curate other facility configs

Minor notes cleanup for tcv, iter, jt-60sa. No structural changes needed.

---

## Line Count Targets

| File | Before | After | Delta |
|------|--------|-------|-------|
| jet.yaml | 748 | ~350 | -398 (notes → private, untyped dicts → typed entries) |
| facility_config.yaml | ~1100 | ~1230 | +130 (4 generic types + extensions) |
| Other facility configs | — | ~same | minor notes cleanup |

---

## Relationship to JET Machine Description Ingestion Plan

This plan is a **prerequisite**:

- **Phase 1** (limiter coverage) → uses `LimiterVersion.source_dir`, `DeviceXMLConfig.limiter_dir`
- **Phase 7** (JEC2020 ingestion) → scanner reads `static_sources` entry with `format: xml`
- **Phase 8** (MCFG sensors) → scanner reads `static_sources` entry with `format: text`
- **Phase 9** (PPF static geometry) → scanner reads `ppf.static_signals[]`

Scanners iterate `static_sources` and dispatch based on `format` and `role` — no hardcoded facility knowledge.

---

## Migration Checklist

- [ ] Add `StaticSourceFormat` enum to schema
- [ ] Add `StaticFileConfig` class to schema
- [ ] Add `StaticSourceConfig` class to schema
- [ ] Add `StaticSignalRef` class to schema
- [ ] Add `static_signals` to `DataSystemBase` mixin
- [ ] Add `static_sources` to `DataSystemsConfig`
- [ ] Add `source_dir`, `n_points` to `LimiterVersion`
- [ ] Add `limiter_dir` to `DeviceXMLConfig`
- [ ] `uv run build-models --force`
- [ ] Restructure jet.yaml: jec2020/mcfg → static_sources, ppf_static → ppf.static_signals
- [ ] Curate jet.yaml notes (43 → ≤10), move to jet_private.yaml
- [ ] Validate all 4 configs
- [ ] Run test suite
- [ ] Curate other facility config notes
- [ ] Commit and push
