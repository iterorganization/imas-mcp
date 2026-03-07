# Facility Config Schema Restructure

**Goal**: Replace the current ad-hoc, data-dump-style facility YAML configs with a clean, schema-typed structure where every field that feeds a scanner or CLI pipeline is defined in the LinkML schema, validated by Pydantic, and accessed via typed accessors. Exploration notes and raw findings move to their proper homes (`exploration_notes` in private YAML, or inline YAML comments) instead of polluting structured config with unstructured prose.

**Scope**: All 4 facility configs (tcv, jet, iter, jt-60sa), the `facility_config.yaml` LinkML schema, generated models, scanner code that consumes config, and the CLI pipelines.

**Priority**: This work **precedes** the JET machine description ingestion plan. The ingestion plan depends on JEC2020, MCFG, and PPF config sections that are currently untyped blobs — they must become schema-typed `DataSystemsConfig` entries before scanner code can consume them reliably.

---

## Problem Analysis

### Current State

The facility YAML files serve **three conflicting purposes**:

1. **Scanner configuration** — structured data that feeds CLI pipelines (data_systems, wiki_sites, discovery_roots)
2. **Exploration journal** — free-text findings from SSH exploration dumped into `data_access_patterns.notes` and inline comments
3. **Reference documentation** — long prose descriptions explaining data provenance chains, system architecture, historical context

This mixing creates several concrete problems:

#### P1: Untyped scanner inputs

JET's `jet.yaml` has grown to 748 lines. Of these:
- `data_systems.device_xml` (14 versions, 5 limiters, systems) — **well-typed** via `DeviceXMLConfig` in the schema
- `data_systems.jec2020` — **completely untyped**, a free-form dict dumped during exploration
- `data_systems.mcfg` — **completely untyped**, same problem
- `data_systems.ppf_static_geometry` — **completely untyped**, same problem
- `data_access_patterns.notes` — **43 multi-line YAML strings** containing a mix of operational knowledge, MDSplus discovery results, machine description architecture, git repo details, calibration chains, and IMAS dead-ends

A scanner trying to consume `jec2020` config must use unvalidated `dict.get()` calls with no schema backing — fragile, undocumented, and invisible to agents reading the schema.

#### P2: Notes as config

The `data_access_patterns.notes` list in jet.yaml is 43 entries long, each a multi-paragraph YAML literal block. This serves as an exploration journal, but it:
- Is loaded into memory on every `get_facility('jet')` call
- Is passed to scanners that don't need it
- Makes the config file unreadable for its primary purpose (scanner configuration)
- Has no structure — a future agent must read all 43 notes to find relevant information

TCV and ITER have smaller versions of the same problem (4-8 notes each).

#### P3: Inconsistent config ↔ scanner contract

Each scanner accesses config differently:
- `TDIScanner.scan(config=...)` receives `data_systems.tdi` as a dict
- `MDSplusScanner.scan(config=...)` receives `data_systems.mdsplus` as a dict
- `DeviceXMLScanner.scan(config=...)` receives `data_systems.device_xml` as a dict
- `PPFScanner.scan(config=...)` (not yet implemented) would receive `data_systems.ppf` as a dict

But `DeviceXMLScanner` also needs `data_systems.device_xml.limiter_versions`, `data_systems.jec2020`, `data_systems.mcfg` — which aren't part of the `DeviceXMLConfig` schema type. There's no contract saying what a scanner can expect.

#### P4: JET-specific types leaked into schema

`DeviceXMLConfig`, `DeviceXMLVersion`, `LimiterVersion` are JET-specific types in the facility-agnostic schema. No other facility has device XMLs or limiter contours in this format. If another facility needs machine description geometry, it will likely come from a different source (MDSplus static tree for TCV, IMAS IDS for ITER).

The schema should model **data source patterns** generically, with facility-specific detailed configuration living in the YAML beyond what the schema validates.

### What Works Well

Before refactoring, it's important to note what the current design gets right:

1. **DataSystemsConfig → scanner dispatch** — `data_systems` keys map to scanner `scanner_type` attributes. `get_scanners_for_facility()` iterates `data_systems` and dispatches. This is clean.

2. **Type-safe scanner configs** — `TDIConfig`, `MDSplusConfig`, `PPFConfig`, `EDASConfig` are well-defined in LinkML with proper attribute types. The generated Pydantic models enforce constraints.

3. **SourceConfig unification** — The recent `SourceConfig` type handles both versioned (static) and shot-scoped (dynamic) trees with a single model. This pattern should extend.

4. **Private/public split** — Private YAML for infrastructure, public for scanner config. Clean separation.

5. **Wiki sites** — `WikiSiteConfig` is well-typed and used consistently across all facilities.

---

## Design Principles

1. **Every field consumed by a scanner or CLI pipeline must be schema-typed** — if code does `config.get("some_key")`, that key must exist in the LinkML schema.

2. **Exploration findings go in `exploration_notes`** (private YAML) — free-text knowledge about a facility belongs in timestamped notes, not in structured config sections.

3. **`data_access_patterns.notes` is capped and curated** — maximum 10 concise notes per facility. Long prose moves to exploration_notes.

4. **Each `data_systems` entry is a typed config class** — every key under `data_systems:` corresponds to a `DataSystemBase`-derived class in the schema. If no typed class exists, the key is invalid.

5. **Scanner config is self-contained** — a scanner's `config` argument contains everything it needs. It should not reach into sibling `data_systems` entries or `data_access_patterns`.

6. **Generic before specific** — prefer composable schema primitives (version lists, file references, coordinate systems) over facility-specific types.

---

## Proposed Changes

### Change 1: Add typed schema classes for JEC2020, MCFG, PPF static geometry

These are currently untyped dicts in jet.yaml. They must become proper `DataSystemBase`-derived classes in the LinkML schema.

#### JEC2020Config

```yaml
JEC2020Config:
  description: >-
    EFIT++ equilibrium code input XML files. Next-generation geometry files
    providing richer metadata than legacy EFIT device XMLs, including dual
    PPF/JPF data source references per probe, multi-element PF coil geometry,
    and iron core boundaries. Facility-specific to JET.
  class_uri: fc:JEC2020Config
  mixins:
    - DataSystemBase
  attributes:
    base_dir:
      description: Directory containing JEC2020 XML files
      required: true
    files:
      description: Map of file role to file config
      range: JEC2020FileConfig
      multivalued: true
    reference_shot:
      description: First shot where JEC2020 geometry applies
      range: integer

JEC2020FileConfig:
  description: A single JEC2020 XML file definition
  class_uri: fc:JEC2020FileConfig
  attributes:
    role:
      description: File role (limiter, magnetics, pf_systems, iron_boundaries)
      required: true
    path:
      description: Filename relative to base_dir
      required: true
    description:
      description: What this file contains
    element_counts:
      description: >-
        Named element counts for validation (e.g., probe_count: 95,
        flux_loop_count: 36). Scanner can verify expected counts.
```

#### MCFGConfig

```yaml
MCFGConfig:
  description: >-
    Magnetics calibration and sensor configuration system. Tracks sensor
    positions (from CATIA CAD models), calibration gains/offsets, and
    per-pulse configuration changes. Facility-specific to JET.
  class_uri: fc:MCFGConfig
  mixins:
    - DataSystemBase
  attributes:
    base_dir:
      description: Base directory for MCFG configuration files
      required: true
    sensor_file:
      description: Path to canonical sensor position file
      required: true
    sensor_reference:
      description: CATIA drawing reference for sensor positions
    calibration_index:
      description: Path to MCFG.ix calibration epoch index file
    calibration_dir:
      description: Directory containing epoch data files
    calibration_files:
      description: List of calibration data file names
      multivalued: true
```

#### PPFStaticGeometryConfig

```yaml
PPFStaticGeometryConfig:
  description: >-
    PPF signals containing static machine description geometry.
    Some PPF DDAs store limiter contours, vessel cross-sections,
    and diagnostic positions that are constant across shots (or
    change only at major hardware changes). These provide an
    alternative access method to geometry also available in device XMLs.
  class_uri: fc:PPFStaticGeometryConfig
  mixins:
    - DataSystemBase
  attributes:
    access_method:
      description: How to access these signals (thin_client, ppfget, sal)
    signals:
      description: Static geometry signal definitions
      multivalued: true
      range: PPFStaticSignal

PPFStaticSignal:
  description: A single PPF signal containing static geometry data
  class_uri: fc:PPFStaticSignal
  attributes:
    dda:
      description: PPF Diagnostic Data Area (e.g., EFIT, VESL, LIDR)
      required: true
    dtype:
      description: PPF data type within the DDA (e.g., RLIM, ZLIM, CROS)
      required: true
    description:
      description: What this signal contains
    shape:
      description: Expected array shape (list of ints)
      multivalued: true
      range: integer
    static:
      description: Whether this signal is truly static (same across all shots)
      range: boolean
```

### Change 2: Register new data system types in DataSystemsConfig

```yaml
DataSystemsConfig:
  attributes:
    # ... existing attributes ...
    jec2020:
      description: JEC2020 EFIT++ XML geometry files
      range: JEC2020Config
    mcfg:
      description: MCFG sensor calibration and configuration
      range: MCFGConfig
    ppf_static_geometry:
      description: PPF signals with static machine description data
      range: PPFStaticGeometryConfig
```

This means `data_systems.jec2020` will be validated against `JEC2020Config` by the generated Pydantic model. A scanner with `scanner_type = "jec2020"` can consume it type-safely.

### Change 3: Restructure LimiterVersion with source tracking

The current `LimiterVersion` class lacks `source` and `file_cc` fields that are used in jet.yaml but not in the schema. Add them:

```yaml
LimiterVersion:
  attributes:
    name:
      required: true
    first_shot:
      range: integer
    last_shot:
      range: integer
    description: ...
    file:
      description: Primary R,Z contour file (relative to limiter dir)
    file_cc:
      description: Coordinate-corrected version of the file (preferred)
    source:
      description: Where the primary file lives (chain1, git)
    source_cc:
      description: Where the cc file lives (chain1, git)
    n_points:
      description: Number of primary contour points
      range: integer
```

### Change 4: Curate data_access_patterns.notes

Move long exploration prose from `data_access_patterns.notes` to `exploration_notes` (private YAML). Keep only concise, actionable notes in `data_access_patterns.notes`:

**Before** (jet.yaml, 43 notes):
```yaml
notes:
  - "PPF Python API at /jet/share/DEPOT/pyppf/21260/lib/python/ppf.py..."
  - >-
    MDSPLUS THIN-CLIENT: JET MDSplus server at mdsplus.jet.uk provides...
    (10 lines of prose)
  - >-
    MACHINE DESCRIPTION OVERVIEW: No single MDSplus machine description tree...
    (15 lines of prose)
  # ... 40 more entries
```

**After** (jet.yaml, ≤10 notes):
```yaml
notes:
  - "PPF via ppfget/ppfdata (libppf.so ctypes), getdat for JPF raw signals"
  - "MDSplus thin-client at mdsplus.jet.uk — TDI functions only, no trees"
  - "1448 TDI functions, 287 JET-specific across jpf/ppf/cmg/dgm/flush"
  - "Machine description in EFIT device XMLs + chain1 files, not MDSplus"
  - "SAL REST at sal.jet.uk — not responding as of exploration (2026)"
  - "IMAS modules exist but import fails on current Python (dead end)"
```

The long prose entries move to `exploration_notes` in `jet_private.yaml` with timestamps.

### Change 5: Add DeviceXMLConfig.chain1_limiter_dir to schema

This field exists in jet.yaml but not in the schema:

```yaml
DeviceXMLConfig:
  attributes:
    # ... existing ...
    chain1_limiter_dir:
      description: >-
        Filesystem directory containing frozen limiter contour files.
        Used when limiter source is "chain1" rather than "git".
    chain1_limiter_list:
      description: >-
        Path to authoritative shot-to-limiter-file mapping.
        Format: "first_shot-last_shot → filename" per line.
```

### Change 6: Scanner config isolation via typed accessors

Instead of scanners receiving raw dicts, provide typed access:

```python
# In scanner base or a config accessor module:
def get_scanner_config(facility: str, scanner_type: str) -> BaseModel:
    """Get typed config for a specific scanner.
    
    Returns the Pydantic model for data_systems.<scanner_type>,
    validated against the schema.
    """
    from imas_codex.config.models import DataSystemsConfig
    config = get_facility(facility)
    ds = config.get("data_systems", {})
    raw = ds.get(scanner_type, {})
    # The DataSystemsConfig model knows the type for each key
    model_cls = DataSystemsConfig.model_fields[scanner_type].annotation
    return model_cls.model_validate(raw)
```

This is aspirational — the immediate change is ensuring all config keys are schema-typed. Full typed accessor can follow.

---

## Implementation Plan

### Phase 1: Schema — Add new typed classes

**Files to modify**:
- `imas_codex/schemas/facility_config.yaml`

**Changes**:
1. Add `JEC2020Config`, `JEC2020FileConfig` classes
2. Add `MCFGConfig` class
3. Add `PPFStaticGeometryConfig`, `PPFStaticSignal` classes
4. Add `jec2020`, `mcfg`, `ppf_static_geometry` attributes to `DataSystemsConfig`
5. Add `source`, `source_cc`, `file_cc`, `n_points` to `LimiterVersion`
6. Add `chain1_limiter_dir`, `chain1_limiter_list` to `DeviceXMLConfig`
7. Add `DataSystemType` enum values: `jec2020`, `mcfg`, `ppf_static`

**Regenerate**: `uv run build-models --force`

**Validation**: `uv run python -c "from imas_codex.config.models import FacilityConfig; print('OK')"`

### Phase 2: Curate jet.yaml — separate notes from config

**Files to modify**:
- `imas_codex/config/facilities/jet.yaml`
- `imas_codex/config/facilities/jet_private.yaml`

**Changes**:
1. Move 43 `data_access_patterns.notes` entries to `exploration_notes` in `jet_private.yaml`, each with a `[2026-03-07]` timestamp prefix
2. Replace with ≤10 concise, actionable notes in `jet.yaml`
3. Move file-level prose comments from `data_systems.jec2020`, `data_systems.mcfg`, `data_systems.ppf_static_geometry` into their schema-typed `description` fields
4. Remove redundant inline comments where the schema `description` is sufficient
5. Ensure all `data_systems.*` entries conform to their new schema types

**What stays in jet.yaml**:
- All `data_systems` entries (device_xml, jec2020, mcfg, ppf_static_geometry, ppf, mdsplus, imas)
- `data_access_patterns` with curated notes (≤10)
- `wiki_sites`, `discovery_roots`, `user_info`
- Structural YAML comments where they add value

**What moves to jet_private.yaml**:
- All 43 verbose exploration notes → `exploration_notes` list
- Any infrastructure-specific details currently in public YAML comments

### Phase 3: Curate all other facility configs

**Files to modify**:
- `imas_codex/config/facilities/tcv.yaml` — minor cleanup, notes already reasonable
- `imas_codex/config/facilities/iter.yaml` — move notes to private
- `imas_codex/config/facilities/jt-60sa.yaml` — move notes to private

**Pattern**: Same as jet.yaml — move verbose notes to `exploration_notes`, keep ≤10 concise notes.

### Phase 4: Validate all configs against updated schema

```bash
uv run python -c "
from imas_codex.discovery.base.facility import validate_facility_config
for f in ['tcv', 'jet', 'iter', 'jt-60sa']:
    errors = validate_facility_config(f)
    print(f'{f}: {len(errors)} errors')
    for e in errors[:5]:
        print(f'  {e}')
"
```

Fix any validation errors.

### Phase 5: Update build_models.py for new types

The `build_models.py` script has type fixes that post-process generated code. Check if new types need additional fixes (e.g., `data_systems: Optional[str]` → `Optional[DataSystemsConfig]` is already handled).

### Phase 6: Tests

1. **Schema validation test**: All 4 facility configs validate against schema
2. **Config loading test**: `get_facility()` returns correct typed sections
3. **Scanner config access**: Each scanner can access its config section
4. **Regression**: Existing signals pipeline tests pass unchanged

```bash
uv run pytest tests/ -k "facility or config or scanner"
```

---

## Line Count Targets

| File | Before | After | Delta |
|------|--------|-------|-------|
| jet.yaml | 748 | ~350 | -398 (notes moved to private) |
| tcv.yaml | 287 | ~260 | -27 |
| iter.yaml | 123 | ~90 | -33 |
| jt-60sa.yaml | 249 | ~230 | -19 |
| facility_config.yaml (schema) | ~1100 | ~1300 | +200 (new types) |

---

## Key Design Decisions

1. **JEC2020/MCFG/PPF as separate `data_systems` entries** — not nested under `device_xml`. They are independent data sources with different file formats, file locations, and (eventually) different scanners. The `device_xml` scanner already does its own thing; JEC2020 XML is a different XML format requiring a different parser.

2. **Keep `data_access_patterns` section** — it has value for wiki scoring and code ingestion even after note curation. The structured fields (`primary_method`, `key_tools`, `wiki_signal_patterns`, `code_import_patterns`) are consumed by scoring heuristics.

3. **No breaking schema changes to existing typed configs** — `TDIConfig`, `MDSplusConfig`, `PPFConfig`, `EDASConfig`, `DeviceXMLConfig` keep their current attributes. We only add new attributes and new config classes.

4. **Notes curation is content-preserving** — nothing is deleted. Long notes move to `exploration_notes` in private YAML. The knowledge is preserved, just relocated to where it belongs.

5. **LimiterVersion extended, not replaced** — add `source`, `file_cc`, `source_cc`, `n_points` as optional fields. Existing configs continue to validate.

6. **No scanner code changes in this plan** — scanner code that consumes the new typed configs is part of the jet machine description ingestion plan. This plan establishes the schema foundation only.

---

## Relationship to JET Machine Description Ingestion Plan

This plan is a **prerequisite** for the jet machine description ingestion plan:

- **Phase 7** (JEC2020 XML ingestion) requires `JEC2020Config` to be schema-typed
- **Phase 8** (MCFG sensor calibration) requires `MCFGConfig` to be schema-typed
- **Phase 9** (PPF static geometry) requires `PPFStaticGeometryConfig` to be schema-typed
- **Phase 1** (complete limiter coverage) requires `LimiterVersion.source/file_cc` in schema

After this plan is complete, the ingestion plan scanners can:
1. Receive typed config objects instead of raw dicts
2. Have their expected inputs documented in the schema
3. Be validated at startup (missing required fields → clear error)
4. Auto-generate API documentation from LinkML descriptions

The jet machine description ingestion plan should reference this plan as a dependency and update its Phase 1+ descriptions to reference the schema-typed config classes.

---

## Migration Checklist

- [ ] Add new LinkML classes to `facility_config.yaml`
- [ ] Run `uv run build-models --force` to regenerate Pydantic models
- [ ] Verify generated `models.py` includes new types
- [ ] Curate jet.yaml notes → jet_private.yaml exploration_notes
- [ ] Restructure jet.yaml `data_systems.jec2020` to match `JEC2020Config`
- [ ] Restructure jet.yaml `data_systems.mcfg` to match `MCFGConfig`
- [ ] Restructure jet.yaml `data_systems.ppf_static_geometry` to match `PPFStaticGeometryConfig`
- [ ] Add `source`, `file_cc`, `source_cc`, `n_points` to jet.yaml limiter versions (already done)
- [ ] Add `chain1_limiter_dir`, `chain1_limiter_list` to jet.yaml (already done)
- [ ] Curate tcv.yaml, iter.yaml, jt-60sa.yaml notes
- [ ] Validate all 4 configs: `validate_facility_config()`
- [ ] Run full test suite: `uv run pytest`
- [ ] Commit and push
