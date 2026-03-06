# JET Machine Description Cataloging Plan

**Status:** Ready for implementation (pending naming generalization)
**Prerequisite:** [naming-generalization.md](naming-generalization.md) must complete first
**Author:** Agent (on behalf of user)
**Date:** 2026-03-06

## Objective

Catalog JET's machine description geometry into the knowledge graph as `DataSource`, `StructuralEpoch`, `DataNode`, and `FacilitySignal` nodes. JET has no MDSplus static tree — geometry lives in device XMLs, Fortran namelists, and R,Z contour files versioned in a git repo.

**Current state:** 0 JET signals, 0 StructuralEpoch, 0 DataSource for machine description.
**Target:** ~330 DataNode per epoch, ~500 FacilitySignal, 10 StructuralEpoch, 1 DataSource.

## Dependency: Naming Generalization

This plan uses the renamed graph schema from [naming-generalization.md](naming-generalization.md):

| Old Name | New Name | Used In This Plan |
|---|---|---|
| `MDSplusTree` | `DataSource` | Parent container for machine description |
| `TreeModelVersion` | `StructuralEpoch` | One per PulseDependencyDB configuration |
| `TreeNode` | `DataNode` | One per geometry element (coil, probe, loop) |
| `TreeNodePattern` | `DataNodePattern` | Indexed repetition (BPME(1)..BPME(191)) |
| `IN_TREE` | `IN_DATA_SOURCE` | DataNode → DataSource relationship |
| `tree_name` | `data_source_name` | Property on DataNode, StructuralEpoch, FacilitySignal |
| `node_path` | `data_source_path` | Full path within the data source |
| `source_node` | `data_source_node` | FacilitySignal → DataNode provenance |

**Implementation cannot begin until the naming generalization is merged.** The scanner plugin will import from the renamed model classes and create nodes using the new labels.

## Data Sources — SSH-Verified Inventory

All paths verified via SSH on 2026-03-06. Access confirmed for all critical files.

### Canonical Source: `/home/chain1/git/efit_f90.git`

Bare git repo, 108 tags (Lx099-0 through Lx103-77), single master branch, HEAD commit `50a3e3b` (2018-01-26). No remote configured — the original SVN server is unreachable. This is the surviving authoritative VCS.

**Access method:** `git show HEAD:<path>` from the bare repo — no checkout needed. Python `xml.etree.ElementTree` confirmed working on JET (system Python 3.x, no extra packages required).

### Device XML Structure (SSH-verified)

Each XML file contains a `<device>` root with these sections:

| Section | Instances | Fields per Instance | IMAS Mapping |
|---|---|---|---|
| `magprobes` | 191 | r, z, angle, abs_error, rel_error | `magnetics.bpol_probe` |
| `flux` | 36 | r, z, dphi, abs_error, rel_error | `magnetics.flux_loop` |
| `pfcoils` | 22 | r, z, dr, dz, turnsperelement, abs_error, rel_error | `pf_active.coil` |
| `pfcircuits` | 16 | coil_connect, supply_connect | `pf_active.circuit` |
| `pfsupplies` | 16 | abs_error, rel_error | `pf_active.supply` |
| `toroidalfield` | 1 | abs_error, rel_error | `tf.coil` |
| `plasmacurrent` | 1 | abs_error, rel_error | `magnetics.ip` |
| `diamagneticflux` | 1 | abs_error, rel_error | `magnetics.diamagnetic_flux` |
| `pfpassive` | 48 | r, z, dr, dz, ang1, ang2, resistance, abs_error, rel_error | `pf_passive.loop` |

**Total: 332 instances per XML file** (consistent across all 4 distinct XML files).

Instance attributes: Each `<instance>` has `id`, `file`, `signal`, `archive`, `owner`, `status`, `seq` as XML attributes. These attributes link each geometry element to its PPF data source (e.g., `file="MAGN" signal="BPME(1)"`).

### Distinct XML Files (SSH-verified)

Three of the five "named" XML files are symlinks:

| File | Type | Notes |
|---|---|---|
| `device_p68613.xml` | **symlink** → `device_p72258.xml` | Same geometry as p72258 |
| `device_p68613_efcc1.xml` | **symlink** → `device_p72258.xml` | EFCC mode, same geometry |
| `device_p68613_efcc2.xml` | **symlink** → `device_p72258.xml` | EFCC mode, same geometry |
| `device_p72258.xml` | **real file** | Baseline 95-probe config |
| `device_p72258_bvch.xml` | **real file** | BVCH variant |
| `device_p87722.xml` | **real file** | P803B→P803A: angle change on instance ~40 |
| `device_p87722_bvch.xml` | **real file** | BVCH variant |
| `device_p88368.xml` | **real file** | UP4E4 coords for BPME(91) |
| `device_p88368_bvch.xml` | **real file** | BVCH variant |
| `device_p89440.xml` | **real file** | P802B→P802A |
| `device_p89440_bvch.xml` | **real file** | BVCH variant |

**Inter-version diffs are minimal:** typically 1-3 probe coordinate changes (mm-scale R,Z or degree-scale angle). Instance counts are always 332. The XML schema itself is stable.

**BVCH variants:** Same instance count (332) as base files. Used for boundary-value-corrected-Hessian EFIT runs. Should be tracked as DataNode properties (bvch=true) rather than separate epochs.

### PulseDependencyDB (SSH-verified)

`JET/input/PulseDependencyDB/EFIT.db` defines pulse ranges mapping to named configurations. `configurationsMAGN` (included by EFIT.db) defines 16 named ranges, each bundling:

| Input File | Purpose | Example Path |
|---|---|---|
| `efitsnap_current` | Active probe enable/disable mask | `Snap_files/EFITSNAP/efitsnap_p90626_bound0` |
| `device` | PPF DDA routing namelist | `Devices/MAGN/device_ppfs` |
| `device.xml` | Geometry (coils, probes, structures) | `Devices/device_p89440.xml` |
| `limiter` | First wall R,Z contour | `Limiters/limiter.mk2ilw_cc` |
| `Greens` | Response function matrices | `Greens/DMSS_105_T_200C_P802A/` |
| `PPF_headers` | Output DDA/Dtype definitions | `PPF_headers/p79854/` |

**Key insight from configurationsMAGN:** Many named ranges share the same `device.xml` but differ in snap files. This means the *geometry* epochs (which XML file is used) are fewer than the *operational* epochs (which probes are enabled). For graph cataloging we track geometry epochs — the XML file determines what geometry elements exist.

### Geometry Epoch Mapping (from configurationsMAGN, SSH-verified)

| Config Name | Device XML | Limiter | Greens | What Changed |
|---|---|---|---|---|
| p68613 | device_p72258.xml (via symlink) | mk2hd_cc | DMSS_091 | Baseline 71 probes |
| p74387 | device_p72258.xml (via symlink) | mk2hd_cc | DMSS_091 | FLME(3) out of order (transient) |
| p78168 | device_p72258.xml (via symlink) | mk2hd_cc | DMSS_091 | P801B broken (snap change only) |
| p78359 | device_p72258.xml (via symlink) | mk2hd_cc | DMSS_105 | New DMSS config (snap/Greens change) |
| p78461 | device_p72258.xml (via symlink) | mk2hd_cc | DMSS_108 | DMSS=108 trial (Greens change only) |
| p79854 | device_p72258.xml (via symlink) | **mk2ilw_cc** | DMSS_105 | **ILW wall installed** |
| p81134 | device_p72258.xml (via symlink) | mk2ilw_cc | DMSS_105 | P804B reinstated (snap change) |
| p87027 | device_p72258.xml (via symlink) | mk2ilw_cc | DMSS_105 | P804B/P805B broken (snap change) |
| p87722 | **device_p87722.xml** | mk2ilw_cc | DMSS_105_P803A | **P803B→P803A probe replacement** |
| p88368 | **device_p88368.xml** | mk2ilw_cc | DMSS_105_UP4E4 | **BPME(91) coord change** |
| p89431 | device_p88368.xml | mk2ilw_cc | DMSS_105_UP4E4 | BPME(42) disabled (snap only) |
| p89440 | **device_p89440.xml** | mk2ilw_cc | DMSS_105_P802A | **P802B→P802A replacement** |
| p90540 | device_p89440.xml | mk2ilw_cc | DMSS_105_P802A | I803 failed (snap only) |
| p90626 | device_p89440.xml | mk2ilw_cc | DMSS_105_P802A | I802 failed (snap only) — **FINAL** |

**Conclusion:** Only 4 distinct device XMLs exist (p72258, p87722, p88368, p89440). The other 10+ PulseDependencyDB configurations differ only in snap files (probe enable/disable masks) or Greens functions. For graph cataloging, we create StructuralEpoch per config (10 in jet.yaml) but DataNodes reference just 4 geometry sets.

### Device PPFs Namelist (SSH-verified)

Fortran namelist (`&ppfin_namelist`) mapping PPF DDAs to data sources:

```fortran
bvac_r  → MAGN   ! toroidal field * R_centre
ip      → MAGN   ! plasma current
dflux   → MAGN   ! diamagnetic flux
flux    → MAGN   ! flux and saddle loops
magn    → MAGN   ! magnetic probes (BPME)
```

12 variants exist (4 DDAs × 3 EFCC modes): `device_ppfs`, `device_ppfs_efcc1`, `device_ppfs_efcc2`.

### Limiter Contour Files (SSH-verified)

Plain text R,Z pairs (one per line). Line counts:

| File | Lines | Points | Wall Era |
|---|---|---|---|
| `limiter.mk2gb_ns_cc` | 221 | 220 | Mk2GB (Mark 2 Gas Box) |
| `limiter.mk2hd_cc` | 117 | 116 | Mk2HD (Mark 2 High Delta) |
| `limiter.mk2ilw_cc` | 252 | 251 | Mk2ILW (ITER-Like Wall) |

Three limiter files exist in git HEAD. The jet.yaml `limiter_versions` config tracks 7 historical wall configurations (L91NEW through Mk2ILW), but only 3 have surviving contour files in this repo.

## Architecture: MachineDescriptionScanner Plugin

### Scanner Design

A new `MachineDescriptionScanner` in `imas_codex/discovery/signals/scanners/machine_description.py` implementing the `DataSourceScanner` protocol:

```
scanner_type = "machine_description"
```

Registered via `register_scanner()` and imported in `_auto_register()`.

### Configuration (jet.yaml — already in place)

The `data_sources.machine_description` section in jet.yaml provides all required config:

```yaml
data_sources:
  machine_description:
    source_format: git_xml
    git_repo: /home/chain1/git/efit_f90.git
    input_prefix: JET/input
    versions:
      - version: p68613
        first_shot: 68613
        last_shot: 74386
        device_xml: Devices/device_p68613.xml
        # ... (10 versions total)
    systems:
      - symbol: PF
        name: Poloidal field coils
        size: 22
        parameters: [R, Z, W, H, TURNS, RESISTANCE]
      # ... (6 systems total)
    limiter_versions:
      - name: Mk2ILW
        first_shot: 79854
        file: Limiters/limiter.mk2ilw_cc
      # ... (7 versions total)
```

**No additional jet.yaml changes needed** — the existing config is sufficient for the scanner.

### scan() Flow

```
1. Read config.git_repo, config.input_prefix, config.versions
2. SSH to JET: git show HEAD:<input_prefix>/<device_xml> for each version
3. Parse XML with xml.etree.ElementTree (stdlib, no dependencies)
4. Create graph nodes:
   a. DataSource(id="jet:machine_description", source_type="xml")
   b. StructuralEpoch per version (10 total)
   c. DataNode per geometry element per epoch
   d. FacilitySignal per unique geometry parameter
5. Create DataNodePattern for indexed elements (BPME(1)..BPME(191))
```

### Remote Execution Strategy

The scanner needs to read files from a bare git repo on JET. Two approaches:

**Option A: SSH + git show (recommended)**
- Use `async_run_python_script()` with a new `parse_device_xml.py` remote script
- Script runs on JET, does `git show HEAD:<path>` + `xml.etree.ElementTree` parsing
- Returns JSON with parsed geometry elements
- Pro: Single SSH call per version, minimal data transfer (~10KB JSON vs ~50KB XML)
- Pro: Uses existing remote execution infrastructure
- Con: Remote script must be Python 3.12+ compatible (venv interpreter)

**Option B: SSH + git show, parse locally**
- Use `run_command()` to `git show HEAD:<path>` and transfer raw XML
- Parse locally with `xml.etree.ElementTree`
- Pro: Simpler remote script (just git show)
- Con: More data transfer, more SSH calls

**Decision: Option A** — a single remote script that parses all versions in one SSH call.

### Remote Script: `parse_device_xml.py`

Location: `imas_codex/remote/scripts/parse_device_xml.py`

```python
"""Parse JET device XML files from a bare git repo.

Python 3.12+ (runs via venv interpreter).
Input: git_repo, input_prefix, versions (list of {version, device_xml})
Output: JSON with parsed geometry per version.
"""
# Uses: subprocess (git show), xml.etree.ElementTree, json
# No external dependencies.
```

Input (JSON via stdin):
```json
{
  "git_repo": "/home/chain1/git/efit_f90.git",
  "input_prefix": "JET/input",
  "versions": [
    {"version": "p72258", "device_xml": "Devices/device_p72258.xml"},
    {"version": "p87722", "device_xml": "Devices/device_p87722.xml"}
  ]
}
```

Output (JSON via stdout):
```json
{
  "versions": {
    "p72258": {
      "magprobes": [{"id": 1, "r": 4.292, "z": 0.604, "angle": -74.1, ...}, ...],
      "flux": [...],
      "pfcoils": [...],
      "pfcircuits": [...],
      "pfsupplies": [...],
      "pfpassive": [...],
      "toroidalfield": [...],
      "plasmacurrent": [...],
      "diamagneticflux": [...]
    }
  }
}
```

### Graph Node Creation

#### DataSource Node

```python
DataSource(
    id="jet:machine_description",
    facility_id="jet",
    name="JET Machine Description",
    data_source_name="machine_description",
    source_type="xml",  # New property from naming generalization
    description="JET tokamak geometry: PF coils, passive structures, magnetic probes, flux loops, circuits, and limiter contours. Versioned in /home/chain1/git/efit_f90.git.",
)
```

#### StructuralEpoch Nodes (10)

One per version in jet.yaml `machine_description.versions`:

```python
StructuralEpoch(
    id="jet:machine_description:p89440",
    facility_id="jet",
    data_source_name="machine_description",
    version="p89440",
    first_shot=89440,
    last_shot=90539,
    description="P802B→P802A, new calibration, new device XML and Green's",
    status=StructuralEpochStatus.ingested,
)
```

#### DataNode Nodes (~332 per epoch × 4 distinct geometries)

Since only 4 distinct XML files exist, DataNodes can reference the geometry set rather than duplicating. However, the graph stores DataNodes per epoch for simple traversal:

```python
DataNode(
    id="jet:machine_description:p89440:magprobes:1",
    facility_id="jet",
    data_source_name="machine_description",
    path="magprobes/1",  # Section/instance_id
    node_type=DataNodeType.SIGNAL,
    description="Magnetic probe BPME(1): R=4.292m, Z=0.604m, angle=-74.1°",
)
```

**Optimization:** For epochs sharing the same XML (e.g., p68613 through p87027 all use device_p72258.xml), DataNodes are created once and linked to all applicable StructuralEpochs via `INTRODUCED_IN`/`REMOVED_IN` relationships rather than duplicated.

#### FacilitySignal Nodes

One per unique geometry parameter. Signals span epochs — they represent the physical measurement, not a specific version's value:

```python
FacilitySignal(
    id="jet:magnetics/bpme_1_r",
    facility_id="jet",
    status=FacilitySignalStatus.discovered,
    physics_domain="magnetic_field_diagnostics",
    name="BPME(1)/R",
    accessor="device_xml:magprobes/1/r",
    data_access="jet:machine_description:xml",
    data_source_name="machine_description",
    data_source_path="magprobes/1/r",
    data_source_node="jet:machine_description:p89440:magprobes:1",
    description="Radial position of magnetic probe BPME(1)",
    unit="m",
    discovery_source="xml_extraction",
)
```

#### DataNodePattern Nodes

Indexed elements get patterns for efficient queries:

```python
DataNodePattern(
    id="jet:machine_description:magprobes:*",
    facility_id="jet",
    data_source_name="machine_description",
    pattern="magprobes/{n}",
    member_count=191,
    description="Magnetic probes BPME(1) through BPME(191)",
)
```

### Signal Count Estimate (Revised)

Based on SSH-verified instance counts and fields:

| System | Instances | Fields | Signals | Notes |
|---|---|---|---|---|
| magprobes | 191 | r, z, angle | 573 | abs_error/rel_error are metadata, not separate signals |
| flux | 36 | r, z, dphi | 108 | |
| pfcoils | 22 | r, z, dr, dz, turnsperelement | 110 | |
| pfcircuits | 16 | coil_connect, supply_connect | 32 | Topology, not numeric signals |
| pfsupplies | 16 | — | 0 | Only error bounds, no geometry |
| toroidalfield | 1 | — | 0 | Only error bounds |
| plasmacurrent | 1 | — | 0 | Only error bounds |
| diamagneticflux | 1 | — | 0 | Only error bounds |
| pfpassive | 48 | r, z, dr, dz, ang1, ang2, resistance | 336 | |
| Limiter contours | 3 | (R,Z point arrays) | 3 | One signal per contour file |

**Subtotal: ~1,162 raw per epoch, but with deduplication across shared XML files.**

**After deduplication:** Since only 4 distinct XML files exist and most signals are constant across epochs, the unique FacilitySignal count is approximately **~500** (distinct geometry parameters) rather than 1,162 × 10 epochs.

Error bounds (abs_error, rel_error) are stored as DataNode properties, not promoted to separate FacilitySignal nodes — they describe measurement uncertainty, not independent signals.

### Graph Structure Summary

```
Facility(id="jet")
  └── DataSource(id="jet:machine_description")
        ├── StructuralEpoch(id="jet:machine_description:p68613")
        │     └── DataNode(id="...:magprobes:1")  ──INTRODUCED_IN──> epoch
        │     └── DataNode(id="...:pfcoils:1")
        │     └── ...
        ├── StructuralEpoch(id="jet:machine_description:p87722")
        │     └── ... (same pattern)
        └── ...
  └── FacilitySignal(id="jet:magnetics/bpme_1_r")
        ──HAS_DATA_SOURCE_NODE──> DataNode
        ──DATA_ACCESS──> DataAccess(id="jet:machine_description:xml")
```

## Implementation Phases

### Phase 0: Naming Generalization (prerequisite)

Complete the [naming generalization plan](naming-generalization.md). This creates the `DataSource`, `StructuralEpoch`, `DataNode`, `DataNodePattern` node types that the machine description scanner depends on.

### Phase 1: Remote Script + Scanner Plugin

1. **Create `imas_codex/remote/scripts/parse_device_xml.py`**
   - Parse device XML files from bare git repo via `git show`
   - Return structured JSON with all geometry elements
   - Test locally on JET: `ssh jet "python3 < parse_device_xml.py"`

2. **Create `imas_codex/discovery/signals/scanners/machine_description.py`**
   - Implement `DataSourceScanner` protocol (`scan()`, `check()`)
   - `scan()`: SSH to JET, run parse script, create graph nodes
   - `check()`: Verify XML files still accessible (git show returns 0)

3. **Register in `_auto_register()` in `base.py`**
   - Add `machine_description` import to the auto-register function

4. **Test locally:**
   ```bash
   uv run imas-codex discover signals jet -s machine_description --scan-only
   ```

### Phase 2: Limiter Contour Integration

1. **Extend parse script** to also read limiter contour files
   - Parse R,Z pairs from `Limiters/limiter.*.cc` files
   - Return as arrays in the JSON output

2. **Create limiter DataNodes**
   - One DataNode per limiter version (from `limiter_versions` config)
   - Store R,Z arrays as node properties or as contour signal

3. **Link limiter versions to StructuralEpoch**
   - Each StructuralEpoch already specifies its limiter file in jet.yaml
   - Create `USES_LIMITER` relationship or store as property

### Phase 3: Enrichment + IMAS Mapping

1. **LLM enrichment** via existing enrich pipeline
   - Physics domain assignment (most will be `magnetic_field_diagnostics`)
   - Description generation for signals lacking wiki context
   - Skip enrichment for signals with sufficient XML-derived descriptions

2. **IMAS mapping**
   - Map XML sections to IMAS IDS paths:
     - `magprobes` → `magnetics.bpol_probe[:].position.r/z`
     - `pfcoils` → `pf_active.coil[:].element[:].geometry.rectangle.r/z`
     - `flux` → `magnetics.flux_loop[:].position[:].r/z`
     - `pfpassive` → `pf_passive.loop[:].element[:].geometry.rectangle.r/z`
     - Limiter → `wall.description_2d[:].limiter.unit[:].outline.r/z`
   - Create `MAPS_TO_IMAS` relationships to IMASPath nodes

### Phase 4: Verification + Monitoring

1. **Verification queries:**
   ```cypher
   -- Count nodes by type
   MATCH (ds:DataSource {id: "jet:machine_description"})
   OPTIONAL MATCH (ds)<-[:IN_DATA_SOURCE]-(dn:DataNode)
   OPTIONAL MATCH (ds)<-[:IN_DATA_SOURCE]-(se:StructuralEpoch)
   RETURN count(DISTINCT se) AS epochs,
          count(DISTINCT dn) AS data_nodes

   -- Count signals
   MATCH (s:FacilitySignal {facility_id: "jet", data_source_name: "machine_description"})
   RETURN s.physics_domain, count(*) AS n
   ORDER BY n DESC

   -- Check IMAS mappings
   MATCH (s:FacilitySignal {facility_id: "jet"})-[:MAPS_TO_IMAS]->(imas:IMASPath)
   WHERE s.data_source_name = "machine_description"
   RETURN imas.ids_name, count(*) AS mapped
   ORDER BY mapped DESC
   ```

2. **CLI verification:**
   ```bash
   uv run imas-codex discover status jet -d signals
   ```

## Tractability Assessment

### Confirmed (SSH-tested 2026-03-06)

| Element | Status | Evidence |
|---|---|---|
| Git repo access | **OK** | `git log`, `git ls-tree`, `git show` all work |
| XML parsing (remote) | **OK** | `xml.etree.ElementTree` parses on JET's Python 3 |
| Device XML content | **OK** | All 4 distinct XML files readable, 332 instances each |
| Device ppfs content | **OK** | Fortran namelist readable via `git show` |
| Limiter files | **OK** | Plain text R,Z pairs, all 3 files readable |
| configurationsMAGN | **OK** | All 16 named ranges parseable |
| EFIT.db | **OK** | Pulse range mappings readable and parseable |
| Python on JET | **OK** | System Python + venv Python (3.12) both available |
| Symlink resolution | **OK** | Git symlinks confirmed (p68613 → p72258) |

### Complexity Assessment

| Concern | Risk | Mitigation |
|---|---|---|
| XML schema stability | Low | Schema unchanged across 4 fileversions (2007-2016), JET shut down |
| Naming generalization dependency | Medium | Must complete first; scanner code uses new names exclusively |
| Number of DataNodes | Low | 332 per epoch × 10 epochs = 3,320 max, well within graph capacity |
| Signal deduplication | Medium | Deduplicate by (section, instance_id, field) across XML-identical epochs |
| BVCH variants | Low | Track as property, don't create separate epochs |
| EFCC variants (n=1, n=2) | Low | Same geometry XML, different snap files — not separate geometry epochs |
| Missing limiter files | Low | 4 of 7 historical versions have no surviving contour file — skip gracefully |

### Not In Scope

- **PPF signal discovery** — handled by existing `PPFScanner` (separate data source)
- **Greens function matrices** — large binary blobs, not geometry signals
- **Snap files** — probe enable/disable masks, operational not geometric
- **PPF headers** — output DDA definitions, not input geometry
- **EFIT reconstruction results** — consumer data, not machine description

## Config Schema Additions

The `limiter_versions` key in jet.yaml is currently ad-hoc (not in the LinkML facility_config.yaml schema). Two options:

**Option A: Formalize in facility_config.yaml** — Add a `LimiterVersion` class to the config schema. This validates the config structure and auto-documents it.

**Option B: Leave ad-hoc** — The machine_description scanner reads it as a raw dict from `get_facility()`. This works but doesn't get schema validation.

**Recommendation: Option A** after the naming generalization is complete. The `LimiterVersion` class would live alongside `SourceVersion` (renamed from `TreeVersion`).

## Execution Order

1. Complete naming generalization (Phase 1-5 of that plan)
2. Implement `parse_device_xml.py` remote script
3. Implement `MachineDescriptionScanner` plugin
4. Register in `_auto_register()`
5. Run: `uv run imas-codex discover signals jet -s machine_description --scan-only`
6. Verify graph state
7. Run enrichment + IMAS mapping
8. Verify signal counts match estimates
