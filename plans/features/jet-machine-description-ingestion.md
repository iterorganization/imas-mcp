# JET Machine Description Ingestion Plan

**Goal**: Complete ingestion of ALL JET tokamak machine description geometry into the knowledge graph — covering shots 1 through 104522 — via the `imas-codex discover signals jet` CLI pipeline. This includes EFIT device XMLs, chain1 limiter contours, JEC2020 XMLs (PF systems, magnetics, iron boundaries, high-res limiter), MCFG sensor calibration data, and PPF static geometry signals. Final deliverable: epoch-evolving composite plot of JET vessel geometry produced entirely from graph queries.

**Prerequisites**:
- [Facility Config Schema Restructure](facility-config-schema-restructure.md) — **must be completed first**. Phases 1, 7, 8, and 9 of this plan depend on schema-typed config classes (`LimiterVersion` extensions, `JEC2020Config`, `MCFGConfig`, `PPFStaticGeometryConfig`) that are defined and implemented in that plan. Without the schema restructure, scanner code would consume unvalidated dicts.

**Constraints**:
- All ingestion via CLI + scanners — not agent-led
- TCV pipeline retains full functionality and feature parity
- Data provenance tracked at every level
- Static data stored in graph (both metadata and geometry values)
- Incremental testing and unit tests at each phase

---

## Current State (Baseline)

### JET Graph (as of commit 9764974)
| Node Type | Count | Notes |
|-----------|-------|-------|
| DataSource | 1 | `device_xml` only |
| StructuralEpoch | 9 | p68613–p90626 (missing 5 intermediate epochs) |
| DataNode | 2,496 | All from device_xml, type=NUMERIC |
| FacilitySignal | 1,054 | All from xml_extraction |
| DataAccess | 1 | `device_xml:git` |
| Limiter DataNodes | 3 | Mk2GB (126 pts), Mk2HD (116 pts), Mk2ILW (251 pts) |

### JET Config (jet.yaml — corrected)
- **14 structural config versions** covering p68613–p90626
- **5 limiter contour files** covering all shots 1–104522:
  - `limiter.mk2a` (shots 1–44414, 108+12+5 pts, chain1)
  - `limiter.mk2gb` (shots 44415–54351, 193 lines, chain1)
  - `limiter.mk2gb_ns` (shots 54352–63445, 220 lines, chain1 + git cc)
  - `limiter.mk2hd` (shots 63446–79853, 117 lines, chain1 + git cc)
  - `limiter.mk2ilw_cc` (shots 79854+, 251 pts, git)
- Authoritative mapping: `/home/chain1/input/efit/limiter_list`
- JEC2020 config section with 4 XML files
- MCFG sensor config with 77 calibration epochs
- PPF static geometry signal definitions

### Data Sources Inventory

| Source | Location | Content | Status |
|--------|----------|---------|--------|
| EFIT device XMLs | `/home/chain1/git/efit_f90.git` | PF coils, probes, structures, circuits | Partially ingested (3 limiter, 9 epochs) |
| Chain1 limiter files | `/home/chain1/input/efit/Limiters/` | 4 R,Z contour files (mk2a–mk2hd) | Not ingested |
| Chain1 limiter_list | `/home/chain1/input/efit/limiter_list` | Shot-to-file mapping (1–99999) | Config only |
| JEC2020 limiter.xml | `/home/chain1/jec2020/limiter.xml` | 248+ pt ILW contour at T=200°C | Not ingested |
| JEC2020 magnetics.xml | `/home/chain1/jec2020/magnetics.xml` | 95 probes + 36 flux loops, dual PPF/JPF sources | Not ingested |
| JEC2020 pfSystems.xml | `/home/chain1/jec2020/pfSystems.xml` | 20 PF coils + 10 circuits, sub-element geometry | Not ingested |
| JEC2020 ironBoundaries3.xml | `/home/chain1/jec2020/ironBoundaries3.xml` | 96 iron segments with permeabilities | Not ingested |
| MCFG sensors | `/home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt` | 238+ sensor positions (R,Z,angle) from CATIA | Not ingested |
| MCFG.ix | `/home/MAGNW/chain1/input/magn_ep_2019-05-14/MCFG.ix` | 77 calibration epochs (pulse 54283–93563) | Not ingested |
| PPF EFIT/RLIM,ZLIM | MDSplus thin-client `ppf("EFIT/RLIM", shot)` | 251-pt limiter via PPF | Not ingested |
| PPF VESL/CROS | MDSplus thin-client `ppf("VESL/CROS", shot)` | Vessel cross-section | Not ingested |

### TCV Graph (reference target for feature parity)
| Node Type | Count | Notes |
|-----------|-------|-------|
| DataSource | 12 | Multiple trees + subtrees |
| StructuralEpoch | 48 | 8 static versions × 6 systems |
| FacilitySignal | 64,000+ | tree_traversal dominant |
| DataAccess | 14 | Per data source |

### What's Missing for Complete JET Machine Description
1. **Limiter coverage**: Only 3 of 5 limiter contour files have R,Z data in graph (Mk2A, Mk2GB from chain1 not ingested)
2. **Intermediate structural epochs**: 5 new versions in config but not yet in graph
3. **Chain1 limiter file support**: Scanner only reads from git repo, not filesystem paths
4. **Per-limiter-epoch DataSource nodes**: All limiter data shares single `device_xml` DataSource
5. **Epoch↔Limiter relationships**: No graph link between StructuralEpoch and its limiter contour
6. **JEC2020 data**: PF systems, iron boundaries, magnetics probe positions from XML not ingested
7. **JEC2020 dual data sources**: `magnetics.xml` has both PPF and JPF data source references per probe — not captured
8. **MCFG sensor positions**: Canonical CATIA-derived sensor positions not ingested
9. **MCFG calibration epochs**: 77 calibration configuration changes not tracked
10. **PPF static geometry**: EFIT/RLIM, EFIT/ZLIM, VESL/CROS not cross-linked to device_xml DataNodes
11. **EFCC variant epochs**: p68967_efcc1, p70468_efcc2 share geometry but tracked separately in PulseDependencyDB


---

## Phase 1: Complete Limiter Contour Coverage (Shots 1–104522)

**Depends on**: Schema restructure Phase 1 — `LimiterVersion.source`, `LimiterVersion.file_cc`, `DeviceXMLConfig.chain1_limiter_dir` must be schema-typed.

**Objective**: Ingest ALL 5 available limiter R,Z contour files into the graph. Every shot from 1 to 104522 will map to a limiter contour.

**Key correction**: The mk2a file covers shots 1–44414 per the authoritative `limiter_list` at `/home/chain1/input/efit/limiter_list`. This spans what were previously called L91NEW (1–28791), Mk1 (28792–35778), and Mk2A (35779–44414) eras. While the physical hardware changed, EFIT used a single contour file for the entire pre-Mk2GB period. No separate L91NEW or Mk1 contour files exist.

### Phase 1.1: Extend `parse_device_xml.py` for chain1 filesystem files

**Current behavior**: Remote script reads limiter files only from git bare repo (`git show HEAD:<path>`).

**Changes required**:
- Add a `chain1_limiter_dir` parameter to the remote script's input JSON
- When `source: chain1`, read from `chain1_limiter_dir/<file>` instead of `git show`
- The R,Z text format is identical between chain1 and git — same parser
- Return both `source` and `file_path` in the parsed output for provenance

**mk2a file format** (verified via SSH):
```
  108                           ← n_points for primary contour
 1.82396       -.02263          ← R,Z pairs (108 lines)
 1.83565       -.12204
 ...
 1.81558        .07711
   12                           ← n_points for vacuum vessel contour
0.9589      0.0                 ← R,Z pairs (12 lines)
...
    5                           ← n_points for centre column
2.3921      -0.3922             ← R,Z pairs (5 lines)
...
Comment: lim012588.033, shots .ge. 57590, 0 degree limiter
comment: lim081390.033, shots .ge. 70074, advance divertor
vvd, 20/05/02
```

The file contains 3 sections (primary contour + 2 auxiliary). The parser should extract the primary 108-pt contour and optionally store the auxiliary contours as additional properties.

**Files to modify**:
- `imas_codex/remote/scripts/parse_device_xml.py`: Add chain1 file reading path
- `imas_codex/discovery/signals/scanners/device_xml.py`: Pass `chain1_limiter_dir` to script

**Unit tests**:
```python
def test_chain1_limiter_parsing():
    """Limiter files from chain1 frozen directory are parsed identically."""
    mock_output = {
        "limiters": {
            "Mk2A": {"r": [1.82396, 1.83565], "z": [-0.02263, -0.12204],
                      "n_points": 108, "source": "chain1",
                      "file_path": "/home/chain1/input/efit/Limiters/limiter.mk2a"},
        }
    }
```

### Phase 1.2: Add limiter provenance to DataNode

**Changes required**:
- Add `file_source` property to limiter DataNode dict (value: "git" or "chain1")
- Add `file_path` property with the absolute path on the remote host
- These are stored as dynamic Neo4j properties (no schema change needed)

**Files to modify**:
- `imas_codex/discovery/signals/scanners/device_xml.py`: `_persist_graph_nodes()` limiter section

### Phase 1.3: Link StructuralEpoch → Limiter DataNode via USES_LIMITER

**Changes required**:
- After creating limiter DataNodes, create `USES_LIMITER` relationships from each StructuralEpoch to its limiter DataNode
- Each config version specifies its `limiter:` path → match to limiter DataNode
- This enables queries like: "Which epochs used the Mk2ILW first wall?"

**Cypher pattern**:
```cypher
MATCH (se:StructuralEpoch {id: $epoch_id})
MATCH (dn:DataNode {path: $limiter_path})
MERGE (se)-[:USES_LIMITER]->(dn)
```

### Phase 1.4: Validate and run

**CLI**: `imas-codex discover signals jet -s device_xml --scan-only`

**Expected outcome**:
- 14 StructuralEpochs (up from 9)
- 5 limiter DataNodes (up from 3): Mk2A (108 pts), Mk2GB, Mk2GB-NS_cc, Mk2HD_cc, Mk2ILW (251 pts)
- Decision: Store only `file_cc` (coordinate-corrected) when available, fall back to `file` (chain1 original) when no cc version exists
- `USES_LIMITER` relationships connecting each epoch to its limiter

**Integration test**:
```python
def test_complete_limiter_coverage():
    """E2E: All 5 limiter contour files produce DataNodes with R,Z data."""
    # Run scanner with mock SSH
    # Assert 5 limiter DataNodes exist with r_contour, z_contour
    # Assert USES_LIMITER relationships exist for all 14 epochs
```

---

## Phase 2: TCV Feature Parity — Data Provenance Model

**Objective**: Ensure the JET pipeline creates the same graph structure as TCV for data provenance.

### Phase 2.1: Audit TCV vs JET graph structure

**TCV reference model**:
```
DataSource (12) → one per MDSplus tree/subtree
  ↓ IN_DATA_SOURCE
StructuralEpoch (48) → 8 versions × 6 systems
  ↓ INTRODUCED_IN (from DataNode)
DataNode (64,000+) → one per tree node path
  ↓ (promotion)
FacilitySignal → with data_access pointing to DataAccess node
  ↓ DATA_ACCESS
DataAccess (14) → per data source, with template_python
```

**JET current model (gaps)**:
```
DataSource (1) → "device_xml" only
StructuralEpoch (9) → one per config version
DataNode (2,496) → not all link to StructuralEpoch
FacilitySignal (1,054) → accessor="device_xml:section/id/field"
DataAccess (1) → "device_xml:git"
```

### Phase 2.2: Add system property for domain filtering

Keep `device_xml` as the single canonical DataSource, but add a `system` property to DataNode and FacilitySignal for filtering. This enables `WHERE dn.system = 'PF'` without breaking the model.

### Phase 2.3: Ensure DataAccess template completeness

Add `template_python` to the device_xml DataAccess showing graph-based access pattern.

### Phase 2.4: Validate DataNode→StructuralEpoch linkage

Verify `INTRODUCED_IN` relationships are created for all DataNodes.

---

## Phase 3: Extend Limiter Scanner for Non-Git Sources

**Objective**: Support reading limiter R,Z files from arbitrary filesystem paths.

### Phase 3.1: Update remote script for dual-source reading

```python
def read_limiter_file(name, file_path, source, chain1_dir, git_repo, input_prefix):
    if source == "git":
        content = git_show(git_repo, f"{input_prefix}/{file_path}")
    elif source == "chain1":
        full_path = os.path.join(chain1_dir, file_path)
        with open(full_path) as f:
            content = f.read()
    return parse_limiter_rz(content)
```

### Phase 3.2: Update scanner to pass limiter source metadata

Include `source`, `chain1_limiter_dir` in the script input.

### Phase 3.3: Unit tests for mixed-source limiter ingestion

Test both git and chain1 sources produce correct DataNodes.

---

## Phase 4: Additional Structural Epochs

**Objective**: Ingest all 14 config versions, including 5 new intermediate epochs.

### Phase 4.1: Run device_xml scanner with updated config

New epochs: p74387, p78168, p78461, p89431, p90540. All share device XMLs with adjacent epochs — deduplication logic handles this.

### Phase 4.2: EFCC variant handling

Do NOT add EFCC variants as separate StructuralEpochs. They share geometry.

### Phase 4.3: Validate epoch shot range continuity

---

## Phase 5: Epoch-Evolving Geometry Plot from Graph

**Objective**: Composite poloidal plot showing all JET limiter configurations from shot 1 to end-of-JET.

### Phase 5.1: Graph query for all limiter geometry

### Phase 5.2: Plot script (`scripts/plot_jet_limiter_epochs.py`)

### Phase 5.3: Composite plot with PF coils and probes overlay

---

## Phase 6: TCV Pipeline Validation

### Phase 6.1: Run existing test suite
### Phase 6.2: Verify TCV pipeline E2E
### Phase 6.3: Cross-facility scanner isolation test

---

## Phase 7: JEC2020 XML Ingestion

**Depends on**: Schema restructure Phase 1 — `JEC2020Config` and `JEC2020FileConfig` must be schema-typed and registered in `DataSystemsConfig`.

**Objective**: Ingest JET's next-generation EFIT++ geometry files, which provide richer metadata than legacy device XMLs including dual PPF/JPF data source references, iron core geometry, and high-resolution limiter contours.

### Verified Evidence

**Files** (all at `/home/chain1/jec2020/`, symlinked to production):

| File | Lines | Content | Element Counts |
|------|-------|---------|---------------|
| `limiter.xml` | 61 | ILW first wall at T=200°C | 248+ R,Z points |
| `magnetics.xml` | 931 | Mag probes + flux loops | 95 probes, 36 flux loops |
| `pfSystems.xml` | 305 | PF coils + circuits | 20 coils, 10 circuits |
| `ironBoundaries3.xml` | 9 | Iron core boundary | 96 segments |

### Phase 7.1: JEC2020 magnetics.xml parser

**Critical feature**: Each magnetic probe has **dual data source references**:
```xml
<magneticProbe id="1" description="Internal Discrete Coil, Oct.3">
    <timeTrace signalName="BPME(1)" signalName2="DA/C2-CX01"
               dataSource="JET::PPF::/magn/$pulseNumber$/0/jetppf"
               dataSource2="JPF::$pulseNumber$"
               errorType="relativeAbsolute"
               errorRelativeAbsolute="[0.02,0.005]"/>
    <geometry rCentre="4.292" zCentre="0.604" angleUnits="degrees"
              poloidalOrientation="-74.1"/>
</magneticProbe>
```

**Implementation**: Parse each `<magneticProbe>` element → create DataNode with:
- `r`, `z`, `poloidal_orientation` from `<geometry>`
- `ppf_signal`: extracted from `signalName` (e.g., "BPME(1)")
- `jpf_signal`: extracted from `signalName2` (e.g., "DA/C2-CX01")
- `ppf_data_source`: parsed URI template from `dataSource`
- `jpf_data_source`: parsed URI template from `dataSource2`
- `error_type`, `error_relative`, `error_absolute`: from error attributes

**Graph relationships**: Create `HAS_PPF_SOURCE` and `HAS_JPF_SOURCE` relationships linking probe DataNodes to their respective DataAccess nodes. This bridges the device_xml geometry world with the PPF/JPF signal world.

**Flux loops** (36): Same dual-source pattern but different geometry attributes.

**Files to create/modify**:
- `imas_codex/remote/scripts/parse_jec2020_magnetics.py`: Remote XML parser
- `imas_codex/discovery/signals/scanners/device_xml.py`: Add JEC2020 scanning path
- OR: New scanner `imas_codex/discovery/signals/scanners/jec2020.py`

**Decision**: Extend `device_xml` scanner with a `jec2020` code path (same DataSource, different parser) rather than creating a new scanner. The data represents the same physical system.

### Phase 7.2: JEC2020 pfSystems.xml parser

**PF coil format** (verified):
```xml
<pfCoil id="1" name="P1/ME">
    <geometry rCentre="0.897" zCentre="0" dR="0.337" dZ="5.427"
              angle1="0" angle2="0" turnCount="710"/>
</pfCoil>
```

Some coils (P2/SUI/8, P2/SUO/20) have **multi-element geometry** with comma-separated arrays:
```xml
<pfCoil id="3" name="P2/SUI/8">
    <geometry
        rCentre="1.967,2.005,2.043,2.081,1.967,2.005,2.043,2.081,..."
        zCentre="3.871,3.871,3.871,3.871,3.918,3.918,3.918,3.918,..."
        dR="0.035,0.035,..." dZ="0.035,0.035,..."
        turnCount="0.500,0.500,..."/>
</pfCoil>
```

This is richer than the legacy device XML format (which stores filamentary sub-coils differently). The parser must handle both single-value and comma-separated array geometry attributes.

**Circuits** (10): `<pfCircuit>` elements mapping coils to power supply connections.

### Phase 7.3: JEC2020 ironBoundaries3.xml parser

**Iron core boundary** (verified, 96 segments):
```xml
<ironBoundary material2Id="1" materialId="3">
    <knotSet basisFunctionCount="96"
             boundaryCoordsR="6.512, 4.952, 3.392, 1.832, ..."
             boundaryCoordsZ="4.45, 4.45, 4.45, 4.45, ..."
             initialPermeabilities="852.82, 724.86, 887.43, ..."
             segmentLengths="1.56, 1.56, 1.56, ..."
             boundaryLength="32.962"/>
</ironBoundary>
```

Store as DataNodes with:
- `r_contour` / `z_contour` arrays (96 boundary coordinate pairs)
- `permeabilities` array (96 values)
- `segment_lengths` array
- `boundary_length` scalar

**Value for graph**: Iron core geometry enables accurate flux calculations and is essential for EFIT++ reconstructions. Links to PF coil nodes via spatial proximity.

### Phase 7.4: JEC2020 limiter.xml — high-resolution ILW contour

**Format** (verified): XML attributes `rValues` and `zValues` containing 248+ comma-separated floats. Shot range ≥79951.

**Decision**: Ingest as a separate DataNode (`jec2020:limiter`) alongside the existing `device_xml:limiter:Mk2ILW` node. Both contain the ILW first wall but at different resolutions (248+ vs 251 pts) and from different sources. Link via `SAME_GEOMETRY` relationship.

### Phase 7.5: Integration test

```python
def test_jec2020_magnetics_dual_source():
    """JEC2020 magnetics parser extracts both PPF and JPF data sources."""
    # Parse magnetics.xml mock
    # Assert 95 probe DataNodes with ppf_signal and jpf_signal
    # Assert 36 flux loop DataNodes
    # Assert HAS_PPF_SOURCE and HAS_JPF_SOURCE relationships
```

---

## Phase 8: MCFG Sensor Calibration Data

**Depends on**: Schema restructure Phase 1 — `MCFGConfig` must be schema-typed and registered in `DataSystemsConfig`.

**Objective**: Ingest canonical sensor positions from CATIA CAD reference and track calibration epoch changes.

### Verified Evidence

**Sensor position file**: `/home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt`
- 423 lines, CATIA reference model @DIA001160_02-A (2002) + @DIA001288_01-A (2004)
- Coordinates at 200°C operating temperature
- Three sections:
  - **Coils** (id 1–238): Pick-up coils with `id R(m) Z(m) angle(deg) gain relErr absErr JPF-name description`
    ```
    181   4.292   0.604   -74.1   1.0   0.020   0.005   C2E-C101   -15.0  Internal Discrete Coil, Oct.1
    ```
  - **Hall probes** (id 1–8): Ex-vessel probes with `id R(m) Z(m) offset gain relErr absErr JPF-name description`
  - **Other**: Rogowski coils (DL8E, TFCI1B/2B) with gain factors

**MCFG.ix calibration epochs**: `/home/MAGNW/chain1/input/magn_ep_2019-05-14/MCFG.ix`
- 77 entries covering pulse 54283 (2005) to 93563 (2019)
- Format: `$:YYYYMMDD PPPPPP user MCFG:NNNN/MAGNW type ! comment`
- Each entry changes calibration gains, probe status, or TF compensation
- Example: `$:20190514 0093563 gszep MCFG:0011/MAGNW MCFG,MCAL,MOFF ! Calib. with sensors_200c_2019-03-11.txt`

### Phase 8.1: Sensor position parser

**Implementation**: Parse `sensors_200c_2019-03-11.txt` into structured records:

```python
@dataclass
class SensorPosition:
    id: int
    r: float       # metres
    z: float        # metres
    angle: float    # degrees (poloidal orientation)
    gain: float
    rel_err: float
    abs_err: float
    jpf_name: str   # e.g., "C2E-C101"
    section: str    # "coil", "hall", "other"
    description: str
```

**Remote script**: `parse_mcfg_sensors.py` — reads the sensor file, sections split by `#END` markers.

**Graph storage**: Create DataNodes per sensor with:
- Path: `jet:mcfg:sensor:<jpf_name>` (e.g., `jet:mcfg:sensor:C2E-C101`)
- Properties: `r`, `z`, `angle`, `gain`, `rel_err`, `abs_err`, `jpf_name`, `description`
- DataSource: `mcfg` (new DataSource node)
- Cross-reference: `SAME_SENSOR` relationship to JEC2020 magnetics probe DataNode (match by R,Z proximity or by BPME index)

### Phase 8.2: MCFG.ix calibration epoch parser

**Implementation**: Parse each `$:` line into a calibration epoch record:

```python
@dataclass
class CalibrationEpoch:
    date: str           # YYYYMMDD
    first_shot: int     # pulse number
    user: str           # operator
    config_id: str      # MCFG:NNNN/MAGNW
    config_type: str    # MCFG, MCAL, MOFF, MTFC
    description: str    # comment
```

**Graph storage**: Consider creating a dedicated node type or storing as properties on sensor DataNodes. The simplest approach: store calibration epoch data as properties on the MCFG DataSource node, avoiding schema changes.

**Alternative**: A dedicated `CalibrationEpoch` node type with `CALIBRATED_AT` relationships from sensors. This is more powerful but requires schema changes.

**Recommendation**: Start with simple per-sensor DataNodes (Phase 8.1) and defer the full calibration epoch tracking to a follow-up, since the sensor positions themselves (R,Z,angle) don't change between calibration epochs — only gains and offsets do.

### Phase 8.3: Cross-reference MCFG ↔ JEC2020 sensors

After both MCFG sensors and JEC2020 magnetics probes are ingested, create `SAME_SENSOR` relationships:

**Matching strategy**:
- MCFG sensor `C2E-C101` at (R=4.292, Z=0.604) matches JEC2020 probe id=1 (BPME(1)) at (R=4.292, Z=0.604)
- Match by exact R,Z coordinate (both from same CATIA model)
- Or match by JPF name: MCFG `C2E-CX01` → JEC2020 `signalName2="DA/C2-CX01"`

```cypher
MATCH (mcfg:DataNode {data_source_name: 'mcfg', jpf_name: $jpf})
MATCH (jec:DataNode {data_source_name: 'jec2020', jpf_signal: $jpf_signal})
WHERE abs(mcfg.r - jec.r) < 0.001 AND abs(mcfg.z - jec.z) < 0.001
MERGE (mcfg)-[:SAME_SENSOR]->(jec)
```

### Phase 8.4: Unit tests

```python
def test_mcfg_sensor_parsing():
    """Parse MCFG sensor position file into structured records."""
    # 238+ coil sensors + 8 hall probes parsed correctly
    # R,Z,angle values match expected

def test_mcfg_jec2020_cross_reference():
    """MCFG and JEC2020 sensors match by coordinates."""
    # Verify SAME_SENSOR relationships created
```

---

## Phase 9: PPF Static Geometry Signals

**Depends on**: Schema restructure Phase 1 — `PPFStaticGeometryConfig` and `PPFStaticSignal` must be schema-typed and registered in `DataSystemsConfig`.

**Objective**: Link PPF signals containing static machine description data to their corresponding device_xml/JEC2020 DataNodes, and ingest geometry data accessible only via PPF.

### Verified Evidence

**PPF static geometry signals** (verified via MDSplus thin-client):
```python
# MDSplus.Connection('mdsplus.jet.uk')
ppf("EFIT/RLIM", 99896)  # → shape (1, 251) — limiter R coordinates
ppf("EFIT/ZLIM", 99896)  # → shape (1, 251) — limiter Z coordinates
ppf("EFIT/RBND", 99896)  # → shape (947, 105) — boundary R (time-resolved)
ppf("EFIT/ZBND", 99896)  # → shape (947, 105) — boundary Z (time-resolved)
```

The EFIT/RLIM and ZLIM signals contain the same 251-point ILW limiter contour as `limiter.mk2ilw_cc` from the git repo, but accessed via the PPF data system rather than filesystem.

### Phase 9.1: Create PPF static geometry DataAccess nodes

For each PPF signal containing geometry, create a DataAccess node:

```python
DataAccess(
    id="jet:ppf:EFIT/RLIM",
    facility_id="jet",
    method_type="ppf",
    template_python='ppf("EFIT/RLIM", {shot})',
    description="Limiter R coordinates (static, 251 pts)",
)
```

### Phase 9.2: Link PPF geometry to device_xml equivalents

Create `SAME_GEOMETRY` relationships between PPF signals and device_xml DataNodes:

```cypher
MATCH (ppf:DataAccess {id: 'jet:ppf:EFIT/RLIM'})
MATCH (dn:DataNode {path: 'jet:device_xml:limiter:Mk2ILW'})
MERGE (ppf)-[:ACCESSES_GEOMETRY]->(dn)
```

This enables queries like: "How can I access the Mk2ILW limiter contour?" → returns both the graph-stored R,Z data AND the PPF access template.

### Phase 9.3: VESL/CROS vessel cross-section

The `VESL/CROS` signal provides the vessel cross-section contour, which is NOT available from device XML files (device XMLs contain passive structures as rectangular elements, not a continuous vessel outline).

**Implementation**: Retrieve `ppf("VESL/CROS", shot)` via remote script, store as a new DataNode:
- Path: `jet:ppf:VESL/CROS`
- Properties: `r_contour`, `z_contour`, `n_points`
- DataSource: `ppf_static` (new DataSource node)

### Phase 9.4: Unit tests

```python
def test_ppf_geometry_data_access():
    """PPF static geometry signals create DataAccess with templates."""

def test_ppf_device_xml_cross_reference():
    """PPF RLIM/ZLIM links to device_xml limiter DataNode."""
```

---

## Phase 10: TCV Pipeline Validation (Final)

**Objective**: Confirm all changes maintain TCV signals pipeline functionality.

### Phase 10.1: Run existing test suite

```bash
uv run pytest tests/discovery/test_signals_pipeline.py -v
uv run pytest tests/discovery/test_device_xml_scanner.py -v
uv run pytest tests/discovery/test_signal_checking.py -v
```

### Phase 10.2: Verify TCV pipeline E2E

### Phase 10.3: Cross-facility scanner isolation test

---

## Phase 11: Composite Geometry Plot (Final Deliverable)

**Objective**: Epoch-evolving composite plot of ALL JET geometry from graph data.

### Phase 11.1: Plot all 5 limiter epochs (shots 1–104522)

Query all limiter DataNodes, plot R,Z contours color-coded by era.

### Phase 11.2: Overlay PF coils and magnetic probes

From both EFIT device XML DataNodes and JEC2020 DataNodes.

### Phase 11.3: Overlay iron core boundary

From JEC2020 `ironBoundaries3.xml` DataNodes.

### Phase 11.4: Overlay MCFG sensor positions

From MCFG sensor DataNodes (238+ sensors with R,Z,angle).

### Phase 11.5: Overlay vessel cross-section

From PPF `VESL/CROS` DataNode.

**Final plot layers**:
1. All limiter contours (5 epochs, color-coded)
2. PF coil rectangles (latest config)
3. Magnetic probe positions (from JEC2020, showing enabled/disabled state)
4. MCFG sensor positions (from CATIA CAD, matching JEC2020)
5. Iron core boundary (96 segments)
6. Vessel cross-section (from PPF VESL/CROS)
7. Flux loop positions

---

## Implementation Order Summary

| Phase | Deliverable | Files Modified | Tests | Priority |
|-------|-------------|----------------|-------|----------|
| 1.1–1.4 | Complete limiter coverage (5 files, shots 1–104522) | parse_device_xml.py, device_xml.py | 3 unit + 1 E2E | P0 |
| 2.1–2.4 | TCV parity audit + fixes | device_xml.py | 2 unit | P1 |
| 3.1–3.3 | Mixed-source limiter support | parse_device_xml.py, device_xml.py | 1 unit | P0 |
| 4.1–4.3 | Full PulseDependencyDB epochs (14 total) | jet.yaml (done), CLI run | 1 unit | P1 |
| 5.1–5.3 | Epoch-evolving limiter plot | New script | Manual | P2 |
| 6.1–6.3 | TCV validation (mid-point) | — | 3 test runs | P1 |
| 7.1–7.5 | JEC2020 XML ingestion (magnetics, PF, iron, limiter) | New parser + scanner extension | 4 unit | P0 |
| 8.1–8.4 | MCFG sensor positions + cross-references | New parser + scanner | 2 unit | P1 |
| 9.1–9.4 | PPF static geometry DataAccess + cross-links | Scanner extension | 2 unit | P1 |
| 10.1–10.3 | Final TCV validation | — | 3 test runs | P0 |
| 11.1–11.5 | Final composite geometry plot | New script | Manual | P2 |

**Total new tests**: ~16 unit tests + 7 validation runs
**Files modified**: 3 (parse_device_xml.py, device_xml.py, test_device_xml_scanner.py)
**Files created**: ~4 (parse_jec2020_magnetics.py, parse_mcfg_sensors.py, plot_jet_geometry_epochs.py, test files)
**Config updated**: jet.yaml (limiter versions corrected, JEC2020/MCFG/PPF sections added)

---

## Key Design Decisions

1. **mk2a covers shots 1–44414**. Per the authoritative `limiter_list` at `/home/chain1/input/efit/limiter_list`, the mk2a contour file is used for ALL shots from 1 to 44414. No separate L91NEW or Mk1 contour files exist on disk. The 5-version scheme (Mk2A → Mk2ILW) replaces the previous 7-version scheme.

2. **Prefer coordinate-corrected limiter files** (`file_cc`) over originals when both exist. The corrected version is the canonical DataNode.

3. **No EFCC variant epochs**. They share geometry with base configs.

4. **JEC2020 extends device_xml scanner**, not a separate scanner. Same DataSource, different parser input.

5. **MCFG sensors create their own DataSource** (`mcfg`). While the sensor positions come from CATIA CAD (not EFIT), they are cross-referenced to JEC2020/device_xml probes via `SAME_SENSOR` relationships.

6. **PPF static geometry creates DataAccess nodes**, not DataNodes. The geometry data itself is already stored in device_xml DataNodes — PPF just provides an alternative access method.

7. **Iron core boundary is new data**. Not available from legacy device XMLs. JEC2020 `ironBoundaries3.xml` is the only source.

---

## File Locations Reference

All paths verified via SSH (March 2026):

### Chain1 EFIT Limiter Files
```
/home/chain1/input/efit/Limiters/limiter.mk2a      3124 bytes, 136 lines, 2002-05-20
/home/chain1/input/efit/Limiters/limiter.mk2gb      4030 bytes, 193 lines, 2002-05-20
/home/chain1/input/efit/Limiters/limiter.mk2gb_ns   5281 bytes, 220 lines, 2002-05-20
/home/chain1/input/efit/Limiters/limiter.mk2hd      2016 bytes, 117 lines, 2006-06-14
/home/chain1/input/efit/limiter_list                 authoritative shot-to-file mapping
```

### Chain1 EFIT_F90 (coordinate-corrected, also in git)
```
/home/chain1/input/efit_f90/Limiters/limiter.mk2gb_ns_cc
/home/chain1/input/efit_f90/Limiters/limiter.mk2hd_cc
/home/chain1/input/efit_f90/Limiters/limiter.mk2ilw_cc
```

### JEC2020 XML Files
```
/home/chain1/jec2020/limiter.xml             61 lines,  248+ R,Z points
/home/chain1/jec2020/magnetics.xml          931 lines,  95 probes + 36 flux loops
/home/chain1/jec2020/pfSystems.xml          305 lines,  20 PF coils + 10 circuits
/home/chain1/jec2020/ironBoundaries3.xml      9 lines,  96 boundary segments
```

### MCFG Sensor Configuration
```
/home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt    423 lines, 238+ sensors
/home/MAGNW/chain1/input/magn_ep_2019-05-14/MCFG.ix            77 calibration epochs
/home/MAGNW/chain1/input/magn_ep_2019-05-14/magn_ep.dat        MAGN DDA epoch data
```

### Git Repository
```
/home/chain1/git/efit_f90.git               bare repo, single 'master' branch
HEAD: 50a3e3b (2018-01-26)                  108 tags (Lx099-0 through Lx103-77)
```

---

## Verification Queries (Final State)

### All limiter epochs with geometry
```cypher
MATCH (dn:DataNode)
WHERE dn.facility_id = 'jet'
  AND dn.path STARTS WITH 'jet:device_xml:limiter:'
  AND dn.r_contour IS NOT NULL
WITH dn ORDER BY dn.first_shot
RETURN dn.path AS epoch, dn.description AS description,
       dn.first_shot AS from_shot, dn.last_shot AS to_shot,
       size(dn.r_contour) AS n_points
```
Expected: 5 rows (Mk2A 108pts, Mk2GB, Mk2GB-NS, Mk2HD, Mk2ILW 251pts)

### Magnetic probes with dual data sources
```cypher
MATCH (dn:DataNode {data_source_name: 'jec2020'})
WHERE dn.path STARTS WITH 'jet:jec2020:magprobe:'
RETURN dn.ppf_signal, dn.jpf_signal, dn.r, dn.z, dn.poloidal_orientation
ORDER BY dn.ppf_signal
```
Expected: 95 rows

### MCFG↔JEC2020 cross-references
```cypher
MATCH (mcfg:DataNode)-[:SAME_SENSOR]->(jec:DataNode)
WHERE mcfg.data_source_name = 'mcfg'
RETURN mcfg.jpf_name, jec.ppf_signal, mcfg.r, mcfg.z
```
Expected: ~95 matched pairs

### Complete machine description for a shot
```cypher
MATCH (dn:DataNode {facility_id: 'jet'})
WHERE dn.first_shot <= $shot AND (dn.last_shot IS NULL OR dn.last_shot >= $shot)
RETURN dn.data_source_name, count(dn) AS node_count
ORDER BY dn.data_source_name
```
Expected: device_xml (PF+PS+MP+FL+CI+LIM), jec2020, mcfg, ppf_static
