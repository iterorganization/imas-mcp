# JET Machine Description Ingestion Plan

**Goal**: Complete ingestion of JET tokamak machine description geometry into the knowledge graph via the `imas-codex discover signals jet` CLI pipeline. Final deliverable: epoch-evolving composite plot of JET vessel geometry (shot 0 → 104522) produced entirely from graph queries.

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

### JET Config (jet.yaml — just updated)
- **14 structural config versions** (was 9) covering p68613–p90626
- **7 limiter versions** with source provenance (git vs chain1)
- `chain1_limiter_dir` added for pre-git-repo files
- L91NEW, Mk1 marked as unavailable (predate digital archiving)

### TCV Graph (reference target for feature parity)
| Node Type | Count | Notes |
|-----------|-------|-------|
| DataSource | 12 | Multiple trees + subtrees |
| StructuralEpoch | 48 | 8 static versions × 6 systems |
| FacilitySignal | 64,000+ | tree_traversal dominant |
| DataAccess | 14 | Per data source |

### What's Missing for Complete JET Machine Description
1. **Limiter coverage**: Only 3 of 7 limiter epochs have R,Z data in graph (Mk2A, Mk2GB-NS, Mk2HD from chain1 not ingested)
2. **Intermediate structural epochs**: 5 new versions in config but not yet in graph
3. **Chain1 limiter file support**: Scanner only reads from git repo, not filesystem paths
4. **Per-limiter-epoch DataSource nodes**: All limiter data shares single `device_xml` DataSource
5. **Epoch↔Limiter relationships**: No graph link between StructuralEpoch and its limiter contour
6. **JEC2020 data**: PF systems, iron boundaries, magnetics probe positions from XML not ingested
7. **Dual data source references**: `magnetics.xml` references both PPF and JPF sources per probe — not captured
8. **EFCC variant epochs**: p68967_efcc1, p70468_efcc2 share geometry but tracked separately in PulseDependencyDB

---

## Phase 1: Complete Limiter Contour Coverage

**Objective**: Ingest ALL 5 available limiter R,Z contours into the graph (Mk2A through Mk2ILW). Two remain unavailable (L91NEW, Mk1).

### Phase 1.1: Extend `parse_device_xml.py` for chain1 files

**Current behavior**: Remote script reads limiter files only from git bare repo (`git show HEAD:<path>`).

**Changes required**:
- Add a `chain1_limiter_dir` parameter to the remote script's input JSON
- When `source: chain1`, read from `chain1_limiter_dir/<file>` instead of `git show`
- The R,Z text format is identical between chain1 and git — same parser
- Return both `source` and `file_path` in the parsed output for provenance

**Files to modify**:
- `imas_codex/remote/scripts/parse_device_xml.py`: Add chain1 file reading path
- `imas_codex/discovery/signals/scanners/device_xml.py`: Pass `chain1_limiter_dir` to script

**Unit tests**:
```python
# test_device_xml_scanner.py — add chain1 limiter test case
def test_chain1_limiter_parsing():
    """Limiter files from chain1 frozen directory are parsed identically."""
    # Mock script output with chain1 source annotation
    mock_output = {
        "limiters": {
            "Mk2A": {"r": [2.0, 2.5], "z": [1.5, 1.8], "source": "chain1"},
            "Mk2ILW": {"r": [2.0, 2.5, 3.0], "z": [1.5, 1.8, 1.5], "source": "git"},
        }
    }
    # Verify both sources create DataNode with identical schema
```

### Phase 1.2: Add limiter provenance to DataNode

**Current behavior**: Limiter DataNodes have `data_source_name: device_xml` and `source: introspection`. No indication of which filesystem source provided the data.

**Changes required**:
- Add `file_source` property to limiter DataNode dict (value: "git" or "chain1")  
- Add `file_path` property with the absolute path on the remote host
- These are stored as dynamic Neo4j properties (no schema change needed)

**Files to modify**:
- `imas_codex/discovery/signals/scanners/device_xml.py`: `_persist_graph_nodes()` limiter section

**Unit tests**:
```python
def test_limiter_data_node_has_provenance():
    """Limiter DataNodes track source filesystem (git vs chain1)."""
    # Check DataNode properties include file_source and file_path
```

### Phase 1.3: Link StructuralEpoch → Limiter DataNode

**Current behavior**: StructuralEpoch nodes have no relationship to their limiter contour. The `limiter` field in config versions specifies the file path but this isn't reflected in the graph.

**Changes required**:
- After creating limiter DataNodes, create `USES_LIMITER` relationships from each StructuralEpoch to its limiter DataNode
- Each config version specifies its `limiter:` path → match to limiter DataNode
- This enables queries like: "Which epochs used the Mk2ILW first wall?"

**Files to modify**:
- `imas_codex/discovery/signals/scanners/device_xml.py`: Add `USES_LIMITER` relationship creation in `_persist_graph_nodes()`

**Schema note**: This is a dynamic relationship — no LinkML schema change needed. If we want it in the schema later, add a `limiter` slot to StructuralEpoch with `range: DataNode` and `relationship_type: USES_LIMITER`.

**Cypher pattern**:
```cypher
MATCH (se:StructuralEpoch {id: $epoch_id})
MATCH (dn:DataNode {path: $limiter_path})
MERGE (se)-[:USES_LIMITER]->(dn)
```

**Unit tests**:
```python
def test_epoch_limiter_relationship():
    """StructuralEpoch nodes link to their limiter DataNode."""
    # After scan, query: MATCH (se)-[:USES_LIMITER]->(dn) RETURN count(*)
    # Should equal number of epochs with limiter config
```

### Phase 1.4: Validate and run

**CLI command**: `imas-codex discover signals jet -s device_xml --scan-only`

**Expected outcome**:
- 14 StructuralEpochs (up from 9)
- 5 limiter DataNodes (up from 3): Mk2A, Mk2GB-NS (chain1), Mk2GB-NS (git cc), Mk2HD (chain1), Mk2HD (git cc), Mk2ILW
  - Decision: Store chain1 originals and git cc versions as separate DataNodes? Or prefer coordinate-corrected only? **Recommendation**: Store only the `file_cc` (coordinate-corrected) version when available, falling back to `file` (chain1 original) when no cc version exists. This gives: Mk2A (chain1), Mk2GB-NS_cc (git), Mk2HD_cc (git), Mk2ILW (git) = 4 with data, 2 without (L91NEW, Mk1).
- `USES_LIMITER` relationships connecting each epoch to its limiter

**Integration test**:
```python
def test_complete_limiter_coverage():
    """E2E: All limiter epochs with files produce DataNodes with R,Z data."""
    # Run scanner with mock SSH
    # Assert 4+ limiter DataNodes exist with r_contour, z_contour
    # Assert USES_LIMITER relationships exist
```

---

## Phase 2: TCV Feature Parity — Data Provenance Model

**Objective**: Ensure the JET pipeline creates the same graph structure as TCV for data provenance. TCV has separate DataSource, DataAccess, and IN_DATA_SOURCE relationships per tree/subtree.

### Phase 2.1: Audit TCV vs JET graph structure

**TCV reference model** (from graph queries):
```
DataSource (12) → one per MDSplus tree/subtree
  ↓ IN_DATA_SOURCE
StructuralEpoch (48) → 8 versions × 6 systems per tree
  ↓ INTRODUCED_IN (from DataNode)
DataNode (64,000+) → one per tree node path
  ↓ (promotion)
FacilitySignal → with data_access pointing to DataAccess node
  ↓ DATA_ACCESS
DataAccess (14) → per data source, with template_python, template_mdsplus
```

**JET current model** (gaps):
```
DataSource (1) → "device_xml" only
  ↓ IN_DATA_SOURCE
StructuralEpoch (9) → one per config version
  ↓ INTRODUCED_IN (from DataNode — but not all DataNodes link!)
DataNode (2,496) → geometry values stored
  ↓ (no promotion — signals created directly)
FacilitySignal (1,054) → accessor="device_xml:section/id/field"
  ↓ DATA_ACCESS
DataAccess (1) → "device_xml:git"
```

### Phase 2.2: Add per-system DataSource nodes for JET

**Current**: Single `DataSource(name=device_xml)`. TCV has one DataSource per tree.

**Change**: Create DataSource nodes per JET system (PF, PS, MP, FL, CI, LIM) to match TCV's per-tree structure. This enables domain-filtered queries analogous to TCV's `MATCH (ds:DataSource {name: 'magnetics'})`.

**Files to modify**:
- `imas_codex/discovery/signals/scanners/device_xml.py`: In `_persist_graph_nodes()`, create DataSource per system

**Decision**: Keep the single `device_xml` DataSource as a parent, and add child DataSources for each system? Or replace with per-system DataSources? **Recommendation**: Keep `device_xml` as the canonical DataSource (it represents the data format), but add a `system` property to DataNode and FacilitySignal for filtering. This matches the existing pattern without breaking the model.

### Phase 2.3: Ensure DataAccess template completeness

**TCV DataAccess** has `template_python` and `template_mdsplus` fields showing how to access data programmatically.

**JET DataAccess** for device_xml currently stores:
- `method_type: xml_extraction`
- No access template (data is static in git, not programmatically queried)

**Change**: Add `template_python` to the device_xml DataAccess showing how to read the geometry data from the graph:
```python
template_python = """
from imas_codex.graph import GraphClient
with GraphClient() as gc:
    result = gc.query('''
        MATCH (dn:DataNode {path: $path})
        RETURN dn.r_contour AS r, dn.z_contour AS z
    ''', path="{node_path}")
"""
```

This isn't for live acquisition (data is static) but documents the access pattern.

### Phase 2.4: Validate DataNode→StructuralEpoch linkage

**Current gap**: Not all DataNodes have `introduced_version` linking to their StructuralEpoch. The `_persist_graph_nodes()` function sets `introduced_version` on DataNode dicts, but `create_nodes()` may not create the relationship.

**Files to modify**:
- `imas_codex/discovery/signals/scanners/device_xml.py`: Verify `INTRODUCED_IN` relationships are created for all DataNodes

**Unit test**:
```python
def test_data_node_epoch_linkage():
    """Every DataNode links to its StructuralEpoch via INTRODUCED_IN."""
    # After scan: MATCH (dn:DataNode)-[:INTRODUCED_IN]->(se:StructuralEpoch)
    # Count should equal total DataNode count
```

---

## Phase 3: Extend Limiter Scanner for Non-Git Sources

**Objective**: Support reading limiter R,Z files from arbitrary filesystem paths (not just git bare repo), enabling ingestion of frozen chain1 files.

### Phase 3.1: Update remote script `parse_device_xml.py`

**Current**: Uses `git show HEAD:<path>` for all file reads.

**Changes**:
```python
def read_limiter_file(name, file_path, source, chain1_dir, git_repo, input_prefix):
    """Read limiter R,Z from git or chain1 filesystem."""
    if source == "git":
        content = git_show(git_repo, f"{input_prefix}/{file_path}")
    elif source == "chain1":
        full_path = os.path.join(chain1_dir, os.path.basename(file_path))
        with open(full_path) as f:
            content = f.read()
    else:
        raise ValueError(f"Unknown source: {source}")
    return parse_limiter_rz(content)
```

**Key constraint**: Remote script runs on JET host via `run_python_script()`. It needs venv Python (3.12+). Chain1 directory is a regular filesystem path accessible from the JET login nodes.

### Phase 3.2: Update scanner to pass limiter source metadata

**Current**: `scan()` builds limiter_files list from config, passing only name + file path.

**Changes**: Include `source`, `chain1_limiter_dir`, and `file_cc` in the script input:
```python
limiter_files.append({
    "name": lv["name"],
    "file": lv.get("file") or lv.get("file_cc"),
    "source": lv.get("source") or lv.get("source_cc", "git"),
    "chain1_dir": config.get("chain1_limiter_dir"),
})
```

### Phase 3.3: Unit tests for mixed-source limiter ingestion

```python
# test_device_xml_scanner.py
CONFIG_WITH_CHAIN1 = {
    "git_repo": "/home/chain1/git/efit_f90.git",
    "input_prefix": "JET/input",
    "chain1_limiter_dir": "/home/chain1/input/efit_f90/Limiters",
    "versions": [MOCK_VERSION],
    "limiter_versions": [
        {"name": "Mk2A", "file": "Limiters/limiter.mk2a", "source": "chain1"},
        {"name": "Mk2ILW", "file": "Limiters/limiter.mk2ilw_cc", "source": "git"},
    ],
}

def test_mixed_source_limiter_ingestion():
    """Limiters from both git and chain1 sources are ingested."""
    # Mock parse_device_xml.py output with both sources
    # Verify both create DataNodes with correct file_source property
```

---

## Phase 4: Additional Structural Epochs (PulseDependencyDB Complete)

**Objective**: Ingest all 14 config versions from the updated jet.yaml, including the 5 new intermediate epochs.

### Phase 4.1: Run device_xml scanner with updated config

The scanner already reads `versions` from config and creates StructuralEpoch + DataNode nodes. Since we added 5 new versions to jet.yaml, simply re-running the scanner should pick them up.

**CLI**: `imas-codex discover signals jet -s device_xml --scan-only`

**Expected new epochs**:
- p74387 (FLME(3) timing issue)
- p78168 (P801B broken)
- p78461 (DMSS=108 transition)
- p89431 (P802B broke)
- p90540 (I803 failed)

**Note**: All 5 new epochs use `device_xml: Devices/device_p68613.xml` (baseline) or `device_p88368.xml`/`device_p89440.xml` — they share geometry with adjacent epochs. The scanner's XML deduplication logic handles this correctly (only parses each XML once, then reuses parsed data for all versions referencing that file).

### Phase 4.2: EFCC variant handling (future consideration)

PulseDependencyDB interleaves EFCC session configs (p68967_efcc1, p70468_efcc2, p78814_efcc2, p79854_efcc1/efcc2) with base configs. These share the same device XML and limiter contour — only snap_file and Green's differ.

**Decision**: Do NOT add EFCC variants as separate StructuralEpochs. They don't represent different machine geometry. If needed later, they can be added as properties on the base epoch node (e.g., `efcc_variants: ["n=1", "n=2"]`).

### Phase 4.3: Validate epoch shot range continuity

**Unit test**:
```python
def test_epoch_shot_range_coverage():
    """Config versions cover continuous shot range from 68613 to end of JET."""
    versions = load_jet_config()["data_systems"]["device_xml"]["versions"]
    # Verify: version[i].last_shot + 1 == version[i+1].first_shot
    # Or at minimum no gaps in coverage
```

---

## Phase 5: Epoch-Evolving Geometry Plot from Graph

**Objective**: Produce a composite poloidal plot showing all JET limiter configurations evolving across epochs, using only graph queries for data retrieval.

### Phase 5.1: Graph query for limiter geometry

```cypher
-- All limiter contours with epoch ranges
MATCH (dn:DataNode)
WHERE dn.facility_id = 'jet'
  AND dn.data_source_name = 'device_xml'
  AND dn.path STARTS WITH 'jet:device_xml:limiter:'
RETURN dn.path AS path,
       dn.description AS description,
       dn.first_shot AS first_shot,
       dn.last_shot AS last_shot,
       dn.r_contour AS r,
       dn.z_contour AS z,
       dn.n_points AS n_points
ORDER BY dn.first_shot
```

### Phase 5.2: Plot script

Create `scripts/plot_jet_limiter_epochs.py`:

```python
"""Plot all JET limiter epoch contours from graph data.

Usage: uv run python scripts/plot_jet_limiter_epochs.py
"""
from imas_codex.graph import GraphClient

def get_limiter_epochs():
    with GraphClient() as gc:
        return gc.query("""
            MATCH (dn:DataNode)
            WHERE dn.facility_id = 'jet'
              AND dn.path STARTS WITH 'jet:device_xml:limiter:'
            RETURN dn.path AS path,
                   dn.description AS desc,
                   dn.first_shot AS first_shot,
                   dn.last_shot AS last_shot,
                   dn.r_contour AS r,
                   dn.z_contour AS z
            ORDER BY dn.first_shot
        """)

def plot_limiter_epochs(epochs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 14))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for epoch, color in zip(epochs, colors):
        label = f"{epoch['desc']} (shot {epoch['first_shot']})"
        ax.plot(epoch['r'], epoch['z'], color=color, label=label, linewidth=1.5)
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('JET First Wall Evolution')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('jet_limiter_epochs.png', dpi=150)
```

### Phase 5.3: Composite plot with PF coils and probes

After limiter contours are validated, overlay PF coil positions and probe locations from the same graph queries:

```cypher
-- PF coil positions for a specific epoch
MATCH (dn:DataNode)-[:INTRODUCED_IN]->(se:StructuralEpoch {id: 'jet:device_xml:p90626'})
WHERE dn.path CONTAINS ':pfcoils:'
RETURN dn.r AS r, dn.z AS z, dn.dr AS width, dn.dz AS height
```

```cypher
-- Magnetic probe positions for baseline epoch
MATCH (dn:DataNode)-[:INTRODUCED_IN]->(se:StructuralEpoch {id: 'jet:device_xml:p68613'})
WHERE dn.path CONTAINS ':magprobes:'
RETURN dn.r AS r, dn.z AS z, dn.angle AS angle, dn.description AS desc
```

The final composite plot shows:
1. All limiter contours (color-coded by epoch)
2. PF coil rectangles (latest epoch)
3. Magnetic probe positions (with enabled/disabled state from snap_file)
4. Flux loop positions
5. Passive structure outlines

This matches the format of the TCV machine description plots already produced.

---

## Phase 6: TCV Pipeline Validation

**Objective**: Confirm that all changes maintain TCV signals pipeline functionality.

### Phase 6.1: Run existing TCV signal tests

```bash
uv run pytest tests/discovery/test_signals_pipeline.py -v
uv run pytest tests/discovery/test_device_xml_scanner.py -v
uv run pytest tests/discovery/test_signal_checking.py -v
```

All must pass unchanged.

### Phase 6.2: Verify TCV pipeline E2E (if graph available)

```bash
imas-codex discover signals tcv --scan-only --scanners mdsplus --time 2
```

Verify no regressions in:
- StructuralEpoch creation
- DataNode extraction
- FacilitySignal promotion
- DataAccess templates

### Phase 6.3: Cross-facility scanner isolation test

```python
def test_scanner_facility_isolation():
    """JET device_xml scanner changes don't affect TCV pipeline."""
    # Run TCV mdsplus scan with mock
    # Run JET device_xml scan with mock
    # Verify no cross-contamination of DataSource/DataAccess nodes
```

---

## Phase 7: Future Extensions (Not in Scope)

These items are documented for future agents but **NOT** part of the current implementation:

### 7.1: JEC2020 XML Ingestion
The `/common/chain1/jec2020/` directory contains richer XML files:
- `pfSystems.xml`: PF coil parameters with COCOS and unit metadata
- `magnetics.xml`: Probe positions with dual `dataSource="JET::PPF::/magn/$pulseNumber$/0/jetppf"` and `dataSource2="JPF::$pulseNumber$"` references
- `ironBoundaries3.xml`: Iron core limbs and yoke geometry
- `limiter.xml`: 248+ point ILW first wall at T=200°C

This requires a new `jec2020_xml` scanner or an extension to `device_xml`. The dual data source references are particularly valuable for linking probe signals to their PPF/JPF access paths.

### 7.2: MCFG Sensor Calibration Epochs
The MCFG system at `/home/MAGNW/chain1/input/` tracks pulse-dependent sensor calibrations (MCFG.ix). This is a distinct epoch system from PulseDependencyDB and could create `CalibrationEpoch` nodes.

### 7.3: PPF Static Geometry Signals
Some PPF DDAs contain static geometry data identical across ILW pulses:
- `EFIT/RLIM,ZLIM` (251-pt limiter)
- `LIDR/Z` (50-pt)
- `VESL/CROS` (vessel cross-section)

These should be ingested with appropriate links to the device_xml DataNodes they duplicate.

### 7.4: Pre-p68613 Geometry
The EFIT device XML system starts at p68613 (2006). JET operated from 1983.
Pre-p68613 geometry is partially available:
- limiter_list.dat covers shots 1–99999 (Mk2A through Mk2HD in chain1)
- No device XML files exist for pre-2006 PF coil/probe configurations
- The CATIA sensor model dates from 2002 (DIA001160_02-A)
- Frozen chain1 efitp_24c/ has 4 limiter files (pre-EFIT-F90 era)

Fully covering pre-2006 geometry would require manual reconstruction from engineering drawings or CAD models.

---

## Implementation Order Summary

| Phase | Deliverable | Files Modified | Tests |
|-------|-------------|----------------|-------|
| 1.1 | Chain1 limiter file reading | parse_device_xml.py, device_xml.py | 1 unit |
| 1.2 | Limiter provenance properties | device_xml.py | 1 unit |
| 1.3 | Epoch↔Limiter relationships | device_xml.py | 1 unit |
| 1.4 | Integration validation | — | 1 E2E |
| 2.1–2.4 | TCV parity audit + fixes | device_xml.py | 2 unit |
| 3.1–3.3 | Mixed-source limiter support | parse_device_xml.py, device_xml.py | 1 unit |
| 4.1–4.3 | Full PulseDependencyDB epochs | jet.yaml (done), run CLI | 1 unit |
| 5.1–5.3 | Epoch-evolving plot | New script | Manual |
| 6.1–6.3 | TCV validation | — | 3 test runs |

**Total new tests**: ~8 unit tests + 3 validation runs
**Files modified**: 3 (parse_device_xml.py, device_xml.py, test_device_xml_scanner.py)
**Files created**: 1 (plot_jet_limiter_epochs.py)
**Config already updated**: jet.yaml (committed 9764974)

---

## Key Design Decisions

1. **Prefer coordinate-corrected limiter files** (`file_cc`) over originals when both exist. Store the corrected version as the canonical DataNode. The original chain1 file path is recorded as `file_path_original` for provenance.

2. **No EFCC variant epochs**. They share geometry with base configs — only snap_file and Green's differ. Not structural changes.

3. **No schema changes for Phase 1–5**. All new properties (file_source, file_path, USES_LIMITER) are dynamic Neo4j properties/relationships. Schema changes deferred to Phase 7 if JEC2020 ingestion proceeds.

4. **device_xml scanner handles both git and filesystem sources**. The scanner is extended, not forked. The `source` field in limiter config controls the code path.

5. **Single DataSource for all device_xml data**. Don't fragment into per-system DataSources (unlike TCV). JET's geometry comes from a unified source (EFIT input files) — creating artificial DataSource boundaries would misrepresent the data architecture.

---

## Verification Query (Final State)

After all phases complete, this single query produces the epoch-evolving geometry:

```cypher
MATCH (dn:DataNode)
WHERE dn.facility_id = 'jet'
  AND dn.path STARTS WITH 'jet:device_xml:limiter:'
  AND dn.r_contour IS NOT NULL
WITH dn ORDER BY dn.first_shot
RETURN dn.path AS epoch,
       dn.description AS description,
       dn.first_shot AS from_shot,
       dn.last_shot AS to_shot,
       size(dn.r_contour) AS n_points,
       dn.r_contour AS r,
       dn.z_contour AS z
```

Expected: 4–5 rows (Mk2A, Mk2GB-NS, Mk2HD, Mk2ILW) with full R,Z contour arrays.
