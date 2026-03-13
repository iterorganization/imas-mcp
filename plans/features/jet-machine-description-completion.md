# JET Machine Description — Remaining Ingestion Work

> **Status**: Planning
> **Priority**: Medium — incremental improvements to already-operational pipeline
> **Scope**: Extend the `device_xml` scanner (`imas-codex discover signals jet -s device_xml`)

## Context

The JET machine description ingestion pipeline is operational and covers shots
1 through 104522. The `device_xml` scanner handles 7 data sources via a single
`scan()` invocation:

| # | Data Source | Remote Script | What It Produces |
|---|-----------|---------------|-----------------|
| 1 | EFIT device XMLs | `parse_device_xml.py` | 20 SignalEpochs, PF coils, probes, structures, limiters |
| 2 | JEC2020 geometry | `parse_jec2020.py` | 4 XML files: limiter, magnetics (131 sensors), PF (20 coils, 10 circuits), iron core (96 segments) |
| 3 | MCFG sensors | `parse_mcfg_sensors.py` | 238+ sensor positions from CATIA reference (2019 production file) |
| 4 | Magnetics config | `parse_magnetics_config.py` | Shot-range → sensor config mapping (12 epochs, 122–328 sensors each), covers shots 1–90538 |
| 5 | PF coil turns | `parse_pf_coil_turns.py` | 12 PF circuit-to-coil mappings with SAME_COMPONENT cross-refs |
| 6 | Greens table | `parse_greens_table.py` | 11 GREENS versions with shot-range mappings and USES_GREENS cross-refs |
| 7 | PPF static geometry | *(local)* | DataAccess nodes for EFIT/RLIM, VESL/CROS etc. with ACCESSES_GEOMETRY links |

All 7 limiter contour versions (Limiter through Mk2ILW) are ingested from both
chain1 and git sources. USES_LIMITER, MATCHES_SENSOR, SAME_GEOMETRY, and
IN_CIRCUIT cross-reference relationships are all created. Four plot scripts
generate epoch-evolving composite geometry from graph queries.

## Why the Pipeline Works Without This Plan

The pipeline is fully functional. Running `imas-codex discover signals jet`
invokes all registered scanners (device_xml, ppf, jpf, wiki). The device_xml
scanner successfully creates all SignalEpoch, SignalNode, FacilitySignal, and
DataAccess nodes needed for signal enrichment and IMAS mapping.

The items below are **incremental enrichments** — they add provenance depth,
historical coverage, and cross-referencing density to already-ingested data.
The core graph topology and signal population are complete without them.

## What Completing This Plan Would Add

1. **Historical sensor position tracking** — currently only the 2019 production
   sensor positions are ingested. Adding earlier versions (2005, 2015) would
   enable validation of position stability over time and match the calibration
   epoch history
2. **Calibration change provenance** — the MCFG.ix file tracks 77 calibration
   changes (gain corrections, probe replacements, TF compensation updates) from
   2006–2019. This is valuable for data quality assessment per shot range
3. **PF circuit real-time access** — adding JPF addresses from the `cturns` file
   enables code generation for real-time coil current monitoring

---

## Phase 1: Sensors-200C Historical Versions

**Priority**: Low | **Effort**: Small

The MCFG scanner already ingests the 2019 production file
(`sensors_200c_2019-03-11.txt`). Three earlier versions exist:

| File | Date | Location |
|------|------|----------|
| `sensors-200c-12-05-04.txt` | Oct 2005 | `/home/chain1/input/magn90/` |
| `sensors_200c_15-09-04_3.txt` | Sep 2015 | `/home/MAGNW/chain1/input/PPFcfg/` |
| `sensors_200c_15-10-05.txt` | Oct 2015 | `/home/MAGNW/chain1/input/PPFcfg/` |

### Implementation

1. Add historical file paths to `jet.yaml` under `static_sources.sensor_calibration`
2. Extend `parse_mcfg_sensors.py` to accept a list of versioned sensor files
3. Parse each file identically (same format, sections: `#FLUX`, `#SADDLE`,
   `#PICK-UP`, `#HALL`, `#OTHER`)
4. Store as versioned SignalNode properties or separate SignalNodes per version
5. Create `SUPERSEDES` relationships between versions (2019 supersedes 2015
   supersedes 2005)

### Value

Enables tracking of sensor position refinements over 14 years. Useful for
validating EFIT reconstructions against the correct sensor geometry for a
given shot range. Low effort because the parser already exists — only the
multi-file input and versioning logic need adding.

---

## Phase 2: MCFG Calibration Epoch History

**Priority**: Low | **Effort**: Medium

The MCFG.ix index at `/home/MAGNW/chain1/input/magn_ep_2019-05-14/MCFG.ix`
tracks 77 calibration changes from pulse 54283 (2005) to 93563 (2019).

### Format

```
$:20190514 0093563 gszepesi MCFG:0001/MAGNW sensors_200c_2019-03-11.txt
$:20151214 0089433 gszepesi MTFC:0204/MAGNW TF comp after P802B -> P802A
$:20151009 0088368 gszepesi MCFG:0162/MAGNW Calib with sensors_200c_15-10-05
```

Fields: date, first_shot, user, config_type (MCFG/MCAL/MOFF/MTFC), description.

### Implementation

1. Add an MCFG.ix parser to `parse_mcfg_sensors.py` (or a separate
   `parse_mcfg_epochs.py`)
2. Store calibration epochs as properties on the MCFG DataSource node
   (simplest) or as related nodes (richer but requires schema addition)
3. Annotate sensor SignalNodes with applicable calibration epoch ranges

### Value

Tracks when sensor gains changed, probes were replaced (P803B→P804B),
or TF compensation was reconfigured. Important for understanding systematic
measurement shifts in historical JET data. However, the sensor R,Z positions
do not change between calibration epochs — only gains and offsets — so this
primarily affects quantitative accuracy rather than spatial mapping.

---

## Phase 3: PF Circuit JPF Addressing

**Priority**: Low | **Effort**: Small

The `cturns` file at `/home/chain1/input/pfcu/cturns` maps PF coil circuits
to JPF addresses. The PF coil scanner already ingests turns counts and creates
SAME_COMPONENT cross-refs to JEC2020 coils. What's missing is the JPF address
for real-time coil current data access.

### Implementation

1. Extend `parse_pf_coil_turns.py` to extract JPF addresses (already present
   in the parsed data, may just need forwarding)
2. Store `jpf_address` property on PF circuit SignalNodes
3. Create DataAccess templates for real-time current access via JPF

### Value

Enables code generation for reading real-time PF coil currents — useful for
equilibrium reconstruction validation and shot comparison tools. Small effort
because the parser and graph structure already exist.
