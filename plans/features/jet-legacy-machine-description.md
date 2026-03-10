# JET Legacy Machine Description — Full Lifecycle Ingestion Plan

## Problem Statement

The graph currently has **20 StructuralEpoch** nodes for JET, but the 6 pre-EFIT++ epochs
(shots 1–68612, spanning ~35 years from 1983 to 2007) have **zero DataNodes**. All machine
description data comes from EFIT++ device XMLs (shots 68613+), leaving the pre-EFIT++ era
completely empty. JEC2020 data is global (not epoch-specific).

## Status Quo (Graph Audit)

| Epoch Group | Shot Range | DataNodes | Source |
|-------------|-----------|-----------|--------|
| limiter_era | 1–28791 | 0 | none |
| mk1_era | 28792–30589 | 0 | none |
| mk2a_era | 30590–44414 | 0 | none |
| mk2gb_era | 44415–54351 | 0 | none |
| mk2gb_sr_era | 54352–63445 | 0 | none |
| mk2hd_pre | 63446–68612 | 0 | none |
| p68613–p90626 (14 epochs) | 68613–99999 | 277 each | device XML |
| JEC2020 (global) | all | 41 | JEC2020 XML |

## Discovered Data Sources on JET

### 1. Legacy Magnetics Config Files (`/home/chain1/input/magn/config/`)

**THE primary source for pre-EFIT++ sensor geometry.** The `indexr` file maps shot ranges
to specific probe configuration files, providing complete sensor definitions including
R, Z coordinates, angles, JPF names, calibration factors, and error bounds.

#### Shot Range → Config File Mapping

| Shots | Config File | Sensor Count | Era |
|-------|------------|-------------|-----|
| 1–27968 | `limves` | 122 (244 lines) | Pre-JET / early limiter |
| 27969–30945 | `pdvessri` | 304 (608 lines) | Mk1 initial |
| 30946–32044 | `pdvessra` | 302 (604 lines) | Mk2A initial |
| 32045–32112 | `pdvessrf` | 303 (606 lines) | Mk2A refined |
| 32113–32448 | `pdvessra` | 302 (604 lines) | Mk2A |
| 32449–35779 | `pdvessr` | 304 (608 lines) | Pre-Mk2GB |
| 35780–36134 | `pd96` | 327 (654 lines) | 1996 |
| 36135–40248 | `pd96e` | 327 (655 lines) | 1996 extended |
| 40249–44590 | `pd97` | 328 (657 lines) | 1997 |
| 44591–50310 | `pd97` | 328 (657 lines) | Mk2GB |
| 50311–54552 | `2000_01` | 324 (649 lines) | 2000 |
| 54553–999999 | `2002_01` | 326 (652 lines) | 2002+ (latest) |

#### File Format

Two-line records: JPF address + PPF signal name + index + calibration + R,Z,angle:
```
'DA/C2-CX01' 'MAGNBPOL  1'     1 0.0E0 0.0E0   4.2992   0.6052  -1.2933
 1.0 1.0 1.0 1.00 1.0 0 0 1.0 0
```
Fields: `jpf_address`, `ppf_signal`, `index`, `cal1`, `cal2`, `R`, `Z`, `angle`,
then `gain1 gain2 gain3 weight gain4 flag1 flag2 gain5 flag3`.

#### Sensor Types Present

| Type | Code | Description | Count (limves) | Count (2002_01) |
|------|------|-------------|----------------|-----------------|
| BPOL | MAGNBPOL | Magnetic probes (Bpol pick-ups) | 18 | 18 |
| FLUX | MAGNFLUX | Full flux loops (Oct 1/7) | 17 | 17 |
| SADX | MAGNSADX | Saddle loop cross-measurements | 14 | 28 |
| TPC | MAGNTPC | Divertor target probes | 0 | 11 |
| TNC | MAGNTNC | TN divertor coils | 0 | 11 |
| TS | MAGNTS | TS saddle loops | 0 | 13 |
| TSFL | MAGNTSFL | TS flux loops | 0 | 12 |
| TSRR | MAGNTSRR | TS RR measurements | 0 | 13 |
| PC | MAGNPC | P8xx probes | 0 | 7 |
| BSAD | MAGNBSAD | Broadband saddle loops | 0 | 16 |
| IC | MAGNIC | ICRF-area coils | 0 | 3 |
| DVC | MAGNDVC | Diamagnetic/vertical field coils | 4 | 4 |
| XPOL | MAGNXPOL | External poloidal coils | 10 | 0 |
| XNOR | MAGNXNOR | External normal coils | 6 | 0 |

**Key insight:** The limiter era (shots 1–27968) had only 18 probes + 17 flux loops +
14 saddle loops (Octant 1/7 original set). By 2002, the system grew to include divertor
probes, EP target coils, saddle loop enhancements, and various auxiliary sensors.

### 2. Sensors-200C Model Files (`/home/MAGNW/chain1/input/PPFcfg/`)

High-precision sensor geometry at 200°C operating temperature with CATIA reference
model coordinates. Multiple timestamped versions:

| File | Date | Description |
|------|------|-------------|
| `sensors-200c-12-05-04.txt` (in magn90) | Oct 2005 | Original CATIA model |
| `sensors_200c_15-09-04_3.txt` | Sep 2015 | |
| `sensors_200c_15-10-05.txt` | Oct 2015 | |
| `sensors_200c_2019-03-11.txt` | Mar 2019 | **Latest production** |

**Format:** Structured text with sections `#FLUX`, `#SADDLE`, `#PICK-UP`, `#HALL`, `#OTHER`:
```
#PICK-UP  R       Z   Angle          Factor  RelErr  AbsErr  JPF-Name    DESCRIPTION
  1   4.292   0.604   -74.1             1.0   0.020   0.005   C2-CX01    Internal Discrete Coil, Oct.3
```

**Contents (latest 2019 file):**
- 6 full flux loops with R,Z
- 60 saddle loops with R1,Z1,R2,Z2
- 180 pick-up coils with R,Z,Angle
- Hall probes (recent additions)

**This is the authoritative geometric reference** — the `pd*/pdvessr*` files derive
from an older (pre-CATIA) coordinate system. The sensors-200c files have higher precision.

### 3. MCFG Calibration Epoch Data (`/home/MAGNW/chain1/input/magn_ep/`)

Production calibration system (MCFG) with versioned sensor configurations.
The `MCFG.ix` index tracks every calibration change from 2006 to 2019:

```
$:20190514 0093563 gszepesi MCFG:0001/MAGNW sensors_200c_2019-03-11.txt
$:20151214 0089433 gszepesi MTFC:0204/MAGNW TF comp after P802B -> P802A
$:20151009 0088368 gszepesi MCFG:0162/MAGNW Calib with sensors_200c_15-10-05
```

Contains ~33 versioned config directories (`magn_ep_4_21` through `magn_ep_4_35.M41`).

### 4. PF Coil Data

**a) PFCU Coil Turns (`/home/chain1/input/pfcu/cturns`):**
12 PF coil entries with JPF addresses: P2SU (upper/lower inner/outer), P3SU/SL,
P4VU/VL, P2SR, P3SR — these are the **main plasma shaping coils** (P1-P4 system).

**b) JEC2020 PF Systems (`pfSystems.xml`)** — already ingested:
20 coils (P1×4 + D1-D4×4) with R,Z,dR,dZ, turns, circuits.

**c) EFIT++ device XMLs** — already ingested:
22 PF coil filaments per version (numbered 1-22), same physical coils as JEC2020.

**d) GREENS tables** — binary Fortran unformatted:
- `RFCOIL` — PF coil filament positions (binary, not text-parseable)
- `RECOIL` — passive structure filament positions (binary)
- `ECONTO` — vessel contour (binary)

The GREENS `green_list` maps shot ranges to GREENS directories, tracking DMSS mode changes:
```
1     63278 pre_JET_EP_DMSS_091_T_285C_nbpol_95/GREEN3333  ! pre-EFIT++, 285°C
63279 63297 pre_JET_EP_DMSS_102_T_285C_nbpol_95/GREEN3333
63298 63445 pre_JET_EP_DMSS_091_T_285C_nbpol_95/GREEN3333
63446 66470 JET_EP_DMSS_091_T_200C_nbpol_95/GREEN3333      ! EFIT++, 200°C
```

### 5. Vessel Forces / Passive Structure Data (`/home/chain1/input/vforce/`)

- `vvlabels` — 32 vessel sector labels (TL/TR/BL/BR × 8 octants)
- `confa01` — Force sensor configuration with JPF GS/MD-* addresses (shot-dependent from 45051)
- `transducer_report` — Sensor fault tracking

### 6. FLUSH Database (`/usr/local/data/flush/`) — Already Ingested

7 wall contour files (L91NEW through mark2ilw), `limiter_list.dat` mapping shot ranges,
`tokamak.dat` with basic parameters (R0≈2.96m).

### 7. EFIT Legacy Limiters (`/home/chain1/input/efit/Limiters/`)

4 limiter files: mk2a, mk2gb, mk2gb_ns, mk2hd. Same R,Z format as FLUSH.
`limiter_list` maps shots:
```
1     44414 limiter.mk2a    (retroactively maps pre-Mk2A to Mk2A)
44415 54351 limiter.mk2gb
54352 63445 limiter.mk2gb_ns
63446 99999 limiter.mk2hd
```

### 8. Fundamental Machine Parameters

From `tokamakData.xml` (JEC2020) and `tokamak.dat` (FLUSH):
- R₀ = 2.96 m (major radius)
- a ≈ 1.25 m (minor radius, from FLUSH par1=1.96 → innermost R)
- B₀×R₀ reading: `bVacRadiusProduct`, `scalingFactor="2.96"`
- Iron core B-H curve (27-point mu table in `tokamakData.xml`)
- Grid boundaries: R=[1.65, 4.05], Z=[-1.90, 2.15] (from `mhdin`)

---

## Ingestion Plan

### Phase 1: Legacy Magnetics Sensors (HIGH PRIORITY)

**Goal:** Fill pre-EFIT++ epochs with magnetic probe, flux loop, and saddle loop data
using the `magn/config/` files.

**New data source:** `legacy_magnetics`

**Implementation:**

1. **Remote parsing script** (`imas_codex/remote/scripts/parse_legacy_magnetics.py`):
   - Read `/home/chain1/input/magn/config/indexr` to get shot-range → file mapping
   - Parse each config file (limves, pdvessri, pdvessra, pdvessrf, pdvessr, pd96, pd96e, pd97, 2000_01, 2002_01)
   - Extract sensor records: JPF address, PPF signal, index, R, Z, angle, gains, weight
   - Return JSON: `{version: {probes: [...], flux_loops: [...], saddle_loops: [...], ...}}`

2. **Scanner handler** in `device_xml.py` (new function `_persist_legacy_magnetics_nodes()`):
   - Map legacy config shot ranges to existing StructuralEpoch boundaries
   - Create DataNodes for each sensor per epoch (respecting sensor availability changes)
   - Create FacilitySignals for each unique sensor
   - Link DataNodes to appropriate StructuralEpoch via `INTRODUCED_IN`

3. **Config** (`jet.yaml`):
   - Add `data_systems.legacy_magnetics` section:
     ```yaml
     legacy_magnetics:
       index_file: /home/chain1/input/magn/config/indexr
       config_dir: /home/chain1/input/magn/config
       high_precision_sensors: /home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt
     ```

4. **Epoch mapping** — map legacy config boundaries to existing epochs:
   | Legacy Config | Shots | Maps To Epoch |
   |--------------|-------|---------------|
   | limves | 1–27968 | limiter_era (1–28791) |
   | pdvessri | 27969–30945 | mk1_era (28792–30589) |
   | pdvessra | 30946–32448 | mk2a_era (30590–44414) |
   | pdvessr | 32449–35779 | mk2a_era |
   | pd96/pd96e/pd97 | 35780–44590 | mk2a_era |
   | pd97 | 44591–50310 | mk2gb_era (44415–54351) |
   | 2000_01 | 50311–54552 | mk2gb_sr_era (54352–63445) |
   | 2002_01 | 54553–68612 | mk2hd_pre (63446–68612) |

**Expected yield:** ~300 DataNodes per epoch × 6 epochs ≈ **1800 new DataNodes**

### Phase 2: Sensors-200C High-Precision Model (MEDIUM PRIORITY)

**Goal:** Ingest the authoritative CATIA-derived sensor positions with full error bounds.

**Implementation:**

1. **Remote parsing script** (`parse_sensors_200c.py`):
   - Parse structured text format (#FLUX, #SADDLE, #PICK-UP, #HALL sections)
   - Extract R, Z, Angle, Factor, RelErr, AbsErr, JPF name, description
   - Return per-section arrays

2. **Scanner handler** (`_persist_sensors_200c_nodes()`):
   - Create a `sensors_200c` DataSource
   - Create DataNodes for 6 flux + 60 saddle + 180 pick-up + Hall probes
   - Cross-reference with legacy magnetics and device XML probes via JPF name matching

3. **Config:**
   ```yaml
   data_systems:
     sensors_200c:
       files:
         - path: /home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt
           version: "2019-03-11"
         - path: /home/chain1/input/magn90/sensors-200c-12-05-04.txt
           version: "2005-10-04"
   ```

**Expected yield:** **246 high-precision sensor nodes** with error bounds

### Phase 3: MCFG Calibration History (LOW PRIORITY)

**Goal:** Track sensor calibration changes and probe enable/disable events.

**Implementation:**

1. Parse `MCFG.ix` for calibration epoch entries
2. Map MCFG calibration shots to StructuralEpoch
3. Store calibration metadata on DataNode properties
4. Track probe failures (P803B, P804B, P805B replacements)

**Expected yield:** ~30 calibration epoch annotations

### Phase 4: PF Coil Turns / Circuit Data (MEDIUM PRIORITY)

**Goal:** Add PF coil circuit information for all epochs.

**Implementation:**

1. Parse `cturns` file for coil-to-circuit JPF mapping
2. Cross-reference with existing JEC2020 PF coils
3. Create `PF_CURRENT_SOURCE` DataNodes linking JPF addresses for real-time
   coil current monitoring

**Expected yield:** 12 PF circuit DataNodes with JPF addresses

### Phase 5: GREENS Table Metadata (LOW PRIORITY)

**Goal:** Capture Green's function table versioning for provenance tracking.

**Implementation:**

1. Parse `green_list` for GREENS version → shot range mapping
2. Store DMSS mode and temperature as StructuralEpoch properties
3. No binary parsing required — just metadata tracking

**Expected yield:** 11 GREENS version annotations on epochs

---

## Implementation Priority Order

1. **Phase 1: Legacy Magnetics** — biggest impact, fills 6 empty epochs with ~1800 nodes
2. **Phase 4: PF Coil Turns** — small but important for circuit completeness
3. **Phase 2: Sensors-200C** — high-precision reference geometry
4. **Phase 5: GREENS Metadata** — provenance tracking
5. **Phase 3: MCFG History** — calibration change tracking

## Architecture Decision

**Extend the existing `device_xml.py` scanner** rather than creating a new scanner.
The legacy magnetics data is part of the same machine description domain. Add new
handler functions (`_persist_legacy_magnetics_nodes`, `_persist_sensors_200c_nodes`)
following the same pattern as `_persist_jec2020_nodes` and `_persist_mcfg_nodes`.

The `DeviceXMLScanner.scan()` method already orchestrates multiple data sources
(device XML, JEC2020, MCFG, PPF static). Adding legacy magnetics and sensors-200c
fits naturally into this pattern.

## New Remote Scripts Needed

1. `parse_legacy_magnetics.py` — parse `indexr` + all config files
2. `parse_sensors_200c.py` — parse structured sensor model files

Both scripts must be Python 3.12+ (they run via `run_python_script()` through
the venv interpreter).

## Schema Changes Required

None — existing `DataNode`, `FacilitySignal`, `StructuralEpoch` schemas are sufficient.
The legacy data maps cleanly to existing properties:
- DataNode: `path`, `r`, `z`, `angle`, `system`, `first_shot`, `last_shot`
- FacilitySignal: `jpf_signal`, `ppf_signal` (already supported)
- StructuralEpoch: `first_shot`, `last_shot`, `pf_configuration` (already present)

## Config Changes Required

Add to `jet.yaml`:
```yaml
data_systems:
  legacy_magnetics:
    index_file: /home/chain1/input/magn/config/indexr
    config_dir: /home/chain1/input/magn/config
    supplementary_dir: /home/chain1/input/magn90/config
  sensors_200c:
    production_file: /home/MAGNW/chain1/input/PPFcfg/sensors_200c_2019-03-11.txt
    historical_file: /home/chain1/input/magn90/sensors-200c-12-05-04.txt
  pfcu:
    cturns_file: /home/chain1/input/pfcu/cturns
```
