# Signal Definition Strategy

## Current Status (June 2025)

### Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Wiki Signal Tables | **Not started** | WikiChunk nodes exist in graph (8,369 chunks), but no wiki→signal extraction implemented |
| Phase 2: TDI Source Mapping | **Complete (TCV)** | 375 TDI signals discovered, 253 enriched+checked, 73 failed (shot-dependent). Cost: $0.09 |
| Phase 3: Runtime Validation | **Working** | check_worker validates via `tdiExecute()`. All bugs fixed (JSON contamination, stdout pollution) |
| Phase 4: Signal ID Schema | **Implemented** | Format: `facility:physics_domain/signal_name` |
| Phase 5: IMAS Mapping | **Not started** | No `tcv_get_ids_*.m` parsing |
| Multi-facility architecture | **Designed** | Plugin-based data source scanners |

### Graph State (post-pipeline run)

- **253 FacilitySignal nodes**: 180 checked + 73 failed (shot-dependent signals like staticgreen, staticwinding)
- 21 TDIFunction nodes with source code cached
- 7 DataAccess nodes: `tcv:tdi:functions`, `tcv:mdsplus:tree_tdi`, `tcv:matlab:mdsplus`, `jet:sal:ppf`, `jet:matlab:ppf`, `jet:idl:ppf`, `jet:cli:getdat`
- 564 WikiPage nodes, 8,369 WikiChunk nodes (TCV wiki)
- Physics domain distribution: equilibrium (127), general (55), magnetic_field_diagnostics (44), EM wave diagnostics (24), edge plasma (15), transport (11), machine ops (10)

### Bugs Fixed

1. **JSON contamination in `run_python_script`**: stderr was appended to stdout, corrupting JSON output from remote scripts when MDSplus/TDI functions printed warnings (e.g., libvaccess.so loading errors). Fix: stderr is now logged separately, never mixed into stdout.

2. **stdout pollution from TDI functions**: Functions like `tile_store` print debug output and try to load hardware libraries (`libvaccess.so`) when executed via `tdiExecute()`. Fix: `check_signals_batch.py` now captures stdout/stderr during TDI execution using fd-level redirection.

3. **Operational functions in signal catalog**: Hardware control functions (`tile_store`, `beckhoff_setstate`, `acq_lente_trigger`, etc.) were being discovered as physics signals. Fix: `OPERATIONAL_TDI_FUNCTIONS` exclusion set in `tdi.py`. 69 stale signals cleaned from graph.

4. **Duplicated LLM infrastructure**: `enrich_worker` reimplemented retry logic, API key management, noise suppression, and cost tracking. Fix: refactored to use `acall_llm_structured()` from `discovery.base.llm`.

### Architecture

The data discovery module (`imas_codex/discovery/data/`) uses the same parallel worker pattern as paths/wiki discovery:

```
scan_worker → enrich_worker → check_worker
(SSH/TDI)     (LLM)           (SSH/MDSplus)
```

Workers coordinate via `claimed_at` timestamps in Neo4j. Status transitions: `discovered → enriched → checked | failed`.

Base infrastructure from `imas_codex/discovery/base/` now shared:
- `acall_llm_structured` — LLM calls with retry, cost tracking, structured output
- `SupervisedWorkerGroup` / `supervised_worker` — crash recovery with backoff
- `WorkerStats` — rate/cost tracking
- `get_facility()` — facility config resolution

## Context

TCV experimental data is accessed through a layered architecture. The wiki (564 pages, 8,369 chunks) documents this extensively.

### Data Access Layers

| Layer | Interface | Language | Example |
|-------|-----------|----------|---------|
| Raw MDSplus | Node paths | Any | `\results::i_p`, `\magnetics::iplasma:trapeze` |
| TDI functions | `.fun` files | TDI | `tcv_get('IP')`, `tcv_eq('KAPPA')`, `fir_aut(shot)` |
| Matlab wrappers | `tcvget.m`, `gdat.m` | Matlab | `tcvget('KAPPA')`, `gdat(shot,'ids','source','summary')` |
| IMAS bridge | `tcv_get_ids_*.m` | Matlab | `tcv2ids2database(shot,run,'ids_names',{'equilibrium'})` |

### TDI Function Inventory (213 .fun files)

| Type | Count | Examples | Characteristics |
|------|-------|----------|-----------------|
| Dispatch | 5 | `tcv_get`, `tcv_eq`, `tcv_psitbx` | `case()` blocks selecting among named quantities |
| Direct | 184 | `fir_aut`, `static`, `magproc` | One function = one signal (or signal bundle) |
| Parametric | 21 | `ts_fitdata`, `ts_rawdata` | Needs argument to select sub-signal |
| Internal | 3 | `_fakir` | Helper functions, not user-facing |

### MDSplus Trees at TCV

| Tree | Signals | Content |
|------|---------|---------|
| tcv_shot | 65,169 | Main shot tree, subtrees for diagnostics |
| atlas | 35,333 | High-speed acquisition (DTACQ) |
| results | 8,198 | Analysis outputs (LIUQE, FIR, proffit, CXRS, ASTRA) |
| magnetics | 514 | Ip, flux loops, Bpol probes |
| base | 1,200 | Photodiodes, basic diagnostics |
| power | — | NBI, ECRH power systems |
| manual | — | Manually-entered metadata (gas species, LH transition times) |
| vsystem | — | Vessel/plant control |
| pcs | — | Plasma control system (MGAMS, FBTE, RT-LIUQE) |

### Key Wiki Pages (Ground Truth)

These pages contain curated signal→path→units→description mappings:

| Page | Content |
|------|---------|
| [Tcvget](https://spcwiki.epfl.ch/wiki/Tcvget) | Complete catalog of `tcvget.m` signals with MDSplus paths, units, descriptions |
| [Frequently used results nodes](https://spcwiki.epfl.ch/wiki/Frequently_used_result_nodes) | Table: path, tree, fill-mode, units, description |
| [Mds trees vms results](https://spcwiki.epfl.ch/wiki/Mds_trees_vms_results) | All RESULTS tree nodes with descriptions and units |
| [Mds trees vms static](https://spcwiki.epfl.ch/wiki/Mds_trees_vms_static) | STATIC tree: geometry, Green's functions, coil data |
| [LIUQE nodes/Update](https://spcwiki.epfl.ch/wiki/LIUQE_nodes/Update) | Equilibrium quantities: names, descriptions, defaults |
| [FBT nodes](https://spcwiki.epfl.ch/wiki/FBT_nodes) | Free-boundary equilibrium under `\PCS::MGAMS.FBTE` with units |
| [Proffit nodes](https://spcwiki.epfl.ch/wiki/Proffit_nodes) | Te/Ne profile fitting results |
| [Chie TCV nodes](https://spcwiki.epfl.ch/wiki/Chie_tcv_to_nodes) | Transport analysis under `\results::conf` |
| [CXRS/Software/MDS nodes](https://spcwiki.epfl.ch/wiki/CXRS/MDS_nodes) | Ti, Vi, Ni, Zeff profiles |
| [FIR MDS nodes](https://spcwiki.epfl.ch/wiki/FIR_MDS_nodes) | Interferometry: `\DIAGZ::FIR`, `\RESULTS::FIR`, `\ATLAS::DT4G` |
| [DIAGZ mds nodes](https://spcwiki.epfl.ch/wiki/DIAGZ_mds_nodes) | Hardware diagnostic nodes (large, last updated 2017) |
| [IMAS](https://spcwiki.epfl.ch/wiki/IMAS) | TCV→IMAS mapping via `tcv_get_ids_*.m` functions |

### Critical Technical Findings

1. **MDSplus nodes have units** — Earlier tests returned blank because of shell escaping issues with backslash paths. With correct escaping (`chr(92)` in Python), `node.units` returns real values: `particles/m^3`, `A`, `V`.

2. **`units_of()` works reliably for `tcv_eq` quantities** — Returns TreePath objects that preserve units (`A`, `Wb`, `Pa`, `m`, `m^2`, `m^3`). `tcv_get` constructs Signal objects that lose units metadata.

3. **Source code contains units via `make_with_units()`** — Functions like `fir_aut` explicitly set units: `"particles/m^3"`, `"seconds"`, `"fringes"`.

4. **Cross-tree paths work from `tcv_shot`** — `tree.getNode('\results::fir:n_average')` resolves correctly and returns units. Many paths from source code use runtime templates that don't exist as literal nodes (NNF errors for those).

5. **`gdat.m` already has `tcv_get_ids_*` for IMAS** — Covers: `core_profiles`, `equilibrium`, `nbi`, `summary`, `thomson_scattering`, `ec_launchers`, `magnetics`, `pf_active`, `tf`, `wall`. These contain the canonical TCV→IMAS mappings.

## Recommended Strategy

### Principle: Wiki-First, Runtime-Validated, Source-Augmented

The wiki pages are the **primary source of signal definitions** at TCV. They are curated by domain experts — the same people who wrote the TDI functions and analysis codes. Rather than reverse-engineering metadata from TDI source alone, we should combine:

1. **Wiki content** (already embedded in graph) — descriptions, units, paths
2. **TDI source parsing** — function→quantity→backing MDSplus path mapping
3. **Runtime probing** — units via `units_of()`, shape/type validation

### Phase 1: Extract Wiki Signal Tables

The Tcvget page and Frequently Used Results Nodes page contain structured tables mapping signal names to MDSplus paths, units, and descriptions. These are already in the graph as WikiChunk nodes with extracted `mdsplus_paths_mentioned` and `units_mentioned`.

**Action:** Parse these wiki chunks into structured signal records. The chunks already have MDSplus paths and units extracted. Cross-reference with the TDI source function mapping to link each signal to its TDI accessor.

Example wiki-to-signal extraction (from Tcvget chunks already in graph):

```
IP → \magnetics::iplasma:trapeze, A, "Plasma current from trapeze integration"
KAPPA → \results::kappa_edge, -, "Elongation of LCFS from LIUQE"
NEL → \results::fir:n_average, m^-3, "Line averaged density"
POHM → \results::surface_flux × IP, W, "Ohmic power from LIUQE"
RGEO → \results::r_contour, m, "Geometric major radius"
Q95 → \results::q_95, -, "Safety factor at 95% flux surface"
```

### Phase 2: TDI Source Mapping

Map every TDI function to its access pattern using source parsing (already working — 348 signals from 213 functions). This gives:

- **Dispatch functions**: quantity name → `build_path()` MDSplus path
- **Direct functions**: function name → one signal (or signal bundle)  
- **Parametric functions**: function name + arg → specific sub-signal

Key output: a function-to-path table linking `tcv_get('IP')` to `\magnetics::iplasma:trapeze`.

### Phase 3: Runtime Metadata Validation

For signals where wiki/source don't provide units, probe at runtime:

```python
# tcv_eq quantities (reliable — returns TreePath with units)
tree.tdiExecute("units_of(tcv_eq('I_P'))").data()  # → "A"

# Backing MDSplus nodes (reliable if path exists)
node = tree.getNode('\results::fir:n_average')
node.units  # → "particles/m^3"
```

`tcv_get` via `units_of()` returns blank (Signal wrapper strips units). Use source code `build_path()` reference and query the backing node directly instead.

### Phase 4: Signal ID Schema

Signals must include function context to avoid collisions:

```
tcv:tcv_get/IP          — plasma current via tcv_get
tcv:tcv_eq/I_P          — plasma current via tcv_eq (LIUQE)  
tcv:tcv_get/KAPPA       — elongation via tcv_get
tcv:fir_aut             — FIR automated analysis (direct accessor)
tcv:ts_fitdata/TE       — fitted Te from Thomson
tcv:ts_fitdata/NE       — fitted Ne from Thomson
```

### Phase 5: IMAS Mapping via `tcv_get_ids_*`

The existing Matlab `tcv_get_ids_*.m` functions contain the authoritative TCV→IMAS mappings. These are under `/home/matlab/crpptbx-*/gdat/TCV_IMAS/` on the LAC machines.

**Action:** Read these Matlab files via SSH, parse the signal-to-IDS-path assignments, and store as MAPS_TO_IMAS relationships in the graph. This gives us the full chain:

```
TDI function → MDSplus path → FacilitySignal → MAPS_TO_IMAS → IMASPath
```

Available IDS mappings: `core_profiles`, `equilibrium`, `nbi`, `summary`, `thomson_scattering`, `ec_launchers`, `magnetics`, `pf_active`, `tf`, `wall`.

### What NOT to Do

- **Don't use LLM for units propagation** — units come from wiki tables, `node.units`, `units_of()`, and `make_with_units()` in source. All programmatic.
- **Don't filter out geometry/non-physics signals** — the STATIC tree and `\results::conf:*` nodes contain valuable machine geometry and confinement data.
- **Don't embed descriptions yet** — focus on building the complete signal catalog with correct metadata first.
- **Don't rely solely on TDI runtime** — `tcv_get` loses units in its Signal wrapper. Wiki + source + backing node is more reliable.

## Metadata Sources by Priority

| Source | Descriptions | Units | MDSplus Path | Reliability |
|--------|-------------|-------|--------------|-------------|
| Wiki (Tcvget, node pages) | ✓ excellent | ✓ good | ✓ explicit | Curated by experts |
| TDI source code | partial (comments) | ✓ via `make_with_units` | ✓ via `build_path` | Authoritative for code paths |
| MDSplus node attributes | ✗ blank at TCV | ✓ where set | n/a | Hit-or-miss (many nodes have units, descriptions always blank) |
| `units_of()` runtime | ✗ | ✓ for `tcv_eq` | ✗ | Reliable for TreePath results only |
| `tcv_get_ids_*.m` | ✗ | ✗ | ✗ | Has IMAS path mappings only |

## Multi-Facility Architecture

### Data Access Methods by Facility

| Facility | Primary | Secondary | Graph State |
|----------|---------|-----------|-------------|
| **TCV** | MDSplus + TDI functions | Matlab gdat/tcvget, IMAS bridge | 253 signals, 2 DataAccess |
| **JET** | PPF (SAL REST), PPF (Matlab/IDL), getdat CLI | IMAS, MDSplus (limited) | 0 signals, 4 DataAccess |
| **JT-60SA** | EDAS (C/Fortran/Python wrappers) | IMAS (modules), MDSplus, UDA | 0 signals, 0 DataAccess |
| **ITER** | IMAS (native) | MDSplus | 0 signals, 0 DataAccess |

### Plugin Architecture

The scan_worker currently handles only TDI function extraction. For multi-facility support, the scan phase needs **data source plugins** — each facility config declares which plugins to use via `data_sources`:

```yaml
# tcv.yaml
data_sources:
  tdi:
    primary_path: /usr/local/CRPP/tdi/tcv
    reference_shot: 85000

# jet.yaml (future)
data_sources:
  ppf:
    sal_endpoint: https://sal.jet.uk/rest/ppf
    reference_pulse: 99999
  imas:
    db_name: jet
    reference_pulse: 99999

# jt-60sa.yaml (future)
data_sources:
  edas:
    api_library: /analysis/src/edas
    reference_shot: 1000
  imas:
    module_name: imas/3.x
    reference_shot: 1000

# iter.yaml (future)
data_sources:
  imas:
    db_name: iter
    reference_shot: 100000
```

### Scanner Interface

Each data source type needs a scanner that produces `FacilitySignal` nodes with a common schema:

```python
class DataSourceScanner(Protocol):
    """Interface for data source-specific signal discovery."""

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict,            # data_sources.{type} config
        reference_shot: int,
    ) -> list[FacilitySignal]:
        """Discover signals from this data source."""
        ...

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        reference_shot: int,
    ) -> list[dict]:
        """Validate signals return data for reference shot."""
        ...
```

**Existing scanners to extract:**
- `TDIScanner` — current scan_worker logic (TCV)
- `PPFScanner` — JET PPF catalog via SAL REST API
- `EDASScanner` — JT-60SA EDAS signal catalog (parse C header files)
- `IMASScanner` — IMAS IDS signal enumeration

**Shared across all scanners:**
- Enrichment (enrich_worker) — LLM physics domain classification is facility-agnostic
- Signal ID schema — `facility:physics_domain/signal_name`
- Graph persistence — FacilitySignal, DataAccess, MAPS_TO_IMAS relationships

### Implementation Sequence

1. **Extract TDI scanner** from scan_worker into `discovery/data/scanners/tdi.py`
2. **Add `data_sources` config** to JET, JT-60SA, ITER facility YAMLs
3. **Implement PPF scanner** for JET (SAL REST API is most accessible)
4. **Implement EDAS scanner** for JT-60SA
5. **Wire scanner dispatch** in scan_worker based on `data_sources` config keys

### JET Signal Discovery Plan

JET has 4 DataAccess methods already in the graph. The SAL REST API is the most accessible:

```
GET https://sal.jet.uk/rest/ppf/{pulse}/{uid}/{dda}/{dtype}
```

PPF is organized: Source → DDA (Diagnostic Data Area) → Node. Key DDAs:
- `EFIT`: Equilibrium (R, Z, Psi, q, p, j profiles)
- `KG1V`: FIR interferometer (ne line-averaged)
- `KK3A`: ECE electron temperature
- `HRTS`: Thomson scattering
- `CXFM`: CXRS ion temperature/rotation
- `BOLO`: Bolometry
- `NB`: Neutral beam power
- `NBR`: ICRF heating power

**Step 1:** Query SAL catalog for all DDAs with signal metadata.
**Step 2:** Enrich with LLM (same enrich_worker).
**Step 3:** Validate data retrieval for a reference pulse.

## Development Direction

### Immediate (this sprint)

1. **Wiki signal table extraction** (Phase 1) — parse Tcvget and Frequently Used Results Nodes wiki chunks to create authoritative signal records with descriptions, units, MDSplus paths. Cross-reference with discovered TDI signals.
2. **Re-run with all signals** — current run has 73 failed signals from shot-dependent functions. Try different reference shots or mark as conditional.
3. **IMAS mapping** (Phase 5) — SSH to TCV, read `tcv_get_ids_*.m` files, parse signal→IDS path mappings, store as `MAPS_TO_IMAS` graph relationships.

### Short-term (next 2 sprints)

4. **Extract TDI scanner** into plugin structure for multi-facility support.
5. **JET PPF discovery** — implement PPF scanner using SAL REST API. JET already has 4 DataAccess nodes in graph.
6. **Signal embeddings** — once catalog is stable, embed signal descriptions for semantic search.

### Medium-term

7. **JT-60SA EDAS discovery** — EDAS has C header files with signal definitions.
8. **Cross-facility signal alignment** — link equivalent signals across facilities (e.g., TCV `tcv_get('IP')` ↔ JET `PPF/MAGN/IPLA` ↔ IMAS `magnetics/ip_plasma`).
9. **Matlab wrapper parsing** — extract `gdat.m`/`tcvget.m` signal catalogs for TCV.

## Open Questions

1. **LAC machine access** — Can we SSH to a LAC (lac911.epfl.ch) to read `tcv_get_ids_*.m` files, or do we need to find them via another route?
2. **Parametric function enumeration** — Functions like `ts_fitdata(quantity)` accept arguments. How do we enumerate all valid arguments? Source parsing of the called function's `case()` blocks is one approach.
3. **Shot-dependent nodes** — Some RESULTS nodes (proffit, ASTRA, chie) are written per-trial-index. Do we represent each trial as a separate signal, or note the trial dimension?
4. **CONF subtree** — `\results::conf:*` nodes exist but queries returned NNF for shot 85000. These are written by `chie_tcv_to_nodes` and may not be filled for all shots.
5. **JET SAL access** — Is the SAL REST API accessible from the JET login nodes, or does it require a specific network route?
6. **JT-60SA EDAS headers** — Where are the EDAS signal definition header files on the analysis servers?
