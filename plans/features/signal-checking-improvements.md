# Signal Checking Improvements Plan

## Investigation Summary

### Root Causes Identified

#### 1. Static Tree Not Accessible via tcv_shot (8,538 NNF + 25,464 unchecked)

**Root cause**: The signal checker routes ALL `tree_traversal` signals through `tcv_shot` (line ~3306 in `parallel.py`):

```python
if node_path and signal.get("discovery_source") == "tree_traversal":
    accessor = node_path
    tree_name = "tcv_shot"
```

But **static is NOT a subtree of tcv_shot**. The tcv_shot tree has 12 subtrees: ATLAS, BASE, DIAGZ, DIAG_ACT, ECRH, HYBRID, MAGNETICS, MANUAL, NOTIFY, POWER, RESULTS, VSYSTEM. Static is a separate, independent tree.

When the checker opens `Tree("tcv_shot", 85000)` and calls `tdiExecute("\STATIC::ANG_A")`, it returns NNF because the static tree context isn't available.

**Additionally**, even when opened correctly (`Tree("static", 1)`), most static nodes are expression nodes:
- `\STATIC::ANG_A` has record: `Build_Param(Build_Signal(Build_With_Units(\STATIC::ACTIVE_COIL:VAL:ANG, "rad"), *, \DIM_A), ...)` 
- These expressions reference TDI functions like `staticgreen()` that require `libblas.so` (missing)
- Out of 47,976 static nodes, ~34,082 are NUMERIC/SIGNAL. Of the first 200 checked, only 1 had actual stored data
- The static tree is a **computational recipe** for machine geometry, not a data store

**Impact**: 34,002 total static signals (100% of static tree signals), all failing

#### 2. Missing Shared Libraries (libjmmshr_gsl.so, libblas.so)

Static tree GREENS computation requires TWO missing libraries:

- **`libjmmshr_gsl.so`** — Custom Green's function computation library used by `greenem.fun` (calls `jmmshr_gsl->greenem()`). NOT installed anywhere on the system. This is the primary blocker for GREENS data.
- **`libblas.so`** — BLAS matrix multiply used by `_matmul.fun` (calls `blas->sgemm_()`). System has `libblas.so.3` at `/usr/lib64/` but the unversioned symlink `libblas.so` is missing. MDSplus TDI dlopen resolves `blas` → `libblas.so`.
- **`libvaccess.so`** — Old VMS access library, only referenced in deprecated `HIDE_2011/` functions. Not a current blocker.

The TDI shared library call syntax (`libname->function()`) resolves to `dlopen("lib<libname>.so")`. Without the exact filename, the call silently fails and returns `$ROPRAND`, which cascades to TreeNODATA.

**Verified via SSH testing**: Even with `libblas.so` symlink created in user dir + LD_LIBRARY_PATH, GREENS still fail because `libjmmshr_gsl.so` is the deeper dependency. BLAS alone isn't sufficient.

**Impact**: All GREENS expression nodes (702 systems × multiple functions each). MECHANICAL geometry data is partially accessible without these libraries (stored data nodes work fine).

#### 2b. Static Tree Data Reality (New Finding)

Deep scan at shot=1 reveals the static tree has a clear data classification:

| Subtree | Stored Data | Expression Nodes | Empty | Notes |
|---------|------------|-----------------|-------|-------|
| MECHANICAL | 135 | 567 | 54 | Geometry: COIL R/Z shape=(32,), MESH R/Z shape=(462+,) |
| GREENS | 0 | ~all | 0 | 702 child systems, all computed via greenem/staticgreen |
| VERSION | 0 | 0 | 1 | Structure node only |

Key findings:
- MECHANICAL stored data **IS readable** at shot=1: coil positions, mesh coordinates, transformation matrices
- MECHANICAL expressions reference `:VAL:X` sub-nodes; many fail with NNF due to version-dependent availability
- GREENS data is **fully computed on-demand** — no stored values, no pre-calculated cache (`:PRE` nodes are empty)
- Even with all libraries installed, GREENS would compute transiently (expensive, not suitable for batch checking)

#### 2c. DataAccess Node Gap (New Finding)

Only 2 DataAccess nodes exist for TCV, both incomplete:

| DataAccess ID | Signals Linked | Has Templates | Has Setup | Issue |
|--------------|---------------|--------------|-----------|-------|
| `tcv:mdsplus:tree_tdi` | 60,256 | No | No | Missing connection/data templates, no env setup |
| `tcv:tdi:functions` | 338 | Yes | No | Has templates but no setup_commands or env vars |

**Critical gap**: ALL 34,002 static tree signals point to `tcv:mdsplus:tree_tdi` which encodes tcv_shot access. The DataAccess node doesn't distinguish between tcv_shot subtrees and independent trees like static. This is why `check_worker()` routes everything through tcv_shot — the DataAccess node doesn't encode the correct tree name.

**Root cause in code**: `graph_ops.py:promote_leaf_nodes_to_signals()` hardcodes `da_id = f"{facility}:mdsplus:tree_tdi"` for ALL trees, regardless of whether the tree is a tcv_shot subtree or independent.

#### 3. Dynamic Tree no_data (5,701 failures)

Expected behavior. Many nodes defined in tree structure don't have data stored for shot 85000. These are legitimately empty nodes (e.g., diagnostic not active for that particular shot, or data not yet analyzed).

**Distribution**: results=2,473, hybrid=2,638, diagz=117, ecrh=113, base=67, diag_act=64, magnetics=36, tcv_shot=157, manual=19

#### 4. tcv_shot NNF from epoch_detection (334 failures)

Epoch-detected signals reference nodes across different subtrees (POWER, MAGNETICS, BASE, PCS, VSYSTEM) with qualified paths like `\POWER::TRCF_REDE_002_AC:ACT_CHANNELS`. These are hardware/acquisition nodes that may not exist at shot 85000 or may have been removed between tree versions.

#### 5. TDI Function Check Results

338 TDI function signals checked: 118 success (35%), 220 failed (65%).

| Function | Success | Failed | Total | Success Rate |
|----------|---------|--------|-------|-------------|
| tcv_get | 78 | 33 | 111 | 70% |
| tcv_eq | 18 | 15 | 33 | 55% |
| magnetics_dependencies | 6 | 0 | 6 | 100% |
| tcv_psitbx | 4 | 2 | 6 | 67% |
| Others (172 functions) | 12 | 170 | 182 | 7% |

Most single-use TDI functions are hardware control, shot management, or specialized operations that don't return data when called without proper arguments.

### Current Signal Check Landscape (TCV)

| Tree | Checked-Success | Checked-Failed | Discovered | Skipped | Primary Failure |
|------|----------------|---------------|------------|---------|-----------------|
| static | 0 | 8,538 (100% NNF) | 25,464 | - | Not a tcv_shot subtree |
| hybrid | 9,186 | 2,685 | - | - | no_data (78%) |
| results | 1,719 | 2,476 | - | - | no_data (99%) |
| diagz | 1,661 | 133 | - | - | no_data (88%) |
| base | 839 | 86 | - | - | no_data (78%) |
| tcv_shot | 484 | 499 | - | - | NNF (67%) |
| ecrh | 379 | 189 | - | - | no_data (60%) |
| magnetics | 297 | 58 | - | - | no_data (62%) |
| power | 120 | 0 | - | - | - |
| TDI | 118 | 220 | - | - | various |
| vsystem | 0 | 0 | 561 | - | Not yet checked |

### TDI Function Inventory

213 `.fun` files in `/usr/local/CRPP/tdi/tcv/`, organized into categories:

**Core Physics Accessors** (return physics data):
- `tcv_get` — Main multi-purpose data accessor
- `tcv_eq` / `tcv_eq2` / `tcv_eq_profile` — Equilibrium reconstruction (LIUQE)
- `tcv_psitbx` — Psi toolbox
- `tcv_ip` — Plasma current
- `tcv_bphi` — Toroidal field
- `tcv_betadml` — Beta
- `fir_*` (6 functions) — Interferometry
- `ts_*` (7 functions) — Thomson scattering
- `li_1`, `li_ga` — Internal inductance
- `static` / `staticgreen` / `staticwinding` — Machine geometry

**Hardware/Control** (not physics data, should be excluded):
- `dt100_*`, `check_*`, `beckhoff_*`, `wavegen_*` — Hardware control
- `tile_*`, `store_*`, `init_*` — Action dispatchers
- `test_*` — Test functions

**Additional TDI directories** (beyond `/usr/local/CRPP/tdi/tcv/`):
- `base/` — 122 .fun files (utility functions)
- `bitbus/` — 96 .fun files (bus communication)
- `trees/` — 43 .fun files (tree management)
- `vsystem/` — 31 .fun files (Vista system)
- `event/` — 36 .fun files (event handling)
- `devices/` — 22 .fun files (device drivers)
- `anasrv/` — 20 .fun files (analysis server)

---

## Improvement Plan

### Phase 1: Fix Static Tree Routing (Immediate)

**Problem**: Static tree signals routed through tcv_shot, where static isn't a subtree.

**Fix** in `check_worker()` in `parallel.py`:

```python
# For tree_traversal signals, determine the correct tree to open
accessor = signal["accessor"]
node_path = signal.get("data_source_path")
if node_path and signal.get("discovery_source") == "tree_traversal":
    accessor = node_path
    tree_name_parts = node_path.split("::")
    if len(tree_name_parts) >= 2:
        tree_prefix = tree_name_parts[0].strip("\\").lower()
        # Static tree is standalone - open directly with shot=1
        if tree_prefix == "static":
            tree_name = "static"
            # Override shot: static tree only opens at shot=1
            # (it's a versioned model tree, not shot-scoped)
        else:
            tree_name = "tcv_shot"
    else:
        tree_name = "tcv_shot"
```

**Also needed**: Teach `check_signals_batch.py` about static tree's shot=1 requirement:
- When `tree_name == "static"`, override shot to 1 regardless of check_shot
- Or: add a `check_shot_override` field to the signal dict

**Expected outcome**: Static nodes will be opened correctly, but most will still fail with NNF/NODATA because they're expression nodes. This is correct — it changes 8538 false-NNF to accurate-NODATA/expression-error.

**Effort**: Small (code change only, ~20 lines)

### Phase 2: Expression Node Classification (Short-term)

**Problem**: Expression nodes (`Build_Signal`, `Build_Param`, etc.) fail with misleading NNF errors when their expression chains can't resolve. We need to distinguish:
1. Node exists with stored data → can read
2. Node exists as expression → depends on runtime resolution  
3. Node doesn't exist → true NNF

**Approach**: During tree extraction (`extract_tree.py`), classify each node's record type:

```python
record: dict[str, Any] = {
    "path": path,
    "name": name,
    "node_type": node_type,
}

# Classify data availability without triggering computation
try:
    rec = node.record
    rec_str = repr(rec)[:100]
    if "Build_" in rec_str:
        record["data_class"] = "expression"
    else:
        record["data_class"] = "stored"
except Exception as e:
    if "NODATA" in str(e):
        record["data_class"] = "empty"
    else:
        record["data_class"] = "unknown"
```

Store `data_class` on DataNode and propagate to FacilitySignal during promotion. This enables:
- **Skip expression nodes during check** (or classify differently)
- **Report accurate statistics**: "1,200 stored data nodes, 32,000 expression nodes"
- **Prioritize checking**: stored-data nodes first, expression nodes separately

**Graph schema addition**:
```yaml
FacilitySignal:
  attributes:
    data_class:
      description: Classification of data storage type (stored, expression, empty)
      range: string
```

**Effort**: Medium (extract_tree.py change, graph_ops.py promotion change, parallel.py check logic)

### Phase 3: Multi-Shot Availability Mapping (Medium-term)

**Problem**: Signals checked at one shot (85000) may have data at other shots. Current fallback_shots mechanism retries on epoch first_shots but doesn't persist a complete availability map.

**Approach**: For each signal, store which shots were tested and the result:

**Graph schema — new relationship properties on CHECKED_WITH**:
```yaml
# Extend CHECKED_WITH relationship
checked_shot:
  description: Shot number where check was performed
  range: integer
failed_shots:
  description: Shots where check failed (before fallback found data)
  multivalued: true
  range: integer
availability_range:
  description: "Shot range where data is available (first_available:last_tested)"
  range: string
```

**Implementation**: 
1. Modify `check_signals_batch.py` to return all tested shots and results
2. Modify `mark_signals_checked()` to store availability data on CHECKED_WITH
3. Add a CLI command `imas-codex discover signals tcv --availability-scan` that:
   - Takes a sample of successful signals
   - Tests them at epoch boundary shots (1, 13551, 49829, ..., 74971, 85000)
   - Builds an availability matrix per diagnostic/tree section
   - Stores results as `SignalAvailability` nodes or relationship properties

**Expected output**: A matrix like:
```
results/THOMSON:NE  → available at: 85000(117,118), 63053(109,148), 40000(23,49), 20000(25,144)
magnetics/AMP:CD_001 → available at: 85000(8,10) only
base/FIR:LID_V01   → not available at any tested shot
```

**Effort**: Large (new scanning mode, graph schema changes, new CLI command)

### Phase 4: Smart Check Strategy by Tree Type (Medium-term)

**Problem**: One-size-fits-all checking doesn't account for tree-specific behavior.

**Tree-specific strategies**:

| Tree | Strategy | Shot | Special Handling |
|------|----------|------|-----------------|
| static | Open directly, shot=1 | 1 | Classify expressions vs stored data; skip expression nodes or mark as "expression-only" |
| results | Via tcv_shot, ref shot | 85000 | High no_data expected — many analysis results only exist for specific shots |
| hybrid | Via tcv_shot, ref shot | 85000 | 16K nodes, mostly heating parameters — batch carefully |
| magnetics | Via tcv_shot, ref shot | 85000 | Expression nodes depend on libvaccess.so |
| vsystem | Via vsystem tree? | ? | 561 unchecked signals — needs investigation |
| TDI | Via tcv_shot, ref shot | 85000 | Exclude hardware/control functions; classify by category |

**Implementation**: Add tree-specific check configuration to facility YAML:
```yaml
data_systems:
  mdsplus:
    trees:
      - tree_name: static
        check_strategy: direct  # Don't route through tcv_shot
        check_shot: 1           # Always use shot=1
        skip_expressions: true  # Don't check Build_* expression nodes
      - tree_name: results
        check_strategy: via_connection_tree
        expected_no_data_rate: 0.6  # Don't alarm on high no_data
```

**Effort**: Medium (config schema, check_worker routing logic)

### Phase 5: Missing Library Detection & Reporting (Short-term)

**Problem**: Missing libblas.so and libvaccess.so cause cascading failures that appear as NNF. Need proactive detection and reporting.

**Implementation**:
1. Add a `tools status` check for required shared libraries:
   ```bash
   imas-codex tools status tcv --check-libs
   ```
   Tests: `ldconfig -p | grep libblas`, `python3 -c "import ctypes; ctypes.CDLL('libblas.so')"`

2. Add library dependency tracking to facility YAML (private):
   ```yaml
   infrastructure:
     missing_libraries:
       - libblas.so    # Affects staticgreen() computation
       - libvaccess.so # Affects magnetics expression evaluation
     library_impact:
       libblas.so: "Static tree Green's function computation (staticgreen)"
       libvaccess.so: "Magnetics calibration (mag_gains, magnetics_dependencies)"
   ```

3. During signal check, detect library errors and classify correctly:
   ```python
   if "Error loading lib" in error or "cannot open shared object" in error:
       error_type = "missing_library"
   ```

**Effort**: Small (error classification, one-time discovery)

### Phase 6: TDI Function Categorization (Short-term)

**Problem**: 213 TDI functions treated uniformly, but only ~30 are core physics accessors. Hardware/control functions waste check cycles and can cause crashes.

**Implementation**:
1. Expand `exclude_functions` list in tcv.yaml with all hardware/control functions:
   ```yaml
   exclude_functions:
     # ... existing ...
     # Hardware acquisition
     - dt100_mds
     - dt100_set_fast_clock
     - ocean_acq_read
     - ocean_acq_start
     # ... 
   ```

2. Add a `core_functions` section categorizing physics accessors:
   ```yaml
   core_functions:
     equilibrium:
       - tcv_eq
       - tcv_eq2
       - tcv_eq_profile
       - tcv_psitbx
     global_parameters:
       - tcv_get
       - tcv_ip
       - tcv_bphi
       - tcv_betadml
       - li_1
       - li_ga
     diagnostics:
       - fir_aut
       - fir_comp_aut
       - fir_lin_int_dens
       - thomson_merge
       - ts_fitdata
       - ts_rawdata
     machine_description:
       - static
       - staticgreen
       - staticwinding
       - static_version
   ```

3. Prioritize checking core functions over peripheral ones

**Effort**: Small (YAML configuration, no code changes needed)

### Phase 7: Graph-Efficient Check Data Persistence (Medium-term)

**Problem**: CHECKED_WITH relationships store individual check results per signal. Need efficient patterns for:
- Batch availability queries ("which signals have data at shot X?")
- Cross-epoch analysis ("which diagnostics work across all epochs?")
- Data extraction planning ("what data can we actually extract?")

**Current schema**:
```
(FacilitySignal)-[:CHECKED_WITH {success, error_type, shape, dtype, checked_shot}]->(DataAccess)
```

**Proposed enhancements**:

1. **Signal availability windows** (new node):
   ```yaml
   SignalAvailability:
     id: "tcv:results/thomson/ne:85000"  # signal_id:shot
     facility_id: tcv
     signal_id: tcv:results/thomson/ne
     shot: 85000
     available: true
     shape: [117, 118]
     dtype: float64
     checked_at: datetime
   ```
   Enables: `MATCH (sa:SignalAvailability {shot: 85000, available: true})`

2. **Diagnostic availability summaries** (materialized view via Cypher):
   ```cypher
   // Aggregate check results by diagnostic and tree
   MATCH (s:FacilitySignal {facility_id: 'tcv'})-[c:CHECKED_WITH]->()
   WITH s.data_source_name AS tree, 
        s.diagnostic AS diagnostic,
        sum(CASE WHEN c.success THEN 1 ELSE 0 END) AS success_count,
        count(s) AS total
   RETURN tree, diagnostic, success_count, total, 
          toFloat(success_count) / total AS success_rate
   ORDER BY success_rate DESC
   ```

3. **Data extraction readiness index** (computed property):
   - Per signal: `extraction_ready = success AND shape IS NOT NULL`
   - Per diagnostic: percentage of signals that are extraction-ready
   - Per tree: overall extraction readiness

**Effort**: Large (schema design, query patterns, CLI reporting)

### Phase 8: Performance Optimizations (Long-term)

1. **Batch tree opens**: Group signals by (tree_name, shot) — already implemented, but ensure static tree signals are grouped separately with shot=1

2. **Parallel tree checking**: Run checks across multiple trees concurrently (currently sequential per batch)

3. **Smart batch sizing**: 
   - Static tree: skip entirely if `skip_expressions=true`
   - Small trees (power, manual): can use larger batches
   - Large trees (hybrid): keep batches small to avoid MDSplus overload

4. **Incremental re-checking**: Don't re-check signals that already succeeded unless the reference shot changes. Currently enforced by status transitions, but a `--recheck-failed` flag exists for retry.

5. **Tree structure caching**: Cache tree node listings so repeated check runs don't re-open trees. The extract phase already does this, but the check phase opens trees independently.

### Phase 9: DataAccess-Driven Signal Checking (Critical — New)

**Problem**: Signal checking currently hardcodes routing logic (all tree_traversal → tcv_shot). DataAccess nodes should be the authoritative source for _how_ to access each signal, but they're unpopulated and unused during checking.

**Goal**: Every FacilitySignal has a properly-linked DataAccess node that tells the check worker exactly how to open the tree, what shot to use, and what environment is needed.

#### 9a. DataAccess Taxonomy for TCV

Create separate DataAccess nodes per access pattern:

| DataAccess ID | data_source | Shot | Signals | Purpose |
|--------------|-------------|------|---------|---------|
| `tcv:mdsplus:tree_tdi` | `tcv_shot` | `reference_shot` | ~26K | Subtree signals (results, hybrid, magnetics, base, diagz, ecrh, power) |
| `tcv:mdsplus:static` | `static` | `1` (version) | ~34K | Independent static tree (geometry, GREENS) |
| `tcv:mdsplus:vsystem` | `vsystem` | `reference_shot` | ~561 | Independent vsystem tree |
| `tcv:tdi:functions` | `tcv_shot` | `reference_shot` | ~338 | TDI function evaluation |

Each DataAccess node should have:
```yaml
# tcv:mdsplus:static
id: "tcv:mdsplus:static"
facility_id: "tcv"
name: "Static Tree (Machine Geometry)"
method_type: "mdsplus"
library: "MDSplus"
access_type: "local"
data_source: "static"
discovery_shot: 1
setup_commands:
  - "source /etc/profile.d/mdsplus.sh"
environment_variables: '{"MDSPLUS_DIR": "/usr/local/mdsplus"}'
imports_template: "import MDSplus"
connection_template: "tree = MDSplus.Tree('static', {shot}, 'readonly')"
data_template: "data = tree.getNode('{node_path}').data()"
time_template: null  # Static data has no time dimension
cleanup_template: "tree.close()"
full_example: |
  import MDSplus
  tree = MDSplus.Tree('static', 1, 'readonly')
  # Mechanical geometry (stored data)
  coil_r = tree.getNode('\\STATIC::TOP.MECHANICAL.COIL:R').data()
  coil_z = tree.getNode('\\STATIC::TOP.MECHANICAL.COIL:Z').data()
  # Note: GREENS nodes require libjmmshr_gsl.so (not installed)
  tree.close()
```

#### 9b. Fix Signal → DataAccess Linkage

**Current bug**: `graph_ops.py:promote_leaf_nodes_to_signals()` hardcodes `da_id = f"{facility}:mdsplus:tree_tdi"` for ALL trees.

**Fix**: Route to the correct DataAccess based on tree independence:

```python
# Determine DataAccess based on tree name and config
independent_trees = {"static", "vsystem"}  # From facility YAML
if tree_name in independent_trees:
    da_id = f"{facility}:mdsplus:{tree_name}"
else:
    da_id = f"{facility}:mdsplus:tree_tdi"
```

The list of independent trees should come from facility YAML config, not be hardcoded.

#### 9c. DataAccess-Driven Check Routing

**Current code** in `check_worker()`:
```python
if node_path and signal.get("discovery_source") == "tree_traversal":
    accessor = node_path
    tree_name = "tcv_shot"  # Hardcoded!
```

**New code**: Read the DataAccess node to determine tree_name and shot:
```python
# Load DataAccess node to determine how to check
da_id = signal.get("data_access")
if da_id:
    # Cache DataAccess nodes to avoid repeated queries
    da_info = data_access_cache.get(da_id)
    if da_info:
        tree_name = da_info.get("data_source", "tcv_shot")
        if da_info.get("discovery_shot"):
            shot = da_info["discovery_shot"]
```

This makes the check worker fully data-driven rather than hardcoded.

#### 9d. Routine DataAccess Population Strategy

DataAccess nodes should be populated automatically during signal discovery:

1. **During tree extraction** (`discover signals`):
   - For each tree in facility YAML: create a DataAccess node
   - Independent trees (not in `connection_tree`'s subtrees): separate DataAccess with `data_source = tree_name`
   - Subtrees: link to the connection tree DataAccess
   - Populate `setup_commands` from facility YAML `data_access_patterns.environment`
   - Populate templates from facility YAML `data_access_patterns.templates`

2. **During TDI scanning**:
   - Create/update `{facility}:tdi:functions` DataAccess
   - Populate with TDI-specific templates

3. **Facility YAML schema addition** (`facility_config.yaml`):
```yaml
data_access_patterns:
  mdsplus:
    independent_trees:
      - static
      - vsystem
    setup_commands:
      - "source /etc/profile.d/mdsplus.sh"
    environment_variables:
      MDSPLUS_DIR: "/usr/local/mdsplus"
    known_missing_libraries:
      - library: "libjmmshr_gsl.so"
        impact: "GREENS computation in static tree"
        severity: "blocks_data"
      - library: "libblas.so"
        impact: "Matrix multiply (_matmul) used by staticgreen"
        severity: "blocks_data"
        workaround: "Symlink libblas.so.3 -> libblas.so"
```

4. **Cross-facility generalization**:
   - Every facility defines its `data_access_patterns` in YAML
   - The pattern is generic: method_type, library, setup_commands, templates
   - JET would have PPF DataAccess, SAL DataAccess
   - ITER would have IMAS DataAccess, MDSplus DataAccess
   - Discovery pipeline reads these patterns and creates DataAccess nodes automatically

5. **Validation**: After creating DataAccess nodes, test one signal per access method:
   ```bash
   imas-codex discover signals tcv --validate-access
   ```
   This opens each DataAccess's data_source with the discovery_shot and tries one tdiExecute.

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 9. DataAccess-driven checking | **Critical** | Medium | Fixes root cause: hardcoded routing for 34K static + all future trees |
| 1. Fix static routing | **Critical** | Small | Immediate fix for 34K signals (quick-win before Phase 9) |
| 5. Missing library detection | **High** | Small | Better error classification, known blockers documented |
| 6. TDI function categorization | **High** | Small | Reduces wasted check cycles |
| 2. Expression node classification | **High** | Medium | Accurate data availability reporting |
| 4. Smart check by tree type | **Medium** | Medium | Reduces false failures |
| 3. Multi-shot availability | **Medium** | Large | Enables epoch-based data planning |
| 7. Graph persistence | **Low** | Large | Better query patterns for downstream |
| 8. Performance optimizations | **Low** | Large | Throughput improvements |

## Immediate Actions

1. **Create DataAccess nodes** for static and vsystem trees with proper templates and setup_commands
2. **Fix DataAccess linkage** in `graph_ops.py`: static signals → `tcv:mdsplus:static`, not `tcv:mdsplus:tree_tdi`
3. **Fix check_worker routing** to read `data_source` from DataAccess node instead of hardcoding `tree_name = "tcv_shot"`
4. **Populate existing DataAccess nodes** (`tree_tdi`, `tdi:functions`) with setup_commands and templates
5. **Add `independent_trees`** to facility YAML config under `data_access_patterns`
6. **Update exclude_functions** in tcv.yaml with hardware/control functions
7. **Add `missing_library` error classification** in `_classify_check_error()`
8. **Reset 8,538 static NNF signals** to `enriched` for re-checking after fix
9. **Record missing libraries** in tcv_private.yaml infrastructure section
