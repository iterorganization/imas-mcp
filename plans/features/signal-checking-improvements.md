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

#### 2. Missing Shared Libraries (libblas.so, libvaccess.so)

Multiple MDSplus TDI functions depend on shared libraries not installed on the login nodes:
- `libblas.so` — Required by `staticgreen()` for Green's function computation
- `libvaccess.so` — Required by `magnetics_dependencies`, `mag_gains_helper`, and others

These cause expression nodes to fail even when accessed correctly. The errors appear as NNF (because the expression chain can't resolve) rather than a clear library error.

**Impact**: Unknown number of signals affected beyond static tree; at minimum affects magnetics expression nodes

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
node_path = signal.get("node_path")
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

Store `data_class` on TreeNode and propagate to FacilitySignal during promotion. This enables:
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
data_sources:
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
   WITH s.tree_name AS tree, 
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

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Fix static routing | **Critical** | Small | Fixes 34K signals (51% of all TCV signals) |
| 5. Missing library detection | **High** | Small | Better error classification |
| 6. TDI function categorization | **High** | Small | Reduces wasted check cycles |
| 2. Expression node classification | **High** | Medium | Accurate data availability reporting |
| 4. Smart check by tree type | **Medium** | Medium | Reduces false failures |
| 3. Multi-shot availability | **Medium** | Large | Enables epoch-based data planning |
| 7. Graph persistence | **Low** | Large | Better query patterns for downstream |
| 8. Performance optimizations | **Low** | Large | Throughput improvements |

## Immediate Actions

1. **Fix static tree routing** in `check_worker()` — open static directly with shot=1
2. **Update `exclude_functions`** in tcv.yaml with hardware/control functions  
3. **Add `missing_library` error classification** in `_classify_check_error()`
4. **Reset 8,538 static NNF signals** to `enriched` for re-checking after fix
5. **Check vsystem tree** — 561 discovered signals never checked
