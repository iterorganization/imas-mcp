# DD Node Kind Classification — Audit & Remaining Work

## Problem Statement

The `node_category` field on `IMASNode` had three values: `data`, `error`, `metadata`. The monolithic `data` bucket (20,037 nodes) lumped physics quantities, coordinates, structural artifacts, and identifier fields together, causing vector space pollution and SN extraction gaps.

## Design (RD Consensus — 3 Rounds)

Expand NodeCategory from 3 to 6 values. Cannot share enum with StandardNameKind (orthogonal taxonomies).

```yaml
NodeCategory:
  quantity:   Physics data. Embedded + SN-extracted.
  coordinate: Independent variable (time, space, flux). NOT embedded, NOT SN-extracted.
  structural: Storage artifact (/data, /validity, containers). NOT embedded.
  identifier: Typed descriptor with HAS_IDENTIFIER_SCHEMA. NOT embedded.
  error:      Uncertainty bounds (_error_*). Unchanged.
  metadata:   Bookkeeping (ids_properties/*, code/*). Unchanged.
```

### Pipeline Participation Matrix

| NodeCategory | Enriched | Embedded | SN Extracted | Searchable |
|---|---|---|---|---|
| `quantity` | ✓ | ✓ | ✓ | ✓ |
| `coordinate` | ✓ | ✗ | ✗ | ✓ |
| `structural` | ✗ | ✗ | ✗ | ✗ |
| `identifier` | ✗ | ✗ | ✗ | ✗ |
| `error` | ✗ | ✗ | ✗ | ✗ |
| `metadata` | ✗ | ✗ | ✗ | ✗ |

## What Was Implemented (Phases A–E)

### Phase A: Schema + Classifier + Constants (committed: `5127c5b9`)

**Created `imas_codex/core/node_classifier.py`** — two-pass classification:
- **Pass 1** (`classify_node_pass1`): 15 rules using path, name, data_type, unit, parent_data_type
- **Pass 2** (`classify_node_pass2`): 3 relational rules using HAS_IDENTIFIER_SCHEMA, HAS_COORDINATE, children evidence

**Created `imas_codex/core/node_categories.py`** — centralized pipeline participation constants.

**Modified `imas_codex/schemas/imas_dd.yaml`** — added quantity, coordinate, structural, identifier to NodeCategory enum.

### Phase B: Consumer Update (committed: `ef0b049a`)

Updated 55 occurrences across 17+ files to import from `node_categories.py`:
- `sources/dd.py`, `build_dd.py`, `dd_graph_ops.py`, `dd_workers.py`
- `dd_ids_enrichment.py`, `dd_progress.py`, `cli/imas_dd.py`, `ids/tools.py`
- `tools/graph_search.py` (9 sites), `tools/migration_guide.py`, `clusters.py`

### Phase C: Wire into Build Pipeline (committed: `6fdaeeed`)

- `_classify_node()` in `build_dd.py` delegates to shared classifier
- Added `_reclassify_relational()` for Pass 2 post-build

### Phase D: Graph Migration (applied to live graph — NOT code-committed)

Executed as inline Cypher + Python via interactive session. Details below.

### Phase E: Remove `data` from enum (local commit `d4234eba` — NOT pushed)

Removed `data` from `imas_dd.yaml` and `node_categories.py`.

---

## Exact Graph Mutations Performed

### Step 1: Shadow-Property Classification

**Query:** Fetched all 20,037 `node_category='data'` nodes with:
- Node properties: `id`, `data_type`, `unit`, `node_category`
- Relationship presence: `EXISTS { (n)-[:HAS_IDENTIFIER_SCHEMA]->() }` (boolean)
- Coordinate target: `EXISTS { ()-[:HAS_COORDINATE]->(n) }` (boolean)

**Processing:** Each node classified in Python via `classify_node_pass1()` then `classify_node_pass2()`.

**Write:** `SET n.node_category_new = $cat` in batches of 2,000.

### Step 2: Coverage Validation

```cypher
MATCH (n:IMASNode) WHERE n.node_category = 'data' AND n.node_category_new IS NOT NULL
RETURN count(n)  -- Result: 20,037 (100% coverage)

MATCH (n:IMASNode) WHERE n.node_category_new = 'data'
RETURN count(n)  -- Result: 0 (no nodes remain as 'data')
```

### Step 3: Atomic Swap

```cypher
MATCH (n:IMASNode) WHERE n.node_category_new IS NOT NULL
SET n.node_category = n.node_category_new
REMOVE n.node_category_new
```

All 20,037 former `data` nodes now have one of: `quantity`, `coordinate`, `structural`, `identifier`, `metadata`.

### Step 4: Stale Embedding Cleanup

**Reasoning:** Before migration, ALL `data` nodes were embedded. After migration, only `quantity` nodes should be embedded. Non-quantity nodes with stale embeddings pollute the vector index.

```cypher
-- Remove embedding properties from non-quantity nodes
MATCH (n:IMASNode)
WHERE NOT (n.node_category IN ['quantity']) AND n.embedding IS NOT NULL
REMOVE n.embedding, n.embedded_at, n.embedding_hash, n.embedding_text
SET n.status = CASE
  WHEN n.node_category = 'coordinate' THEN 'enriched'
  ELSE 'built'
END
```

**Result:** 10,044 non-quantity nodes had embeddings removed. Status remapped:
- `coordinate` → `enriched` (terminal status — enrichable but never embedded)
- `structural`, `identifier` → `built` (terminal — no further processing)

---

## Resulting Graph State

```
| Category   | Count  | %     |
|------------|--------|-------|
| error      | 31,281 | 51.0% |
| metadata   | 10,715 | 17.5% |
| quantity   |  9,993 | 16.3% |
| structural |  7,395 | 12.1% |
| coordinate |  1,739 |  2.8% |
| identifier |    243 |  0.4% |
| TOTAL      | 61,366 |       |
```

Embedding state: exactly 9,993 `quantity` nodes embedded. Zero non-quantity nodes embedded.

---

## Bugs Found in Migration

### BUG 1: Pass 2 R3 Uses Non-Existent PARENT_OF Relationship (CRITICAL)

**Location:** `imas_codex/graph/build_dd.py` line 2050

```python
# BUG: This relationship type does not exist in the graph
OPTIONAL MATCH (n)<-[:PARENT_OF]-(child:IMASNode)
```

**The graph uses `HAS_PARENT` (child→parent), NOT `PARENT_OF` (parent→child).** There are ZERO `PARENT_OF` relationships in the entire graph. The correct query is:

```python
OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(n)
```

**Impact:** Pass 2 R3 was designed to validate STRUCTURE+unit nodes by checking for data/time/validity children. Because `PARENT_OF` doesn't exist, the children check ALWAYS returned empty. Every STRUCTURE+unit node was demoted from `quantity` to `structural`.

**Misclassified nodes:** 1,703 STRUCTURE+unit nodes that DO have children (verified via `HAS_PARENT` reverse lookup). Verified that ZERO of these have only-metadata/error/identifier children — all 1,703 have at least one quantity/coordinate/structural child.

**Examples of misclassified nodes (should be `quantity`, are `structural`):**
```
magnetics/flux_loop/flux                   STRUCTURE   Wb         children=7
summary/global_quantities/v_loop           STRUCTURE   V          children=5
summary/local/divertor_target/t_i_average  STRUCTURE   eV         children=5
iron_core/segment/magnetisation_z          STRUCTURE   T          children=5
lh_antennas/power                          STRUCTURE   W          children=5
ec_launchers/launcher/steering_angle_tor   STRUCTURE   rad        children=5
```

### BUG 2: Pass 2 R2 Over-Aggressive Coordinate Classification (CRITICAL)

**Location:** `node_classifier.py` line 282-284 (R2 rule)

**Issue:** `HAS_COORDINATE` targets are unconditionally reclassified as `coordinate`. This is wrong because `HAS_COORDINATE` has different semantic meanings in different contexts:

1. **"This IS a coordinate axis"** (correct for R2): `j_phi -[:HAS_COORDINATE]-> psi` means psi is the coordinate for j_phi
2. **"This data shares grid indexing"** (wrong for R2): `j_parallel/coefficients -[:HAS_COORDINATE]-> j_parallel/values` means coefficients reconstruct these values on a GGD grid

**Detailed breakdown of 577 misclassified physics-unit coordinate targets:**

| Category | Count | Pattern | Correct Category |
|---|---|---|---|
| GGD `/values` paths | 452 | `*/ggd/*/values`, `ggd_fluid/*/values`, `field_map/*/values` | quantity |
| GGD `/radial` components | 39 | `*/ggd/*/radial` | quantity |
| Fit measurements | 28 | `*_fit/measured` | quantity |
| Non-GGD `/values` data | ~31 | `plasma_transport/.../values`, `tf/field_map/.../values`, `waves/.../values` | quantity |
| **Genuine coordinate axes** | **~27** | Named axes: psi, photon_energy, b_field, energies, frequencies | **coordinate** (correctly classified) |

**The 27 genuine coordinate axes with physics units:**
- `equilibrium/time_slice/profiles_1d/psi` (Wb) — flux coordinate for profiles
- `camera_x_rays/photon_energy` (eV) — energy axis for X-ray spectrum
- `ferritic/permeability_table/b_field` (T) — B-field lookup table axis
- `magnetics/*/non_linear_response/b_field_linear` (T) × 4 — response curve axes
- `neutron_diagnostic/*/energies` (eV) × 5 — energy band/spectrum axes
- `hard_x_rays/channel/energy_band/energies` (eV) — energy band axis
- `spectrometer_x_ray_crystal/channel/energies` (eV) — energy axis
- `reflectometer_fluctuation/.../frequencies_fourier` (Hz) — frequency axis
- `gas_injection/valve/response_curve/voltage` (V) — response curve axis
- `iron_core/segment/b_field` (T) — magnetization curve axis
- `pf_active/coil/b_field_max` (T), `pf_active/coil/temperature` (K), `pf_active/force_limits/limit_max` (N)
- `spectrometer_mass/residual_spectrum/a` (AMU) — mass spectrum axis
- `wall/global_quantities/neutral/incident_species/energies` (eV) — energy axis
- `mhd_linear/.../alfven_frequency_spectrum/real` (s^-1)
- `amns_data/coordinate_system/coordinate/values` — coordinate with dynamic unit

**Root cause:** R2 cannot use unit alone to distinguish coordinate axes from grid-stored data. The distinguishing feature is the **path ending**: `/values`, `/radial`, `/measured` endings indicate grid-stored data, while named endpoints (psi, photon_energy, b_field, energies, r, z) are physical axes.

**Full breakdown of all 1,460 HAS_COORDINATE targets by path ending:**

| Path Ending | Count | Classification | Reasoning |
|---|---|---|---|
| `time` | 548 | coordinate ✓ | Time axis |
| `values` | 514 | quantity ✗ (513 fix + 1 exception) | GGD data container — NOT a coordinate axis |
| `r` | 64 | coordinate ✓ | R-component of cylindrical (R,Z,φ) — IS a genuine spatial coordinate |
| `radial` | 44 | quantity ✗ | Radial flux component (e.g., `flux_limiter/radial`) — data, not axis |
| `x1` | 41 | coordinate ✓ | Grid coordinate |
| `measured` | 28 | quantity ✗ | Fit measurements |
| `rho_tor_norm` | 23 | coordinate ✓ | Normalised toroidal flux coordinate |
| Named physics axes | ~198 | coordinate ✓ | wavelengths, dim1, dim2, n_tor, energies, phi, rho_pol_norm, etc. |

**Key insight from RD Round 2: `/r` must NOT be a data-storage ending.** All 64 `/r` nodes are genuine R-coordinates (cylindrical R) with unit `m`. Treating `/r` as data-storage would misclassify all of them.

**Corrected R2 heuristic (RD Round 2):**
```
R2. Node is HAS_COORDINATE target → coordinate
    EXCEPT when Pass 1 classified as quantity AND path ends with
    a data-storage indicator: /values, /radial, /measured
    (NOT /r — that is a genuine spatial coordinate)
    
    Special case: paths containing 'coordinate' in ancestor
    (e.g., amns_data/coordinate_system/coordinate/values)
    are genuine coordinates even with /values ending.
```

**Scope (RD Round 2 Finding 2): The graph fix must audit ALL 1,460 HAS_COORDINATE targets**, not just the 577 with physics units. Unitless targets can also have data-storage endings (e.g., 10 unitless `/values` nodes, 5 unitless `/radial` nodes). The corrected classifier is path-based, not unit-based, so the repair scope must match.

### BUG 3: Identifier Off-by-83 (Not Off-by-1)

**Observed:** 243 identifier nodes. Pre-migration count of 244 HAS_IDENTIFIER_SCHEMA was total, not per-category.

**Root cause:** 83 nodes have HAS_IDENTIFIER_SCHEMA but are in `ids_properties/` subtrees (all are `ids_properties/occurrence_type`). Pass 1 Rule 2 (metadata subtree) fires first and classifies them as metadata. The migration excluded metadata nodes from R1 processing. This is **correct** — these are metadata with associated schemas, not identifiers.

**Verified:** `MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->() WHERE n.node_category <> 'identifier' RETURN count(n)` = 83, all `ids_properties/occurrence_type`.

**Action:** No fix needed. Document in classifier comments that metadata subtree takes precedence over identifier schema.

---

## Remaining Work

### Task 1: Fix Bug 1 — PARENT_OF → HAS_PARENT (Code + Graph)

**Code fix** (`build_dd.py` line 2050):
```python
# Before (wrong):
OPTIONAL MATCH (n)<-[:PARENT_OF]-(child:IMASNode)

# After (correct):
OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(n)
```

**Also tighten the R3 children check** (`node_classifier.py` `classify_node_pass2`):
```python
# Before: any child category counts
has_data_child = any(c in ("quantity", "coordinate", "structural") for c in children)

# After: same logic, but document the intent — metadata/error/identifier
# children alone don't confirm signal_value pattern
```
The current logic is correct (checks for quantity/coordinate/structural children). Verified that all 1,703 nodes have such children — but the code should remain tight to prevent future regressions.

**Graph fix** (inline Cypher, one-off — requires children evidence):
```cypher
-- Reclassify the 1,703 misclassified structural+unit nodes
-- Only promote nodes that have data-bearing children (quantity/coordinate/structural)
MATCH (n:IMASNode {node_category: 'structural'})
WHERE n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
  AND n.unit IS NOT NULL AND n.unit <> '' AND n.unit <> '-'
MATCH (child:IMASNode)-[:HAS_PARENT]->(n)
WHERE child.node_category IN ['quantity', 'coordinate', 'structural']
WITH DISTINCT n
SET n.node_category = 'quantity', n.status = 'built'
RETURN count(n) AS fixed
-- Expected: 1,703
```

Setting `status='built'` means they'll be picked up by the next enrichment run. They should NOT get embeddings until enriched.

### Task 2: Fix Bug 2 — R2 Coordinate Path-Aware Override (Code + Graph)

**Code fix** (`node_classifier.py` `classify_node_pass2`, R2 rule):
```python
# Data-storage endings — HAS_COORDINATE means "indexed by grid", not "is coordinate"
# NOTE: /r is NOT here — it is a genuine spatial coordinate (cylindrical R, unit=m)
_DATA_STORAGE_ENDINGS = frozenset({"values", "radial", "measured"})

# R2: Coordinate target → coordinate
if is_coordinate_target and current_category != "error":
    if current_category == "quantity":
        # Physics quantity that is also a coordinate target.
        # Only override if path does NOT end with a data-storage indicator.
        path_parts = (path or "").split("/")
        last_segment = path_parts[-1] if path_parts else ""
        if last_segment in _DATA_STORAGE_ENDINGS:
            # Exception: paths with an explicit 'coordinate' ancestor segment
            # (e.g., amns_data/coordinate_system/coordinate/values)
            # Use exact segment match, NOT substring — avoids false positives
            # from 'coordinate_system', 'coordinates_type', etc.
            if "coordinate" not in path_parts[:-1]:
                return None  # Keep as quantity — grid-stored data
    return "coordinate"
```

**Graph fix** — re-run corrected classifier on ALL 1,460 HAS_COORDINATE targets, apply by ID list:

The graph fix must NOT be unit-scoped (RD Round 2). Since the corrected rule is path-based, the repair set must be ALL HAS_COORDINATE targets currently classified as `coordinate`, not just unitful ones.

1. Fetch ALL 1,460 current `coordinate` nodes that are HAS_COORDINATE targets
2. Run corrected `classify_node_pass2()` in Python for each
3. Collect IDs where the corrected classifier says `None` (keep as quantity)
4. Apply via explicit `WHERE n.id IN $ids` Cypher

```python
# Pseudocode for the one-off fix
from imas_codex.core.node_classifier import classify_node_pass2
misclassified_ids = []
for node in fetch_all_coordinate_target_nodes():  # ALL 1,460, not just unitful
    result = classify_node_pass2(
        "quantity",  # what Pass 1 would have said
        is_coordinate_target=True,
        path=node["id"],
        data_type=node["data_type"],
        unit=node["unit"],
    )
    if result is None:  # corrected classifier says: keep as quantity
        misclassified_ids.append(node["id"])

# Apply in batches
query("""
    UNWIND $ids AS id
    MATCH (n:IMASNode {id: id})
    SET n.node_category = 'quantity', n.status = 'built'
""", ids=misclassified_ids)
# Expected: ~586 reclassified (514 values + 44 radial + 28 measured),
#           ~874 kept as coordinate (time, r, x1, rho_tor_norm, named axes, etc.)
```

**Note:** `classify_node_pass2` needs a new `path` parameter for the path-aware R2 logic.

### Task 3: Update Tests (BEFORE graph fixes)

Per RD feedback, tests must be written and passing before graph one-offs are applied. This ensures we can verify the corrected classifier produces the expected results.

**Required regression tests for `test_node_category.py`:**

1. `classify_node_pass1` tests:
   - All 6 output categories with representative examples
   - Include `data_type` and `unit` parameters
   - Edge cases: STRUCTURE+unit (provisional quantity), dimensionless physics

2. `classify_node_pass2` regression tests (prevent recurrence):
   - `HAS_PARENT` vs `PARENT_OF`: STRUCTURE+unit with children → quantity
   - STRUCTURE+unit WITHOUT data-bearing children → structural (R3 demotion)
   - GGD `/values` quantity with incoming HAS_COORDINATE → stays quantity (not coordinate)
   - True coordinate target (psi, time) → becomes coordinate
   - Identifier schema → identifier regardless of other signals

3. Identity audit test:
   - `MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->() WHERE n.node_category <> 'identifier' AND NOT (n.id CONTAINS '/ids_properties/')` → empty

**Update hardcoded `'data'` references** (20 occurrences, 9 files):

| File | Occurrences | Action |
|---|---|---|
| `tests/graph/test_node_category.py` | Entire file | Rewrite: test shared classifier |
| `tests/graph/test_dd_build.py` | 2 | Use `SEARCHABLE_CATEGORIES` |
| `tests/graph/test_vector_search.py` | 2 | Use `EMBEDDABLE_CATEGORIES` |
| `tests/test_imas_search_remediation.py` | 4 | Update assertions |
| `tests/graph_mcp/conftest.py` | 1 | Update fixture |
| `tests/search/generate_expected_paths.py` | 3 | Update queries |
| `tests/search/test_search_evaluation.py` | 4 | Update queries |
| `tests/search/test_search_benchmarks.py` | 3 | Update queries |
| `tests/search/conftest.py` | 1 | Update fixture |

### Task 4: Apply Graph Fixes (after tests pass)

Execute the one-off inline Cypher for Bug 1 and Bug 2 (as described above).

### Task 5: Verify Corrected Distribution

```cypher
-- Category distribution
MATCH (n:IMASNode) RETURN n.node_category AS cat, count(n) AS cnt ORDER BY cnt DESC

-- Zero stale data
MATCH (n:IMASNode {node_category: 'data'}) RETURN count(n)  -- must be 0

-- Embedding integrity: only quantity has embeddings
MATCH (n:IMASNode) WHERE NOT (n.node_category IN ['quantity']) AND n.embedding IS NOT NULL
RETURN count(n)  -- must be 0

-- No null categories
MATCH (n:IMASNode) WHERE n.node_category IS NULL RETURN count(n)  -- must be 0

-- Identifier audit (excluding ids_properties metadata)
MATCH (n:IMASNode)-[:HAS_IDENTIFIER_SCHEMA]->()
WHERE n.node_category <> 'identifier' AND NOT (n.id CONTAINS '/ids_properties/')
RETURN count(n)  -- must be 0
```

### Task 6: Push Phase E Commit + Code Fixes

After verification, push the combined commit:
- Bug 1 fix (PARENT_OF → HAS_PARENT in build_dd.py)
- Bug 2 fix (path-aware R2 in node_classifier.py)
- Phase E (remove `data` from schema + constants)

### Task 7: Re-enrich + Re-embed Reclassified Quantity Nodes

After graph fixes, ~2,289 new `quantity` nodes (1,703 from Bug 1 + ~586 from Bug 2) need enrichment and embedding. They'll be picked up automatically by the pipelines since they have `status='built'`.

**Cost estimate:** ~2,289 nodes × $0.01/node enrichment + free embedding = ~$23.

### Task 8: Documentation

| Target | Update Required |
|---|---|
| `AGENTS.md` | NodeCategory expansion (6 values), pipeline matrix, `node_categories.py` imports |
| `plans/README.md` | Mark dd-node-kind-classification as complete |
| Schema reference | Auto-generated by `uv run build-models` |

---

## Execution Order (RD-reviewed)

1. **Code fixes** — Bug 1 (PARENT_OF → HAS_PARENT) + Bug 2 (path-aware R2) + Phase E cleanup
2. **Write regression tests** — classifier tests, data references
3. **Run tests** — verify all pass with corrected classifier
4. **Graph one-offs** — apply Bug 1 + Bug 2 fixes via inline Cypher
5. **Verify counts/invariants** — run verification queries
6. **Push code** — combined commit with all fixes
7. **Re-enrich/embed** — pipeline picks up new quantity nodes
8. **Documentation** — AGENTS.md, plans/README.md
9. **Resume SN bootstrap loop**

---

## Post-Fix Expected Distribution

```
| Category   | Count  | Change        |
|------------|--------|---------------|
| error      | 31,281 | unchanged     |
| metadata   | 10,715 | unchanged     |
| quantity   | 12,282 | +2,289        |
| structural |  5,692 | −1,703        |
| coordinate |  1,153 | −586          |
| identifier |    243 | unchanged     |
```

*Breakdown: +1,703 from Bug 1 (structural→quantity) + ~586 from Bug 2 (coordinate→quantity) = ~2,289 new quantity nodes. 1 coordinate node (amns_data/coordinate_system/coordinate/values) stays coordinate via path-ancestor exception.*

## SN Bootstrap Impact

- **Before migration:** SN extraction used `node_category='data'` + `node_type='dynamic'` → 11,441 extractable
- **After migration (fixed):** `node_category='quantity'` + `node_type='dynamic'` → need to verify count
- **After relaxing node_type filter:** All ~12,282 quantity nodes become extractable
- **Magnetics specifically:** `magnetics/flux_loop/flux` (STRUCTURE, Wb) will be `quantity` after Bug 1 fix. Its children (`flux/data`, `flux/time`) are already correctly classified. The SN extraction can now find these via the `quantity` filter.

## Lessons Learned

1. **Always verify relationship direction in Neo4j.** `HAS_PARENT` (child→parent) is the opposite of `PARENT_OF` (parent→child). The graph schema uses `HAS_PARENT`. Running `MATCH ()-[r:PARENT_OF]->() RETURN count(r)` before writing the query would have caught this instantly.
2. **Don't bulk-fix by a single property.** The unit-based R2 fix (RD round 1) would have misclassified 27 genuine coordinate axes. Path-aware classification + explicit ID lists is safer.
3. **`/r` is a genuine coordinate, not a data-storage ending.** 64 HAS_COORDINATE targets ending in `/r` are R-components in cylindrical (R,Z,φ) systems with unit `m`. A global data-storage exclusion for `/r` would misclassify all of them.
4. **Fix scope must match rule scope.** When the corrected rule is path-based (not unit-based), the repair audit must cover ALL affected nodes, not just the unitful subset. 15 unitless `/values` and `/radial` targets would have been missed.
3. **Run migrations as inline Cypher** per AGENTS.md. The CLI command was unnecessary overhead for a one-off operation.
4. **Spot-check with physics knowledge** after automated classification. Understanding what `HAS_COORDINATE` means in GGD vs profiles_1d contexts is essential.
5. **Tests before graph mutations.** Write regression tests that verify the corrected classifier, THEN apply graph changes. This order catches logic bugs before they hit production data.
