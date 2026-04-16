# DD Unified Classification & Enrichment Pipeline

## Problem Statement

The DD classification system needs a principled taxonomy that distinguishes physics
measurements from machine geometry, surfaces correct classifier semantics, and uses
a capable enrichment model. The current system has a monolithic `quantity` bucket that
conflates physics measurements (T_e, I_p, B_tor) with hardware geometry (coil positions,
vessel outlines, diagnostic lines of sight), classifier bugs that misclassify ~2,300 nodes,
and enrichment performed by the weakest available model (flash-lite). This plan designs
the correct classification and enrichment pipeline, then applies it to all DD nodes.

This is a **prerequisite** for the [SN Greenfield Pipeline](standard-names/28-sn-greenfield-pipeline.md) —
`node_category` labels drive EXTRACT source filtering, and enrichment quality directly
affects standard name generation context.

The plan addresses:

1. **Splitting `quantity` → `physical_quantity` + `geometry`**
2. **Fixing classifier bugs** (Bug 1: reversed traversal, Bug 2: overbroad coordinate)
3. **Restoring broken search** (dirty worktree reverted Phase B; tests hardcode `'data'`)
4. **Renaming pass1/pass2** to meaningful function names
5. **TDD test suite** for full labeling pipeline
6. **MCP tool augmentation** to expose category and coordinate metadata
7. **Enrichment model upgrade** (flash-lite → sonnet, with benchmark evidence)
8. **Full reclassification and re-enrichment** via status machine

### Why Split Quantity?

| Need | How the Split Helps |
|------|---------------------|
| **Standard name grammar** | Geometry nodes use ISN `geometric_base` (position, outline, trajectory) vs physical nodes use `physical_base` (temperature, current). The compose prompt can tailor guidance per type |
| **Search UX** | Users can filter "show me only hardware geometry" or "only physics measurements" |
| **Embedding quality** | Both types ARE embedded (both are meaningful). The split is informational, not a pipeline barrier |
| **Validation** | Geometry nodes always have spatial units (m, rad, deg). A `geometry` node with unit `eV` would be suspicious |

### Non-Goals

- Backwards compatibility (we can regenerate all nodes)
- Sharing enum with StandardNameKind (orthogonal taxonomies — see ISN plan)
- LLM-based classification for the split (rule-based via path patterns is sufficient)
- Capturing plasma shape params as geometry (they are physics outputs → `physical_quantity`)

---

## Current State

### What Works

- **NodeCategory enum** has 6 values in schema (quantity/coordinate/structural/identifier/error/metadata)
- **node_classifier.py** has 15 attribute rules + 3 relational rules
- **node_categories.py** centralizes pipeline participation constants
- **Phase B** (ef0b049a, on origin/main) correctly parameterized `graph_search.py` with `SEARCHABLE_CATEGORIES`

### What's Broken

| Issue | Severity | Detail |
|-------|----------|--------|
| **Dirty worktree reverted Phase B** | 🔴 CRITICAL | Another agent rewrote `graph_search.py` (-634 lines), reverting all 9 `IN $categories` changes back to `'data'`. 5 hardcoded `'data'` references on disk. **DD search returns zero results.** |
| **Test suite hardcodes `'data'`** | 🟡 HIGH | 17 references across 6 test files still assert `node_category = 'data'`. Tests would fail against the live graph (no `data` nodes exist). |
| **Bug 1: reversed traversal** | 🟡 HIGH | `_reclassify_relational()` in `build_dd.py` uses `PARENT_OF` but the schema relationship is `HAS_PARENT` (child→parent). 1,703 structural nodes should be quantity. |
| **Bug 2: overbroad R2** | 🟡 HIGH | Pass 2 R2 rule reclassifies ALL coordinate targets to `coordinate`, including data-storage nodes like `.../profiles_1d/psi/values` (~586 nodes should be quantity). |
| **Two classification systems** | 🟡 MEDIUM | `node_classifier.py` (pass1/pass2) and `dd_enrichment.py` (concept/accessor) are independent and partially overlapping. The enrichment classifier decides LLM vs template enrichment; the node classifier decides pipeline participation. |

### Graph State (Current)

| Category | Count | After Bug Fixes | After Quantity Split |
|----------|-------|-----------------|---------------------|
| error | 31,281 | 31,281 | 31,281 |
| metadata | 10,715 | 10,715 | 10,715 |
| quantity | 9,993 | 12,282 | ~11,412 physical_quantity |
| structural | 7,395 | 5,692 | 5,692 |
| coordinate | 1,739 | 1,153 | 1,153 |
| identifier | 243 | 243 | 243 |
| geometry | — | — | ~870 |

---

## Design

### 1. NodeCategory Enum (Final: 8 Values)

> **Major revision (2026-04-16):** Research in `plans/research/geometric-quantity-taxonomy.md`
> showed the original `geometry` (17 leaf names, 36 nodes of plasma shape params)
> was wrong. Elongation, triangularity, and aspect_ratio are **physics outputs** computed by
> equilibrium codes. The correct `geometry` category captures **machine and diagnostic
> hardware positions** (~870 nodes): coil r/z, vessel outlines, diagnostic lines of sight,
> apertures. Classification is **structural** (path patterns + spatial units), not leaf-name
> matching.

```yaml
NodeCategory:
  physical_quantity:
    description: >-
      Measurable physics quantity (temperature, current, pressure, field,
      density, flux, elongation, triangularity, beta, safety_factor).
      Enriched, embedded, searchable, SN-extractable.
  geometry:
    description: >-
      Machine and diagnostic hardware geometry (coil r/z, vessel outline,
      diagnostic LoS, aperture positions, detector dimensions). Things that
      exist in real space before plasma. Enriched, embedded, searchable,
      SN-extractable.
  coordinate:
    description: >-
      Independent variable (time, spatial coords, flux coordinates).
      Enriched and searchable, NOT embedded or SN-extracted.
  structural:
    description: >-
      Storage artifact (/data arrays, /validity flags, containers).
      Neither enriched, embedded, nor SN-extracted.
  identifier:
    description: >-
      Typed descriptor with HAS_IDENTIFIER_SCHEMA relationship.
      Neither enriched, embedded, nor SN-extracted.
  error:
    description: Uncertainty bound fields (_error_upper, _error_lower, _error_index)
  metadata:
    description: Bookkeeping subtrees (ids_properties, code, identifier descriptors)
  constant:
    description: >-
      Future — for physical/engineering constants if needed.
```

**Removed**: `quantity` (replaced by `physical_quantity` + `geometry`)

**Key taxonomy decisions:**
- Elongation, triangularity, squareness, shift → `physical_quantity` (equilibrium outputs)
- Coil r/z, vessel outline, LoS, apertures → `geometry` (engineering inputs)
- Plasma boundary outline → `physical_quantity` (NOT geometry — it's an equilibrium output)

### 2. Pipeline Participation Matrix (Updated)

| NodeCategory | Enriched | Embedded | SN Extracted | Searchable |
|---|---|---|---|---|
| `physical_quantity` | ✓ | ✓ | ✓ | ✓ |
| `geometry` | ✓ | ✓ | ✓ | ✓ |
| `coordinate` | ✓ | ✗ | ✗ | ✓ |
| `structural` | ✗ | ✗ | ✗ | ✗ |
| `identifier` | ✗ | ✗ | ✗ | ✗ |
| `error` | ✗ | ✗ | ✗ | ✗ |
| `metadata` | ✗ | ✗ | ✗ | ✗ |

```python
# node_categories.py (updated)
QUANTITY_CATEGORIES = frozenset({"physical_quantity", "geometry"})
EMBEDDABLE_CATEGORIES = QUANTITY_CATEGORIES
SN_SOURCE_CATEGORIES = QUANTITY_CATEGORIES
SEARCHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"}
ENRICHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"}
```

### 3. Classifier Architecture (Renamed)

**Current names**: `classify_node_pass1` / `classify_node_pass2`
**New names**: `classify_from_attributes` / `refine_from_relationships`

Rationale: The names describe WHAT the function does (uses XML attributes vs graph
relationships), not WHEN it runs in a sequence. This makes the code self-documenting
and eliminates the phase1/phase2 confusion.

#### `classify_from_attributes(path, name, *, data_type, unit, parent_data_type) → str`

Rules 1–8 unchanged. Rules 9–15 modified:

```
Rule 9a: FLT/CPX + spatial unit + geometry path pattern → geometry
Rule 9b: FLT/CPX + physics unit → physical_quantity
Rule 10a: FLT/CPX + no unit + coordinate segment → coordinate
Rule 10b: FLT/CPX + no unit → physical_quantity  (dimensionless physics: beta, q, zeff)
Rule 11a: STRUCTURE + unit + geometry path pattern → geometry
Rule 11b: STRUCTURE + unit → physical_quantity
Rule 12: STRUCTURE without unit → structural
Rule 13: INT_0D + no unit + no structural keywords → physical_quantity
Rule 14a: INT + spatial unit + geometry path pattern → geometry
Rule 14b: INT + physics unit → physical_quantity
Rule 15: Remaining INT → structural
```

#### Geometry Detection: Structural Path Patterns

> **Replaces the 17-name leaf set.** Geometry is identified by the DD subtree structure
> (parent path patterns) combined with spatial units, not by leaf name matching.

Geometry is classified when ALL of the following hold:
1. **Spatial unit**: `m`, `rad`, `m^2`, `m^-1`, `m^3`, `deg` (or dimensionless for angles)
2. **Path contains a geometry-indicating segment** (any ancestor):

```python
GEOMETRY_PATH_PATTERNS: frozenset[str] = frozenset({
    # Machine hardware
    "geometry",          # Coil geometry, iron core, optical elements
    "outline",           # Wall, vessel, coil cross-sections
    "aperture",          # Diagnostic aperture positions
    "line_of_sight",     # Diagnostic viewing geometry
    "position",          # Gauge, probe, channel positions
    "detector",          # Detector hardware dimensions
    "mirror",            # Optical mirrors
    "waveguide",         # Microwave waveguides
    "launcher",          # Heating launchers
    "antenna",           # RF antennas
    "annular_grid",      # Bolometer grids
    "first_wall",        # First wall contour
    "limiter",           # Limiter geometry
    "divertor",          # Divertor target geometry (coils, structure)
    "vessel",            # Vacuum vessel
    "pf_active",         # Poloidal field coils (structure)
    "pf_passive",        # Passive conductors
})
```

3. **NOT in an exclusion path** (plasma/computational outputs masquerading as geometry):

```python
GEOMETRY_EXCLUSION_PATTERNS: frozenset[str] = frozenset({
    "boundary",          # Plasma boundary (equilibrium output)
    "separatrix",        # Plasma separatrix (equilibrium output)
    "grid",              # Computational grids (not physical structure)
    "ggd",               # General grid description
    "flux_surface",      # Magnetic flux surfaces
})
```

**Why path-pattern instead of leaf-name:**
- Leaf names like `r`, `z`, `phi` appear in BOTH geometry (coil r/z) and physics (position r/z)
- The distinguishing factor is the PARENT: `pf_active/coil/element/geometry/outline/r` is geometry
  but `equilibrium/time_slice/boundary/outline/r` is physics (plasma boundary)
- Path patterns capture ~870 nodes (8.7% of quantities); the old leaf-name set captured only 36

**Plasma shape parameters stay as `physical_quantity`:**
- elongation, triangularity, squareness, tilt, ovality → equilibrium-derived physics outputs
- minor_radius, major_radius → describe the plasma, not the machine
- aspect_ratio → dimensionless ratio derived from equilibrium
- shift (only quantity "shift" in DD = Doppler frequency shift [Hz], not Shafranov)

**IDS distribution of geometry nodes (~870):**
- bolometer: 89% geometry (channel lines of sight, apertures)
- camera_visible: 70% geometry (detector positions, optical paths)
- pf_active: 55% geometry (coil outlines, conductor cross-sections)
- interferometer, polarimeter, reflectometer: high geometry fraction (diagnostic hardware)
- equilibrium, core_profiles: <1% geometry (these are physics IDSs)

#### `refine_from_relationships(current_category, *, has_identifier_schema, is_coordinate_target, children_categories, data_type, unit, path) → str | None`

**New parameter: `path`, `unit`** (needed for Bug 2 fix).

```
R1: has_identifier_schema → identifier  (unchanged)
R2: is_coordinate_target
    IF name is in COORDINATE_SEGMENTS (r, z, phi, rho_tor_norm, time, etc.)
      → coordinate  (canonical coordinate names are always coordinates, even with units)
    ELSE IF has physics unit (not '-', not empty) AND name NOT in COORDINATE_SEGMENTS
      → keep current category (data-storage node with real physics content)
    ELSE IF path segments[:-1] contain "coordinate" (exact match)
      → keep current category (skip — ancestor is already a coordinate container)
    ELSE → coordinate
R3: STRUCTURE+unit validation  (update for split: `current_category in QUANTITY_CATEGORIES` and
    children evidence checks `QUANTITY_CATEGORIES | {"coordinate", "structural"}`)
```

**R3 update detail**: Current code at `node_classifier.py:288` checks
`current_category == "quantity"` and at line 295 checks `c in ("quantity", "coordinate", "structural")`.
After the split, both must use `QUANTITY_CATEGORIES`:
- Entry guard: `current_category in QUANTITY_CATEGORIES` (catches both physical and geometric)
- Child evidence: `c in (QUANTITY_CATEGORIES | {"coordinate", "structural"})` (data-bearing children)

**Migration parity note**: The bottom-up processing order in the migration script (deepest
paths first) ensures that when R3 runs on a parent STRUCTURE node, its children already
have split categories (`physical_quantity`, `geometry`) — not legacy `"quantity"`.
However, as a safety net, the child evidence check should also accept legacy `"quantity"` as
data-bearing (it IS a quantity, just unsplit). This means the actual check should be:
`c in (QUANTITY_CATEGORIES | {"quantity", "coordinate", "structural"})`. The legacy `"quantity"`token will never appear after a fresh build, but prevents R3 from incorrectly demoting parents
if bottom-up ordering is somehow violated during migration.

**Why this two-check approach**: Pure unit-based guard (round 1 fix) was too broad —
real coordinates like `r`, `z`, `phi` have unit `m` but are genuine coordinates.
The fix combines:
1. **Name check first**: canonical coordinate names (already in `COORDINATE_SEGMENTS`)
   are always coordinates, regardless of unit
2. **Unit check second**: non-coordinate names with physics units are data-storage
   nodes that should keep their quantity classification
This is robust because `COORDINATE_SEGMENTS` is already maintained as the single
source of truth for coordinate identification in `node_classifier.py`.

### 4. Dirty Worktree Resolution

**Root cause**: Runaway `--yolo` agent(s) from April 14-15 with stale context applied
develop-branch code onto the main worktree. At least 6 copilot processes are running
against imas-codex simultaneously.

**Resolution strategy** (safe for parallel agents):
1. **Save a diff first**: `git diff -- imas_codex/tools/graph_search.py > /tmp/dirty_graph_search.patch`
   (in case the other agent's changes contain any useful additions)
2. **Kill stale agents**: Identify and kill all `--yolo` copilot processes with stale context
   (especially those from April 14-15 with `--resume`)
3. **Verify no active writers**: `lsof imas_codex/tools/graph_search.py` to confirm no process
   has the file open for writing
4. **Restore committed versions**:
   ```bash
   git checkout HEAD -- imas_codex/tools/graph_search.py
   git checkout HEAD -- imas_codex/llm/search_formatters.py
   git checkout HEAD -- imas_codex/settings.py
   ```
5. **Verify restoration**: `git diff --stat` should show only intentional changes
6. If the dirty patch contains useful work, it can be re-applied AFTER Phase B is confirmed working

**Test files**: Update all 17 hardcoded `'data'` references in test files to use
`SEARCHABLE_CATEGORIES` or the appropriate category constants.

### 5. MCP Tool Augmentation

#### 5a. Category-Aware Search

The committed `graph_search.py` already uses `$categories` parameter. Once the dirty
worktree is resolved, search works with `SEARCHABLE_CATEGORIES`. After the quantity split,
`SEARCHABLE_CATEGORIES = {"physical_quantity", "geometry", "coordinate"}` —
search automatically picks up both.

#### 5b. New `node_category` Filter

Add `node_category` as an optional filter parameter to `search_dd_paths` and `list_dd_paths`:

```python
# New parameter on search_dd_paths, list_dd_paths
node_category: str | None  # "physical_quantity", "geometry", "coordinate", etc.
```

This enables queries like "show me only geometric shape parameters" or "only physics measurements".

#### 5c. Coordinate Metadata in Results

Currently, `fetch_dd_paths` returns coordinate specs as path IDs only. Augment to include:

```
Coordinates: rho_tor_norm (FLT_1D, -) [coordinate], time (FLT_0D, s) [coordinate]
```

Each coordinate reference expands to include data_type, unit, and node_category. This
makes the response self-contained without requiring follow-up queries.

#### 5d. Geometric vs Physical Filter Shorthand

Add a convenience parameter `quantity_type` to search tools:
- `"physical"` → filter to `physical_quantity` only
- `"geometric"` → filter to `geometry` only
- `"all"` (default) → both physical and geometric quantities

### 6. Enrichment Pipeline Assessment

#### Current State

`dd_enrichment.py` has its own classification system (`classify_node()`) with 5 layers:
1. Error/metadata patterns → accessor (template)
2. Force-include physics concepts → concept (LLM)
3. Explicit accessor names → accessor (template)
4. Regex suffix patterns → accessor (template)
5. Frequency heuristic → accessor (template)
6. Default → concept (LLM)

This is SEPARATE from `node_classifier.py`. "Concept" maps roughly to "quantity" and
"accessor" maps roughly to "structural/coordinate/identifier".

#### Unification Recommendation

Do NOT merge the two systems. They serve different purposes:
- `node_classifier.py` → pipeline participation (what to embed/search/extract)
- `dd_enrichment.py` → enrichment strategy (LLM vs template descriptions)

However, the enrichment classifier should CONSULT `node_category` with a **positive
override** for quantity categories — preventing false demotion to "accessor":

```python
def classify_node(path_id, name, node_stats=None, node_category=None):
    # POSITIVE OVERRIDE: quantity categories must stay "concept" (LLM-enriched)
    # This prevents the frequency heuristic or accessor suffix rules from
    # demoting genuine physics/geometry quantities (e.g., "surface" appears
    # in the accessor vocabulary but is a real quantity when node_category says so)
    if node_category and node_category in QUANTITY_CATEGORIES:
        return "concept"
    # NEGATIVE OVERRIDE: non-enrichable categories always get template enrichment
    if node_category and node_category not in ENRICHABLE_CATEGORIES:
        return "accessor"
    # ... existing 5-layer logic for concept vs accessor (unchanged)
```

This ensures:
- Quantities always get LLM enrichment (even if they match accessor patterns)
- Non-enrichable nodes always get template enrichment (even if they match concept patterns)
- When `node_category` is unavailable (e.g., during initial build), the existing logic applies

**Wiring requirement**: The enrichment claim query and worker must fetch and pass
`node_category` to `classify_node()`. Add `n.node_category AS node_category` to the
enrichment claim/batch query, and thread it through the worker's call to `classify_node()`.
Also add an audit log for cases where `node_category in QUANTITY_CATEGORIES` but the
leaf name appears in the accessor vocabulary — this catches classifier false positives
that would be wastefully LLM-enriched.

#### Model Selection

> **Critical finding (2026-04-16):** ALL current DD enrichment was done with
> `google/gemini-3.1-flash-lite-preview` — the cheapest/weakest model available.
> The plan previously framed this as "opus → sonnet downgrade"; the reality is
> `flash-lite → sonnet` = massive quality UPGRADE at negligible cost difference.

**5-Node Pilot Benchmark (real graph writes):**

| Model | Time | Cost | Quality |
|-------|------|------|---------|
| gemini-3.1-flash-lite | 2.6s | $0.019 | Adequate — generic, thin descriptions |
| claude-sonnet-4.6 | 14.4s | $0.021 | Excellent — precise physics context, practical |
| claude-opus-4.6 | 25.7s | $0.021 | Excellent — most detailed, includes definitions |

**Example: triangularity description**
- **flash-lite:** "measures the degree of plasma indentation" — vague
- **sonnet:** "quantifying the D-shape asymmetry... key shaping parameter influencing MHD stability, confinement, and power exhaust" — precise
- **opus:** "defined as the horizontal offset of the maximum vertical extent from the geometric axis normalized by the minor radius" — includes formula

**Cost projection for full re-enrichment:**

| Scenario | Nodes | Cost | Time |
|----------|-------|------|------|
| Sonnet (all enrichable) | 11,699 | ~$50 | ~3 hours |
| Flash-lite (all enrichable) | 11,699 | ~$45 | ~1 hour |
| Marginal cost of sonnet | — | ~$5 | ~2 hours |

**Recommendation**: Switch to `claude-sonnet-4.6` for DD enrichment. The $5 marginal
cost buys dramatically better descriptions that directly improve:
1. Embedding quality (better descriptions → better vectors → better search)
2. Standard name generation (enriched descriptions are compose prompt context)
3. MCP tool responses (descriptions surfaced to users)

Use a DD-specific model override for enrichment, NOT a global change:
1. **CLI override** (preferred): `uv run imas-codex imas dd enrich --model openrouter/anthropic/claude-sonnet-4.6`
2. **Config section**: Add `[tool.imas-codex.dd-enrichment]` with its own `model` key

Hash-based idempotency (`compute_enrichment_hash(context_text, model_name)`) means
changing the model automatically invalidates the cache, forcing re-enrichment of all
nodes — which is exactly what we want.

#### Multi-Pass Enrichment Assessment

The discovery paths pipeline uses a two-pass pattern (triage → detailed scoring) that
produces high-quality results. The same multi-pass principle is applied at the SN pipeline
level in the [SN Greenfield Pipeline](standard-names/28-sn-greenfield-pipeline.md), which
separates naming (LLM call #1) from documentation (LLM call #2) with dynamic context
retrieval between them.

For DD enrichment specifically:

**Pass 1** (current): Generate description + keywords from path context
**Pass 2** (future): Refine description using sibling descriptions, cluster membership,
and coordinate relationships as additional context

**Verdict**: Defer multi-pass DD enrichment to post-reclassification. The immediate
priority is fixing classification and switching to sonnet. The model upgrade alone
(flash-lite → sonnet) provides a dramatic quality improvement. Multi-pass DD enrichment
is a separate optimization for a future iteration — the SN pipeline's multi-pass design
addresses the quality gap where it matters most (standard name documentation).

### 7. Two Classification Systems: Reconciliation Path

| Feature | `node_classifier.py` | `dd_enrichment.py` |
|---------|---------------------|---------------------|
| Purpose | Pipeline participation | LLM vs template enrichment |
| Input | XML attributes + graph rels | Path name + frequency stats |
| Output | 7 categories | concept / accessor |
| Used by | Build, search, SN extract, embed | Enrichment only |

**Reconciliation**: The enrichment classifier's `FORCE_INCLUDE_CONCEPTS` set should be
reviewed to ensure geometry nodes (identified by path patterns, not leaf names) are
still force-included as concepts (not demoted to accessors).

No code merge needed — just ensure the enrichment classifier consults `node_category`
(see section 6 above).

---

## TDD Test Strategy

### Test File: `tests/core/test_node_classifier.py`

```python
# Parametrized tests for classify_from_attributes()
@pytest.mark.parametrize("path,name,dt,unit,parent_dt,expected", [
    # Error fields
    ("eq/ts/p1d/psi_error_upper", "psi_error_upper", "FLT_1D", "Wb", None, "error"),
    # Metadata
    ("core_profiles/ids_properties/comment", "comment", "STR_0D", None, None, "metadata"),
    # Physical quantity (unitful)
    ("core_profiles/profiles_1d/electrons/temperature", "temperature", "FLT_1D", "eV", None, "physical_quantity"),
    ("eq/ts/profiles_1d/pressure", "pressure", "FLT_1D", "Pa", None, "physical_quantity"),
    # Geometry (hardware/diagnostic structure — path-pattern based)
    ("pf_active/coil/element/geometry/outline/r", "r", "FLT_1D", "m", None, "geometry"),
    ("bolometer/channel/line_of_sight/first_point/r", "r", "FLT_0D", "m", None, "geometry"),
    ("camera_visible/channel/detector/geometry/outline/r", "r", "FLT_1D", "m", None, "geometry"),
    ("interferometer/channel/line_of_sight/first_point/phi", "phi", "FLT_0D", "rad", None, "geometry"),
    # Plasma shape params → physical_quantity (NOT geometry — equilibrium outputs)
    ("eq/ts/boundary/elongation", "elongation", "FLT_0D", "-", None, "physical_quantity"),
    ("eq/ts/boundary/triangularity_upper", "triangularity_upper", "FLT_0D", "-", None, "physical_quantity"),
    ("eq/ts/boundary/minor_radius", "minor_radius", "FLT_0D", "m", None, "physical_quantity"),
    ("eq/ts/boundary/major_radius", "major_radius", "FLT_0D", "m", None, "physical_quantity"),
    # Plasma boundary outline → physical_quantity (NOT geometry — equilibrium output)
    ("eq/ts/boundary/outline/r", "r", "FLT_1D", "m", None, "physical_quantity"),
    # Physics dimensionless (NOT geometric — regression)
    ("eq/ts/gq/beta_tor", "beta_tor", "FLT_0D", "-", None, "physical_quantity"),
    ("eq/ts/p1d/q", "q", "FLT_1D", "-", None, "physical_quantity"),
    ("core_profiles/profiles_1d/zeff", "zeff", "FLT_1D", "-", None, "physical_quantity"),
    ("eq/ts/gq/li_3", "li_3", "FLT_0D", "-", None, "physical_quantity"),
    # Ambiguous terms (default to physical_quantity — regression)
    ("eq/ts/gq/volume", "volume", "FLT_0D", "m^3", None, "physical_quantity"),
    ("eq/ts/gq/area", "area", "FLT_0D", "m^2", None, "physical_quantity"),
    ("eq/ts/gq/surface", "surface", "FLT_0D", "m^2", None, "physical_quantity"),
    ("eq/ts/gq/length", "length", "FLT_0D", "m", None, "physical_quantity"),
    ("eq/ts/gq/perimeter", "perimeter", "FLT_0D", "m", None, "physical_quantity"),
    # Coordinate
    ("core_profiles/profiles_1d/grid/rho_tor_norm", "rho_tor_norm", "FLT_1D", "-", None, "coordinate"),
    ("eq/ts/time", "time", "FLT_0D", "s", None, "coordinate"),
    # Structural (data-storage under STRUCTURE parent)
    ("magnetics/flux_loop/flux/data", "data", "FLT_1D", "Wb", "STRUCTURE", "structural"),
    # Containers (NOT geometric even if name looks it)
    # geometric_axis, strike_point, x_point are STRUCTURE → structural
    ("eq/ts/gq/geometric_axis", "geometric_axis", "STRUCTURE", None, None, "structural"),
    ("eq/ts/boundary/strike_point", "strike_point", "STRUCTURE", None, None, "structural"),
    # Identifier
    # ... (tested via refine_from_relationships)
])
def test_classify_from_attributes(path, name, dt, unit, parent_dt, expected):
    result = classify_from_attributes(path, name, data_type=dt, unit=unit, parent_data_type=parent_dt)
    assert result == expected

# Parametrized tests for refine_from_relationships()
@pytest.mark.parametrize("current,kwargs,expected", [
    # R1: identifier schema overrides
    ("physical_quantity", {"has_identifier_schema": True}, "identifier"),
    # R2: coordinate target with physics unit but NOT a coordinate name → keep
    ("physical_quantity", {"is_coordinate_target": True, "path": "eq/ts/p1d/psi/values",
     "unit": "Wb", "name": "values"}, None),
    # R2: coordinate target that IS a canonical coordinate name → coordinate (even with unit)
    ("physical_quantity", {"is_coordinate_target": True, "path": "eq/ts/p1d/rho_tor_norm",
     "unit": "-", "name": "rho_tor_norm"}, "coordinate"),
    # R2: real coordinate with unit m → still coordinate (r, z, phi are always coordinates)
    ("physical_quantity", {"is_coordinate_target": True, "path": "eq/ts/boundary/outline/r",
     "unit": "m", "name": "r"}, "coordinate"),
    ("physical_quantity", {"is_coordinate_target": True, "path": "eq/ts/boundary/outline/z",
     "unit": "m", "name": "z"}, "coordinate"),
    ("physical_quantity", {"is_coordinate_target": True, "path": "wall/description/limiter/unit/outline/phi",
     "unit": "rad", "name": "phi"}, "coordinate"),
    # R2: ancestor coordinate exception
    ("physical_quantity", {"is_coordinate_target": True, "path": "ggd/grid/space/coordinate/r",
     "unit": "m", "name": "r"}, "coordinate"),
    # Bug 2 regression: R2 should NOT reclassify data nodes with units
    ("physical_quantity", {"is_coordinate_target": True, "path": "eq/ts/p1d/phi/radial",
     "unit": "Wb", "name": "radial"}, None),
    # State transition: reclassified coordinate → embeddable quantity
    ("coordinate", {"has_identifier_schema": False, "is_coordinate_target": False}, None),
])
def test_refine_from_relationships(current, kwargs, expected):
    result = refine_from_relationships(current, **kwargs)
    assert result == expected
```

### Test File: `tests/core/test_node_categories.py`

```python
def test_quantity_categories_includes_both():
    assert "physical_quantity" in QUANTITY_CATEGORIES
    assert "geometry" in QUANTITY_CATEGORIES

def test_embeddable_equals_quantity():
    assert EMBEDDABLE_CATEGORIES == QUANTITY_CATEGORIES

def test_searchable_includes_coordinate():
    assert SEARCHABLE_CATEGORIES == QUANTITY_CATEGORIES | {"coordinate"}

def test_old_quantity_not_in_any_set():
    """Ensure 'quantity' (old monolithic) is not present."""
    for cat_set in [EMBEDDABLE_CATEGORIES, SEARCHABLE_CATEGORIES, SN_SOURCE_CATEGORIES]:
        assert "quantity" not in cat_set

def test_r3_structure_unit_with_quantity_children():
    """R3 must use QUANTITY_CATEGORIES, not 'quantity' literal.
    
    A STRUCTURE+unit parent whose children are physical_quantity/geometry
    should remain a quantity (not be demoted to structural). This proves that the
    classifier works correctly with split categories — parity between fresh build
    and migration.
    """
    # Parent: STRUCTURE with unit, children are split quantities
    cat = classify_from_attributes(
        "equilibrium/time_slice/boundary",
        "boundary", data_type="STRUCTURE", unit="m",
        parent_data_type="STRUCTURE",
    )
    result = refine_from_relationships(
        cat,
        has_identifier_schema=False,
        is_coordinate_target=False,
        children_categories=["physical_quantity", "geometry", "coordinate"],
        data_type="STRUCTURE",
        unit="m",
    )
    # Should keep quantity (not demoted to structural) because children are data-bearing
    assert result is None or result in QUANTITY_CATEGORIES

def test_r3_structure_unit_with_legacy_quantity_children():
    """Migration parity: R3 must NOT depend on legacy 'quantity' child category.
    
    After bottom-up migration, children should already be split. If somehow a
    legacy 'quantity' child remains, R3 should still treat it as data-bearing.
    """
    result = refine_from_relationships(
        "physical_quantity",
        has_identifier_schema=False,
        is_coordinate_target=False,
        children_categories=["quantity"],  # legacy — should not break
        data_type="STRUCTURE",
        unit="m",
    )
    # Must not demote to structural — 'quantity' is a data-bearing child
    assert result is None or result in QUANTITY_CATEGORIES
```

### Test File: `tests/graph/test_schema_compliance.py`

Existing schema compliance tests are parametrized from the schema. After updating the
LinkML YAML, rebuilding models, and running `uv run pytest tests/graph/test_schema_compliance.py`
will verify that all graph nodes have valid `node_category` values.

### Regression: No Legacy Category Strings in Production Code

```python
def test_no_hardcoded_data_or_quantity_categories():
    """Ensure production code does not hardcode 'data' or 'quantity' as category values.
    
    After the split, all category checks should use constants from node_categories.py.
    This test greps the source to catch regressions.
    """
    import re
    from pathlib import Path
    
    src = Path("imas_codex")
    violations = []
    skip_files = {"node_classifier.py", "node_categories.py"}  # these define the constants
    
    for py in src.rglob("*.py"):
        if py.name in skip_files:
            continue
        text = py.read_text()
        # Check for hardcoded category string comparisons
        for pattern in [
            r"node_category\s*==?\s*['\"]data['\"]",
            r"node_category\s*==?\s*['\"]quantity['\"]",
            r"_classify_node\([^)]+\)\s*==\s*['\"]data['\"]",
        ]:
            for match in re.finditer(pattern, text):
                violations.append(f"{py}:{match.group()}")
    
    assert violations == [], f"Legacy category hardcoding found:\n" + "\n".join(violations)
```

### Integration Tests (Graph-Backed)

```python
# In tests/search/ — update all 'data' references
# Replace: WHERE p.node_category = 'data'
# With: WHERE p.node_category IN $categories
# Parameter: categories=list(SEARCHABLE_CATEGORIES)
```

---

## Graph Recovery Strategy

### Step 1: Fix Dirty Worktree

```bash
# Restore Phase B version of graph_search.py
git checkout HEAD -- imas_codex/tools/graph_search.py
# Check other dirty files — assess if they should be restored or committed
git diff HEAD --stat
```

### Step 2: Apply Schema + Classifier Changes

1. Update `imas_dd.yaml`: Replace `quantity` with `physical_quantity` + `geometry`
2. Update `node_classifier.py`: Rename functions, add geometric rules, fix Bug 2 path parameter
3. Update `node_categories.py`: Add `QUANTITY_CATEGORIES`, update all sets
4. Update `build_dd.py`:
   - Fix Bug 1: `PARENT_OF` → `HAS_PARENT` in `_reclassify_relational`
   - **Rename call sites**: update import at ~line 2020 from `classify_node_pass2` →
     `refine_from_relationships`. Update all 3 call sites (~lines 2065, 2073, 2088) to
     use the new name and pass `path=row["id"]` (needed for Bug 2 fix in R2).
   - **Update `_classify_node()` wrapper** (~line 3061): import `classify_from_attributes`
     instead of `classify_node_pass1`. Wrapper stays as a convenience for embed filtering.
   - Update `_reclassify_relational()` identifier/coord override queries (~lines 2026, 2035):
     replace `['quantity', 'coordinate', 'structural', 'data']` with `QUANTITY_CATEGORIES | {"coordinate", "structural"}`
     (these passes must still examine coordinate and structural nodes for reclassification)
   - Update `_reclassify_relational()` STRUCTURE+unit validation query (~line 2047):
     replace `['quantity', 'data']` with `list(QUANTITY_CATEGORIES)`
   - Update `phase_embeddings()`: replace `_classify_node(...) == "data"` with `EMBEDDABLE_CATEGORIES` check
5. Update `clusters/preprocessing.py`: replace `_classify_node(path, name) != "data"` with constant-based check
6. Rebuild models: `uv run build-models --force`

### Step 3: One-Off Graph Migration

**Strategy: Python-driven, not Cypher-duplicated.** RD review correctly identified
that duplicating classifier logic in Cypher is error-prone. Instead, run the full
two-step Python classifier (attributes + relationships) over affected nodes.

```python
# migration_classify.py — run via: uv run python migration_classify.py
# (disposable script, do NOT commit)
from imas_codex.graph.client import GraphClient
from imas_codex.core.node_classifier import classify_from_attributes, refine_from_relationships
from imas_codex.core.node_categories import QUANTITY_CATEGORIES

def fetch_relational_context(gc, node_id):
    """Fetch the relational facts needed by refine_from_relationships.
    
    Direction matters:
    - HAS_PARENT: (child)-[:HAS_PARENT]->(parent) — we want n's parent
    - HAS_COORDINATE: (array)-[:HAS_COORDINATE]->(coord_target) — incoming means n IS a coordinate target
    - HAS_IDENTIFIER_SCHEMA: (n)-[:HAS_IDENTIFIER_SCHEMA]->(schema)
    - Children: (child)-[:HAS_PARENT]->(n) — nodes whose parent is n
    """
    row = gc.query("""
        MATCH (n:IMASNode {id: $id})
        OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
        OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(n)
        WITH n, parent,
             EXISTS { (n)-[:HAS_IDENTIFIER_SCHEMA]->() } AS has_id_schema,
             EXISTS { (:IMASNode)-[:HAS_COORDINATE]->(n) } AS is_coord_target,
             collect(DISTINCT child.node_category) AS children_categories
        RETURN parent.data_type AS parent_data_type,
               has_id_schema, is_coord_target,
               children_categories
    """, id=node_id)
    return row[0] if row else {}

def full_classify(gc, node):
    """Run both classification steps — mirrors build_dd.py exactly."""
    # Step 1: attribute-based
    ctx = fetch_relational_context(gc, node["id"])
    cat = classify_from_attributes(
        node["id"], node["name"],
        data_type=node["data_type"],
        unit=node["unit"],
        parent_data_type=ctx.get("parent_data_type"),
    )
    # Step 2: relationship-based refinement (full contract)
    refined = refine_from_relationships(
        cat,
        has_identifier_schema=ctx.get("has_id_schema", False),
        is_coordinate_target=ctx.get("is_coord_target", False),
        path=node["id"],
        unit=node["unit"],
        name=node["name"],
        data_type=node["data_type"],
        children_categories=ctx.get("children_categories", []),
    )
    return refined if refined is not None else cat

reclassified = []

def apply_one(gc, node_id, new_cat):
    """Apply a single reclassification immediately (for bottom-up consistency)."""
    gc.query("""
        MATCH (n:IMASNode {id: $id})
        SET n.node_category = $cat, n._migration_reclassified = true
    """, id=node_id, cat=new_cat)

def apply_batch(gc, batch):
    """Write a batch of reclassifications to the graph immediately."""
    for node_id, new_cat in batch:
        apply_one(gc, node_id, new_cat)

with GraphClient() as gc:
    # Phase 1: Re-classify ALL nodes currently labeled 'quantity'
    # Sort by path depth (deepest first = bottom-up) so that when R3
    # checks children_categories, child nodes already have split values.
    nodes = list(gc.query("""
        MATCH (n:IMASNode)
        WHERE n.node_category = 'quantity'
        RETURN n.id AS id, n.data_type AS data_type, n.unit AS unit,
               split(n.id, '/')[-1] AS name, size(split(n.id, '/')) AS depth
        ORDER BY depth DESC
    """))
    phase1_count = 0
    for node in nodes:
        new_cat = full_classify(gc, node)
        if new_cat != 'quantity':
            # Apply immediately — parent nodes processed later will see
            # updated child categories via fetch_relational_context()
            apply_one(gc, node["id"], new_cat)
            reclassified.append((node["id"], new_cat))
            phase1_count += 1
    print(f"Phase 1: {phase1_count} quantity nodes reclassified (bottom-up)")

    # Phase 2: Fix Bug 1 — structural nodes under quantity parents
    # Now safe to query parent.node_category — phase 1 results are applied
    bug1_nodes = list(gc.query("""
        MATCH (n:IMASNode)-[:HAS_PARENT]->(parent:IMASNode)
        WHERE n.node_category = 'structural'
          AND parent.node_category IN $qty_cats
          AND (n.data_type STARTS WITH 'FLT' OR n.data_type STARTS WITH 'CPX')
        RETURN n.id AS id, n.data_type AS data_type, n.unit AS unit,
               split(n.id, '/')[-1] AS name
    """, qty_cats=list(QUANTITY_CATEGORIES)))
    phase2_batch = []
    for node in bug1_nodes:
        new_cat = full_classify(gc, node)
        if new_cat != 'structural':
            phase2_batch.append((node["id"], new_cat))
    
    apply_batch(gc, phase2_batch)
    reclassified.extend(phase2_batch)
    print(f"Phase 2: {len(phase2_batch)} structural→quantity nodes reclassified")

    # Phase 3: Fix Bug 2 — coordinate targets that should be quantities
    bug2_nodes = list(gc.query("""
        MATCH (n:IMASNode)<-[:HAS_COORDINATE]-(:IMASNode)
        WHERE n.node_category = 'coordinate'
          AND n.unit IS NOT NULL AND n.unit <> '-' AND n.unit <> ''
        RETURN n.id AS id, n.data_type AS data_type, n.unit AS unit,
               split(n.id, '/')[-1] AS name
    """))
    phase3_batch = []
    for node in bug2_nodes:
        new_cat = full_classify(gc, node)
        if new_cat != 'coordinate':
            phase3_batch.append((node["id"], new_cat))
    
    apply_batch(gc, phase3_batch)
    reclassified.extend(phase3_batch)
    print(f"Phase 3: {len(phase3_batch)} coordinate→quantity nodes reclassified")

    # Summary
    from collections import Counter
    summary = Counter()
    for _, new_cat in reclassified:
        summary[new_cat] += 1

    print(f"Migration complete: {len(reclassified)} nodes reclassified")
    for cat, count in sorted(summary.items()):
        print(f"  {cat}: {count}")
```

**Why Python instead of Cypher**: The Python classifier is the single source of truth.
Running both `classify_from_attributes()` and `refine_from_relationships()` (with the
full contract: `data_type`, `children_categories`, identifier/coordinate flags) over
candidate nodes guarantees consistency between the build pipeline and the migration.
Note: `parent_data_type` is fetched via `HAS_PARENT` join (not stored on IMASNode).

**Why phases are applied sequentially**: Phase 2 queries `parent.node_category IN $qty_cats`
which includes the newly split `physical_quantity` and `geometry` values. If
phase 1 reclassifications aren't written before phase 2 queries, the parent lookup would
fail to match — those parents would still show `quantity` (the old unsplit value).

### Step 4: Reset Reclassified Nodes for Re-Enrichment

**Critical**: Reset ALL nodes whose category changed, regardless of current status.
Nodes reclassified from `structural` or `coordinate` to a quantity category need
full re-enrichment and re-embedding — even if they already have descriptions
(generated under the wrong classification context).

```cypher
// Reset ALL reclassified nodes — includes 'built', 'enriched', AND 'embedded'
// Field set is a superset of _RESET_CLEAR_FIELDS["built"] from dd_graph_ops.py (lines 362-374)
// PLUS embedding_text (not in the dict but needed for complete vector cleanup) and
// claimed_at/claim_token (handled separately by reset_imas_nodes at lines 453-457).
MATCH (n:IMASNode)
WHERE n._migration_reclassified = true
SET n.status = 'built',
    n.description = null,
    n.keywords = null,
    n.enrichment_hash = null,
    n.enrichment_model = null,
    n.enrichment_source = null,
    n.enriched_at = null,
    n.physics_domain = null,
    n.embedding = null,
    n.embedding_hash = null,
    n.embedded_at = null,
    n.embedding_text = null,
    n.claimed_at = null,
    n.claim_token = null,
    n._migration_reclassified = null
RETURN count(n) AS reset_for_enrichment;
```

The migration script (Step 3) sets `n._migration_reclassified = true` on every
node it reclassifies. This temporary marker ensures we reset exactly the affected
nodes. The reset clears ALL enrichment and embedding state — a superset of
`_RESET_CLEAR_FIELDS["built"]` from `dd_graph_ops.py` (lines 362–374): the dict
fields (`description`, `keywords`, `enrichment_hash`, `enrichment_model`,
`enrichment_source`, `enriched_at`, `physics_domain`, `embedding`, `embedding_hash`,
`embedded_at`) plus `embedding_text` (not in the dict but needed for complete
vector cleanup) and `claimed_at`/`claim_token` (handled separately by
`reset_imas_nodes()` at lines 453–457). This is critical because nodes
reclassified from a quantity category to `structural` or `identifier` must not
retain stale `description`, `keywords`, or `physics_domain` generated under the
wrong classification context. Without clearing `embedding_hash`, embed workers
could skip on a stale hash match and mark nodes as embedded with no vector.

### Step 5: Re-Run Enrichment + Embedding

```bash
# Re-enrich nodes that were reclassified (status=built)
uv run imas-codex imas dd enrich --version 4.0.0

# Re-embed nodes that were enriched (status=enriched)
uv run imas-codex imas dd embed --version 4.0.0
```

**Cost estimate**: ~2,289 nodes × ~$0.01/node ≈ $23 for enrichment. Embedding is GPU-only (free).

### Step 6: Update Test Suite

Replace all 17 hardcoded `'data'` references in test files with constants from
`node_categories.py`. The test assertions should check for `IN $categories` patterns,
not specific category values.

---

## Execution Order

| Step | What | Depends On | Risk |
|------|------|------------|------|
| 0 | Kill stale agents, save dirty patches | — | Low |
| 1 | Restore dirty worktree (`git checkout HEAD -- ...`) | 0 | Low: restores committed code |
| 2 | Write TDD tests (test_node_classifier.py, test_node_categories.py) | — | Low: tests only |
| 3 | Update schema (imas_dd.yaml): physical_quantity + geometry | — | Low: schema only |
| 4 | Rebuild models (`uv run build-models --force`) | 3 | Low: auto-generated |
| 5 | Update node_classifier.py (rename, add geometric rules, fix R2) | 2, 3 | Medium: core logic |
| 6 | Update node_categories.py (QUANTITY_CATEGORIES, update sets) | 3 | Low: constants |
| 7 | Fix build_dd.py: Bug 1 + ALL legacy category predicates. Two distinct sets: identifier/coord override → `QUANTITY_CATEGORIES \| {coordinate, structural}`, STRUCTURE+unit → `QUANTITY_CATEGORIES`, embed filter → `EMBEDDABLE_CATEGORIES` | 6 | Medium: multiple sites |
| 7a | Update clusters/preprocessing.py: replace `_classify_node != "data"` with constants | 6 | Low: one predicate |
| 8 | Update dd_enrichment.py (positive override), dd_graph_ops.py (return node_category), dd_workers.py (thread through partitioning) | 6 | Low: guard clause + field addition |
| 9 | Run TDD tests — all should pass | 2–8 | — |
| 10 | Update graph_search.py: ensure `SEARCHABLE_CATEGORIES` (not `QUANTITY_CATEGORIES`) at search boundaries | 6 | Low: already parameterized |
| 11 | Update test files (remove 'data' hardcoding). Delete `tests/graph/test_node_category.py` (superseded by new TDD tests) | 6 | Low: string replacement + delete |
| 12 | Run full test suite | 9–11 | — |
| 13 | Graph migration (Python-driven, disposable script) | 3–8 | Medium: data migration |
| 14 | Reset reclassified nodes + re-enrich + re-embed | 13 | Medium: LLM cost ~$23 |
| 15 | MCP tool augmentation (node_category filter, coordinate metadata). Thread new optional `node_category` param through: `server.py` tool signatures (`search_dd_paths` ~line 2483, `list_dd_paths` ~line 2660) → `graph_search.py` backing methods (`GraphSearchTool.search_dd_paths` ~line 229, `GraphListTool.list_dd_paths` ~line 1017). Also update `ids/tools.py` wrapper layer for parity. | 6, 10 | Low: additive |
| 16 | DD enrichment model override (CLI or config) | — | Low: config change |

**Key dependency note**: Step 13 (migration) depends on steps 3–8 ALL being complete
and tested. The migration runs the Python classifier which must be updated first.
Step 10 must use `SEARCHABLE_CATEGORIES` (includes `coordinate`), not `QUANTITY_CATEGORIES`,
because coordinates remain searchable even though they are not embedded or SN-extracted.

---

## Files to Modify

| File | Change |
|------|--------|
| `imas_codex/schemas/imas_dd.yaml` | Replace `quantity` with `physical_quantity` + `geometry` |
| `imas_codex/core/node_classifier.py` | Rename functions, add `GEOMETRIC_QUANTITY_NAMES`, add geometric rules, fix R2 with name+unit two-check guard |
| `imas_codex/core/node_categories.py` | Add `QUANTITY_CATEGORIES`, update all sets |
| `imas_codex/graph/build_dd.py` | Fix Bug 1: `PARENT_OF` → `HAS_PARENT` in `_reclassify_relational`. Update ALL legacy category predicates with **two distinct sets**: (a) identifier/coordinate override queries (lines ~2026, ~2035) → `QUANTITY_CATEGORIES \| {"coordinate", "structural"}` (these passes must still examine coordinate and structural nodes for reclassification), (b) STRUCTURE+unit validation query (line ~2047) → `QUANTITY_CATEGORIES` only. Update `phase_embeddings()` filter → `EMBEDDABLE_CATEGORIES`. |
| `imas_codex/graph/dd_enrichment.py` | Add positive override: quantity categories → always "concept" |
| `imas_codex/graph/dd_graph_ops.py` | Return `p.node_category AS node_category` from `claim_paths_for_enrichment()` step-2 query (~line 88) |
| `imas_codex/graph/dd_workers.py` | Thread `node_category` through template/LLM partitioning: quantity categories → always LLM path (override `is_accessor_terminal`) |
| `imas_codex/clusters/preprocessing.py` | Replace `_classify_node(path, name) != "data"` with `node_category not in EMBEDDABLE_CATEGORIES` (or equivalent constant check). This file filters paths for cluster generation — dropping all quantity nodes would break clusters. |
| `imas_codex/tools/graph_search.py` | Restore from git (Phase B). Already uses `SEARCHABLE_CATEGORIES` (committed at ef0b049a). Add optional `node_category` filter param to `GraphSearchTool.search_dd_paths()` (~line 229) and `GraphListTool.list_dd_paths()` (~line 1017). When provided, narrows the category filter beyond `SEARCHABLE_CATEGORIES` (e.g., `physical_quantity` only). This is the actual MCP execution layer — `server.py` delegates directly here. |
| `imas_codex/llm/server.py` | Thread `node_category` optional param through `search_dd_paths` (~line 2483) and `list_dd_paths` (~line 2660) MCP tool signatures. Pass through to `GraphSearchTool`/`GraphListTool` methods in `graph_search.py`. |
| `imas_codex/ids/tools.py` | Accept `node_category` filter param in wrapper functions for parity (~lines 74, 134). Separate from MCP execution path but used by non-MCP callers. |
| `imas_codex/llm/search_formatters.py` | Restore from git if dirty |
| `imas_codex/settings.py` | Restore from git if dirty |
| `tests/core/test_node_classifier.py` | New: TDD tests for classifier |
| `tests/core/test_node_categories.py` | New: TDD tests for constants |
| `tests/search/*.py` (6 files) | Replace `'data'` → `SEARCHABLE_CATEGORIES` |
| `tests/test_imas_search_remediation.py` | Replace `'data'` assertions |
| `tests/graph/test_vector_search.py` | Replace `'data'` references |
| `tests/graph/test_node_category.py` | **Delete**: superseded by `tests/core/test_node_classifier.py` + `tests/core/test_node_categories.py`. Legacy file tests a defunct 3-value classification (`error`/`metadata`/`data`) that no longer exists. |

---

## Open Questions

1. **Enrichment model switch timing**: Should the model switch happen with the quantity split
   (Step 16) or as a separate follow-up? Using a DD-specific model override avoids invalidating
   all enrichment hashes globally, but still re-enriches reclassified nodes.

2. **Other agent's graph_search.py rewrite**: The dirty worktree contains a major rewrite.
   Should we attempt to merge it with Phase B after restoration, or discard it entirely?

3. **Post-migration audit**: After migration, report the top remaining `physical_quantity`
   names with unit `-` or `m` and geometry-heavy docs/clusters. This identifies candidates
   for safely expanding `GEOMETRIC_QUANTITY_NAMES` in a future iteration. The current 17-name
   set is intentionally conservative (high precision, acceptable recall for v1).

## RD Review History

### Round 1 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | GEOMETRIC_QUANTITY_NAMES too broad | Blocking | Removed `surface`, `length`, `volume`, `area`, `perimeter`, `geometric_axis`, `strike_point`, `x_point`. Set now 17 entries (all high confidence). |
| 2 | Rename is clear | Non-blocking | Kept as-is |
| 3 | Execution order under-specified | Blocking | Added Step 0, Step 8. Migration now depends on 3-8. Clarified SEARCHABLE vs QUANTITY at search boundaries. |
| 4 | `git checkout HEAD --` unsafe with parallel agents | Blocking | Added save-patch-first + kill-agents + lsof-verify protocol |
| 5 | Enrichment positive override needed | Non-blocking | Added dual guard: positive (quantities → concept) + negative (non-enrichable → accessor) |
| 6 | Model switch affects all subsystems | Blocking | Changed to DD-specific CLI override or config section, not global change |
| 7 | R2 data-storage fix too brittle | Blocking | Replaced 3-ending whitelist with unit-based guard (has physics unit → keep) |
| 8 | Migration Cypher has bugs | Blocking | Replaced Cypher-based migration with Python-driven script using the classifier |
| 9 | Reset query wrong | Blocking | Reset ALL reclassified nodes (migration marker), not just null-description |
| 10 | TDD coverage gaps | Non-blocking | Added: ambiguous names, container regression, state transitions, Bug 2 regression |

### Round 2 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | R2 unit-guard too broad for real coordinates (r, z, phi) | Blocking | Added two-check: COORDINATE_SEGMENTS name check first, then unit guard. Real coordinates always classified as coordinates. |
| 2 | Migration script only calls classify_from_attributes | Blocking | Added full_classify() that runs both classify_from_attributes + refine_from_relationships with relational context |
| 3 | Script uses nonexistent parent_data_type property | Blocking | Added fetch_relational_context() that joins through HAS_PARENT for parent.data_type |
| 4 | Reset excludes embedded status, misses embedding fields | Blocking | Include all statuses; clear embedding_hash + embedding_text too |
| 5 | Enrichment override not wired through claims | Non-blocking | Added wiring requirement: fetch node_category in claim query, thread to classify_node, add audit log |
| 6 | pyproject.toml in Files to Modify contradicts DD-specific model | Non-blocking | Removed pyproject.toml entry |

### Round 3 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Phase 2 migration query can't see newly split quantity parents | Blocking | Applied phase 1 reclassifications to graph before phase 2 query. Added `apply_batch()` helper with per-phase sequential application. |
| 2 | `is_coord_target` HAS_COORDINATE direction reversed | Blocking | Fixed direction: `(:IMASNode)-[:HAS_COORDINATE]->(n)` — incoming means n IS a coordinate target. Added Cypher direction documentation comment. |
| 3 | `full_classify()` missing children_categories and data_type | Blocking | Added children_categories collection via `(child)-[:HAS_PARENT]->(n)` join. Passed `data_type` and `children_categories` to `refine_from_relationships()` for full contract parity with build_dd.py. |

### Round 4 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | R3 uses `"quantity"` literal — broken after split | Blocking | Updated R3 to use `QUANTITY_CATEGORIES` for entry guard and child evidence. Added legacy `"quantity"` to child evidence as safety net during migration. Bottom-up processing order ensures children are split before parents. Added 2 TDD regression tests for migration parity. |

### Round 5 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `build_dd.py` under-scoped — hardcoded category lists in reclassify + embed filter | Blocking | Expanded plan: update ALL legacy category predicates in `_reclassify_relational()`, `phase_embeddings()`, and `_classify_node()` checks. Added to Files to Modify and execution order. |
| 2 | `clusters/preprocessing.py` hardcodes `"data"` filter — drops all quantities | Blocking | Added `clusters/preprocessing.py` to Files to Modify. Replace `_classify_node != "data"` with `EMBEDDABLE_CATEGORIES` constant check. Added regression test for no-hardcoded-categories sweep. |

### Round 6 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `_reclassify_relational()` replacement too narrow — identifier/coord override needs coordinate+structural too | Blocking | Split into two distinct category sets: (a) identifier/coord override → `QUANTITY_CATEGORIES \| {"coordinate", "structural"}`, (b) STRUCTURE+unit validation → `QUANTITY_CATEGORIES` only. Updated Files to Modify, execution order step 7, and Step 2 with line-specific guidance. |

### Round 7 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `build_dd.py` call sites not updated for function rename (`classify_node_pass1/2` → `classify_from_attributes/refine_from_relationships`) | Blocking | Added explicit call site updates: import rename at ~line 2020, 3 `classify_node_pass2` call sites → `refine_from_relationships` with `path=` arg, `_classify_node()` wrapper → `classify_from_attributes` import. 5 total call sites documented with line numbers. |

### Round 8 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `tests/graph/test_node_category.py` hardcodes legacy `"data"` — not in plan | Blocking | Added to Files to Modify: delete file (superseded by `tests/core/test_node_classifier.py` + `tests/core/test_node_categories.py`). Updated execution order step 11. |

### Round 9 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Enrichment override not wired: `dd_graph_ops.py` doesn't return `node_category`, `dd_workers.py` doesn't use it for partitioning | Blocking | Added `dd_graph_ops.py` and `dd_workers.py` to Files to Modify. Claim query returns `node_category`, worker threads it through template/LLM partitioning (quantity categories → always LLM). Updated execution order step 8. |

### Round 10 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | MCP augmentation not fully wired — `server.py` and `ids/tools.py` tool entry surfaces omitted | Blocking | Corrected MCP execution path: `server.py` → `graph_search.py` (direct delegation, not via `ids/tools.py`). Added `server.py` (~2483, ~2660), `graph_search.py` (`GraphSearchTool.search_dd_paths` ~229, `GraphListTool.list_dd_paths` ~1017), and `ids/tools.py` (wrapper parity) to Files to Modify and Step 15. |
| 2 | Step 4 reset leaves stale enrichment metadata on nodes reclassified to non-enrichable categories | Blocking | Expanded Step 4 Cypher to clear ALL enrichment fields: a superset of `_RESET_CLEAR_FIELDS["built"]` (lines 362-374) plus `embedding_text` and `claimed_at`/`claim_token` (handled separately in `reset_imas_nodes` at lines 453-457). Fixed plan text from "mirrors" to "superset of" to accurately describe the relationship. |

### Round 11 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Step 4 plan claims "mirrors" `_RESET_CLEAR_FIELDS["built"]` but includes `embedding_text`, `claimed_at`, `claim_token` not in the dict | Blocking | Changed language from "mirrors" to "superset of". Documented that `embedding_text` is migration-specific cleanup, `claimed_at`/`claim_token` are handled separately in `reset_imas_nodes()` at lines 453-457 (not in the dict itself). |
| 2 | MCP execution path wrong: `server.py` calls `graph_search.py` directly, not via `ids/tools.py` | Blocking | Corrected Step 15 and Files to Modify: MCP path is `server.py` → `GraphSearchTool.search_dd_paths()` / `GraphListTool.list_dd_paths()` in `graph_search.py`. `ids/tools.py` is a separate wrapper for non-MCP callers — listed for parity, not as the MCP execution layer. |
