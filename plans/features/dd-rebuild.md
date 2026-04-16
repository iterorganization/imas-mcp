# DD Rebuild: Classification, Multi-Pass Enrichment & Model Upgrade

> Full rebuild of IMAS Data Dictionary graph nodes with geometry classification,
> two-pass LLM enrichment using sonnet 4.6, and MCP tool augmentation.

## Goal

Reclassify all ~61K DD nodes with a `quantity` + `geometry` split, re-enrich
~12K enrichable nodes using a two-pass LLM pipeline with Claude Sonnet 4.6,
re-embed, and re-cluster. All external relationships preserved.

## Why a Full Rebuild

The current graph has:
- No `geometry` category — all physics and hardware paths are `quantity`
- Enrichment from an older model (language config currently `anthropic/claude-opus-4.6`)
- Single-pass enrichment — no cross-referencing between sibling descriptions
- Bug 1: `PARENT_OF` relationship used in `build_dd.py:2050` (should be `HAS_PARENT`)
- Bug 2: Pass 2 R2 reclassifies physics data nodes as coordinates (no unit guard)
- Legacy `'data'` category still referenced in `build_dd.py:2026,2035,2047` and `preprocessing.py:44`

A full rebuild is cheaper and safer than surgical migration. Cost: ~$100. Time: ~6 hours.

## Current Graph State

| Category | Status | Count |
|----------|--------|-------|
| quantity | embedded | 9,993 |
| coordinate | enriched | 1,739 |
| error | built | 31,281 |
| metadata | built | 10,715 |
| structural | built | 7,395 |
| identifier | built | 243 |

**All DD relationships are rebuilt by the pipeline** — `HAS_PARENT`, `HAS_COORDINATE`,
`IN_CLUSTER`, `HAS_UNIT`, `HAS_IDENTIFIER_SCHEMA`, `INTRODUCED_IN`, `DEPRECATED_IN`,
`RENAMED_TO`, `FOR_IMAS_PATH`, `COORDINATE_SAME_AS`, `HAS_ERROR`, `IN_IDS`.

**StandardName relationships** (`HAS_STANDARD_NAME`, `HAS_STANDARD_NAME_VOCAB_GAP`) will
be cleared before DD rebuild as part of the SN greenfield plan (P1.6). No preservation needed.

**Reset strategy**: `--reset-to built` clears enrichment fields and preserves build-time
properties + all relationships. A full `--reset-to extracted` (DETACH DELETE + rebuild
from XML) is also safe since all relationships are pipeline-internal.

---

## 1. NodeCategory Enum

Add `geometry` alongside existing `quantity`. No rename needed — `quantity` is unambiguous
now that geometry has its own category.

```yaml
NodeCategory:
  quantity:
    description: >-
      Measurable physics quantity (temperature, current, pressure, field,
      density, flux, elongation, triangularity, beta, safety_factor).
      Enriched, embedded, searchable, SN-extractable.
  geometry:
    description: >-
      Machine and diagnostic hardware positions and dimensions.
      Coil r/z, vessel outline, diagnostic lines of sight, aperture
      positions, detector geometry. Things that exist in real space
      before plasma. Enriched, embedded, searchable, SN-extractable.
  coordinate:
    description: >-
      Independent variable (time, spatial coords, flux coordinates).
      Enriched and searchable, NOT embedded or SN-extracted.
  structural:
    description: Storage artifact (/data, /validity, containers without unit)
  identifier:
    description: Typed descriptor with HAS_IDENTIFIER_SCHEMA relationship
  error:
    description: Uncertainty bound fields (_error_upper, _error_lower, _error_index)
  metadata:
    description: Bookkeeping subtrees (ids_properties, code, identifier descriptors)
```

### Pipeline Participation

| Category | Enriched | Embedded | SN Source | Searchable |
|----------|----------|----------|-----------|------------|
| `quantity` | ✓ | ✓ | ✓ | ✓ |
| `geometry` | ✓ | ✓ | ✓ | ✓ |
| `coordinate` | ✓ | ✗ | ✗ | ✓ |
| `structural` | ✗ | ✗ | ✗ | ✗ |
| `identifier` | ✗ | ✗ | ✗ | ✗ |
| `error` | ✗ | ✗ | ✗ | ✗ |
| `metadata` | ✗ | ✗ | ✗ | ✗ |

### Constants (node_categories.py)

```python
QUANTITY_CATEGORIES = frozenset({"quantity", "geometry"})
EMBEDDABLE_CATEGORIES = QUANTITY_CATEGORIES
SN_SOURCE_CATEGORIES = QUANTITY_CATEGORIES
SEARCHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"}
ENRICHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"}
```

---

## 2. Geometry Classification

**Definition**: geometry = things that exist in real space before plasma. Machine
hardware, diagnostic optics, vessel structures. If you can measure it with a ruler
on the cold machine, it's geometry.

**NOT geometry**: elongation, triangularity, plasma boundary, magnetic axis position,
pedestal location, safety factor profile. These are equilibrium outputs computed from
plasma measurements.

### Detection Algorithm

A node is classified as `geometry` when BOTH conditions hold:
1. **Spatial unit** — unit ∈ {`m`, `rad`, `m^2`, `m^3`, `deg`}
   (excludes `m^-1`, `m^-2`, `m^-3` which are densities/inverse lengths)
2. **Geometry ancestor** — at least one ancestor path segment matches a geometry pattern,
   AND no ancestor segment matches an exclusion pattern

```python
GEOMETRY_PATH_PATTERNS = frozenset({
    "geometry", "outline", "aperture", "line_of_sight",
    "first_point", "second_point", "position", "detector",
    "mirror", "waveguide", "launcher", "antenna",
    "annular_grid", "first_wall", "limiter", "divertor",
    "vessel", "pf_active", "pf_passive",
})

GEOMETRY_EXCLUSION_PATTERNS = frozenset({
    "boundary", "separatrix", "grid", "ggd", "flux_surface",
    "constraint", "magnetic_axis", "pedestal", "itb", "etb",
    "x_point", "strike_point",
})

SPATIAL_UNITS = frozenset({"m", "rad", "m^2", "m^3", "deg"})
```

Pattern matching uses **substring** for exclusions (catches `boundary_separatrix`)
and **exact segment** for inclusions (prevents false matches on substrings).

### Expected Distribution

From live graph analysis (~1,047 geometry nodes, 10.5% of current quantities):

| IDS | Geometry Count | Example |
|-----|---------------|---------|
| bolometer | 88 | channel/line_of_sight/first_point/r |
| spectrometer_visible | 80 | channel/detector/geometry/outline/r |
| neutron_diagnostic | 76 | detectors/detector/x1_unit_vector/x |
| camera_visible | 58 | channel/detector/geometry/outline/r |
| pf_active | 49 | coil/element/geometry/outline/r |
| ic_antennas | 46 | antenna/module/strap/geometry/rectangle/width |
| equilibrium | 0 | (all excluded: boundary, separatrix, constraint) |
| core_profiles | 0 | (no geometry patterns) |

### Classifier Rules (Updated)

**Pass 1** (`classify_from_attributes`): Rules 1–8 unchanged. Rules 9–15 split:

```
Rule 9a: FLT/CPX + spatial unit + geometry ancestor → geometry
Rule 9b: FLT/CPX + physics unit → quantity
Rule 10a: FLT/CPX + no unit + coordinate segment → coordinate
Rule 10b: FLT/CPX + no unit → quantity (dimensionless: beta, q, zeff)
Rule 11a: STRUCTURE + unit + geometry ancestor → geometry
Rule 11b: STRUCTURE + unit → quantity
Rule 12:  STRUCTURE without unit → structural
Rule 13:  INT_0D + no unit + no structural keywords → quantity
Rule 14a: INT + spatial unit + geometry ancestor → geometry
Rule 14b: INT + physics unit → quantity
Rule 15:  Remaining INT → structural
```

**Pass 2** (`refine_from_relationships`): R1 unchanged. R2 fixed:

```
R1: has_identifier_schema → identifier (unchanged)
R2: is_coordinate_target
    IF name ∈ COORDINATE_SEGMENTS → coordinate (canonical names always coordinate)
    ELIF unit is physics (not empty, not '-') AND name ∉ COORDINATE_SEGMENTS → keep (Bug 2 fix)
    ELSE → coordinate
R3: STRUCTURE+unit with children ∈ QUANTITY_CATEGORIES → keep; else → structural
```

### Key Test Cases

```python
# Geometry (machine hardware)
("pf_active/coil/element/geometry/outline/r", "r", "FLT_1D", "m", None, "geometry"),
("bolometer/channel/line_of_sight/first_point/r", "r", "FLT_0D", "m", None, "geometry"),
("camera_visible/channel/detector/geometry/outline/r", "r", "FLT_1D", "m", None, "geometry"),

# NOT geometry — equilibrium outputs
("eq/ts/boundary/elongation", "elongation", "FLT_0D", "-", None, "quantity"),
("eq/ts/boundary/outline/r", "r", "FLT_1D", "m", None, "quantity"),
("eq/ts/boundary/minor_radius", "minor_radius", "FLT_0D", "m", None, "quantity"),

# NOT geometry — density (m^-3 is not a spatial unit)
("summary/local/limiter/n_e/value", "value", "FLT_0D", "m^-3", None, "quantity"),

# NOT geometry — plasma positions
("summary/local/magnetic_axis/position/r", "r", "FLT_0D", "m", None, "quantity"),
("summary/local/pedestal/position/rho_tor", "rho_tor", "FLT_0D", "m", None, "quantity"),

# Geometry — NBI hardware position
("summary/heating_current_drive/nbi/position/r/value", "value", "FLT_0D", "m", None, "geometry"),
```

---

## 3. Status Lifecycle

```
built → enriched (Pass 1) → refined (Pass 2) → embedded
```

Add `refined` to `IMASNodeStatus` in `imas_dd.yaml`:

```yaml
IMASNodeStatus:
  permissible_values:
    built:
      description: Node created from DD XML
    enriched:
      description: Pass 1 description generated by LLM
    refined:
      description: Pass 2 description refined with sibling and peer context
    embedded:
      description: Vector embedding generated from refined description
```

Add to `IMASNode` properties:
- `refinement_hash` (str) — hash of Pass 2 inputs (Pass 1 description + siblings + peers + model)
- `refined_at` (datetime) — timestamp of Pass 2 completion

---

## 4. Multi-Pass Enrichment

### Pass 1: Description Generation (enrich_worker — existing)

Generates initial descriptions using path context. Unchanged except:
- Model: sonnet 4.6 (via `--model` flag or env var)
- `quantity` + `geometry` nodes → enrichment type `"concept"` (positive override)
- `coordinate` nodes → enrichment type `"concept"` (already handled)

### Pass 2: Refinement (refine_worker — new)

After Pass 1 completes for a sibling group, Pass 2 refines each description using
context that was unavailable during Pass 1:

| Context | Source | Value |
|---------|--------|-------|
| Own Pass 1 description | Node property | Starting point for refinement |
| Sibling descriptions | Same parent's children | Disambiguates similarly-named paths |
| Cluster peer descriptions | IN_CLUSTER relationship | Cross-IDS standardization |
| Cross-IDS duplicates | Same leaf name, different IDS | Consistency across IDSs |

### Sibling-Readiness Barrier

The refine_worker claims only nodes whose enrichable siblings are ALL past `built`:

```cypher
MATCH (n:IMASNode)
WHERE n.status = 'enriched'
  AND n.node_category IN $enrichable
  AND n.claimed_at IS NULL
  AND NOT EXISTS {
    MATCH (n)-[:HAS_PARENT]->(parent)<-[:HAS_PARENT]-(sib:IMASNode)
    WHERE sib.node_category IN $enrichable AND sib.status = 'built'
  }
WITH n ORDER BY rand() LIMIT $limit
SET n.claimed_at = datetime(), n.claim_token = $token
```

This creates a natural wavefront — no global synchronization needed.

### Refinement Hash

```python
def compute_refinement_hash(
    pass1_description: str,
    sibling_descriptions: list[str],
    cluster_peers: list[str],
    model_name: str,
) -> str:
    combined = (
        f"{model_name}:{pass1_description}:"
        + ":".join(sorted(sibling_descriptions))
        + ":".join(sorted(cluster_peers))
    )
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
```

### Refinement Prompt

New template: `imas_codex/llm/prompts/imas/refinement.md`

System prompt instructs:
- Preserve accurate physics from Pass 1
- Disambiguate from siblings (e.g., `r` under `outline` vs `r` under `position`)
- Standardize terminology with cluster peers across IDSs
- Maintain 150–300 character target

### Reset Levels

| Reset Target | Statuses Reset | Fields Cleared | Use Case |
|--------------|---------------|----------------|----------|
| `built` | enriched, refined, embedded | description, keywords, enrichment_*, refinement_*, embedding_* | Full re-enrichment (~$100) |
| `enriched` | refined, embedded | refinement_*, embedding_* | Re-refinement only (~$50) |
| `refined` | embedded | embedding_* | Re-embedding only (free) |

---

## 5. Pipeline Wiring

### Worker Sequence

```python
workers = [
    WorkerSpec("extract", extract_worker, ...),   # DD XML → nodes
    WorkerSpec("build", build_worker, ...),        # relationships + Pass 2 classification
    WorkerSpec("enrich", enrich_worker, ...),      # Pass 1: LLM descriptions
    WorkerSpec("refine", refine_worker, ...),      # Pass 2: sibling/peer refinement
    WorkerSpec("embed", embed_worker, ...),        # vector embeddings
    WorkerSpec("cluster", cluster_worker, ...),    # semantic clustering
]
```

### Model Threading

Add `--model` CLI flag to `imas dd build`:

```python
@click.option("--model", default=None, help="Override LLM model for enrichment")
```

Thread through `DDBuildState` → `enrich_worker` → `enrich_imas_paths()`.
The `enrich_imas_paths()` function already accepts a `model` parameter at `dd_enrichment.py:875`.

Quick-start alternative: `IMAS_CODEX_LANGUAGE_MODEL=openrouter/anthropic/claude-sonnet-4.6`

Hash-based idempotency means changing model forces re-enrichment automatically.

### Graph Operations (new functions)

| Function | Purpose |
|----------|---------|
| `claim_paths_for_refinement()` | Claim `enriched` nodes with sibling-readiness barrier |
| `mark_paths_refined()` | Set status=refined, write refinement_hash, refined_at |
| `release_refinement_claims()` | Clear claimed_at on failure |
| `has_pending_refinement()` | Check if any enriched nodes remain |

Update existing:
- `claim_paths_for_embedding()` — claim `refined` instead of `enriched`
- `_RESET_CLEAR_FIELDS` — add `refined` level clearing refinement fields
- `_RESET_SOURCE_STATUSES` — add `refined` to source statuses for `built` and `enriched` targets

---

## 6. Bug Fixes

### Bug 1: Reversed Relationship in build_dd.py

**Location**: `build_dd.py:2050`
```cypher
-- Current (WRONG — PARENT_OF does not exist in schema):
OPTIONAL MATCH (n)<-[:PARENT_OF]-(child:IMASNode)
-- Fixed:
OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(n)
```

### Bug 2: Overbroad Coordinate Reclassification

**Location**: `node_classifier.py:282-284` (Pass 2 R2)
```python
# Current (WRONG — reclassifies psi values node as coordinate):
if is_coordinate_target and current_category != "error":
    return "coordinate"
# Fixed — add name + unit guard:
if is_coordinate_target:
    if name in COORDINATE_SEGMENTS:
        return "coordinate"
    if unit and unit not in _NO_UNIT and name not in COORDINATE_SEGMENTS:
        return None  # keep current — physics data with unit
    return "coordinate"
```

### Bug 3: Legacy 'data' Category References

**Locations**:
- `build_dd.py:2026` — `WHERE n.node_category IN ['quantity', 'coordinate', 'structural', 'data']`
- `build_dd.py:2035` — same
- `build_dd.py:2047` — `WHERE n.node_category IN ['quantity', 'data']`
- `preprocessing.py:44` — `if _classify_node(path, name) != "data":`

Replace with `QUANTITY_CATEGORIES` and `ENRICHABLE_CATEGORIES` imports from `node_categories.py`.

---

## 7. MCP Tool Augmentation

### Category Filter

Add `node_category` optional parameter to `search_dd_paths` and `list_dd_paths`:

```python
node_category: str | None  # "quantity", "geometry", "coordinate"
```

When provided, narrows the Cypher `WHERE n.node_category IN $categories` clause.

### Coordinate Metadata

Expand `fetch_dd_paths` coordinate display:
```
Coordinates: rho_tor_norm (FLT_1D, -) [coordinate], time (FLT_0D, s) [coordinate]
```

---

## 8. Execution Order

| Step | Task | Depends On | Risk |
|------|------|------------|------|
| 1 | Write TDD tests (classifier + categories) | — | Low |
| 2 | Update schema: add `geometry`, `refined` status, refinement fields | — | Low |
| 3 | Rebuild models (`uv run build-models --force`) | 2 | Low |
| 4 | Update `node_classifier.py`: rename functions, add geometry rules, fix R2 | 1, 3 | Medium |
| 5 | Update `node_categories.py`: add `QUANTITY_CATEGORIES`, update all sets | 3 | Low |
| 6 | Fix `build_dd.py`: Bug 1 (PARENT_OF), Bug 3 (legacy 'data'), import constants | 4, 5 | Medium |
| 7 | Fix `preprocessing.py`: replace `_classify_node != "data"` with constants | 5 | Low |
| 8 | Update `dd_enrichment.py`: positive override for quantity categories | 5 | Low |
| 9 | Add refinement: `gather_refinement_context()`, `build_refinement_messages()`, `compute_refinement_hash()`, prompt template | 5 | Medium |
| 10 | Update `dd_graph_ops.py`: refinement claim/mark/release, reset levels, return node_category | 3 | Medium |
| 11 | Update `dd_workers.py`: thread model + add `refine_worker` + update embed claims | 9, 10 | Medium |
| 12 | Add `--model` CLI flag to `imas_dd.py` | 11 | Low |
| 13 | Run TDD tests — all should pass | 1–12 | — |
| 14 | Run full test suite | 13 | — |
| 15 | In-place reclassification script (~3 min) | 4, 5 | Low |
| 16 | Full rebuild: `imas-codex imas dd build --reset-to built --model openrouter/anthropic/claude-sonnet-4.6` | 15 | Medium |
| 17 | MCP tool augmentation (node_category filter) | 5 | Low |

**Steps 1–14**: Code changes and testing (~2–3 agents, ~2 hours)
**Step 15**: In-place reclassification (~3 minutes)
**Step 16**: Full rebuild (~6 hours, ~$100)
**Step 17**: MCP tool augmentation (~1 agent, ~30 minutes)

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `imas_codex/schemas/imas_dd.yaml` | Add `geometry` to NodeCategory. Add `refined` to IMASNodeStatus. Add `refinement_hash`, `refined_at` to IMASNode. |
| `imas_codex/core/node_classifier.py` | Rename `classify_node_pass1` → `classify_from_attributes`, `classify_node_pass2` → `refine_from_relationships`. Add geometry path patterns, spatial unit check, exclusion patterns. Fix R2 with name+unit guard. Add `path` parameter to `refine_from_relationships`. |
| `imas_codex/core/node_categories.py` | Add `QUANTITY_CATEGORIES`. Update all sets to use it. |
| `imas_codex/graph/build_dd.py` | Fix Bug 1: `PARENT_OF` → `HAS_PARENT` at line 2050. Fix Bug 3: replace legacy `'data'` with constants at lines 2026, 2035, 2047. Update imports. |
| `imas_codex/clusters/preprocessing.py` | Replace `_classify_node != "data"` with `EMBEDDABLE_CATEGORIES` check. |
| `imas_codex/graph/dd_enrichment.py` | Positive override: `node_category in QUANTITY_CATEGORIES` → enrichment type `"concept"`. Add `gather_refinement_context()`, `build_refinement_messages()`, `compute_refinement_hash()`. |
| `imas_codex/graph/dd_graph_ops.py` | Return `node_category` from claim functions. Add refinement claim/mark/release. Update `_RESET_CLEAR_FIELDS` and `_RESET_SOURCE_STATUSES` for `refined`. Update `claim_paths_for_embedding` to claim `refined`. |
| `imas_codex/graph/dd_workers.py` | Thread `model` through `DDBuildState`. Add `refine_worker` with sibling-readiness barrier. Update `embed_worker` to claim `refined`. Wire into `run_dd_build_engine()`. |
| `imas_codex/cli/imas_dd.py` | Add `--model` option to `build` command. Thread through `DDBuildState`. |
| `imas_codex/llm/prompts/imas/refinement.md` | **New**: Pass 2 refinement prompt template. |
| `imas_codex/tools/graph_search.py` | Add optional `node_category` filter to `search_dd_paths` and `list_dd_paths`. |
| `imas_codex/llm/server.py` | Thread `node_category` parameter through MCP tool signatures. |

### New Test Files

| File | Tests |
|------|-------|
| `tests/core/test_node_classifier.py` | Parametrized tests for `classify_from_attributes` (30+ cases) and `refine_from_relationships` (15+ cases) |
| `tests/core/test_node_categories.py` | Constant set membership, mutual exclusion, no legacy values |
| `tests/graph/test_refine_worker.py` | Sibling barrier, hash idempotency, status cascades |

### Superseded Test

| File | Action |
|------|--------|
| `tests/graph/test_node_category.py` | Delete (superseded by `test_node_classifier.py` + `test_node_categories.py`) |

---

## 10. Cost Estimate

| Phase | Nodes | Model | Cost |
|-------|-------|-------|------|
| Pass 1 (enrich) | ~12,780 enrichable | sonnet 4.6 | ~$50 |
| Pass 2 (refine) | ~12,780 enrichable | sonnet 4.6 | ~$50 |
| Embedding | ~11,040 embeddable | local GPU | $0 |
| **Total** | | | **~$100** |

---

## Naming Rationale

**`quantity`** (not `physical_quantity`): Unambiguous now that geometry has its own
category. Matches the existing schema enum value — no migration needed. Shorter and
cleaner in code, Cypher, and MCP tool output.

**`geometry`** (not `machine_geometry`): The DD itself uses `geometry` as a path segment
for machine hardware (e.g., `pf_active/coil/element/geometry/outline/r`). Our category
name mirrors the DD's own vocabulary. The classification algorithm makes the scope
crystal clear — only paths under geometry-indicating ancestors with spatial units qualify.
The enum description documents the definition.

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | NodeCategory enum (7→8 values), `--model` CLI flag, `refined` status in lifecycle |
| `plans/README.md` | Replace P1.5 + P1.5a with single dd-rebuild plan |
| Schema reference | Auto-generated by `uv run build-models` |
