# DD Graph Enrichment & Migration Guide

**Status:** Planned
**Created:** 2026-03-31
**Scope:** `imas_codex/graph/build_dd.py`, `imas_codex/tools/`, `imas_codex/schemas/imas_dd.yaml`,
`imas_codex/cocos/`, `imas_codex/llm/server.py`, tests

## Motivation

A/B testing of the `imas` (production) and `imas-dd` (graph-backed) MCP servers revealed that the
graph-backed architecture has **unique capabilities** not available in the production server — version
history, semantic clusters, cross-IDS relationship discovery, and graph schema introspection — but
these capabilities are underexploited. The graph contains 17,170 version change records, 3,471
semantic clusters, and 2,696 rename links, yet much of this data is poorly connected and hard to
query for practical migration tasks.

This plan addresses 9 enhancement areas identified during testing, centred on a flagship
**DD Migration Guide** feature that generates actionable, version-pair migration reports including
COCOS sign-flip tables, breaking-change classification, and code update recipes.

### Context: PR #214 (COCOS Labels)

[iterorganization/IMAS-Data-Dictionary#214](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/214)
re-adds `psi_like`, `dodpsi_like`, and missing `b0_like` COCOS labels to the DD schema, standardizes
sign convention terminology, and documents sign semantics on all labelled fields. This addresses the
**DD 4.0.0–4.1.1 gap** where COCOS labels were erroneously removed from the XML. The build pipeline
must handle this gap gracefully — backfilling labels for affected versions — and leverage the richer
metadata from future DD releases.

### imas-python as the Canonical Source

The build pipeline (`build_dd.py`) already extracts 45+ metadata fields per path via two imas-python
channels:

1. **Metadata API** (`imas.IDSFactory(version).new(ids_name).metadata`) — Python objects with
   `cocos_label_transformation`, units, documentation, coordinates, identifiers
2. **Raw XML** (`imas.dd_zip.dd_etree(version)`) — XPath access to `cocos_transformation_expression`,
   `cocos_leaf_name_aos_indices`, NBC metadata, coordinate_same_as

Several fields are extracted but **not persisted to the graph or not used for change detection**:
- `cocos_transformation_expression` — stored on node but not tracked for changes
- `change_nbc_version`, `change_nbc_previous_name`, `change_nbc_previous_type` — stored but not
  linked to change events
- `alternative_coordinate1`, `coordinate{N}_same_as` — stored as strings, not resolved to relationships
- Identifier enum values — extracted but not persisted

Each phase below identifies which imas-python fields need deeper extraction and which require a
graph rebuild to surface.

---

## Implementation Status

| Phase | Description | Status | Graph Rebuild |
|-------|-------------|--------|---------------|
| 1 | Graceful degradation: non-search tools without embeddings | Planned | No |
| 2 | COCOS label backfill for DD 4.0.0–4.1.1 | Planned | Yes |
| 3 | Path lifecycle: additions, removals, renames per version | Planned | Yes |
| 4 | Version chain: NEXT_VERSION relationships | Planned | Yes (lightweight) |
| 5 | COCOS-aware semantic clusters | Planned | Yes |
| 6 | Cross-IDS cluster densification | Planned | Yes |
| 7 | Migration guide MCP tool | Planned | No |
| 8 | Breaking vs non-breaking change classification | Planned | Yes |
| 9 | Cypher query optimizations | Planned | No |

---

## Phase 1: Graceful Degradation (No Rebuild)

### Problem

The imas-dd server gates **every** tool call on embedding warmup success. When the embedding server
is down (SLURM job pending, GPU contention), tools that don't need embeddings fail:
`check_imas_paths`, `list_imas_paths`, `fetch_imas_paths`, `get_dd_versions`,
`get_dd_version_context`, `get_imas_overview`, `get_imas_identifiers`, `analyze_imas_structure`,
`export_imas_ids`, `export_imas_domain`, `fetch_error_fields`. Only `get_graph_schema` works.

### Solution

Categorize tools into **embedding-required** and **graph-only** tiers:

| Tier | Tools | Requires Embeddings |
|------|-------|-------------------|
| **Search** | `search_imas`, `search_imas_clusters`, `find_related_imas_paths` (semantic mode) | Yes |
| **Lookup** | `check_imas_paths`, `fetch_imas_paths`, `list_imas_paths`, `fetch_error_fields` | No |
| **Overview** | `get_imas_overview`, `get_imas_identifiers`, `get_dd_versions`, `get_dd_version_context` | No |
| **Analysis** | `analyze_imas_structure`, `export_imas_ids`, `export_imas_domain` | No |

### Implementation

**File:** `imas_codex/llm/server.py`

The `_warmup_encoder()` background thread already catches failures and logs a warning. The fix is to
make the warmup gate **conditional** — only applied to search tools, not lookup/overview/analysis:

```python
# Current: all DD tools call _require_warmup() 
# Fix: only search tools require warmup; others work with graph-only
def _require_warmup():
    """Block until encoder is ready. Only called by search tools."""
    ...

def _require_graph():
    """Light check — only verifies Neo4j connection. Used by lookup tools."""
    ...
```

**Scope:** `imas_codex/llm/server.py` — tool registration, warmup gate
**Tests:** Verify all lookup/overview tools work when embedding server is unreachable

---

## Phase 2: COCOS Label Backfill (Rebuild Required)

### Problem

DD versions 4.0.0–4.1.1 removed `cocos_label_transformation` from the DD XML schema. The build
pipeline's `extract_paths_for_version()` gets `None` for these fields and never writes them to the
graph. Result: ~500 COCOS-dependent paths lose their labels in the 4.x series until 4.2.0+ re-adds
them (per PR #214).

### Current Graph State

```
COCOS labels by version:
  3.28.1: 307 fields first labelled
  3.42.2: 680 fields labelled (peak)
  4.0.0:  407 fields (dropped psi_like, some b0_like)
  4.1.1:  435 fields
  4.2.0+: ~700 fields expected (PR #214)
```

### Solution: Multi-Source Label Inference

**Strategy:** For DD 4.0.0–4.1.1, infer COCOS labels from multiple sources in priority order:

1. **Forward-port from DD 3.42.2:** Match paths by ID across versions. If a path exists in both
   3.42.2 (labelled) and 4.0.0 (unlabelled), carry the label forward.
2. **Back-port from DD 4.2.0+:** When PR #214 lands and imas-python includes 4.2.0, use those
   labels for paths that are new in 4.x.
3. **Parse `cocos_transformation_expression`:** The XML still contains expressions like
   `"- {psi_like}"` in 4.0.0. Parse these to extract label names.
4. **`imas.ids_convert._3to4_sign_flip_paths`:** imas-python's hardcoded mapping of which paths
   flip sign in DD3→DD4 provides ground truth for `psi_like` identification.

### Implementation

**File:** `imas_codex/graph/build_dd.py`

Add a new function `_backfill_cocos_labels()` called after all versions are extracted:

```python
def _backfill_cocos_labels(client: GraphClient, version_data: dict) -> int:
    """Backfill COCOS labels for versions where XML lacks them.
    
    Priority: forward-port from last labelled version > back-port from
    next labelled version > parse transformation expressions > imas-python
    sign flip paths.
    
    Returns count of labels backfilled.
    """
```

**Schema change:** Add `cocos_label_source` property to `IMASNode`:

```yaml
cocos_label_source:
  description: >-
    Source of the cocos_label_transformation value.
    'xml' = from DD XML directly, 'inferred_forward' = carried from prior version,
    'inferred_backward' = from later version, 'inferred_expression' = parsed from
    cocos_transformation_expression, 'inferred_sign_flip' = from imas-python sign flip paths.
  range: string
```

**Tests:**
- Verify label count for 4.0.0 after backfill matches 3.42.2 minus removed paths
- Verify `cocos_label_source` correctly tracks provenance
- Verify backfilled labels match forward-ported values from PR #214 branch

---

## Phase 3: Path Lifecycle Tracking (Rebuild Required)

### Problem

`compute_version_changes()` separately tracks `added` and `removed` path sets but never correlates
them. A renamed path appears as one removal + one addition with no link. The existing `RENAMED_TO`
relationships (2,696 in graph) are populated by an external `load_path_mappings()` reading a
pre-computed JSON file — the build pipeline itself doesn't detect renames.

Meanwhile, NBC metadata (`change_nbc_previous_name`, `change_nbc_previous_type`) is extracted from
the DD XML but **not used for rename detection or persisted as change events**.

### Current State

```
IMASNodeChange types: units, documentation, data_type, node_type,
  cocos_label_transformation, lifecycle_status, timebasepath
Missing: path_added, path_removed, path_renamed, structure_changed
```

### Solution

#### 3a. New ChangeType Values

**File:** `imas_codex/schemas/imas_dd.yaml`

```yaml
# Add to ChangeType enum:
path_added:
  description: Path introduced in this version
path_removed:
  description: Path removed/deprecated in this version
path_renamed:
  description: Path renamed (old_value=previous path, new_value=new path)
structure_changed:
  description: Structural change (array dims, parent hierarchy)
coordinates_changed:
  description: Coordinate specification changed
maxoccur_changed:
  description: Max occurrences changed
```

#### 3b. NBC-Based Rename Detection

**File:** `imas_codex/graph/build_dd.py`

The DD XML provides `change_nbc_previous_name` on renamed fields. Currently extracted (line 998) but
not used for change tracking. Wire this into `compute_version_changes()`:

```python
def _detect_renames(added: set, removed: set, version_data: dict) -> list[dict]:
    """Detect renames using NBC metadata and fuzzy matching.
    
    Priority:
    1. NBC previous_name metadata (ground truth from DD maintainers)
    2. Levenshtein similarity on (removed, added) pairs in same IDS
    3. Field signature matching (same units + type + docs similarity)
    """
```

#### 3c. Path Addition/Removal as Change Events

Create `IMASNodeChange` nodes for path additions and removals with:
- `change_type = 'path_added'` / `'path_removed'`
- `old_value` / `new_value` = path ID
- Link to version via `IN_VERSION`

#### 3d. Coordinate Change Tracking

Add `coordinates` to the compared fields in `compute_version_changes()`. Currently only 7 fields are
compared; coordinates are silently ignored despite being extracted.

**Tests:**
- Verify 4.0.0 renames are detected (label→name, momentum_tor→momentum_phi known from graph)
- Verify NBC metadata drives rename detection where available
- Verify path_added/path_removed counts match INTRODUCED_IN/DEPRECATED_IN edge counts

---

## Phase 4: Version Chain Relationships (Lightweight Rebuild)

### Problem

DDVersion nodes have `HAS_PREDECESSOR` relationships but no `NEXT_VERSION` relationship. Traversing
the version chain forward requires sorting by ID string, which is fragile and can't be done in a
single Cypher hop. Forward traversal is essential for migration guides ("what changes between
version A and version B?").

### Solution

**File:** `imas_codex/graph/build_dd.py` — `_create_version_nodes()`

Add `NEXT_VERSION` relationships alongside the existing `HAS_PREDECESSOR`:

```cypher
UNWIND $versions AS v
MATCH (ver:DDVersion {id: v.id})
MATCH (next:DDVersion {id: v.next_version})
MERGE (ver)-[:NEXT_VERSION]->(next)
```

**Schema change:** Add `NEXT_VERSION` relationship to `imas_dd.yaml`.

Also add convenience properties to `DDVersion`:
- `is_major_boundary: bool` — true if this version starts a new major series (4.0.0)
- `breaking_change_count: int` — precomputed count of breaking changes in this version

**Tests:**
- Verify full chain traversal: `(v:DDVersion {id: '3.22.0'})-[:NEXT_VERSION*]->(latest)` reaches
  current version in correct order
- Verify bidirectional: every NEXT_VERSION has a matching HAS_PREDECESSOR

---

## Phase 5: COCOS-Aware Semantic Clusters (Rebuild Required)

### Problem

Zero clusters exist with COCOS in their label or description. COCOS information is tracked via
`IMASNodeChange` nodes (`cocos_label_transformation` change type) but paths sharing the same COCOS
sensitivity are not grouped. A user asking "which paths are affected by COCOS 11→17 migration?"
must manually aggregate across all IDS.

### Solution

Create **COCOS transformation clusters** — one cluster per `cocos_label_transformation` value that
groups all paths sharing that sensitivity:

```yaml
# Example clusters:
- label: "psi_like COCOS-dependent fields"
  scope: global
  cross_ids: true
  members: [equilibrium/.../psi, core_profiles/.../psi, ...]

- label: "ip_like COCOS-dependent fields"
  scope: global
  cross_ids: true
  members: [equilibrium/.../ip, summary/.../ip, ...]
```

### Implementation

**File:** `imas_codex/graph/build_dd.py` — add `_create_cocos_clusters()` after HDBSCAN clustering

```python
def _create_cocos_clusters(client: GraphClient) -> int:
    """Create deterministic clusters grouping paths by COCOS label type.
    
    Unlike HDBSCAN clusters (statistical), these are authoritative groupings
    derived from DD metadata. They serve as ground-truth cross-IDS links
    for COCOS-sensitive paths.
    """
    labels = query("""
        MATCH (p:IMASNode)
        WHERE p.cocos_label_transformation IS NOT NULL
        RETURN p.cocos_label_transformation AS label, collect(p.id) AS paths,
               collect(DISTINCT p.ids) AS ids_names
        ORDER BY label
    """)
    # Create IMASSemanticCluster per label with scope='global', cross_ids=true
```

These clusters are **deterministic** (not statistical) and complement the existing HDBSCAN clusters.
Tag them with a `source` property distinguishing `'hdbscan'` from `'cocos_metadata'`.

**Tests:**
- Verify cluster exists for each distinct `cocos_label_transformation` value
- Verify `psi_like` cluster spans ≥5 IDS
- Verify `search_imas_clusters(query="COCOS psi")` returns the psi_like cluster

---

## Phase 6: Cross-IDS Cluster Densification (Rebuild Required)

### Problem

A/B testing found cross-IDS cluster coverage is sparse: psi returns only 1 cluster relative,
electron temperature returns only 1. The root cause is 13.5% of data paths (2,703) are unclustered,
and existing clusters average only ~5 members. Key physics quantities like `psi`, `Te`, `ip` should
link densely across `core_profiles ↔ edge_profiles ↔ equilibrium ↔ summary ↔ transport`.

### Solution

Two-pronged approach:

#### 6a. Reduce Unclustered Paths

Current HDBSCAN parameters may be too conservative. Tune:
- Lower `min_cluster_size` to capture smaller groups
- Use softer `cluster_selection_epsilon`
- Re-embed unclustered paths with enriched text (IDS context + units + COCOS label)

#### 6b. Physics-Aware Cross-IDS Links

Create **deterministic physics clusters** for canonical quantities:

```python
CANONICAL_CROSS_IDS = {
    "electron_temperature": {
        "paths_pattern": r".*/electrons/temperature$",
        "ids": ["core_profiles", "edge_profiles", "summary", "ece", "thomson_scattering"],
    },
    "poloidal_flux": {
        "paths_pattern": r".*/psi$|.*/psi_axis$|.*/psi_boundary$",
        "ids": ["equilibrium", "core_profiles", "core_transport", "mhd_linear"],
    },
    "plasma_current": {
        "paths_pattern": r".*/ip$|.*/current$",
        "ids": ["equilibrium", "summary", "pf_active", "magnetics"],
    },
    # ... 20-30 canonical quantities
}
```

These supplement statistical clusters with physics ground-truth links.

**Tests:**
- Verify `find_related_imas_paths("equilibrium/time_slice/profiles_1d/psi")` returns ≥5 cross-IDS
  paths after densification
- Verify unclustered path count drops below 5%

---

## Phase 7: Migration Guide MCP Tool (No Rebuild)

### Problem

No tool exists to answer "what changed between DD version A and version B?" with actionable
guidance. The `get_dd_version_context()` tool shows per-path history but provides no aggregation,
no version-pair comparison, and no code recipes.

### Solution

New MCP tool: `get_dd_migration_guide(from_version, to_version)`

#### 7a. Tool Interface

```python
@mcp_tool(
    name="get_dd_migration_guide",
    description="Generate a migration guide between two DD versions. Returns breaking changes, "
                "COCOS sign-flip tables, path renames, unit changes, and code update recipes.",
)
async def get_dd_migration_guide(
    self,
    from_version: str,  # e.g. "3.39.0"
    to_version: str,    # e.g. "4.0.0"
    ids_filter: str | None = None,  # restrict to single IDS
    include_recipes: bool = True,   # include code update snippets
) -> str:
```

#### 7b. Migration Guide Structure

The guide is rendered as structured markdown with these sections:

```markdown
# DD Migration Guide: 3.39.0 → 4.0.0

## Summary
- **COCOS convention change:** 11 → 17
- **Total changes:** 1,234
- **Breaking changes:** 89
- **Path renames:** 45
- **Paths removed:** 123
- **Paths added:** 567
- **IDS most affected:** equilibrium (23), core_profiles (15), summary (12)

## COCOS Sign-Flip Table
| IDS | Path | Label | Factor | Action |
|-----|------|-------|--------|--------|
| equilibrium | time_slice/.../psi | psi_like | ×(−1) | Negate stored values |
| equilibrium | time_slice/.../ip | ip_like | ×(+1) | No change needed |
| ... | ... | ... | ... | ... |

## Breaking Changes

### Removed Paths (123)
| IDS | Path | Replacement | Migration |
|-----|------|-------------|-----------|
| equilibrium | time_slice/profiles_1d/label | → .../name | Rename reference |
| ... | ... | ... | ... |

### Unit Changes
Only dimensionally-incompatible changes (breaking) and equivalent-symbol changes (advisory)
are shown — cosmetic notation changes (e.g. `A/m^2` → `A.m^-2`) are filtered at build time.

| IDS | Path | Old Unit | New Unit | Severity | Conversion |
|-----|------|----------|----------|----------|------------|
| core_profiles | .../pressure | J.m^-3 | Pa | advisory | Same dimension |
| spectrometer | .../power | W | Wb | **breaking** | Different dimension! |
| ... | ... | ... | ... | ... | ... |

### Type Changes (12)
| IDS | Path | Old Type | New Type | Action |
|-----|------|----------|----------|--------|
| ... | ... | FLT_0D | FLT_1D | Add time dimension |

## Code Update Recipes

### Python (imas-python)
```python
# COCOS sign flip for psi_like fields
from imas_codex.cocos.transforms import cocos_sign
factor = cocos_sign("psi_like", cocos_in=11, cocos_out=17)  # → -1
equilibrium.time_slice[0].profiles_1d.psi *= factor
```

### Path Renames
```python
# Old (DD 3.39.0):
value = ids.time_slice[0].profiles_1d.label
# New (DD 4.0.0):
value = ids.time_slice[0].profiles_1d.name
```
```

#### 7c. Cypher Queries

The migration guide is assembled from multiple graph queries:

```cypher
-- 1. Version-pair change summary
MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
WHERE v.id IN $version_range  -- all versions between from and to
WITH c.change_type AS type, c.semantic_type AS semantic, count(c) AS cnt
RETURN type, semantic, cnt ORDER BY cnt DESC

-- 2. COCOS sign-flip table
MATCH (p:IMASNode)
WHERE p.cocos_label_transformation IS NOT NULL
  AND EXISTS { (p)-[:INTRODUCED_IN]->(iv:DDVersion)
    WHERE toInteger(split(iv.id, '.')[0]) <= $to_major }
  AND NOT EXISTS { (p)-[:DEPRECATED_IN]->(dv:DDVersion)
    WHERE toInteger(split(dv.id, '.')[0]) <= $to_major }
RETURN p.ids AS ids, p.id AS path, p.cocos_label_transformation AS label
ORDER BY p.ids, p.id

-- 3. Path renames (via RENAMED_TO + NBC changes)
MATCH (old:IMASNode)-[:RENAMED_TO]->(new:IMASNode)
WHERE EXISTS { (old)-[:DEPRECATED_IN]->(v:DDVersion) WHERE v.id IN $version_range }
RETURN old.id AS old_path, new.id AS new_path, old.ids AS ids

-- 4. Unit changes in range (only real changes — cosmetic filtered at build)
MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)-[:FOR_IMAS_PATH]->(p:IMASNode)
WHERE v.id IN $version_range AND c.change_type = 'units'
RETURN p.ids, p.id, c.old_value, c.new_value,
       c.unit_change_subtype, c.breaking_level
```

#### 7d. COCOS Sign Computation

Leverage existing `cocos_sign()` from `imas_codex/ids/transforms.py`:

```python
from imas_codex.ids.transforms import cocos_sign

def compute_sign_flip_table(from_cocos: int, to_cocos: int, labeled_paths: list[dict]) -> list[dict]:
    """Compute sign/scale factors for all COCOS-labeled paths."""
    results = []
    for p in labeled_paths:
        factor = cocos_sign(p["label"], cocos_in=from_cocos, cocos_out=to_cocos)
        action = "No change needed" if factor == 1 else f"Multiply by {factor}"
        results.append({**p, "factor": factor, "action": action})
    return results
```

**Tests:**
- Verify guide for 3.39.0→4.0.0 includes COCOS table with ≥300 entries
- Verify guide for 3.30.0→3.31.0 (minor version) shows incremental changes only
- Verify `ids_filter="equilibrium"` restricts output to equilibrium paths
- Verify code recipes are syntactically valid Python

---

## Phase 8: Pint-Normalized Unit Comparison (Rebuild Required)

### Problem

`compute_version_changes()` compares raw unit strings with `!=`, producing **6,379 unit
"changes"** of which **6,002 are in DD 4.1.0 alone**. Investigation reveals a 3-tier noise
problem:

| Category | Count | % | Example |
|----------|-------|---|---------|
| Cosmetic (identical after pint) | 3,867 | 60.6% | `A/m^2` → `A.m^-2`, `Ohm` → `ohm`, `-` → `1` |
| Sentinel resolved (`as_parent` → real unit) | 2,318 | 36.3% | `as_parent_level_2` → `m^-3` |
| Dimensionally equivalent | 98 | 1.5% | `J.m^-3` → `Pa` (both are pressure) |
| **Genuine physics change** | **96** | **1.5%** | `W` → `Wb`, `m` → `rad`, `V.m^-1` → `T` |

DD 4.1.0 performed a **mass unit standardization** — resolving `as_parent*` placeholders,
switching to UDUNITS-compatible notation (`/` → `.^-1`), and standardizing case. These are
formatting changes, not physics changes. Only 96 of 6,379 changes represent actual dimensional
incompatibilities.

Meanwhile, the Unit node graph already has 450 nodes including garbage sentinels like
`as_parent_level_2` and `Toroidal angle`. We already have `normalize_unit_symbol()` in
`imas_codex/units/` backed by pint with a custom UDUNITS formatter and DD-specific aliases —
but it's only used for Unit node creation, not for change detection.

### Solution

Normalize units at build time using pint's base representation so that all DD version paths
point to the same canonical Unit node regardless of the raw string used in each version's XML.

#### 8a. Normalize Before Comparison in `compute_version_changes()`

**File:** `imas_codex/graph/build_dd.py` (line ~1188)

Replace the naive `old_val != new_val` comparison for the `units` field with pint-based
comparison at three tiers:

```python
from imas_codex.units import normalize_unit_symbol, unit_registry

def _units_changed(old_raw: str, new_raw: str) -> tuple[bool, str]:
    """Compare units via pint normalization.
    
    Returns:
        (changed, change_subtype) where change_subtype is one of:
        - 'cosmetic'             — same after normalization (not stored)
        - 'sentinel_resolved'    — as_parent/placeholder → concrete unit
        - 'dim_equivalent'       — different symbol, same dimension (e.g. J.m^-3 → Pa)
        - 'dim_incompatible'     — genuinely different physics (e.g. W → Wb)
    """
    old_norm = normalize_unit_symbol(old_raw)
    new_norm = normalize_unit_symbol(new_raw)
    
    # Tier 1: Identical after normalization → not a change
    if old_norm == new_norm:
        return False, 'cosmetic'
    
    # Tier 2: One or both are sentinels/unparseable
    if old_norm is None or new_norm is None:
        return True, 'sentinel_resolved'
    
    # Tier 3: Both parse — check dimensional compatibility
    try:
        old_p = unit_registry.parse_expression(old_norm)
        new_p = unit_registry.parse_expression(new_norm)
        if old_p.is_compatible_with(new_p):
            return True, 'dim_equivalent'
    except Exception:
        pass
    
    return True, 'dim_incompatible'
```

Update the comparison loop:

```python
for field in ("units", "documentation", ...):
    old_val = old_info.get(field, "")
    new_val = new_info.get(field, "")
    
    if field == "units":
        changed, subtype = _units_changed(old_val, new_val)
        if not changed:
            continue  # cosmetic — don't create IMASNodeChange
        changes.append({
            "field": "units",
            "old_value": str(old_val) if old_val else "",
            "new_value": str(new_val) if new_val else "",
            "unit_change_subtype": subtype,  # stored on IMASNodeChange
        })
    else:
        if old_val != new_val:
            changes.append({...})
```

#### 8b. Store `unit_change_subtype` on IMASNodeChange

**File:** `imas_codex/schemas/imas_dd.yaml`

Add a new enum and property:

```yaml
UnitChangeSubtype:
  description: Classification of unit change based on pint normalization
  permissible_values:
    sentinel_resolved:
      description: Placeholder (as_parent, etc.) resolved to concrete unit
    dim_equivalent:
      description: Different symbol, same physical dimension (e.g. J.m^-3 → Pa)
    dim_incompatible:
      description: Genuinely different physical dimension (breaking change)

# On IMASNodeChange:
unit_change_subtype:
  description: For units changes, the pint-based classification
  range: UnitChangeSubtype
```

#### 8c. Canonical Unit Nodes — No Sentinels

**File:** `imas_codex/graph/build_dd.py` (`_create_unit_nodes`, `HAS_UNIT` creation)

Currently, unparseable raw strings like `as_parent_level_2` become Unit node IDs.
Change: only create Unit nodes for pint-parseable units. Paths with sentinel units get
no `HAS_UNIT` relationship (they have no physical unit). Paths from older DD versions that
used `as_parent_level_2` get `HAS_UNIT` to the *resolved* unit from later versions when
available (forward-port the resolution).

```python
def _create_unit_nodes(client, units: set[str]) -> None:
    unit_list = []
    for u in units:
        normalized = normalize_unit_symbol(u)
        if normalized is None:
            continue  # skip sentinels — no Unit node
        unit_list.append({"id": normalized, "symbol": normalized})
    ...
```

For `HAS_UNIT` relationships, always normalize:

```python
normalized = normalize_unit_symbol(p["unit"])
if normalized:
    unit_paths.append({**p, "unit": normalized})
# else: no HAS_UNIT for this path (sentinel unit)
```

This collapses `A/m^2`, `A.m^-2`, and `A/m²` to a single `A.m^-2` Unit node.
All paths across all DD versions that have the same physical unit will point to the
same Unit node, enabling proper cross-version comparisons.

**Expected impact:**
- Unit nodes: ~450 → ~200 (remove sentinels, collapse equivalents)
- Unit changes: 6,379 → ~2,512 (194 real + 2,318 sentinel resolutions)
- Cosmetic noise eliminated: 3,867 spurious IMASNodeChange nodes removed

#### 8d. Breaking Change Classification

With clean unit data, breaking change classification becomes reliable:

```yaml
BreakingChangeLevel:
  description: Severity classification for DD changes
  permissible_values:
    breaking:
      description: Requires code changes (type change, incompatible unit, sign flip, removal)
    advisory:
      description: May affect interpretation (doc clarification, lifecycle, equivalent unit)
    informational:
      description: No code impact (doc wording, formatting)
```

Add `breaking_level` property to `IMASNodeChange`:

```yaml
breaking_level:
  description: Whether this change requires code updates
  range: BreakingChangeLevel
```

Classification rules — now unit-aware:

```python
BREAKING_RULES = {
    'path_removed': 'breaking',
    'path_renamed': 'breaking',
    'data_type': 'breaking',
    'cocos_label_transformation': 'breaking',
    'coordinates_changed': 'advisory',
    'lifecycle_status': 'advisory',
    'node_type': 'advisory',
    'timebasepath': 'informational',
    'maxoccur_changed': 'advisory',
    'documentation': lambda c: (
        'advisory' if c.semantic_type in ('sign_convention', 'coordinate_convention')
        else 'informational'
    ),
}

def classify_unit_change(change: dict) -> str:
    """Unit changes classified by pint subtype."""
    subtype = change.get('unit_change_subtype')
    if subtype == 'dim_incompatible':
        return 'breaking'       # W → Wb: different physics
    if subtype == 'dim_equivalent':
        return 'advisory'       # J.m^-3 → Pa: same physics, different symbol
    return 'informational'      # sentinel resolved: metadata improvement
```

#### 8e. Precomputed Aggregates on DDVersion

Add to DDVersion node:
- `breaking_change_count: int` — count of breaking changes introduced in this version
- `advisory_change_count: int`
- `total_change_count: int`

**Tests:**
- Verify `A/m^2` → `A.m^-2` does NOT produce an IMASNodeChange (cosmetic)
- Verify `J.m^-3` → `Pa` produces advisory change (dim_equivalent)
- Verify `W` → `Wb` produces breaking change (dim_incompatible)
- Verify `as_parent_level_2` → `m^-3` produces informational change (sentinel_resolved)
- Verify Unit node count drops (~200 real units, no sentinels)
- Verify all `HAS_UNIT` relationships point to pint-normalized Unit nodes
- Verify doc wording changes are informational
- Verify sign_convention doc changes are advisory
- Verify total unit changes drop from 6,379 to ~2,512

---

## Phase 9: Cypher Query Optimizations (No Rebuild)

### Problem

Code inspection identified several query performance issues:

| Issue | Severity | Location |
|-------|----------|----------|
| CONTAINS fallback does full IMASNode scan | High | `graph_search.py:1911` |
| `toLower()` on every row in fallback | High | `graph_search.py:1911-1940` |
| `split()` to calculate depth on every row | Medium | `graph_search.py:1563` |
| Doc length >10 check per row | Medium | `graph_search.py:1887` |
| Per-path validation not batched | Low | `graph_search.py:392` |
| Hardcoded score threshold 0.3 | Low | `graph_search.py:1159` |

### Solution

#### 9a. Precomputed Properties on IMASNode

Add to schema and populate during build:

```yaml
depth:
  description: Nesting depth (number of '/' in path)
  range: integer
is_leaf:
  description: Whether this is a leaf data node (not STRUCTURE/STRUCT_ARRAY)
  range: boolean
path_lower:
  description: Lowercase path for case-insensitive CONTAINS queries
  range: string
doc_length:
  description: Length of documentation string
  range: integer
```

#### 9b. Composite Indexes

```cypher
CREATE INDEX imas_node_category_ids IF NOT EXISTS
  FOR (p:IMASNode) ON (p.node_category, p.ids)

CREATE INDEX imas_node_is_leaf IF NOT EXISTS
  FOR (p:IMASNode) ON (p.is_leaf)
```

#### 9c. Batch Path Validation

Replace per-path loop with single UNWIND query:

```cypher
UNWIND $paths AS path_id
OPTIONAL MATCH (p:IMASNode {id: path_id})
WHERE p.node_category = 'data'
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
RETURN path_id AS requested, p.id AS found, p.ids AS ids,
       p.data_type AS data_type, u.id AS units
```

**Tests:**
- Benchmark CONTAINS fallback before/after precomputed properties
- Verify batch validation returns same results as per-path loop

---

## Dependency Graph

```
Phase 1 (graceful degradation) ─── independent, ship first
Phase 4 (version chain) ─── independent, lightweight
Phase 9 (query optimizations) ─── independent

Phase 2 (COCOS backfill) ──┐
Phase 3 (path lifecycle) ──┼── require graph rebuild together
Phase 8 (pint units +     ─┘
         breaking changes)
                            │
                            ▼
Phase 5 (COCOS clusters) ──┐
Phase 6 (cluster density) ──┼── rebuild on enriched graph
                            │
                            ▼
Phase 7 (migration guide) ─── builds on phases 2-6,8 — no rebuild itself
```

### Suggested Execution Order

**Sprint 1 (no rebuild):** Phases 1, 4, 9 — immediate value, unblocks imas-dd testing
**Sprint 2 (first rebuild):** Phases 2, 3, 8 — enriched change tracking with pint-normalized units
**Sprint 3 (second rebuild):** Phases 5, 6 — cluster improvements
**Sprint 4 (tool):** Phase 7 — migration guide tool leveraging all prior work

Each rebuild is a `uv run imas-codex dd build --force` cycle (~30 min with embeddings, ~5 min
without). Schema changes require `uv run build-models --force` before rebuild.

---

## Graph Rebuild Strategy

The phased approach requires multiple graph rebuilds as we iteratively enrich the data. To minimize
disruption:

1. **Schema-first:** Add all new properties/enums to `imas_dd.yaml` and rebuild models before any
   code changes. New properties with no data are harmless.
2. **Incremental extraction:** Each phase adds extraction logic to `build_dd.py` that populates new
   properties. Existing data is preserved.
3. **Build hash versioning:** Bump `_BUILD_SCHEMA_VERSION` when extraction logic changes. This
   invalidates the build cache and triggers a full re-extract on next build.
4. **Test-driven:** Each phase adds tests that verify the new data before the next phase begins.
   Tests run against the graph after rebuild.
5. **Release gating:** Push graph to GHCR after each successful rebuild + test pass. The migration
   guide tool (Phase 7) should be developed against the Sprint 2 graph.

---

## Migration Guide: Extended Design

The migration guide is the flagship deliverable. Here's the extended design.

### Use Cases

1. **"I'm upgrading from DD 3.39 to 4.0 — what code do I change?"**
   → Full guide with COCOS table, renames, unit changes, code recipes

2. **"What happened to psi in DD 4.0?"**
   → Per-path history showing COCOS label removal, sign convention doc addition

3. **"Which IDS are most affected by the DD 3→4 migration?"**
   → Impact assessment: paths changed per IDS, breaking change counts

4. **"Give me a COCOS transformation table for equilibrium"**
   → Filtered sign-flip table for one IDS

5. **"What's new in DD 4.1.0?"**
   → Single-version changelog: new paths, changes, deprecations

### Output Formats

The tool should support multiple output modes via a `format` parameter:

- **`summary`** (default): Compact overview with counts and top changes
- **`full`**: Complete guide with all tables and code recipes
- **`cocos_only`**: Just the COCOS sign-flip table
- **`breaking_only`**: Only breaking changes
- **`json`**: Machine-readable for downstream tools

### Integration with imas-python

The code recipes should use real imas-python API syntax:

```python
# Reading with version-aware access
import imas
entry = imas.DBEntry("imas:hdf5?path=/data", "r")
equilibrium = entry.get("equilibrium")

# COCOS transformation (built into imas-python)
from imas.ids_convert import convert_ids
equilibrium_v4 = convert_ids(equilibrium, source_version="3.39.0", target_version="4.0.0")
```

### Graph-Backed Advantages

This migration guide is only possible with the graph architecture because it requires:
- **Cross-version path matching:** INTRODUCED_IN/DEPRECATED_IN/RENAMED_TO relationships
- **Semantic change classification:** IMASNodeChange nodes with breaking_level
- **COCOS metadata aggregation:** Cluster-based grouping of COCOS-sensitive paths
- **Multi-hop traversal:** Version chain → changes → affected paths → clusters → related paths

The production `imas` server has none of this infrastructure.
