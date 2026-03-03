# MDSplus Tree Discovery Unification

> Merge the static and dynamic MDSplus discovery pipelines into a single,
> shared-core system where all trees flow through the same infrastructure.

## Status: Planning

## Problem Statement

The MDSplus scanner plugin (`discovery/signals/scanners/mdsplus.py`) currently
joins two independent implementations behind a single `scan()` method:

- **Static trees** go through `discovery/static/` — a rich pipeline with
  TreeNode creation, version tracking, unit extraction, pattern detection,
  and LLM enrichment with parent/sibling context.
- **Dynamic trees** go through `enumerate_mdsplus.py` — a flat SSH script
  that returns raw JSON, converted directly to FacilitySignal objects with
  no TreeNode intermediary, no graph structure, and no enrichment context.

This produces a two-class signal system where static signals are deeply
characterized and dynamic signals are barely described. The infrastructure
for rich discovery already exists — it's just not available to dynamic trees.

### What we keep from each system

**From the static pipeline (best features):**
- TreeNode as the universal intermediary — all signals originate from tree nodes
- Graph-backed claim coordination via `claimed_at` timestamps
- Pattern detection (indexed parameters, member-suffix grouping)
- Unit extraction via SSH with pint normalization → canonical `Unit` nodes
- Parent/sibling context injection into LLM enrichment prompts
- Version/epoch tracking via TreeModelVersion nodes
- Supervised async workers with orphan recovery

**From the signals pipeline (best features):**
- Dynamic context injection: wiki chunks, code chunks, TDI source code
  all fetched at enrichment time and injected into the LLM prompt
- Group-level semantic wiki search (per-diagnostic, per-tree)
- Facility-level wiki context (sign conventions, coordinate systems)
- Diagnostic node creation and BELONGS_TO_DIAGNOSTIC linkage
- Re-enrichable design — signals can be re-enriched as new context
  arrives (after code ingestion, wiki scraping, etc.)
- Plugin architecture with facility-agnostic scanner protocol

## Design Principles

1. **One `TreeConfig`, not `StaticTreeConfig` vs a flat string list.**
   Every MDSplus tree — regardless of whether it has versions or is
   shot-scoped — is described by the same schema class. Properties like
   `versions`, `accessor_function`, `systems` are optional fields.
   The code path chosen at runtime is a function of what's configured,
   not a type tag.

2. **TreeNode is always the intermediary.**
   No signal is created directly from SSH output. The flow is always:
   SSH extract → TreeNode in graph → promote to FacilitySignal.
   This guarantees SOURCE_NODE provenance, parent/sibling context for
   enrichment, and the ability to re-derive signals from tree structure.

3. **Shared core, specialized only when necessary.**
   Extract, units, pattern detection, promote — these operations are
   tree-generic. The only thing that differs between versioned and
   shot-scoped trees is *how many versions to extract* and *what shot
   number to use*. That difference lives in config, not in code paths.

4. **All context available to all signals.**
   Static signals currently never get wiki/code context. Dynamic signals
   never get parent/sibling context. The unified enrichment path provides
   both: tree hierarchy context from SOURCE_NODE traversal, plus wiki/code
   context from semantic search. Signals can be re-enriched as the knowledge
   graph grows.

5. **Facility-agnostic.**
   Nothing about this system is TCV-specific. All facility-specific data
   lives in `<facility>.yaml`. The pipeline code works unchanged across
   any MDSplus facility. Schema definitions enforce this separation.

## Architecture

### Unified Pipeline

```
TreeConfig (YAML)
     │
     ▼
┌─────────────────────────────────────────────────┐
│  MDSplus Scanner Plugin  (scan method)          │
│                                                 │
│  for each tree in config.trees:                 │
│    1. EXTRACT — SSH tree walk → TreeNode nodes  │
│       (versioned: one extraction per version)   │
│       (shot-scoped: one extraction at ref_shot) │
│    2. UNITS — SSH unit extraction → Unit nodes  │
│    3. PATTERNS — detect indexed/member groups   │
│    4. PROMOTE — TreeNode → FacilitySignal       │
│       + SOURCE_NODE + AT_FACILITY + DATA_ACCESS │
│       + INTRODUCED_IN (if versioned)            │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Signals Enrichment Pipeline (shared)           │
│                                                 │
│  claim discovered FacilitySignals               │
│  fetch context:                                 │
│    - tree hierarchy via SOURCE_NODE→TreeNode    │
│    - wiki chunks via semantic search            │
│    - code chunks via code_chunk_embedding       │
│    - TDI source code (if tdi_function set)      │
│  LLM structured output → enriched signals       │
│  create Diagnostic nodes + linkage              │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Check Pipeline (shared)                        │
│  validate data access for reference shot        │
└─────────────────────────────────────────────────┘
```

### Key difference from current design

Currently, static trees promote signals as `status=enriched`, skipping
the signals enrichment pipeline entirely. The tree-level LLM enrichment
(parent/sibling context) is the only enrichment they get.

In the unified design, **all promoted signals enter as `status=discovered`**
and flow through the signals enrichment pipeline. The enrichment pipeline
is enhanced to also fetch tree hierarchy context via SOURCE_NODE when
available. This means every signal gets the union of both context sources.

The tree-level LLM enrichment step in the static workers (the `enrich_worker`
in `discovery/static/workers.py`) is **removed**. Its work moves into the
signals enrichment pipeline where it can benefit from wiki and code context
too. This eliminates the duplicate enrichment infrastructure.

### Re-enrichment

Signals have an `enriched_at` timestamp. When new context arrives (code
ingestion completes, wiki re-scraped, new trees discovered), an
`--enrich-only` run can re-claim signals where:
- `enriched_at < latest_ingestion_date` for the facility, or
- `enriched_at IS NULL`, or
- explicitly requested via `--force-enrich`

This is already partially supported by the signals pipeline's
`--enrich-only` flag. The enhancement is to make the enrichment prompt
incorporate *all* available context — not just what was available at first
discovery.

## Schema Changes

### Config Schema (`facility_config.yaml`)

**Rename `StaticTreeConfig` → `TreeConfig`.**
Remove `static_trees` and `trees` (flat list) from `MDSplusConfig`.
Replace with a single `trees` list of `TreeConfig`.

```yaml
TreeConfig:
  description: >-
    Configuration for an MDSplus tree to scan during signal discovery.
    Covers both versioned (machine-description) trees and shot-scoped
    (experimental data) trees. The presence or absence of `versions`
    determines the extraction strategy — code paths are not typed.
  attributes:
    tree_name:
      description: MDSplus tree name (e.g., "static", "results", "magnetics")
      required: true
    subtree_of:
      description: >-
        Parent tree if this is a subtree. Shot-scoped subtrees are opened
        via the parent tree's shot context (e.g., "results" is a subtree
        of "tcv_shot"). Standalone trees (like "static") leave this null.
    accessor_function:
      description: >-
        TDI function that abstracts data access for this tree.
        For versioned trees, selects the correct version automatically.
        For subtrees, may provide the canonical access pattern.
    versions:
      description: >-
        Known structural versions with first applicable shot. When present,
        each version is extracted separately to build the "super tree" with
        shot-range applicability. When absent, a single extraction at
        reference_shot is performed.
      multivalued: true
      range: TreeVersion
    member_parent_types:
      description: >-
        Node types whose member children should be grouped into patterns
        for batch enrichment. Facility-specific.
      multivalued: true
    systems:
      description: >-
        Named subsystems for structural documentation. Optional.
      multivalued: true
      range: TreeSystem

TreeVersion:
  # Renamed from StaticTreeVersion — these aren't static-specific
  attributes:
    version:
      required: true
      range: integer
    first_shot:
      range: integer
    description: {}

TreeSystem:
  # Renamed from StaticTreeSystem
  attributes:
    symbol:
      required: true
    name:
      required: true
    description: {}
    size:
      range: integer
    parameters:
      multivalued: true
```

**Updated `MDSplusConfig`:**
```yaml
MDSplusConfig:
  attributes:
    connection_tree: ...     # unchanged
    trees:                   # NOW a list of TreeConfig (was flat strings)
      multivalued: true
      range: TreeConfig
    subtrees: ...            # unchanged (documentary)
    reference_shot: ...      # unchanged (default for all trees)
    exclude_node_names: ...  # unchanged
    max_nodes_per_tree: ...  # unchanged
    node_usages: ...         # unchanged
    setup_commands: ...      # unchanged
    # REMOVED: static_trees (merged into trees)
```

### Facility YAML (`tcv.yaml`)

The `trees` flat list and `static_trees` structured list merge into one:

```yaml
mdsplus:
  connection_tree: tcv_shot
  reference_shot: 85000
  setup_commands:
    - source /etc/profile.d/mdsplus.sh
  trees:
    # Versioned tree — has versions, accessor_function, systems
    - tree_name: static
      accessor_function: static
      versions:
        - version: 1
          first_shot: 1
          description: Original configuration
        # ... all 8 versions
      member_parent_types: [SIGNAL]
      systems:
        - symbol: W
          name: Individual coil turns
          # ...

    # Shot-scoped subtrees — no versions, opened via tcv_shot
    - tree_name: results
      subtree_of: tcv_shot
    - tree_name: magnetics
      subtree_of: tcv_shot
    - tree_name: diagz
      subtree_of: tcv_shot
    - tree_name: base
      subtree_of: tcv_shot
    - tree_name: ecrh
      subtree_of: tcv_shot
    - tree_name: power
      subtree_of: tcv_shot
    - tree_name: diag_act
      subtree_of: tcv_shot
    - tree_name: manual
      subtree_of: tcv_shot
    - tree_name: vsystem
      subtree_of: tcv_shot
    - tree_name: hybrid
      subtree_of: tcv_shot
```

### Graph Schema (`facility.yaml`)

Minimal changes — the graph schema is already mostly generic:

- `TreeNode.is_static` → **remove**. Replace with checking whether
  the tree has versions. Or simply track `temporality` on the node
  if needed for queries. The boolean was a type proxy.
- `TreeModelVersion` — works as-is for versioned trees. For shot-scoped
  trees, a single TreeModelVersion is created at the reference shot to
  establish the INTRODUCED_IN relationship and serve as a baseline.
- `FacilitySignal.temporality` — keep, derived from whether the tree
  has versions. Not duplicated on TreeNode.

## Module Restructure

### Current layout (two parallel systems)

```
discovery/
  static/           # Static-only pipeline
    parallel.py     # run_parallel_static_discovery
    workers.py      # extract_worker, units_worker, enrich_worker
    graph_ops.py    # seed, claim, mark, pattern detection
    state.py        # StaticDiscoveryState
  signals/
    scanners/
      mdsplus.py    # Shallow join of both systems
    parallel.py     # Signals pipeline (enrich_worker, check_worker)
mdsplus/
  static.py         # SSH extraction (async_discover_static_tree_version)
  ingestion.py      # TreeNode graph ingestion
  enrichment.py     # LLM prompt building for static nodes
  discovery.py      # Legacy TreeDiscovery class
  metadata.py       # Legacy metadata extraction
remote/scripts/
  extract_static_tree.py   # Remote: versioned tree extraction
  extract_units.py         # Remote: unit extraction
  enumerate_mdsplus.py     # Remote: dynamic tree enumeration
```

### Target layout (shared core)

```
discovery/
  mdsplus/              # Unified MDSplus tree discovery
    __init__.py
    pipeline.py         # run_tree_discovery (replaces both parallel.py files)
    workers.py          # extract_worker, units_worker (shared for all trees)
    graph_ops.py        # Merged: seed, claim, mark, pattern detection, promote
    state.py            # TreeDiscoveryState (replaces StaticDiscoveryState)
  signals/
    scanners/
      mdsplus.py        # Scanner plugin — thin, delegates to discovery/mdsplus/
    parallel.py         # Signals pipeline (enrichment, check — unchanged role)
mdsplus/
  extraction.py         # SSH extraction (replaces static.py — works for all trees)
  ingestion.py          # TreeNode graph ingestion (mostly unchanged)
  enrichment.py         # REMOVE — enrichment moves to signals pipeline
  discovery.py          # REMOVE — legacy, replaced by discovery/mdsplus/
  metadata.py           # REMOVE — legacy, replaced by extract_units.py
remote/scripts/
  extract_tree.py       # Renamed from extract_static_tree.py — works for any tree
  extract_units.py      # Unchanged
  enumerate_mdsplus.py  # REMOVE — replaced by extract_tree.py
```

### Key renames

| Old | New | Reason |
|-----|-----|--------|
| `StaticTreeConfig` | `TreeConfig` | Not static-specific |
| `StaticTreeVersion` | `TreeVersion` | Not static-specific |
| `StaticTreeSystem` | `TreeSystem` | Not static-specific |
| `StaticDiscoveryState` | `TreeDiscoveryState` | Applies to all trees |
| `discovery/static/` | `discovery/mdsplus/` | MDSplus is the domain, not static |
| `extract_static_tree.py` | `extract_tree.py` | Works for any tree |
| `static_trees` (YAML key) | removed, merged into `trees` | Unified list |
| `run_parallel_static_discovery` | `run_tree_discovery` | Not static-specific |
| `mdsplus/static.py` | `mdsplus/extraction.py` | Clearer name |
| `_scan_static_tree` / `_scan_dynamic_trees` | single loop in `scan()` | No type split |

### What gets deleted

| File | Reason |
|------|--------|
| `mdsplus/enrichment.py` | LLM enrichment moves to signals pipeline |
| `mdsplus/discovery.py` | Legacy `TreeDiscovery` class, replaced by `discovery/mdsplus/` |
| `mdsplus/metadata.py` | Legacy script-in-string approach, replaced by remote scripts |
| `remote/scripts/enumerate_mdsplus.py` | Replaced by `extract_tree.py` |
| `discovery/static/` (entire package) | Moved to `discovery/mdsplus/` |

## Implementation Phases

### Phase 1: Schema and Config

**Goal:** Unified `TreeConfig` schema, updated facility YAML.

1. Rename `StaticTreeConfig` → `TreeConfig`, `StaticTreeVersion` → `TreeVersion`,
   `StaticTreeSystem` → `TreeSystem` in `facility_config.yaml`
2. Replace `trees` (flat list) + `static_trees` in `MDSplusConfig` with
   single `trees: list[TreeConfig]`. Add `subtree_of` field.
3. Update `tcv.yaml` — merge the two lists, add `subtree_of: tcv_shot` to
   shot-scoped trees
4. Rebuild models: `uv run build-models --force`
5. Update `get_facility()` consumers (the scanner plugin, any config readers)

**Decision guidance:** Don't try to support both old and new config formats.
Just migrate. No backwards compatibility.

### Phase 2: Unified Remote Extraction

**Goal:** One remote script that extracts any MDSplus tree.

1. Extend `extract_tree.py` (renamed from `extract_static_tree.py`) to accept
   either a version number or a shot number. The script already walks `***`
   and extracts structure — the only difference is `Tree(name, version)` vs
   `Tree(name, shot)`.
2. Keep `extract_units.py` unchanged — it already works with any tree given a
   tree name and version/shot.
3. Delete `enumerate_mdsplus.py` — its functionality is subsumed.
4. Rename `mdsplus/static.py` → `mdsplus/extraction.py`. Make
   `async_discover_static_tree_version` generic: accept `shot` parameter
   (which is the version number for versioned trees, or `reference_shot` for
   shot-scoped trees).

**Decision guidance:** The remote script must stay Python 3.12+ (runs in venv).
The extraction function should take `tree_name` and `shot` — the caller decides
what shot means (version number or experimental shot).

### Phase 3: Unified Pipeline Core — `discovery/mdsplus/`

**Goal:** Move `discovery/static/` to `discovery/mdsplus/`, make it tree-generic.

1. Move files: `static/{parallel,workers,graph_ops,state}.py` →
   `mdsplus/{pipeline,workers,graph_ops,state}.py`
2. Rename `StaticDiscoveryState` → `TreeDiscoveryState`
3. Rename `run_parallel_static_discovery` → `run_tree_discovery`
4. Make `run_tree_discovery` handle both versioned and shot-scoped trees:
   - If `tree_config.versions` exists: seed TreeModelVersions, extract per version
   - If no versions: create single TreeModelVersion at `reference_shot`,
     extract once
5. Remove `enrich_worker` from `discovery/mdsplus/workers.py` — enrichment
   moves to the signals pipeline. The tree pipeline stops after EXTRACT → UNITS
   → PATTERNS → PROMOTE.
6. Add `promote_worker` to `discovery/mdsplus/workers.py` — promotes enriched
   TreeNodes to FacilitySignals. Actually, since we're removing tree-level
   enrichment, promote creates FacilitySignals with `status=discovered` from
   leaf TreeNodes (no description yet — that comes from signals enrichment).
7. Pattern detection runs after extraction, before promote. Patterns still
   create TreeNodePattern nodes that the enrichment pipeline can use for
   batch efficiency.

**Decision guidance:** The `extract_worker` and `units_worker` are already
tree-generic in their core logic. The only static-specific assumption is that
there are multiple versions to iterate. For shot-scoped trees, there's one
"version" (the reference shot). The worker loop handles this naturally —
it claims pending TreeModelVersions and there happens to be exactly one.

### Phase 4: Scanner Plugin Simplification

**Goal:** `MDSplusScanner.scan()` becomes a thin loop over `config.trees`.

1. For each `TreeConfig` in `config.trees`:
   - Call `run_tree_discovery(facility, ssh_host, tree_config, ...)` which
     handles extract → units → patterns → promote internally
   - Collect stats
2. Return `ScanResult` with empty signals list (all signals are already
   in the graph as `status=discovered` via promote). The scan result
   provides `data_access` and `stats` metadata.
3. Delete `_scan_static_tree`, `_scan_dynamic_trees`,
   `_promote_static_signals` — all absorbed into `run_tree_discovery`.

**Decision guidance:** The scanner plugin should be very thin — just config
parsing and delegation. All the intelligence lives in `discovery/mdsplus/`.

### Phase 5: Enrichment Pipeline Enhancement

**Goal:** Signals enrichment pipeline fetches tree context via SOURCE_NODE.

1. Modify `claim_signals_for_enrichment()` to also return SOURCE_NODE
   TreeNode data when available (path, parent, siblings via graph traversal).
2. Modify `enrich_worker` prompt construction:
   - When `source_node` is set: traverse `SOURCE_NODE→TreeNode` to get
     parent structure path, sibling value nodes, pattern membership
   - Add this as a "tree context" section in the prompt, alongside
     the existing wiki/code context sections
3. Delete `mdsplus/enrichment.py` (the `StaticNodeBatch` model and prompt
   builders) — these are superseded by the signals enrichment models and
   Jinja2 prompt template.
4. Update the Jinja2 prompt template (`signals/enrichment.md`) to include
   a "Tree Structure Context" section for MDSplus-sourced signals.

**Decision guidance:** Don't build a separate enrichment prompt for
tree-contextualized signals. Extend the existing prompt template with a
conditional section. The prompt already has sections for TDI, PPF, EDAS —
add one for "MDSplus Tree" that includes parent path, sibling nodes, and
pattern information when available.

### Phase 6: Cleanup

**Goal:** Remove dead code and legacy modules.

1. Delete `mdsplus/discovery.py` (legacy `TreeDiscovery` class)
2. Delete `mdsplus/metadata.py` (legacy inline scripts)
3. Delete `mdsplus/enrichment.py` (moved to signals pipeline)
4. Delete `remote/scripts/enumerate_mdsplus.py` (replaced by extract_tree.py)
5. Delete `discovery/static/` directory (moved to `discovery/mdsplus/`)
6. Remove `TreeNode.is_static` from graph schema (no longer needed)
7. Update all imports across the codebase
8. Update tests: rename `test_static_parallel.py` → `test_mdsplus_parallel.py`

## Cross-Cutting Concerns

### Signal ID construction

Currently, static and dynamic signals use different ID formats:
- Static: `{facility}:{domain}/{tree}/{path_segments}`
- Dynamic: `{facility}:{tree}/{group}/{name}`

Unify to: `{facility}:{tree}/{normalized_path}` where normalized_path
is derived from the TreeNode path. The physics_domain is a property of
the FacilitySignal, not part of its ID. This prevents ID changes when
the domain classification is corrected during enrichment.

### DataAccess nodes

Currently one DataAccess per facility. The unified system creates one per
tree (or per accessor pattern):
- `{facility}:mdsplus:{tree_name}` for direct tree access
- `{facility}:mdsplus:{accessor_function}` for accessor-mediated access

This gives agents the information they need to construct data access code
that is specific to the tree being queried.

### Enrichment batch grouping

The signals enrichment pipeline groups signals by `_signal_context_key()`.
Currently MDSplus tree signals group by `tree:{tree_name}`. With SOURCE_NODE
available, grouping can be smarter:
- Group by parent STRUCTURE node (siblings enriched together, like static)
- Group by TreeNodePattern (pattern members enriched together)
- Fall back to tree-level grouping when no structure is available

This is an optimization, not a blocker. The tree-level grouping works
initially; pattern-aware grouping can be added later.

### TreeModelVersion for shot-scoped trees

Shot-scoped trees get a single TreeModelVersion:
```
id: {facility}:{tree_name}:v1
version: 1
first_shot: {reference_shot}
description: "Baseline at shot {reference_shot}"
status: ingested
```

This establishes the INTRODUCED_IN relationship infrastructure. If epoch
detection is added later (comparing structure across shots), additional
versions can be created without changing the pipeline.

### Backwards incompatibility

This plan explicitly breaks:
- Config format: `static_trees` / flat `trees` → unified `trees: [TreeConfig]`
- Import paths: `discovery.static.*` → `discovery.mdsplus.*`
- Module names: `mdsplus.static` → `mdsplus.extraction`
- Signal IDs: format changes (re-run discovery to regenerate)
- Graph data: existing FacilitySignal nodes from old format should be cleared
  and re-discovered

This is expected and desired — no backwards compatibility required.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Removing static enrich_worker loses tree context | Signals enrich_worker enhanced to fetch tree context via SOURCE_NODE |
| Shot-scoped tree extraction much larger than versioned | `max_nodes_per_tree` limit + node_usages filtering already exist |
| Re-enrichment produces inconsistent descriptions | Track `enrichment_version` counter; description quality should monotonically improve |
| Multi-version extraction slow for shot-scoped trees | Only one "version" extracted for shot-scoped trees — no performance regression |
| Pattern detection on large dynamic trees | Channel deduplication in extract_tree.py reduces node count before graph ingestion |

## Success Criteria

1. `imas-codex discover signals tcv -s mdsplus -n 50` discovers signals from
   both static and dynamic trees through the same pipeline
2. All signals have SOURCE_NODE→TreeNode edges
3. All signals' enrichment prompts include tree hierarchy context when available
4. `TreeConfig` schema validated for TCV, JET, JT-60SA configs
5. No code references "static" as a tree type — only as a specific tree name
6. `discovery/static/` directory no longer exists
7. Re-enrichment via `--enrich-only` produces improved descriptions when new
   wiki/code context is available
