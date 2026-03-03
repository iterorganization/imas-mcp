# MDSplus Discovery Unification

> Merge the static and dynamic MDSplus discovery pipelines into a single,
> shared-core system where all trees — versioned, shot-scoped, and
> structurally epoched — flow through the same infrastructure. Integrate
> TDI function discovery as a higher-level access layer that links back to
> tree-scoped signals.

## Status: Implemented

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

Additionally, two parallel discovery channels operate independently:

- **TDI function discovery** (`discovery/signals/tdi.py`) creates
  FacilitySignal nodes from TDI `.fun` file parsing, with no linkage to
  the TreeNode structure those functions abstract over.
- **Epoch detection** (`mdsplus/batch_discovery.py`) discovers structural
  version boundaries for dynamic trees via SSH, but this infrastructure is
  not accessible from the unified discovery pipeline.

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

**From epoch detection (`batch_discovery.py`):**
- Batched SSH structure fingerprinting (50 shots/query)
- Binary search for exact epoch boundaries
- Checkpoint/resume for long-running discovery
- Incremental detection from existing graph state
- Structure diff with subtree-level tracking

**From TDI discovery (`discovery/signals/tdi.py`):**
- Source parsing of `.fun` files for quantities, build_paths, dependencies
- Runtime probing via MDSplus (`extract_tdi_signals.py`)
- Function type classification (dispatch, direct, parametric, inventory)
- Physics domain heuristic classification from quantity names
- TDIFunction graph nodes with source_code for LLM context

## Design Principles

1. **One `TreeConfig`, not `StaticTreeConfig` vs a flat string list.**
   Every MDSplus tree — regardless of whether it has known versions,
   discovered epochs, or is shot-scoped — is described by the same schema
   class. Properties like `versions`, `accessor_function`, `systems` are
   optional fields. The code path chosen at runtime is a function of what's
   configured, not a type tag.

2. **TreeNode is always the intermediary.**
   No signal is created directly from SSH output. The flow is always:
   SSH extract → TreeNode in graph → promote to FacilitySignal.
   This guarantees SOURCE_NODE provenance, parent/sibling context for
   enrichment, and the ability to re-derive signals from tree structure.

3. **Tree→subtree nesting in configuration.**
   Dynamic trees at TCV are subtrees of `tcv_shot` — this is reflected in
   config by nesting: `tcv_shot` has a `subtrees` list, not individual
   entries with `subtree_of`. The parent tree provides shared context
   (reference_shot, setup_commands); subtrees inherit it. This mirrors the
   actual MDSplus hierarchy and eliminates repetitive config.

4. **All trees are epoched — some just have more data.**
   The static tree has 8 *known* versions from facility documentation.
   Dynamic trees have *discoverable* epochs via structural fingerprinting.
   The unified pipeline treats both the same: TreeModelVersion nodes with
   INTRODUCED_IN/REMOVED_IN relationships. The difference is only in
   discovery method (pre-configured vs runtime detection).

5. **All context available to all signals.**
   Static signals currently never get wiki/code context. Dynamic signals
   never get parent/sibling context. The unified enrichment path provides
   both: tree hierarchy context from SOURCE_NODE traversal, plus wiki/code
   context from semantic search. Signals can be re-enriched as the knowledge
   graph grows.

6. **TDI as a higher-level access layer, not a separate discovery path.**
   TDI functions abstract over raw MDSplus tree paths. A TDI signal
   (`tcv_eq('IP')`) resolves to one or more TreeNode paths at runtime.
   The graph should capture this: TDIFunction → RESOLVES_TO → TreeNode(s).
   This enriches both sides: tree-sourced signals gain a preferred accessor
   expression, TDI-sourced signals gain structural context.

7. **Progressive async architecture.**
   Extract workers feed downstream workers (units, patterns, promote)
   as soon as tree structures arrive. Epoch detection and tree extraction
   can run concurrently across trees. The pipeline never blocks on one
   tree finishing before another starts.

8. **Facility-agnostic.**
   Nothing about this system is TCV-specific. All facility-specific data
   lives in `<facility>.yaml`. The pipeline code works unchanged across
   any MDSplus facility.

## Architecture

### Unified Pipeline

```
TreeConfig (YAML)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  MDSplus Scanner Plugin  (scan method)                         │
│                                                                │
│  for each tree in config.trees:                                │
│                                                                │
│  ┌─ EPOCH DETECT ──────────────────────────────────────────┐   │
│  │  (optional) batch fingerprint → binary search → epochs  │   │
│  │  creates TreeModelVersion nodes with shot ranges        │   │
│  │  skipped if tree has pre-configured versions            │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                    │
│           ▼                                                    │
│  ┌─ EXTRACT ───────────────────────────────────────────────┐   │
│  │  SSH tree walk → TreeNode nodes (per version/epoch)     │   │
│  │  versioned: one extraction per version                  │   │
│  │  epoched: one extraction per discovered epoch           │   │
│  │  shot-scoped (no epochs): one extraction at ref_shot    │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                    │
│           ▼                                                    │
│  ┌─ UNITS ─────────────────────────────────────────────────┐   │
│  │  SSH unit extraction → pint normalize → Unit nodes      │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                    │
│           ▼                                                    │
│  ┌─ PATTERNS ──────────────────────────────────────────────┐   │
│  │  detect indexed and member-suffix groups                │   │
│  │  create TreeNodePattern nodes                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                    │
│           ▼                                                    │
│  ┌─ PROMOTE ──────────────────────────────────────────────┐    │
│  │  TreeNode → FacilitySignal (status=discovered)         │    │
│  │  + SOURCE_NODE + AT_FACILITY + DATA_ACCESS             │    │
│  │  + INTRODUCED_IN (if versioned/epoched)                │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TDI Linkage (post-extraction, optional)                       │
│                                                                │
│  Match TDI function build_path() refs → TreeNode paths         │
│  Create RESOLVES_TO edges: TDIFunction → TreeNode              │
│  Set preferred_accessor on FacilitySignal from TDI expression  │
│  Inject TDI source_code into enrichment context                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Signals Enrichment Pipeline (shared)                          │
│                                                                │
│  claim discovered FacilitySignals                              │
│  fetch context:                                                │
│    - tree hierarchy via SOURCE_NODE→TreeNode                   │
│    - wiki chunks via semantic search                           │
│    - code chunks via code_chunk_embedding                      │
│    - TDI source code (via RESOLVES_TO→TDIFunction)             │
│    - epoch/version applicability ranges                        │
│  LLM structured output → enriched signals                      │
│  create Diagnostic nodes + linkage                             │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Check Pipeline (shared)                                       │
│  validate data access for reference shot                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key differences from current design

1. **Static→enriched bypass eliminated.** Currently, static trees promote
   signals as `status=enriched`, skipping the signals enrichment pipeline.
   In the unified design, all promoted signals enter as `status=discovered`
   and flow through the signals enrichment pipeline.

2. **Epoch detection integrated.** The `BatchDiscovery` infrastructure
   (currently a standalone script) becomes a stage in the tree pipeline.
   This means `results`, `magnetics`, and other dynamic trees can
   automatically discover their structural versions.

3. **TDI linkage as post-processing.** After tree extraction, TDI
   `build_path()` references are matched to TreeNode paths. This creates
   bidirectional context: tree signals know their TDI accessor, TDI signals
   know their backing tree structure.

4. **Progressive workers.** Extract workers for different trees run
   concurrently. As each tree's extraction completes, its units/patterns/
   promote workers start immediately — no serial bottleneck.

### Re-enrichment

Signals have an `enriched_at` timestamp. When new context arrives (code
ingestion completes, wiki re-scraped, new trees discovered, epoch detection
runs), an `--enrich-only` run can re-claim signals where:
- `enriched_at < latest_ingestion_date` for the facility, or
- `enriched_at IS NULL`, or
- new TDI linkage established (TDI function ingested after initial enrichment)
- new epoch detected (structural context changed)
- explicitly requested via `--force-enrich`

The enrichment prompt incorporates *all* available context. As the
knowledge graph grows (wiki scraped, code ingested, TDI functions parsed,
epochs detected), re-enrichment produces progressively better descriptions.

## Schema Changes

### Config Schema (`facility_config.yaml`)

**Rename `StaticTreeConfig` → `TreeConfig`.**
Remove `static_trees` and `trees` (flat list) from `MDSplusConfig`.
Replace with a single `trees` list of `TreeConfig`.

```yaml
TreeConfig:
  description: >-
    Configuration for an MDSplus tree to scan during signal discovery.
    A tree may be standalone (like "static"), or serve as a parent that
    organizes subtrees (like "tcv_shot" containing "results", "magnetics",
    etc.). The subtrees field groups related trees that share the parent's
    shot context. The presence of `versions` determines whether extraction
    uses pre-configured version numbers; otherwise, epoch detection or a
    single reference_shot extraction is used.
  attributes:
    tree_name:
      description: >-
        MDSplus tree name (e.g., "static", "tcv_shot").
        For parent trees with subtrees, this is the connection tree.
      required: true
    subtrees:
      description: >-
        Named subtrees to extract. Each subtree is opened as an independent
        MDSplus tree using the parent's reference_shot. The parent tree
        itself is not directly scanned — only its subtrees are extracted.
        If empty or absent, the tree itself is scanned directly.
      multivalued: true
    reference_shot:
      description: >-
        Reference shot for extraction. For parent trees, inherited by all
        subtrees. Overrides the global MDSplusConfig.reference_shot.
      range: integer
    accessor_function:
      description: >-
        TDI function that abstracts data access for this tree.
        For versioned trees, selects the correct version automatically.
      range: string
    detect_epochs:
      description: >-
        Whether to run epoch detection for this tree or its subtrees.
        When true, the pipeline runs batch fingerprinting + binary search
        to discover structural version boundaries before extraction.
        Default false for trees with pre-configured versions.
      range: boolean
    epoch_config:
      description: Configuration for epoch detection when detect_epochs is true.
      range: EpochConfig
    versions:
      description: >-
        Known structural versions with first applicable shot. When present,
        each version is extracted separately to build the "super tree" with
        shot-range applicability. When absent, either epoch detection runs
        (if detect_epochs=true) or a single extraction at reference_shot.
      multivalued: true
      range: TreeVersion
    member_parent_types:
      description: >-
        Node types whose member children should be grouped into patterns
        for batch enrichment. Facility-specific.
      multivalued: true
    systems:
      description: Named subsystems for structural documentation. Optional.
      multivalued: true
      range: TreeSystem

EpochConfig:
  description: Configuration for structural epoch detection.
  attributes:
    coarse_step:
      description: Shot step for coarse fingerprint scan (default 1000).
      range: integer
    start_shot:
      description: Start of epoch search range (default 3000).
      range: integer
    shot_tree:
      description: >-
        Tree to query for current shot number (default: parent tree_name).
      range: string

TreeVersion:
  attributes:
    version:
      required: true
      range: integer
    first_shot:
      range: integer
    description: {}

TreeSystem:
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
    trees:                   # NOW a list of TreeConfig (was flat strings + static_trees)
      multivalued: true
      range: TreeConfig
    reference_shot: ...      # unchanged (global default for all trees)
    exclude_node_names: ...  # unchanged
    max_nodes_per_tree: ...  # unchanged
    node_usages: ...         # unchanged
    setup_commands: ...      # unchanged
    # REMOVED: static_trees (merged into trees)
    # REMOVED: subtrees (documentary list, replaced by per-tree subtrees field)
```

### Facility YAML (`tcv.yaml`)

The `trees` flat list and `static_trees` structured list merge into one.
Subtrees are nested under their parent tree, not listed flat with
`subtree_of`:

```yaml
mdsplus:
  connection_tree: tcv_shot
  reference_shot: 85000
  setup_commands:
    - source /etc/profile.d/mdsplus.sh
  node_usages: [SIGNAL, NUMERIC]
  exclude_node_names: [COMMENTS, DATE, LAST_TRIAL, ...]
  trees:
    # Parent tree with subtrees — subtrees share reference_shot
    - tree_name: tcv_shot
      subtrees:
        - results      # Equilibrium, analysis outputs (~1500 signals)
        - magnetics    # Magnetic field measurements (~215 signals)
        - diagz        # Diagnostic measurements (~220 signals)
        - base         # Core plasma parameters (~287 signals)
        - ecrh         # Electron cyclotron heating (~416 signals)
        - power        # Power supply data
        - diag_act     # Diagnostic actions
        - manual       # Manually-populated signals
        - vsystem      # Vista control parameters
        - hybrid       # Hybrid heating scenarios (16k nodes)
      detect_epochs: true
      epoch_config:
        coarse_step: 1000
        start_shot: 3000

    # Standalone versioned tree — has pre-configured versions, no subtrees
    - tree_name: static
      accessor_function: static
      versions:
        - version: 1
          first_shot: 1
          description: Original configuration
        - version: 2
          first_shot: 13551
          description: Belt limiter removed
        - version: 3
          first_shot: 49829
          description: NBH ports added
        - version: 4
          first_shot: 63053
          description: 1st-gen HFS baffle only (SI-NO)
        - version: 5
          first_shot: 63528
          description: 1st-gen HFS and LFS baffles (SI-LO)
        - version: 6
          first_shot: 65794
          description: Un-baffled (with port protection tiles)
        - version: 7
          first_shot: 70972
          description: 1st-gen HFS and 2nd-gen LFS baffles (SI-SO)
        - version: 8
          first_shot: 74971
          description: 2nd-gen HFS and 1st-gen LFS baffles (LI-LO)
      member_parent_types: [SIGNAL]
      systems:
        - symbol: W
          name: Individual coil turns
          size: 830
          parameters: [R, Z, W, H, A, NA, NB, NT, N, XSECT]
        # ... (remaining systems unchanged)
```

**Config parsing logic:**
```python
for tree_config in mdsplus_config["trees"]:
    if tree_config.get("subtrees"):
        # Parent tree: iterate subtrees, each inherits parent's reference_shot
        for subtree_name in tree_config["subtrees"]:
            yield subtree_name, tree_config  # subtree inherits parent config
    elif tree_config.get("versions"):
        # Versioned tree: extract per version
        yield tree_config["tree_name"], tree_config
    else:
        # Standalone tree: extract at reference_shot
        yield tree_config["tree_name"], tree_config
```

### Graph Schema (`facility.yaml`)

Minimal changes — the graph schema is already mostly generic:

- `TreeNode.is_static` → **remove**. Replace with `temporality` enum
  (already on FacilitySignal, extend to TreeNode if needed for queries).
- `TreeModelVersion` — works as-is for all tree types. Versioned trees
  have pre-configured versions, epoched trees have discovered versions,
  un-epoched trees get a single version at reference_shot.
- Add `TDIFunction` → `RESOLVES_TO` → `TreeNode` relationship for
  TDI linkage.
- FacilitySignal gains `preferred_accessor` field for TDI expression
  when available (distinct from `accessor` which is the raw tree path).

## Existing Infrastructure Inventory

### Epoch Detection (ready to integrate)

**Location:** `imas_codex/mdsplus/batch_discovery.py` (805 lines)

The `BatchDiscovery` class provides the core epoch detection algorithm:

1. **Coarse scan** — `batch_query_structures()` queries 50 shots per SSH
   call, computing MD5 fingerprints of sorted path lists. Scans from
   current shot back to start in configurable steps.
2. **Binary search** — `binary_search_boundary()` finds exact shot where
   structure changed between two fingerprints. Handles missing shots by
   probing nearby candidates in batches.
3. **Incremental mode** — checks existing graph epochs and only scans
   shots beyond the latest known epoch boundary.
4. **Checkpoint/resume** — `DiscoveryCheckpoint` saves progress to JSON
   for interrupted runs.

The CLI wrapper (`scripts/discover_mdsplus.py`) orchestrates:
- Epoch discover → boundary refine → TreeModelVersion ingest →
  super tree build → metadata enrichment → legacy cleanup

**Key data structures:**
```python
# Epoch record (output of discover_epochs_optimized)
{
    "id": "tcv:results:v3",
    "tree_name": "results",
    "facility_id": "tcv",
    "version": 3,
    "first_shot": 65000,
    "last_shot": 72000,
    "node_count": 1523,
    "nodes_added": 42,
    "nodes_removed": 3,
    "added_subtrees": ["LIUQE3"],
    "added_paths": [...],
    "is_new": True
}
```

**Integration path:** The standalone CLI becomes a phase in the unified
pipeline. `discover_epochs_optimized()` is called for trees with
`detect_epochs: true`. Its output feeds directly into the existing
`seed_versions` → `extract_worker` flow — discovered epochs become
TreeModelVersion nodes, just like pre-configured versions.

### TDI Discovery (ready to integrate)

**Location:** `imas_codex/discovery/signals/tdi.py` (555 lines)

The TDI discovery pipeline:

1. **Remote extraction** — `extract_tdi_functions()` runs
   `extract_tdi_functions.py` via SSH, parsing `.fun` files for:
   - Function signatures and parameters
   - `case()` quantities (dispatch functions like `tcv_get`)
   - `build_path()` references to MDSplus tree paths
   - Function dependencies (TDI→TDI call graph)
   - Shot-conditional logic

2. **Signal creation** — `discover_tdi_signals()` converts quantities
   to FacilitySignal nodes with heuristic physics domain classification.

3. **Enhanced extraction** — `extract_tdi_signals.py` (649 lines) provides
   a richer two-phase approach:
   - Source parsing: case blocks, build_path refs, make_with_units, section
     comments (physics domain hints from `/* CASE --- PLASMA CURRENT ---*/`)
   - Runtime probing: executes TDI expressions via MDSplus to get actual
     resolved paths, units, shapes, dtypes

4. **Function type classification:**
   - `dispatch` — case-based multi-quantity (tcv_get, tcv_eq)
   - `direct` — single-value accessor (tcv_ip, li_1)
   - `parametric` — parameterized but not dispatch
   - `inventory` — returns path arrays
   - `internal` — underscore-prefixed helpers

**Current gaps:**
- TDI signals and tree-extracted signals exist independently in the graph
- No `RESOLVES_TO` relationship links TDI functions to TreeNode paths
- TDI `build_path()` references contain the information needed to create
  this linkage, but it's not used
- The TDI scanner plugin (`scanners/tdi.py`) wraps this as a separate
  scanner type — signals from TDI never get tree hierarchy context

**Integration path:** After tree extraction creates TreeNode nodes, a
TDI linkage phase matches `build_path()` references against TreeNode paths.
This creates bidirectional relationships and enriches both discovery channels.

### Legacy Sequential Discovery

**Location:** `imas_codex/mdsplus/discovery.py` (333 lines)

The `TreeDiscovery` class and `discover_epochs()` function implement the
original per-shot scanning approach. This is **superseded** by
`BatchDiscovery` in `batch_discovery.py` which is 10-50x faster. The
sequential mode is retained only as a fallback (`--sequential` flag).

**Plan:** Delete `discovery.py` after confirming all functionality is
covered by `batch_discovery.py`.

## Module Restructure

### Current layout (fragmented systems)

```
discovery/
  static/               # Static-only pipeline
    parallel.py         # run_parallel_static_discovery
    workers.py          # extract_worker, units_worker, enrich_worker
    graph_ops.py        # seed, claim, mark, pattern detection
    state.py            # StaticDiscoveryState
  signals/
    scanners/
      mdsplus.py        # Shallow join of static + dynamic
      tdi.py            # TDI scanner plugin (separate)
    parallel.py         # Signals pipeline (enrich_worker, check_worker)
    tdi.py              # TDI discovery orchestration
mdsplus/
  static.py             # SSH extraction (async_discover_static_tree_version)
  ingestion.py          # TreeNode graph ingestion
  enrichment.py         # LLM prompt building for static nodes
  discovery.py          # Legacy TreeDiscovery class (sequential)
  batch_discovery.py    # BatchDiscovery + epoch detection (optimized)
  metadata.py           # Legacy metadata extraction
remote/scripts/
  extract_static_tree.py   # Remote: versioned tree extraction
  extract_units.py         # Remote: unit extraction
  enumerate_mdsplus.py     # Remote: dynamic tree enumeration
  extract_tdi_functions.py # Remote: TDI .fun file parsing (basic)
  extract_tdi_signals.py   # Remote: TDI signal extraction (enhanced)
scripts/
  discover_mdsplus.py      # CLI: epoch detection + super tree (standalone)
```

### Target layout (shared core)

```
discovery/
  mdsplus/                 # Unified MDSplus tree discovery
    __init__.py
    pipeline.py            # run_tree_discovery (replaces both parallel.py files)
    workers.py             # extract_worker, units_worker, promote_worker
    graph_ops.py           # Merged: seed, claim, mark, pattern detect, promote
    state.py               # TreeDiscoveryState (replaces StaticDiscoveryState)
    epochs.py              # Epoch detection integration (wraps BatchDiscovery)
    tdi_linkage.py         # TDI→TreeNode linkage post-processing
  signals/
    scanners/
      mdsplus.py           # Scanner plugin — thin, delegates to discovery/mdsplus/
      tdi.py               # TDI scanner — delegates to tdi.py, triggers linkage
    parallel.py            # Signals pipeline (enrichment, check — enhanced)
    tdi.py                 # TDI discovery (mostly unchanged)
mdsplus/
  extraction.py            # SSH extraction (replaces static.py — any tree)
  ingestion.py             # TreeNode graph ingestion (mostly unchanged)
  batch_discovery.py       # BatchDiscovery + epoch detection (unchanged)
  # REMOVED: enrichment.py, discovery.py, metadata.py
remote/scripts/
  extract_tree.py          # Renamed from extract_static_tree.py — any tree
  extract_units.py         # Unchanged
  extract_tdi_signals.py   # TDI extraction (enhanced version kept)
  # REMOVED: enumerate_mdsplus.py, extract_tdi_functions.py (superseded)
scripts/
  # REMOVED: discover_mdsplus.py (absorbed into pipeline)
```

### Key renames

| Old | New | Reason |
|-----|-----|--------|
| `StaticTreeConfig` | `TreeConfig` | Not static-specific |
| `StaticTreeVersion` | `TreeVersion` | Not static-specific |
| `StaticTreeSystem` | `TreeSystem` | Not static-specific |
| `StaticDiscoveryState` | `TreeDiscoveryState` | Applies to all trees |
| `discovery/static/` | `discovery/mdsplus/` | MDSplus is the domain |
| `extract_static_tree.py` | `extract_tree.py` | Works for any tree |
| `static_trees` + `trees` (YAML) | `trees: [TreeConfig]` | Unified list |
| `run_parallel_static_discovery` | `run_tree_discovery` | Not static-specific |
| `mdsplus/static.py` | `mdsplus/extraction.py` | Clearer name |

### What gets deleted

| File | Reason |
|------|--------|
| `mdsplus/enrichment.py` | LLM enrichment moves to signals pipeline |
| `mdsplus/discovery.py` | Legacy sequential mode, superseded by `batch_discovery.py` |
| `mdsplus/metadata.py` | Legacy script-in-string, replaced by remote scripts |
| `remote/scripts/enumerate_mdsplus.py` | Replaced by `extract_tree.py` |
| `remote/scripts/extract_tdi_functions.py` | Superseded by `extract_tdi_signals.py` |
| `discovery/static/` (entire package) | Moved to `discovery/mdsplus/` |
| `scripts/discover_mdsplus.py` | CLI absorbed into unified pipeline |

## Implementation Phases

### Phase 1: Schema and Config — Tree→Subtree Nesting ✅

**Goal:** Unified `TreeConfig` schema with subtree nesting, updated facility YAML.

1. Rename `StaticTreeConfig` → `TreeConfig`, `StaticTreeVersion` → `TreeVersion`,
   `StaticTreeSystem` → `TreeSystem` in `facility_config.yaml`
2. Add `subtrees`, `detect_epochs`, `epoch_config` fields to `TreeConfig`
3. Add `EpochConfig` class to schema
4. Replace `trees` (flat list) + `static_trees` in `MDSplusConfig` with
   single `trees: list[TreeConfig]`
5. Update `tcv.yaml`:
   - `tcv_shot` becomes a tree entry with `subtrees: [results, magnetics, ...]`
   - `static` becomes a tree entry with `versions: [...]`
   - Remove old `trees:` flat list and `static_trees:` section
   - Remove documentary `subtrees:` list (now expressed in tree config)
6. Rebuild models: `uv run build-models --force`
7. Update `get_facility()` consumers (scanner plugin, config readers)

**Config parsing changes:**
```python
def iter_extraction_targets(mdsplus_config, global_ref_shot):
    """Yield (tree_name, shot_or_version, tree_config) tuples."""
    for tree in mdsplus_config["trees"]:
        ref = tree.get("reference_shot", global_ref_shot)
        if tree.get("subtrees"):
            # Parent: yield each subtree with parent's config
            for subtree in tree["subtrees"]:
                yield subtree, ref, tree
        elif tree.get("versions"):
            # Versioned: yield tree for each version
            for v in tree["versions"]:
                yield tree["tree_name"], v["version"], tree
        else:
            # Standalone: yield tree at reference_shot
            yield tree["tree_name"], ref, tree
```

### Phase 2: Epoch Detection Integration ✅

**Goal:** Integrate `BatchDiscovery` into the unified pipeline for
trees with `detect_epochs: true`.

1. Create `discovery/mdsplus/epochs.py` — thin wrapper around
   `BatchDiscovery.discover_epochs_optimized()`:
   ```python
   async def detect_tree_epochs(
       facility: str,
       tree_name: str,
       tree_config: dict,
       client: GraphClient,
   ) -> list[dict]:
       """Discover structural epochs for a tree.

       Returns epoch records compatible with seed_versions().
       """
       epoch_config = tree_config.get("epoch_config", {})
       epochs, structures = discover_epochs_optimized(
           facility=facility,
           tree_name=tree_name,
           start_shot=epoch_config.get("start_shot", 3000),
           coarse_step=epoch_config.get("coarse_step", 1000),
           client=client,
       )
       return epochs
   ```

2. Integrate into the extraction pipeline:
   - Before extraction, check `detect_epochs`:
     - If true: run epoch detection, create TreeModelVersion nodes from results
     - If false and versions configured: seed from config (existing behavior)
     - If false and no versions: create single version at reference_shot
   - Extraction then iterates discovered epochs just like configured versions

3. For parent trees with subtrees and `detect_epochs: true`:
   - Each subtree is independently epoched (structure changes independently)
   - Epoch detection runs per subtree, not per parent tree
   - Progress reporting shows per-subtree epoch counts

4. Checkpoint integration — the pipeline's checkpoint mechanism stores
   epoch detection state alongside extraction state, enabling resume
   across both phases.

**Epoch detection for `tcv_shot` subtrees (example run):**
```
Detecting epochs for tcv:results...
  Coarse scan: 85000 → 3000 (step=1000, 82 queries)
  Found 3 boundaries, refining...
  Discovered 4 epochs: v1(3000-42000), v2(42001-65000), v3(65001-78000), v4(78001-current)
Detecting epochs for tcv:magnetics...
  Coarse scan: 85000 → 3000 (step=1000)
  No structural changes detected — single epoch.
Extracting tcv:results v4 (shot 78001)... 1523 nodes
Extracting tcv:results v3 (shot 65001)... 1481 nodes
...
```

### Phase 3: Unified Remote Extraction ✅

**Goal:** One remote script that extracts any MDSplus tree.

1. Rename `extract_static_tree.py` → `extract_tree.py`. Accept either
   a version number or a shot number via the same `shot` parameter. The
   script already walks `***` and extracts structure — the only difference
   is `Tree(name, version)` vs `Tree(name, shot)`.
2. Keep `extract_units.py` unchanged — already works with any tree.
3. Delete `enumerate_mdsplus.py` — its functionality is subsumed.
4. Rename `mdsplus/static.py` → `mdsplus/extraction.py`. Make
   `async_discover_static_tree_version` generic: accept `shot` parameter
   (the caller decides what shot means).

**Remote script changes:** Minimal — MDSplus `Tree(name, shot)` already
works for both version numbers and experimental shot numbers. The main
change is removing the "static" naming assumption and accepting node_usages
filtering for dynamic trees.

### Phase 4: Unified Pipeline Core — `discovery/mdsplus/` ✅

**Goal:** Move `discovery/static/` to `discovery/mdsplus/`, make it tree-generic.

1. Move files: `static/{parallel,workers,graph_ops,state}.py` →
   `mdsplus/{pipeline,workers,graph_ops,state}.py`
2. Rename `StaticDiscoveryState` → `TreeDiscoveryState`
3. Rename `run_parallel_static_discovery` → `run_tree_discovery`
4. Make `run_tree_discovery` handle all tree types:
   - Versioned (config): seed from `tree_config.versions`
   - Epoched (detected): seed from epoch detection results
   - Shot-scoped: create single version at `reference_shot`
5. Remove `enrich_worker` from workers — enrichment moves to signals pipeline.
   Pipeline stops after EXTRACT → UNITS → PATTERNS → PROMOTE.
6. Add `promote_worker` — creates FacilitySignal (status=discovered) from
   leaf TreeNodes. No description yet — that comes from signals enrichment.
7. Pattern detection runs after extraction, before promote.

**Progressive async architecture:**
```python
async def run_tree_discovery(facility, ssh_host, trees_config, state):
    """Run tree discovery for all configured trees concurrently."""
    async with anyio.create_task_group() as tg:
        for tree_config in trees_config:
            tg.start_soon(process_tree, facility, ssh_host, tree_config, state)

async def process_tree(facility, ssh_host, tree_config, state):
    """Process a single tree (or parent with subtrees)."""
    trees_to_extract = []

    if tree_config.get("subtrees"):
        for subtree in tree_config["subtrees"]:
            trees_to_extract.append((subtree, tree_config))
    else:
        trees_to_extract.append((tree_config["tree_name"], tree_config))

    # Phase 1: Epoch detection (concurrent per subtree)
    async with anyio.create_task_group() as tg:
        for tree_name, config in trees_to_extract:
            if config.get("detect_epochs"):
                tg.start_soon(detect_and_seed_epochs, tree_name, config, state)
            elif config.get("versions"):
                seed_versions_from_config(tree_name, config, state)
            else:
                seed_single_version(tree_name, config, state)

    # Phase 2: Extract → Units → Patterns → Promote (workers claim from queue)
    async with SupervisedWorkerGroup(state) as group:
        group.add_workers("extract", extract_worker, count=2)
        group.add_workers("units", units_worker, count=1)
        group.add_workers("patterns", patterns_worker, count=1)
        group.add_workers("promote", promote_worker, count=1)
        await group.run()
```

### Phase 5: TDI Linkage ✅

**Goal:** Link TDI functions to TreeNode paths for bidirectional context.

1. Create `discovery/mdsplus/tdi_linkage.py`:
   ```python
   async def link_tdi_to_tree_nodes(
       gc: GraphClient,
       facility: str,
   ):
       """Match TDI build_path() refs to TreeNode paths.

       For each TDIFunction with build_path references:
       1. Normalize the build_path to canonical TreeNode ID format
       2. MATCH against existing TreeNode nodes
       3. Create RESOLVES_TO edges
       4. Set preferred_accessor on matching FacilitySignals
       """
       gc.query('''
           MATCH (tf:TDIFunction {facility_id: $facility})
           WHERE tf.mdsplus_trees IS NOT NULL
           UNWIND tf.mdsplus_trees AS tree_name
           MATCH (tn:TreeNode {facility_id: $facility})
           WHERE tn.tree_name = tree_name
             AND any(bp IN tf.build_paths WHERE tn.path ENDS WITH bp)
           MERGE (tf)-[:RESOLVES_TO]->(tn)
       ''', facility=facility)
   ```

2. After tree extraction + TDI discovery both complete, run linkage.
   This is a graph-only operation — no SSH needed.

3. Update FacilitySignal nodes that have matching TDI accessors:
   - Set `preferred_accessor` to the TDI expression
   - FacilitySignals from tree extraction gain TDI context
   - FacilitySignals from TDI scanning gain tree context via reverse traversal

4. The enrichment pipeline uses RESOLVES_TO edges to inject TDI source
   code into the prompt for tree-sourced signals, and tree hierarchy
   context for TDI-sourced signals.

**Why post-processing, not inline?** TDI and tree extraction may run at
different times or in different sessions. The linkage is idempotent —
running it after either discovery completes creates whatever edges are
possible with current graph state. Re-running after both adds the
remaining edges.

### Phase 6: Scanner Plugin Simplification ✅

**Goal:** `MDSplusScanner.scan()` becomes a thin loop over `config.trees`.

1. For each `TreeConfig` in `config.trees`:
   - Call `run_tree_discovery(facility, ssh_host, tree_config, ...)` which
     handles epoch detect → extract → units → patterns → promote internally
   - Collect stats
2. After tree extraction, run TDI linkage if TDI signals exist in graph
3. Return `ScanResult` with empty signals list (signals already in graph
   as `status=discovered` via promote)
4. Delete `_scan_static_tree`, `_scan_dynamic_trees`,
   `_promote_static_signals` — all absorbed into `run_tree_discovery`

### Phase 7: Enrichment Pipeline Enhancement ✅

**Goal:** Signals enrichment pipeline fetches all available context.

1. Modify `claim_signals_for_enrichment()` to also return:
   - SOURCE_NODE TreeNode data (path, parent, siblings)
   - TreeNodePattern membership (for batch optimization)
   - RESOLVES_TO TDIFunction data (source_code, signature)
   - Epoch/version applicability ranges

2. Modify `enrich_worker` prompt construction:
   - When `source_node` is set: traverse SOURCE_NODE→TreeNode for
     parent structure, sibling nodes, pattern membership
   - When TDI linkage exists: include TDI function source_code and
     quantity-level comments from `.fun` file parsing
   - When epoch data exists: note signal's applicability range
   - All of this alongside existing wiki/code semantic search context

3. Update Jinja2 prompt template (`signals/enrichment.md`):
   ```
   {% if tree_context %}
   ## MDSplus Tree Structure Context
   Path: {{ tree_context.path }}
   Parent: {{ tree_context.parent_path }}
   Siblings: {{ tree_context.sibling_paths | join(', ') }}
   {% if tree_context.pattern %}
   Pattern: {{ tree_context.pattern.name }} ({{ tree_context.pattern.type }})
   {% endif %}
   {% if tree_context.epochs %}
   Applicability: shots {{ tree_context.first_shot }}-{{ tree_context.last_shot }}
   {% endif %}
   {% endif %}

   {% if tdi_context %}
   ## TDI Function Context
   Function: {{ tdi_context.name }}({{ tdi_context.signature }})
   Source: {{ tdi_context.source_snippet }}
   {% endif %}
   ```

4. Delete `mdsplus/enrichment.py` — superseded by the unified template.

### Phase 8: Cleanup ✅

**Goal:** Remove dead code and legacy modules.

1. ~~Delete `mdsplus/discovery.py` (legacy sequential `TreeDiscovery`)~~ ✅
2. ~~Delete `mdsplus/metadata.py` (legacy inline scripts)~~ ✅
3. Delete `mdsplus/enrichment.py` — deferred (still used by `discovery/static/workers.py`)
4. ~~Delete `remote/scripts/enumerate_mdsplus.py` (replaced by extract_tree.py)~~ ✅
5. Delete `remote/scripts/extract_tdi_functions.py` — deferred (still used by `discovery/signals/tdi.py`)
6. ~~Delete `scripts/discover_mdsplus.py` (absorbed into pipeline)~~ ✅
7. `discovery/static/` converted to backward-compat re-export shims (graph_ops content moved to `discovery/mdsplus/graph_ops.py`)
8. Remove `TreeNode.is_static` — deferred (schema changes are additive-only per AGENTS.md; used in 30+ Cypher queries)
9. ~~Update all imports across the codebase~~ ✅
10. ~~Update tests: rename test files, add epoch detection tests~~ ✅

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
- `{facility}:tdi:functions` for TDI function access (existing)

### Enrichment batch grouping

With SOURCE_NODE available, grouping can be smarter than tree-level:
- Group by parent STRUCTURE node (siblings enriched together)
- Group by TreeNodePattern (pattern members enriched together)
- Group by TDI function (quantities from same function share context)
- Fall back to tree-level grouping when no structure is available

### TreeModelVersion for all tree types

| Tree type | TreeModelVersion strategy |
|-----------|--------------------------|
| Versioned (static) | One per configured version, from YAML `versions` |
| Epoched (detect_epochs) | One per discovered epoch, from `BatchDiscovery` |
| Shot-scoped (no epochs) | One at reference_shot, as baseline |

All types use the same INTRODUCED_IN/REMOVED_IN relationship infrastructure.
Epoch detection for a previously un-epoched tree just adds more
TreeModelVersion nodes — no schema or pipeline changes needed.

### TDI↔Tree signal deduplication

After TDI linkage, some signals may exist twice: once from tree extraction
(path-based accessor) and once from TDI extraction (function-based accessor).
These are **not** duplicates — they represent different access patterns for
the same physical quantity. Both are kept. The TDI signal gets a
`source_node` link to the same TreeNode. The tree signal gets a
`preferred_accessor` pointing to the TDI expression. An agent choosing
how to access a signal can see both options and pick the higher-level one.

### Backwards incompatibility

This plan explicitly breaks:
- Config format: `static_trees` / flat `trees` → unified `trees: [TreeConfig]`
  with subtree nesting
- Import paths: `discovery.static.*` → `discovery.mdsplus.*`
- Module names: `mdsplus.static` → `mdsplus.extraction`
- Signal IDs: format changes (re-run discovery to regenerate)
- Graph data: existing FacilitySignal nodes should be cleared and re-discovered
- CLI: `discover-mdsplus` standalone script → integrated into `imas-codex
  discover signals`

This is expected and desired — no backwards compatibility required.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Removing static enrich_worker loses tree context | Signals enrich_worker enhanced to fetch tree context via SOURCE_NODE |
| Shot-scoped tree extraction much larger than versioned | `max_nodes_per_tree` limit + `node_usages` filtering already exist |
| Epoch detection for all subtrees is slow | Epoch detection runs concurrently per subtree; coarse_step is configurable; incremental mode skips known epochs |
| TDI linkage produces false matches | `build_path()` references are exact MDSplus paths — match is deterministic, not heuristic |
| Re-enrichment produces inconsistent descriptions | Track `enrichment_version` counter; context only grows, so quality should monotonically improve |
| Pattern detection on large dynamic trees | Channel deduplication in extract_tree.py reduces node count before graph ingestion |
| `BatchDiscovery` uses raw SSH subprocess | Migrate to `run_python_script()` / `async_run_python_script()` executor in Phase 2 for zombie prevention |

## Success Criteria

1. `imas-codex discover signals tcv -s mdsplus -n 50` discovers signals from
   both static and dynamic trees through the same pipeline
2. All signals have SOURCE_NODE→TreeNode edges
3. All signals' enrichment prompts include tree hierarchy context when available
4. `TreeConfig` schema validated for TCV (and future JET, JT-60SA configs)
5. No code references "static" as a tree type — only as a specific tree name
6. `discovery/static/` directory no longer exists
7. Dynamic trees with `detect_epochs: true` produce TreeModelVersion nodes
   with shot ranges comparable to the standalone `discover-mdsplus` CLI
8. TDI functions with `build_path()` references have RESOLVES_TO edges
   to matching TreeNode nodes
9. Re-enrichment via `--enrich-only` produces improved descriptions when
   new context (wiki, code, TDI, epochs) is available
10. No standalone CLI scripts for epoch detection — all integrated into
    the `imas-codex discover signals` command

## Phase Dependencies

```
Phase 1 (Schema/Config)
  │
  ├──→ Phase 2 (Epoch Detection)
  │       │
  ├──→ Phase 3 (Remote Extraction)
  │       │
  │       ▼
  └──→ Phase 4 (Pipeline Core) ──→ Phase 5 (TDI Linkage)
                │                          │
                └──────────────────────────┘
                              │
                              ▼
                    Phase 6 (Scanner Plugin)
                              │
                              ▼
                    Phase 7 (Enrichment Enhancement)
                              │
                              ▼
                    Phase 8 (Cleanup)
