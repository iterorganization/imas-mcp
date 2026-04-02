# Staged Mapping Pipeline: Data → Error → Metadata

**Status**: Plan  
**Created**: 2025-07-24  
**Scope**: `imas_codex/ids/mapping.py`, `imas_codex/ids/tools.py`, `imas_codex/ids/models.py`

## Executive Summary

The current mapping pipeline treats all IMAS fields uniformly: signal sources are
matched against every node type (data, error, metadata) in a single LLM pass.
This proposal restructures the pipeline into three sequential stages, each
targeting a distinct node category with purpose-built logic:

1. **Stage 1 — Data Mapping** (LLM): Map facility signals to IMAS data fields
   (current pipeline, with error/metadata excluded from search)
2. **Stage 2 — Error Mapping** (Programmatic): Derive error field mappings from
   Stage 1 data mappings using `HAS_ERROR` graph relationships
3. **Stage 3 — Metadata Population** (LLM): Populate `ids_properties/*` and
   `code/*` fields from facility context (see companion plan)

## Justification

### Why Stages?

**The three field categories have fundamentally different mapping semantics:**

| Category | Source | Target | Method | Cardinality |
|----------|--------|--------|--------|-------------|
| Data | Facility signal | IMAS leaf node | Semantic match (LLM) | 1 signal → 1+ paths |
| Error | Facility signal with uncertainty | IMAS `_error_*` node | Graph traversal | 1 data mapping → 3 error paths |
| Metadata | Facility config, DD version, pipeline identity | IMAS `ids_properties/*`, `code/*` | Template + LLM extraction | 1 per IDS occurrence |

**Data mapping** requires expensive LLM reasoning about physics semantics, unit
compatibility, and coordinate structure. **Error mapping** requires no LLM at all —
it's a deterministic graph traversal. **Metadata population** requires a different
kind of LLM reasoning: extracting structured metadata from facility documentation,
not matching signals to physics concepts.

Combining these in a single LLM call wastes tokens (error fields dominate search
results), produces lower-quality results (the LLM sees noise it can't distinguish
from real targets), and makes the pipeline harder to debug and validate.

### Current Pain Points

1. **70% of vector search results are error fields** (see error-metadata-filtering
   plan) — the LLM wastes its context window evaluating `_error_index` candidates
   that are structurally determined, not semantically ambiguous.

2. **No error mappings exist today** — 0 out of 55,120 `FacilitySignal` nodes map
   to error fields, despite 311 facility signals having error-related data (density
   errors, temperature uncertainties, calibration errors). These will never be
   correctly mapped by the current single-stage approach because error fields
   crowd out their parent data fields in search.

3. **ids_properties/code fields are unmappable by signal matching** — These are
   per-IDS-occurrence metadata (who wrote it, what version, what software), not
   signal-level data. They require a fundamentally different population strategy.

## Stage 1: Data Mapping (Enhanced Current Pipeline)

### Changes from Current Pipeline

The only change to the existing pipeline is **search filtering**: all search
functions (`search_imas_semantic`, `compute_semantic_matches`, `fetch_imas_subtree`)
pass `include_error_fields=False, include_metadata=False` to exclude noise.

This is implemented by the error-metadata-filtering plan (companion document).
No structural changes to the pipeline orchestrator.

### Expected Impact

With error/metadata fields excluded from search:
- kNN vector search returns 2-3× more relevant data field candidates per query
- The LLM sees cleaner candidate lists, reducing forced low-confidence bindings
- Mapping quality should improve significantly for the same token budget

## Stage 2: Error Mapping (New — Programmatic)

### Design: Graph-Traversal Error Derivation

For each validated data mapping from Stage 1, derive error field mappings by
traversing `HAS_ERROR` relationships in the graph:

```
Data Mapping: signal_source → data_path (e.g., equilibrium/time_slice/global_quantities/ip)
     ↓
Graph Query: MATCH (d:IMASNode {id: $data_path})-[r:HAS_ERROR]->(e:IMASNode)
             RETURN e.id, r.error_type
     ↓
Error Mappings:
  - signal_source → equilibrium/time_slice/global_quantities/ip_error_upper
  - signal_source → equilibrium/time_slice/global_quantities/ip_error_lower
  - signal_source → equilibrium/time_slice/global_quantities/ip_error_index
```

### Phase 2a: Basic Error Derivation

For every Stage 1 data mapping, create error field mappings by graph traversal.
The source signal gets the same `MAPS_TO_IMAS` relationship to each error child,
but with a `mapping_type: 'error_derived'` property to distinguish from direct
signal mappings.

```python
def derive_error_mappings(
    data_mappings: list[ValidatedSignalMapping],
    gc: GraphClient,
) -> list[ValidatedSignalMapping]:
    """Derive error mappings from validated data mappings via HAS_ERROR."""
    error_mappings = []
    for dm in data_mappings:
        errors = gc.query(
            "MATCH (d:IMASNode {id: $path})-[r:HAS_ERROR]->(e:IMASNode) "
            "RETURN e.id AS error_path, r.error_type AS error_type",
            path=dm.target_path,
        )
        for err in errors:
            error_mappings.append(ValidatedSignalMapping(
                source_id=dm.source_id,
                target_path=err["error_path"],
                transform_expression=dm.transform_expression,
                source_units=dm.source_units,
                target_units=dm.target_units,
                confidence=dm.confidence,
                mapping_type="error_derived",
                derived_from=dm.target_path,
                error_type=err["error_type"],
            ))
    return error_mappings
```

**Cost**: Zero LLM tokens. One graph query per data mapping (~10ms each).

### Phase 2b: Direct Error Signal Matching

Handle the 311 facility signals that directly represent measurement uncertainties
(e.g., "HRTS Electron Density Error", "Ion Temperature Error"). These signals
should map to `_error_upper`/`_error_lower` fields directly.

**Approach**:
1. Identify facility signals whose description/name indicates uncertainty data
   (heuristic: contains "error", "uncertainty", "sigma", "err bar", etc.)
2. For each, search with `include_error_fields=True` to find the matching error
   field
3. Cross-reference with Stage 1 data mappings: if `magnetics/.../field/data` is
   mapped, and a signal named "field error" exists, validate that it maps to
   `magnetics/.../field/data_error_upper`

**Implementation**: This may require a lightweight LLM call to distinguish:
- Measurement errors → `_error_upper`/`_error_lower`
- Error indices → `_error_index`
- "Error field" physics concepts → regular data fields (e.g., magnetic error
  field correction coils map to `pf_active` or `magnetics`, not `_error_*`)

**Cost**: 10-50 LLM calls for the ~311 error-related signals (most facilities),
vs 0 for most signals.

### Phase 2c: Error Mapping Validation

Validate error mappings the same way data mappings are validated:
- Target path exists in graph
- Data type is compatible (error fields share parent's type)
- Units match (error fields share parent's units)
- No duplicate mappings

### Graph Schema Additions

```
(sg:SignalSource)-[:MAPS_TO_IMAS {
    mapping_type: 'error_derived',   # NEW: vs 'direct' for data mappings
    derived_from: 'equilibrium/...',  # NEW: parent data path
    error_type: 'upper'              # NEW: upper/lower/index
}]->(ip:IMASNode)
```

## Stage 3: Metadata Population

See companion plan: `ids-properties-population.md`

This stage addresses `ids_properties/*` and `code/*` fields which are per-IDS
metadata, not signal-level data. It requires a different input (facility config,
DD version, pipeline identity) and a different LLM strategy (template filling
vs. semantic matching).

## Pipeline Orchestration

### Updated Flow

```
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Data Mapping (existing pipeline + filter) │
│                                                      │
│  gather_context ──► assign_sections ──► map_signals │
│       │                                     │       │
│       │         (error/metadata EXCLUDED    │       │
│       │          from all searches)         │       │
│       │                                     ▼       │
│  discover_assembly ──► validate ──► persist_data    │
└──────────────────────────────┬──────────────────────┘
                               │ validated data mappings
                               ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Error Mapping (programmatic + optional LLM)│
│                                                      │
│  derive_error_mappings (graph traversal, ~0 tokens) │
│       │                                              │
│       ├── basic derivation (HAS_ERROR traversal)    │
│       ├── direct error signals (LLM for 311 sigs)   │
│       │                                              │
│       ▼                                              │
│  validate_error_mappings ──► persist_errors          │
└──────────────────────────────┬──────────────────────┘
                               │ mapping + error state
                               ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: Metadata Population (LLM per IDS)         │
│                                                      │
│  gather_facility_context                             │
│       │                                              │
│       ├── DD version → version_put fields           │
│       ├── pipeline identity → code/* fields          │
│       ├── facility config → ids_properties fields    │
│       │                                              │
│       ▼                                              │
│  populate_ids_properties (1 LLM call per IDS)       │
│       ▼                                              │
│  validate_metadata ──► persist_metadata             │
└─────────────────────────────────────────────────────┘
```

### CLI Interface

```bash
# Full pipeline (all stages)
imas map run jet pf_active

# Individual stages
imas map run jet pf_active --stage data      # Stage 1 only
imas map run jet pf_active --stage error     # Stage 2 only (requires Stage 1)
imas map run jet pf_active --stage metadata  # Stage 3 only

# Skip stages
imas map run jet pf_active --skip-errors     # Stage 1 + 3
imas map run jet pf_active --skip-metadata   # Stage 1 + 2
```

## Implementation Phases

### Phase 1: Search Filtering (prerequisite)

Implement the error-metadata-filtering plan. Without this, Stage 1 still sees
error noise. **Must land first.**

### Phase 2: Stage 2 Basic Error Derivation

Implement `derive_error_mappings()` as a post-processing step after Stage 1
validation. Wire into `generate_mapping()` orchestrator.

- Add `mapping_type`, `derived_from`, `error_type` to `ValidatedSignalMapping`
- Add `derive_error_mappings()` to `mapping.py`
- Add `persist_error_mappings()` to `models.py`
- Integration test: run Stage 1 + 2 for a known IDS, verify error mappings
  exist for every data mapping with HAS_ERROR children

### Phase 3: Stage 2b Direct Error Signal Matching

Add heuristic classifier for error-related signals and targeted LLM matching
with `include_error_fields=True`.

### Phase 4: Stage 3 Metadata Population

Implement per-IDS metadata population (see companion plan).

### Phase 5: CLI + Orchestration

Add `--stage` and `--skip-*` flags to `imas map run` CLI.

## Cost-Benefit Summary

| Stage | LLM Cost | Graph Queries | New Mappings | Quality Impact |
|-------|----------|---------------|-------------|----------------|
| Data (filtered) | Same as today | +1 WHERE clause | +0 (same count, better quality) | 2-3× more relevant candidates |
| Error (basic) | **$0** | 1 per data mapping | +3× data mapping count | Fills 31,281 previously unreachable error paths |
| Error (direct) | ~$0.10/IDS | 10-50 per facility | +311 direct error signals | Captures measurement uncertainty signals |
| Metadata | ~$0.05/IDS | 5-10 per IDS | +91 per IDS (78 ids_properties + 13 code) | Fills per-IDS provenance/versioning |

**Stage 2 basic error derivation is the highest-value, lowest-cost addition
to the pipeline** — it maps 31K error fields with zero LLM cost. 

---

## Priority & Dependencies

**Priority: P3 — Medium**

| Depends On | Enables |
|-----------|---------|
| error-metadata-filtering (✅ implemented and deleted) | ids-properties-population |

The error-metadata-filtering prerequisite is complete. This plan can proceed.

**Dependency chain:** This plan → ids-properties-population → complete mapping quality.

## Documentation Updates

When this work is complete, update:
- [ ] `AGENTS.md` — Mapping pipeline description if CLI flags change (e.g., `--stage`)
- [ ] Prompt templates — new/updated mapping prompts for each stage
- [ ] `plans/README.md` — mark as complete or move to pending if gaps remain
- [ ] `ids-properties-population.md` — unblock and update dependency status
