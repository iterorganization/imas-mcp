# Standard Name Generation Pipeline

> DD-enriched standard name generation: classify IMAS paths, batch by
> physics concept, compose names via LLM, validate grammar, consolidate
> across batches, and persist to the knowledge graph.

## Overview

Standard names give each IMAS Data Dictionary path a canonical, human-readable
physics identity — e.g. `electron_temperature` for
`core_profiles/profiles_1d/electrons/temperature`. They bridge the structural
DD namespace and the semantic physics vocabulary, enabling cross-facility and
cross-IDS data discovery.

The naming system follows a **composable grammar** (from `imas_standard_names`)
where a name is built from ordered segments:

```
[subject_] [coordinate_] [position_] physical_base [_component] [_process]
```

Examples: `electron_temperature`, `toroidal_magnetic_field_at_magnetic_axis`,
`ion_deuterium_density`.

## Pipeline Flow

```
 ┌───────────┐     ┌───────────┐     ┌───────────┐
 │  EXTRACT  │────▶│  COMPOSE  │────▶│  REVIEW   │  (optional)
 │           │     │           │     │           │
 │ DD query  │     │ LLM call  │     │ LLM judge │
 │ classify  │     │ per-batch │     │ accept/   │
 │ enrich    │     │ unit from │     │ reject/   │
 │ group     │     │ DD, not   │     │ revise    │
 └───────────┘     │ LLM       │     └─────┬─────┘
                   └───────────┘           │
                                           ▼
 ┌───────────┐     ┌───────────┐     ┌───────────┐
 │  PERSIST  │◀────│CONSOLIDATE│◀────│ VALIDATE  │
 │           │     │           │     │           │
 │ Neo4j     │     │ dedup     │     │ grammar   │
 │ conflict  │     │ conflicts │     │ round-trip│
 │ detection │     │ coverage  │     │ fields    │
 └───────────┘     └───────────┘     │ check     │
                                     └───────────┘
```

**Phase summary:**

| Phase | Module | What it does |
|-------|--------|--------------|
| EXTRACT | `workers.extract_worker` | Query graph for DD paths, classify, enrich with clusters, group into batches |
| COMPOSE | `workers.compose_worker` | LLM generates standard names per batch; unit injected from DD |
| REVIEW | `workers.review_worker` | Optional LLM review: accept/reject/revise each candidate |
| VALIDATE | `workers.validate_worker` | Grammar round-trip via `parse_standard_name()`, fields consistency |
| CONSOLIDATE | `workers.consolidate_worker` | Cross-batch dedup, conflict detection, coverage accounting |
| PERSIST | `workers.persist_worker` | Conflict-detecting Neo4j writes with coalesce semantics |

Orchestrator: `imas_codex/sn/pipeline.py` — wires workers into the generic
`run_discovery_engine()` with a DAG dependency graph.

## Naming Scope Classifier

**Module:** `imas_codex/sn/classifier.py`

Every DD path is classified into one of three scopes before entering the
pipeline. The classifier is **rule-based** — no LLM, no cost, deterministic.

```python
Scope = Literal["quantity", "metadata", "skip"]
```

| # | Rule | Condition | Result |
|---|------|-----------|--------|
| 1 | Structure `/data` with unit | Last segment is `data`, parent is STRUCTURE type, has unit | `metadata` |
| 2 | `/time` leaf | Last segment is `time` | `metadata` |
| 3 | Validity fields | Last segment in `{validity, validity_timed}` | `metadata` |
| 4 | Error fields | Path ends with `_error_upper`, `_error_lower`, `_error_index` | `skip` |
| 5 | String fields | `data_type == "STR_0D"` | `skip` |
| 6 | Index/flag integers | `INT_0D`, no unit, description matches index/flag pattern | `skip` |
| 7 | Unitless integers | `INT_0D`, no unit, passes rule 6 check | `quantity` |
| 8–9 | Physics leaf types | `data_type` in `{FLT_0D..6D, INT_1D..2D, CPX_0D..2D}` | `quantity` |
| 10 | Structure with unit | `STRUCTURE` or `STRUCT_ARRAY` with unit | `quantity` |
| 11 | Fallback | Everything else | `skip` |

Only `quantity` paths proceed to COMPOSE. `metadata` and `skip` are filtered
out during EXTRACT.

## Unit Safety Model

Units flow from the Data Dictionary to the graph — **never from LLM output**.

```
DD (HAS_UNIT relationship)
        │
        ▼
 EXTRACT phase reads unit from graph
        │
        ▼
 COMPOSE prompt marks unit as "read-only, authoritative"
        │
        ▼
 LLM output has no unit field
        │
        ▼
 workers.py injects DD unit into candidate dict
        │
        ▼
 PERSIST writes unit → (StandardName)-[:CANONICAL_UNITS]->(Unit)
```

The compose prompt (`compose_dd.md`) includes a **Unit Policy** section:

> The unit value shown for each path is authoritative — it comes from the IMAS
> Data Dictionary. Do NOT include a `unit` field in your output.

If the LLM emits a unit anyway, the worker discards it and uses the DD value.

## DD Enrichment

**Module:** `imas_codex/sn/enrichment.py`

### Primary Cluster Selection

Each DD path may belong to multiple semantic clusters (IDS-scope, domain-scope,
global-scope). The enrichment layer selects **one primary cluster** per path
using a deterministic resolution order:

1. **IDS-scope** clusters (most specific) — scope rank 0
2. **Domain-scope** clusters — scope rank 1
3. **Global-scope** clusters (least specific) — scope rank 2
4. Within same scope: highest `similarity_score`, then lexicographic tie-break

```python
def select_primary_cluster(clusters: list[dict]) -> dict | None
```

### Global Grouping

Paths are grouped by **(primary_cluster × unit)** for LLM batching:

```python
def group_by_concept_and_unit(
    items: list[dict],
    max_batch_size: int = 25,
    existing_names: set[str] | None = None,
) -> list[ExtractionBatch]
```

**Key design choice:** grouping is **global** across all IDSs, not per-IDS.
If `electron_temperature` appears in `core_profiles`, `core_sources`, and
`equilibrium`, all three paths land in the same batch. This ensures the LLM
sees the full cross-IDS picture and assigns the same standard name.

Oversized groups are split into chunks of `max_batch_size` (token budget
guard). Unclustered paths are sub-grouped by `parent_path` for coherent
context.

Each batch carries context: cluster description, cross-IDS path summary,
sibling paths within the cluster, and existing standard names for reuse.

## Cross-Batch Consolidation

**Module:** `imas_codex/sn/consolidation.py`

After VALIDATE, all candidates are consolidated in a single pass:

```python
def consolidate_candidates(
    candidates: list[dict],
    *,
    source_paths: set[str] | None = None,
    existing_registry: dict[str, dict] | None = None,
) -> ConsolidationResult
```

### Deduplication

When multiple batches produce the same standard name:
- Keep the entry with the **longest documentation**
- **Union** `imas_paths` and `tags` across duplicates
- Keep the **highest confidence** score

### Conflict Detection

Five conflict checks, applied in order:

| Check | Type | What it catches |
|-------|------|-----------------|
| 1 | `unit_mismatch` | Same name proposed with different units |
| 2 | `kind_mismatch` | Same name with different kind (scalar vs vector) |
| 3 | `duplicate_source` | Same source path claimed by multiple names |
| 4 | Coverage gaps | Source paths with no candidate mapping |
| 5 | Registry reuse | Existing accepted names in graph → reuse instead of mint |

Conflicting entries are **filtered out**, not failed — the pipeline makes
partial progress rather than aborting.

### Result

```python
@dataclass
class ConsolidationResult:
    approved: list[dict]         # Ready for PERSIST
    conflicts: list[ConflictRecord]
    coverage_gaps: list[str]     # Source paths not mapped
    reused: list[dict]           # Existing names reused
    skipped_vocab_gaps: list[dict]
    stats: dict                  # Aggregate counts
```

## Prompt Architecture

### Static-First Ordering

The LLM pipeline uses a **two-message pattern** optimized for prompt caching:

1. **System message** (static, ~6k tokens) — rendered from `sn/compose_system`
   - Grammar rules, canonical segment order, exclusive pairs
   - Full vocabulary reference per segment
   - Curated examples, tokamak parameter ranges
   - 9 composition rules, output format schema
   - Cached by OpenRouter across calls (32% cache hit measured)

2. **User message** (dynamic, per-batch) — rendered from `sn/compose_dd`
   - Unit policy and anti-patterns
   - Batch context (IDS, cluster, siblings)
   - Existing standard names for reuse
   - Per-path detail: description, documentation, unit, type, coordinates

**Prompt files:**
- `imas_codex/llm/prompts/sn/compose_system.md` — static system context
- `imas_codex/llm/prompts/sn/compose_dd.md` — dynamic user prompt template

### Context Assembly

`imas_codex/sn/context.py` builds the compose context:

```python
def build_compose_context() -> dict[str, Any]
```

Assembles grammar rules, vocabulary sections, segment descriptions, curated
examples, and tokamak parameter ranges. Module-level cache avoids redundant
computation across batches.

### Response Model

```python
class SNComposeBatch(BaseModel):
    candidates: list[SNCandidate]
    skipped: list[str]
    vocab_gaps: list[SNVocabGap]
```

`vocab_gaps` captures cases where the grammar vocabulary lacks a needed token
(e.g. a new particle species). These are logged and tracked for vocabulary
extension.

## Graph Persistence

**Module:** `imas_codex/sn/graph_ops.py`

### Conflict-Detecting Writes

Before writing to Neo4j, `write_standard_names()` checks for unit conflicts:

```python
# If existing node has different canonical_units → skip, don't overwrite
if existing sn.canonical_units != incoming unit:
    logger.warning("Unit conflict for %s: existing=%s incoming=%s",
                   name, existing_unit, incoming_unit)
    # Filter out conflicting entry
```

### Coalesce Semantics

Node creation uses `coalesce()` — new values fill in blanks but never
overwrite existing data:

```cypher
MERGE (sn:StandardName {id: b.id})
SET sn.source_type    = coalesce(b.source_type, sn.source_type),
    sn.physical_base  = coalesce(b.physical_base, sn.physical_base),
    sn.canonical_units = coalesce(b.unit, sn.canonical_units),
    sn.created_at     = coalesce(sn.created_at, datetime())
```

This is safe for re-runs: imported catalog data is never clobbered by a
subsequent `sn mint` execution.

### Provenance Fields

Each StandardName node carries a full audit trail:

| Field | Source | Purpose |
|-------|--------|---------|
| `model` | Compose worker | LLM model used for generation |
| `confidence` | LLM output | 0–1 confidence score |
| `generated_at` | Compose worker | Timestamp of LLM generation |
| `reviewer_model` | Review worker | Model used for review |
| `reviewer_score` | Review worker | 0–100 quality score |
| `reviewer_scores` | Review worker | JSON: grammar, semantic, docs, convention, completeness |
| `reviewer_comments` | Review worker | Reasoning text |
| `reviewed_at` | Review worker | Review timestamp |
| `review_tier` | Review worker | outstanding/good/adequate/poor |
| `vocab_gap_detail` | Compose worker | JSON: segment, needed_token, reason |
| `catalog_commit_sha` | Import | Git SHA of catalog source |

### Relationships

```
(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)
(FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)
(StandardName)-[:CANONICAL_UNITS]->(Unit)
```

## Standard Name Lifecycle

```
drafted → published → accepted
                    ↘ rejected
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `drafted` | `sn mint` | Generated by LLM pipeline |
| `published` | `sn publish` | Exported to YAML catalog for human review |
| `accepted` | `sn import` | Imported from reviewed catalog (authoritative) |
| `rejected` | `sn import` | Reviewed and rejected in catalog |

Additional transient statuses: `reviewed`, `validation_failed`, `vocab_gap`,
`skipped` — used internally during pipeline execution.

**Safety model:** `sn reset` and `sn clear` require `--include-accepted` to
touch accepted names. Accepted names are catalog-authoritative and rarely need
graph modification.

## File Map

| File | Purpose |
|------|---------|
| `imas_codex/sn/pipeline.py` | Six-phase orchestrator (DAG wiring) |
| `imas_codex/sn/workers.py` | Async worker functions for each phase |
| `imas_codex/sn/classifier.py` | 11-rule path classifier (quantity/metadata/skip) |
| `imas_codex/sn/enrichment.py` | Primary cluster selection + global grouping |
| `imas_codex/sn/consolidation.py` | Cross-batch dedup, conflict detection, coverage |
| `imas_codex/sn/graph_ops.py` | Neo4j read/write helpers with conflict detection |
| `imas_codex/sn/models.py` | Pydantic response models (`SNComposeBatch`, `SNReviewBatch`) |
| `imas_codex/sn/context.py` | Grammar context builder (vocabulary, examples, ranges) |
| `imas_codex/sn/state.py` | `SNBuildState` dataclass (shared pipeline state) |
| `imas_codex/sn/sources/` | Extraction source plugins (DD and signals) |
| `imas_codex/sn/publish.py` | Export to YAML catalog |
| `imas_codex/sn/catalog_import.py` | Import reviewed catalog entries |
| `imas_codex/sn/benchmark.py` | LLM model quality benchmarking |
| `imas_codex/llm/prompts/sn/compose_system.md` | Static system prompt |
| `imas_codex/llm/prompts/sn/compose_dd.md` | Dynamic user prompt template |
| `imas_codex/schemas/standard_name.yaml` | LinkML schema (v0.5.0) |
