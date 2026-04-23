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
 │  EXTRACT  │────▶│  COMPOSE  │────▶│  VALIDATE │
 │           │     │           │     │           │
 │ DD query  │     │ LLM call  │     │ ISN 3-    │
 │ classify  │     │ per-batch │     │ layer +   │
 │ enrich    │     │ unit from │     │ grammar   │
 │ group     │     │ DD, not   │     │ round-trip│
 └───────────┘     │ LLM       │     └─────┬─────┘
                   └───────────┘           │
                                           ▼
                                    ┌───────────┐     ┌───────────┐
                                    │CONSOLIDATE│────▶│  PERSIST  │
                                    │           │     │           │
                                    │ dedup     │     │ Neo4j     │
                                    │ conflicts │     │ conflict  │
                                    │ coverage  │     │ detection │
                                    └───────────┘     └───────────┘
```

**RD-quorum review** runs as a standalone `sn review` command (not part of
`sn generate`). Each axis has an independent review pipeline:

```
compose → validate → review_names (RD-quorum) → review_docs (RD-quorum)

  review_names:
    cycle 0: primary   (BLIND)  ──┐
    cycle 1: secondary (BLIND)  ──┼─ disagree on any dim? → cycle 2
    cycle 2: escalator (SEES 0+1) ┘  (only if len(models)==3)

  review_docs:  same structure, independent rubric
```

**Phase summary:**

| Phase | Module | What it does |
|-------|--------|--------------|
| EXTRACT | `workers.extract_worker` | Query graph for DD paths, classify, enrich with clusters, group into batches |
| COMPOSE | `workers.compose_worker` | LLM generates standard names per batch; unit injected from DD |
| VALIDATE | `workers.validate_worker` | ISN 3-layer validation + grammar round-trip via `parse_standard_name()`, fields consistency |
| CONSOLIDATE | `workers.consolidate_worker` | Cross-batch dedup, conflict detection, coverage accounting |
| PERSIST | `workers.persist_worker` | Conflict-detecting Neo4j writes with coalesce semantics |

Orchestrator: `imas_codex/standard_names/pipeline.py` — wires workers into the generic
`run_discovery_engine()` with a DAG dependency graph.

## ISN Integration

The architecture boundary between imas-codex and imas-standard-names (ISN) is
documented in detail at [`docs/architecture/boundary.md`](boundary.md).

**Single import boundary:** `imas_codex/standard_names/context.py` calls
`get_grammar_context()` from ISN v0.7.0rc3 as its single entry point for all
grammar data (19 context keys: vocabulary, patterns, naming guidance, etc.).
No private ISN modules are imported.

**Three-layer validation** (`_validate_via_isn()` in `workers.py`):

1. **Pydantic** — `create_standard_name_entry()` fires 18 field validators
2. **Semantic** — `run_semantic_checks()` runs 9 grammar-semantic checks
3. **Description** — `validate_description()` checks for metadata leakage

Validation is **annotation-only**: entries are never rejected by ISN validation.
Issues are persisted to the graph as `validation_issues` (list of tagged
strings) and `validation_layer_summary` (JSON with per-layer counts). The only
hard rejection is an unparseable name (grammar round-trip failure).

## Scoring

Review scoring uses 6 dimensions × 0–20 integer scores, normalized to a 0–1
aggregate via `sum / 120.0`. The LLM scores in integers to avoid float
clustering (LLMs distribute poorly across continuous ranges). The graph stores
the normalized 0–1 float as `reviewer_score`.

**6 dimensions:** grammar, semantic, documentation, convention, completeness,
compliance. Defined in `imas_codex/llm/config/sn_review_criteria.yaml`.

**Tier thresholds:**

| Tier | Minimum Score |
|------|---------------|
| outstanding | ≥0.85 |
| good | ≥0.60 |
| inadequate | ≥0.40 |
| poor | <0.40 |

**Verdict rules:** accept (≥0.60, no zero dimensions), reject (<0.40 or any
zero dimension), revise (otherwise).

## Review Pipeline

`sn review` runs as a **standalone CLI command** (not part of `sn generate`).
It scores all `valid` (not quarantined) drafted names using an **RD-quorum** flow.
The two review axes (`names` and `docs`) are independent; each has its own rubric,
model chain, and per-axis score columns on the `StandardName` node.

### RD-Quorum Architecture

```
 ┌─────────────┐     ┌─────────────────────────────────────────┐
 │  FETCH      │────▶│  QUORUM                                  │
 │             │     │                                          │
 │ Query graph │     │  cycle 0: primary (BLIND)                │
 │ valid names │     │  cycle 1: secondary (BLIND)              │
 │ (no score)  │     │  disagree on any dim? ──► cycle 2        │
 └─────────────┘     │  cycle 2: escalator (sees 0+1 critiques) │
                     └──────────────────┬──────────────────────┘
                                        │
                                        ▼
                     ┌─────────────┐     ┌─────────────┐
                     │  AGGREGATE  │────▶│  PERSIST    │
                     │             │     │             │
                     │ winning-    │     │ Review nodes│
                     │ group select│     │ + axis slots│
                     └─────────────┘     └─────────────┘
```

**Cycle semantics:**

| Cycle | Role | Sees prior reviews? | When runs |
|-------|------|---------------------|-----------|
| 0 | primary | No (BLIND) | Always |
| 1 | secondary | No (BLIND) | When `len(models) ≥ 2` |
| 2 | escalator | Yes — both cycle 0+1 critiques | Only if disputed items exist AND `len(models) == 3` |

**Disputed items:** an item is disputed if any dimension score differs by >
`disagreement-threshold` (default 0.15, normalised 0–1) between cycles 0 and 1.
Cycle 2 receives only the disputed items as a per-item mini-batch.

**Immediate persistence:** each cycle's `Review` nodes are written to the graph
before the next cycle starts (crash-safety).

**`resolution_method` enum:**

| Value | Meaning |
|-------|---------|
| `quorum_consensus` | Cycles 0+1 agreed within tolerance; final = mean |
| `authoritative_escalation` | Cycle 2 ran; its result is authoritative for disputed items |
| `max_cycles_reached` | Cycles 0+1 disagreed; no escalator configured; final = mean with disagreement flag |
| `retry_item` | Both cycles failed; item quarantined |
| `single_review` | Only one cycle produced a score |

**Winning-group selection:** `update_review_aggregates` picks the most-recent
Review group with `resolution_method` ∈ {`quorum_consensus`,
`authoritative_escalation`, `single_review`} and mirrors scores onto the
StandardName axis slots.

**Partial-failure ladder:** both missing → retry; second failure → `retry_item`;
one missing → `single_review`.

### 1:1 Scoring Invariant

Each submitted name must receive exactly one score per cycle. The pipeline enforces
this invariant within each cycle:

1. Names are sent to the reviewer in batches (typically 10–20 per call)
2. After each batch, the response is matched back to submitted names by ID
3. Any names not returned by the LLM are collected as **unmatched**
4. Unmatched names are automatically retried individually (single-item batches)

This ensures no name is silently skipped due to LLM truncation or output format errors.

### Cost Reference

Approximate cost using `anthropic/claude-opus-4.6` as primary reviewer (2-model chain,
no escalator):

| Scope | Names | Cost |
|-------|-------|------|
| 4 IDSs (equilibrium, core_profiles, edge_profiles, summary) | ~300 | ~$60 (2× with secondary) |
| Full DD (~3,000 paths → ~1,200 names) | ~1,200 | ~$792 projected |

With a 3-model quorum and typical ~15% dispute rate, the escalator adds ~15% to the
two-cycle cost. Use `-c/--cost-limit` to cap spending per run.

## A/B Comparison (Plan 21 Outcomes)

Plan 21 closed two categories of gaps relative to ISN's built-in capabilities:

**ISN validation leads — closed.** The `_validate_via_isn()` three-layer
validation now runs 27+ checks via Pydantic model construction, semantic
analysis, and description quality checking. All issues are persisted to the
graph for reviewer context.

**Context gaps — closed.** Collision avoidance via vector search
(`imas_codex/standard_names/search.py`) finds similar existing StandardName nodes before
compose and review. ISN grammar context keys (`quick_start`, `common_patterns`,
`critical_distinctions`) are rendered in the compose prompt for richer LLM
guidance.

**Remaining ISN lead:** Iterative retry loop (ISN supports validate → fix →
revalidate cycles). Assessed as lower priority — the validate-once +
persistent-issues architecture catches issues upfront and surfaces them to
the reviewer rather than attempting automated repair.

## Naming Scope Classifier

**Module:** `imas_codex/standard_names/classifier.py`

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
| 11a | Structural keywords | Description contains structural keywords (e.g. "array of structure", "index of", "identifier of") | `skip` |
| 11b | Fit diagnostics | Path matches fit-diagnostic segment patterns (e.g. `chi_squared`, `fit_quality`, `convergence`) | `skip` |
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
 PERSIST writes unit → (StandardName)-[:HAS_UNIT]->(Unit)
```

The compose prompt (`compose_dd.md`) includes a **Unit Policy** section:

> The unit value shown for each path is authoritative — it comes from the IMAS
> Data Dictionary. Do NOT include a `unit` field in your output.

If the LLM emits a unit anyway, the worker discards it and uses the DD value.

## DD Enrichment

**Module:** `imas_codex/standard_names/enrichment.py`

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

**Module:** `imas_codex/standard_names/consolidation.py`

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

`imas_codex/standard_names/context.py` builds the compose context:

```python
def build_compose_context() -> dict[str, Any]
```

Assembles grammar rules, vocabulary sections, segment descriptions, curated
examples, and tokamak parameter ranges. Module-level cache avoids redundant
computation across batches.

### Per-Item DD Context Injection

Beyond the static grammar context, each DD path item is enriched with four
channels of Data Dictionary context during the extract phase:

1. **Hybrid DD search neighbours** (`hybrid_neighbours`) — concept-similar DD
   paths found via vector similarity + keyword search. The `hybrid_dd_search`
   function in `imas_codex/graph/dd_search.py` runs description-based and
   path-based queries in parallel, deduplicates results, and pre-resolves
   existing standard names for each neighbour.

2. **Related DD paths** (`related_neighbours`) — cross-IDS structural siblings
   discovered via explicit graph relationships: cluster membership, shared
   coordinates, matching units, identifier schemas, and COCOS transformation
   type. Uses `find_related_dd_paths` from `dd_search.py`.

3. **Error companions** (`error_fields`) — uncertainty/error field paths
   (`_error_upper`, `_error_lower`, `_error_index`) associated with each DD
   path. Gives the composer awareness of the quantity's measurement context.

4. **Identifier enum values** (`identifier_values`) — when a DD path references
   an identifier schema, the allowed enumeration values (name, index,
   description) are injected so the composer can use them for disambiguation.

**Compose retry loop:** On grammar/validation failure, the compose worker
retries up to `retry_attempts` times (configurable via `[tool.imas-codex.sn]`),
re-enriching items with expanded hybrid search (`search_k=retry_k_expansion`)
before resubmission.

**Scored-example injection:** Both compose and review prompts include
dynamically selected exemplar StandardName nodes at target score thresholds
`(1.0, 0.8, 0.65, 0.4)`. Examples are graph-backed and selected at runtime
by the example loader. Context keys: `compose_scored_examples`,
`review_scored_examples`.

### Axis Reviewer Output

The review worker persists per-dimension data per axis. Each axis has five paired
slots on `StandardName`:

| Axis | Scalar | Per-dim JSON | Comments | Verdict | Model |
|------|--------|--------------|----------|---------|-------|
| names | `reviewer_score_name` | `reviewer_scores_name` | `reviewer_comments_name` | `reviewer_verdict_name` | `reviewer_model_name` |
| docs | `reviewer_score_docs` | `reviewer_scores_docs` | `reviewer_comments_docs` | `reviewer_verdict_docs` | `reviewer_model_docs` |

Scores are written by the winning Review group's `update_review_aggregates` call (after
all quorum cycles complete).

**Review node fields (RD-quorum, p39-4):**
- `review_axis` — `names` or `docs`
- `cycle_index` — 0 (primary), 1 (secondary), 2 (escalator)
- `review_group_id` — UUID per quorum session (links cycles within a review run)
- `resolution_role` — `primary`, `secondary`, or `escalator`
- `resolution_method` — outcome enum (see table in Review Pipeline section)

### Response Model

```python
class StandardNameComposeBatch(BaseModel):
    candidates: list[StandardNameCandidate]
    skipped: list[str]
    vocab_gaps: list[StandardNameVocabGap]
```

`vocab_gaps` captures cases where the grammar vocabulary lacks a needed token
(e.g. a new particle species). These are logged and tracked for vocabulary
extension.

### Open vs Closed Segments

The ISN grammar distinguishes **closed-vocabulary** segments (fixed token
lists — `transformation`, `subject`, `position`, `component`, `coordinate`,
`object`, `geometry`, `process`, `device`, `geometric_base`, `region`) from
**open-vocabulary** segments (free-form compounds — `physical_base`). The LLM
composer may incidentally report a "missing token" against the open
`physical_base` segment when it packs a compound (e.g.
`electron_temperature`, `parallel_component_of_ion_momentum_diffusivity`)
into that slot — but by design *any* snake_case compound is admissible there.

Such reports are **not real vocabulary gaps**: they would pollute the
`VocabGap` node population and drown out genuine closed-segment gaps during
release filtering.  The single source of truth for segment openness lives in
`imas_codex/standard_names/segments.py`:

- `open_segments()` — derived from `imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP` (any segment with an empty token list).
- `PSEUDO_SEGMENTS` — `{"grammar_ambiguity"}`, a composer-reported structural finding that is not a real segment.
- `is_open_segment(seg)` — unified predicate used by `write_vocab_gaps()` and `_update_sources_after_vocab_gap()` to drop non-gaps before persistence.

`sn gaps` hides open/pseudo segments by default; pass
`--include-open-segments` for diagnostics.

## Graph Persistence

**Module:** `imas_codex/standard_names/graph_ops.py`

### Conflict-Detecting Writes

Before writing to Neo4j, `write_standard_names()` checks for unit conflicts:

```python
# If existing node has different unit → skip, don't overwrite
if existing sn.unit != incoming unit:
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
    sn.unit = coalesce(b.unit, sn.unit),
    sn.created_at     = coalesce(sn.created_at, datetime())
```

This is safe for re-runs: imported catalog data is never clobbered by a
subsequent `sn generate` execution.

### Axis-Split Review Storage (rc22 C3 + p39-2)

Name-axis and docs-axis reviews persist to independent column families to
prevent a docs-only review from clobbering a prior name-only review's
per-dimension JSON (and vice-versa). The shared scalar+JSON slots that existed
prior to p39-2 (`reviewer_score`, `reviewer_scores`, `reviewer_comments`,
`reviewer_comments_per_dim`, `reviewer_verdict`, `reviewer_model`, `reviewed_at`,
`review_mode`) have been **removed from the schema**. Only axis-specific columns remain:

```cypher
// --target names (name-axis review)
SET sn.reviewer_score_name = b.reviewer_score_name,
    sn.reviewer_scores_name = coalesce(b.reviewer_scores_name, sn.reviewer_scores_name),
    sn.reviewer_comments_name = coalesce(b.reviewer_comments_name, sn.reviewer_comments_name),
    sn.reviewer_verdict_name = b.reviewer_verdict_name,
    sn.reviewer_model_name = b.reviewer_model_name

// --target docs (docs-axis review) — same pattern with _docs suffix
```

**Reader fallback.** Queries use `coalesce(sn.reviewer_score_name, sn.reviewer_score)`
so pre-p39-2 nodes with old shared slots still surface correctly on the name axis until
backfilled.

**Downgrade guard.**
`imas_codex.standard_names.review.pipeline._axis_overwrite_blocked(name, incoming)`
rejects axis writes when same-axis data is already present. `--force` bypasses the guard.

### Provenance Fields

Each StandardName node carries a full audit trail:

| Field | Source | Purpose |
|-------|--------|---------|
| `model` | Compose worker | LLM model used for generation |
| `confidence` | LLM output | 0–1 confidence score |
| `generated_at` | Compose worker | Timestamp of LLM generation |
| `reviewer_score_name` | Review worker | 0–1 name-axis quality score |
| `reviewer_scores_name` | Review worker | JSON: grammar, semantic, convention, completeness (each 0–20) |
| `reviewer_comments_name` | Review worker | Name-axis reasoning text |
| `reviewer_verdict_name` | Review worker | accept/reject/revise — name-axis verdict |
| `reviewer_model_name` | Review worker | Winning-cycle model (name axis) |
| `reviewed_name_at` | Review worker | Timestamp of most recent name-axis review |
| `reviewer_score_docs` | Review worker | 0–1 docs-axis quality score |
| `reviewer_scores_docs` | Review worker | JSON: description_quality, documentation_quality, completeness, physics_accuracy (each 0–20) |
| `reviewer_comments_docs` | Review worker | Docs-axis reasoning text |
| `reviewer_verdict_docs` | Review worker | accept/reject/revise — docs-axis verdict |
| `reviewer_model_docs` | Review worker | Winning-cycle model (docs axis) |
| `reviewed_docs_at` | Review worker | Timestamp of most recent docs-axis review |
| `review_tier` | Review worker | outstanding/good/inadequate/poor (from `reviewer_score_name`) |
| `review_count` | Review worker | Number of Review nodes attached via REVIEWS |
| `review_mean_score` | Review worker | Arithmetic mean of Review.score across all cycles |
| `review_disagreement` | Review worker | True iff quorum cycles disagreed beyond threshold |
| `vocab_gap_detail` | Compose worker | JSON: segment, needed_token, reason |
| `catalog_commit_sha` | Import | Git SHA of catalog source |

### Relationships

```
(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)
(FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)
(StandardName)-[:HAS_UNIT]->(Unit)
```

## Standard Name Lifecycle

```
drafted → published → accepted
                    ↘ rejected
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `drafted` | `sn generate` | Generated by LLM pipeline |
| `published` | `sn publish` | Exported to YAML catalog for human review |
| `accepted` | `sn import` | Imported from reviewed catalog (authoritative) |
| `rejected` | `sn import` | Reviewed and rejected in catalog |

Additional transient statuses: `reviewed`, `validation_failed`, `vocab_gap`,
`skipped` — used internally during pipeline execution.

**Safety model:** `sn generate --reset-to` and `sn clear` require `--include-accepted` to
touch accepted names. Accepted names are catalog-authoritative and rarely need
graph modification.

## Validation Gating

A separate `validation_status` field (independent of `review_status`) gates
names before they reach review, consolidation, or publish:

```
pending → valid | quarantined
```

| Status | Set by | Meaning |
|--------|--------|---------|
| `pending` | `sn generate` (PERSIST phase) | Default; awaiting validation |
| `valid` | VALIDATE phase | Passed all critical checks; eligible for downstream stages |
| `quarantined` | VALIDATE phase | Failed a critical check; excluded from review/consolidation/publish |

**Critical failures (→ quarantine):**
- Grammar round-trip failure (`parse_standard_name()` rejects the name)
- Pydantic construction error (`create_standard_name_entry()` raises)
- Detected ambiguity (name maps to multiple distinct physical quantities)

**Non-critical issues (→ valid):** semantic warnings, description quality hints —
recorded in `validation_issues` and surfaced to the reviewer, but do not quarantine.

Only `valid` names participate in `sn review`, consolidation, and `sn publish`.

## File Map

| File | Purpose |
|------|---------|
| `imas_codex/standard_names/pipeline.py` | Six-phase orchestrator (DAG wiring) |
| `imas_codex/standard_names/workers.py` | Async worker functions for each phase |
| `imas_codex/standard_names/classifier.py` | 11-rule path classifier (quantity/metadata/skip) |
| `imas_codex/standard_names/enrichment.py` | Primary cluster selection + global grouping |
| `imas_codex/standard_names/consolidation.py` | Cross-batch dedup, conflict detection, coverage |
| `imas_codex/standard_names/graph_ops.py` | Neo4j read/write helpers with conflict detection |
| `imas_codex/standard_names/models.py` | Pydantic response models (`StandardNameComposeBatch`, `StandardNameReviewBatch`) |
| `imas_codex/standard_names/context.py` | Grammar context builder (vocabulary, examples, ranges) |
| `imas_codex/standard_names/state.py` | `StandardNameBuildState` dataclass (shared pipeline state) |
| `imas_codex/standard_names/sources/` | Extraction source plugins (DD and signals) |
| `imas_codex/standard_names/publish.py` | Export to YAML catalog |
| `imas_codex/standard_names/catalog_import.py` | Import reviewed catalog entries |
| `imas_codex/standard_names/benchmark.py` | LLM model quality benchmarking |
| `imas_codex/llm/prompts/sn/compose_system.md` | Static system prompt |
| `imas_codex/llm/prompts/sn/compose_dd.md` | Dynamic user prompt template |
| `imas_codex/llm/prompts/sn/review_names.md` | Names-axis review prompt (4-dim: grammar/semantic/convention/completeness) |
| `imas_codex/llm/prompts/sn/review_docs.md` | Docs-axis review prompt (4-dim: description_quality/documentation_quality/completeness/physics_accuracy) |
| `imas_codex/llm/prompts/shared/sn/_grammar_reference.md` | Shared grammar fragment (included by compose_system.md) |
| `imas_codex/llm/prompts/shared/sn/_scoring_rubric.md` | Shared scoring rubric reference |
| `imas_codex/llm/config/sn_review_criteria.yaml` | Review scoring config (dimensions, tiers, verdict rules) |
| `imas_codex/standard_names/search.py` | Vector search for similar existing StandardName nodes |
| `imas_codex/schemas/standard_name.yaml` | LinkML schema (v0.5.0) |
| `imas_codex/graph/dd_search.py` | Pure functions: `hybrid_dd_search`, `find_related_dd_paths` |
| `imas_codex/standard_names/example_loader.py` | Graph-backed scored example selection for prompts |

## Migration from Pre-p39-2 Graphs

p39-2 removed the shared reviewer slots (`reviewer_score`, `reviewer_scores`,
`reviewer_comments`, `reviewer_comments_per_dim`, `reviewer_verdict`, `reviewer_model`,
`reviewed_at`, `review_mode`) from the `StandardName` schema. These fields are no longer
written but **will not cause runtime errors** if present on existing nodes — they become
stale orphan properties until the node is re-reviewed.

Reader queries use `coalesce(sn.reviewer_score_name, sn.reviewer_score)` for backward
compatibility so pre-migration nodes surface correctly on the name axis.

**Backfill procedure:** Run `sn review --force` to re-review all valid names and populate
the axis-specific columns.

The `review_mode` enum has two values: `names` and `docs`. The removed `full` variant
no longer exists — use `--target names` + `--target docs` in sequence for full coverage.

The tier rename (`adequate` → `inadequate`) applies to new reviews; stale
`review_tier = 'adequate'` values persist harmlessly until re-reviewed.
