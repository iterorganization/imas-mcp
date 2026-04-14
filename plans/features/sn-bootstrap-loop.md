# Standard Names: Batched Review + Bootstrap Loop

> Priority: P1 — Wire `sn review` as a scalable batched pipeline mirroring
> generate architecture. Enable bootstrap to 10,000+ names.

## Problem

The catalog will grow to ~10,000 standard names across all IDSs. The original
plan assumed "full catalog oversight" — every review batch sees all names:

- **Token budget**: 10k names × ~200 tokens = 2M tokens per batch (impossible)
- **Cost**: full-catalog context in every batch multiplies cost by batch count
- **Signal/noise**: reviewing `electron_temperature` doesn't need to see
  `wall_erosion_rate` — it needs `ion_temperature`, `electron_density`

The generate pipeline already solves this with **(cluster × unit) batching** +
**semantic neighborhood search**. Review should mirror this architecture, adding
**deterministic audits** for consistency checks that probabilistic vector search
cannot guarantee.

## Key Insight

At generate time, `persist_worker` already embeds every StandardName description
into the `standard_name_desc_embedding` vector index. This means:

1. Every name is searchable by semantic similarity
2. For any review batch, we can find the K most similar names from the entire
   catalog — cross-catalog visibility without sending the full catalog
3. **Targeted neighborhood context** replaces full catalog context

## Verified Code Facts

- **62 names in graph**: 21 accepted (seed/import), 41 drafted (LLM generate)
- **review_worker exists**: `workers.py` — processes batches via `SNQualityReviewBatch`
- **6-dimension scoring**: grammar, semantic, documentation, convention,
  completeness, compliance (each 0-20, normalized to 0-1 via sum/120)
- **Scoring criteria**: `sn_review_criteria.yaml`
- **Calibration data**: `benchmark_calibration.yaml` — 27 curated reference entries
- **Persistence stores**: `reviewer_score`, `reviewer_scores` (JSON),
  `reviewer_comments`, `reviewed_at`, `review_tier`, `reviewer_model`
- **Write semantics**: `write_standard_names()` uses coalesce — safe to re-run
- **Embedding at generate time**: `persist_worker` calls `embed_descriptions_batch()`
- **Vector search**: `search_similar_names()` queries `standard_name_desc_embedding`
- **Cluster grouping**: `enrichment.py` groups by `(primary_cluster × unit)`

## Architecture: Three-Layer Review

```
Layer 1: DETERMINISTIC AUDITS (no LLM, full catalog, runs first)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Embedding   │  │   Lexical    │  │    Link      │  │  Duplicate   │
  │  Preflight   │  │   Pattern    │  │  Integrity   │  │  Candidate   │
  │              │  │   Linting    │  │              │  │  Generation  │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └──────────────────┴──────────────────┴──────────────────┘
                                    │
                          audit findings per name
                                    │
                                    ▼
Layer 2: BATCHED LLM REVIEW (scoped targets, neighborhood context)
  ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
  │  EXTRACT  │────▶│  ENRICH   │────▶│  REVIEW   │────▶│  PERSIST  │
  │           │     │           │     │           │     │           │
  │ cluster×  │     │ semantic  │     │ 6-dim LLM │     │ coalesce  │
  │ unit grp  │     │ neighbor  │     │ scoring + │     │ write to  │
  │           │     │ + audits  │     │ batch     │     │ graph     │
  └───────────┘     └───────────┘     │ consist.  │     └───────────┘
                                      └───────────┘
                                    │
                              review scores
                                    │
                                    ▼
Layer 3: CROSS-BATCH CONSOLIDATION (post-review, mostly deterministic)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Duplicate   │  │  Convention  │  │    Score     │  │   Summary    │
  │  Resolution  │  │    Drift     │  │   Outlier    │  │   Report     │
  │              │  │  Detection   │  │  Detection   │  │              │
  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**Layer boundaries** (strict ownership):
- **Layer 1**: candidate generation + deterministic findings only (full catalog)
- **Layer 2**: per-name scoring + batch-level judgment (scoped targets)
- **Layer 3**: resolve/report cross-batch findings (never recomputes Layer 1)

## Layer 1: Deterministic Audit Phase

Runs before LLM review. No LLM cost. Produces findings that feed into review
prompts. **Always operates on full catalog**, even when Layer 2 targets a
subset via `--ids`/`--domain` — otherwise incremental review misses conflicts
with accepted names elsewhere.

### 1A: Embedding Preflight

Ensures neighborhood search works correctly for all names:

- Query all StandardNames (full catalog, not just review targets)
- Identify **missing** embeddings: `embedding IS NULL`
- Identify **stale** embeddings: compute `review_input_hash` over review-relevant
  fields (`id`, `description`, `documentation`, `kind`, `unit`, `tags`, `links`,
  grammar fields, `cocos_transformation_type`, `imas_paths`) — stale when
  `review_input_hash != stored_hash`
- Re-embed via `embed_descriptions_batch()`
- Persist updated `review_input_hash` and `embedded_at`

**Why hash, not timestamps**: timestamp-only staleness (`embedded_at < generated_at`)
misses changes to tags, links, or grammar fields that don't update generation
timestamps. A content fingerprint over review-relevant fields is robust.

### 1B: Lexical Pattern Linting

Deterministic checks against grammar rules (ISN `get_grammar_context()`):

- Grammar round-trip failures: `parse(name) → compose() ≠ name`
- Segment vocabulary gaps: tokens not in ISN vocabulary
- Naming convention violations detected by pattern:
  - Mixed position forms within a domain (e.g., `_at_magnetic_axis` vs `_on_axis`)
  - Processing verbs that should be absent (`_reconstructed`, `_measured`)
- Overlaps with ISN validation but runs on **full catalog at once** for
  cross-name pattern detection

### 1C: Link Integrity

- Unresolved links: `link_status = 'unresolved'`
- Missing reverse links: A→B exists but B→A doesn't
- Dead links: target name doesn't exist in graph
- All deterministic, no LLM needed

### 1D: Duplicate Candidate Generation

Efficient O(n·K) approach using **multi-pass blocking** + vector search:

**Pass 1**: block by `(unit × kind × physics_domain)`
**Pass 2**: block by `(unit × kind × physical_base)` (when present)
**Pass 3**: block by `(unit × kind × geometric_base)` (when present)

Union candidate pairs across passes.

Within each block:
1. For each name, query K=5 nearest neighbors via `standard_name_desc_embedding`
2. Filter: description similarity > 0.92 AND different `id` → candidate pair
3. Also check: lexical normalized-name similarity (Levenshtein on snake_case tokens)
4. Component detection: union-find over candidate pairs → connected components
5. Output: `DuplicateComponent(names=[...], max_similarity=float)`

**Why multi-pass**: `(unit × kind)` alone creates huge blocks for dimensionless
scalars. Adding `physics_domain` or `physical_base` keeps blocks tractable while
union across passes avoids false negatives.

## Layer 2: Batched LLM Review Pipeline

Uses `run_discovery_engine()` — same DAG orchestrator as generate.

### EXTRACT

Load StandardName nodes from graph with filters:

| Flag | Effect |
|------|--------|
| `--ids IDS` | Names linked to IMASNode in that IDS |
| `--domain DOMAIN` | Names with matching `physics_domain` |
| `--status STATUS` | Filter by `review_status` (default: `drafted`) |
| `--unreviewed` | `reviewer_score IS NULL` OR stale (`review_input_hash` changed) |
| `--re-review` | Force re-scoring of already-reviewed names |

**Cluster reconstruction** (live from graph, never cached on StandardName):

For DD-sourced names:
```
(sn:StandardName)<-[:HAS_STANDARD_NAME]-(node:IMASNode) → cluster memberships
```
Compute **dominant cluster** using aggregated evidence:
1. Count linked sources per cluster (most sources wins)
2. Then scope priority (IDS > domain > global)
3. Then mean similarity score within cluster
4. Deterministic tie-break on cluster label

For signal-sourced/manual names:
- Group by `(physics_domain × unit)` with **lexical family sub-bucketing**:
  names sharing `physical_base` or `geometric_base` go together

Group by **(dominant_cluster × unit)** → batches.

**Token-budget batch sizing** (not fixed count):
- Estimate tokens per name: `len(description) + len(documentation)` chars ÷ 4
  (rough char-to-token ratio) + 80 tokens scaffolding per item
- Fixed overhead per batch: system prompt (~6k tokens) + scoring criteria +
  calibration examples + neighborhood context (~2k tokens) + completion
  reserve (30% of input)
- Fill each batch until token estimate reaches budget, hard cap at 25 names
- Default effective batch size: ~12-15 names (varies by documentation richness)

### ENRICH

For each review batch, build **semantic neighborhood**:

1. **Batch-level query**: embed cluster label + first 3 representative
   descriptions → search `standard_name_desc_embedding` for K=8 nearest names
   NOT in current batch
2. **Per-name fallback**: for unclustered names or names with < 3 batch-level
   neighbors, search K=3 per individual name → union + dedupe with batch results
3. **Attach Layer 1 audit findings**: duplicate candidates, lint issues, link
   problems — cap verbosity (one-line summary per finding, not full component
   dumps)
4. Neighborhood metadata (summary only): `id`, `description`, `kind`, `unit`,
   `review_tier`

### REVIEW

LLM scores each batch using existing 6-dimension rubric.

**Each batch receives:**
- **Batch members** (8-25 names): full detail — name, description,
  documentation, grammar fields, kind, unit, COCOS, validation_issues,
  audit findings from Layer 1
- **Neighborhood context** (~10 names): summary — id, description, kind,
  unit, review_tier
- **Scoring criteria** from `sn_review_criteria.yaml`

**Response model**: `SNQualityReviewBatch` (unchanged)

**LLM checks:**
- Individual quality (grammar, docs, conventions) — 6 dimensions
- Within-batch consistency (names in same cluster follow same patterns)
- Audit-informed checks (reviewer sees "flagged as potential duplicate of X"
  and can confirm/dismiss)

**Re-review context**: Blind to prior numeric scores (avoids anchoring bias).
May include prior `reviewer_comments` as secondary "delta" context for
tracking improvement across iterations.

**Budget manager** (centralized, concurrent-safe):
```
ReviewBudgetManager:
  - total_budget: float (from --cost-limit)
  - remaining: AtomicFloat
  - reserve(estimated_cost) → bool  # atomically deduct, reject if insufficient
  - reconcile(reserved, actual)     # return unused reservation
  - exhausted() → bool
```
- Estimate per-batch cost: `(input_tokens + completion_reserve) × model_price`
- Include retry headroom: reserve 1.3× estimated cost per batch
- If reservation fails: persist all completed reviews, report partial progress
- Concurrency default: 2 parallel batches with rate-limit backoff

### PERSIST

Write review scores via `write_standard_names()` (coalesce semantics).

Fields written: `reviewer_model`, `reviewer_score`, `reviewer_scores`,
`reviewer_comments`, `reviewed_at`, `review_tier`.

Update `review_input_hash` to mark review as current.

## Layer 3: Cross-Batch Consolidation

Runs after all LLM review batches complete. Mostly deterministic.

### 3A: Duplicate Resolution

- Take duplicate components from Layer 1D
- Cross-reference with LLM review comments (reviewer may have
  confirmed/dismissed)
- Unresolved components: report as actionable items with similarity scores
- Optional: LLM arbitration for genuinely ambiguous cases (e.g.,
  `electron_thermal_energy` vs `electron_temperature`)

### 3B: Convention Drift Detection

Group reviewed names by `physics_domain`:
- Mixed position suffixes (`_at_magnetic_axis` vs `_on_axis`)
- Inconsistent process suffixes (`_reconstructed` vs no suffix)
- Divergent documentation depth within same concept family
- Report as warnings, not auto-fixes

### 3C: Score Outlier Detection

- Score distribution per cluster/domain
- Names scoring > 1σ below their cluster mean → regeneration candidates
- Accepted-name anchors: compare drafted names against accepted names in
  same cluster for consistency

### 3D: Summary Report

Console output:
- Total reviewed, LLM cost, tier distribution
- Duplicate candidates found (count + top examples)
- Convention drift warnings
- Lowest scorers → regeneration candidates
- Coverage: % of catalog reviewed this pass

## Review Staleness Policy

Reviews become stale when review-relevant content changes:

- **Primary rule**: `review_input_hash` (computed over id, description,
  documentation, kind, unit, tags, links, grammar fields, cocos fields,
  imas_paths) differs from `stored_review_hash`
- **Fallback rule**: `reviewed_at < coalesce(generated_at, imported_at)`
  (for names without hash — transition period)
- Stale reviews are included in `--unreviewed` targeting

## CLI Design

```
sn review [OPTIONS]
  --ids IDS            Scope to names linked to specific IDS
  --domain DOMAIN      Scope to physics domain
  --status STATUS      Filter by review_status (default: drafted)
  --unreviewed         Only names with no reviewer_score (or stale review)
  --re-review          Force re-review of already-scored names
  --model MODEL        Override review model (default: get_model("language"))
  --batch-size INT     Max names per batch (default: 15, hard cap: 25)
  --neighborhood INT   Similar names for context (default: 10)
  --cost-limit FLOAT   Max LLM spend in USD
  --dry-run            Run Layer 1 audits, show batch plan, no LLM calls
  --skip-audit         Skip Layer 1 audits (debug only, not normal workflow)
  --concurrency INT    Parallel review batches (default: 2)
```

**Design decisions:**
1. **`--ids`/`--domain` scoping works** — narrows Layer 2 targets but Layer 1
   audits always run on full catalog; neighborhood context spans full catalog
   via vector search
2. **`--dry-run` runs audits** — useful for running deterministic checks
   without LLM cost (duplicate candidates, lint, link integrity)
3. **`--skip-audit` is debug-only** — the audit layer justifies the
   architecture; skipping it defeats the purpose
4. **Batch size is a cap** — actual size determined by token budget estimation

## Phase 1: Implementation

### 1a: Layer 1 — Deterministic Audits

New module: `imas_codex/standard_names/review/audits.py`
- `run_embedding_preflight(names) → EmbeddingReport`
- `run_lexical_lint(names) → list[LintFinding]`
- `run_link_integrity(names) → list[LinkFinding]`
- `run_duplicate_detection(names) → list[DuplicateComponent]`
- `run_all_audits(names) → AuditReport`

New schema field: `review_input_hash` on StandardName (add to
`standard_name.yaml`, rebuild models).

### 1b: Layer 2 — Batched LLM Review Pipeline

New modules:
- `review/enrichment.py` — cluster reconstruction + neighborhood search
- `review/pipeline.py` — DAG wiring for EXTRACT → ENRICH → REVIEW → PERSIST
- `review/budget.py` — `ReviewBudgetManager` with atomic reservation

Additions to existing:
- `graph_ops.py`: `get_review_candidates()`, `get_names_with_cluster_info()`
- `cli/sn.py`: `sn review` command
- Review prompt template: include audit findings + neighborhood in context

### 1c: Layer 3 — Cross-Batch Consolidation

New module: `imas_codex/standard_names/review/consolidation.py`
- `resolve_duplicates(components, review_results) → list[DuplicateReport]`
- `detect_convention_drift(reviewed_names) → list[DriftWarning]`
- `detect_score_outliers(reviewed_names) → list[OutlierReport]`
- `build_summary_report(...) → ReviewSummary`

### Implementation Reuse

| Existing Component | Used For |
|-------------------|----------|
| `select_primary_cluster()` from `enrichment.py` | Cluster grouping in EXTRACT |
| `search_similar_names()` from `search.py` | Neighborhood context in ENRICH |
| `embed_descriptions_batch()` from `workers.py` | Embedding preflight |
| `write_standard_names()` from `graph_ops.py` | Score persistence |
| `run_discovery_engine()` from `engine.py` | DAG orchestration |
| `SNQualityReviewBatch` from `models.py` | LLM response model |
| `sn_review_criteria.yaml` | Scoring dimensions and tiers |
| `get_grammar_context()` from ISN | Lexical lint rules |

### Tests

- `test_audit_embedding_preflight` — detect missing/stale embeddings via hash
- `test_audit_duplicate_candidates` — multi-pass blocking + vector similarity
- `test_audit_link_integrity` — unresolved/dead link detection
- `test_review_cluster_reconstruction` — dominant cluster via aggregated evidence
- `test_review_batch_token_budget` — batch sizing by token estimate, not count
- `test_review_neighborhood_enrichment` — vector search excludes batch members
- `test_review_budget_manager` — atomic reservation, partial-progress persist
- `test_review_consolidation_drift` — convention drift detection
- `test_review_staleness_policy` — hash-based invalidation
- `test_review_cli_dry_run` — audit-only mode without LLM calls

## Phase 2: Bootstrap Loop (Operational)

Scoped batches make this tractable at scale. Run after Phase 1 implemented.

### Wave 1: Audit + review existing (62 names)
```bash
uv run imas-codex sn review --unreviewed --cost-limit 5.0
uv run imas-codex sn status  # Check tier distribution
```

### Wave 2: Generate + review by IDS (incremental)
```bash
for ids in edge_profiles summary mhd core_transport; do
  uv run imas-codex sn generate --source dd --ids $ids --cost-limit 2.0
  uv run imas-codex sn review --ids $ids --unreviewed --cost-limit 2.0
done
```

### Wave 3: Regenerate poor-quality names
```bash
# Identify low-tier names from sn status, regenerate specific paths
uv run imas-codex sn generate --source dd --paths "equilibrium/..." --force
uv run imas-codex sn review --re-review --domain equilibrium --cost-limit 2.0
```

### Wave 4: Cross-domain consistency
```bash
# Full catalog audit + targeted re-review of flagged names
uv run imas-codex sn review --re-review --cost-limit 10.0
# Target: ≥1000 names, ≥80% good/outstanding tier
```

## Phase 3: Prompt Quality Improvements

After catalog reaches ≥500 reviewed names:

### 3A: Showcase examples in compose prompt
Top-scoring names from bootstrap as few-shot examples in `context.py`.
Replace abstract examples with real, high-quality generated names.

### 3B: Documentation depth gates
Soft content gates in `compose_system.md`:
- Physics quantities SHOULD have ≥1 display equation
- Variables in equations SHOULD be defined with units
- COCOS-dependent quantities MUST have sign convention paragraph

## Implementation Notes

Lock down during coding — none are architectural blockers:

1. **Hash update semantics**: `review_input_hash` must be **overwritten** (not
   coalesced) whenever review-relevant fields change. Watch existing coalesce-
   heavy write paths in `write_standard_names()` — don't accidentally preserve
   stale hashes.

2. **Metadata-poor name fallback**: names with null `physics_domain`,
   `physical_base`, and `geometric_base` won't appear in any blocking pass.
   Add a lexical/family fallback bucket for these names in duplicate detection.

3. **Batch vector queries**: O(n·K) individual Neo4j vector queries may be slow
   at 10k names. Batch or server-side where possible — don't do one RPC per
   name in the hot path.

4. **Budget reconciliation on failures**: reservations must be reconciled in
   `finally` blocks, including timeout/cancel/retry-exhaustion paths.

5. **Cross-unit near-duplicates**: blocking assumes true duplicates share unit.
   Add a separate low-priority audit for "same/near-same name, different unit"
   — this signals a serious data quality problem, not a naming duplicate.

## Explicitly Deferred

- **Full-catalog LLM context**: replaced by semantic neighborhood (scales to 10k+)
- **A/B holdout validation**: needs benchmark infrastructure
- **Hierarchical review**: review-of-reviews for cross-domain issues (future)
- **Automated regeneration loop**: review→regenerate cycle (manual in bootstrap)

## Phase Dependencies

```
Phase 1a (deterministic audits) → implement first
Phase 1b (batched LLM review)   → depends on 1a (audits feed into review)
Phase 1c (consolidation)        → depends on 1b (needs review scores)
Phase 2 (bootstrap loop)        → after Phase 1, operational
Phase 3A (showcase examples)    → after ≥500 reviewed names
Phase 3B (documentation gates)  → after Phase 3A
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Add `sn review` CLI with flags, batched architecture note |
| `docs/architecture/standard-names.md` | Add review pipeline section |
| `plans/README.md` | Update status |

## References

- `completed/23-quality-parity.md` — original quality plan
- `completed/25-standalone-review.md` — original review plan
- `standard-names/20-consistency-and-prompt-enrichment.md` — DD enrichment
