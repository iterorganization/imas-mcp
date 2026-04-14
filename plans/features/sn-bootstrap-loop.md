# Standard Names: Standalone Review CLI + Bootstrap Loop

> Priority: P1 — 62 names exist (21 accepted, 41 drafted). Wire standalone
> review to score and iterate quality upward across the full catalog.

## Problem

The `sn generate` pipeline is fully functional and has produced 62 standard
names. The review_worker exists inside the generate pipeline but there is no
**standalone** `sn review` CLI command. A standalone review tool is needed to:

1. Score all existing names (including 21 accepted seed names) for consistency
2. Detect cross-name issues: naming convention drift, link integrity, duplicate concepts
3. Iterate quality by feeding review scores back into regeneration decisions

The review task differs fundamentally from generation:
- **Generate** scopes by IDS/domain and groups by `(cluster × unit)` to ensure
  same-concept paths get composed together
- **Review** needs **full catalog oversight** — it must see all names to check
  cross-name consistency, naming conventions, and duplicate detection

## Verified Code Facts

- **62 names in graph**: 21 accepted (seed/import), 41 drafted (LLM generate)
- **LLM compose is schema-enforced**: `acall_llm_structured(..., response_model=SNComposeBatch)` in `workers.py:612-615`
- **Units are LLM-excluded + DD-injected**: prompt says "Do NOT output a `unit` field"; `workers.py:629-667` injects from DD
- **review_worker exists**: `workers.py:793+` — processes batches via `SNQualityReviewBatch`
- **review_worker uses `get_model("language")`** (not "reasoning")
- **Review batch size**: `_REVIEW_BATCH_SIZE = 10` (hardcoded `workers.py:752`)
- **6-dimension scoring**: grammar, semantic, documentation, convention, completeness, compliance (each 0-20, normalized to 0-1 via sum/120)
- **Scoring criteria**: `sn_review_criteria.yaml`
- **Calibration data**: `benchmark_calibration.yaml` — 27 curated reference entries
- **Persistence already stores**: `reviewer_score`, `reviewer_scores` (JSON), `reviewer_comments`, `reviewed_at`, `review_tier`, `reviewer_model` (`graph_ops.py:340-345`)
- **State machine**: `drafted` → `published` → `accepted`
- **Write semantics**: `write_standard_names()` uses `coalesce(b.field, sn.field)` — safe to re-run reviews

## Phase 1: Wire `sn review` CLI Command

### CLI Design

The review command deliberately avoids the generate-style scoping (`--ids`, `--domain`)
because review needs full catalog oversight. Instead, it uses **selection filters**
that control _which names to review_ without restricting the LLM's view of the catalog:

```
sn review [OPTIONS]
  --status STATUS      Filter by review_status (default: all statuses)
  --unreviewed         Only review names with no reviewer_score (shortcut)
  --re-review          Force re-review of already-scored names
  --model MODEL        Override model (default: get_model("language"))
  --batch-size INT     Names per LLM call (default: 10)
  --cost-limit FLOAT   Max spend in USD
  --dry-run            Show what would be reviewed, no LLM calls
```

**Key design decisions:**

1. **No `--ids` / `--domain` / `--score-below` scoping** — the review must see the
   full catalog for cross-name consistency. The existing `get_validated_standard_names()`
   already loads from the graph with status filters; we load ALL names for context but
   only score the subset matching the selection filters.

2. **No `--score-below`** — names don't have scores at generation time. After first
   review pass, `--re-review` explicitly opts into re-scoring already-reviewed names.
   The `--unreviewed` flag is the natural starting point.

3. **Full catalog context in each batch** — each LLM review batch receives a "catalog
   summary" (all name IDs + descriptions) so the reviewer can check for naming
   convention consistency and duplicate concepts across the entire catalog.

4. **All metadata persisted** — every review writes: `reviewer_model`, `reviewer_score`,
   `reviewer_scores` (JSON), `reviewer_comments`, `reviewed_at`, `review_tier`,
   review `cost` accumulated on the CLI summary.

### Wiring

1. Load ALL StandardName nodes from graph (full catalog context)
2. Filter to review targets: `--status`, `--unreviewed`, `--re-review`
3. Build catalog summary for LLM context (name IDs + descriptions + kinds)
4. Batch review targets into groups of `--batch-size` (default 10)
5. For each batch: call `_review_batch()` with catalog summary as context
6. Persist scores via `write_standard_names()` (coalesce semantics = safe)
7. Console summary: total reviewed, cost, tier distribution, lowest scorers

### New graph_ops function

Add `get_all_standard_names_for_review()` to `graph_ops.py`:
- Returns ALL StandardName nodes with their properties + linked unit + source IDS
- No status filter (loads entire catalog for context building)
- Returns both the full catalog list and the filtered review targets

### Tests

- `test_sn_review_cli_dry_run` — verifies selection filters and catalog loading
- `test_sn_review_score_persistence` — mock review, verify graph writes include all metadata fields
- `test_sn_review_catalog_context` — verify each batch receives catalog summary

## Phase 2: Bootstrap Loop (Operational — Agent Prompt)

This is an operational sequence, not new code. Run on the live graph after
Phase 1 is implemented. See the Wave 3 agent prompt below for deployment.

### Step 1: Review existing catalog
```bash
uv run imas-codex sn review --unreviewed --cost-limit 5.0
uv run imas-codex sn status  # Check tier distribution
```

### Step 2: Generate for uncovered IDS
```bash
uv run imas-codex sn generate --source dd --ids edge_profiles --cost-limit 2.0
uv run imas-codex sn generate --source dd --ids summary --cost-limit 2.0
uv run imas-codex sn generate --source dd --ids mhd --cost-limit 2.0
```

### Step 3: Review new names
```bash
uv run imas-codex sn review --unreviewed --cost-limit 3.0
```

### Step 4: Regenerate poor names
```bash
# Use sn status to identify low-tier names, then regenerate specific paths
uv run imas-codex sn generate --source dd --paths "equilibrium/time_slice/..." --force
uv run imas-codex sn review --re-review --cost-limit 2.0
```

### Step 5: Iterate until quality targets met
Target: ≥100 names with `review_tier` of "good" or "outstanding"

## Phase 3: Prompt Quality Improvements (After Bootstrap)

Only pursue after Phase 2 produces ≥100 reviewed names:

### 3A: Showcase examples in compose prompt
Use top-scoring names from the bootstrap as few-shot examples in `context.py`.
Replace abstract examples with real, high-quality generated names from the graph.

### 3B: Documentation depth gates
Add soft content gates to `compose_system.md`:
- Physics quantities SHOULD have at least one display equation
- Variables in equations SHOULD be defined with units
- COCOS-dependent quantities MUST have sign convention paragraph

## Explicitly Deferred

- **Benchmarking**: On hold until graph has ≥200 reviewed names with good tier distribution
- **Group consistency review**: Folded into review tool (catalog context provides this)
- **A/B holdout validation**: Needs benchmark infrastructure, deferred

## Phase Dependencies

```
Phase 1 (review CLI)         → implement first
Phase 2 (bootstrap loop)     → after Phase 1, operational agent prompt
Phase 3A (showcase examples) → after Phase 2 produces ≥100 reviewed names
Phase 3B (content gates)     → after Phase 3A
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Add `sn review` to CLI commands table with flags |
| `docs/architecture/standard-names.md` | Update phase table to clarify review is standalone |
| `plans/README.md` | Update status |

## References

- `completed/23-quality-parity.md` — original quality plan
- `completed/25-standalone-review.md` — original review plan
- `standard-names/20-consistency-and-prompt-enrichment.md` — DD enrichment (85% done, reference only)
