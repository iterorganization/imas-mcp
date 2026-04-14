# Standard Names: Bootstrap Generation Loop

> Priority: P1 — The StandardName graph is empty. This must run first.

## Problem

Zero StandardName nodes exist in the graph. The `sn generate` pipeline is
fully functional (schema-enforced LLM compose, automatic DD unit injection,
COCOS propagation, 3-layer ISN validation, conflict-detecting persistence).
What's missing is the `sn review` CLI command to score generated names, and
an operational loop to iterate quality upward.

## Verified Code Facts (rubber-duck confirmed)

- **LLM compose is schema-enforced**: `acall_llm_structured(..., response_model=SNComposeBatch)` in `workers.py:612-615`
- **Units are LLM-excluded**: prompt says "Do NOT output a `unit` field" (`compose_system.md:275`), `SNCandidate` has no `unit` field
- **Units are DD-injected**: `workers.py:629-667` injects from DD, persists via `HAS_UNIT` relationship
- **review_worker exists**: `workers.py:793+` — processes batches with LLM scoring via `SNQualityReviewBatch`
- **review_worker uses `get_model("language")`** (not "reasoning" as previously speculated)
- **Review batch size is 10** (hardcoded `workers.py:752`)
- **`review_dd.md` does NOT exist** — only `review.md` is present
- **6-dimension scoring**: grammar, semantic, documentation, convention, completeness, compliance (each 0-20, normalized to 0-1)
- **Scoring criteria**: `sn_review_criteria.yaml`
- **Calibration data**: `benchmark_calibration.yaml` — 27 curated reference entries

## Phase 1: Wire `sn review` CLI Command

### Implementation

Add `sn review` to `cli/sn.py`:

```
sn review [OPTIONS]
  --ids IDS           Review names for specific IDS
  --domain DOMAIN     Review by physics domain
  --score-below FLOAT Review only names scoring below threshold
  --status STATUS     Filter by review_status (default: drafted)
  --model MODEL       Override model (default: get_model("language"))
  --batch-size INT    Names per LLM call (default: 10)
  --limit INT         Max names to review
  --cost-limit FLOAT  Max spend in USD
  --dry-run           Show what would be reviewed
```

### Wiring

1. Load StandardName nodes from graph filtered by `--ids`, `--domain`, `--status`, `--score-below`
2. Group by IDS for context coherence
3. Call existing `review_worker()` / `_review_batch()` with loaded names
4. Persist scores: `reviewer_score`, `reviewer_scores` (JSON), `reviewer_comments`, `reviewed_at`, `review_tier`, `reviewer_model`
5. Console summary: score distribution, tier breakdown, lowest scorers

### Tests

- `test_sn_review_cli_dry_run` — verifies filtering and grouping without LLM call
- `test_sn_review_score_persistence` — mock review, verify graph writes

## Phase 2: Bootstrap Loop (Operational)

This is an operational sequence, not new code. Run on the live graph:

### Step 1: Initial Generation
```bash
uv run imas-codex sn generate --source dd --ids equilibrium --cost-limit 2.0
uv run imas-codex sn generate --source dd --ids core_profiles --cost-limit 2.0
uv run imas-codex sn generate --source dd --ids magnetics --cost-limit 2.0
```

### Step 2: Review Generated Names
```bash
uv run imas-codex sn review --ids equilibrium
uv run imas-codex sn review --ids core_profiles
uv run imas-codex sn review --ids magnetics
```

### Step 3: Regenerate Low Scorers
```bash
uv run imas-codex sn status  # Check tier distribution
# Regenerate names scoring below 0.5
uv run imas-codex sn generate --source dd --ids equilibrium --force --score-below 0.5
```

### Step 4: Review Again
```bash
uv run imas-codex sn review --ids equilibrium --score-below 0.6
```

### Step 5: Expand to More IDS
```bash
# Once quality is good for core IDS, expand
uv run imas-codex sn generate --source dd --ids edge_profiles --cost-limit 2.0
uv run imas-codex sn generate --source dd --ids summary --cost-limit 2.0
# ... continue for all physics-relevant IDS
```

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
- **Group consistency review**: Needs enough related names to form meaningful groups
- **A/B holdout validation**: Needs benchmark infrastructure, deferred
- **Concept registry / drift detection**: Low priority refinement

## Phase Dependencies

```
Phase 1 (review CLI)        → independent, implement first
Phase 2 (bootstrap loop)    → after Phase 1
Phase 3A (showcase examples) → after Phase 2 produces ≥100 names
Phase 3B (content gates)     → after Phase 3A
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Add `sn review` to CLI commands table |
| `plans/README.md` | Update status |

## References

- `completed/23-quality-parity.md` — original quality plan
- `completed/25-standalone-review.md` — original review plan
- `standard-names/20-consistency-and-prompt-enrichment.md` — DD enrichment (85% done, reference only)
