# Standard Names: Quality Parity & Standalone Review

> Consolidates remaining work from plans 23 (quality parity) and 25
> (standalone review). Benchmark harness, --paths flag, DD enrichment,
> COCOS injection, and link resolution are all complete. What remains
> is quality refinement and the standalone review CLI.

## Priority: P2

## Phase 1: Standalone Review CLI (from plan 25)

The review infrastructure is fully built — what's missing is the CLI wiring.

### What exists
- `review_worker()` in `workers.py` — processes names in batches with LLM scoring
- `SNQualityReviewBatch`, `SNQualityReview`, `SNQualityScore` — Pydantic models
- `review.md` (system), `review_dd.md` (user context) — prompts
- `sn_review_criteria.yaml` — 6-dimension scoring rubric
- `benchmark_calibration.yaml` — 27 curated reference entries

### Implementation
1. Add `sn review` CLI command to `cli/sn.py`:
   - `--ids IDS` — review names for specific IDS
   - `--domain DOMAIN` — review by physics domain
   - `--score-below FLOAT` — review only low-scoring names
   - `--model MODEL` — override model (default: `get_model("reasoning")`)
   - `--dry-run` — show what would be reviewed
   - `--batch-size INT` — names per LLM call (default: 30)
2. Load catalog snapshot from graph (all StandardName nodes + relationships)
3. Group by IDS then physics domain
4. Call existing `review_worker()` / `_review_batch()` with group context
5. Persist scores to graph (reviewer_score, reviewer_scores, reviewer_comments, etc.)
6. Console summary: score distribution, tier breakdown, low-scorers list

### Tests
- Test CLI invocation with `--dry-run`
- Test score persistence to graph
- Test grouping logic (by IDS, by domain)

## Phase 2: Quality Content Improvements (from plan 23)

### 2A: Showcase examples in compose prompt
Split examples in `context.py` into:
- `_load_showcase_examples()` — 3-5 COMPLETE entries with full documentation (1500+ chars each)
- `_load_vocabulary_examples()` — all abbreviated for reference
Use showcase examples from `benchmark_calibration.yaml` (not holdout paths).

### 2B: Documentation depth gates
Add soft content gates to `compose_system.md`:
- Physics quantities SHOULD have at least one display equation
- Variables in equations SHOULD be defined with units
- Typical values SHOULD reference specific machines
- COCOS-dependent quantities MUST have sign convention paragraph

### 2C: A/B holdout validation (optional)
Framework for comparing pipeline output against ISN catalog entries on
held-out paths. Deferred until Phase 1 review CLI provides scoring.

## Phase 3: Group Consistency (from plan 25, Phase 2)

### 3A: GroupConsistencyReport model
```python
class GroupConsistencyReport(BaseModel):
    naming_consistency: float     # 0-1: parallel naming patterns
    documentation_consistency: float  # 0-1: template adherence
    link_coverage: float          # 0-1: bidirectional link ratio
    overall_coherence: float      # 0-1: group-level assessment
    issues: list[str]
```

### 3B: Cross-name consistency in review
Enhance review prompt to assess group-level patterns:
- Do `*_electron_*` and `*_ion_*` names have parallel structure?
- Are links bidirectional?
- Is documentation style consistent within a domain?

## Phase Dependencies

```
Phase 1 (review CLI)     → independent, do first
Phase 2A (examples)      → independent, parallel with Phase 1
Phase 2B (content gates) → after Phase 2A
Phase 2C (A/B holdout)   → after Phase 1 (uses review scores)
Phase 3 (consistency)    → after Phase 1 (extends review)
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Add `sn review` command documentation |
| `docs/architecture/standard-names.md` | Add review tool section |
| `plans/README.md` | Update status |

## References
- `standard-names/23-quality-parity.md` — original quality plan (keep for reference)
- `standard-names/25-standalone-review.md` — original review plan (keep for reference)
- `standard-names/completed/14-mcp-tools-benchmark.md` — benchmark (done)
- `standard-names/completed/19-benchmark-and-lifecycle.md` — lifecycle (done)
