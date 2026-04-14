# 25: Standalone Standard Name Review Tool

**Status:** Draft
**Created:** 2025-07-18
**Depends on:** Generate pipeline (auto-attach, no inline review)

## Problem Statement

The `sn generate` pipeline now focuses on composition: EXTRACT → COMPOSE →
VALIDATE → CONSOLIDATE → PERSIST. Review/scoring has been removed from this
pipeline because:

1. **Flash-lite rubber-stamps everything** — gives 20/20 on every dimension,
   providing zero signal. Only premium models (opus) discriminate quality.
2. **Per-name review lacks cross-catalog visibility** — reviewing names one at
   a time misses consistency issues (naming pattern divergence, link coverage
   gaps, documentation style drift).
3. **Review should operate on resolved state** — links need to be resolved,
   all related names need to be visible, before quality scoring is meaningful.

A standalone `sn review` tool solves these problems by operating at the
catalog level with full visibility.

## Research Findings

### How ISN Interactive Review Works

The ISN interactive workflow (human + Copilot + MCP tools) reviews names by:

1. Loading the full catalog context (all 309+ names)
2. Grouping names by physics domain or IDS
3. For each group, checking:
   - Naming consistency (do related quantities follow the same pattern?)
   - Documentation completeness (equations, sign conventions, typical values)
   - Link coverage (are cross-references bidirectional?)
   - Unit consistency across related names
4. Using the 6-dimension scoring rubric (grammar, semantic, documentation,
   convention, completeness, compliance)

### Scoring Rubric (from sn_review_criteria.yaml)

Six dimensions, each 0-20 (total /120, normalized to 0-1):

| Dimension | What it measures |
|-----------|------------------|
| grammar | Correct segment order, valid tokens, canonical form |
| semantic | Name accurately captures the physics quantity |
| documentation | Richness: equations, sign conventions, typical values, links |
| convention | Follows IMAS/fusion naming conventions |
| completeness | All required fields present and properly populated |
| compliance | Passes ISN validation (grammar round-trip, semantic checks) |

Tiers: outstanding (≥0.85), good (0.60-0.84), adequate (0.40-0.59), poor (<0.40)

### Existing Review Infrastructure

- `review_worker` function in `workers.py` — processes names in batches with
  LLM scoring. Currently unused by generate pipeline but still functional.
- `_review_batch()` helper — builds review context with calibration entries
  and existing names, calls LLM for scoring.
- `SNQualityReviewBatch`, `SNQualityReview`, `SNQualityScore` models — full
  Pydantic models for review responses.
- `sn_review_criteria.yaml` — scoring dimensions, tiers, verdict rules.
- `benchmark_calibration.yaml` — 27 curated reference entries for consistency.
- Prompts: `review.md` (system), `review_dd.md` (user context).

## Design

### Architecture

```
sn review [--ids IDS] [--domain DOMAIN] [--score-below FLOAT] [--model MODEL]
    │
    ├── 1. Load catalog snapshot from graph
    │      - All StandardName nodes with relationships
    │      - Group by IDS, domain, or naming pattern
    │
    ├── 2. Resolve links (ensure all links point to real names)
    │      - Call resolve_links_batch() if unresolved links exist
    │
    ├── 3. Build review groups
    │      - Group by IDS (primary) then physics domain (secondary)
    │      - Each group gets full cross-name visibility
    │      - Include calibration entries as quality anchors
    │
    ├── 4. LLM Review (premium model — opus or sonnet)
    │      - System prompt: scoring rubric + calibration examples
    │      - User prompt: full group context with all names + docs
    │      - Response: per-name scores + group-level consistency notes
    │
    ├── 5. Persist scores to graph
    │      - Update reviewer_score, reviewer_scores, reviewer_comments
    │      - Set review_tier, reviewed_at, reviewer_model
    │
    └── 6. Report
           - Console summary: score distribution, low-scorers, consistency issues
           - Optional: export review report to YAML/markdown
```

### Key Design Decisions

1. **Group-level review, not per-name**: The LLM sees all names in a group
   at once. This enables consistency checks (e.g., "electron_temperature" and
   "ion_temperature" should have parallel documentation structure).

2. **Premium model only**: Review quality requires opus-class models. The
   review tool should default to `get_model("reasoning")` or accept
   `--model` override. No point using budget models for quality assessment.

3. **Score gate for regeneration**: `sn review` scores but does not modify
   names. Low-scoring names can be fed back to `sn generate --paths` for
   regeneration. Future: `sn review --regenerate-below 0.5` could automate
   this loop.

4. **Link resolution prerequisite**: Review should check that all links are
   resolved before scoring. Unresolved links degrade the completeness score.
   The tool should warn if unresolved links exist and optionally resolve them
   first.

5. **Cross-name consistency scoring**: Beyond per-name scoring, the review
   should assess group-level consistency:
   - Do related names follow the same pattern? (e.g., `*_electron_*` and
     `*_ion_*` should be structurally parallel)
   - Are links bidirectional? (if A links to B, does B link to A?)
   - Are documentation templates consistent within a domain?

### Review Groups Strategy

| Group By | Use Case |
|----------|----------|
| IDS | Default — reviews names within each IDS for internal consistency |
| Domain | Cross-IDS review for physics domain coherence |
| Score | Review only names below a threshold (re-review after regeneration) |
| Pattern | Group by naming pattern (all `*_temperature_*`, all `*_flux_*`) |

### CLI Interface

```bash
# Review all names in an IDS
imas-codex sn review --ids equilibrium

# Review names below quality threshold
imas-codex sn review --score-below 0.7

# Review a specific physics domain across all IDSs
imas-codex sn review --domain magnetics

# Review with specific model
imas-codex sn review --model anthropic/claude-opus-4.6

# Dry run — show what would be reviewed without LLM calls
imas-codex sn review --dry-run

# Review and auto-regenerate low scorers
imas-codex sn review --regenerate-below 0.5  # future
```

### LLM Prompt Design

**System prompt** (cached — same for all review groups):
- Full scoring rubric (6 dimensions with detailed criteria)
- Calibration examples (3-5 high-quality reference entries)
- Group-level consistency instructions
- Output schema (per-name scores + group notes)

**User prompt** (per-group):
- IDS overview (from graph)
- All names in the group with full metadata:
  - name, description, documentation, unit, kind, tags, links
  - imas_paths, grammar_fields, constraints, validity_domain
- Cross-reference map (which names link to which)
- Any unresolved links flagged

**Response model**:
```python
class SNReviewGroupResult(BaseModel):
    reviews: list[SNQualityReview]  # Per-name scores (reuse existing model)
    group_consistency: GroupConsistencyReport
    recommendations: list[str]  # Actionable improvement suggestions

class GroupConsistencyReport(BaseModel):
    naming_consistency: float  # 0-1: do related names follow the same pattern?
    documentation_consistency: float  # 0-1: template adherence across group
    link_coverage: float  # 0-1: ratio of bidirectional links
    overall_coherence: float  # 0-1: group-level quality assessment
    issues: list[str]  # Specific consistency problems found
```

## Implementation Phases

### Phase 1: Core Review Command
- New `sn review` CLI command with `--ids`, `--domain`, `--score-below`, `--model`
- Catalog loading from graph (all StandardName nodes + relationships)
- Group building (by IDS, then by domain)
- Reuse existing `_review_batch()` with enhanced group context
- Persist scores to graph
- Console summary report

### Phase 2: Cross-Name Consistency
- Group consistency model and scoring
- Bidirectional link checking
- Documentation template adherence checking
- Pattern consistency across related quantities

### Phase 3: Review-Regenerate Loop
- `--regenerate-below` flag to auto-feed low scorers back to generate
- Iteration tracking (review round number)
- Score improvement tracking

## Open Questions

1. **Batch size for review groups**: How many names can fit in one LLM context?
   With opus at 200K context, we can fit ~50-100 names with full documentation.
   Groups larger than this need splitting.

2. **Incremental vs full review**: Should `sn review` always review everything,
   or only names that haven't been reviewed since their last generation?
   Recommendation: default to unreviewed/stale names, `--all` for full catalog.

3. **Score persistence strategy**: Should review scores from different models
   coexist (reviewer A scores vs reviewer B scores)? Or does each review
   overwrite? Recommendation: overwrite — the latest review is authoritative.
   History is preserved in review audit trail (reviewed_at, reviewer_model).
