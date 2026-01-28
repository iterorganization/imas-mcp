# Wiki Scoring Update Strategy

**Date**: January 19, 2026  
**Status**: Analysis Complete - Awaiting Decision

## Executive Summary

The wiki scoring schema was recently redesigned (commit `72c043d`, Jan 19 2026) to implement **content-aware LLM scoring** with new fields (`page_type`, `is_physics_content`, `value_rating`, `preview_text`, `preview_summary`). However, **only 1 out of 3,513 pages** in the graph has been scored with the new schema.

**Recommendation**: **YES - Re-run the complete scoring cycle** to populate the new schema fields across all pages.

---

## Current State Analysis

### Graph Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total WikiPages** | 3,513 | 100% |
| **Pages with new schema** | 1 | 0.03% |
| **Pages without new schema** | 3,512 | 99.97% |
| **Scored (old schema)** | 2,804 | 79.8% |
| **Ingested** | 705 | 20.1% |

### Facility Breakdown

| Facility | Total | Scored | Ingested |
|----------|-------|--------|----------|
| **EPFL** | 2,972 | 2,263 | 705 |
| **ITER** | 541 | 541 | 0 |

### Schema Migration Status

**Old schema fields** (populated on 3,512 pages):
- ✓ `interest_score` (0.0-1.0)
- ✓ `score_reasoning` (text)
- ✓ `status` ('scored', 'ingested', etc.)

**New schema fields** (populated on only 1 page):
- ✗ `page_type` (enum: data_source, documentation, code, etc.)
- ✗ `is_physics_content` (boolean)
- ✗ `value_rating` (0-10 integer)
- ✗ `preview_text` (first 2000 chars of page content)
- ✗ `preview_summary` (LLM-generated summary)

---

## Why the Redesign Happened

### Problem: Facility Bias in Old Scoring

The old metric-based scoring system was **EPFL-biased** and systematically underscored ITER Confluence pages:

| Metric | EPFL | ITER | Gap |
|--------|------|------|-----|
| Pages scored < 0.4 | 60.9% | 73.0% | **+12.1%** |
| Pages scored >= 0.8 | 1.4% | 0.6% | -0.8% |
| Pages scored >= 0.6 | 13.1% | 7.2% | **-5.9%** |

**Root cause**: The old system relied on graph topology metrics (`in_degree`, `link_depth`) which penalized ITER Confluence pages due to architectural differences, not content quality.

### Example Failures (Old Scoring)

| Page | Old Score | Should Be | Problem |
|------|-----------|-----------|---------|
| JOREK disruption cases | 0.45 | 0.75+ | Critical ML dataset, penalized for low in_degree |
| DINA disruption cases | 0.45 | 0.75+ | Same issue |
| SOLPS User Forum | 0.20 | 0.65 | "Forum" keyword triggered negative scoring |
| ITER code docs | 0.35 | 0.70 | Generic title, low in_degree |

### Solution: Content-Aware Scoring

The new system (commit `72c043d`) implements a **two-stage LLM-centric approach**:

1. **Prefetch Stage**: Fetch page content + LLM summarize → store `preview_text` + `preview_summary`
2. **Scoring Stage**: LLM evaluates title + summary + metrics → content-aware score + metadata

**Key improvements**:
- Scores based on **actual page content**, not just topology
- Facility-agnostic (works equally for EPFL wiki and ITER Confluence)
- Stores rich metadata (`page_type`, `is_physics_content`, `value_rating`)
- Expected accuracy improvement: **40% → 85%**

---

## What Needs to Be Done

### Phase 1: Prefetch Content (NEW)

**Command**: `uv run imas-codex wiki prefetch <facility>`

**What it does**:
1. Fetch page content via HTTP (with auth handling)
2. Extract clean text (first 2000 chars)
3. LLM summarize content (max 300 chars)
4. Store `preview_text` + `preview_summary` in graph

**Status**: ✗ Not run yet (0 pages have preview data)

**Estimated cost**: ~$10.50 for 3,513 pages ($0.003/page)

### Phase 2: Re-score with New Schema

**Command**: `uv run imas-codex wiki score <facility> --rescore`

**What it does**:
1. Read `preview_summary` + graph metrics
2. LLM classifies page type and content value
3. Store `page_type`, `is_physics_content`, `value_rating`, updated `interest_score`

**Status**: ✗ Only 1 page scored with new schema

**Estimated cost**: ~$3.50 for 3,513 pages ($0.001/page)

### Total Cost: ~$14.00

**ROI**: $14 investment for:
- 2x accuracy improvement (40% → 85%)
- Elimination of facility bias
- Rich metadata for downstream analysis
- Content-aware filtering and search

---

## Implementation Strategy

### Option A: Full Re-score (Recommended)

**Scope**: All 3,513 pages  
**Timeline**: 2-3 days  
**Cost**: ~$14.00

**Steps**:
```bash
# 1. Backup graph first (CRITICAL)
uv run imas-codex neo4j dump

# 2. Prefetch all pages
uv run imas-codex wiki prefetch tcv
uv run imas-codex wiki prefetch iter

# 3. Re-score all pages
uv run imas-codex wiki score tcv --rescore
uv run imas-codex wiki score iter --rescore

# 4. Validate results
uv run imas-codex wiki stats
```

**Pros**:
- Complete schema migration
- Consistent scoring across all pages
- Enables content-aware filtering
- Fixes ITER bias immediately

**Cons**:
- Higher cost ($14 vs incremental)
- Requires 2-3 days of processing
- May need auth setup for ITER Confluence

### Option B: Incremental Re-score

**Scope**: Only high-value pages (score >= 0.5) + ITER pages  
**Timeline**: 1 day  
**Cost**: ~$7.00

**Steps**:
```bash
# 1. Backup graph
uv run imas-codex neo4j dump

# 2. Prefetch high-value + ITER pages
uv run imas-codex wiki prefetch tcv --filter "interest_score >= 0.5"
uv run imas-codex wiki prefetch iter

# 3. Re-score filtered pages
uv run imas-codex wiki score tcv --filter "interest_score >= 0.5" --rescore
uv run imas-codex wiki score iter --rescore
```

**Pros**:
- Lower cost
- Faster completion
- Fixes ITER bias
- Focuses on valuable content

**Cons**:
- Incomplete schema migration
- Low-value pages remain on old schema
- May miss hidden gems in low-scored pages

### Option C: Pilot + Full Rollout

**Scope**: 200 pages pilot → full 3,513 pages  
**Timeline**: 1 week  
**Cost**: ~$15.00 (pilot + full)

**Steps**:
```bash
# 1. Pilot (200 pages)
uv run imas-codex wiki prefetch tcv --max-pages 100
uv run imas-codex wiki prefetch iter --max-pages 100
uv run imas-codex wiki score tcv --max-pages 100 --rescore
uv run imas-codex wiki score iter --max-pages 100 --rescore

# 2. Validate pilot results
# - Check JOREK/DINA disruption cases: 0.45 → 0.75+?
# - Check ITER average score: 0.38 → 0.55+?
# - Manual review 20 pages

# 3. If successful, full rollout
uv run imas-codex wiki prefetch tcv
uv run imas-codex wiki prefetch iter
uv run imas-codex wiki score tcv --rescore
uv run imas-codex wiki score iter --rescore
```

**Pros**:
- Risk mitigation via pilot
- Validates prompts before full rollout
- Allows prompt refinement
- Builds confidence

**Cons**:
- Longer timeline
- Slightly higher cost (duplicate pilot work)
- More manual validation required

---

## Recommended Approach: **Option A (Full Re-score)**

### Rationale

1. **Schema is already deployed**: The code, prompts, and graph schema are ready
2. **Cost is justified**: $14 for 2x accuracy + facility-agnostic scoring is excellent ROI
3. **Incomplete migration is technical debt**: Having 99.97% of pages on old schema creates inconsistency
4. **ITER bias is a known issue**: Needs fixing across all 541 ITER pages
5. **Prefetch enables future features**: Content summaries unlock semantic search, clustering, etc.

### Success Criteria

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| JOREK disruption cases score | 0.45 | 0.75+ | Direct query |
| DINA disruption cases score | 0.45 | 0.75+ | Direct query |
| ITER average score | 0.38 | 0.55+ | `AVG(interest_score)` |
| ITER pages >= 0.6 | 7.2% | 15%+ | `COUNT` query |
| Pages with new schema | 0.03% | 100% | Schema migration complete |
| Prefetch success rate | N/A | 90%+ | `fetched / total` |

### Validation Queries

After re-scoring, run these queries to validate:

```cypher
// 1. Check schema migration
MATCH (wp:WikiPage)
RETURN count(*) AS total,
       count(CASE WHEN wp.page_type IS NOT NULL THEN 1 END) AS with_new_schema,
       count(CASE WHEN wp.preview_summary IS NOT NULL THEN 1 END) AS with_summary

// 2. Check ITER score improvement
MATCH (wp:WikiPage {facility_id: 'iter'})
RETURN AVG(wp.interest_score) AS avg_score,
       count(CASE WHEN wp.interest_score >= 0.6 THEN 1 END) AS high_value_count

// 3. Check specific test cases
MATCH (wp:WikiPage)
WHERE wp.title CONTAINS 'JOREK' AND wp.title CONTAINS 'disruption'
RETURN wp.id, wp.title, wp.interest_score, wp.page_type, wp.value_rating

// 4. Check page type distribution
MATCH (wp:WikiPage)
RETURN wp.page_type, count(*) AS count
ORDER BY count DESC
```

---

## Risk Mitigation

### Risk 1: Authentication Barriers (ITER Confluence)

**Problem**: Many ITER Confluence pages require login.

**Mitigation**:
- Prefetch module has graceful fallback to title-only scoring
- Stores `preview_fetch_error` for debugging
- Summary: "[Auth required - scoring based on title only]"
- Still better than current: at least we know it failed

**Action**: Check if ITER Confluence credentials are configured in `.env`

### Risk 2: Cost Overrun

**Problem**: LLM costs may exceed $14 estimate.

**Mitigation**:
- Set hard budget limits in code
- Monitor costs during execution
- Use cheaper model for summarization (Haiku)
- Batch aggressively (50 pages per call)

**Action**: Set `OPENROUTER_BUDGET_USD=20` in `.env` as safety limit

### Risk 3: Regression on EPFL Scores

**Problem**: New scoring may break EPFL pages that work well.

**Mitigation**:
- Store old scores in `interest_score_old` before re-scoring
- Compare old vs new scores after completion
- Rollback plan: `UPDATE WikiPage SET interest_score = interest_score_old`

**Action**: Add `interest_score_old` field to schema before re-scoring

### Risk 4: Prefetch Failures

**Problem**: HTTP timeouts, 404s, auth failures.

**Mitigation**:
- Prefetch module handles errors gracefully
- Stores error messages in `preview_fetch_error`
- Continues processing remaining pages
- Can retry failed pages later

**Action**: Monitor prefetch success rate, aim for 90%+

---

## Rollback Plan

If the new scoring system fails validation:

1. **Immediate**: Restore old scores from `interest_score_old` field
   ```cypher
   MATCH (wp:WikiPage)
   WHERE wp.interest_score_old IS NOT NULL
   SET wp.interest_score = wp.interest_score_old
   ```

2. **Revert**: Restore graph from backup
   ```bash
   uv run imas-codex neo4j load imas-codex-graph-<version>.dump
   ```

3. **Investigate**: Analyze failure cases
   - Which pages scored worse?
   - What patterns in `score_reasoning`?
   - Are prompts too strict/lenient?

4. **Iterate**: Refine prompts and retry pilot

---

## Next Steps

### Immediate Actions (Before Re-scoring)

1. **Backup graph** (CRITICAL)
   ```bash
   uv run imas-codex neo4j dump
   ```

2. **Add rollback field** to schema
   ```cypher
   MATCH (wp:WikiPage)
   WHERE wp.interest_score IS NOT NULL
   SET wp.interest_score_old = wp.interest_score
   ```

3. **Check ITER Confluence auth**
   - Verify credentials in `.env`
   - Test prefetch on 1-2 ITER pages
   - Document auth setup if needed

4. **Set budget limits**
   ```bash
   export OPENROUTER_BUDGET_USD=20
   ```

### Execution (2-3 days)

**Day 1: Prefetch**
```bash
# Morning: EPFL (2,972 pages, ~6 hours)
uv run imas-codex wiki prefetch tcv

# Afternoon: ITER (541 pages, ~1 hour)
uv run imas-codex wiki prefetch iter

# Validate prefetch
# - Check success rate >= 90%
# - Review error messages
# - Spot-check summaries
```

**Day 2: Re-score**
```bash
# Morning: EPFL (2,972 pages, ~3 hours)
uv run imas-codex wiki score tcv --rescore

# Afternoon: ITER (541 pages, ~30 min)
uv run imas-codex wiki score iter --rescore

# Validate scoring
# - Run validation queries
# - Check ITER score improvement
# - Review test cases (JOREK, DINA, SOLPS)
```

**Day 3: Validation & Documentation**
- Manual review of 50 pages (25 EPFL + 25 ITER)
- Compare old vs new scores
- Document findings
- Update wiki ingestion pipeline if needed

### Post-Execution

1. **Update documentation**
   - Mark redesign as complete in `plans/features/wiki-scoring-redesign.md`
   - Document new schema fields in `docs/architecture/graph.md`
   - Update CLI help text if needed

2. **Clean up old code**
   - Remove old metric-based scoring logic (if any)
   - Archive old prompts
   - Update tests

3. **Enable downstream features**
   - Semantic search on `preview_summary`
   - Page type filtering in ingestion
   - Physics content prioritization

---

## Cost-Benefit Summary

| Aspect | Current | After Re-score | Improvement |
|--------|---------|----------------|-------------|
| **Schema completeness** | 0.03% | 100% | +99.97% |
| **ITER bias** | HIGH | LOW | Fixed |
| **Accuracy (specialized)** | ~40% | ~85% | +45% |
| **Content metadata** | None | Rich | New capability |
| **Total cost** | $0 | $14 | One-time investment |
| **Facility-agnostic** | NO | YES | Architectural win |

**Recommendation**: **Proceed with Option A (Full Re-score)** - the benefits far outweigh the costs.

---

## References

- **Redesign Plan**: `plans/features/wiki-scoring-redesign.md`
- **Redesign Commit**: `72c043d` (Jan 19, 2026)
- **Prefetch Module**: `imas_codex/wiki/prefetch.py`
- **Scoring Module**: `imas_codex/wiki/discovery.py`
- **Schema**: `imas_codex/schemas/facility.yaml`
- **CLI Commands**: `imas_codex/cli.py` (lines 2500+)

---

## Appendix: Current vs New Scoring

### Old Scoring (Metric-Based)

**Input**: Title + URL + graph metrics only  
**Process**: Pattern matching on keywords + topology heuristics  
**Output**: `interest_score` (0.0-1.0)

**Example**:
```json
{
  "id": "iter:559745500",
  "title": "The JOREK disruption cases",
  "interest_score": 0.45,
  "score_reasoning": "Depth 3 with low in_degree (1). JOREK disruption cases."
}
```

### New Scoring (Content-Aware)

**Input**: Title + content summary + graph metrics  
**Process**: LLM semantic analysis of actual page content  
**Output**: `interest_score` + `page_type` + `is_physics_content` + `value_rating`

**Example**:
```json
{
  "id": "iter:559745500",
  "title": "The JOREK disruption cases",
  "preview_summary": "Database of JOREK MHD simulation cases for disruption scenarios. Contains validated simulation results for code benchmarking and ML training.",
  "interest_score": 0.78,
  "page_type": "data_source",
  "is_physics_content": true,
  "value_rating": 9,
  "score_reasoning": "Critical disruption database for ML training and code validation. High scientific value despite low in_degree (expected for external data source)."
}
```

**Key difference**: New scoring understands that low `in_degree` doesn't mean low value for specialized data sources.
