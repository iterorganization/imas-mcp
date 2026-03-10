# Prompt Optimization

Status: **Phase 1 Complete**
Priority: Medium — cost reduction and score consistency across discovery pipelines
Source: [plans/discovery-prompt-optimization.md](../discovery-prompt-optimization.md)

## Summary

Actionable implementation plan for all recommendations from the discovery
prompt optimization analysis. Three phases: infrastructure fixes (done),
calibration consistency, and wiki calibration extension.

## What's Already Landed

These changes were implemented during the analysis and are on `main`:

| Commit | Change | Status |
|--------|--------|--------|
| `7c0dd3b` | Fix proxy cache: `openai/` → `openrouter/` prefix in LiteLLM proxy path | **Done** |
| `7c0dd3b` | Add `_log_cache_metrics()` for DEBUG-level cache hit/miss logging | **Done** |
| `ecf0271` | Align code scorer TTL: 60s → 300s (matches paths) | **Done** |
| `099af8c` | Deterministic calibration queries: `ORDER BY rand()` → `ORDER BY abs(score - target), id` | **Done** |
| `099af8c` | Push facility preference into Cypher `ORDER BY` (remove Python post-filter) | **Done** |
| `099af8c` | Increase paths/triage `per_level` from 3 → 5 | **Done** |

---

## Phase 1: Verify Deterministic Calibration ✅

> All items complete — code is on `main`.

### 1.1 Verify cache hit rate with deterministic queries

**What:** Run a paths triage against a populated facility and confirm
provider-side cache hits are stable across the full run (not just within
a single 300s window).

**How:**
```bash
# Run with DEBUG logging to see cache metrics
IMAS_CODEX_LOG_LEVEL=DEBUG imas-codex discover paths triage tcv --limit 100
# Grep for cache metrics
grep "cache_" ~/.local/share/imas-codex/logs/discover-*.log | tail -20
```

**Verify:**
- [ ] Cache hit rate > 95% across the entire run (not just first 300s window)
- [ ] No cache misses after the first call per worker
- [ ] `_log_cache_metrics()` output shows `cached_tokens / total_tokens > 0.9`

### 1.2 Verify per_level=5 doesn't degrade triage quality

**What:** Compare triage scores before/after the `per_level` increase
for paths/triage. Spot-check 20 paths to confirm scores are reasonable.

**How:**
```bash
# Run a small triage batch and inspect results
imas-codex discover paths triage tcv --limit 50
# Check triage scores in graph
imas-codex graph query "MATCH (p:FacilityPath {facility_id: 'tcv', status: 'triaged'})
RETURN p.path, p.triage_composite ORDER BY p.triaged_at DESC LIMIT 20"
```

**Verify:**
- [ ] Triage scores are in expected ranges (no systematic inflation/deflation)
- [ ] High-value paths (equilibrium, diagnostic directories) still score > 0.7
- [ ] Low-value paths (config, build artifacts) still score < 0.3

---

## Phase 2: Prompt Ordering Audit

> Verify template structure maximizes the cacheable prefix.

### 2.1 Audit template ordering across all 4 calibrated pipelines

**What:** Confirm all 4 templates with calibration examples place content
in the correct order for maximum cache prefix:
1. Static text (task, philosophy, seed examples)
2. Schema-derived content (enums, JSON schemas)
3. Focus area (`{{ focus }}` — constant per run)
4. Calibration examples (deterministic, stable per graph state)

**Files to audit:**
- [ ] `imas_codex/agentic/prompts/paths/triage.md`
- [ ] `imas_codex/agentic/prompts/paths/scorer.md`
- [ ] `imas_codex/agentic/prompts/code/triage.md`
- [ ] `imas_codex/agentic/prompts/code/scorer.md`

**Current state:** All 4 templates already follow the correct order —
`{% if focus %}` block appears before `{% include "schema/dimension-calibration.md" %}`.
Since focus is constant per run and calibration is now deterministic,
the entire system prompt is stable within a run.

**Verify:**
- [ ] Each template: static → schema → focus → calibration (bottom)
- [ ] `cache_control` breakpoint is on the system message (not user) — confirmed in `inject_cache_control()`
- [ ] No template has volatile content ABOVE the calibration section

**Action required:** If ordering is already correct (expected), mark complete.
If any template has focus AFTER calibration, swap the blocks.

### 2.2 Measure cache benefit empirically

**What:** Compare cache hit rates for runs with and without `--focus`.

**How:**
```bash
# Run without focus
IMAS_CODEX_LOG_LEVEL=DEBUG imas-codex discover paths triage tcv --limit 50
# Run with focus
IMAS_CODEX_LOG_LEVEL=DEBUG imas-codex discover paths triage tcv --limit 50 --focus "equilibrium"
# Compare cache metrics in both log files
```

**Verify:**
- [ ] Without focus: ~100% cache hit rate after first call
- [ ] With focus: ~100% cache hit rate after first call (focus is per-run constant)
- [ ] Changing focus between runs: cache miss on first call only, then hits

---

## Phase 3: Wiki Calibration Extension

> Add graph-backed calibration examples to wiki page and document scoring.

### Rationale

Wiki scoring (`wiki/scorer`, `wiki/document-scorer`) are the only remaining
**scoring** tasks without calibration examples. They process ~150 LLM calls
per facility. Calibration anchors the 0–1 scale to real examples, reducing
score drift across batches and facilities.

Enrichment tasks (signals, static, clusters) do NOT get calibration — they
produce categorical outputs, not scores.

### 3.1 Add wiki page calibration query function

**What:** Create a function to sample scored WikiPage nodes by score range,
matching the pattern used by paths/code calibration.

**File:** `imas_codex/discovery/wiki/scoring.py`

**Implementation:**
```python
_wiki_page_calibration_cache: dict[str, tuple[float, dict]] = {}
_WIKI_CALIBRATION_TTL_SECONDS = 300

def sample_wiki_page_calibration(
    facility: str | None = None,
    per_level: int = 2,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Sample calibration examples from scored WikiPage nodes.

    Same pattern as paths/code calibration: deterministic ordering,
    facility preference, in-process TTL cache.

    Uses wiki score dimensions (score_technical_depth, score_data_access,
    score_imas_relevance, score_operational_value, score_calibration_content).
    """
```

Key design decisions:
- `per_level=2` (not 5) — wiki has fewer scored items and fewer dimensions
- Same 5 score buckets as paths/code (lowest/low/medium/high/highest)
- Same deterministic `ORDER BY` pattern (facility preference, closest to target, id)
- Wiki score dimensions are a subset — query `WikiScoreResult` model fields

**Properties to return per example:**
```python
{"id": str, "title": str, "score": float, "purpose": str, "description": str}
```

**Verify:**
- [ ] Function returns populated dict when WikiPage nodes have scores
- [ ] Function returns empty lists when no scored WikiPages exist (bootstrap)
- [ ] Results are deterministic (same graph → same output)
- [ ] TTL cache prevents redundant graph queries within 300s

### 3.2 Add wiki document calibration query function

**What:** Same as 3.1 but for WikiDocument nodes.

**File:** `imas_codex/discovery/wiki/scoring.py`

**Implementation:** Near-identical to 3.1, querying `WikiDocument` nodes
instead of `WikiPage`. WikiDocuments have the same score dimensions.

**Verify:**
- [ ] Same verification as 3.1 for WikiDocument nodes

### 3.3 Create wiki calibration template

**What:** Create a wiki-specific calibration template, or reuse the
existing `dimension-calibration.md` template.

**Decision:** Reuse `dimension-calibration.md` — the template is generic.
It iterates `dimension_calibration` and renders `ex.path`, `ex.facility`,
`ex.score`, `ex.description`. Wiki examples need `ex.title` instead of
`ex.path`, but can map `title` → `path` in the query function for
template compatibility.

Alternatively, add a guard:
```jinja2
- `{{ ex.title or ex.path }}` [{{ ex.facility }}] → {{ ex.score }} - {{ ex.description }}
```

**Verify:**
- [ ] Template renders wiki examples correctly (title shows, not path)
- [ ] Template handles empty calibration gracefully (no examples yet)

### 3.4 Wire calibration into wiki prompt builders

**What:** Inject `dimension_calibration` into the context dict for
`wiki/scorer` and `wiki/document-scorer` prompt rendering.

**Files:**
- `imas_codex/discovery/wiki/scoring.py` — `score_pages_batch()` (~L704)
- `imas_codex/discovery/wiki/scoring.py` — `score_documents_batch()` (~L310)

**Implementation:**
```python
# In score_pages_batch():
from imas_codex.discovery.wiki.scoring import sample_wiki_page_calibration

dimension_calibration = sample_wiki_page_calibration(facility=facility, per_level=2)
has_calibration = any(
    any(examples for examples in levels.values())
    for levels in dimension_calibration.values()
)
if has_calibration:
    context["dimension_calibration"] = dimension_calibration
```

**Files to update:**
- [ ] `imas_codex/agentic/prompt_loader.py` — add `"dimension_calibration"` to
  the `wiki/scorer` and `wiki/document-scorer` provider lists (if needed, or
  pass directly via context)

**Verify:**
- [ ] `render_prompt("wiki/scorer", context)` includes calibration section when data exists
- [ ] `render_prompt("wiki/scorer", context)` omits calibration section during bootstrap
- [ ] System prompt token count increases by ~500-800 tokens with calibration (still well above cache threshold)

### 3.5 Add `{% include "schema/dimension-calibration.md" %}` to wiki templates

**What:** Add the calibration include to the bottom of `wiki/scorer.md`
and `wiki/document-scorer.md`, after the focus/data_access_patterns sections.

**Files:**
- [ ] `imas_codex/agentic/prompts/wiki/scorer.md` — add at end:
  ```
  {% if dimension_calibration %}
  {% include "schema/dimension-calibration.md" %}
  {% endif %}
  ```
- [ ] `imas_codex/agentic/prompts/wiki/document-scorer.md` — same addition

**Verify:**
- [ ] Template renders without error when `dimension_calibration` is not in context
- [ ] Template renders calibration block when `dimension_calibration` is provided
- [ ] Calibration section appears AFTER focus and data_access_patterns (correct cache ordering)

### 3.6 End-to-end verification

**What:** Run wiki scoring with calibration enabled and verify output quality.

**How:**
```bash
# Ensure some wiki pages are already scored (from prior runs)
imas-codex graph query "MATCH (w:WikiPage {facility_id: 'jet'})
WHERE w.score_composite IS NOT NULL RETURN count(w)"

# Run scoring with DEBUG logging
IMAS_CODEX_LOG_LEVEL=DEBUG imas-codex discover wiki score jet --limit 40

# Check calibration appears in prompt (look for "Calibration Examples" in debug output)
grep -i "calibration" ~/.local/share/imas-codex/logs/discover-*.log | head -5

# Check scores are reasonable
imas-codex graph query "MATCH (w:WikiPage {facility_id: 'jet'})
WHERE w.scored_at IS NOT NULL
RETURN w.title, w.score_composite, w.page_purpose
ORDER BY w.scored_at DESC LIMIT 10"
```

**Verify:**
- [ ] Calibration examples appear in system prompt (when scored WikiPages exist)
- [ ] Scores are consistent with calibration ranges
- [ ] Cache hit rate remains high (~100% after first call)
- [ ] No regression in existing wiki scoring behavior when calibration data is absent

---

## Phase 4: Batch Size Tuning (Optional)

> Lower priority. Evaluate whether larger batch sizes reduce cost without
> degrading quality.

### 4.1 Measure quality vs batch size for paths/triage

**What:** Paths triage currently processes 25 directories per batch.
Test with 35 and 50 to see if scores remain consistent.

**How:**
```bash
# Run with different batch sizes on same set of paths
imas-codex discover paths triage tcv --batch-size 25 --limit 100
# Record scores, then reset and re-run
imas-codex discover paths triage tcv --batch-size 50 --limit 100
# Compare scores for the same paths
```

**Verify:**
- [ ] Score correlation > 0.9 between batch sizes
- [ ] No increase in parse errors at larger batch sizes
- [ ] Cost per path decreases with larger batches (system prompt amortized)

### 4.2 Measure quality vs batch size for code/triage

**What:** Same as 4.1 but for code triage (currently 20 files per batch).

**Verify:**
- [ ] Score correlation > 0.9 between batch sizes
- [ ] No increase in structured output parse errors

---

## Phase 5: Score Calibration Deduplication (Optional)

> Lower priority. Remove redundant `score_calibration` examples from
> paths/scorer if they overlap with `dimension_calibration`.

### 5.1 Assess overlap between score_calibration and dimension_calibration

**What:** The paths/scorer prompt includes both `dimension_calibration`
(per-dimension examples at 5 score levels) AND `score_calibration`
(enriched path examples by category, 8 categories × 2 examples).
These may provide redundant signal.

**File:** `imas_codex/agentic/prompts/paths/scorer.md`

**How:**
```bash
# Check what score_calibration adds that dimension_calibration doesn't
# Score calibration shows enriched evidence (patterns, read/write counts)
# Dimension calibration shows raw scores per dimension
```

**Decision criteria:**
- If `score_calibration` provides unique enrichment-evidence context not
  in `dimension_calibration` → keep both
- If they're redundant → remove `score_calibration` (~950 tokens saved per call)

**Verify:**
- [ ] Run scorer with and without `score_calibration` on same paths
- [ ] Compare score distributions — if within ±0.05 mean, removal is safe
- [ ] If removed: ~950 fewer tokens per scorer call × 133 calls = ~127k tokens saved (~$0.01/facility)

---

## Cost Impact Summary

| Phase | Optimization | Input Cost Impact | Accuracy Impact |
|-------|-------------|-------------------|-----------------|
| 1 ✅ | Deterministic queries + TTL alignment | ~100% cache hits (−$0.05/facility) | Positive — reproducible |
| 1 ✅ | Proxy cache fix (openai/ → openrouter/) | Enables ALL caching (was 0% through proxy) | None |
| 1 ✅ | per_level 3→5 for paths/triage | +$0.002/facility (cached) | Positive — better boundary calibration |
| 2 | Prompt ordering audit | Validates existing cache behavior | None |
| 3 | Wiki calibration extension | +$0.005/facility (cached) | Positive — score consistency |
| 4 | Batch size tuning | Up to −$0.30/facility | Test carefully |
| 5 | score_calibration dedup | −$0.01/facility | Test carefully |

**Baseline:** ~$2.88/facility (no caching) → ~$1.95/facility (with all Phase 1 fixes).
