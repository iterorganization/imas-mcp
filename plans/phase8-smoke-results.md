# Phase 8 Smoke Test Results

**Date**: 2026-04-29  
**Commit**: ffc5b32e (fairness deadlock fix)  
**Command**: `uv run imas-codex sn run -c 5`

## Result: ✅ PASS

The smoke test ran to completion with `stop_reason=budget_exhausted`.

### SNRun Node (from graph)

| Field | Value |
|-------|-------|
| `cost_spent` | $5.23 |
| `stop_reason` | `budget_exhausted` |
| `names_composed` | 0 (regen, not compose) |
| `started_at` | 2026-04-29T03:09:45Z |
| Duration | ~30 minutes |

### Pool Activity Summary

Active pools (from final PoolHealth log):

| Pool | `consecutive_empty_claims` | Role |
|------|--------------------------|------|
| `generate` | 0 | ✅ Active — composed new names from StandardNameSources |
| `regen` | 0 | ✅ Active — regenerated names with reviewer_score < 0.75 |
| `enrich` | 62 | ⛔ Excluded (all 26 eligible names have confidence < 0.75) |
| `review_names` | 62 | ⛔ Excluded (no newly composed names to review in this run) |
| `review_docs` | 63 | ⛔ Excluded (all eligible names have score < 0.75, reserved for regen) |

### Root Cause Fixed: Fairness Deadlock

**Symptom**: `sn run -c 5` would hang indefinitely with all pools started but
only the first batch processed. The display showed `graph:pending llm:pending`
forever.

**Root cause**: The weighted admission gate (`pool_admit()`) computed effective
weights over `active_pools` — pools with `pending_count > 0` from the display
query. The display query used loose eligibility criteria (e.g. `pipeline_status='named'`
for enrich → 463 nodes). The actual claim functions used stricter criteria
(e.g. `confidence >= 0.75` for enrich → 0 claimable nodes).

This caused `enrich` and `review_docs` to hold weight share in the admission
denominator but consistently return `None` from claim, spending no budget.
Since `total_spent` never grew, the spend shares of `generate` and `regen`
(which had successfully processed one batch each) permanently exceeded their
effective weight thresholds — admission denied forever.

**Fix**: Added `consecutive_empty_claims: int = 0` to `PoolHealth`. In
`pool_loop`, the counter increments each time a pool is admitted but claim
returns `None`, and resets to 0 on any successful claim. In `active_pools_fn`,
pools with `consecutive_empty_claims >= 3` are excluded from the denominator.
After 3 consecutive empty claims (~21 seconds of backoff), the stalled pool
is excluded, productive pools get fair admission, and work proceeds.

**Files changed**:
- `imas_codex/standard_names/pools.py` — `PoolHealth` field + `pool_loop` tracking + `active_pools_fn` logic
- `tests/standard_names/test_phase8_bugfixes.py` — `TestFairnessDeadlockBreaker` (4 tests)

### Timeline

```
05:09:45  sn run -c 5 starts
05:09:56  First LLM calls (cache WRITE, expensive — ~$0.44)
05:10–05:22  generate + regen active; enrich/review_docs reach threshold=3 → excluded
05:22–05:35  regen dominates (all regen candidates); generate claiming from compose queue
05:31:25  Budget pool at $0.03 remaining
05:35:31  Budget pool at $0.03 (last extension)
05:39:17  Budget pool exhausted → $0.00
05:39:21  run_pools: budget exhausted — signalling graceful shutdown
05:39:21  enrich, review_docs, review_names, generate exit cleanly
05:40:21  regen cancelled mid-batch (grace period expired)
05:40:21  All pools exited — SNRun updated with stop_reason=budget_exhausted
```

### Observations

- **Cache hit rate**: 93–95% (most LLM calls hit prompt cache)
- **Unit conflicts**: Several regen candidates had unit mismatches (e.g. `s^-1 vs 1`,
  `m^-3 vs 1`) — these were skipped with a warning but didn't crash the pool
- **Truncated LLM responses**: Two batches hit `completion=32000` (max tokens)
  with invalid JSON; the retry mechanism correctly handled both
- **In-flight batch at shutdown**: regen pool had 1 claim in-flight during grace;
  cancelled cleanly after 60s timeout
- **names_composed=0**: The `names_composed` field tracks generate→compose
  write events; regen regenerates via `write_standard_names` which doesn't
  update this counter. This is expected behaviour.

### Follow-up Issues

1. **`review_docs` never activated**: All 96 eligible nodes had
   `reviewer_score_name < 0.75`, making them reserved for regen. After regen
   improves scores above 0.75, review_docs will activate naturally.
2. **`enrich` never activated**: All 26 valid nodes had `confidence < 0.75`.
   The confidence gate prevents low-quality enrichment. As generate produces
   higher-confidence names (or the confidence threshold is tuned), enrich
   will activate.
3. **Display shows `graph:pending llm:pending`**: The server status panel is
   not updating from the health check results. Separate display bug — does not
   affect pipeline correctness.
4. **names_composed counter**: Consider incrementing `names_composed` in the
   regen persist path for a more accurate progress metric.
