# Cycle Trace: $2 Domain Cycle — `divertor_physics`

**Run ID:** `8bc68bff-a656-4b64-94f3-342554e1eb67`
**Domain:** `divertor_physics` (10 DD leaf paths, never previously cycled)
**CLI command:**
```bash
uv run imas-codex sn run --physics-domain divertor_physics -c 2.0 --turn-number 1 -v
```
**Start:** `2026-04-25T05:40:00Z`  |  **End:** `2026-04-25T05:42:25Z`  |  **Duration:** 145s
**Budget:** $2.00  |  **Spent:** $1.53  |  **Stop reason:** `budget_exhausted`
**Commit:** `650caab2` on `main`

---

## Phase-by-Phase Trace

| Phase | Items In | Items Out | Cost | Errors | Duration | Status |
|-------|----------|-----------|------|--------|----------|--------|
| reconcile | — | 0 | $0.00 | 0 | <1s | ✅ |
| generate (extract) | 45 DD paths | 44 items / 40 batches | $0.00 | 0 | <1s | ✅ |
| generate (compose) | 40 batches | 10 names composed | $1.3547 | 0 | 13s | ⚠️ 31 batches budget-starved |
| generate (validate) | 7 pending | 7 validated | $0.00 | 0 | 8s | ✅ |
| generate (consolidate) | 43 candidates | 43 approved | $0.00 | 0 | <1s | ✅ |
| generate (persist) | 0 needing embed | 0 embedded | $0.00 | 0 | 8s | ✅ |
| enrich (extract) | all `named` | 7 items / 1 batch | $0.00 | 0 | <1s | ✅ |
| enrich (contextualise) | 7 items | 7 contextualised | $0.00 | 0 | 4s | ✅ |
| enrich (document) | 7 items | 7 documented | $0.1734 | 0 | 58s | ✅ |
| enrich (validate) | 7 items | 6 valid, 1 quarantined | $0.00 | 1 quarantine | <1s | ⚠️ |
| enrich (persist) | 7 items | 6 written, 1 skipped | $0.00 | 0 | <1s | ✅ |
| link | — | 0 resolved | $0.00 | 0 | <1s | ✅ |
| review_names (extract) | 155 total SN | 6 targets / 5 batches | $0.00 | 0 | <1s | ✅ |
| review_names (enrich) | 5 batches | 5 enriched | $0.00 | 0 | 1s | ✅ |
| review_names (review) | 5 batches | **0 scored** | **$0.00** | 0 | <1s | ❌ BUDGET EXHAUSTED |
| review_names (persist) | 0 results | 0 persisted | $0.00 | 0 | <1s | ❌ NOTHING TO PERSIST |
| review_docs | 6 targets | **0 scored** | $0.00 | 0 | 2s | ❌ GATE BLOCKED |
| regen | — | — | $0.00 | 0 | — | ⏭️ SKIPPED (no min_score) |

### Final Graph State (divertor_physics)

| Metric | Value |
|--------|-------|
| Total StandardName nodes | 8 |
| validation_status = valid | 6 |
| validation_status = quarantined | 1 |
| validation_status = pending | 1 |
| pipeline_status = enriched | 6 |
| pipeline_status = named | 2 |
| With reviewer_score_name | 1 (pre-existing, score=0.31) |
| With reviewer_score_docs | 1 (pre-existing) |
| StandardNameSource: composed | 10 |
| StandardNameSource: extracted (unprocessed) | 34 |

---

## Blocker #1 — Compose Cost Explosion (CRITICAL)

**Root cause:** Every compose batch sends a ~39,000-token prompt to `anthropic/claude-sonnet-4.6`, regardless of batch size. With 90% singleton batches (1 item per batch), the amortized cost per item is ~$0.15.

**Observed cost per batch:**

| Batch | Items | Composed | Cost | Prompt Tokens | Completion Tokens |
|-------|-------|----------|------|---------------|-------------------|
| unclustered/.../current_incident/s | 2 | 0+1skip | $0.1436 | 39,320 | 38 |
| unclustered/.../particle_flux_recycled_total/s | 2 | 1+1skip | $0.1465 | 39,591 | 176 |
| unclustered/.../power_black_body/s | 2 | 1+1skip | $0.1464 | 39,460 | 198 |
| unclustered/.../power_conducted/s | 2 | 1+1skip | $0.1461 | 39,472 | 175 |
| 5f572b.../K | 1 | 1 | $0.1495 | 39,718 | 351 |
| e82e83.../m | 2 | 2 | $0.1576 | 41,414 | 555 |
| 5f572b.../eV | 2 | 2 | $0.1588 | 41,378 | 639 |
| 5faeec.../m^-2.W | 1 | 1 | $0.1536 | 40,255 | 516 |
| 6c6320.../rad | 1 | 1 | $0.1528 | 40,027 | 509 |

**Key observation:** Prompt size is essentially fixed at ~39–41K tokens. Output is tiny (38–639 tokens). Cost is dominated entirely by prompt tokens. Adding more items to a batch barely changes the cost.

**Budget arithmetic:**
- Compose budget allocation: 30% × $2.00 = **$0.60**
- Actual cost for 9 batches: **$1.3547** (2.26× over budget)
- Cost to compose all 40 batches: 40 × $0.15 = **$6.00** (10× over budget)
- Budget reserve per batch (expected): $0.60 / 40 = **$0.015**
- Actual cost per batch: **$0.15** (10× higher than reservation)

**Impact:** 196 budget-reserve failures logged. After 9 batches exhausted the pool, the remaining 31 batches had no budget. Compose consumed 67.7% of the total $2 budget.

**Log excerpt:**
```
07:40:48 - DEBUG - sn_compose_worker: Worker 9: budget reserve failed for unclustered/.../power_convected/s, re-enqueuing (retry 1/5)
...  [196 total reserve failures across 31 batches × multiple retries]
07:40:59 - INFO - sn_compose_worker: L7: Skipped — remaining budget $-0.75 < $0.50 threshold
07:40:59 - INFO - sn_compose_worker: Composition complete: 10 composed, 0 attached, 0 errors (cost=$1.3547)
```

**Cross-reference:** → `rd1-budget-trace` should investigate prompt template bloat (39K tokens for a 1-item batch is extreme).

---

## Blocker #2 — Review Budget Starvation (CRITICAL)

**Root cause:** Adaptive budget for `review_names` computes remaining budget after actual prior spend, then takes 45% of it. But compose overspent so severely that even a 1-name review batch can't be funded.

**Budget arithmetic:**
```
Prior spend:  generate=$1.3547 + enrich=$0.1734 + link=$0.00 = $1.5281
Remaining:    $2.00 - $1.5281 = $0.4719
Review share: $0.4719 × 0.45 = $0.2124
Min reservation for 1-name batch: 1 × $0.05 × 3 models × 1.5× = $0.225
$0.2124 < $0.225 → EVERY BATCH REJECTED
```

**Result:** 6 eligible names identified, 0 scored. The review worker immediately rejects all 5 batches.

**Invariant violation triggered:**
```
07:42:23 - ERROR - Phase review_names invariant violated:
    6 eligible names identified but zero persisted (not budget-exhausted)
```

This invariant check (`total_cost < budget * 0.5`) classifies the failure as "not budget-exhausted" because the review worker spent $0.00. But the *root cause* IS budget exhaustion — just at the reservation check, not during actual LLM calls.

**Log excerpt:**
```
07:42:23 - INFO - sn_review_review: Budget exhausted at batch 0 — stopping review
07:42:23 - INFO - sn_review_review: Budget exhausted at batch 1 — stopping review
07:42:23 - INFO - sn_review_review: Budget exhausted at batch 2 — stopping review
07:42:23 - INFO - sn_review_review: Budget exhausted at batch 3 — stopping review
07:42:23 - INFO - sn_review_review: Budget exhausted at batch 4 — stopping review
07:42:23 - INFO - sn_review_review: Review complete: 0 scored, 0 unscored, 0 revised, 0 errors (cost=$0.0000, tokens=0)
```

**Cross-reference:** → `rd2-reviewer-pattern` should consider whether the 3-model quorum's reservation formula ($0.225/name) is too aggressive for small budgets.

---

## Blocker #3 — Review Docs Gate Blocked (CASCADING)

**Root cause:** `review_docs` requires `reviewed_name_at IS NOT NULL`, which is only set by a successful `review_names` pass. Since review_names scored 0 names, all 6 are gated.

**Log excerpt:**
```
07:42:23 - DEBUG - sn_review_extract: Docs gate: skipping 'maximum_surface_temperature_of_divertor_target' — reviewed_name_at IS NULL
07:42:23 - DEBUG - sn_review_extract: Docs gate: skipping 'vertical_extent_of_divertor_target' — reviewed_name_at IS NULL
... [all 6 skipped]
07:42:23 - INFO - sn_review_extract: Filter result: 0 targets from 155 total
```

**Impact:** The sequential gate (name_review → docs_review) means a failure in review_names blocks the entire downstream chain.

---

## Blocker #4 — Enrich Quarantine: ISN Vocabulary Gap (MINOR)

**Name quarantined:** `major_radius_extent_of_divertor_target`

**Reasons (3 validation failures):**
1. **Grammar gap:** Token `major_radius` used as coordinate prefix is missing from `coordinate_axes` vocabulary
2. **Link not found:** `name:vertical_extent_of_divertor_target` (link target exists but not yet in links vocabulary at validate time)
3. **Link not found:** `name:maximum_steady_state_heat_flux_limit_of_divertor_target` (same)

**Impact:** 1/7 enriched names quarantined → only 6 of 8 total names reached `enriched` status.

**Cross-reference:** → `rd3-dd-classifier` may find that `major_radius` should be added to `coordinate_axes` vocabulary.

---

## Blocker #5 — Compose Error Siblings: ISN Token Misses (MODERATE)

Every batch produced names that failed ISN validation for error-sibling generation (uncertainty fields). Missing physical_base tokens:

| Token Miss | ISN Residue | Nearest Candidates |
|------------|-------------|-------------------|
| `electron_temperature` | `temperature` | temperature, critical_temperature, coolant_inlet_temperature |
| `surface_temperature` | — | (none) |
| `major_radius_extent` | — | major_radius, minor_radius, larmor_radius |
| `vertical_extent` | — | (none) |
| `electron_temperature_of_divertor_target_sputtering_limit` | — | (none) |
| `steady_state_heat_flux_limit` | — | (none) |
| `angle_of_divertor_tile_outline_point` | — | (none) |
| `pulse_schedule_event_time_stamp` | — | (none) |

**Impact:** Error-sibling generation (upper/lower uncertainty) was suppressed for ALL names. No `_error_upper`, `_error_lower`, or `_error_index` companion names were created.

---

## Blocker #6 — Domain Misclassification: `pulse_schedule_event_time_stamp` (MINOR)

The name `pulse_schedule_event_time_stamp` (score=0.31, tier="poor") was classified as `divertor_physics` but is semantically a generic pulse-schedule concept. It was created at `2026-04-24T22:12` (prior run) and re-composed during this cycle, absorbing $0.5169 in compose costs across 3 batches (each batch contained the time_stamp path paired with a divertor-specific path).

---

## Cost Breakdown

| Phase | Actual Cost | Budget Allocation | Ratio |
|-------|------------|-------------------|-------|
| generate (compose) | $1.3547 | $0.60 (30%) | **2.26×** |
| enrich (document) | $0.1734 | $0.50 (25%) | 0.35× |
| review_names | $0.0000 | $0.2124 (adaptive 45% of remaining) | **0.00×** |
| review_docs | $0.0000 | $0.0 (gated) | — |
| regen | $0.0000 | $0.0 (skipped) | — |
| **Total** | **$1.5281** | **$2.00** | **0.76×** |

Paradox: 76% of budget was spent, but only compose ran LLM calls. The other $0.47 was stranded — too small for any phase to use.

---

## The "100% Throughput" Problem

To get all N items through compose → enrich → review with a score persisted:

### Current math for `divertor_physics` (10 DD paths)

| Phase | Min Cost | Explanation |
|-------|----------|-------------|
| Compose (40 batches × ~$0.15) | $6.00 | Fixed ~39K token prompt per batch |
| Enrich (1 batch × opus) | $0.17 | Scales linearly, efficient |
| Review (3 models × 5 batches) | $0.75 | ~$0.05/name × 3 models × 1.5× |
| **Total** | **$6.92** | For just 10 DD paths |

**The $2 budget can fund at most 13 of 40 compose batches.** Even with a generous budget, review still needs at least $0.225 per batch.

### Shortest Path to 100% Throughput

1. **[CRITICAL] Reduce compose prompt size from 39K to ~8K tokens.** The prompt is 98% input, 2% output. The system prompt, examples, grammar rules, and existing-name catalog dominate. Options:
   - Strip or summarize the existing-name catalog (likely the biggest contributor)
   - Use a smaller/cheaper model for compose (currently claude-sonnet-4.6 at $3/M input)
   - Batch more items together (90% singletons is wasteful when each batch pays the same ~39K prompt tax)

2. **[CRITICAL] Fix budget split.** The 30/25/15/15/15 split is wrong when compose costs $0.15/batch. For a 40-batch domain, compose alone needs 30% × $2 = $0.60 but actually requires $6.00. Either:
   - Raise cost-limit to $8+ for small domains
   - Or restructure batching to have ≤6 large batches instead of 40 singletons

3. **[MODERATE] Fix review reservation formula.** `worst_case = n_names × $0.05 × 3 × 1.5 = $0.225` per 1-name batch is too conservative. Actual review cost per name is closer to $0.05 per model call. The 1.5× retry multiplier, applied across all 3 models, inflates the reservation by 4.5× vs single-model expectation.

4. **[MODERATE] Fix invariant violation false positive.** The check `total_cost < budget * 0.5` misclassifies reservation failures as "not budget-exhausted". Add a case for `budget < min_reservation_per_batch`.

---

## Cross-References

| Sibling R&D | Overlap with this trace |
|-------------|------------------------|
| `rd1-budget-trace` | Compose prompt bloat (39K tokens) and budget split (30/25/15/15/15) are the root cause of all downstream failures. |
| `rd2-reviewer-pattern` | Review never ran because no budget reached it. The 3-model quorum reservation ($0.225/name) is too high for residual budgets, but the real fix is in compose. |
| `rd3-dd-classifier` | ISN vocab gaps (electron_temperature, surface_temperature, vertical_extent, etc.) and domain misclassification (pulse_schedule_event_time_stamp in divertor_physics). |

---

## Summary

**0 of 8 names completed the full pipeline with a review score.** The single root cause is compose prompt cost: a fixed ~39K token prompt per batch makes each batch cost $0.15 regardless of content size. With 40 singleton batches, compose alone requires $6 — triple the total $2 budget. After compose exhausts the budget at batch 9, every downstream phase (review_names, review_docs, regen) is starved. The enrich phase succeeds (efficient opus-based batch), but its output is wasted because no review can follow.

**Fix priority:** Reduce compose prompt tokens from 39K → 8K (via catalog pruning, template compression, or larger batches) to bring per-batch cost from $0.15 to ~$0.03. This single fix would make a $2 budget sufficient for the full pipeline.
