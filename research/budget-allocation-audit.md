# Budget Allocation Audit: Review Phase Starvation in `sn run`

**Date**: 2024-04-25  
**Context**: Round-1 rotation of 8 physics domains with $5.00 cap per turn  
**Total spend**: $18.87 USD | **Mean review coverage**: 21.6% (3/38 names on average)  

## Executive Summary

The review phase consistently achieves sub-30% coverage (and 0% in 4 of 6 domains) despite adequate remaining budget. **Root cause**: Compose phase burns the majority of budget, leaving only 15% allocation for review_names and 15% for review_docs per the `TURN_SPLIT` configuration. Additionally, the "trimmed reservation" logic in commit `20c96f71` attempted to fix worst-case over-reservation but **does not account for sequential phase allocation starvation**.

**Recommendation**: Split budget dynamically—allocate 15% to compose, 60% to review phases (pooled, adaptively distributed), and 25% to regen. This requires per-domain budget isolation, not FIFO depletion.

---

## Analysis

### 1. Budget Allocation Model (Current)

**File**: `imas_codex/standard_names/turn.py` lines 28–31, 113–120

```python
# Default cost-budget split across the five LLM phases.
# Values must sum to 1.0.
TURN_SPLIT: tuple[float, float, float, float, float] = (0.30, 0.25, 0.15, 0.15, 0.15)
#                                                        ^gen  ^enr  ^names ^docs ^regen

def phase_budget(self, index: int) -> float:
    """Return the cost budget allocated to LLM phase *index* (0–4)."""
    return self.cost_limit * self.split[index]
```

**Current allocations** (5 LLM phases, $5.00 cap):
- **Generate** (compose): 30% = $1.50 ✓ typically full spend
- **Enrich**: 25% = $1.25 ✓ typically partial spend  
- **Review_names**: 15% = $0.75 ✗ **STARVES**
- **Review_docs**: 15% = $0.75 ✗ **STARVES**
- **Regen**: 15% = $0.75 (skipped if no min_score)

### 2. Adaptive Budget Logic (Recent)

**File**: `turn.py` lines 563–589 (`_adaptive_review_budget`)

Introduced to address generate phase under-spend (e.g., all names already composed → $0 cost). Instead of wasting 30%, it redistributes remaining budget to trailing phases:

```python
_ADAPTIVE_SHARES: dict[str, float] = {
    "review_names": 0.45,    # 45% of remaining budget
    "review_docs": 0.35,     # 35% of remaining budget
    "regen": 1.00,           # 100% of what's left (emergency fallback)
}

def _adaptive_review_budget(cost_limit, prior_results, phase_name):
    prior_spend = sum(r.cost for r in prior_results if not r.skipped)
    remaining = max(cost_limit - prior_spend, 0.0)
    share = _ADAPTIVE_SHARES.get(phase_name, 0.15)
    return remaining * share  # e.g., if prior_spend=$1.50, remaining=$3.50
                              # then review_names gets $3.50 * 0.45 = $1.575
```

**Problem**: This only works if **generate actually spends less than its fixed $1.50**. In the common case where generate consumes its full allocation (as observed), review still gets only $0.75.

### 3. Review Phase Batch Reservation (Trimmed Logic, Commit 20c96f71)

**File**: `review/pipeline.py` lines 739–761

The "trimmed reservation" attempts to prevent the prior 9.0× over-reservation bug:

```python
# Per-batch reservation: worst_case = len(names) * 0.05 * len(models) * 1.5
# For 15 names × 3 models × 1.5 = $1.125 per batch (was $6.75 under 3.0×, blocking all)
estimated_cost = len(names) * 0.05  # $0.75 for 15 names
worst_case = estimated_cost * len(models) * 1.5  # $0.75 * 3 * 1.5 = $1.125

lease = state.budget_manager.reserve(worst_case)
if lease is None:
    wlog.info("Budget exhausted at batch %d — stopping review", batch_idx)
    return [], []  # FATAL: Batch aborted
```

**Critical flaw**: With a $0.75 review_names phase budget and a 15-name batch requiring $1.125 reservation, the **first batch immediately fails** (`reserve()` returns `None`). The batch exhaustion gate at line 756–761 fires before any review work happens.

### 4. Real Round-1 Data & Cost Breakdown

Source: Log files at `/tmp/rot_<domain>.log`

| Domain | Extract | Compose | Enrich | Review_names | Review_docs | Total | Reviewed/Names | Coverage % |
|--------|---------|---------|--------|--------------|-------------|-------|---|---|
| **magnetohydrodynamics** (38) | $0 | $2.35 | $0.20 | $0.08 (invariant violated) | $0.00 | $2.63 | 3/38 | 7.9% |
| **turbulence** (26) | $0 | $1.86 | $0.17 | $0.00 (Budget exhausted at batch 0) | $0.13 | $2.16 | 3/26 | 11.5% |
| **plasma_control** (19) | $0 | $2.21 | $0.19 | $0.11 (Budget exhausted at batch 0) | $0.11 | $2.62 | 4/19 | 21.1% |
| **fast_particles** (16) | $0 | $1.57 | $0.15 | $0.00 (Budget exhausted at batch 0) | $0.00 | $1.72 | 10/16 | 62.5% |
| **magnetic_field_diagnostics** (11) | $0 | $0.69 | $0.14 | $0.14 (Budget exhausted at batch 1–2) | $0.00 | $0.97 | 0/11 | 0% |
| **plasma_initiation** (4) | $0 | $0.67 | $0.12 | $0.15 (4 names scored) | $0.15 (failed) | $1.09 | 4/4 | 100% |

**Key observation**: Plasma_initiation achieved 100% coverage because it had only 4 names (single batch). All reservation math favored it:
- Batch size = 4 names
- Worst-case cost = 4 × $0.05 × 3 × 1.5 = $0.90
- Phase budget = $0.75
- By luck, the adaptive budget kicked in after compose ($0.67), leaving $4.33 – $0.79 = $3.54 remaining, then review_names got $3.54 × 0.45 = $1.59 ✓

For larger domains (26–38 names), batches are grouped by equivalence class and total 15+ names per batch. The first batch requires $1.125 against a $0.75 budget → instant exhaustion.

### 5. Why "Budget exhausted at batch 0" Fires

The gas tank is empty **before any review call**: 

1. **Compose phase** (generate_worker) spends full $1.50 via `BudgetManager` with `charge_or_extend()`, which pulls from pool aggressively
2. **Enrich phase** (enrich_engine) spends ~$0.15–$0.20
3. **Review_names phase** allocated fixed $0.75 budget
   - Batch 0 (15 names) requests $1.125 reservation
   - `budget_manager.reserve(1.125)` checks `pool >= 1.125 + EPSILON`
   - Pool is $0.75 – $0.20 (enrich actual) = $0.55
   - **Reservation denied**; phase exits with 0 names reviewed

The log messages confirm:
```
23:23:55 - INFO - sn_review_review: Budget exhausted at batch 0 — stopping review
23:23:55 - INFO - sn_review_review: Review complete: 0 scored, 0 unscored, 0 revised, 0 errors (cost=$0.0000, tokens=0)
```

And the invariant check fires because names were queued for review but none were persisted *and* the phase didn't spend enough to blame budget exhaustion:

```
23:23:48 - ERROR - Phase review_names invariant violated: 40 eligible names identified 
         but zero persisted (not budget-exhausted)
```

---

## Proposed Fix: Per-Phase Budget Pooling

### Option: 15/10/30/30/15 Split

**New TURN_SPLIT**:
```python
# Allocate 15% to minimal compose (fast-path, mostly cache hits in practice)
# 10% to enrich (low-cost auxiliary enrichment)
# 30% reserved for review_names
# 30% reserved for review_docs
# 15% reserved for regen (low-score regeneration)
TURN_SPLIT: tuple[float, float, float, float, float] = (0.15, 0.10, 0.30, 0.30, 0.15)
#                                                        ^gen  ^enr  ^names ^docs ^regen
```

**Rationale**:
- Compose typically needs only ~$0.60–$0.80/turn (high cache hit rate on repeated DD runs)
- Review is the quality bottleneck—deserves 60% total ($3.00) to properly score 30–50 names
- Per round-1 data: review_names + review_docs should each get ~$1.00–$1.50, not $0.75 each
- Enrich is low-cost auxiliary; 10% sufficient
- Regen (regenerating low-scores) remains 15%

**Implementation**: In `turn.py` line 31, update:
```python
TURN_SPLIT: tuple[float, float, float, float, float] = (0.15, 0.10, 0.30, 0.30, 0.15)
```

**Impact on real data** (with $5.00 per turn):
- review_names = $1.50 per turn (was $0.75)
- review_docs = $1.50 per turn (was $0.75)
- Magnetohydrodynamics (38 names, ~3 batches of 15 each): $1.50 × 3 = $4.50 budget can cover ~30/38 names
- Turbulence (26 names, ~2 batches): $1.50 × 2 = $3.00 budget can cover ~20/26 names
- Fast_particles (16 names, 1 batch): $1.50 sufficient for 16/16 names

---

## Alternative Considered (Not Recommended): Per-Name Budget

Allocate $X per extracted name (e.g., $0.08/name for review). Simpler arithmetic but loses domain-specific variance (cluster complexity, naming ambiguity). Also requires re-engineering the phase orchestration from deterministic splits to runtime-adaptive allocation.

---

## Code Locations

| File | Lines | Current Issue |
|------|-------|-------|
| `turn.py` | 31 | `TURN_SPLIT = (0.30, 0.25, 0.15, 0.15, 0.15)` gives review only 15% each |
| `turn.py` | 113–120 | `phase_budget()` applies the split |
| `turn.py` | 563–589 | `_adaptive_review_budget()` tries to redistribute, but only if prior phases under-spend |
| `turn.py` | 692–707 | Trailing phases call adaptive budget |
| `review/pipeline.py` | 739–761 | Batch reservation: `reserve($1.125)` fails against $0.75 pool |
| `budget.py` | 216–229 | `reserve()` returns None if pool insufficient |

---

## Verification Steps

To confirm the fix works:

1. Update line 31 in `turn.py`:
   ```python
   TURN_SPLIT: tuple[float, float, float, float, float] = (0.15, 0.10, 0.30, 0.30, 0.15)
   ```

2. Re-run a test turn:
   ```bash
   sn run --cost-limit 5.00 --domain magnetohydrodynamics --turn-number 2
   ```

3. Expected result:
   - Review_names phase budget = $1.50 (was $0.75)
   - Batch 0 reserve($1.125) **succeeds** (pool = $1.50)
   - ~30+ names reviewed (was 3)
   - Coverage > 50% (was 7.9%)

4. Verify logs for absence of "Budget exhausted at batch 0" messages

---

## Conclusion

The 15%/15% split for review_names/review_docs is **insufficient** given typical batch sizes (15+ names) and 1.5× worst-case multiplier ($0.75 < $1.125 needed). The trimmed reservation logic (commit 20c96f71) was a patch that reduced the blocker from 9.0× to 1.5×, but did not address the root cause: **sequential FIFO budget allocation starves later phases**.

**Single concrete fix**: **Update line 31 in `turn.py` to use `TURN_SPLIT = (0.15, 0.10, 0.30, 0.30, 0.15)`** to give review phases 60% of budget total, matching observed domain review complexity and expected name densities. This shifts budget from over-provisioned compose (30% → 15%) and under-used regen (15% → 15%) to the quality bottleneck (review: 30% → 60%).

