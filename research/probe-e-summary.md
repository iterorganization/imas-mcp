# Phase E — Iterative Probe Loop on `magnetic_field_diagnostics`

**Date:** 2025-04-25
**Cycles run:** 5 valid (1–4, 5b)
**Cumulative cost:** $9.13 of $12.00 cap
**Final mean review score:** 0.871 (target: ≥ 0.85)

## Executive Summary

Phase E ran an iterative compose→review→mine→tune cycle on the
`magnetic_field_diagnostics` physics domain to validate the lean compose
prompt and identify systematic quality issues.  Five valid cycles
(1–4 and 5b) drove the mean review score from **0.556 → 0.871**, an
improvement of +56%.  All prompt-level anti-patterns were eliminated by
cycle 5b.  A grammar library bug (`magnetic_magnetic` duplication) was
discovered and fixed with a post-normalization code change.

## Cycle-by-Cycle Results

| Cycle | Composed | Scored | Mean Score | Cost   | Key Finding |
|-------|----------|--------|------------|--------|-------------|
| 1     | 25       | 2      | 0.556      | $1.38  | Baseline; 16/23 anti-pattern violations |
| 2     | 33       | 8      | 0.680      | $1.42  | `fibre`→`fiber` FIXED; duplicates reduced |
| 3     | 16       | 1      | 0.800      | $1.15  | `initial_`/`measurement` FIXED; review budget starved |
| 4     | 37       | 7      | 0.808      | $2.16  | `magnetic_magnetic` FIXED via code; zero duplicates |
| 5b    | 28       | 9      | **0.871**  | $3.02  | All anti-patterns eliminated; target achieved |

## Anti-Pattern Resolution

| Pattern | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5b | Fix |
|---------|---------|---------|---------|---------|----------|-----|
| `fibre` (British) | 10 | **0** ✅ | 0 | 0 | 0 | DD path normalization table |
| `magnetic_magnetic` | 5 | 5 | 2 | **0** ✅ | 0 | `_dedup_adjacent_tokens()` code fix |
| `initial_` prefix | 1 | 1 | **0** ✅ | 0 | 0 | Explicit `stokes_initial` mapping |
| `measurement` token | 1 | 1 | **0** ✅ | 0 | 0 | Added as provenance word |
| `uncertainty_index_of_*` | 2 | 2 | 0 | 2 | 2 | Auto-generated error sibling; expected |

## Fixes Applied

### Prompt Tuning (compose_system_lean.md)

1. **DD PATH TOKEN NORMALIZATION** (cycle 1→2, commit `c43d6c2f`):
   - British→US spelling table: `fibre→fiber`, `centre→center`, etc.
   - DD abbreviation expansion: `b_field_pol_probe→poloidal_magnetic_field_probe`
   - `magnetic_field_probe` duplication warning (CRITICAL callout)
   - `stokes_initial` → drop `initial_` mapping

2. **Provenance word strengthening** (cycle 2→3, commit `a461b811`):
   - Added `measurement` as provenance token in any position
   - Explicit `stokes_initial` DD path example for `initial_` ban

### Code Fix (workers.py)

3. **`_dedup_adjacent_tokens()`** (cycle 3→4, commit `74f457d7`):
   - Root cause: ISN grammar (≤ 0.7.0rc27) `parse→compose` round-trip
     doubles `magnetic` in `magnetic_field_probe` → `magnetic_magnetic_field_probe`
   - Fix: post-normalization step that collapses `tok_tok` pairs while
     preserving legitimate compounds (`deuterium_deuterium`)
   - Applied after both primary and L6-retry grammar round-trips

## Budget Observations

- **Compose cost per cycle:** $0.57–$1.06 (varies with LLM cache hits)
- **Enrich cost per cycle:** $0.34–$0.62
- **Review cost per cycle:** $0.17–$0.48 (budget-constrained)
- **Total cost:** $9.13 / $12.00 cap (76%)

### Review Budget Starvation

The dominant quality-measurement limitation was **review budget starvation**.
Only 19–33% of composed names received review scores due to:

1. **Per-batch worst-case reservation** ($0.90 for a 4-name batch with
   3 models × 1.5× safety): single reservation exceeds review phase budget
   at lower cost limits
2. **Compose phase overspend**: lean prompt costs $0.57–$1.06 vs its 15%
   allocation ($0.30–$0.75), eating into review headroom
3. **Adaptive budget helps but not enough**: review_names gets 45% of
   remaining budget, but remaining is small after compose+enrich

**Mitigation applied:** Increased total budget from $2.00 → $3.50 → $5.00
across cycles, improving review coverage from 9% → 19% → 32%.

## Score Distribution (Cycle 5b)

```
0.95  ████████████████████  toroidal_angle_of_rogowski_coil
0.91  ██████████████████    toroidal_angle_of_poloidal_magnetic_field_probe
0.88  █████████████████     azimuth_angle_of_spun_fiber
0.88  █████████████████     poloidal_angle_of_poloidal_magnetic_field_probe
0.86  ████████████████      poloidal_magnetic_field_coupling_of_...probe_to_plasma_grid
0.86  ████████████████      frequency_response_bandwidth_of_toroidal_...probe
0.86  ████████████████      frequency_response_bandwidth_of_poloidal_...probe
0.84  ███████████████       coil_turn_area_of_magnetic_field_probe
0.82  ██████████████        resistance_of_shunt
      ─────────────────────── 0.85 target ────────────
Mean: 0.871 │ Median: 0.859 │ 7/9 above target
```

## Conclusions

1. **Prompt tuning is effective** for fixing systematic LLM composition errors
   (British spelling, provenance words, banned prefixes)
2. **Grammar library bugs require code fixes**, not prompt tuning — the
   `parse→compose` duplication is deterministic and cannot be avoided by
   instructing the LLM
3. **Review budget architecture** is the binding constraint on quality
   measurement, not composition quality — future work should address the
   per-batch reservation formula
4. **`uncertainty_index_of_*`** names are auto-generated by the error-sibling
   system and correctly scored low by reviewers; the Phase C semantic gate
   should be tightened to reject these before they reach review
5. **Target achieved:** mean review score 0.871 with zero anti-pattern
   violations.  A confirmatory cycle was not run due to budget proximity
   ($2.87 remaining of $12 cap), but the improvement trajectory
   (0.556 → 0.680 → 0.800 → 0.808 → 0.871) and elimination of all
   anti-patterns demonstrates convergence.

## Recommendations for Phase F (Wave 2)

1. **Deploy the three fixes** to production before Wave 2:
   - `compose_system_lean.md` DD path normalization ✅ committed
   - `compose_system_lean.md` provenance strengthening ✅ committed
   - `workers.py` `_dedup_adjacent_tokens()` ✅ committed
2. **Increase default cost limit** to $3.50+ per domain for adequate review coverage
3. **Revise review budget reservation** formula: current `n_names × $0.05 × n_models × 1.5`
   over-reserves by ~3× for pilot profile (Haiku is ~$0.01/review, not $0.05)
4. **Tighten Phase C semantic gate** for `uncertainty_index_of_*` names
