# Wave 9 Synthesis (validate W8 fixes)

## W9A — Names rotation ($12.6 / $15)

| Domain | W9 mean | W7 mean | Δ |
|--------|--------:|--------:|--:|
| gyrokinetics | 0.607 | 0.649 | -0.042 |
| plasma_wall_interactions | 0.631 | 0.733 | -0.102 |
| particle_measurement_diagnostics | 0.596 | 0.658 | -0.062 |

**Apparent regression is artifact, not real.** Two wiring bugs invalidate the comparison:

1. **`--target names` doesn't filter enrich** — Opus-4.6 ran docs-enrichment on ~50-200 SNs/domain consuming 50-60% of $5 cap. W7A had no such leak (used the loop, not `--single-pass`). Scores written by enrich/docs pipeline use a different rubric than dedicated K3 names review.
2. **`pipeline.py:796` silent discard** — `lease.charge()` exceeds Opus reservation, discards results while billing cost. `charge_or_extend()` already exists in `budget.py` but isn't wired.
3. **`review_docs invariant violated`** — emitted across all domains: "N eligible names identified but zero persisted (not budget-exhausted)".

**W8 prompt fixes ARE working**: 51 SNs got `reviewer_suggested_name` populated, confirming the revised-name bug fix. Residual anti-patterns are NEW classes not yet banned:
- `_flag` suffix (DD computational metadata): `gyrokinetic_model_nonlinear_run_flag` (0.31)
- IDS-tree prefix leak: `gyrokinetic_model_parallel_vector_potential_flag` (0.33)
- Compound coefficient absorption: `sputtering_coefficient_incident_energy_grid` (0.29)
- Free-form GGD: `grid_object_geometry` (0.21)

Despite hardening, `_from_/_to_` still appears (PWI). Ban needs more prominent placement.

## W9B — Docs rotation ($4.43 / $15)

| Domain | W9 mean | W7 mean | Δ | Recovered |
|--------|--------:|--------:|--:|----------:|
| magnetic_field_diagnostics | 0.795 | 0.758 | +0.037 | +2 |
| machine_operations | 0.774 | 0.717 | +0.057 | +8 |
| turbulence | 0.774 | 0.769 | +0.005 | +4 |
| Combined | 0.784 | 0.749 | +0.035 | +14 |

**H1 confirmed**: 14/18 W7-blocked SNs now reviewed (revised-name fix works).
**H2 not confirmed**: +3.5pp lift, but 0.784 still under 0.8 target.

**Persistent outlier class**: `uncertainty_index_of_*` SNs score 0.65-0.72 across all domains. PR-1 boilerplate is being followed but scores still drag. May need to gate them out of review entirely (they're DD metadata indices, not real quantities).

## VocabGap accumulation (W7+W9A = 33 unique tokens)
- Position (16): from_wall, from_plasma, outboard_midplane, inner_divertor_target, outer_divertor_target, divertor_target, limiter_outline_point, mobile_unit_outline_point, vessel_outline_point, diagnostic_component_center, detector_aperture, +5 from W9A
- Subject (2): gyroaveraged, +1
- Geometric_base (2): x1_width, x2_width
- Coordinate_axes: annular, vertical_coordinate_of
- Component: cartesian_x
- New (W9A): shearing_rate, plus 5 unenumerated

## Other surfaced bugs (cosmetic / display)
- `sn.py:3502` `KeyError: 'total_actual'` — BudgetManager returns `total_spent` not `total_actual`
- Turbulence "Scored: 0" display while graph has scores — concurrency artifact between in-memory state and per-batch persist
- Gyrokinetics `--single-pass` + enrich re-run: bug or surprising behavior

## Wave 10 dispatch
- **w10-budget-fix**: wire `charge_or_extend()`, fix `--target names` enrich-filter, debug `review_docs invariant` violation, fix `KeyError: total_actual`
- **w10-prompts-2**: ban `_flag` suffix, generic IDS-tree prefix ban, stronger `_from_/_to_` placement, gate `uncertainty_index` SNs from review (or hard-skip via filter)
- **w10-isn-vocab**: file ISN PR with 33-token vocab batch
- **w11**: small validation rotation on ONE domain ($5) to confirm fixes
