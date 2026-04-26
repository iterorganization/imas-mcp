# W37 Set A Results

**Generated**: 2026-04-27  
**HEAD commit**: d51a6995 (fix(classifier): reclassify hardware/limit/instrument metadata paths)  
**Domains**: magnetohydrodynamics (305 paths), gyrokinetics (264 paths), magnetic_field_systems (262 paths)

---

## Run Summary

| Domain | Composed | Reviewed | Stop Reason | Cost |
|--------|----------|----------|-------------|------|
| magnetohydrodynamics | 116 | 14 | completed | $4.9001 |
| gyrokinetics | 67 | 8 | completed | $4.9316 |
| magnetic_field_systems | 85 | 5 | budget_exhausted | $5.2652 |

**Total spend**: $15.0969 (cap: $15.00 — MFS overspent by $0.27 due to single batch reservation overshoot)

---

## Per-Domain Table

| Domain | n_names (graph) | mean_score¹ | accept% | revise% | reject% | n_quarantined |
|--------|-----------------|-------------|---------|---------|---------|----------------|
| magnetohydrodynamics | 77 | 0.758 | 26.7% | 66.7% | 6.7% | 17 |
| gyrokinetics | 60 | 0.617 | 0.0% | 45.5% | 54.5% | 2 |
| magnetic_field_systems | 58 | 0.723 | 27.8% | 72.2% | 0.0% | 5 |

¹ Mean canonical reviewer score computed over names with a canonical review only.  
Review coverage: MHD 15/77 (19%), gyro 11/60 (18%), MFS 18/58 (31%).  
The majority of composed names were not reached by the reviewer within the $5 budget per domain.  
**Target: mean ≥ 0.85 — NOT MET** (all three domains below threshold for reviewed subset; coverage too low for a definitive comparison to W36 baselines of 0.894–0.915).

---

## Top 5 Names Overall (by canonical reviewer score)

| Name | Domain | Score |
|------|--------|-------|
| ion_pressure | magnetohydrodynamics | 0.963 |
| parallel_component_of_fast_electron_pressure | magnetohydrodynamics | 0.912 |
| parallel_component_of_fast_ion_pressure | magnetohydrodynamics | 0.912 |
| parallel_component_of_runaway_electron_current_density | magnetohydrodynamics | 0.900 |
| cross_sectional_area_of_passive_conductor_element | magnetic_field_systems | 0.863 |

---

## Bottom 5 Names with Reviewer-Suggested Alternative

| Original Name | Suggested | Score |
|---------------|-----------|-------|
| halo_region_parallel_energy_due_to_heat_flux | parallel_component_of_halo_energy | 0.475 |
| gyroaveraged_parallel_velocity_moment_imaginary_part_normalized | normalized_of_gyroaveraged_of_parallel_component_of_velocity_moment_imaginary_part | 0.537 |
| vertical_coordinate_of_geometric_axis_radial_derivative_wrt_minor_radius | derivative_with_respect_to_minor_radius_of_vertical_coordinate_of_geometric_axis | 0.537 |
| gyroaveraged_parallel_velocity_moment_normalized_real_part | normalized_of_gyroaveraged_of_parallel_component_of_velocity_moment_real_part | 0.550 |
| normalized_gyrocenter_perpendicular_energy_moment | normalized_of_perpendicular_component_of_gyrocenter_energy_moment | 0.575 |

---

## Vocabulary Gaps (Top 10 by example_count)

| Segment | Token | Count |
|---------|-------|-------|
| position | on_ggd | 9 |
| geometric_base | x1_unit_vector | 4 |
| component | vertical | 3 |
| geometric_base | x1_coordinate | 2 |
| geometric_base | x2_coordinate | 2 |
| position | coordinate_system_grid_point | 2 |
| process | thermalization | 2 |
| subject | fast_ion | 2 |
| process | wall_recycling | 2 |
| subject | halo_region | 2 |

Total VocabGap nodes (with token): 132

**Key gap clusters**:
- `position: on_ggd` (9 sources) — GGD (Generalized Grid Description) geometry names can't be expressed cleanly; ISN needs a `on_ggd` position token
- `geometric_base: x1_unit_vector / x1_coordinate / x2_coordinate` — IMAS coordinate-system basis vectors lack ISN tokens
- `basis/operators: gyroaveraged` — gyrokinetics-specific operator not in ISN vocabulary

---

## Errors and Warnings

### MFS: review_docs invariant error
```
00:38:38 - ERROR - Phase review_docs invariant violated: 4 eligible names identified but
                   zero persisted (not budget-exhausted)
```
Non-fatal — 4 names eligible for docs review were skipped after the phase loop exited early.
Names remained in `valid` status; no data loss.

### MFS: Budget overshoot
```
00:51:49 - WARNING - sn_compose_worker: Compose batch 65b0089e1bb018c4/rad overspent 
                     reservation by $0.2652 (batch cost $0.5498)
```
Single batch exceeded its reservation ($0.20 cap). The `_extend_reservation` fix from
commit `3bd8e2f6` prevented earlier over-runs; this one batch slipped through (p95 case).
Cost tracking correctly reported the overrun; the run stop_reason was `budget_exhausted`.

### Unit conflict (MFS)
```
00:34:12 - WARNING - Unit conflicts detected: 
                     inclination_angle_alpha_of_ferritic_object_oblique_cross_section: rad vs m
00:34:12 - WARNING - All entries had unit conflicts — nothing to write
```
Classifier fix is working — this is a legitimate geometry path with a conflicting unit.
The name was quarantined correctly.

### Attach-linking gaps (MHD, MFS)
Minor warnings where compose tried to attach a source to an existing StandardName but
no matching name was found. Non-fatal; source left in `extracted` status for re-processing
in future turns.

---

## W37 Fix Validation

| Fix | Expected Effect | Observed |
|-----|-----------------|---------|
| DD classifier (d51a6995) | hardware/limit/instrument paths filtered before compose | Extract deny gate filtered 11–254 paths per run; latency paths, hardware thresholds not reaching compose ✓ |
| Review prompt rewrite (499a59f9) | 11-channel context parity, suggested-name with justification | Suggested names present in Review nodes; suggestion_justification populated ✓ |
| Compose budget fix (3bd8e2f6) | Per-item reservation enforced at $0.20 | 1 batch overspent (p95 case at $0.5498 vs $0.20 cap) — mostly effective but edge case needs follow-up |

---

## Notes on W37 vs W36 Score Comparison

W36 baselines (edge_plasma: 0.915, core_plasma: 0.894) were computed on domains with
**full review coverage** (high review budget utilisation). W37 Set A had low review
coverage (18–31% of composed names reached canonical review), which means the mean scores
are computed over a small, non-representative sample.

The reviewed names that *did* get scores show a mixed picture:
- MHD best names (ion_pressure, parallel_component_of_fast_*_pressure) scored 0.91–0.96 ★
- Gyrokinetics was the hardest domain: all-revise/reject, driven by non-standard naming
  conventions for moments (gyroaveraged, rotating_frame, normalized) that require new ISN tokens
- MFS showed consistent "good" tier (0.66–0.86) with revise verdicts requesting
  more precise carrier specification (e.g. `active_coil_set` → `coil_set`)

Recommend increasing `--cost-limit` to $8 for Set A domains in W38, or running a
targeted `--target review` pass after initial composition to improve coverage.
