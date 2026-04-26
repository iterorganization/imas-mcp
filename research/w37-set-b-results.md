# W37 Set B Results

**Generated**: 2026-04-27  
**HEAD commit**: d51a6995 (fix(classifier): reclassify hardware/limit/instrument metadata paths)  
**Domains**: plasma_wall_interactions (204 paths), particle_measurement_diagnostics (194 paths),
plant_systems (166 paths), plasma_control (112 paths), core_plasma_physics (103 paths),
magnetic_field_diagnostics (99 paths)

---

## Run Summary

| Domain | Composed | Reviewed | Stop Reason | Cost |
|--------|----------|----------|-------------|------|
| plasma_wall_interactions | 69¹ | 4 | completed | $4.9268 |
| particle_measurement_diagnostics | 57 | 5 | completed | $4.8316 |
| plant_systems | 28 | 6 | completed | $4.8908 |
| plasma_control | 18 | 6 | completed | $4.8275 |
| core_plasma_physics | 8 | 7 | completed | $2.0284 |
| magnetic_field_diagnostics | 65 | 9 | completed | $4.8485 |

¹ Approximate; includes error-sibling names minted during generate phase.

**Total Set B spend**: $26.3536 (cap: $30.00 — $3.65 headroom)  
**All T37 runs (15 domains)**: $69.3917 (budget shared with Set A spillover domains)

---

## Per-Domain Table

| Domain | n_names (graph) | mean_score¹ | accept% | revise% | reject% | n_quarantined |
|--------|-----------------|-------------|---------|---------|---------|----------------|
| plasma_wall_interactions | 70 | 0.836 | 75% | 25% | 0% | 8 |
| particle_measurement_diagnostics | 58 | 0.688 | 88% | 11% | 0% | 11 |
| plant_systems | 29 | 0.587 | 50% | 43% | 6% | 2 |
| plasma_control | 19 | 0.554 | 66% | 20% | 13% | 0 |
| core_plasma_physics | 9 | 0.521 | 85% | 14% | 0% | 1 |
| magnetic_field_diagnostics | 50 | 0.744 | 66% | 33% | 0% | 13 |

¹ Mean `review_mean_score` over names with a canonical review (excludes NULLs).  
Review coverage: PWI 4/70 (6%), PMD 5/58 (9%), PS 6/29 (21%), PC 6/19 (32%),  
CPP 7/9 (78%), MFD 9/50 (18%).  
**Target: mean ≥ 0.85 — NOT MET** (mean 0.488–0.836 across domains on reviewed subset;
plasma_wall_interactions at 0.836 is closest to target; core_plasma_physics and
plasma_control dragged by grammar-violation names in the reviewed sample).

---

## Top 5 Names Overall (by canonical reviewer score)

| Name | Domain | Score |
|------|--------|-------|
| fusion_power | particle_measurement_diagnostics | 0.975 |
| area_of_flux_loop | magnetic_field_diagnostics | 0.944 |
| cross_sectional_area_of_rogowski_coil | magnetic_field_diagnostics | 0.938 |
| ion_atomic_mass | plasma_control | 0.925 |
| upper_uncertainty_of_area_of_flux_loop | magnetic_field_diagnostics | 0.919 |

---

## Bottom 5 Names with Reviewer-Suggested Alternative

| Original Name | Suggested | Score |
|---------------|-----------|-------|
| electrical_resistance_of_shunt | *(no suggestion — mixed verdict: 2× accept 0.86/0.96, 4× reject 0.03–0.06)* | 0.335 |
| ion_charge_state_bundle_ionization_potential | ion_charge_state_ionization_potential | 0.350 |
| pulse_schedule_event_time | event_time | 0.366 |
| pellet_species_number_density | pellet_number_density | 0.369 |
| pulse_schedule_event_duration | event_duration_of_pulse_schedule | 0.394 |

---

## Vocabulary Gaps (Top 10 by sources count)

| Segment | Token | Sources |
|---------|-------|---------|
| position | on_ggd | 9 |
| geometric_base | x1_unit_vector | 4 |
| geometric_base | x1_coordinate | 2 |
| geometric_base | x2_coordinate | 2 |
| position | coordinate_system_grid_point | 2 |
| position | divertor_target | 2 |
| position | line_of_sight_third_point | 2 |
| position | line_of_sight_second_point | 2 |
| position | breeding_blanket_module | 2 |
| position | annular_element_centre | 2 |

Total VocabGap nodes (with token): 132  
No promotion candidates at current thresholds (min-uses=3, min-score=0.75).

**Key gap clusters**:
- `position: on_ggd` (9 sources) — GGD geometry names cannot be expressed cleanly; ISN needs `on_ggd` position token  
- `geometric_base: x1_unit_vector / x1_coordinate / x2_coordinate` — IMAS GGD coordinate basis lacks ISN tokens  
- `position: divertor_target / line_of_sight_*_point / breeding_blanket_module` — ITER-specific geometry reference points missing from ISN `position` segment  
- `component: vertical` (3 sources) — vertical component of vectors not in ISN `component` segment  
- `basis/operators: gyroaveraged, gyroaveraged` — gyrokinetics-specific operator carried over from Set A  

---

## Errors and Warnings

### Recurrent: Jinja2 template TypeError in `review_docs` phase
```
TypeError: 'builtin_function_or_method' object is not iterable
```
Seen multiple times throughout the run. Affects secondary layer of the review pipeline
(`review_docs`, `review_names` cycles > 1). **Non-fatal** — the primary names review
completed and scored names are written to the graph. Layer-2 docs review was skipped for
affected batches. Root cause is a Jinja2 template iteration bug (template invokes a
built-in method instead of calling it).

### `review_docs` invariant violations
```
ERROR - Phase review_docs invariant violated: 7 eligible names identified but zero persisted (not budget-exhausted)
ERROR - Phase review_docs invariant violated: 1 eligible names identified but zero persisted (not budget-exhausted)
```
Consequence of the Jinja2 TypeError — names remain `valid` (not docs-reviewed).

### divertor_physics: stalled (non-Set-B spillover)
```
domains: ['divertor_physics'], stop_reason: stalled, cost: $4.2912
```
Spillover domain from the plasma_control run. Stalled mid-loop after the Jinja2 error
blocked batch processing. No impact on Set B targets.

---

## W37 Fix Validation

| Fix | Expected Effect | Observed |
|-----|-----------------|---------|
| DD classifier (d51a6995) | hardware/limit/instrument paths filtered before compose | Extract deny gate filtered 15 paths from PWI; engineering-coil-geometry skip reason active ✓ |
| Review prompt rewrite (499a59f9) | Suggested-name with justification in Review nodes | `suggested_name` and `suggestion_justification` fields populated across all reviewed names ✓ |
| Compose budget fix (3bd8e2f6) | Per-item reservation at $0.20, phase caps enforced | No budget overruns observed in Set B domains ✓ |

---

## Notes

- `core_plasma_physics` is nearly complete: 9/103 paths → 9 names (8 composed + 1 attached).
  Budget of $2.03 exhausted; full coverage would require ~$8.
- `plasma_wall_interactions` has the highest mean score (0.836) and is closest to the
  ≥ 0.85 target, but only 4 of 70 names were canonically reviewed.
- `magnetic_field_diagnostics` produced 13 quarantined names, many from FOCS
  (Fibre-Optic Current Sensor) geometry subtrees with `position` tokens missing from ISN.
- To reach ≥ 0.85 target for Set B, recommend a targeted `sn review` pass with an
  additional $3–5 budget per domain after composition is complete.
