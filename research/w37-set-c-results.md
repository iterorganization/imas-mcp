# W37 Set C Results

**HEAD commit:** d51a6995  
**Total spend:** ~$44.73 (11 active domains; 5 domains $0.00 — work already exhausted)  
**Run date:** 2026-04-27

## Per-Domain Table

| domain | n_names | mean_reviewer_score | accept% | revise% | reject% | n_quarantined |
|--------|---------|---------------------|---------|---------|---------|---------------|
| fast_particles | 49 | 0.7789 | 59% | 18% | 24% | 18 |
| neutronics | 22 | 0.7562 | 0% | 100% | 0% | 1 |
| computational_workflow | 26 | 0.7812 | 12% | 88% | 0% | 0 |
| divertor_physics | 14 | 0.8620 | 65% | 35% | 0% | 0 |
| structural_components | 48 | 0.7307 | 0% | 100% | 0% | 3 |
| waves | 39 | 0.7662 | 60% | 40% | 0% | 0 |
| spectroscopy | 22 | 0.6896 | 14% | 86% | 0% | 0 |
| machine_operations | 22 | 0.7983 | 18% | 82% | 0% | 2 |
| data_management | 18 | 0.6817 | 7% | 93% | 0% | 0 |
| mechanical_measurement_diagnostics | 20 | 0.6833 | 33% | 47% | 20% | 0 |
| plasma_measurement_diagnostics | 7 | 0.7812 | 25% | 75% | 0% | 0 |
| fueling | 0 | — | — | — | — | 0 |
| current_drive | 0 | — | — | — | — | 0 |
| runaway_electrons | 0 | — | — | — | — | 0 |
| general | 0 | — | — | — | — | 0 |
| plasma_initiation | 0 | — | — | — | — | 0 |
| **TOTAL (Set C)** | **287** | **0.7523** | **~24%** | **~64%** | **~12%** | **24** |

> Note: fueling, current_drive, runaway_electrons, general, plasma_initiation all returned $0.00 spend — all StandardNameSource nodes were already in terminal state from prior turns.

## Run Summary (cost per domain)

| domain | stop_reason | cost_spent |
|--------|-------------|------------|
| fast_particles | completed | $4.917 |
| neutronics | completed | $4.862 |
| computational_workflow | completed | $4.885 |
| divertor_physics | stalled | $4.291 |
| structural_components | completed | $4.803 |
| waves | stalled | $4.183 |
| spectroscopy | stalled | $4.490 |
| machine_operations | completed | $4.181 |
| data_management | budget_exhausted | $4.962 |
| mechanical_measurement_diagnostics | completed | $4.877 |
| plasma_measurement_diagnostics | completed | $2.286 |
| fueling | completed | $0.000 |
| current_drive | completed | $0.000 |
| runaway_electrons | completed | $0.000 |
| general | completed | $0.000 |
| plasma_initiation | completed | $0.000 |
| **TOTAL** | | **~$48.74** |

## Top 5 Names (by reviewer_score_name)

| name | domain | score |
|------|--------|-------|
| toroidal_component_of_wave_vector | waves | 0.95 |
| parallel_component_of_ion_current_density | plasma_measurement_diagnostics | 0.95 |
| surface_area_of_divertor_tile | divertor_physics | 0.95 |
| toroidal_component_of_current_density | fast_particles | 0.9375 |
| ion_atomic_mass | machine_operations | 0.925 |

## Bottom 5 Names with Suggested Alternatives

| name | domain | score | suggested_name |
|------|--------|-------|----------------|
| x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound | spectroscopy | 0.5125 | photon_energy_lower_bound |
| x_ray_crystal_spectrometer_pixel_photon_energy_upper_bound | spectroscopy | 0.5188 | photon_energy_upper_bound |
| z_coordinate_of_sensor_direction_unit_vector | mechanical_measurement_diagnostics | 0.5188 | z_component_of_direction_unit_vector |
| toroidal_collisional_torque_density_on_fast_electron | fast_particles | 0.525 | toroidal_component_of_fast_electron_torque_density_due_to_collisions |
| toroidal_collisional_torque_density_on_fast_ion | fast_particles | 0.525 | toroidal_component_of_fast_ion_torque_density_due_to_collisions |

## Top 10 Vocab Gaps

| segment | token | sources |
|---------|-------|---------|
| position | on_ggd | 9 |
| geometric_base | x1_unit_vector | 4 |
| component | vertical | 3 |
| geometric_base | x1_coordinate | 2 |
| geometric_base | x2_coordinate | 2 |
| position | coordinate_system_grid_point | 2 |
| position | divertor_target | 2 |
| position | divertor_tile_surface_outline_point | 2 |
| position | line_of_sight_third_point | 2 |
| position | line_of_sight_second_point | 2 |

## Errors from Logs

- No crashes or exceptions. 3 domains stalled (divertor_physics, waves, spectroscopy), 1 hit budget_exhausted (data_management) — all expected.
- 5 small domains (fueling, current_drive, runaway_electrons, general, plasma_initiation) had no pending work and completed immediately at $0.00.

## Notes

- **fast_particles** has 18 quarantined names — highest quarantine rate (37%) in Set C. These are mainly toroidal torque/current density variants with instrument-specific prefixes rejected by the grammar.
- **divertor_physics** is the only domain meeting the ≥0.85 target at 0.862 mean score.
- **data_management** and **mechanical_measurement_diagnostics** are the weakest domains (0.68–0.68), with heavy revise/reject rates and time-epoch naming issues.
- Mean Set C score is **0.7523** — below the 0.85 target, suggesting W38 should focus revision passes on spectroscopy, data_management, mechanical_measurement_diagnostics, and fast_particles (quarantine clearance).
