# Standard Name Regen Cycle — Low-Scoring Domains

**Date**: 2025-07-27
**Commit baseline**: `a60b1fcf`
**Graph baseline**: 1,406 standard names
**ISN version**: v0.7.0rc16 (56 new tokens)
**Reviewer**: Opus 4.6 (primary), cross-family diversity enabled
**Budget**: $2.00 generate + $0.50 review per domain = $17.50 total cap

## Baseline Summary

| Domain | N | Mean Score | Good (≥0.8) | Poor (<0.5) |
|--------|---|------------|-------------|-------------|
| waves | 8 | 0.083 | 0 | 1 |
| fast_particles | 47 | 0.134 | 0 | 38 |
| structural_components | 16 | 0.125 | 0 | 13 |
| electromagnetic_wave_diagnostics | 64 | 0.180 | 1 | 32 |
| turbulence | 135 | 0.244 | 7 | 54 |
| plasma_wall_interactions | 71 | 0.367 | 8 | 37 |
| gyrokinetics | 132 | 0.322 | 8 | 46 |

VocabGap nodes: 0 across all domains (pre-regen baseline).

---

## Per-Domain Results

### 1. waves

**Baseline**: n=8, mean=0.083, good=0, poor=1
**Post-regen**: n=27, mean=0.861, good=18, poor=3
**Δ**: +19 names, +0.778 mean score, +18 good, +2 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 8 | 27 | +19 |
| Mean score | 0.083 | 0.861 | +0.778 |
| Good (≥0.8) | 0 | 18 | +18 |
| Poor (<0.5) | 1 | 3 | +2 |

**Cost**: generate=$1.44, review=$0.012, total=$1.45
**VocabGap**: 0 before → 0 after (VocabGap nodes not created; ~29 token-miss warnings during generation for `wave_absorbed_power_per_toroidal_mode`, `wave_electric_field_*_per_toroidal_mode`, etc.)

**Sample names (top 5)**:
1. `wave_absorbed_power_per_toroidal_mode` — score=1.0, outstanding
2. `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode` — score=1.0, outstanding
3. `right_hand_circularly_polarized_electric_field_phase_per_toroidal_mode` — score=1.0, outstanding
4. `left_hand_circularly_polarized_electric_field_phase_per_toroidal_mode` — score=1.0, outstanding
5. `wave_driven_toroidal_current_inside_flux_surface` — score=1.0, outstanding

**Notes**: Massive improvement. Domain went from near-zero quality (0.083 mean) to outstanding (0.861 mean). 14 of 18 reviewed names scored outstanding (1.0). Remaining 3 poor-scored names are quarantined/needs_revision validation status (not yet reviewable). Grammar decomposition fields empty (expected for `--name-only` mode). Key vocab gap: `wave_absorbed_power_per_toroidal_mode` family needs ISN tokens.

### 2. fast_particles

**Baseline**: n=47, mean=0.134, good=0, poor=38
**Post-regen**: n=44, mean=0.751, good=10, poor=0
**Δ**: -3 names, +0.617 mean score, +10 good, -38 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 47 | 44 | -3 |
| Mean score | 0.134 | 0.751 | +0.617 |
| Good (≥0.8) | 0 | 10 | +10 |
| Poor (<0.5) | 38 | 0 | -38 |

**Cost**: generate=$1.92, review=$0.14, total=$2.06
**VocabGap**: 0 → 0 (token-miss warnings: `toroidal_fast_ion_torque_due_to_coulomb_collisions`, `thermal_ion_total_toroidal_torque`)

**Sample names (top 5)**:
1. `trapped_toroidal_current_density` — score=1.0, outstanding
2. `trapped_fast_ion_toroidal_current_density` — score=1.0, outstanding
3. `co_passing_fast_ion_toroidal_current_density` — score=1.0, outstanding
4. `counter_passing_fast_ion_toroidal_current_density` — score=1.0, outstanding
5. `co_passing_thermal_electron_collisional_toroidal_torque_density` — score=0.925, outstanding

**Notes**: Eliminated all 38 poor-scored names. Count dropped slightly (-3) due to better deduplication. Mean score improved dramatically from 0.134 to 0.751.

### 3. structural_components

**Baseline**: n=16, mean=0.125, good=0, poor=13
**Post-regen**: n=13, mean=0.839, good=5, poor=0
**Δ**: -3 names, +0.714 mean score, +5 good, -13 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 16 | 13 | -3 |
| Mean score | 0.125 | 0.839 | +0.714 |
| Good (≥0.8) | 0 | 5 | +5 |
| Poor (<0.5) | 13 | 0 | -13 |

**Cost**: generate=$1.15, review=$0.025, total=$1.17
**VocabGap**: 0 → 0 (token-miss: `coolant_outlet_temperature_of_breeding_blanket_module`, `radiated_thermal_power_absorbed_by_breeding_blanket_module`, etc.)

**Sample names (top 5)**:
1. `coolant_inlet_pressure_of_breeding_blanket_module` — score=1.0, outstanding
2. `tritium_breeding_ratio` — score=0.975, outstanding
3. `blanket_energy_multiplication_factor` — score=0.975, outstanding
4. `coolant_outlet_temperature_of_breeding_blanket_module` — score=0.888, outstanding
5. `coolant_inlet_temperature_of_breeding_blanket_module` — score=0.888, outstanding

**Notes**: All 13 poor-scored names eliminated. Breeding blanket module quantities dominate. Key vocab gap: `breeding_blanket_module` as an object modifier not in ISN.

### 4. electromagnetic_wave_diagnostics

**Baseline**: n=64, mean=0.180, good=1, poor=32
**Post-regen**: n=19, mean=0.760, good=7, poor=1
**Δ**: -45 names, +0.580 mean score, +6 good, -31 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 64 | 19 | -45 |
| Mean score | 0.180 | 0.760 | +0.580 |
| Good (≥0.8) | 1 | 7 | +6 |
| Poor (<0.5) | 32 | 1 | -31 |

**Cost**: generate=$1.08, review=$0.009, total=$1.09
**VocabGap**: 0 → 0 (token-miss: `electron_density_profile_approximation_parameters`, `refractometer_detector_timebase`)

**Sample names (top 5)**:
1. `normalized_poloidal_flux_coordinate` — score=1.0, outstanding
2. `normalized_toroidal_flux_coordinate` — score=1.0, outstanding
3. `major_radius_of_measurement_position` — score=0.975, outstanding
4. `toroidal_angle_of_measurement_position` — score=0.975, outstanding
5. `radial_width_of_reflectometer_antenna` — score=0.925, outstanding

**Notes**: Major count reduction (64→19) reflects better dedup and filtering of unnameable paths. Mean score jumped from 0.18 to 0.76. Diagnostic-specific terms like `refractometer_detector_timebase` still need ISN tokens.

### 5. turbulence

**Baseline**: n=135, mean=0.244, good=7, poor=54
**Post-regen**: n=30, mean=0.799, good=18, poor=3
**Δ**: -105 names, +0.555 mean score, +11 good, -51 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 135 | 30 | -105 |
| Mean score | 0.244 | 0.799 | +0.555 |
| Good (≥0.8) | 7 | 18 | +11 |
| Poor (<0.5) | 54 | 3 | -51 |

**Cost**: generate=$1.42, review=$0.036, total=$1.46
**VocabGap**: 0 → 0 (token-miss: `perturbed_parallel_vector_potential_parity`, `cosine_shape_coefficient_radial_derivative_wrt_normalized_minor_radius`, etc.)

**Sample names (top 5)**:
1. `normalized_perturbed_parallel_vector_potential_weight` — score=1.0, outstanding
2. `gyrokinetic_eigenmode_normalized_gyrocenter_density_moment` — score=1.0, outstanding
3. `gyrokinetic_eigenmode_perturbed_parallel_vector_potential_weight` — score=1.0, outstanding
4. `ion_temperature` — score=1.0, outstanding
5. `normalized_perturbed_parallel_magnetic_field_weight` — score=1.0, outstanding

**Notes**: Largest count reduction (135→30) — original had many low-quality duplicates. Mean score tripled from 0.244 to 0.799. 51 poor-scored names eliminated. Remaining 3 poor are quarantined.

### 6. plasma_wall_interactions

**Baseline**: n=71, mean=0.367, good=8, poor=37
**Post-regen**: n=15, mean=0.715, good=4, poor=3
**Δ**: -56 names, +0.348 mean score, -4 good, -34 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 71 | 15 | -56 |
| Mean score | 0.367 | 0.715 | +0.348 |
| Good (≥0.8) | 8 | 4 | -4 |
| Poor (<0.5) | 37 | 3 | -34 |

**Cost**: generate=$1.25, review=$0.093, total=$1.34
**VocabGap**: 0 → 0 (token-miss: `bidirectional_reflectance_distribution_function_parameters`, `electron_particle_flux_from_wall`, `neutral_particle_flux_from_plasma`)

**Sample names (top 5)**:
1. `vertical_component_of_magnetic_vector_potential` — score=1.0, outstanding
2. `neutral_particle_flux_from_plasma` — score=0.95, outstanding
3. `neutral_particle_flux_from_wall` — score=0.95, outstanding
4. `electron_particle_flux_from_wall` — score=0.938, outstanding
5. `neutral_chemical_sputtering_coefficient_at_wall` — score=0.788, good

**Notes**: Count reduced from 71→15 (79% reduction). Mean score improved from 0.367 to 0.715. Good-count dropped slightly (-4) because fewer total names, but quality per name is much higher. Wall interaction terms (`particle_flux_from_wall`, `sputtering_coefficient_at_wall`) named well.

### 7. gyrokinetics

**Baseline**: n=132, mean=0.322, good=8, poor=46
**Post-regen**: n=30, mean=0.864, good=20, poor=2
**Δ**: -102 names, +0.542 mean score, +12 good, -44 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 132 | 30 | -102 |
| Mean score | 0.322 | 0.864 | +0.542 |
| Good (≥0.8) | 8 | 20 | +12 |
| Poor (<0.5) | 46 | 2 | -44 |

**Cost**: generate=$1.92, review=$0.036, total=$1.96
**VocabGap**: 0 → 0 (token-miss: `gyrokinetic_eigenmode_normalized_gyrocenter_parallel_heat_flux_moment`, `flux_tube_parallel_domain_poloidal_turn_count`, `normalized_toroidal_velocity_gradient`)

**Sample names (top 5)**:
1. `gyrokinetic_eigenmode_normalized_parallel_velocity_moment_real_part` — score=1.0, outstanding
2. `gyrokinetic_eigenmode_normalized_parallel_temperature_moment_gyroaveraged_real_part` — score=1.0, outstanding
3. `gyrokinetic_eigenmode_normalized_gyrocenter_density_moment` — score=1.0, outstanding
4. `perturbed_electrostatic_potential_parity` — score=1.0, outstanding
5. `gyrokinetic_eigenmode_normalized_gyrocenter_perpendicular_pressure_moment` — score=1.0, outstanding

**Notes**: Second-best improvement alongside waves. Mean score jumped from 0.322 to 0.864. The gyrokinetic eigenmode naming convention is clean and consistent. Count dropped from 132→30 due to heavy dedup of similar moment-type names.

---

## Aggregate Summary

| Domain | Before Mean | After Mean | Δ Mean | Before N | After N | Gen $ | Rev $ | Total $ |
|--------|-------------|------------|--------|----------|---------|-------|-------|---------|
| waves | 0.083 | 0.861 | **+0.778** | 8 | 27 | 1.44 | 0.01 | 1.45 |
| fast_particles | 0.134 | 0.751 | **+0.617** | 47 | 44 | 1.92 | 0.14 | 2.06 |
| structural_components | 0.125 | 0.839 | **+0.714** | 16 | 13 | 1.15 | 0.03 | 1.17 |
| electromagnetic_wave_diagnostics | 0.180 | 0.760 | **+0.580** | 64 | 19 | 1.08 | 0.01 | 1.09 |
| turbulence | 0.244 | 0.799 | **+0.555** | 135 | 30 | 1.42 | 0.04 | 1.46 |
| plasma_wall_interactions | 0.367 | 0.715 | **+0.348** | 71 | 15 | 1.25 | 0.09 | 1.34 |
| gyrokinetics | 0.322 | 0.864 | **+0.542** | 132 | 30 | 1.92 | 0.04 | 1.96 |
| **TOTAL** | **0.208** | **0.805** | **+0.597** | **473** | **178** | **$10.18** | **$0.35** | **$10.53** |

### Top 3 Improvements (Best Δ)

1. **waves** (+0.778): 0.083 → 0.861. Near-total transformation — previously almost all names unscored/poor, now 14/18 outstanding.
2. **structural_components** (+0.714): 0.125 → 0.839. Small domain but clean naming of breeding blanket module quantities.
3. **fast_particles** (+0.617): 0.134 → 0.751. Eliminated all 38 poor-scored names, consistent torque/current density naming.

### Top 3 Non-Improvements (Smallest Δ)

1. **plasma_wall_interactions** (+0.348): Still the lowest post-regen score (0.715). Root cause: domain has many diagnostic/engineering quantities (BRDF parameters, sputtering coefficients) that don't decompose cleanly with current ISN grammar. The `particle_flux_from_wall/plasma` pattern works but engineering terms lack tokens.
2. **turbulence** (+0.555): Good improvement but 3 names remain poor (quarantined). Root cause: shape coefficient derivatives (`cosine_shape_coefficient_radial_derivative_wrt_normalized_minor_radius`) are too compositionally complex for current grammar.
3. **electromagnetic_wave_diagnostics** (+0.580): Diagnostic-specific terms like `refractometer_detector_timebase` don't have ISN tokens. These are instruments, not physics quantities.

### Residual VocabGap Analysis

VocabGap nodes were not created during this cycle (node creation may be gated by a feature flag or the generation pipeline version). However, token-miss warnings were logged extensively. Key missing ISN tokens by domain:

| Domain | Key Missing Tokens |
|--------|-------------------|
| waves | `wave_absorbed_power_per_toroidal_mode`, `wave_electric_field_amplitude_per_toroidal_mode`, `beam_tracing_wavevector` |
| fast_particles | `toroidal_torque_due_to_coulomb_collisions` (process modifier) |
| structural_components | `breeding_blanket_module` (object), `coolant_outlet_temperature` (base), `radiated_thermal_power_absorbed` (base) |
| electromagnetic_wave_diagnostics | `refractometer_detector_timebase`, `density_profile_approximation_parameters` |
| turbulence | `parallel_vector_potential_parity`, `shape_coefficient_radial_derivative_wrt_normalized_minor_radius` |
| plasma_wall_interactions | `bidirectional_reflectance_distribution_function_parameters`, `particle_flux_from_wall` (base), `magnetic_vector_potential` (base) |
| gyrokinetics | `gyrocenter_parallel_heat_flux_moment`, `flux_tube_parallel_domain_poloidal_turn_count`, `toroidal_velocity_gradient` |

### Recommendation

**Is another ISN release candidate needed?** Yes — rc17 should target:

1. **High-impact bases** (cover multiple domains):
   - `particle_flux_from_wall`, `particle_flux_from_plasma` — used by PWI, edge
   - `magnetic_vector_potential` — used by PWI, MHD
   - `wave_absorbed_power` — used by waves, auxiliary heating

2. **Domain-specific compound bases** (reduce quarantined names):
   - `breeding_blanket_module` as object modifier — structural_components
   - `beam_tracing_wavevector` — waves
   - `reflectance_distribution_function` — PWI

3. **Process modifiers**:
   - `coulomb_collisions` — fast_particles torque terms

Estimated impact: 15-20 currently quarantined names would validate with these ~10 new tokens.
