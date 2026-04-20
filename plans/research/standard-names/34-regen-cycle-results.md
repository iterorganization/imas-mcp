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

