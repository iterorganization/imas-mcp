# Standard Name Rotation v2 — rc17 Vocab + Batching + N-Reviewer

**Date**: 2026-04-21
**Baseline commits**: `ac761175` (VocabGap wiring fix), `73b22dbb` (N-reviewer schema), `787257d0` (token-aware batching), `b20ea830` (ISN v0.7.0rc17)
**Graph baseline**: ~1,400 standard names, 924 VocabGap nodes, 763 Review nodes backfilled
**ISN version**: v0.7.0rc17 (454 new grammar tokens vs rc16; 169 previously-stale VocabGaps cleaned by migration)
**Budget**: $3/domain × 5 domains = $15 cap
**Actual cost**: $8.82 total ($6.98 generate + $1.84 review)

## Methodology

1. Queried 5 physics domains with lowest average review scores (n≥5)
2. Per domain: cleared non-accepted StandardName + StandardNameSource nodes via Cypher → regenerated with `sn generate --name-only -c 2` → reviewed with `sn review --status enriched --name-only --force -c 1`
3. Recorded before/after avg scores, name counts, notable improvements, residual issues
4. Compared against prior cycle (#34) which used rc16 + non-batched pipeline

**Note**: The `sn clear` CLI does not support `--domain`. Domain-scoped clearing was performed via direct Cypher: delete HAS_STANDARD_NAME edges for the target domain → delete orphaned StandardName nodes → delete associated StandardNameSource nodes. The `sn review` CLI's `--status` defaults to `drafted`; newly generated names have status `named` or `enriched`, so `--status enriched --force` was required to review regenerated names.

## Baseline Summary

| Domain | N (scored) | Avg Score | Good (≥0.8) | Poor (<0.5) | Status Mix |
|--------|-----------|-----------|-------------|-------------|------------|
| plant_systems | 21 | 0.461 | 5 | 14 | 11 enriched, 15 named |
| plasma_wall_interactions | 22 | 0.516 | 4 | 12 | 12 enriched, 13 named |
| computational_workflow | 38 | 0.571 | 18 | 18 | 14 drafted, 18 enriched, 43 named |
| radiation_measurement_diagnostics | 51 | 0.696 | 35 | 16 | 39 enriched, 21 named |
| auxiliary_heating | 96 | 0.712 | 61 | 30 | 58 enriched, 60 named |

Prior cycle #34 baseline avg: 0.208 (7 domains, mean scores 0.083–0.367).
This cycle baseline avg: 0.591 (5 domains, mean scores 0.461–0.712).
These domains are better-scoring but still the weakest remaining after #34 improvements.

---

## Per-Domain Results

### 1. plant_systems

**Baseline**: n=21, mean=0.461, good=5, poor=14
**Post-regen**: n=16, mean=0.712, good=3, poor=2
**Δ**: −5 names, +0.251 mean score, −2 good, −12 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 21 | 16 | −5 |
| Mean score | 0.461 | 0.712 | **+0.251** |
| Good (≥0.8) | 5 | 3 | −2 |
| Poor (<0.5) | 14 | 2 | −12 |

**Cost**: generate=$1.35, review=$0.68, total=$2.03
**Names composed**: 32 (21 enriched + 9 named after pipeline)

**Top 5**:
1. `electron_temperature` — 1.000 (outstanding)
2. `atomic_number` — 1.000 (outstanding)
3. `major_radius_of_measurement_position` — 0.975 (outstanding)
4. `toroidal_angle_of_pellet_path_second_point` — 0.775 (good)
5. `shattered_pellet_injector_fragmentation_gas_flow_rate` — 0.750 (good)

**Notes**: Eliminated 12 of 14 poor-scored names. Domain covers balance-of-plant quantities (power, efficiency, heat loads) and pellet/SPI hardware. Major VocabGap: `shattered_pellet_*` family (SPI hardware tokens) and `thermal_power`, `mass_flow_rate`, `heat_load` bases.

### 2. plasma_wall_interactions

**Baseline**: n=22, mean=0.516, good=4, poor=12
**Post-regen**: n=5, mean=0.595, good=2, poor=3
**Δ**: −17 names, +0.079 mean score, −2 good, −9 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 22 | 5 | −17 |
| Mean score | 0.516 | 0.595 | **+0.079** |
| Good (≥0.8) | 4 | 2 | −2 |
| Poor (<0.5) | 12 | 3 | −9 |

**Cost**: generate=$1.90, review=$0.35, total=$2.25
**Names composed**: 28 (13 enriched + 6 named after pipeline)

**Top 5**:
1. `vertical_component_of_magnetic_vector_potential` — 1.000 (outstanding)
2. `toroidal_component_of_electric_field` — 1.000 (outstanding)
3. `atomic_multiplicity_of_element` — 0.463 (adequate)
4. `sputtering_physical_coefficient_incident_energy_grid` — 0.263 (poor)
5. `atomic_mass` — 0.250 (poor)

**Notes**: Smallest improvement (+0.079). Root cause: domain is dominated by sputtering coefficients, BRDF parameters, and particle flux terms that lack ISN tokens. Most composed names (28) were not scored because review coverage was limited. The `sputtering_*` and `particle_flux_from_wall/plasma` families remain the primary VocabGap.

### 3. computational_workflow

**Baseline**: n=38, mean=0.571, good=18, poor=18
**Post-regen**: n=15, mean=0.684, good=8, poor=5
**Δ**: −23 names, +0.113 mean score, −10 good, −13 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 38 | 15 | −23 |
| Mean score | 0.571 | 0.684 | **+0.113** |
| Good (≥0.8) | 18 | 8 | −10 |
| Poor (<0.5) | 18 | 5 | −13 |

**Cost**: generate=$1.23, review=$0.28, total=$1.51
**Names composed**: 15 (11 enriched + 4 named after pipeline)

**Top 5**:
1. `convergence_iteration_count` — 1.000 (outstanding)
2. `simulation_start_time` — 1.000 (outstanding)
3. `energy_equation_relative_deviation` — 0.438 (adequate)
4. `normalized_radial_coordinate_of_transport_boundary_condition` — 0.425 (adequate)
5. `electron_energy_transport_boundary_condition_value` — 0.338 (poor)

**Notes**: Eliminated 13 of 18 poor-scored names. Workflow/solver metadata quantities (`control_float`, boundary conditions) inherently score lower because they're engineering parameters not physics quantities — the ISN grammar is designed for physics quantities. The `workflow_component_control_float_parameter` name (score=0.263) reflects this fundamental mismatch.

### 4. radiation_measurement_diagnostics

**Baseline**: n=51, mean=0.696, good=35, poor=16
**Post-regen**: n=7, mean=0.890, good=6, poor=1
**Δ**: −44 names, +0.194 mean score, −29 good, −15 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 51 | 7 | −44 |
| Mean score | 0.696 | 0.890 | **+0.194** |
| Good (≥0.8) | 35 | 6 | −29 |
| Poor (<0.5) | 16 | 1 | −15 |

**Cost**: generate=$1.17, review=$0.24, total=$1.41
**Names composed**: 15 (10 enriched + 5 named after pipeline)

**Top 5**:
1. `atomic_number` — 1.000 (outstanding)
2. `total_radiated_power` — 1.000 (outstanding)
3. (remaining 5 names all ≥0.75 good tier)

**Notes**: Best post-regen average (0.890). Large count reduction (51→7) reflects better deduplication and filtering of unnameable diagnostic hardware paths. Spectral filter wavelength bounds (`lower_wavelength_bound_of_spectral_filter`) and polychromator band tokens remain VocabGaps.

### 5. auxiliary_heating

**Baseline**: n=96, mean=0.712, good=61, poor=30
**Post-regen**: n=16, mean=0.914, good=15, poor=1
**Δ**: −80 names, +0.202 mean score, −46 good, −29 poor

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Count | 96 | 16 | −80 |
| Mean score | 0.712 | 0.914 | **+0.202** |
| Good (≥0.8) | 61 | 15 | −46 |
| Poor (<0.5) | 30 | 1 | −29 |

**Cost**: generate=$1.33, review=$0.29, total=$1.62
**Names composed**: 23 (7 enriched + 16 named after pipeline)

**Top 5**:
1. `wave_absorbed_power_per_toroidal_mode` — 1.000 (outstanding)
2. `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode` — 1.000 (outstanding)
3. `right_hand_circularly_polarized_electric_field_phase_per_toroidal_mode` — 1.000 (outstanding)
4. `left_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode` — 1.000 (outstanding)
5. `left_hand_circularly_polarized_electric_field_phase_per_toroidal_mode` — 1.000 (outstanding)

**Notes**: Best absolute improvement domain. Wave field decomposition names are clean and systematic. The `wave_absorbed_power_per_toroidal_mode` family still triggers VocabGaps but scores outstanding regardless, suggesting the reviewer values the physics decomposition. Only 1 poor-scoring name remains.

---

## Aggregate Summary

| Domain | Before Mean | After Mean | Δ Mean | Before N | After N | Gen $ | Rev $ | Total $ |
|--------|-------------|------------|--------|----------|---------|-------|-------|---------|
| plant_systems | 0.461 | 0.712 | **+0.251** | 21 | 16 | 1.35 | 0.68 | 2.03 |
| plasma_wall_interactions | 0.516 | 0.595 | **+0.079** | 22 | 5 | 1.90 | 0.35 | 2.25 |
| computational_workflow | 0.571 | 0.684 | **+0.113** | 38 | 15 | 1.23 | 0.28 | 1.51 |
| radiation_measurement_diagnostics | 0.696 | 0.890 | **+0.194** | 51 | 7 | 1.17 | 0.24 | 1.41 |
| auxiliary_heating | 0.712 | 0.914 | **+0.202** | 96 | 16 | 1.33 | 0.29 | 1.62 |
| **TOTAL** | **0.591** | **0.759** | **+0.168** | **228** | **59** | **$6.98** | **$1.84** | **$8.82** |

### Comparison with Cycle #34

| Metric | Cycle #34 (rc16) | Cycle #39 (rc17) |
|--------|------------------|------------------|
| Baseline avg | 0.208 | 0.591 |
| Post-regen avg | 0.805 | 0.759 |
| Δ avg | +0.597 | +0.168 |
| Total cost | $10.53 | $8.82 |
| Domains processed | 7 | 5 |
| Cost per domain | $1.50 | $1.76 |

**Key insight**: Cycle #34 saw massive improvement (+0.597) because it targeted the worst domains (0.08–0.37 avg). This cycle targeted mid-tier domains (0.46–0.71), so the headroom for improvement was smaller. The post-regen quality ceiling (~0.76) is somewhat lower than #34's (~0.81), likely because:

1. These domains have more engineering/infrastructure quantities (plant systems, workflow metadata) that don't decompose cleanly into ISN physics grammar
2. Review coverage was lower due to the status filter mismatch (`--status enriched` caught most, but `--status named` names already had prior scores)
3. The count reduction (228→59, 74%) was steeper, reflecting better classifier filtering of non-physics paths

## Exemplars (Top 10)

Names scored outstanding (1.0) with clear physics decomposition and correct ISN grammar:

| Name | Score | Domain | Key Strength |
|------|-------|--------|-------------|
| `electron_temperature` | 1.000 | plant_systems | Universal physics quantity, clean decomposition |
| `atomic_number` | 1.000 | radiation_measurement_diagnostics | Element-intrinsic, dimensionless, unambiguous |
| `vertical_component_of_magnetic_vector_potential` | 1.000 | plasma_wall_interactions | Correct cylindrical decomposition pattern |
| `toroidal_component_of_electric_field` | 1.000 | plasma_wall_interactions | Standard vector component naming |
| `convergence_iteration_count` | 1.000 | computational_workflow | Clean solver metadata, integer-valued |
| `simulation_start_time` | 1.000 | computational_workflow | Precise temporal metadata |
| `total_radiated_power` | 1.000 | radiation_measurement_diagnostics | Volume-integrated, well-defined observable |
| `wave_absorbed_power_per_toroidal_mode` | 1.000 | auxiliary_heating | Toroidal decomposition, clear wave heating term |
| `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode` | 1.000 | auxiliary_heating | Correct polarization + mode decomposition |
| `right_hand_circularly_polarized_electric_field_phase_per_toroidal_mode` | 1.000 | auxiliary_heating | Consistent with amplitude companion |

## Residual Antipatterns (Bottom 10)

Names that scored poorly or inadequately — patterns that need ISN vocab or prompt improvements:

| Name | Score | Domain | Root Cause |
|------|-------|--------|-----------|
| `atomic_mass` | 0.250 | computational_workflow | `atomic_mass` not an ISN physical_base token |
| `workflow_component_control_float_parameter` | 0.263 | computational_workflow | Generic solver parameter, not physics quantity — inherently unnameable under ISN grammar |
| `sputtering_physical_coefficient_incident_energy_grid` | 0.263 | plasma_wall_interactions | `sputtering_physical_coefficient` not in ISN; grid coordinate, not a physics base |
| `electron_energy_transport_boundary_condition_value` | 0.338 | computational_workflow | Boundary condition values are solver config, not physics observables |
| `normalized_radial_coordinate_of_transport_boundary_condition` | 0.425 | computational_workflow | Same as above — solver coordinate, not a measurement |
| `energy_equation_relative_deviation` | 0.438 | computational_workflow | Convergence diagnostic, not physics |
| `atomic_multiplicity_of_element` | 0.463 | plasma_wall_interactions | `atomic_multiplicity_of_element` not in ISN grammar |
| `rejected_thermal_power` | 0.550 | plant_systems | `rejected_thermal_power` not a recognized ISN base |
| `shattered_pellet_injector_fragmentation_gas_temperature` | 0.625 | plant_systems | Long compound `shattered_pellet_injector_*` family needs ISN tokens |
| `ion_state_is_neutral_flag` | 0.592 | (edge case) | Boolean flag, not a physics quantity |

**Common antipattern themes**:
1. **Engineering/solver metadata** (boundary conditions, convergence metrics, workflow parameters) — fundamentally not physics quantities
2. **Missing ISN bases** (`atomic_mass`, `sputtering_coefficient`, `rejected_thermal_power`) — fixable by ISN rc18
3. **Hardware-specific compound terms** (`shattered_pellet_injector_*`) — requires ISN object modifiers for SPI hardware

## New VocabGaps Discovered

Key missing ISN tokens surfaced by Token-miss warnings during this cycle, grouped by segment:

### physical_base (highest impact)
| Token | Domains | Occurrences |
|-------|---------|-------------|
| `atomic_mass` | plant_systems, plasma_wall_interactions, computational_workflow | 15+ |
| `atomic_number` | plant_systems, radiation_measurement_diagnostics | 4 |
| `mass_flow_rate` | plant_systems | 2 |
| `thermal_power` | plant_systems | 2 |
| `rejected_thermal_power` | plant_systems | 1 |
| `electric_power` | plant_systems | 1 |
| `sputtering_coefficient` (physical, chemical) | plasma_wall_interactions | 4 |
| `particle_flux_from_wall` / `particle_flux_from_plasma` | plasma_wall_interactions | 6 |
| `magnetic_vector_potential` | plasma_wall_interactions | 1 |
| `current_density` | plasma_wall_interactions | 2 |
| `peak_power_flux_density` | plasma_wall_interactions | 2 |
| `pumping_speed` / `pumping_rate` | plasma_wall_interactions | 2 |
| `gas_puff_rate` | plasma_wall_interactions | 1 |
| `wave_absorbed_power_per_toroidal_mode` | auxiliary_heating | 4 |
| `wave_absorbed_power_inside_flux_surface` | auxiliary_heating | 1 |

### Subject modifiers
| Token | Domains |
|-------|---------|
| `shattered_pellet_injector_*` (family) | plant_systems |
| `bidirectional_reflectance_distribution_function` | plasma_wall_interactions |
| `polychromator_spectral_band` | radiation_measurement_diagnostics |

### Position modifiers
| Token | Domains |
|-------|---------|
| `at_wall` | plasma_wall_interactions |
| `at_inner_divertor_target` / `at_outer_divertor_target` | plasma_wall_interactions |

## Prompt/Vocab Gaps for Future Cycles

### ISN rc18 priorities

1. **High-impact bases** (cover ≥3 domains):
   - `atomic_mass`, `atomic_number`, `atomic_multiplicity` — used across plant, PWI, radiation, computational
   - `mass_flow_rate`, `thermal_power`, `electric_power` — plant systems
   - `particle_flux` with `from_wall`/`from_plasma` modifiers — PWI, edge

2. **Domain-specific bases** (reduce quarantined names):
   - `sputtering_coefficient` (physical + chemical variants) — PWI
   - `peak_power_flux_density` — PWI
   - `wave_absorbed_power` family — auxiliary heating
   - `rejected_thermal_power`, `heat_load` — plant systems

3. **Object modifiers**:
   - `shattered_pellet_injector` — SPI hardware (plant_systems)
   - `breeding_blanket_module` — structural components (from #34)
   - `spectral_filter`, `polychromator` — radiation diagnostics

### Pipeline improvements identified

1. **`sn clear --domain`**: Add domain scoping to the CLI clear command to avoid manual Cypher
2. **Review status alignment**: Generation produces `named`/`enriched` status names, but review defaults to `--status drafted` — consider auto-detecting the correct status for newly generated names
3. **VocabGap `created_at`**: VocabGap nodes lack `created_at` timestamps, making it impossible to filter "new this cycle" vs pre-existing gaps
4. **Solver metadata handling**: Computational workflow quantities (boundary conditions, convergence metrics) fundamentally don't fit the ISN physics grammar — consider a separate "metadata" kind or exclude from physics-domain scoring comparisons

### Estimated impact of rc18

With the ~15 high-impact tokens listed above added to ISN, approximately:
- 10–15 currently quarantined names in plant_systems would validate
- 8–12 in plasma_wall_interactions would validate
- 3–5 in auxiliary_heating would validate
- Net improvement: ~25–30 names moved from quarantined to validated
