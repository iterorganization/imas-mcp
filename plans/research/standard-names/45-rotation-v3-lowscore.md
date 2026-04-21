# Rotation v3 — Low-Score Domain Rotation (rc18 Pipeline)

**Date**: 2025-07-26
**ISN version**: v0.7.0rc18
**Pipeline features**: Open-segment VocabGap filter, decomposition audit, grammar-slot search
**Budget cap**: $12.00
**Actual cost**: $2.39

## Summary

Targeted the 4 lowest-scoring physics domains (by reviewed avg score) for a
full rotation cycle (generate→enrich→review→regen). Used the rc18 pipeline
with open-base VocabGap filtering and decomposition audit.

## Targets

Domains selected by lowest `avg(review_mean_score)` among domains with ≥10 names:

| Rank | Domain | Avg Score | Total | Reviewed | Below 0.70 | Unreviewed |
|------|--------|-----------|-------|----------|------------|------------|
| 1 | edge_plasma_physics | 0.666 | 81 | 67 | 25 | 14 |
| 2 | computational_workflow | 0.683 | 19 | 17 | 7 | 2 |
| 3 | fast_particles | 0.748 | 40 | 27 | 11 | 13 |
| 4 | waves | 0.761 | 14 | 11 | 3 | 3 |

## Rotation Results

### 1. edge_plasma_physics

| Phase | Count | Cost | Elapsed |
|-------|-------|------|---------|
| generate | 0 | $0.00 | 22s |
| enrich | 4 | $0.19 | 58s |
| review | 0 | $0.00 | 2s |
| regen | 10 | $0.81 | 211s |
| **Total** | **14** | **$0.99** | **293s** |

**Key outcomes**:
- 4 names enriched with documentation
- 10 names regenerated from `needs_revision` with reviewer feedback
- 4 new replacement names created: `ion_particle_flux`,
  `ion_particle_convective_velocity`, `sobol_sensitivity_index_input_path_reference`,
  `radial_component_of_ion_convective_velocity`
- VocabGaps surfaced: `anomalous_current_density`, `particle_convective_velocity`,
  `reference_major_radius`, `reference_major_radius_of_vacuum_toroidal_field`
- Residual: 6 names still `needs_revision` (truncated `*_due_to_anomalous` pair persists)

### 2. computational_workflow

| Phase | Count | Cost | Elapsed |
|-------|-------|------|---------|
| generate | 0 | $0.00 | 19s |
| enrich | 0 | $0.00 | 2s |
| review | 0 | $0.00 | 2s |
| regen | 4 | $0.48 | 148s |
| **Total** | **4** | **$0.48** | **171s** |

**Key outcomes**:
- 4 names regenerated; `needs_revision` reduced from 5 to 3
- 2 new replacement names: `simulation_time_step`, `vacuum_toroidal_magnetic_field_at_reference_major_radius`
- VocabGaps: `position_of_transport_solver_boundary_condition`,
  `relative_deviation_of_electron_energy_equation`,
  `energy_transport_boundary_condition_value`,
  `workflow_component_control_parameter`
- Residual: solver metadata names (`workflow_component_control_float_parameter`,
  `normalized_radial_coordinate_of_transport_boundary_condition`) fundamentally
  don't fit ISN physics grammar — these are engineering parameters

### 3. fast_particles

| Phase | Count | Cost | Elapsed |
|-------|-------|------|---------|
| generate | 0 | $0.00 | 19s |
| enrich | 0 | $0.09 | 30s |
| review | 0 | $0.00 | 2s |
| regen | 3 | $0.37 | 228s |
| **Total** | **3** | **$0.46** | **279s** |

**Key outcomes**:
- 3 names regenerated; `needs_revision` reduced from 4 to 2
- VocabGaps: `current_density` (in `toroidal_component_of_fast_ion_current_density`),
  `current_density_co_passing`
- `toroidal_current_density_co_passing` and `toroidal_counter_passing_current_density`
  scored 0.45 adequate — co-passing/counter-passing particle current decomposition
  needs ISN tokens

### 4. waves

| Phase | Count | Cost | Elapsed |
|-------|-------|------|---------|
| generate | 0 | $0.00 | 19s |
| enrich | 0 | $0.04 | 24s |
| review | 0 | $0.00 | 2s |
| regen | 3 | $0.42 | 108s |
| **Total** | **3** | **$0.46** | **153s** |

**Key outcomes**:
- 3 names regenerated; `needs_revision` reduced from 2 to 0
- All `needs_revision` names cleared from waves domain
- VocabGaps: `beam_tracing_wave_vector_varying_toroidal_mode_number_flag`,
  `wave_electric_field_phase_per_toroidal_mode`
- `beam_tracing_wave_vector_varying_toroidal_mode_number_flag` scored 0.0 — boolean
  flag name, fundamentally not a physics quantity

## Aggregate Summary

| Domain | Before Mean | After Mean | Before N | After N | needs_rev Δ | Cost |
|--------|-------------|------------|----------|---------|-------------|------|
| edge_plasma_physics | 0.666 | 0.666 | 81 | 83 | 6→6 | $0.99 |
| computational_workflow | 0.683 | 0.695 | 19 | 21 | 5→3 | $0.48 |
| fast_particles | 0.748 | 0.748 | 40 | 41 | 4→2 | $0.46 |
| waves | 0.761 | 0.761 | 14 | 14 | 2→0 | $0.46 |
| **TOTAL** | — | — | **154** | **159** | **17→11** | **$2.39** |

**Notes on score stagnation**: The reviewed avg scores didn't change because:
1. The regen phase creates replacement names with `pending` validation status
2. These need a subsequent review pass to be scored
3. The existing reviewed scores for old names persist
4. Net effect is seen in `needs_revision` reduction (-6 names) and new name generation (+7)

### Global State Change

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Total names | 918 | 925 | +7 |
| Reviewed | 702 | 702 | 0 |
| Unreviewed | 216 | 223 | +7 |
| Below 0.70 | 130 | 130 | 0 |
| Avg reviewed score | 0.838 | 0.838 | 0 |
| needs_revision | 17 | 11 | -6 |
| Total VocabGaps | 195 | 195 | 0 |

## Anti-Pattern Audit

Ran all anti-pattern queries from `41-rotation-v3-antipatterns.md`:

| Anti-pattern | Count | Status |
|--------------|-------|--------|
| Duplicated tokens (`_word_word_`) | 0 | ✓ Clean |
| Placeholder names (`constant_*_value`) | 0 | ✓ Clean |
| Model-specific suffixes (`*_hager_bootstrap` etc.) | 4 | Pre-existing (general domain) |
| Truncated names (`*_due_to_anomalous`) | 2 | Pre-existing (edge_plasma) |
| Very long names (>10 tokens) | 15 | Mostly transport 2nd-derivatives |
| Boolean flag names (`*_flag`) | 15 | Mixed quality (0.0–0.98) |

**No new anti-patterns introduced by this rotation.** The 4 model-suffix and 2
truncated names are residuals from prior cycles documented in plan 41.

## VocabGaps Surfaced

Key missing ISN tokens from this rotation:

### physical_base (open segment)
| Token | Domain | Occurrences |
|-------|--------|-------------|
| `anomalous_current_density` | edge_plasma_physics | 2 |
| `particle_convective_velocity` | edge_plasma_physics | 1 |
| `reference_major_radius` | edge_plasma_physics | 1 |
| `reference_major_radius_of_vacuum_toroidal_field` | edge_plasma_physics | 3 |
| `sobol_sensitivity_index_input_path_reference` | edge_plasma_physics | 1 |
| `position_of_transport_solver_boundary_condition` | computational_workflow | 1 |
| `relative_deviation_of_electron_energy_equation` | computational_workflow | 1 |
| `energy_transport_boundary_condition_value` | computational_workflow | 1 |
| `workflow_component_control_parameter` | computational_workflow | 1 |
| `current_density` (fast_ion context) | fast_particles | 2 |
| `current_density_co_passing` | fast_particles | 1 |
| `beam_tracing_wave_vector_varying_toroidal_mode_number_flag` | waves | 1 |
| `wave_electric_field_phase_per_toroidal_mode` | waves | 2 |

### Patterns
- **Engineering metadata**: `workflow_component_control_parameter`,
  `position_of_transport_solver_boundary_condition` — inherently not physics quantities
- **Complex decompositions**: `reference_major_radius_of_vacuum_toroidal_field` — long
  compound that could decompose into position + subject + base
- **Missing particle species modifiers**: `anomalous_current_density`, `co_passing`,
  `counter_passing` — need ISN tokens for anomalous transport and passing-particle species

## Cost Breakdown

| Domain | Generate | Enrich | Review | Regen | Total |
|--------|----------|--------|--------|-------|-------|
| edge_plasma_physics | $0.00 | $0.19 | $0.00 | $0.81 | $0.99 |
| computational_workflow | $0.00 | $0.00 | $0.00 | $0.48 | $0.48 |
| fast_particles | $0.00 | $0.09 | $0.00 | $0.37 | $0.46 |
| waves | $0.00 | $0.04 | $0.00 | $0.42 | $0.46 |
| **TOTAL** | **$0.00** | **$0.32** | **$0.00** | **$2.07** | **$2.39** |

**Observations**:
- Generate phase found zero new paths across all domains — extraction is saturated
- Review phase had zero candidates — all existing names were already reviewed
- Cost dominated by regen phase ($2.07, 87%) — regenerating with reviewer feedback
  is the most expensive phase due to longer prompts with critique context
- Total cost $2.39 of $12.00 budget (20%) — significant underrun

## Recommendations for Next Cycle

1. **Review the pending names**: Run `sn review --domain <domain>` on all 4 domains
   to score the 7 newly created names and 10 regenerated names (now pending)
2. **Model-suffix cleanup**: The 4 `*_hager_bootstrap`/`*_sauter_bootstrap` names in
   `general` domain need targeted regeneration without model suffixes
3. **Truncated name fix**: `*_due_to_anomalous` names need ISN vocab for
   `anomalous_transport` as a physics process
4. **Flag name policy**: Decide whether boolean flags (`*_flag`) should be standard-named
   or classified as metadata and excluded from the physics grammar
5. **Transport 2nd-derivatives**: 15+ names with >10 tokens — consider whether these
   need abbreviated naming conventions
