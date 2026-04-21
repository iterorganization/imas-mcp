# Rotation v3 — Anti-pattern Audit

**Date**: 2025-07-25
**Status**: In progress

## Summary

Systematic audit of anti-patterns in the standard names corpus across
under-reviewed physics domains. Identifies four anti-pattern classes,
quantifies prevalence, applies fixes via classifier rules, graph
corrections, and reviewer prompt updates.

## Anti-pattern Inventory

### 1. Duplicated Tokens (7 names)

Names containing repeated consecutive tokens (e.g. `_magnetic_magnetic_`):

| Name | Domain | Status |
|------|--------|--------|
| `fall_time_of_magnetic_magnetic_field_probe` | neutronics | enriched |
| `fall_time_of_magnetic_magnetic_magnetic_field_probe` | particle_measurement_diagnostics | enriched |
| `poloidal_component_of_magnetic_field_of_poloidal_magnetic_magnetic_field_probe` | equilibrium | named |
| `poloidal_component_of_magnetic_field_of_poloidal_magnetic_magnetic_magnetic_field_probe` | equilibrium | enriched |
| `poloidal_magnetic_magnetic_field_probe_voltage` | equilibrium | enriched |
| `rise_time_of_magnetic_magnetic_field_probe` | neutronics | enriched |
| `rise_time_of_magnetic_magnetic_magnetic_field_probe` | particle_measurement_diagnostics | enriched |

**Root cause**: LLM composer stuttering on repeated `magnetic` when
composing names for `b_field_sensor` paths inside magnetic probe contexts.

**Fix**: Delete and regenerate from source DD paths.

### 2. Placeholder Names (2 names)

Names that describe data containers, not physical quantities:

| Name | Domain | Status |
|------|--------|--------|
| `constant_float_value` | general | named |
| `constant_integer_value` | general | named |

**Root cause**: DD paths for generic constant storage (`summary/local/parameter/*/value`)
passed through to extraction despite not representing physics.

**Fix**: Add classifier rule S3 to skip placeholder patterns.

### 3. Wrong Domain — Neutronics (5 names)

All 5 neutronics-domain names are detector timing or magnetic probe
quantities that belong in `particle_measurement_diagnostics`:

| Name | IMAS Path | IMAS Domain |
|------|-----------|-------------|
| `fall_time_of_magnetic_magnetic_field_probe` | neutron_diagnostic/detector/b_field_sensor/fall_time | particle_measurement_diagnostics |
| `rise_time_of_magnetic_magnetic_field_probe` | neutron_diagnostic/detector/b_field_sensor/rise_time | particle_measurement_diagnostics |
| `neutron_detector_data_acquisition_start_time` | neutron_diagnostic/detectors/start_time | neutronics |
| `neutron_detector_recording_end_time` | neutron_diagnostic/detectors/end_time | neutronics |
| `neutron_detector_spectrum_sampling_time` | neutron_diagnostic/detectors/spectrum_sampling_time | neutronics |

**Note**: The first 2 are also duplicated-token names and will be
deleted. The last 3 are legitimately from the `neutronics` IDS —
their IMAS domain IS neutronics. No domain reassignment needed for
these; they correctly describe neutron detector timing.

### 4. Model-Specific Suffixes (4 names)

Names containing author/model identifiers as suffixes:

| Name | Domain |
|------|--------|
| `critical_normalized_pressure_gradient_at_pedestal_hager_bootstrap` | general |
| `critical_normalized_pressure_gradient_at_pedestal_sauter_bootstrap` | general |
| `ratio_of_critical_to_experimental_normalized_pressure_gradient_hager_bootstrap` | general |
| `ratio_of_critical_to_experimental_normalized_pressure_gradient_sauter_bootstrap` | general |

**Root cause**: DD paths contain model author names (hager, sauter) that
the composer preserved verbatim. Standard names should be model-agnostic.

**Fix**: Add anti-pattern guidance to reviewer prompt; add calibration
examples.

## Review Coverage (pre-intervention)

| Domain | Total | Enriched | Named | Review % |
|--------|-------|----------|-------|----------|
| neutronics | 5 | 2 | 3 | 0% |
| plasma_measurement_diagnostics | 3 | 1 | 2 | 0% |
| magnetic_field_systems | 14 | 8 | 6 | 0% |
| fast_particles | 29 | 26 | 3 | 0% |
| plasma_wall_interactions | 17 | 13 | 4 | 0% |
| particle_measurement_diagnostics | 32 | 18 | 14 | 0% |
| edge_plasma_physics | 75 | 46 | 29 | 0% |
| general | 142 | 37 | 97 | 0% |
| transport | 238 | 180 | 58 | 0% |

## Fixes Applied

### Classifier Rule S3 — Placeholder Skip

Added rule to `classifier.py` skipping paths whose leaf segment matches
`^constant_(float|integer|boolean)_value$` or similar non-physics
container patterns.

### Reviewer Prompt — Model Suffix Anti-pattern

Added guidance to the review prompt flagging names ending in known model
author surnames as convention violations.

### Calibration — Model Suffix Example

Added poor-tier calibration entry for model-suffix anti-pattern.

## Review Results

### Review Cost Summary
Total LLM spend: **$2.35** (of $20 cap)

### Domains Reviewed

| Domain | Names Scored | Tier Distribution | Cost |
|--------|-------------|-------------------|------|
| neutronics | 3 | 3 good | $0.08 |
| plasma_measurement_diagnostics | 2+2=4 | 1 outstanding, 3 good | $0.14 |
| magnetic_field_systems | 6 | 5 outstanding, 1 good | $0.12 |
| fast_particles | 18 | 14 good, 4 adequate | $0.30 |
| plasma_wall_interactions | 12 | 1 outstanding, 10 good, 1 adequate | $0.24 |
| particle_measurement_diagnostics | 2 | 2 good | $0.07 |
| general | 92 | 50 outstanding, 38 good, 3 adequate, 1 poor | $1.03 |
| edge_plasma_physics | 22 | 7 outstanding, 10 good, 3 adequate, 2 poor | $0.45 |

### Overall Post-Review State

| Domain | Total | Scored | Avg% | Outstanding | Good | Adequate | Poor |
|--------|-------|--------|------|-------------|------|----------|------|
| magnetic_field_systems | 14 | 14 | 88.4 | 11 | 1 | 2 | 0 |
| plasma_control | 25 | 25 | 93.6 | 23 | 0 | 2 | 0 |
| equilibrium | 60 | 57 | 78.3 | 37 | 5 | 13 | 2 |
| transport | 238 | 183 | 89.7 | 159 | 4 | 5 | 15 |
| general | 170 | 130 | 87.6 | 86 | 38 | 5 | 1 |
| edge_plasma_physics | 81 | 67 | 66.3 | 32 | 10 | 6 | 19 |
| fast_particles | 40 | 27 | 74.8 | 8 | 15 | 4 | 0 |
| neutronics | 6 | 3 | 70.8 | 0 | 3 | 0 | 0 |
| plasma_wall_interactions | 21 | 16 | 73.1 | 3 | 10 | 2 | 1 |
| particle_measurement_diagnostics | 31 | 17 | 90.6 | 12 | 4 | 0 | 1 |
| plasma_measurement_diagnostics | 4 | 2 | 84.4 | 1 | 1 | 0 | 0 |
| **Totals** | **920** | **702** | — | — | — | — | — |

**Overall coverage: 76.3% scored (702/920)**

### Score Outliers (flagged for investigation)

- `ratio_of_critical_to_experimental_normalized_pressure_gradient_at_pedestal`: 0.51 → regenerate
- `poloidal_component_of_current_density_due_to_anomalous`: 0.39 → poor, investigate
- `toroidal_component_of_current_density_due_to_anomalous`: 0.39 → poor, investigate
- `vertical_component_of_momentum_flux_limiter`: 0.42 → regenerate
- `pedestal_fit_coefficients`: 0.56 → investigate
- `bidirectional_reflectance_distribution_function_coefficient_at_wall`: 0.57 → verbose

## Post-fix Counts

### Fixed
- **7 duplicate-token names** deleted (all contained `_magnetic_magnetic_`)
- **2 placeholder names** deleted (`constant_float_value`, `constant_integer_value`)
- Classifier rule S3 added to prevent future placeholder names
- Review prompt updated with [I1.7] model-suffix anti-pattern check
- Calibration entry added for model-suffix anti-pattern (poor tier)

### Residual (next cycle)
- **4 model-suffix names** remain (`*_hager_bootstrap`, `*_sauter_bootstrap`) — need regeneration
  without model suffixes
- **2 poor edge_plasma names** (`*_due_to_anomalous`) truncated by LLM
- `general` domain still acts as catch-all for 170 names (up from 142 after
  reviews revised and reclassified some names)
- edge_plasma_physics has highest poor-tier count (19) — many need revision

### New Anti-patterns Discovered
1. **Truncated names**: `poloidal_component_of_current_density_due_to_anomalous`
   and `toroidal_component_of_current_density_due_to_anomalous` — the LLM
   truncated `anomalous_transport` to just `anomalous`
2. **Grammar parse failures**: `fast_ion_toroidal_current_density` fails grammar
   (component must precede subject) — needs rewrite to
   `toroidal_component_of_fast_ion_current_density`
3. **Verbose compounds**: `bidirectional_reflectance_distribution_function_coefficient_at_wall`
   (10+ tokens) — too long even for an open-vocabulary physical_base
4. **COCOS type=one_like without integer**: Multiple names flagged with
   `cocos_transformation_type='one_like'` but no COCOS integer — graph edge
   creation skipped
