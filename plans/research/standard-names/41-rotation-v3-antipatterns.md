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

*(To be filled after running reviews)*

## Post-fix Counts

*(To be filled after applying graph corrections)*
