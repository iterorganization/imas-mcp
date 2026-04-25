# Wave 7 Synthesis (NAMES + DOCS rotation, $21.02 spent)

## Names rotation (W7A — gyrokinetics, plasma_wall_interactions, particle_measurement_diagnostics)

| Domain | Composed | Reviewed | Mean | %≥0.85 | Cost |
|--------|---------:|---------:|-----:|-------:|-----:|
| gyrokinetics | 35 | 25 | 0.649 | 12% | $4.25 |
| plasma_wall_interactions | 54 | 26 | **0.737** | **58%** | $4.46 |
| particle_measurement_diagnostics | 42 | 37 | 0.658 | 27% | $4.44 |

Anti-patterns (88 reviews): vocab_gap 26%, provenance words 16%, wrong_kind 14%, `_from_/_to_` 8%, `uncertainty_index` 7%.

## Docs rotation (W7B — magnetic_field_diagnostics, machine_operations, turbulence)

| Domain | Coverage | Mean | Cost |
|--------|---------:|-----:|-----:|
| magnetic_field_diagnostics | 26/28 (93%) | 0.758 | $4.18 |
| machine_operations | 13/22 (59%) | 0.717 | $2.56 |
| turbulence | 8/15 (53%) | 0.769 | $1.15 |

**Combined SN quality (names + docs)**: ≈0.69 mean — **below 0.8 target**. Highest-quality domain (pwa) at 0.737 still under target.

## Critical bugs found
1. **revised-name overwrite** in `review/pipeline.py:1456-1462` — entry id replaced by reviewer's `revised_name` → 386 orphan Review nodes, 16 SNs blocked from docs review.
2. **SNS stuck-state**: `reconcile_standard_name_sources()` revives 'stale' SNS; no clean reset path. Required DETACH DELETE workaround.
3. **`uncertainty_index_of_*` leak**: still 7% post-Phase-C gate; needs compose-prompt ban.

## Prompt gaps (compose_system_lean.md)
- BANNED additions: `uncertainty_index`, `reconstruction`, `measured`, `inferred`, `_from_`, `_to_` prepositions
- Flux-surface locus must appear as POSITION specifier (not qualifier prefix) — 28% of gyrokinetics
- Multi-concept batching produces nonsensical names — composer must reject when paths semantically incompatible

## Prompt gaps (enrich_system.md + enrich_user.md)
- Dimensionless-index rule (uncertainty_index_of_* → "Dimensionless integer index" + flag DD unit inconsistency)
- GGD container rule (grid_object_*/grid_element_* → describe access pattern, not enumerate)
- Cross-reference inline-link format `[name](name:bare_id)`
- Anti-speculation for calibration parameters (no Jones-matrix without sources)
- Ban "typically" hedging
- Descriptions must respect grammar segments (no Z-axis if axis-agnostic, no "normalized" if not in grammar, no COCOS handedness unless flagged)

## Vocab additions (queue for ISN PR)
- Position (11): from_wall, from_plasma, outboard_midplane, inner/outer_divertor_target, divertor_target, limiter/mobile_unit/vessel_outline_point, diagnostic_component_center, detector_aperture
- Subject (1): gyroaveraged
- Geometric_base (2): x1_width, x2_width
- Coordinate_axes: annular, vertical_coordinate_of
- Component: cartesian_x

## Wave 8 dispatch
- 8A engineer: revised-name fix + SNS reset CLI
- 8B engineer: compose + enrich prompt hardening
- 9 (after 8): re-run rotations to verify lift
