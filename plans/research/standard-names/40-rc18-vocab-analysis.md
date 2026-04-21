# rc18 VocabGap analysis

**Date**: 2026-04-21 (session snapshot)
**Source**: `MATCH (vg:VocabGap) ...` on the codex graph, 999 rows total.
**Goal**: classify every gap and drive an ISN rc18 release that closes the ones that can be closed with vocabulary additions.

## Summary totals

| Classification                                                  | Count | Action                                     |
|------------------------------------------------------------------|------:|--------------------------------------------|
| TRUE vocab gap (enumerated segments, needs ISN token)           |    88 | Add to rc18 YAML                           |
| COMPOSITION OVERRUN (`physical_base` packs multiple segments)   |   715 | Reviewer/composer-prompt concern, not ISN  |
| SYNC ISSUE (transformation tokens exist in YAML but not in graph)|    39 | Codex grammar-sync / SEGMENT_ORDER concern |
| GRAMMAR AMBIGUITY (`diamagnetic`, real/imag prefix chains)      |    15 | Deferred — senior-review ruling pending    |
| BAD NAMING (speculative one-off compounds below signal threshold)|   142 | Classifier-exclusion / prompt tightening   |
| **Total**                                                       | **999** |                                          |

### Why `physical_base` cannot be an ISN "gap"

ISN specification.yml explicitly defines `physical_base` as **open-ended** and
controlled via catalog entries, not grammar vocabulary:

> `physical_base`: Physical quantity, measurement, or hardware property
> (open-ended). … New physical base terms are defined through catalog entries,
> not the grammar.

`SEGMENT_TOKEN_MAP["physical_base"]` is `()` in
`imas_standard_names/grammar/constants.py`. Consequently, the codex token-miss
detector's `OPTIONAL MATCH (t:GrammarToken {segment:'physical_base'})` always
misses, and every used physical_base token becomes a VocabGap node.

The 715 physical_base rows fall into two qualitative buckets:

- **COMPOSITION OVERRUN** (majority): the composer stuffed multiple grammar
  concepts into a single base term, e.g.

  - `wave_absorbed_power_per_toroidal_mode` → physical_base=`wave_absorbed_power`
    + transformation=`per_toroidal_mode`
  - `wave_absorbed_power_density_per_toroidal_mode` → physical_base=
    `wave_absorbed_power_density` + transformation=`per_toroidal_mode`
  - `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode`
    → component prefix + physical_base + transformation
  - `perturbed_magnetic_field_imaginary_part` → transformation=`imaginary_part_of`
    + physical_base=`perturbed_magnetic_field`
  - `collisional_toroidal_torque_density` → OK as base, but torque_density
    already exists
  - `diamagnetic_drift_velocity`, `exb_drift_velocity`, `perturbed_plasma_velocity`
    → already expressible via component + base

  → **Fix side**: tighten the R1/R2 reviewer prompts (imas-codex) to forbid
  transformation suffixes from being folded into physical_base.

- **LEGITIMATE SELF-CONTAINED BASES** that the graph cannot recognise because
  physical_base has no token list. These are not ISN grammar gaps; they are
  catalog candidates. Top-20 examples:
  - `reference_waveform` (41) — metadata wrapper, not a physics quantity;
    reject as BAD NAMING.
  - `magnetic_flux` (30) — valid; already usable as a physical_base today.
  - `atomic_mass` (15) — valid base; not a grammar-vocabulary concern.
  - `halo_current` (9), `particle_source_rate` (7), `surface_current_spectrum`
    (7), `non_axisymmetric_magnetic_field` (7), `constraint_weight` (10),
    `transport_boundary_condition_value` (9), `accumulated_gas_injection` (8)
    — all acceptable physical bases the composer is free to use.

### Transformation segment (39 rows, all SYNC ISSUE)

All 39 transformation gaps are tokens **already present** in
`imas_standard_names/grammar/vocabularies/transformations.yml`:

| needed_token                                    | present since | count |
|-------------------------------------------------|---------------|------:|
| `per_toroidal_mode`                             | rc12          | 13    |
| `per_toroidal_mode_number`                      | rc12          | 12    |
| `normalized`                                    | pre-rc9       | 8     |
| `second_derivative_of`                          | rc16          | 7     |
| `cumulative_inside_flux_surface`                | rc12          | 6     |
| `real_part` / `imaginary_part`                  | (of-suffixed) | 11    |
| `flux_surface_averaged`                         | pre-rc9       | 3     |
| `derivative_with_respect_to_rho_tor`            | rc16          | 1     |
| `time_integrated`, `volume_averaged`, …         | pre-rc9       | 5     |

Root cause: `SEGMENT_ORDER` in `imas_standard_names/grammar/constants.py` does
**not** include `transformation`. `get_grammar_graph_spec()` iterates only
SEGMENT_ORDER, so `GrammarToken{segment:'transformation'}` nodes are never
written to the graph. Every transformation a composer uses therefore registers
as a token-miss even though the token is in canonical ISN vocabulary.

→ **Fix side**: codex-side `sync_isn_grammar` (or ISN's `spec.py`) must opt
`transformation` into the exported segment list. Out of scope for this rc18
cut; logged in this analysis for follow-up.

### Grammar ambiguity (15 rows)

All variants of `diamagnetic_*` plus a couple of velocity_tor / velocity_phi
complaints. Senior review D.3 and D.5 both **deferred** the `diamagnetic`
token pending parser refactoring (see header of
`vocabularies/components.yml`). No action in rc18.

## rc18 additions (TRUE vocab gaps)

Selected from enumerated-segment gaps with `example_count ≥ 2` (and a small
set of high-confidence singletons) where the token is clearly a physics-valid
vocabulary extension, not composer noise.

### subjects.yml (+23 tokens)

Fast-ion / fast-particle orbital-class compositions that the composer
consistently emits as atomic subjects (IMAS DD distribution_sources,
distributions, core_fast_particles):

- `ion_state`, `fast_ion_state`, `ion_charge_state`, `neutral_state`
- `trapped_fast_ion`, `co_passing_fast_ion`, `counter_passing_fast_ion`
- `trapped_fast_particle`, `co_passing_fast_particle`, `counter_passing_fast_particle`
- `trapped_particle`
- `gyrocenter`
- `halo`, `pellet`
- `shattered_pellet`, `shattered_pellet_fragment`, `shattered_pellet_species`
- `hard_xray`
- `bulk_plasma`, `total_plasma`
- `ammonia`, `silane` (recycling / diagnostic gas species)
- `edge_localized_mode`

### processes.yml (+10 tokens)

- `disruption_event`
- `wave_particle_interaction`
- `fast_particle_thermalization`
- `wave_driven_current_drive`
- `conductive_losses`
- `fast_particle_source`
- `distribution_function_driven`
- `radiation_emission`
- `recombination_emission`
- `fusion_born_alpha`

### positions.yml (+25 tokens; shared by position & geometry segments)

- `halo_region`, `divertor_plate`
- `pellet_path`, `pellet_path_point`, `pellet_path_first_point`,
  `pellet_path_second_point`, `along_pellet_path`, `shattering_position`
- `ntm_onset`, `neoclassical_tearing_mode_onset`
- `beam_path`, `along_beam_path`
- `ece_channel`, `ece_channel_emission_position`
- `outboard_midplane_first_wall`, `outboard_midplane_separatrix`,
  `first_wall_midplane`
- `ggd_grid_point`, `ggd_grid_node`, `ggd_grid_subset`, `ggd_element`,
  `ggd_node`
- `inlet`, `outlet`, `cooling_circuit_inlet`

### objects.yml (+14 tokens)

- `breeder_blanket_module`, `breeder_blanket_layer`, `breeder_blanket_shield`
- `beamlet_group`
- `working_fluid`, `filter`, `temperature_sensor`, `optical_element`
- `mass_spectrometer`, `mass_spectrometer_channel`, `spectrometer_channel`,
  `neutron_detector_converter`
- `ec_launcher_mirror` (alias for `electron_cyclotron_launcher_mirror`)
- `diagnostic_antenna`
- `port`

### geometric_bases.yml (+1 token)

- `unit_vector`

Total: **73 new tokens** targeting **~180 gap rows** (~18% of all VocabGaps,
~78% of actionable enumerated-segment gaps).

## Out-of-scope follow-ups (tracked here, not executed)

1. **Transformation segment graph sync** (39 gaps). Add `transformation`
   to `SEGMENT_ORDER` in ISN `grammar/constants.py`, regenerate
   `GrammarSpec`, teach codex's `sync-isn-grammar` to upload transformation
   tokens. Closes the rc16+ transformation-existing false-positives.

2. **physical_base composition-overrun prompt fix** (≈500 gaps). Update
   reviewer R1/R2 rubrics to REJECT physical_base tokens that contain
   known transformation suffixes (`_per_toroidal_mode`, `_real_part`,
   `_imaginary_part`, `_amplitude`, `_phase`, `_density_per_*`).

3. **Classifier exclusion for container/metadata fields** (`reference_waveform`
   × 41, `constraint_weight` × 10, `constraint_measurement_point`,
   `transport_boundary_condition_value`). These IMAS paths are scaffolding,
   not physics quantities, and should not emit standard names at all.

4. **Senior review resume on `diamagnetic`** — grammar-ambiguity cluster.
