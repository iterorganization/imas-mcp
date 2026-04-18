# D.3 Senior Physics Review — 7-Domain Rotation (2026-04)

**Reviewer role:** Senior tokamak physicist + IMAS Standard-Names co-author.
**Scope:** 438 StandardName nodes produced by the fresh D.2 rotation across 7 physics
domains (`equilibrium`, `magnetohydrodynamics`, `transport`, `edge_plasma_physics`,
`magnetic_field_diagnostics`, `auxiliary_heating`, `turbulence`).
**Evidence base:** `plans/features/standard-names/reviews/D3-data.json` (graph snapshot
exported 2026-04 via `imas-codex-repl`). Raw corpus = canonical source; quotations
below are verbatim `sn.id` values.
**Model under review:** all 438 names composed by
`openrouter/anthropic/claude-sonnet-4.6` (plan-29 §D.2 winner).

> **Status gate:** This review closes plan 29 §D.3 and conditionally approves the
> transition to §D.4 *only* if the HIGH-ROI actions in §4 are executed first. The
> corpus is publishable as a *draft* vocabulary, not yet as an authoritative
> controlled vocabulary.

---

## 1. Graph snapshot

| Metric | Value |
| --- | ---: |
| Total StandardName nodes | **438** |
| Valid (validation_status) | **335** (76.5%) |
| Quarantined | **52** (11.9%) |
| Pending (attached / not validated) | **51** (11.6%) |
| Mean `validation_issues` / name | **0.16** |
| `cocos_transformation_type` populated | **16 / 438** (3.7%) — *plan target ≥35* |
| StandardNameSource nodes | 1 359 (628 extracted, 599 composed, 109 attached, 23 vocab_gap) |
| VocabGap nodes | **290** (physical_base 169, geometry 26, object 22, position 22, subject 17, transformation 12, process 11, grammar_ambiguity 5) |
| `reviewer_score` populated | **0 / 438** — `sn review` has not yet run (see §6) |
| `documentation` populated | **0 / 438** — D.2 is name-only; enrichment is D.4 |

### 1.1 Per-domain distribution

| Domain | Total | Valid | Quar | Pend | Issues/name | J ≥ 0.75 pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary_heating | 95 | 78 | 5 | 12 | 0.05 | **51** ← synonym cluster |
| edge_plasma_physics | 75 | 51 | 5 | 19 | 0.12 | 2 |
| equilibrium | **107** | 84 | **23** | 0 | **0.32** ← worst | 20 |
| magnetic_field_diagnostics | **4** | 4 | 0 | 0 | 0.00 | 6 |
| magnetohydrodynamics | 83 | 64 | 13 | 6 | 0.16 | **36** |
| transport | 63 | 49 | 6 | 8 | 0.14 | 7 |
| turbulence | **11** | **5** | 0 | 6 | 0.09 | 1 |

### 1.2 Headline findings

* **Synonym density is too high.** Jaccard ≥0.75 pair count = 123 across 335 valid
  names (36.7 per 100). Two domains account for 87 of the 123 pairs:
  `auxiliary_heating` (51) and `magnetohydrodynamics` (36). This is the *opposite*
  of the plan-29 target (§7.a "≥30% reduction in Jaccard cluster density"). The
  "one decomposition, many subjects" refactor is deferred.
* **Equilibrium is the weakest domain.** Issue density 0.32/name is 2× the corpus
  mean, almost entirely from DD-path leakage
  (`equilibrium_reconstruction_constraint_weight_of_*`, `measurement_time`, g-index
  metric components). Quarantine rate 23/107 = 21%.
* **Two domains are under-populated.** `magnetic_field_diagnostics` has only 4
  names (Stokes parameters), and `turbulence` has only 5 valid names. These
  domains were effectively skipped by the extractor — re-running D.2 targeted on
  these domains is the highest-coverage win.
* **Name-only mode works.** Unit auditor quarantines are reliable; the compose
  prompt respects grammar structure; cross-domain `attached` deduplication is
  working (109 sources attached without LLM re-generation).
* **Documentation quality cannot be audited.** Spot-audit of 20 random names
  confirmed `description` and `documentation` are empty (D.2 `--name-only`
  rotation). Plan-29 §7.e (documentation rubric ≥0.75) can only be evaluated
  after D.4.
* **Reviewer score cannot be audited.** No `sn review` run on this rotation;
  plan-29 §7.b (reviewer rubric ≥0.75 vs 0.68 baseline) must be deferred. Below
  we use `validation_issues` density + qualitative physicist judgement as proxy.

---

## 2. Per-domain audit

For each domain: a short diagnosis, 5–10 **strong** names (idiomatic, ISN-grammar
clean), 5–10 **weak** names (anti-pattern with NC-tag), identified vocabulary gaps
(from VocabGap + quarantine corpus), and grammar-shortcoming diagnosis.

### NC (Name-Criticism) tag legend

| Tag | Anti-pattern |
| --- | --- |
| NC-1 | Combinatorial explosion — same physics expressed as N separate names instead of one base × decomposition |
| NC-2 | Tautology / repeated token (e.g. `_beam_injector_beam`) |
| NC-3 | Provenance / solver-semantic leak (`measured_`, `reconstructed_`, `explicit_`, `implicit_`) |
| NC-4 | DD-path leakage (name reads like an IDS path, e.g. `equilibrium_reconstruction_constraint_weight_of_*`) |
| NC-5 | Wrong `kind` — vector expressed as scalar or vice-versa |
| NC-6 | Abbreviation / jargon (`ntm`, `norm`, `exb`) |
| NC-7 | Name/unit mismatch (e.g. a Tesla-valued quantity with unit `m^-1.V`) |
| NC-8 | COCOS metadata missing (T/Wb/Pa/A quantity with `cocos_transformation_type` null) |

---

### 2.1 Equilibrium (107 nodes · 84 valid · 23 quar · issues/name 0.32)

**Diagnosis.** Worst domain by every metric: highest issue density, highest
quarantine rate, largest DD-path-leakage footprint. The compose prompt has
allowed the LLM to lift IDS structural names verbatim
(`equilibrium_reconstruction_constraint_*`, `iron_core_segment_*_measurement_time`,
`g11_contravariant_metric_tensor_component`). These are data-model identifiers,
not physics quantities.

**Strong (exemplars to keep).**

1. `poloidal_magnetic_flux_at_magnetic_axis` — `Wb`, clean `_at_` for
   named feature (fixed-point).
2. `safety_factor_at_magnetic_axis` — `1`, canonical scalar.
3. `minor_radius_of_flux_surface` — `m`, well-scoped geometry.
4. `major_radius_of_magnetic_axis` — `m`, canonical.
5. `flux_surface_averaged_plasma_pressure` — `Pa`, correct transformation prefix.
6. `normalized_poloidal_flux_coordinate` — `1`, proper radial coordinate.
7. `toroidal_magnetic_field_at_magnetic_axis` — `T`, exemplar.
8. `lower_triangularity` / `upper_triangularity` — `1`, dimensionless shaping.
9. `plasma_volume` — `m^3`, minimal clean scalar.
10. `vertical_coordinate_of_magnetic_axis` — `m`, canonical component.

**Weak (anti-patterns).**

1. `equilibrium_reconstruction_constraint_weight_of_poloidal_magnetic_field_probe`
   — **NC-4** (DD-path leak) + **NC-1** (one of 8 identical `_constraint_weight_of_*`
   variants).
2. `equilibrium_reconstruction_constraint_flux_loop_measurement_time` — **NC-4**,
   `measurement_time` is diagnostic provenance, not a physical quantity.
3. `g11_contravariant_metric_tensor_component` … `g33_*` — **NC-2/NC-5**: the
   index `11…33` is a *decomposition axis*, not part of the name. Canonical form
   would be one base name `contravariant_metric_tensor` with a
   `(flux_coordinate, flux_coordinate)` index decomposition.
4. `upper_triangularity_of_plasma_boundary` duplicates `upper_triangularity` —
   **NC-1** (the unqualified name already means the plasma boundary).
5. `radial_component_of_iron_core_segment_magnetization_constraint_measurement_time`
   — **NC-4** + **NC-7** concern (unit is `s` but name reads like a field value).
6. `measured_rotational_pressure_constraint` / `reconstructed_rotational_pressure_constraint`
   — **NC-3** (provenance leak); `measured_` and `reconstructed_` encode
   *equilibrium solver I/O role*, not physics.
7. `equilibrium_reconstruction_constraint_measured_vertical_coordinate_of_x_point`
   — **NC-3 + NC-4**, reads like `equilibrium/time_slice/constraints/x_point/.../measured`.
8. `maximum_magnetic_field_of_flux_surface` / `minimum_magnetic_field_of_flux_surface`
   — **NC-5**: current `kind=scalar`, but the underlying quantity is a
   flux-surface *extremum of a field magnitude* — acceptable as scalar but
   `_magnetic_field_magnitude_` wording would be clearer.
9. `poloidal_magnetic_field_probe_constraint_weight` — **NC-4**, a data-model
   weight, not a physical observable.
10. `plasma_filament_current` — **NC-5** candidate (filament current in a
    multi-filament reconstruction is a *vector over filament index*; marking as
    `scalar` forces consumers to re-decorate the index).

**Vocabulary gaps surfaced.**

* **Process:** (none in this domain — good.)
* **Position:** `inside_flux_surface` appears in both `geometry` and `position`
  segments (ambiguity — ISN has no canonical assignment).
* **Transformation:** `variation_of` used as a free prefix but not in
  canonical transformations.
* **Subject/object:** `iron_core_segment`, `passive_structure`, `passive_loop`,
  `flux_loop`, `x_point`, `strike_point` — all DD machine-description objects
  being conflated with physical quantities.

**Grammar shortcomings.** ISN has no segment for "reconstruction-solver
provenance" (measured-vs-reconstructed), nor for "diagnostic observable vs
reconstructed estimator". These concepts are *metadata about a timeseries*,
not parts of a standard name. The composer needs an explicit rule to refuse
any name containing `measured_`, `reconstructed_`, `constraint_weight_`,
`measurement_time`, or `_constraint` as a suffix.

---

### 2.2 Magnetohydrodynamics (83 nodes · 64 valid · 13 quar · issues/name 0.16)

**Diagnosis.** The second most populous domain. Grammar is generally sound,
but Fourier/mode-decomposition is handled ad-hoc as a `per_toroidal_mode(_number)`
suffix rather than a `transformation` segment — this is the root cause of the
36 J≥0.75 pairs (second-worst synonym density). ISN ADR-3 R1 F3 (Fourier
decomposition as a first-class segment) is the pending blocker.

**Strong.**

1. `perturbed_electron_temperature_eigenfunction_real_part` / `_imaginary_part` —
   canonical complex-amplitude pair, ADR-3 R3 compliant.
2. `perturbed_radial_component_of_magnetic_field_real_part` — clean.
3. `seed_island_full_width` (valid quarantine recovery) — good descriptor.
4. `poloidal_mode_number` / `toroidal_mode_number` — `1`, canonical decomp tag.
5. `resistive_wall_mode_growth_rate` — `s^-1`, clean.
6. `neoclassical_tearing_mode_saturated_island_width` — `m`, clean.
7. `poloidal_component_of_vacuum_perturbed_magnetic_field_imaginary_part` — `T`,
   clean compound but verbose (candidate for shortening if `vacuum` becomes a
   decomposition tag).
8. `sawtooth_period` — `s`, canonical.
9. `plasma_displacement_eigenfunction` — good base.
10. `edge_localized_mode_frequency` — `Hz`.

**Weak.**

1. `ntm_mode_onset_time_offset` — **NC-6** (`ntm` = "neoclassical tearing mode")
   + **NC-2** (`ntm_mode` = "neoclassical tearing mode mode").
2. `neoclassical_tearing_mode_detailed_evolution_time` vs
   `neoclassical_tearing_mode_onset_time` — inconsistent phrasing; the first
   reads like a DD path. **NC-4**.
3. `mode_rotation_frequency` — too generic; qualifier (what mode?) is missing.
4. `perturbed_plasma_velocity_eigenfunction` — no `_real_part`/`_imaginary_part`
   suffix — violates ADR-3 R3 for complex-valued eigenfunctions.
5. `poloidal_component_of_perturbed_magnetic_field_imaginary_part` vs
   `perturbed_poloidal_component_of_magnetic_field_imaginary_part` — both exist,
   different subject-ordering — **NC-1**.
6. `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode`
   — **NC-1**: LH/RH polarisation should be a `decomposition(polarisation)` axis,
   not part of the name.
7. `neoclassical_toroidal_viscosity_stress_tensor_imaginary_part` — tensor being
   marked as a scalar; no rank/index metadata.
8. `reynolds_stress_tensor_real_part` / `maxwell_stress_tensor_real_part` —
   tensors being named as scalars (no index axis decomposition).
9. `plasma_displacement_imaginary_part` with no `_eigenfunction_` qualifier —
   inconsistent with `plasma_pressure_eigenfunction_imaginary_part`.
10. `perturbed_electromagnetic_super_potential_imaginary_part` — "super_potential"
    is non-standard (usually "Hertz potential" or "magnetic-vector-potential").
    **NC-6** (jargon).

**Vocabulary gaps.**

* **Process:** `neoclassical_tearing_mode`, `resistive_dissipation`,
  `j_cross_b_force`, `bootstrap`.
* **Transformation:** `real_part_of_perturbed`, `per_toroidal_mode`,
  `per_toroidal_mode_number`, `normalized`, `volume_averaged`.
* **Grammar ambiguity:** `diamagnetic`, `velocity_tor vs velocity_phi`.
* **Object:** `neoclassical_tearing_mode` recurs as a subject/object token.

**Grammar shortcomings.** Fourier decomposition is the single biggest missing
piece. Until ISN adds a `decomposition(toroidal_mode_number, poloidal_mode_number)`
first-class axis, the composer will keep inventing `_per_toroidal_mode_*`
variants. Similarly, `_real_part` / `_imaginary_part` should be ADR-3 R3
enforced on *every* `perturbed_*_eigenfunction` name.

---

### 2.3 Transport (63 nodes · 49 valid · 6 quar · issues/name 0.14)

**Diagnosis.** Decent grammar, moderate synonym density, but the domain is
polluted with solver-semantic leak (`explicit_`, `implicit_part_of_`). These
are numerics bookkeeping, not physics.

**Strong.**

1. `volume_averaged_ion_temperature` — canonical.
2. `electron_energy_source_density` — `m^-3.W`, canonical source density.
3. `thermal_ion_temperature` / `electron_temperature` — canonical scalars.
4. `particle_diffusivity` / `thermal_diffusivity` — canonical transport coeffs.
5. `bootstrap_current_density` — `A.m^-2`, canonical.
6. `plasma_current_due_to_bootstrap` — `A`, clean `due_to_` phrasing.
7. `total_thermal_pressure` — clean.
8. `toroidal_component_of_plasma_rotation_velocity` — `m.s^-1`, canonical.
9. `poloidal_current_function` (`f = R·Bt`) — canonical shape name.
10. `electron_collisionality` — `1`.

**Weak.**

1. `explicit_electron_energy_source_density` / `explicit_ion_energy_source_density`
   — **NC-3** (solver semantic leak). The `explicit_` prefix means "treated
   explicitly in the time-integration scheme" — that is numerical-method
   metadata, not physics.
2. `implicit_part_of_electron_energy_source` / `implicit_part_of_ion_particle_source`
   — same **NC-3**; these belong as solver annotations on a scheme object.
3. `total_thermal_ion_pressure` vs `electron_thermal_pressure` — subject ordering
   is reversed (`total_thermal_ion_` vs `electron_thermal_`). **NC-1**.
4. `plasma_current_due_to_bootstrap` vs `bootstrap_current_density` — double-named
   bootstrap (one with `_due_to_`, one with flat compound) — **NC-1**.
5. `tendency_of_electron_number` / `tendency_of_ion_number` — "tendency" is
   CF-Conventions terminology but ISN has not adopted it; looks out of place.
   **NC-6**.
6. `neutral_diamagnetic_drift_velocity` — **NC-5** (vector, currently `scalar`).
7. `ion_to_electron_density_ratio` — non-canonical expression; should be a
   derived quantity, not a named one.
8. `cumulative_ionization_potential` — DD-path-sounding. **NC-4**.
9. `plasma_filament_current` ambiguous (transport vs equilibrium domain — same
   name in both).
10. `ohmic_energy` — should be `ohmic_dissipation_energy`, and `ohmic` should be
    a process token.

**Vocabulary gaps.** `bootstrap`, `resistive_flux_consumption`, `wave_driven`
(process). `diamagnetic_drift_velocity`, `exb_drift_velocity` (physical_base —
both really want `{diamagnetic, exb}_drift` as a process with `velocity` base).

**Grammar shortcomings.** The composer should *reject* any name containing
`explicit_` or `implicit_part_of_` because both encode solver-decomposition.
The ISN-side fix is to add a `decomposition(solver_scheme)` so numerical
bookkeeping lives out-of-band.

---

### 2.4 Edge plasma physics (75 nodes · 51 valid · 5 quar · issues/name 0.12)

**Diagnosis.** Solid domain. `component_of_*` is consistently used; unit
handling is correct. Main weaknesses: viscosity-current nomenclature conflates
different physical processes, and diamagnetic/ExB drifts appear as base names
rather than `{drift}_{velocity}` composites.

**Strong.**

1. `poloidal_component_of_vorticity_over_major_radius` — `m^-1.s^-1`, exemplary
   composite geometry.
2. `radial_component_of_pfirsch_schlueter_current_density` — `A.m^-2`, clean
   use of eponymous process (well-known PS current).
3. `poloidal_component_of_perpendicular_viscosity_current_density` — clean.
4. `vertical_component_of_ion_diamagnetic_drift_velocity` — canonical composite
   (but see NC below for `diamagnetic` gap).
5. `plasma_mass_density` — `kg.m^-3`, canonical.
6. `ion_number_density` — canonical.
7. `parallel_component_of_plasma_velocity` — canonical.
8. `vertical_component_of_current_density` — canonical.
9. `radial_component_of_magnetic_field` — canonical.
10. `poloidal_magnetic_flux` — canonical (pending-status but clean).

**Weak.**

1. `radial_component_of_current_density_due_to_viscous` — **NC-3** ("viscous" is
   an adjective, not a process noun); should be
   `radial_component_of_viscous_current_density` or decompose into
   `{parallel_viscosity, perpendicular_viscosity}` processes.
2. `diamagnetic_heat_viscosity_current_density` — conflates "diamagnetic" and
   "viscosity"; should probably be two separate currents.
3. `inertial_current_density` — ambiguous without qualifier (which fluid
   component? Species?).
4. `halo_current` — "halo" is used as both region and process — ISN vocab gap.
5. `parallel_viscosity_current_density` vs `perpendicular_viscosity_current_density`
   — inconsistent with companion names using `_component_of_*_viscosity_current_density`.
6. `vorticity_over_major_radius` — mathematical derived form; prefer a named base
   (e.g. `vorticity` with a `divided_by(major_radius)` decomposition) over the
   implicit `_over_` convention.
7. `exb_drift_velocity` — **NC-6** (`exb` abbreviation for "E cross B").
8. `diamagnetic_drift_velocity` — **NC-5** (vector, marked `scalar`); also
   `diamagnetic` is a grammar-ambiguity vocab gap.
9. `parallel_velocity_over_magnetic_field` — same `_over_` convention issue as
   (6); should be a derived ratio.
10. `neutral_diamagnetic_drift_velocity` — same as (8) for neutrals.

**Vocabulary gaps.** `diamagnetic` (both as adjective and process-role),
`parallel_viscosity`, `perpendicular_viscosity`, `heat_viscosity`,
`j_cross_b_force`, `halo` (subject/region), `halo_region` (region token).

**Grammar shortcomings.** The `_over_X` convention (e.g. `vorticity_over_major_radius`)
is not formally specified — it is a de-facto division operator. ISN should
either formalise it as a `decomposition(divided_by=...)` or deprecate it in
favour of derived-quantity handling.

---

### 2.5 Magnetic field diagnostics (4 nodes · 4 valid · 0 quar · issues/name 0.00)

**Diagnosis.** Severely under-populated. The domain contains 4 names, all
Stokes parameters for fibre-optic current sensors. Key magnetics observables
(pickup-coil signals, flux loops, Rogowski coils, Mirnov coils, diamagnetic
loops) are either absent or (in the case of flux loops/probes) living in the
`equilibrium` domain as `_constraint_*` names — a serious domain-assignment
failure.

**All 4 names (all `stokes_sN_parameter_of_fiber_optic_current_sensor`, unit `1`, N=0..3).**

Strong/weak distinction is moot for 4 names. The Stokes naming is idiomatic.

**Vocabulary gaps.** `poloidal_magnetic_field_probe_measured`,
`poloidal_magnetic_field_probe_reconstructed`, `flux_loop_voltage`, `rogowski_coil_current`,
`mirnov_coil_signal`, `diamagnetic_loop_flux_difference`, `hall_probe_field` —
none of these exist. `ips_delta_faraday_rotation_angle`, `poincare_sphere_*`
surfaced in VocabGap but were not composed.

**Grammar shortcomings.** The `equilibrium` domain is absorbing diagnostics
that should live here. The extractor is classifying by DD-path physics_domain
tag (which puts magnetics-probe constraints under `equilibrium`), not by
semantic role. Either: (i) re-map certain `equilibrium/time_slice/constraints/*`
DD paths to `magnetic_field_diagnostics`, or (ii) widen the extractor query
for `magnetic_field_diagnostics` to pull from `magnetics` IDS directly.

---

### 2.6 Auxiliary heating (95 nodes · 78 valid · 5 quar · issues/name 0.05 · **J≥0.75 pairs: 51**)

**Diagnosis.** Lowest issue density (0.05) — individual names are clean — but
the **worst synonym-cluster density** in the corpus. The `wave_absorbed_power`
family spans 19 variants generated by combinatorial expansion across
`{electron, ion, fast_ion, fast_electron, ∅}` × `{density, inside_flux_surface,
flux_surface_averaged, per_toroidal_mode, ∅}` × {absorbed_power / power}. This
is the most actionable refactor opportunity in the entire corpus.

**Strong (clean base cases).**

1. `wave_absorbed_power_density` — `m^-3.W`, canonical (root of the family).
2. `flux_surface_averaged_wave_absorbed_power_density` — clean transformation.
3. `toroidal_angle_of_lower_hybrid_antenna` — `rad`, canonical geometry.
4. `tangency_radius_of_neutral_beam_injector_beam` — `m` (but see NC-2).
5. `vertical_component_of_wave_vector` — `m^-1`, canonical.
6. `neutral_beam_injector_beam_energy` — `eV`, canonical (rename suggested).
7. `poloidal_component_of_wave_electric_field_real_part` — canonical.
8. `fast_ion_wave_absorbed_power_density` — exemplar of species decomposition.
9. `electron_wave_absorbed_power_density` — ditto.
10. `lower_hybrid_wave_power_density_spectrum` — spectrum-specific, canonical.

**Weak.**

1. **The whole `wave_absorbed_power` family** — 19 variants where 3–4 would
   suffice with proper decomposition axes:
   ```
   wave_absorbed_power  [W]          wave_absorbed_power_density  [m^-3.W]
   wave_absorbed_power_per_toroidal_mode  [W]
   wave_absorbed_power_density_per_toroidal_mode  [m^-3.W]
   wave_absorbed_power_inside_flux_surface  [W]
   flux_surface_averaged_wave_absorbed_power_density  [m^-3.W]
   + each multiplied by {electron, ion, fast_ion, fast_electron} species
   + one cross-combination flux_surface_averaged_X_wave_absorbed_power_density_per_toroidal_mode
   ```
   Canonical form: **one** base name `wave_absorbed_power_density` with
   decomposition axes `species`, `toroidal_mode_number`, and transformation
   `{∅, flux_surface_averaged, inside_flux_surface}`. **NC-1.**
2. `tangency_radius_of_neutral_beam_injector_beam` — **NC-2**: "beam injector
   beam" → prefer `of_neutral_beam` (the injector is the object, the beam is
   the subject).
3. `binormal_component_of_wave_electric_field` — unit `m^-1.V` (OK for E-field,
   but inconsistent with same-family `T` units elsewhere).
4. `right_hand_circularly_polarized_electric_field_amplitude_per_toroidal_mode`
   / `left_hand_circularly_polarized_*` — LH/RH polarisation as part of the name;
   should be `decomposition(polarisation)`. **NC-1**.
5. `electric_field_phase_per_toroidal_mode` vs `electric_field_amplitude_per_toroidal_mode`
   — amplitude/phase encoded in name; should be complex-amplitude pair with
   `_real_part`/`_imaginary_part`.
6. `toroidal_angle_of_electron_cyclotron_launcher_beam_launching_position` —
   **NC-4** (long DD-path lift).
7. `ion_cyclotron_heating_antenna_surface_current_spectrum` — **NC-5** (spectrum
   is 1-D array over frequency / wavenumber; marked `scalar`).
8. `ec_launcher_mirror` / `ec_launcher_launching_position` — **NC-6** (`ec`
   abbreviation).
9. `beam_tracing_electric_field_component` — **NC-4** (reads like a DD path).
10. `phase_ellipse_rotation_angle` / `spot_ellipse_rotation_angle` / `tilt_angle_variation`
    — beam-shaping jargon without clear physics grounding.

**Vocabulary gaps.** `per_toroidal_mode`, `per_toroidal_mode_number`,
`electron_cyclotron_beam_launching_position`, `electron_cyclotron_launcher_mirror`,
`ion_cyclotron_heating_antenna`, `lower_hybrid_antenna`, `neutral_beam_injector`,
`beam_tracing_*`, `polarisation_ellipticity_angle`, `o_mode_fraction`,
`ordinary_mode_fraction`, `stokes_sN_parameter`, `refractive_index`,
`poloidal_refractive_index`, `toroidal_refractive_index`, `wave_driven` (process).

**Grammar shortcomings.** Two big missing axes:

* `decomposition(species)` — would collapse the `{electron, ion, fast_ion,
  fast_electron}` proliferation.
* `decomposition(toroidal_mode_number, poloidal_mode_number)` (ADR-3 R1 F3) —
  would collapse `_per_toroidal_mode(_number)` variants.
* `decomposition(polarisation)` — would collapse LH/RH circular and O/X
  linear polarisation.

---

### 2.7 Turbulence (11 nodes · 5 valid · 6 pend · 0 quar · issues/name 0.09)

**Diagnosis.** Severely under-populated. 5 valid names is too few to even audit
properly — most expected turbulence quantities (fluctuation amplitudes,
correlation lengths, turbulent transport fluxes, turbulent Prandtl numbers,
Reynolds stress tensor as a turbulence quantity, cross-phase spectra) are
absent.

**All valid names.**

1. `element_atoms_count` — **misdomained** (this is a species/element metadata
   integer, not a turbulence quantity). **NC-4**.
2. `normalized_wave_vector` — generic, not turbulence-specific.
3. `wave_power_density_spectrum` — arguably turbulence, but better placed in
   auxiliary_heating.
4. `wave_electric_field` — ditto.
5. `vorticity` — arguably turbulence, but really a transport/edge quantity.

**Pending (attached — not re-validated).** `perturbed_electrostatic_potential_*`,
`plasma_displacement_imaginary_part` (should live in MHD).

**Vocabulary gaps.** Every core turbulence quantity is absent. The D.2
extractor evidently did not target the `edge_profiles/ggd` fluctuation fields,
`turbulence` IDS, or gyrokinetic code outputs (GENE, GKW, GYRO schemas).

**Grammar shortcomings.** Not applicable — the problem is *coverage*, not
grammar. Fix this by re-running D.2 targeted on `turbulence` and `gyrokinetics`
IDSs.

---

## 3. Cross-cutting findings

### 3.1 Systemic anti-patterns (count across corpus)

| Pattern | Count | Domains affected |
| --- | ---: | --- |
| NC-1 synonym proliferation (J≥0.75 pairs) | 123 | aux_heating, MHD, equilibrium |
| NC-3 solver-semantic leak (`measured_`, `reconstructed_`, `explicit_`, `implicit_part_of_`) | 14 | equilibrium, transport |
| NC-4 DD-path leakage (`equilibrium_reconstruction_constraint_*`, `ggd_object_*`, `iron_core_segment_*`) | 23 | equilibrium |
| NC-5 vector/scalar misclassification | ≥14 | edge, transport, equilibrium, aux_heating, mag_diag |
| NC-6 abbreviation/jargon (`ntm`, `ec`, `exb`, `norm`) | ≥10 | MHD, aux_heating, edge |
| NC-7 name/unit mismatch (caught by auditor) | 5 (quarantined) | aux_heating, equilibrium |
| NC-8 missing `cocos_transformation_type` | **422** | all (16/438 set) |

### 3.2 Why ADR-3 Fourier decomposition matters

49 names out of 438 (11%) contain `_per_toroidal_mode` or `_per_toroidal_mode_number`
as a free suffix. Every one of these would collapse under proper Fourier
decomposition into a `decomposition(toroidal_mode_number)` axis on its base name.
This is single-handedly responsible for:

* 51 / 51 Jaccard pairs in `auxiliary_heating`.
* 20+ / 36 Jaccard pairs in `magnetohydrodynamics`.
* The bulk of VocabGap entries in `transformation` and `position` segments.

**Until ADR-3 R1 F3 is implemented in ISN, the corpus cannot hit the plan-29
§7.a target (Jaccard density reduction ≥30%).**

### 3.3 Species decomposition

35+ names contain `{electron, ion, fast_ion, fast_electron, neutral}_` prefixed
on otherwise-identical bases. A first-class `decomposition(species)` axis
(or an equivalent convention that the composer can see) would similarly
collapse these.

### 3.4 COCOS under-population

Only 16/438 names have `cocos_transformation_type` set. Of the 422 missing,
the majority are scalar quantities that genuinely have `cocos_type=one_like`
and should be tagged as such (the extractor should default to `one_like` for
scalars with units `m`, `eV`, `Pa`, `kg.m^-3`, `s^-1`, etc.). T-/Wb-/A-valued
quantities need explicit audit against the COCOS sign-flip table.
**This is the lowest-effort lift for the largest metadata-coverage gain.**

### 3.5 Domain assignment failures

* `magnetic_field_diagnostics` is under-populated because magnetics probes
  and flux loops are being absorbed into `equilibrium` (via the `constraints`
  DD subtree).
* `turbulence` is under-populated because the extractor did not target
  fluctuation/gyrokinetic data models.
* `element_atoms_count` in `turbulence` is stray metadata.

### 3.6 Provenance / solver-decomposition leakage

`measured_`, `reconstructed_`, `explicit_`, `implicit_part_of_`,
`_measurement_time`, `_constraint_weight_of_` — all encode *metadata about how
the data was produced*, not physics. These should be rejected wholesale by the
compose prompt. Roughly **35 names** would collapse if this rule were enforced.

### 3.7 Tautologies and verbose compounds

Confirmed tautologies (same token repeated in one name):
`tangency_radius_of_neutral_beam_injector_beam` (beam×2),
`toroidal_angle_of_neutral_beam_injector_beam_tilting_offset` (beam×2),
`wave_driven_toroidal_current_per_toroidal_mode` (toroidal×2),
`flux_surface_area_derivative_with_respect_to_toroidal_flux_coordinate` (flux×2),
`flux_surface_averaged_inverse_major_radius_squared_grad_rho_squared` (squared×2),
`flux_surface_averaged_inverse_magnetic_field_squared_grad_rho_squared` (squared×2).

---

## 4. HIGH-ROI actions (ordered)

### 4.1 ⭐ #1 — ISN: add 15 vocabulary tokens to unblock ~30 quarantined + pending names

**File:** `imas-standard-names/imas_standard_names/grammar/vocabularies/` (split
by segment file). One PR. See §5 for the `proposed_vocab_additions.yaml` payload.

Estimated impact: unblocks 23+ currently-quarantined names and reduces
VocabGap hits by ~80.

### 4.2 ⭐ #2 — imas-codex: tighten compose prompt to reject DD-path / solver-semantic leakage

**File:** `imas_codex/llm/prompts/sn/compose_system.md`

Add an explicit **"REJECT" clause** listing forbidden name prefixes/suffixes:

```
REJECT any candidate name that contains any of the following tokens or substrings,
because they encode data-model structure or solver semantics, not physics:

Forbidden prefixes:
  - measured_
  - reconstructed_
  - explicit_
  - implicit_part_of_

Forbidden suffixes:
  - _measurement_time
  - _constraint
  - _constraint_weight
  - _constraint_weight_of_*

Forbidden tokens:
  - equilibrium_reconstruction_
  - ggd_object_
  - _constraint_reconstructed_
  - _constraint_measured_
  - ntm_ (use neoclassical_tearing_mode_)
  - ec_ (use electron_cyclotron_)
  - exb_ (use e_cross_b_ or decomposition(drift_type))
  - norm_ (use normalized_)

When a DD path would produce one of these, SKIP and record as vocab_gap rather
than composing.
```

Estimated impact: prevents regeneration of the 23 equilibrium NC-4 names + 14
NC-3 names on next rotation. This is the single highest-ROI prompt change.

### 4.3 ⭐ #3 — imas-codex: default `cocos_transformation_type` to `one_like` for scalar/unit-safe quantities

**File:** `imas_codex/standard_names/graph_ops.py` (in the graph-write path where
`cocos_transformation_type` is set, post-LLM injection).

For names with `kind=scalar` and units in `{1, m, m^2, m^3, eV, Pa, kg.m^-3,
s, s^-1, Hz, m^-3, m.s^-1, A.m^-2}`, default `cocos_transformation_type=one_like`
unless the extractor finds a more specific annotation on the DD node.

Estimated impact: pushes coverage from 16/438 → ~350/438 without any LLM
intervention; closes plan-29 §7.h gap.

### 4.4 #4 — imas-codex: re-run D.2 for `turbulence` and `magnetic_field_diagnostics`

**Command:**
```bash
imas-codex sn generate --source dd --domain turbulence -c 5
imas-codex sn generate --source dd --domain magnetic_field_diagnostics -c 5
```

Plus: extend extractor to pull `magnetics/bpol_probe`, `magnetics/flux_loop`,
`magnetics/rogowski_coil`, `magnetics/diamagnetic_flux`, `magnetics/b_field_*`
into `magnetic_field_diagnostics` (currently routed to `equilibrium` via the
constraint subtree).

Estimated impact: 5 → ~30 names in each under-populated domain; closes the
biggest coverage gap in the corpus.

### 4.5 #5 — imas-codex: re-generate `equilibrium` with the new prompt

After #2 ships, reset equilibrium with
```
imas-codex sn clear --status drafted --ids equilibrium
imas-codex sn generate --source dd --ids equilibrium
```

Estimated impact: issue density 0.32 → ~0.10 (in line with other domains);
quarantine rate 21% → ~5%.

### 4.6 #6 — ISN: specify ADR-3 R1 F3 (Fourier / mode-number decomposition)

**File:** `imas-standard-names/docs/adr/ADR-003-*.md` + grammar R3 implementation.

Formalise `decomposition(toroidal_mode_number, poloidal_mode_number)` as a
first-class axis. Update composer to emit `base_name + decomposition` rather
than `base_name_per_toroidal_mode`.

Estimated impact: ~49 names collapse into ~15; Jaccard density drops by ~40%
in auxiliary_heating and MHD.

### 4.7 #7 — ISN: add `decomposition(species)` for `{electron, ion, fast_ion, fast_electron, neutral, thermal_ion, thermal_electron}` subject axis

This collapses the species-prefix proliferation across all power/density
families.

### 4.8 #8 — Run `sn review` on current corpus

So that plan-29 §7.b (reviewer_score ≥0.75) becomes measurable. Currently
**not a single name** has a reviewer score.

```bash
imas-codex sn review -c 15
```

### 4.9 #9 — Audit vector/scalar `kind` for 14 flagged names

List in §2 — assign `kind=vector` and add a proper `component` decomposition
where appropriate (e.g. `diamagnetic_drift_velocity`).

### 4.10 #10 — ISN: formalise or deprecate the `_over_X` division convention

Names like `vorticity_over_major_radius`, `parallel_velocity_over_magnetic_field`
use `_over_` as an implicit division. Either formalise with a
`decomposition(divided_by=major_radius)` convention or deprecate and demand a
named quantity.

---

## 5. Proposed vocabulary additions

```yaml
# imas-standard-names/grammar/vocabularies/ updates
# Source: D.3 senior-review evidence from 52 quarantined names + 290 VocabGap
#         nodes in the 2026-04 corpus.

component:
  # Drift-type components
  - diamagnetic
  - e_cross_b                     # replaces `exb_`
  - pfirsch_schlueter
  # Spatial decomposition
  - normalized_radial
  - normalized_vertical
  - parallel_component_of_plasma  # already in composites but missing as token
  - perpendicular_component_of_plasma

process:
  - bootstrap
  - neoclassical_tearing_mode
  - ohmic_dissipation               # replaces ad-hoc `ohmic_`
  - resistive_diffusion             # replaces ad-hoc `resistive_`
  - resistive_flux_consumption
  - non_inductive_drive             # replaces ad-hoc `non_inductive_`
  - wave_driven
  - parallel_viscosity
  - perpendicular_viscosity
  - heat_viscosity
  - j_cross_b_force
  - poloidal_current

position:
  - sawtooth_inversion_radius
  - neoclassical_tearing_mode_center
  - inside_flux_surface             # disambiguate from geometry segment
  - halo_boundary
  - strike_point
  - secondary_separatrix_strike_point
  - active_limiter_point

transformation:
  - per_toroidal_mode_number        # interim; replace with ADR-3 Fourier decomp
  - per_toroidal_and_poloidal_mode_number
  - cumulative_inside_flux_surface
  - flux_surface_averaged           # already canonical; ensure listed
  - volume_averaged
  - variation_of
  - real_part
  - imaginary_part
  - normalized

subject:
  # Species axis (interim — prefer decomposition(species) per HIGH-ROI #7)
  - fast_electron
  - fast_ion
  - fast_ion_state
  - fast_neutral
  - thermal_electron
  - thermal_ion
  - thermal_neutral
  - ion_state
  - ion_charge_state
  - ion_species
  - molecular_ion
  - halo                            # edge/energetic-particle halo
  # Polarisation (interim — prefer decomposition(polarisation))
  - right_hand_circularly_polarized
  - left_hand_circularly_polarized

object:
  - neoclassical_tearing_mode
  - electron_cyclotron_beam
  - electron_cyclotron_launcher
  - electron_cyclotron_launcher_mirror
  - lower_hybrid_antenna
  - ion_cyclotron_heating_antenna
  - neutral_beam_injector           # replace `_beam_injector_beam`
  - fiber_optic_current_sensor      # already used
  - passive_loop
  - passive_structure
  - x_point
  - secondary_separatrix_x_point
  - iron_core_segment               # machine-description; needs ADR on whether to include

geometry:
  - beam_tracing_point
  - beamlet_group
  - delta_position
  - end_point
  - start_point
  - halo_current_area_end_point
  - halo_current_area_start_point
  - plasma_boundary_gap_reference_point
  - surface_normal
  - x_point
  - outline_point

region:
  - halo_region

# Clarified (remove ambiguity reported in grammar_ambiguity segment)
clarifications:
  - token: diamagnetic
    ruling: |
      `diamagnetic` may appear as a `component` (in drift-velocity composites)
      and as a qualifier on fluxes (`diamagnetic_flux`). It is NEVER a subject.
      Composer must not emit `diamagnetic_X` as a standalone base name.
  - token: velocity_phi_vs_velocity_tor
    ruling: |
      Toroidal component in cylindrical coordinates is always `toroidal_component`.
      `velocity_phi` is forbidden; `velocity_tor` is deprecated.
```

---

## 6. Plan-29 §7 success-criteria check

| # | Criterion | Target | Actual | Status |
| --- | --- | --- | --- | --- |
| 7.a | Jaccard cluster density reduction | ≥30% vs prior | 36.7 pairs/100 SNs (**baseline**) | **Not measurable** — no prior to compare (D.1 wiped 141-SN corpus); adopting this as the new baseline to beat in D.5. |
| 7.b | Reviewer rubric score | ≥0.75 (from 0.68) | **no data** (`sn review` not run) | ❌ Must run reviewer before gating D.4. |
| 7.c | StandardName validation pass rate | ≥90% | 335/438 = **76.5%** | ❌ below target; 11.9% quarantine is driven by vocab gaps (§5 fixes this). |
| 7.d | Vocab-gap surfacing | `sn gaps` exports yaml | 290 gaps recorded | ✅ mechanism works. |
| 7.e | Documentation rubric | ≥0.75 | **N/A** (no docs yet) | Deferred to D.4. |
| 7.f | No provenance/solver leak | 0 names | 14 leaked | ❌ — see §4.2 prompt-rejection rule. |
| 7.g | No DD-path leakage | 0 names | 23 leaked | ❌ — see §4.2. |
| 7.h | `cocos_transformation_type` coverage | ≥35 SNs | **16 / 438** | ❌ — §4.3 fixes cheaply. |
| 7.i | Per-domain coverage balance | all domains ≥20 valid | `mag_diag=4`, `turbulence=5` | ❌ — §4.4 regenerate. |

**Overall verdict on §D.3 gate:** **CONDITIONAL GO** for D.4 (enrich) *only*
if §4.1, §4.2, §4.3 are shipped first. Without them, the enrich rotation will
lock in NC-3/NC-4 anti-patterns and waste LLM budget documenting quantities
that should not exist in the vocabulary.

---

## 7. Recommended next iteration

**Before running D.4 enrich:**

1. **Ship §4.1** (ISN vocab PR — 15 tokens) — 1 day, independent PR against
   `imas-standard-names`.
2. **Ship §4.2** (composer REJECT clause) — 1 hour.
3. **Ship §4.3** (default `cocos_transformation_type=one_like` for scalars) —
   30 minutes.
4. **Run §4.8** (`sn review -c 15`) — overnight, produces the reviewer_score
   baseline needed to measure §7.b on *this* corpus.
5. **Regenerate targeted domains** (§4.4, §4.5): `turbulence`,
   `magnetic_field_diagnostics`, `equilibrium` (reset + re-gen).

**Expected post-action state (projection):**

| Metric | Current | Projected after §4.1–4.5 |
| --- | --- | --- |
| Total names | 438 | ~470 (+30 in under-populated domains, −35 in equilibrium cleanup) |
| Valid | 335 (76.5%) | ~430 (≈92%) |
| Quarantined | 52 | ~15 |
| Issues / name | 0.16 | ~0.08 |
| COCOS populated | 16 | ~370 |
| J≥0.75 pairs | 123 | 123 (species/mode-number collapse needs ADR-3 §4.6/§4.7) |

Then **D.4 enrich** is green-lit. ADR-3 Fourier + species-decomposition
(§4.6, §4.7) are ISN-side refactors that should feed the *next* rotation
(D.6 or later) rather than blocking D.4.

---

**Signed off.** Evidence: `D3-data.json` (438 standard_names + 290 vocab_gaps +
metrics block). Reviewer: senior physics + ISN co-author role,
acting on behalf of plan 29 §D.3.
