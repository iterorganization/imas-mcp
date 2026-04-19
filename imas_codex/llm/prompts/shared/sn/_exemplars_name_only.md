## Curated Naming Exemplars

These exemplars teach by contrast. Study the *reasons* — many anti-patterns
look plausible until you see the canonical form. All exemplars follow
American spelling (NC-17).

### Positive exemplars — imitate these patterns

#### P1. Component decomposition (scalar from vector)

- ✅ `radial_component_of_magnetic_field` — scalar, unit `T`
  - *Why good:* `{component}_of_{vector_base}` is the canonical decomposition.
    The decomposed scalar has the same unit as the parent vector. Never use
    `_r_`, `_rad_`, or `_radial_magnetic_field` (the last is ambiguous
    between "radial component of B" and "B in the radial direction").
- ✅ `poloidal_component_of_plasma_velocity`
- ✅ `perpendicular_component_of_electron_pressure_gradient`
  - *Why good:* perpendicular/parallel are first-class components; chain
    them through gradients and fluxes naturally.

#### P2. Species subjects

- ✅ `electron_temperature`, `deuterium_density`, `helium_ash_density`
  - *Why good:* single-token species go before the physical base noun.
    Do not say `temperature_of_electrons`; `electron_temperature` is shorter
    and follows the conventional physics word order.
- ✅ `tungsten_density`, `beryllium_impurity_density`
  - *Why good:* impurity species use their element name, not Zeff or a
    numeric index.

#### P3. Position qualifiers: `of_` vs `at_` (semantic match)

Positional qualifiers distinguish intrinsic geometric properties from
field values evaluated at a locus.

- **`of_<position>`** — intrinsic geometric property of a locus.
  GOOD: `major_radius_of_magnetic_axis`, `area_of_plasma_boundary`,
       `vertical_coordinate_of_x_point`.
- **`at_<position>`** — value of a field quantity sampled at a point,
  line, or surface.
  GOOD: `electron_temperature_at_magnetic_axis`,
       `poloidal_magnetic_flux_at_plasma_boundary`,
       `safety_factor_at_minimum_safety_factor`.

FORBIDDEN: `at_` for a geometric coordinate (wrong preposition).
   BAD: `major_radius_at_magnetic_axis` → use `major_radius_of_magnetic_axis`.

#### P4. Transformations on a base quantity

- ✅ `normalized_poloidal_magnetic_flux`
- ✅ `volume_averaged_electron_density`
- ✅ `line_averaged_electron_density`
- ✅ `surface_integrated_toroidal_current_density` (→ `toroidal_plasma_current`)
  - *Why good:* transformation prefixes attach cleanly; they preserve the
    base physics while signaling the operation applied.

#### P5. Geometry of a structural entity

- ✅ `major_radius_of_magnetic_axis`
- ✅ `vertical_coordinate_of_x_point`
- ✅ `major_radius_of_plasma_boundary_outline_point`
  - *Why good:* `of_{entity}` reads as "the property belonging to this
    entity". Do not switch to `at_{entity}` — the property *is* a feature
    of the entity, not a measurement taken at it.
  - Note: `vertical_coordinate_of_` is the canonical Z coordinate form
    (Rule 17); prefer it over the legacy `vertical_position_of_` form.

#### P6. Distance-between form

- ✅ `distance_from_magnetic_axis_to_separatrix_along_midplane`
- ✅ `distance_between_strike_points_along_divertor_target`
  - *Why good:* the `distance_between_A_and_B` or
    `distance_from_A_to_B_along_C` pattern removes ambiguity about which
    direction is measured.

#### P7. Spectral / Fourier decomposition

- ✅ `fourier_coefficient_of_plasma_boundary_radius` (with
  `description` starting "Fourier cosine coefficient of ...")
- ✅ `mode_amplitude_of_magnetic_perturbation`
  - *Why good:* when the quantity *is* a spectral coefficient, the name
    must carry `fourier_coefficient_*`, `mode_amplitude_*`, or an explicit
    `harmonic_*` marker. Name and description must agree.

#### P8. Poloidal-plane coordinates

- ✅ `vertical_coordinate_of_<position>` (preferred; Rule 17).
- ✅ `major_radius_of_<position>` (preferred; Rule 17).
- ✅ `toroidal_angle_of_<position>` (preferred; Rule 17).

DEPRECATED: `vertical_position_of_<position>` (old form;
    `vertical_coordinate_` is the canonical Z coordinate segment).

- ✅ For 2-D fields, use `radial_coordinate` / `vertical_coordinate` for
  the independent-axis arrays themselves when they appear as grid paths.

### Anti-patterns — never emit these

#### A1. Abbreviations (NC-5, NC-12)

- ❌ `norm_poloidal_flux`, `perp_velocity`, `temp_pedestal`, `sep_distance`
  - *Why bad:* every abbreviation has a canonical long form. Long forms
    dedupe across a domain; abbreviations proliferate.
  - *Fix:* `normalized_poloidal_flux`, `perpendicular_velocity_component`,
    `pedestal_electron_temperature`, `separatrix_distance`.

#### A2. Provenance verbs in the name

- ❌ `measured_plasma_current`, `reconstructed_safety_factor`,
  `fitted_electron_temperature`
  - *Why bad:* provenance (measured / reconstructed / fitted) is a
    property of the data instance, not of the quantity. Same physical
    concept, different pipeline.
  - *Fix:* drop the provenance verb — use `plasma_current`,
    `safety_factor`, `electron_temperature`. Store provenance as path
    metadata.

#### A3. Tautology with `of`

- ❌ `poloidal_magnetic_flux_of_poloidal_plane`
  - *Why bad:* the base quantity already implies the geometric setting.
    `of` should contribute new information.
  - *Fix:* drop the tautological `of` phrase, or select a different
    qualifier (`at_plasma_boundary`, `on_flux_surface`).

#### A4. Mixed R/Z entity (inconsistency within a pair)

- ❌ `r_of_magnetic_axis` paired with `vertical_coordinate_of_magnetic_axis`
  - *Why bad:* the pair is asymmetric — R uses a letter abbreviation, Z
    uses a full phrase. Grep for one and you miss the other.
  - *Fix:* pair as `major_radius_of_magnetic_axis` /
    `vertical_coordinate_of_magnetic_axis`.

#### A5. Multi-subject naming (NC-2)

- ❌ `electron_ion_temperature_ratio`
  - *Why bad:* this encodes two species into one name and bakes in a
    specific operation. The ratio is a derived diagnostic, not a
    primitive quantity.
  - *Fix:* two standard names — `electron_temperature`, `ion_temperature`
    — and compute the ratio at analysis time.

#### A6. Single-token generic nouns

- ❌ `geometry`, `value`, `coefficient`, `parameter`
  - *Why bad:* the name must be self-describing; no external context
    tells the consumer what the quantity is.
  - *Fix:* either (a) a specific physics name, or (b) classify the path
    as metadata and skip naming.

#### A7. British spelling (NC-17)

- ❌ `normalised_poloidal_flux`, `polarised_cross_section`,
  `centre_of_plasma_boundary`
  - *Why bad:* the ISN catalog is American-English only. British
    variants would create silent synonyms.
  - *Fix:* `normalized_*`, `polarized_*`, `center_of_*`. Apply the same
    rule to all prose fields.

#### A8. Spectral name/description mismatch

- ❌ Name `normal_component_of_magnetic_field` with description
  "Fourier coefficients of the normal component..."
  - *Why bad:* the name promises a scalar field; the description
    describes a spectral coefficient. They disagree.
  - *Fix:* either rename to `fourier_coefficient_of_normal_component_of_magnetic_field`
    or rewrite the description to describe the underlying field.

#### A9. Trivial surface-of-definition names

- ❌ `normalized_poloidal_flux_at_plasma_boundary`
  - *Why bad:* normalized flux equals 1 on the boundary by construction.
    The "name" encodes a definitional tautology.
  - *Fix:* skip the path — no standard name is warranted.

#### A10. Duplicated preposition (exclusive-pair violation)

- ❌ `poloidal_magnetic_flux_of_plasma_boundary_at_plasma_boundary`
  - *Why bad:* `of` and `at` are exclusive for the same entity.
  - *Fix:* pick one preposition per name.

### Forbidden patterns (anti-exemplars)

1. `due_to_<adjective>` — always use the process noun.
   BAD: `due_to_halo`, `due_to_ohmic`, `due_to_fast_ion`, `due_to_non_inductive`.
   GOOD: `due_to_halo_currents`, `due_to_ohmic_dissipation`,
         `due_to_fast_ions`, `due_to_non_inductive_drive`.

2. `_over_<quantity>` as a division surrogate.
   BAD: `ion_velocity_over_magnetic_field_strength`.
   GOOD: `ion_velocity_per_magnetic_field_strength`.
   Note: `over_<region>` (e.g. `over_halo_region`) is the valid Region
   segment.

3. `_ggd_coefficients`, `_finite_element_interpolation_coefficients_on_ggd`,
   `_on_ggd`, `_coefficient_on_ggd` — basis-function storage, not
   physics. Classifier excludes these; LLM must never propose them.

4. `_reference_waveform`, `_reference` on pulse_schedule paths —
   controller setpoints. Classifier excludes; LLM must never propose.

5. `diamagnetic_component_of_<vector>` — the diamagnetic drift is a
   vector quantity (`v_dia = (B × ∇p) / (q n B²)`), not a spatial axis.
   If you need its projection, first name the drift:
   `ion_diamagnetic_drift_velocity`; then project:
   `radial_component_of_ion_diamagnetic_drift_velocity`.

6. Duplicate-subject splitting on compound species.
   BAD: `deuterium_tritium_*` interpreted as two subjects.
   GOOD: treat `deuterium_tritium`, `deuterium_deuterium`,
         `tritium_tritium` as single compound-subject tokens.

### Checklist before emitting a name

1. Does the name use American spelling throughout?
2. Are all words spelled in full — no `norm_`, `perp_`, `temp_`, `max_`?
3. Is there at most one subject and one position qualifier?
4. If the name implies a Fourier/spectral quantity, does it carry an
   explicit spectral marker (`fourier_coefficient_`, `mode_amplitude_`)?
5. Does the name describe a physical quantity (not a provenance label,
   a diagnostic pipeline, or a trivially-defined constant)?
6. Does the description use American spelling and correctly define every
   `$...$` symbol it introduces?
