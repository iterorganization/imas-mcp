## vNext Exemplars

Study the **reasoning**, not just the names. The canonical form often surprises.

### ✅ Positive Exemplars — produce names like these

**E1. Zero-operator base** — `electron_temperature`
Qualifier `electron` + base `temperature`. No operators, no locus. The simplest valid form.

**E2. Unary prefix operator** — `gradient_of_electron_pressure`
Operator `gradient` wraps the inner name `electron_pressure` with explicit `_of_` scope.

**E3. Unary postfix operator** — `magnetic_field_magnitude`
Operator `magnitude` appends directly to the base. Postfix operators never use `_of_`.

**E4. Binary operator** — `ratio_of_electron_to_ion_temperature`
Binary `ratio` uses `_of_` scope + `_to_` separator between the two operands.

**E5. Axis projection** — `radial_component_of_ion_velocity`
Projection `radial_component_of_` precedes the qualified base. The projection is always a prefix.

**E6. Postfix locus (_at_)** — `electron_density_at_magnetic_axis`
The locus `_at_magnetic_axis` follows the base. `_at_` marks a field value sampled at a point.

**E7. Mechanism qualifier** — `plasma_current_due_to_bootstrap`
The process `bootstrap` is attached via `_due_to_` after the base quantity.

**E8. Prefix operator with scope** — `line_averaged_of_electron_density`
The operator `line_averaged` applies `_of_` scope to the full inner name. Operators always carry explicit scope.

**E9. Projection + locus combo** — `radial_component_of_magnetic_field_at_separatrix`
Projection prefix + locus postfix coexist cleanly: `[projection] base [locus]`.

**E10. Nested operators** — `time_derivative_of_volume_averaged_of_electron_density`
Outer operator `time_derivative` wraps inner operator `volume_averaged` wraps base `electron_density`. Each `_of_` is an operator scope marker; the last `_of_` (if any) would be a locus.

### ❌ Negative Exemplars — rc20 forms now rejected

**N1.** ❌ `real_part_of_perturbed_electrostatic_potential`
*Rejected:* `real_part` is a postfix operator. Prefix `real_part_of_X` creates nested `_of_` ambiguity.
✅ Use: `perturbed_electrostatic_potential_real_part`

**N2.** ❌ `amplitude_of_parallel_component_of_wave_electric_field`
*Rejected:* `amplitude` is postfix. Prefix form with nested `_of_` chains is unparseable.
✅ Use: `parallel_component_of_wave_electric_field_amplitude`

**N3.** ❌ `volume_averaged_electron_density` (bare concatenation)
*Rejected:* Prefix operators require `_of_` scope marker. Bare concatenation is the old transformation-segment form.
✅ Use: `volume_averaged_of_electron_density`

**N4.** ❌ `diamagnetic_component_of_ion_velocity`
*Rejected:* `diamagnetic` is a drift mechanism, not a projection axis. The diamagnetic drift IS a velocity, not a component of another velocity.
✅ Use: `ion_diamagnetic_drift_velocity`

**N5.** ❌ `reconstructed_safety_factor`
*Rejected:* Provenance verbs (`reconstructed_`, `measured_`, `fitted_`) describe the data pipeline, not the physics quantity.
✅ Use: `safety_factor`

**N6.** ❌ `ion_rotation_frequency_toroidal`
*Rejected:* Trailing component suffix violates the canonical projection-prefix pattern.
✅ Use: `toroidal_component_of_ion_rotation_frequency`

**N7.** ❌ `electron_thermal_pressure`
*Rejected:* Population qualifier `thermal` must precede species `electron`. Species is a qualifier; population class is also a qualifier — they compose as `[population]_[species]`.
✅ Use: `thermal_electron_pressure`

**N8.** ❌ `electron_temperature_profile`
*Rejected:* The `_profile` suffix encodes data rank (1D array), not physics. Every standard name is a point value; profiles are implicit.
✅ Use: `electron_temperature`

**N9.** ❌ `norm_poloidal_flux`
*Rejected:* Abbreviations fragment the vocabulary. All tokens must be spelled in full.
✅ Use: `normalized_of_poloidal_magnetic_flux`

**N10.** ❌ `poloidal_magnetic_flux_of_plasma_boundary_at_plasma_boundary`
*Rejected:* Duplicated preposition — `_of_` and `_at_` for the same entity. Only one locus per name.
✅ Use: `poloidal_magnetic_flux_of_plasma_boundary`
