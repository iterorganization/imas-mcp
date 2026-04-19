# DD Unit Defect Catalog

**Status:** Enumerated from graph (DD version 4.1.0) · 2026-04-19  
**Audience:** IMAS DD maintainers / imas-standard-names team  
**Prepared by:** imas-codex automated audit + graph query

---

## Executive Summary

This report catalogs **5 distinct defect classes** affecting **1,403 IMAS Data Dictionary
paths** in DD 4.1.0.  Three classes involve invalid unit strings (unparseable by UDUNITS2
or semantically wrong), one involves a Jinja placeholder that was never resolved at DD
generation time, and one is a sentinel-value issue in `pulse_schedule` reference nodes.
All defects propagate into the `imas-codex` Standard Name (SN) pipeline, causing either
direct SN quarantine (invalid unit fails pydantic audit) or incorrect unit inheritance by
downstream SNs.  Four SNs are currently flagged in the test allow-list as "known DD-side
bugs" pending upstream fixes.

---

## Defect Table

| # | Defect Class | Affected paths | Current (wrong) unit | Suggested unit | Justification |
|---|---|---|---|---|---|
| 1 | Unresolved Jinja placeholder | 57 | `m^dimension` | `m`, `m^2`, or `m^3` per object dimension | The string `m^dimension` is a compile-time template variable that was never expanded; `measure` of a 0-D / 1-D / 2-D / 3-D GGD object has a concrete spatial unit determined by the `dimension` field |
| 2 | Prose string instead of UDUNITS symbol | 511 | `Elementary Charge Unit` | `1` (multiplicity, z_n, vibrational level) / `eV` (ionisation_potential) | Multi-word prose strings cannot be parsed by any UDUNITS library; the correct UDUNITS symbol for elementary charge is `e`, but most quantities using this string are actually dimensionless (counts or ratios) |
| 3 | Prose string instead of UDUNITS symbol | 15 | `Atomic Mass Unit` | `u` (unified atomic mass unit, UDUNITS symbol) | Same class of bug as #2; `u` is the correct UDUNITS2 symbol |
| 4 | Wrong dimension on charge/state ratios | 805 | `e` | `1` | Quantities like `z_ion` (average charge number), `z_average`, `z_square_average`, `z_max`, `z_min`, `z_n`, and `vibrational_level` are dimensionless ratios or integer quantum numbers; the elementary charge `e` is a physical constant, not the unit of a dimensionless quantity |
| 5 | Sentinel `unit=1` on `pulse_schedule/*/reference` nodes | ~350 (est.) | `1` | Physical unit of parent quantity (e.g. `ohm`, `W`, `Hz`, `V`, `m`) | All `reference` and `reference_waveform/data` nodes under `pulse_schedule` carry a hard-coded `unit=1` regardless of the physical quantity they reference; the correct unit should match the controlled quantity |

**Total affected paths (deduplicated): ≥ 1,403**  
*(Defect 5 path count is estimated from DD-04 analysis; exact count not enumerated in graph
because `pulse_schedule` reference nodes mostly lack `HAS_UNIT` edges in the current graph.)*

---

## Per-Defect Detail

### Defect 1 — Unresolved Jinja placeholder `m^dimension` (57 paths, 18 IDSs)

**Root cause.** The GGD (Generic Grid Description) DD generator uses a Jinja-style
template `m^{dimension}` to express that the `measure` field of a grid object has spatial
units determined by the grid object's `dimension` integer (0 = point count, 1 = length,
2 = area, 3 = volume).  This template variable was not resolved at DD compile time and
was emitted verbatim into the `units` attribute of every `objects_per_dimension/object/measure`
node.  The string `m^dimension` is not a valid UDUNITS2 expression.

**Affected IDSs (18):**  
`distribution_sources`, `distributions`, `edge_profiles`, `edge_sources`,
`edge_transport`, `em_coupling`, `equilibrium`, `ferritic`, `mhd`,
`plasma_profiles`, `plasma_sources`, `plasma_transport`, `radiation`,
`runaway_electrons`, `tf`, `transport_solver_numerics`, `wall`, `waves`

**Affected paths (all 3 variants per IDS):**
```
<IDS>/*/space/objects_per_dimension/object/measure
<IDS>/*/space/objects_per_dimension/object/measure_error_lower
<IDS>/*/space/objects_per_dimension/object/measure_error_upper
```

Full list of 57 paths:

<details>
<summary>Expand all 57 paths</summary>

```
distribution_sources/source/ggd/grid/space/objects_per_dimension/object/measure
distribution_sources/source/ggd/grid/space/objects_per_dimension/object/measure_error_lower
distribution_sources/source/ggd/grid/space/objects_per_dimension/object/measure_error_upper
distributions/distribution/ggd/grid/space/objects_per_dimension/object/measure
distributions/distribution/ggd/grid/space/objects_per_dimension/object/measure_error_lower
distributions/distribution/ggd/grid/space/objects_per_dimension/object/measure_error_upper
edge_profiles/grid_ggd/space/objects_per_dimension/object/measure
edge_profiles/grid_ggd/space/objects_per_dimension/object/measure_error_lower
edge_profiles/grid_ggd/space/objects_per_dimension/object/measure_error_upper
edge_sources/grid_ggd/space/objects_per_dimension/object/measure
edge_sources/grid_ggd/space/objects_per_dimension/object/measure_error_lower
edge_sources/grid_ggd/space/objects_per_dimension/object/measure_error_upper
edge_transport/grid_ggd/space/objects_per_dimension/object/measure
edge_transport/grid_ggd/space/objects_per_dimension/object/measure_error_lower
edge_transport/grid_ggd/space/objects_per_dimension/object/measure_error_upper
em_coupling/grid_ggd/space/objects_per_dimension/object/measure
em_coupling/grid_ggd/space/objects_per_dimension/object/measure_error_lower
em_coupling/grid_ggd/space/objects_per_dimension/object/measure_error_upper
equilibrium/grids_ggd/grid/space/objects_per_dimension/object/measure
equilibrium/grids_ggd/grid/space/objects_per_dimension/object/measure_error_lower
equilibrium/grids_ggd/grid/space/objects_per_dimension/object/measure_error_upper
equilibrium/time_slice/ggd/grid/space/objects_per_dimension/object/measure
equilibrium/time_slice/ggd/grid/space/objects_per_dimension/object/measure_error_lower
equilibrium/time_slice/ggd/grid/space/objects_per_dimension/object/measure_error_upper
ferritic/grid_ggd/space/objects_per_dimension/object/measure
ferritic/grid_ggd/space/objects_per_dimension/object/measure_error_lower
ferritic/grid_ggd/space/objects_per_dimension/object/measure_error_upper
mhd/grid_ggd/space/objects_per_dimension/object/measure
mhd/grid_ggd/space/objects_per_dimension/object/measure_error_lower
mhd/grid_ggd/space/objects_per_dimension/object/measure_error_upper
plasma_profiles/grid_ggd/space/objects_per_dimension/object/measure
plasma_profiles/grid_ggd/space/objects_per_dimension/object/measure_error_lower
plasma_profiles/grid_ggd/space/objects_per_dimension/object/measure_error_upper
plasma_sources/grid_ggd/space/objects_per_dimension/object/measure
plasma_sources/grid_ggd/space/objects_per_dimension/object/measure_error_lower
plasma_sources/grid_ggd/space/objects_per_dimension/object/measure_error_upper
plasma_transport/grid_ggd/space/objects_per_dimension/object/measure
plasma_transport/grid_ggd/space/objects_per_dimension/object/measure_error_lower
plasma_transport/grid_ggd/space/objects_per_dimension/object/measure_error_upper
radiation/grid_ggd/space/objects_per_dimension/object/measure
radiation/grid_ggd/space/objects_per_dimension/object/measure_error_lower
radiation/grid_ggd/space/objects_per_dimension/object/measure_error_upper
runaway_electrons/grid_ggd/space/objects_per_dimension/object/measure
runaway_electrons/grid_ggd/space/objects_per_dimension/object/measure_error_lower
runaway_electrons/grid_ggd/space/objects_per_dimension/object/measure_error_upper
tf/field_map/grid/space/objects_per_dimension/object/measure
tf/field_map/grid/space/objects_per_dimension/object/measure_error_lower
tf/field_map/grid/space/objects_per_dimension/object/measure_error_upper
transport_solver_numerics/boundary_conditions_ggd/grid/space/objects_per_dimension/object/measure
transport_solver_numerics/boundary_conditions_ggd/grid/space/objects_per_dimension/object/measure_error_lower
transport_solver_numerics/boundary_conditions_ggd/grid/space/objects_per_dimension/object/measure_error_upper
wall/description_ggd/grid_ggd/space/objects_per_dimension/object/measure
wall/description_ggd/grid_ggd/space/objects_per_dimension/object/measure_error_lower
wall/description_ggd/grid_ggd/space/objects_per_dimension/object/measure_error_upper
waves/coherent_wave/full_wave/grid/space/objects_per_dimension/object/measure
waves/coherent_wave/full_wave/grid/space/objects_per_dimension/object/measure_error_lower
waves/coherent_wave/full_wave/grid/space/objects_per_dimension/object/measure_error_upper
```

</details>

**Recommended DD PR strategy.**  
Three options in order of preference:

- **Option A (preferred — split leaves):** Replace the single `measure` leaf with
  per-dimension concrete leaves: `measure_count` (`1`), `measure_length` (`m`),
  `measure_area` (`m^2`), `measure_volume` (`m^3`).  Each element of
  `objects_per_dimension` already carries a `dimension` integer (0–3), so the
  correct leaf is unambiguous at read time.  This also improves schema clarity for
  downstream readers.

- **Option B (template resolution):** Introduce a `units_template` mechanism in the
  DD generator that resolves `m^{dimension}` at compile time by expanding the
  integer `dimension` into the concrete unit string.  This preserves the
  single-leaf structure but requires tooling support.

- **Option C (minimal / unblocking):** Set `units=mixed` and add a prose
  `description` tag explaining the dimension dependency.  Allows imas-codex to
  skip the path as a non-standard-name candidate without fixing the ambiguity.

---

### Defect 2 — Prose string `Elementary Charge Unit` (511 paths)

**Root cause.** The DD emits the English prose string `Elementary Charge Unit`
for certain quantity types instead of a UDUNITS2-parseable symbol.  This string
contains whitespace and is not recognised by any UDUNITS library.

**Affected quantity types:**

| Quantity | # paths | DD unit | Correct unit | Justification |
|---|---|---|---|---|
| `element/multiplicity` | 88 | `Elementary Charge Unit` | `1` | Atom count per molecule; dimensionless integer |
| `element/z_n` (atomic number) | 198 | `Elementary Charge Unit` | `1` | Nuclear charge number Z; dimensionless integer |
| `ion/state/ionisation_potential` | 13 | `Elementary Charge Unit` | `eV` | Energy; elementary charge × voltage = eV |
| Error fields of above | 212 | `Elementary Charge Unit` | (same as parent) | Error fields inherit incorrect parent unit |

Note: A sub-set of `ionisation_potential` nodes correctly carries `eV` (the US-spelling
`ionization_potential` variant).  The DD therefore has an *internal inconsistency*
between UK/US spelling variants of the same field.

**Recommended DD PR strategy.**  
File one PR: `Fix prose 'Elementary Charge Unit' — use UDUNITS2 symbol or dimensionless`.
In that PR:
1. Change all `element/multiplicity` and `element/z_n` units to `1` (they are dimensionless).
2. Change all `ionisation_potential` units to `eV` (ionisation energy is always measured
   in electron-volts in plasma physics).
3. Reconcile UK/US spelling (`ionisation` vs `ionization`) to a single canonical spelling.

---

### Defect 3 — Prose string `Atomic Mass Unit` (15 paths)

**Root cause.** Same class of bug as Defect 2.  The DD uses the prose string
`Atomic Mass Unit` rather than the UDUNITS2 symbol `u` (unified atomic mass unit,
also accepted as `amu`).

**Affected paths (all `element/a` — atomic mass number):**  
`gas_injection/*/element/a`, `spectrometer_mass/a`, and related error fields.

**Recommended DD PR strategy.**  
File one PR: `Fix prose 'Atomic Mass Unit' — use UDUNITS2 symbol u`.
Change all instances of `Atomic Mass Unit` to `u`.

---

### Defect 4 — Dimensionless charge/state quantities tagged as unit `e` (805 paths)

**Root cause.** Several families of dimensionless quantities — ion charge numbers,
charge bounds, average charges, and vibrational quantum numbers — have been assigned
`unit=e` (elementary charge) in the DD.  While the *scale* of these quantities is
determined by the elementary charge, the values are pure ratios (charge-to-charge)
or integer quantum numbers and therefore dimensionless (`unit=1`).

**Affected quantity families:**

| Quantity | # paths | DD unit | Correct unit | Justification |
|---|---|---|---|---|
| `vibrational_level` | 198 | `e` | `1` | Vibrational quantum number; pure integer |
| `z_ion` (average ion charge) | 157 | `e` | `1` | ⟨Z⟩ is charge / elementary charge; dimensionless |
| `z_min` (minimum charge bound) | 126 | `e` | `1` | Dimensionless integer charge state |
| `z_max` (maximum charge bound) | 126 | `e` | `1` | Dimensionless integer charge state |
| `z_n` (atomic number) | 114 | `e` | `1` | Nuclear charge Z; dimensionless integer |
| `z_square_average` (⟨Z²⟩) | 32 | `e` | `1` | ⟨Z²⟩ = ⟨(charge/e)²⟩; dimensionless |
| `z_average` | 20 | `e` | `1` | Same as z_ion above |
| `ionisation_potential` (edge_profiles GGD) | 14 | `e` | `eV` | Energy; should be `eV`, not `e` |
| Other | 18 | `e` | (context-dependent) | `summary/*/z_n`, edge GGD ionisation |

Note: There is also an *internal DD inconsistency*: `z_ion_1d` (profile-averaged)
correctly carries `unit=1` in some IDSs while the scalar `z_ion` carries `unit=e`.
This inconsistency is what triggers the SN pipeline allow-list entry for
`ion_average_charge_of_ion_state`.

**Recommended DD PR strategy.**  
File one PR: `Fix unit 'e' on dimensionless charge ratios and quantum numbers`.  
Bulk-change all `z_ion`, `z_average`, `z_square_average`, `z_max`, `z_min`, `z_n`,
and `vibrational_level` fields to `unit=1`.  For `ionisation_potential` paths that
currently carry `e`, change to `eV`.

---

### Defect 5 — Sentinel `unit=1` on `pulse_schedule/*/reference` nodes (~350 paths)

**Root cause.** All `reference` and `reference_waveform/data` nodes under
`pulse_schedule` carry a hard-coded `unit=1` sentinel value regardless of the
physical quantity they represent.  The sentinel appears to be a DD convention
meaning "normalised / controller-internal", but it is not documented as such and
is not valid physics.

**Sample affected paths:**

| DD path | Current unit | Correct unit |
|---|---|---|
| `pulse_schedule/ec/beam/power_launched/reference/data` | `1` | `W` |
| `pulse_schedule/ic/antenna/power/reference/data` | `1` | `W` |
| `pulse_schedule/lh/antenna/power/reference/data` | `1` | `W` |
| `pulse_schedule/nbi/unit/power/reference/data` | `1` | `W` |
| `pulse_schedule/flux_control/loop_voltage/reference/data` | `1` | `V` |
| `pulse_schedule/pf_active/coil/resistance_additional/reference` | `1` | `ohm` |
| `pulse_schedule/ec/beam/frequency/reference/data` | `1` | `Hz` |
| `pulse_schedule/position_control/*/reference/data` | `1` | `m` |

**Recommended DD PR strategy.**  
File one PR: `Publish physical unit on pulse_schedule reference waveform nodes`.
Each `reference/data` node should inherit or explicitly declare the unit of its
parent controlled quantity.  Where normalised waveforms are intended (e.g. a
fractional power reference), the documentation should say so explicitly.

---

## Impact on the SN Pipeline

### Quarantined standard names (unit-invalid, blocked from publication)

These SNs fail the `pydantic:scalar.unit` audit because their source DD path
has an invalid unit string:

| Standard name | Quarantine reason | Blocking defect |
|---|---|---|
| `grid_object_measure` | `m^dimension` not a valid UDUNITS expression | Defect 1 |
| `atom_multiplicity_of_isotope` | `Elementary Charge Unit` contains whitespace | Defect 2 |
| `atomic_mass_number_of_isotope` | `Atomic Mass Unit` contains whitespace | Defect 3 |

### Valid SNs with wrong unit (test allow-list entries)

These SNs were generated with incorrect units inherited from DD defects, but are
considered *valid* because the SN unit was corrected manually or the DD source
path has an inconsistency that makes the SN unit technically correct for some paths:

| Standard name | SN unit | DD unit on linked path | Root defect |
|---|---|---|---|
| `electron_temperature_peaking_factor` | `1` (correct) | `eV` (from `summary/t_e/value` — wrong source path) | DD-side: `summary/*/t_e` is absolute temperature, not a peaking factor |
| `ion_average_charge_of_ion_state` | `1` (correct) | `e` (from `z_average` GGD path) | Defect 4: DD inconsistency; `z_ion_1d` has `1`, `z_average` has `e` |
| `ion_average_square_charge_of_ion_state` | `1` (correct) | `e` (from `z_square_average` path) | Defect 4: same |
| `resistance_of_poloidal_field_coil` | `1` (wrong — should be `ohm`) | `ohm` (from `pf_active/coil/resistance_additional`) | Defect 5: SN unit was inherited from `pulse_schedule/reference` sentinel=1 |

### SNs with multiple quarantine causes (pulse_schedule reference cluster)

The DD-04 `pulse_schedule/*/reference` sentinel bug alone accounts for **~16
quarantined SNs** (see `plans/research/standard-names/dd-issues-rc14/DD-04-pulse-schedule-reference-unit.md`),
the largest single-issue quarantine cluster in the rc13/rc14 SN corpus.

---

## Suggested PR Strategy

File **one PR per defect class** (not one per path).  Suggested titles:

| PR # | Title | Defect | Paths |
|---|---|---|---|
| 1 | `Fix unresolved m^dimension placeholder in GGD grid_object_measure` | Defect 1 | 57 |
| 2 | `Fix prose 'Elementary Charge Unit' — use UDUNITS2 symbol or dimensionless` | Defect 2 | 511 |
| 3 | `Fix prose 'Atomic Mass Unit' — use UDUNITS2 symbol u` | Defect 3 | 15 |
| 4 | `Fix unit e on dimensionless charge ratios and vibrational quantum numbers` | Defect 4 | 805 |
| 5 | `Publish physical unit on pulse_schedule reference waveform nodes` | Defect 5 | ~350 |

PRs 2 and 4 can be filed simultaneously since they address different quantity
families.  PR 5 is the highest-impact fix for the SN pipeline (16 quarantined names
unblocked).  PR 1 is the highest-impact fix for GGD users.

---

## Allow-List Reconciliation

The following allow-list entries are maintained in `tests/graph/test_sn_unit_integrity.py`
to prevent CI failures until upstream DD fixes are applied.  These should be removed
when the corresponding DD PR is merged and the graph is rebuilt.

| Path pattern (SN name) | SN unit | DD unit | Root defect | Remove when |
|---|---|---|---|---|
| `electron_temperature_peaking_factor` | `1` | `eV` (via summary path) | Source path mismatch: `summary/t_e` is absolute temperature | DD fixes `t_e_peaking` source linkage or removes `summary` from SN source paths |
| `ion_average_charge_of_ion_state` | `1` | `e` (via `z_average` path) | Defect 4 | DD PR 4 fixes `z_average` to `1` |
| `ion_average_square_charge_of_ion_state` | `1` | `e` (via `z_square_average` path) | Defect 4 | DD PR 4 fixes `z_square_average` to `1` |
| `resistance_of_poloidal_field_coil` | `1` | `ohm` (correct) | Defect 5: SN inherited sentinel=1 from `pulse_schedule/reference` | DD PR 5 fixes reference units AND SN is regenerated with `ohm` |

---

## References

- `plans/research/standard-names/dd-issues-rc14/DD-01-diamagnetic-axis.md`
- `plans/research/standard-names/dd-issues-rc14/DD-02-grid-object-measure-unit.md`
- `plans/research/standard-names/dd-issues-rc14/DD-03-multiplicity-unit.md`
- `plans/research/standard-names/dd-issues-rc14/DD-04-pulse-schedule-reference-unit.md`
- `plans/research/standard-names/dd-issues-rc14/DD-05-neutron-fluxes-rates-duplicate.md`
- `plans/research/standard-names/dd-issues-rc14/DD-06-thermalisation-spelling.md`
- `plans/research/standard-names/sn-unit-mismatches.json`
- `scripts/backfill_sn_unit_mismatches.py`
- `tests/graph/test_sn_unit_integrity.py`
