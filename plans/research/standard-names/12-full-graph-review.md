# 12 — Full Graph Standard-Name Review (705 names, 2026 rc13)

> **Role:** senior standard-name reviewer (CF/ISN conventions + IMAS physics).
> **Corpus:** 705 StandardName nodes in the imas-codex Neo4j graph, exported
> to `sn-review-dump.md` / `sn-full-dump.json` at the end of the rc13
> bootstrap rotation.
> **Companion plan:** `plans/features/standard-names/31-quality-bootstrap-v2.md`
> consumes this report and dispatches parallel Opus workstreams.
>
> **Prior context (not repeated here):**
> - R1 grammar findings, R2 semantic audit, R3 enrichment architecture
>   (archived under `plans/research/standard-names/archived-v1/`).
> - Plan 28 greenfield framing; plan 29 architectural pivot (generate/enrich
>   split + ISN grammar rc bump); plan 30 DD semantic node categories
>   (`fit_artifact`, `representation`).
> - D.3 / D.5 / E grammar-graph design notes — already landed in ISN rc12/rc13
>   (see `subjects.yml`, `processes.yml`, `positions.yml`, `components.yml`,
>   `transformations.yml`, `objects.yml`, `regions.yml`).

---

## 0. Executive summary — 5 blockers for the next bootstrap cycle

| # | Blocker | Owner | Estimated impact |
|---|---------|-------|------------------|
| **B1** | DD representation/fit-artifact paths continue to leak into the SN queue because plan 30 (`fit_artifact`, `representation` NodeCategory) has not yet landed. **63/121 (52%) of quarantined names** are GGD coefficients, reference-waveforms, or constraint-fit children. | imas-codex classifier | Quarantine drops from 17% → ~8% overnight. |
| **B2** | Prompt/grammar/exemplar contradictions: compose_system.md Rules 17/18/21 require `vertical_coordinate_of_`, `_of_plasma_boundary`, `_of_magnetic_axis`; exemplars P3/P5/P8 still advertise the older `_at_` / `_position_of_` forms. The LLM sees both and regresses on rotation. | imas-codex prompts | Eliminates ~15 recurrent anti-patterns per rotation. |
| **B3** | Silent unit-bug propagation on `*_reference_waveform` / `*_reference`: 16 quarantined names have `unit=1` while the name mentions `power`/`angle`/`frequency`/`voltage`. DD pulse_schedule reference signals publish `unit=1` as a sentinel (reference-normalized waveform) — but codex treats them as physical quantities. | imas-codex classifier + DD | Cuts `plasma_control` quarantine rate from **51% → ≤10%**. |
| **B4** | Grammar vocabulary gaps force ~12 near-valid quarantines. Critical missing tokens: `e_cross_b_drift` (Process), `thermalization_of_fast_particles` / `thermal_fusion` (Process), `suprathermal_electrons` (Subject or Process), `ferritic_element_centroid` (Position), `neutron_detector` (Position), `atom_multiplicity_of_isotope` unit sanitization (DD). | iter-standard-names + DD | Unblocks 12 quarantines; adds 2–3 valid names. |
| **B5** | DD-upstream sentinel leaks into names: `diamagnetic_component_of_*` (edge_profiles/ggd/*/diamagnetic treats a drift as a spatial axis), `vacuum_toroidal_field_function` (r·B0 product called "`field`"), `grid_object_measure` with `unit=m^dimension`, `atom_multiplicity_of_isotope` unit `Elementary Charge Unit`. These must be filed as DD issues; codex can only suppress them at the classifier. | IMAS DD | Ends a class of false quarantines forever. |

Corpus health on rc13 export:

- Valid: 584 / 705 (82.8%); Quarantined: 121 / 705 (17.2%).
- Descriptions populated: 84 / 705 (11.9%) — **enrichment rotation has not yet run**.
- Domains with >30% quarantine (priority order):
  `current_drive 100%` (1/1), `unknown 100%` (1/1), `plasma_control 51%` (18/35),
  `magnetohydrodynamics 40%` (32/80), `computational_workflow 38%` (5/13),
  `fast_particles 32%` (10/31), `particle_measurement_diagnostics 32%` (6/19).
- Issue classes (total 126 audit hits, multi-hit names common):
  representation_artifact 36, pydantic:scalar.name 25, name_unit_consistency 25,
  multi_subject 10, implicit_field 6, diamagnetic_component 6,
  density_unit_consistency 5, causal_due_to 5, cumulative_prefix 4,
  parse_error 6, pydantic:scalar.unit 2, unit_validity 1.

---

## 1. Quarantine heat-map

```
current_drive            (1)  ██████████████████████████████ 100%   [dummy placeholder]
unknown                  (1)  ██████████████████████████████ 100%   [dummy placeholder]
plasma_control          (35)  ███████████████░░░░░░░░░░░░░░░  51%   pulse_schedule reference_waveform unit=1
magnetohydrodynamics    (80)  ████████████░░░░░░░░░░░░░░░░░░  40%   mhd/ggd/*/coefficients (representation)
computational_workflow  (13)  ███████████░░░░░░░░░░░░░░░░░░░  38%   transport_solver_numerics boundary_condition
fast_particles          (31)  ██████████░░░░░░░░░░░░░░░░░░░░  32%   causal_due_to + multi_subject on distributions
particle_meas_diag      (19)  ██████████░░░░░░░░░░░░░░░░░░░░  32%   multi_subject on dt/dd pairs + missing Position tokens
auxiliary_heating       (91)  █████░░░░░░░░░░░░░░░░░░░░░░░░░  14%   wave ggd coefficients
core_plasma_physics      (7)  █████░░░░░░░░░░░░░░░░░░░░░░░░░  14%   D-T pair
edge_plasma_physics     (50)  █████░░░░░░░░░░░░░░░░░░░░░░░░░  14%   diamagnetic_component + on_ggd
equilibrium             (96)  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  11%   constraint children + coordinate_of_
general                 (62)  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  11%   multi_subject on fusion pairs
plant_systems           (34)  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░   9%   e_cross_b_drift + coordinate_of_
structural_components   (15)  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7%   energy_multiplication_factor unit=1
electromagnetic_w_diag  (39)  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5%   suprathermal_electrons
transport               (40)  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5%   diamagnetic_component of e_field + temperature_validity
radiation_meas_diag     (37)  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3%   distance_between_measurement_and_sep
(all other domains)                                             0%
```

Two signals dominate: **representation artifacts concentrated in ggd-heavy
IDSs** (mhd, auxiliary_heating, equilibrium ggd subsets) and
**reference-waveform sentinels in pulse_schedule**. Neither is a grammar
failure — both are classifier gaps.

---

## 2. Findings by category

Each finding: concrete names, root cause, proposed resolution, owner.
Owners: **CLS** = codex classifier (`imas_codex/standard_names/classifier.py`
and `source_paths.py`); **CMP** = codex compose prompts
(`imas_codex/llm/prompts/sn/*`); **AUD** = codex audits
(`imas_codex/standard_names/audits.py`); **ISN** = iter-standard-names
vocabularies (`imas_standard_names/grammar/vocabularies/*.yml`);
**DD** = IMAS Data Dictionary upstream filing;
**GRF** = graph-side remediation (purge / consolidate / link); **ENR** =
enrichment prompt (descriptions).

### 2.1 Representation artifacts — classifier gap (B1)

**Quarantine contribution:** 36 audit hits + ~25 secondary (GGD-related
pydantic failures). Dominant in `magnetohydrodynamics` (22 of 32 quarantines)
and `auxiliary_heating` (6 of 13). All already flagged by
`representation_artifact_check`.

**Concrete names (sample):**
- `electron_temperature_finite_element_interpolation_coefficients_on_ggd`
  (dd:mhd/ggd/electrons/temperature/coefficients)
- `mass_density_ggd_coefficients` (dd:mhd/ggd/mass_density/coefficients)
- `ion_temperature_ggd_coefficients`, `ion_number_density_ggd_coefficients`
- `poloidal_magnetic_flux_finite_element_interpolation_coefficients_on_ggd`
- `toroidal_current_density_ggd_coefficients`
- `radial_component_of_magnetic_field_ggd_coefficients`,
  `vertical_component_of_*`, `toroidal_component_of_*` (×10 variants)
- `effective_charge_interpolation_coefficient_on_ggd`
- `radial_component_of_magnetic_vector_potential_interpolation_coefficients`
- `toroidal_component_of_plasma_velocity_ggd_coefficients` (+ r/z siblings)
- `vorticity_per_major_radius_ggd_coefficients`
- `poloidal_component_of_perturbed_magnetic_field_finite_element_coefficients_{real,imaginary}_part`
  (mhd_linear coordinate{1,2,3})

**Root cause.** The DD stores field values at quadrature nodes under
`*/values` *and* basis-function coefficients under `*/coefficients`. Both
share the same parent DataNode → both are classified as physics leaves.
The LLM then faithfully appends `_finite_element_interpolation_coefficients_on_ggd`
or `_ggd_coefficients`.

**Resolution.** Plan 30 already specifies the fix: add `NodeCategory=representation`
for `*/coefficients`, `grid_subset/*`, `grids_ggd/*`, `fourier_mode/*`, and
finite-element basis paths. **Blocker: plan 30 has not landed.** Until it
does, re-running the rotation will regenerate these names.

**Owner:** CLS (plan 30 dependency).

**Secondary defect:** 6 names wrongly frame the coefficient as a
component (e.g. `radial_component_of_magnetic_field_ggd_coefficients` is
treated as a vector_component). Even after classifier exclusion, the
`representation_artifact_check` regex should also match `_ggd_coefficients`,
`_interpolation_coefficient`, `_coefficient_on_ggd`, `_coefficients_imaginary`,
`_coefficients_real` (currently misses `_coefficients_{imaginary,real}_part`
on mhd_linear). Add as AUD patch.

---

### 2.2 Reference-waveform sentinels — classifier gap (B3)

**Quarantine contribution:** 16 names, all in `plasma_control` with
`name_unit_consistency_check` on `power`/`angle`/`frequency`/`voltage`.

**Concrete names:**
- `electron_cyclotron_beam_frequency_reference_waveform` [unit=1]
- `electron_cyclotron_beam_launched_power_reference_waveform` [unit=1]
- `electron_cyclotron_beam_poloidal_steering_angle_reference_waveform` [unit=1]
- `electron_cyclotron_beam_toroidal_steering_angle_reference_waveform` [unit=1]
- `electron_cyclotron_launched_power_reference_waveform` [unit=1]
- `ion_cyclotron_heating_antenna_reference_frequency` [unit=1]
- `ion_cyclotron_heating_antenna_reference_power` [unit=1]
- `ion_cyclotron_heating_reference_power` [unit=1]
- `loop_voltage_reference_waveform` [unit=1]
- `lower_hybrid_antenna_frequency` [unit=1]
- `lower_hybrid_heating_power` [unit=1]
- `neutral_beam_injection_power_reference` [unit=1]
- `poloidal_field_coil_current_reference_waveform` [unit=1]
- `poloidal_field_coil_supply_voltage_reference` [unit=1]
- `major_radius_of_active_limiter_point`, `major_radius_of_geometric_axis`,
  `major_radius_of_magnetic_axis`, `major_radius_of_strike_point`,
  `vertical_coordinate_of_*` (×4), `resistance_of_poloidal_field_coil`
  — all `unit=1` for geometry/resistance under pulse_schedule/position_control.

**Root cause.** DD paths of the form
`dd:pulse_schedule/<system>/<control>/reference` and
`dd:pulse_schedule/<system>/<control>/reference/data` carry `unit=1` because
pulse_schedule reference signals are **controller setpoints/waveforms**,
not physical quantities. The pulse_schedule IDS intentionally publishes
them dimensionless as reference traces (normalized or in controller units).
Classifier has no rule excluding this subtree.

**Resolution.**
- **CLS:** Exclude all DD paths matching
  `^pulse_schedule/.+/reference(/.+)?$` and
  `^pulse_schedule/.+/reference_waveform$`. These are setpoint/waveform
  signals — not suitable for standard-name extraction. Consumers that want
  a standard name for the controlled quantity should look up the underlying
  physics path. (Note: some existing valid names like
  `plasma_current_reference_waveform`, `effective_charge_reference_waveform`,
  `gas_valve_flow_rate_reference` also live here — these pass audits but
  are semantically identical to the quarantined ones. Policy must be
  consistent: **either all pulse_schedule/reference paths are excluded, or
  a `_reference_setpoint` standard-name surface is defined**. Recommend the
  former for rc14; adopt explicit SNs only for controller-specific concepts
  like `ip_reference_waveform` that have no physics peer.)
- **ISN (optional):** If the graph retains some reference-waveforms, add
  `reference_waveform` as a *transformation* token with the explicit rule
  "outputs a dimensionless controller reference; the physical quantity
  has its own SN". This is probably over-engineering; prefer exclusion.

**Owner:** CLS primary; DD secondary (DD could publish `unit=<same as target>`
or at minimum annotate `lifecycle=obsolescent` for the bare `/reference`
path now that `/reference_waveform` carries the data).

---

### 2.3 Diamagnetic-component drift masquerading as a projection axis (B5)

**Quarantine contribution:** 6 audit hits (diamagnetic_component_check).

**Concrete names:**
- `diamagnetic_component_of_anomalous_current_density`
- `diamagnetic_component_of_ion_exb_drift_velocity`
- `diamagnetic_component_of_ion_velocity_per_magnetic_field_magnitude`
- `diamagnetic_component_of_magnetic_field`
- `diamagnetic_component_of_magnetic_vector_potential`
- `diamagnetic_component_of_electric_field`

**Root cause.** DD paths like
`edge_profiles/ggd/e_field/diamagnetic`,
`core_profiles/profiles_1d/e_field/diamagnetic`, and
`edge_profiles/ggd/ion/velocity/diamagnetic` use `/diamagnetic` as if it
were a spatial projection alongside `/radial`, `/poloidal`, `/toroidal`.
Physically wrong — diamagnetic drift is a **vector quantity defined by the
drift relation `v_dia = (B × ∇p)/(qnB²)`**, not a component axis of the
embedded field. The correct standard name for the projection of `v_dia`
onto a spatial direction is e.g. `radial_component_of_ion_diamagnetic_drift_velocity`.

**Resolution.**
- **DD:** File issue upstream — rename `/diamagnetic` siblings under
  `*/velocity/` and `*/e_field/` to reflect that they are drift magnitudes
  or velocity scalars, not spatial components. E.g. rename
  `edge_profiles/ggd/ion/velocity/diamagnetic` →
  `edge_profiles/ggd/ion/diamagnetic_drift_velocity`.
- **CLS:** Until DD fix, add a path-level exclusion for any DD path ending
  in `/diamagnetic` under a parent that is itself a vector container
  (`/velocity/`, `/e_field/`, `/j_tot/`, `/a_field/`, `/b_field/`). Emit the
  path into `representation` or simply skip. The audit
  `diamagnetic_component_check` should fire as a **classifier veto**, not
  just a post-hoc audit.
- **ISN:** The existing `components.yml` comment already documents that
  `diamagnetic` is deferred as a component token — but the error message
  surfaced to compose (`"Token 'diamagnetic' used with 'component_of'
  template is missing from Component vocabulary"`) is misleading. Upgrade
  the parser error to explicitly say: *"`diamagnetic` is a drift qualifier
  (v_dia), not a spatial axis; name the drift as a base
  (`ion_diamagnetic_drift_velocity`) and project with
  `<axis>_component_of_`."*

**Owner:** DD primary; CLS secondary; ISN tertiary (error-message patch).

---

### 2.4 Multi-subject on D-T / D-D / T-T pairs

**Quarantine contribution:** 10 audit hits (multi_subject_check).

**Concrete names:**
- `deuterium_tritium_fusion_power_density`
- `deuterium_tritium_neutron_emissivity`
- `deuterium_tritium_neutron_flux_due_to_{beam_beam,beam_thermal,thermal}_fusion`
- `deuterium_tritium_total_neutron_flux`
- `toroidal_component_of_collisional_torque_density_on_fast_ion_due_to_neutral_beam_injection`
  (ion + neutral)
- `tritium_gas_injection_accumulated_electron_count` (electron + tritium)
- `tritium_to_deuterium_density_ratio` (deuterium + tritium)
- `hydrogen_to_deuterium_density_ratio` (deuterium + hydrogen)

**Root cause.** ISN `subjects.yml` already lists `deuterium_tritium`,
`deuterium_deuterium`, `tritium_tritium` as Tier 3 compound-species
subjects. The audit splits on `_` boundaries and flags **any** token pair
matching two Subject entries, but does not look up the compound forms.

**Resolution.**
- **AUD:** Fix `multi_subject_check` to preferentially match the
  longest compound subject first (`deuterium_tritium` → one subject,
  not `['deuterium', 'tritium']`). Zero-rename; pure false-positive removal.
- **AUD:** Add a grammar-aware exception for explicit binary-operator
  compounds (`<species_A>_to_<species_B>_density_ratio`). These are
  canonical ratio compounds; not a multi-subject violation. See §2.6.
- **AUD:** The `electron + tritium` hit on
  `tritium_gas_injection_accumulated_electron_count` is genuinely
  confusing — the intent is "tritium gas injection quantified in
  electron-equivalents (i.e. ionization electrons)." Rename to
  `tritium_accumulated_gas_injection_electron_equivalent` to match the
  sibling `deuterium_gas_injection_accumulated_electron_equivalent` and
  move `electron_equivalent` to a recognized transformation or qualifier.

**Owner:** AUD primary; GRF secondary (rename electron_count → electron_equivalent).

---

### 2.5 Causal `due_to_<adjective>` — grammar misuse

**Quarantine contribution:** 5 audit hits (causal_due_to_check).

**Concrete names:**
- `heating_power_due_to_halo` → use `due_to_halo_currents`
- `heating_power_due_to_ohmic` → use `due_to_ohmic_dissipation`
- `parallel_heat_flux_power_due_to_halo` → `due_to_halo_currents`
- `toroidal_component_of_collisional_torque_density_on_thermal_ion_due_to_fast_ion`
  → `due_to_fast_ions` or `due_to_fast_ion_slowing_down`
- `toroidal_current_due_to_non_inductive` → `due_to_non_inductive_drive`

**Root cause.** The LLM picks up DD segment names (`power_ohm`,
`halo_currents`, `fast_ion`, `non_inductive`) and splices them with
`due_to_`. The existing ISN `processes.yml` already has
`ohmic_dissipation`, `non_inductive_drive`, and the D.5/rc13 additions
include `halo` (with commentary that bare `halo` is an adjective; the
process is `halo_currents`). The audit fires correctly; the prompt lets
the pattern through.

**Resolution.**
- **CMP:** Add a FORBIDDEN PATTERN exemplar to
  `_exemplars_name_only.md`: *"`due_to_halo` / `due_to_ohmic` /
  `due_to_fast_ion` are adjective forms; always use the process noun
  (`due_to_halo_currents`, `due_to_ohmic_dissipation`,
  `due_to_fast_ions`)."*
- **ISN:** Add plural/noun forms `halo_currents`, `fast_ions` as Process
  tokens (currently `halo` and `fast_ion` are present but their
  adjectival surface forms confuse the LLM).
- **AUD:** Extend `causal_due_to_check` with the `fast_ion` case (already
  caught) and add `halo_currents` / `ohmic_dissipation` as canonical
  replacements in the error message.

**Owner:** CMP + ISN.

---

### 2.6 `_ratio` compounds and near-duplicate density ratios

**Concrete names:**
- `tritium_to_deuterium_density_ratio` (quarantined, multi_subject)
- `hydrogen_to_deuterium_density_ratio` (quarantined, multi_subject)
- `hydrogen_isotope_density_ratio` (valid)
- `neutron_emissivity_reconstruction_relative_error` (valid)

**Root cause.** Binary-operator segment for `ratio_of_X_to_Y` is not in
ISN grammar (confirmed: `binary_operators.yml` lists only the operator
markers; there is no canonical `ratio_of_<A>_to_<B>_density` template in
the catalog). The LLM invents `X_to_Y_density_ratio`, which the audit
then rejects as multi-subject.

**Resolution.**
- **ISN:** Add a formal *ratio* transformation token or canonical
  binary-operator form:
  `ratio_of_<subject_A>_<base>_to_<subject_B>_<base>` (density ratio,
  temperature ratio, pressure ratio). Document example:
  `ratio_of_tritium_number_density_to_deuterium_number_density`.
- **GRF:** Rename the two quarantined entries to the new form.
- **AUD:** Multi_subject_check gets an exception for explicit ratio
  compounds (see §2.4).

**Owner:** ISN primary.

---

### 2.7 Vocabulary gaps (B4)

Each entry: the missing token, example quarantined name(s), proposed
ISN enum addition, rationale.

| Missing token | Segment | Example name | Rationale |
|---|---|---|---|
| `e_cross_b_drift` | Process | `vertical_component_of_ion_velocity_due_to_e_cross_b_drift`, `normalized_toroidal_flux_coordinate_displacement_due_to_e_cross_b_drift` | E×B drift is a physics mechanism; already referenced in DD (`pellets/.../rho_tor_norm_drift`) and plasma rotation literature. |
| `thermalization_of_fast_particles` | Process | `heating_power_due_to_thermalization_of_fast_particles`, `thermal_particle_number_density_source_due_to_thermalization_of_fast_particles` | Distribution-source mechanism in `distributions/.../thermalisation/*`. Extending the existing `thermalization` token with the `_of_fast_particles` qualifier (or permitting `thermalization_of_<subject>` template) resolves both. |
| `thermal_fusion` | Process | `deuterium_deuterium_neutron_flux_due_to_thermal_fusion`, `tritium_tritium_neutron_flux_due_to_thermal_fusion` | ISN has `beam_beam_fusion`, `beam_thermal_fusion`, but not the pure-thermal variant. Add `thermal_fusion`. |
| `suprathermal_electrons` | Subject or Population-qualifier | `suprathermal_electrons` (quarantined, electromagnetic_wave_diagnostics) | DD has `electrons_suprathermal`; population tier between `thermal` and `fast`. Add to `subjects.yml` or to a new population axis. |
| `ferritic_element_centroid` | Position | `toroidal_component_of_magnetic_field_at_ferritic_element_centroid` | rc13 already added `ferritic_insert_centroid`; the `_element_` variant comes from the DD path `ferritic/element/*/time_slice/b_field_tor`. Add as synonym or rename the DD ingest to use `insert` consistently. |
| `neutron_detector` | Position | `neutron_flux_at_neutron_detector` | Diagnostic detector position; currently only `neutron_detector` is in Objects, not Positions. Either promote or add an Objects→Position mirror rule. |
| `normalized_parallel` | Component | none in rc13 (but next rotation will hit `normalized_parallel_refractive_index`) | rc13 added `normalized_radial`, `normalized_vertical`; `normalized_parallel` and `normalized_perpendicular` are the natural complements. |
| `translation` | Component or **remove** | `translation_component_of_plasma_displacement` (from mhd_linear; currently mislabeled as Component) | "Translation" is not a spatial axis; it is the zeroth-order mode of a perturbation expansion. Drop from Component; promote to a Process or transformation (`due_to_translation_mode`) or spell out as `normal_component_of_plasma_boundary_displacement`. |
| `boundary_condition_value` / `boundary_condition_on_ggd` | **exclude** | `electron_energy_transport_boundary_condition_value`, etc. | Solver-configuration metadata, not physics. DD `transport_solver_numerics/boundary_conditions_1d/*`. Classifier should exclude the subtree. See §2.8. |
| `extensive_cumulative` / `extensive_integrated` | **remove** | `line_integrated_electron_density_constraint_measurement_time` | Audit already catches `integrated_` misuse; see §2.11. |
| `electron_equivalent` | Transformation | `tritium_gas_injection_accumulated_electron_count` (rename to `..._electron_equivalent`) | DD publishes several gas-injection counts measured in ionization-electron equivalents. Add as transformation + documentation. |

---

### 2.8 Solver / boundary-condition paths (classifier gap)

**Concrete names (all quarantined under `computational_workflow`):**
- `electron_energy_transport_boundary_condition_value`
- `electron_particle_boundary_condition_on_ggd`
- `ion_energy_transport_boundary_condition_value`
- `electron_particle_boundary_condition_value`
- `momentum_transport_boundary_condition_on_ggd`

**Root cause.** DD subtree `transport_solver_numerics/boundary_conditions_*`
exposes solver-internal Dirichlet/Neumann choices as leaf nodes. Every
leaf becomes an SN because the classifier sees
`electron_energy_transport` + `value` as a physics child.

**Resolution.**
- **CLS:** Exclude all paths matching
  `transport_solver_numerics/boundary_conditions_(1d|ggd)/.*` and
  `transport_solver_numerics/solver_1d/.*/coefficient.*`. These are solver
  configuration artefacts — analogous to the `fit_artifact` category in
  plan 30. Propose adding `solver_artifact` as a third NodeCategory,
  *or* subsume under `fit_artifact` with broadened description.

**Owner:** CLS (plan 30 extension).

---

### 2.9 Contradictions inside the compose prompt (B2)

**File:** `imas_codex/llm/prompts/sn/compose_system.md` and
`imas_codex/llm/prompts/shared/sn/_exemplars_name_only.md`.

**Contradiction 1 — `_at_X` vs `_of_X` for positional qualifiers.**
Compose_system.md Rules 18/21 (lines 676–691) declare that
`_of_plasma_boundary`, `_of_magnetic_axis`, `_of_x_point` are preferred
over `_at_plasma_boundary`, `_at_magnetic_axis`, `_at_x_point`.
Exemplar P3 in `_exemplars_name_only.md` (lines 33–38) still advertises
`at_separatrix`, `at_magnetic_axis`, `at_plasma_boundary` as **GOOD
EXAMPLES**. ISN `positions.yml` permits both. Real corpus shows both
forms (`safety_factor_at_magnetic_axis` [valid],
`vertical_coordinate_of_magnetic_axis` [valid]).

*Recommendation:* split the Position vocabulary semantically — `of_` for
intrinsic geometric properties, `at_` for field values evaluated at a
location (already suggested in `positions.yml` comments). Rewrite P3 to
exhibit both, each in its intended context. Remove the Rule 18/21
prohibition; replace with a rule that enforces *semantic match* instead of
*syntactic preference*.

**Contradiction 2 — `vertical_position_of_` vs `vertical_coordinate_of_`.**
Rule 17 in compose_system.md mandates `vertical_coordinate_of_x_point`.
Exemplar P8 (`_exemplars_name_only.md` line 77) still says
`vertical_position_of_x_point`. The corpus uses `vertical_coordinate_of_`
exclusively (`vertical_coordinate_of_x_point`,
`vertical_coordinate_of_plasma_boundary`, etc.).

*Recommendation:* update P8 to match Rule 17. Add `geometric_base=position`
as the ISN canonical base; `vertical_position_of_*` remains grammatical
but `vertical_coordinate_of_*` is preferred for poloidal-plane Z coords.
Document the distinction.

**Contradiction 3 — allowed transformations.**
`_grammar_reference.md` lists 4 canonical transformation tokens:
`flux_surface_averaged`, `line_integrated`, `norm`, `square_root`.
But `transformations.yml` in ISN rc13 defines 30+ tokens including
`volume_averaged`, `time_integrated`, `time_derivative_of`,
`maximum_of`, `per_toroidal_mode`, `cumulative_inside_flux_surface`,
`amplitude_of`, etc. Compose_system.md NC-20 treats
`volume_averaged`, `surface_averaged`, `line_averaged`,
`density_averaged` as valid.

*Recommendation:* delete the 4-token allow-list from
`_grammar_reference.md`; replace with a pointer to the live
`transformations.yml` enum (generated/kept-in-sync by `build-models`).
Add a rule that the LLM must ONLY use tokens from the enum — the
compose_system.md prompt loader should inject the live enum at render time.

**Contradiction 4 — `_over_` preposition.**
Compose_system.md (lines 59–61) forbids `_over_<quantity>` as a
division surrogate (must be `_per_`). But the corpus contains
`toroidal_component_of_ion_velocity_over_magnetic_field_strength` (valid,
edge_plasma_physics) and its sibling `*_per_magnetic_field_magnitude`
(valid, edge_plasma_physics). DD uses `velocity_parallel_over_b_field` in
the path names.

*Recommendation:*
- **AUD:** Add an audit that flags bare `_over_<quantity>_strength|_magnitude`
  as a division surrogate and renames to `_per_magnetic_field_strength`.
- **ISN:** Confirm `over_` is reserved for **region** segments
  (`over_halo_region`) per `regions.yml`; it is NEVER a mathematical
  division surrogate. Make this explicit in `_grammar_reference.md`.
- **GRF:** Merge the two variants
  (`*_over_magnetic_field_strength` → `*_per_magnetic_field_strength`;
  or, if both refer to different underlying DD paths, consolidate on the
  DD-canonical `_per_` form).

---

### 2.10 Near-duplicate families — consolidation candidates

These are **valid-but-proliferating** names that should be collapsed
via a population/surface qualifier axis (R1 F3 / R2 A.1 already
recommend this; confirmed in rc13 export).

**Family A — `wave_absorbed_power` (8 variants):**
- `wave_absorbed_power`, `electron_wave_absorbed_power`,
  `thermal_electron_wave_absorbed_power`, `fast_electron_wave_absorbed_power`,
  `ion_wave_absorbed_power`, `thermal_ion_wave_absorbed_power`,
  `fast_ion_wave_absorbed_power`,
  `ion_wave_absorbed_power_at_beam_tracing_point`
- Plus density and inside-flux-surface variants: 23 total.

*Resolution:* consolidate to `wave_absorbed_power` with axes
{species: electron|ion, population: thermal|fast|total, surface: global|
flux_surface|flux_surface_integral, per_mode: null|toroidal_mode,
location: null|beam_tracing_point}.

**Family B — `flux_surface_averaged_*_metric_coefficient` (gm1–gm9):**
- `flux_surface_averaged_gradient_of_toroidal_flux_coordinate` (gm7)
- `flux_surface_averaged_squared_gradient_of_toroidal_flux_coordinate` (gm3)
- `flux_surface_averaged_inverse_magnetic_field_squared` (gm4)
- `flux_surface_averaged_inverse_magnetic_field_squared_gradient_squared` (gm6)
- `flux_surface_averaged_inverse_major_radius` (gm9)
- `flux_surface_averaged_inverse_major_radius_squared` (gm1)
- `flux_surface_averaged_inverse_major_radius_squared_toroidal_flux_gradient_squared` (gm2)
- `flux_surface_averaged_major_radius` (gm8)

*Resolution:* correct-as-is. Each is a distinct metric coefficient and
the names already match the DD labels. Add descriptions (enrichment rotation).
Cross-link `equilibrium.gm1..gm9` to the canonical ISN surface via
`derived_from`/`REFERENCES`.

**Family C — metric tensor `g{ij}` components (12 variants):**
- `contravariant_metric_tensor_g11 … g33` (6)
- `covariant_metric_tensor_g11 … g33` (6)

*Resolution:* consolidate to `metric_tensor_component` with structured
axes `{variance: covariant|contravariant, i: 1..3, j: 1..3}`. Requires
ISN support for **indexed tensor axes** (explicitly deferred in plan 29,
ADR-1 non-goal). Recommend: keep as 12 distinct names for rc14 but add a
parent `metric_tensor` SN and link each component to it via `PART_OF`
(see plan 27 sn-vector-hierarchy).

**Family D — `radial|poloidal|toroidal_component_of_perturbed_magnetic_field_{imaginary,real}_part`:**
8 variants (3 axes × 2 parts) + 6 more with `_finite_element_coefficients_`.

*Resolution:* the 8 non-coefficient variants are valid and stay. The 6
coefficient variants are classifier-excluded by plan 30.
Add vector parent `perturbed_magnetic_field` (kind=vector) and link
components via `PART_OF`.

**Family E — pulse_schedule `*_reference_waveform`:**
See §2.2; policy is exclusion, not consolidation.

**Family F — `deuterium_deuterium_*_neutron_flux_*`, `tritium_tritium_*`, `deuterium_tritium_*`:**
15+ variants across `fusion/neutron_fluxes/*` and `fusion/neutron_rates/*`.
Reactions are physically distinct; keep separate names but:
- Normalize `*_neutron_flux_*` vs `*_neutron_rate_*` — DD ships both
  under sibling paths with **identical units (Hz)**. This is a DD
  duplication; codex should pick one canonical name per (reaction,
  drive-mechanism) and collapse the 2× path count.
- Canonicalize compound subjects using existing
  `deuterium_tritium`/`deuterium_deuterium`/`tritium_tritium` tokens to
  silence the multi_subject_check (see §2.4).

**Family G — `*_gas_injection_accumulated_*` (~18 variants per species):**
- `accumulated_gas_injection_of_helium_3` (one form)
- `ammonia_gas_injection_accumulated_molecule_count` (another)
- `beryllium_accumulated_gas_injection` (a third)
- `deuterium_gas_injection_accumulated_electron_equivalent`, etc.

**Problem:** three different segment orderings across species
— `accumulated_<species>_gas_injection`, `<species>_gas_injection_accumulated_<count_type>`,
`<species>_accumulated_gas_injection`.

*Resolution:* canonicalize to
`<species>_accumulated_gas_injection_<count_type>` for the ~15 species
that carry distinct count-type flavors (electron_equivalent, molecule_count,
bare count). Add an alias SN (`accumulated_gas_injection_of_<species>`)
that points to the canonical via `REFERENCES`. Fixes ordering
inconsistency R1-F7.

---

### 2.11 Audit coverage gaps

**Cases currently missed:**

1. **Trailing `_on_ggd` suffix** — `representation_artifact_check` uses a
   regex on substrings like `_finite_element_interpolation_coefficients_`,
   `_ggd_coefficients`, `_coefficient_on_ggd`. But **plain
   `_on_ggd`** (e.g. `radial_component_of_electron_velocity_on_ggd`,
   `electron_particle_boundary_condition_on_ggd`) slips through when the
   content is not clearly a basis-function coefficient.
   **Fix (AUD):** add suffix match `_on_ggd$` as a hint to the
   classifier, not just the audit; surface to compose as "GGD node
   indicator, strip suffix or re-route".

2. **Bare `_field` implicit-qualifier** — `implicit_field_check` catches
   `vacuum_toroidal_field_function` and
   `poloidal_field_coil_current_reference_waveform`. Good. But the
   `poloidal_field_coil_supply_voltage_reference` case fires
   `implicit_field_check` + `name_unit_consistency_check` + nothing else;
   the correct rename is `poloidal_magnetic_field_coil_supply_voltage`
   — audit should **suggest** the fix, not just flag.

3. **Stacked transformations** — the name
   `line_integrated_electron_density_constraint_measurement_time` hits
   `cumulative_prefix_check` correctly, but the root cause is a stacked
   qualifier: `line_integrated` + `electron_density` + `constraint` +
   `measurement_time`. This kind of name is produced when the classifier
   treats `/n_e_line/time_measurement` as a physics leaf instead of as a
   constraint child. The audit should add a sibling check: *if
   `_constraint_measurement_time` suffix is present, physical_base should
   match the sibling constraint's name, not include cumulative/integration
   prefixes.* Alternatively, plan 30's `fit_artifact` category sweeps
   these out.

4. **Compound-subject multi_subject false positive** — §2.4 already
   addresses: `deuterium_tritium`, `deuterium_deuterium`, `tritium_tritium`
   must be matched greedy-first.

5. **`_reference` vs `_reference_waveform` inconsistency** — 5 names
   end in `_reference` (`neutral_beam_injection_power_reference`,
   `ion_cyclotron_heating_antenna_reference_power`, etc.); ~12 end in
   `_reference_waveform`. All are pulse_schedule sentinels (§2.2).
   Add audit pattern: if name ends in `_reference` or `_reference_waveform`
   **and** DD path matches `pulse_schedule/.*/reference`, **quarantine
   regardless of unit.**

6. **Missing audit: "trivial definitional duplicate."** Pairs like
   `parallel_velocity_per_magnetic_field_strength` (valid) and
   `parallel_component_of_plasma_velocity_per_magnetic_field_magnitude`
   (valid) differ only in `velocity` vs `plasma_velocity` and `strength`
   vs `magnitude`. An audit should detect two names with shared physics
   semantics and identical units (see 2.12).

7. **Unit validity on placeholder units.** `grid_object_measure` has
   unit `m^dimension`; `atom_multiplicity_of_isotope` has unit
   `Elementary Charge Unit`. The current `unit_validity` audit fires
   once (1 hit) but should fire on any DD unit containing whitespace,
   "e" as a unit-atom (vs the dimensionless `e` electron charge), or
   non-UDUNITS syntax. Add specific pattern matches.

---

### 2.12 Silent near-duplicates (different names, same physics)

**Concrete pairs:**
- `parallel_velocity_per_magnetic_field_strength` [scalar]
  ≈ `parallel_component_of_plasma_velocity_per_magnetic_field_magnitude`
  [vector_component] — unit `m.s^-1.T^-1`, same DD family
  (`mhd/ggd/velocity_parallel_over_b_field`).
- `plasma_mass_density` [valid] ≈ `ion_number_density_times_average_mass`
  (not yet minted, but the DD path `mhd/ggd/mass_density/values` encodes
  it). Currently just one SN.
- `major_radius_of_current_center` [valid] ≈ `major_radius_of_plasma_filament`
  [valid, sources the same `current_centre` DD nodes in some IDS
  variants] — 7-path overlap.
- `electron_diamagnetic_drift_velocity` [valid, scalar] vs a would-be
  vector-component form
  `<axis>_component_of_electron_diamagnetic_drift_velocity`
  (not yet minted; six DD paths are available).
- `faraday_rotation_angle` [valid, scalar] vs `faraday_angle_constraint_weight`
  [valid] — the latter is a fit artifact (plan 30 would classify as
  `fit_artifact`); one name only survives after plan 30.
- `toroidal_component_of_magnetic_field_of_magnetic_axis` (valid, vector)
  vs `toroidal_component_of_magnetic_field_at_reference_major_radius`
  (valid, scalar) vs `vacuum_toroidal_magnetic_field_at_reference_major_radius`
  (valid, scalar) — three names for what physicists call `B_tor(R_ref)`
  or `B_tor(R0)`. Consolidation candidate.
- `passive_loop_current` + `passive_loop_current_constraint_weight` +
  `iron_core_segment_{radial,vertical}_magnetization_constraint_{weight,measurement_time}`
  — all fit-artifact children; plan 30 eliminates the `_weight` /
  `_measurement_time` siblings.

*Resolution:* **GRF** — introduce a `NEAR_DUPLICATE_OF` relationship in
the graph (undirected, weighted by unit+cluster agreement) after plan 30
lands. Use it as an enrichment input (cross-reference each other in
documentation). Surface these pairs to human review in the next rotation.

---

### 2.13 DD-upstream issue list (B5 detail)

Each line: DD path / pattern, issue, proposed filing.

| DD path / pattern | Issue | Proposed fix |
|---|---|---|
| `*/ggd/*/diamagnetic` under vector containers | `/diamagnetic` used as a spatial-axis sibling to `/radial`, `/poloidal`, `/toroidal`. Physically incorrect. | Rename `/diamagnetic` → `/diamagnetic_drift_velocity` (under `/velocity/`) and equivalent qualified names under `/e_field/`, `/j_tot/`, `/a_field/`. |
| `equilibrium/time_slice/ggd/grid_object/measure` | Unit `m^dimension` (placeholder). | Replace with `units` that depend on `dimension` via a `units_template` or split into four leaves (`measure_point_count` [1], `measure_length` [m], `measure_area` [m²], `measure_volume` [m³]). |
| `summary/particles/ion/state/atom_multiplicity` | Unit `Elementary Charge Unit` — whitespace-containing UDUNITS non-standard. | Rename unit to `e` (elementary charge) or `1` (integer multiplicity). Codex unit parser rejects whitespace. |
| `equilibrium/time_slice/constraints/*/exact` (integer flag) | Flag nodes treated as physics quantities. | Add lifecycle `obsolescent` or `constant` + documentation; codex classifier needs plan-30 `fit_artifact` anyway. |
| `pulse_schedule/*/reference` and `pulse_schedule/*/reference/data` | All publish `unit=1` regardless of the underlying quantity's physical unit. | Either (a) publish unit of the controlled quantity, (b) add explicit `description` tag `"controller-reference; dimensionless normalized"`, or (c) add `lifecycle=alpha` so codex can filter. |
| `summary/fusion/neutron_fluxes/*` vs `summary/fusion/neutron_rates/*` | Duplicate leaves with identical unit `Hz`. | DD should deprecate one (preferably `neutron_rates` — "flux" is the standard term) or explicitly document which is authoritative. |
| `*/_error_upper`, `*/_error_lower`, `*/_error_index` | Expected to be discovered as companion error fields (fetch_dd_error_fields tool exists). | No fix required; note that codex already excludes these from enumeration. |
| `ferritic/element/*` vs `ferritic/object/*` (DD 4 renaming) | Same semantic content under two different parent names during DD 3→4 migration. | Fold `/element/` into `/object/` or add an explicit `RENAMED_TO` record; ferritic_element_centroid vocab gap traces to this. |
| `waves/coherent_wave/*` full_wave/values | Same as GGD: both `values` and basis-function coefficient leaves. | Tag basis-function leaves with `NodeCategory=representation` (plan 30 sweeps). |
| `distributions/distribution/*/thermalisation/*` | British spelling `thermalisation`; DD also has `thermalization`. | Pick one spelling; DD currently mints both in some IDSs (see sources of `thermal_particle_number_density_source_due_to_thermalization_of_fast_particles`). |
| `summary/gas_injection_accumulated/.../value` with unit `1` | Same DD-units issue: counts published as dimensionless. Correct (counts are dimensionless), but the audit `name_unit_consistency_check` trips when the name says `electron_equivalent` or `molecule_count`. | Either add `description` clarifying the count semantics, or publish unit as `count` (UCUM). |

---

### 2.14 Description / documentation quality (ENR)

**Coverage:** 84 / 705 names have descriptions (11.9%). The enrichment
rotation has not yet run on rc13 output; this is expected per plan 29
ADR-1 (split generate/enrich) and §1 of this report.

**Spot-check of the 84 existing descriptions:**

- **Length distribution:** median ~190 chars, max 350 chars, min 45 chars.
  Catalog target (GAP-1 in plan 11) is ≥1200 chars. Current descriptions
  are exposition-only; they lack the 7-section structure
  (opening / governing equation / physical significance / measurement /
  typical values / sign convention / cross-references).
- **Good examples (near-catalog quality):**
  - `magnetic_shear`: equation + definition + normalization.
  - `plasma_inductance`: equation + defining relation.
  - `normalized_toroidal_beta`: physical significance + stability context.
- **Weak examples (need enrichment):**
  - `poloidal_angle`: one sentence, no sign convention.
  - `vertical_coordinate_of_plasma_boundary_gap_reference_point`: pure
    tautology of the name.
  - `major_radius_of_plasma_boundary_gap_reference_point`: pure tautology.
- **Description-tag mismatch:**
  - `magnetic_shear`: description says "radial variation of the safety
    factor" but tag is `spatial-profile`. Audit flagged it. Enrichment
    should either rewrite the description or change the tag.

**Cross-linking signals:** 45+ `link_not_found:name:*` issues in the
audit log. Root cause is compose-time ordering: the LLM references names
that don't yet exist in the graph. R3-architecture §B.3 already
recommends a two-phase POSTLINK (plan 29 split generate/enrich); confirm
it populates REFERENCES after enrich sees all minted names.

---

### 2.15 Length / readability — objective measurements

Using the `head -c` byte count on the 705 names:

- **Mean:** 47 chars · **Median:** 44 chars · **Max:** 113 chars
  (`flux_surface_averaged_inverse_major_radius_squared_toroidal_flux_gradient_squared`)
- **≥70 chars:** 58 names (8.2%). Typical offenders:
  - `toroidal_component_of_collisional_torque_density_on_thermal_ion_due_to_fast_ion` (82)
  - `distance_between_measurement_position_and_separatrix_mapped_to_outboard_midplane` (83)
  - `thermal_particle_number_density_source_due_to_thermalization_of_fast_particles` (80)
  - `electron_cyclotron_beam_poloidal_steering_angle_reference_waveform` (67)
  - `ion_wave_absorbed_power_inside_flux_surface_per_toroidal_mode` (62)
  - `poloidal_component_of_perturbed_magnetic_field_finite_element_coefficients_imaginary_part` (90)

- **≥100 chars: 2 names.** Both come from the GGD subtree or chained
  qualifiers that plan 30 / §2.2 will classifier-exclude.

**Recommendation:** no hard cap (catalog has a 113-char name that is
intrinsic to the physics). Target a **p95 ≤ 75 chars** after classifier
fixes and consolidation (§2.10).

---

### 2.16 Coordinate-prefix pydantic errors (B2 secondary)

**Pattern:** 6 quarantines with
`Token 'X_coordinate_of' used as coordinate prefix is missing from
Component vocabulary`.

**Examples:**
- `major_radius_of_outline_point`,
- `vertical_coordinate_of_outline_point`,
- `vertical_coordinate_of_control_surface_normal_vector`,
- `normalized_toroidal_flux_coordinate_displacement_due_to_e_cross_b_drift`
- `toroidal_angle_of_outline_point`

**Root cause.** ISN grammar distinguishes **coordinate prefix** (axis
label of a Cartesian/cylindrical coordinate) from **Component segment**
(vector projection). `major_radius_of_`, `vertical_coordinate_of_`,
`toroidal_angle_of_` are coordinate prefixes when applied to a
Position/Object and Components when applied to a vector quantity. The
compose Rule 17 is about coordinate prefixes; the pydantic validator
thinks the same tokens are Components. Double-definition.

**Resolution.**
- **ISN:** Split Component vocabulary into **axis** (radial, toroidal,
  vertical, parallel, perpendicular, …) and **coordinate_prefix**
  (major_radius_of, vertical_coordinate_of, toroidal_angle_of,
  minor_radius_of). The R1-F6 proposal (deferred in plan 29) covers this;
  rc14 should land it.
- **AUD:** Until ISN is split, suppress the pydantic error for names
  matching `^(major_radius_of|vertical_coordinate_of|toroidal_angle_of|normalized_toroidal_flux_coordinate)_` and route them through a
  `coordinate_prefix_check` audit that confirms the suffix is a
  Position/Object token.

**Owner:** ISN primary (R1-F6); AUD as interim.

---

### 2.17 `_reference_waveform` inside plasma_control — a policy gap

See §2.2. Summary: until the classifier excludes
`pulse_schedule/.*/reference.*`, the LLM will keep generating names
whose units are mismatched to their `power`/`angle`/`frequency`/`voltage`
mentions. **Prompt cannot fix this** because the DD unit is
authoritative. Classifier exclusion is the only path.

---

## 3. Prompt root-cause analysis — file-by-file

| File | Line(s) | Issue | Fix |
|---|---|---|---|
| `compose_system.md` | 54–56 | Forbids `_over_<quantity>` but does not cross-reference the legitimate `over_<region>` Region segment. Causes ambiguity. | Replace the line with *"`_over_<quantity>` is forbidden (use `_per_`); `over_<region>` is the Region segment — see `_grammar_reference.md`."* |
| `compose_system.md` | 249–474 (NC-1..NC-26) | NC-7 forbids dimension-suffix `_2d`, `_3d`, yet `grid_object_geometry_2d` [valid] passes. | Promote to AUD `structural_dim_tag_check` that runs on names, not just docs. |
| `compose_system.md` | 633–698 (Rules 1..22) | Rule 22 bans `diamagnetic_component_of_*`; LLM keeps producing it because the DD path is literal. | Classifier-level veto (§2.3) — prompt cannot prevent it. Downgrade Rule 22 to a reminder. |
| `compose_system.md` | 676–691 (Rules 17/18/21) | Contradicts exemplars P3/P8. | Update exemplars (see below); align rule wording with `positions.yml` of/at semantics. |
| `_exemplars_name_only.md` | 33–38 (P3) | Advertises `at_` for positional qualifiers that Rules 18/21 now demote. | Rewrite P3: split into P3a (`at_<position>` for field-at-locus) and P3b (`of_<position>` for intrinsic geometric property). Provide one example of each. |
| `_exemplars_name_only.md` | 77 (P8) | `vertical_position_of_x_point` vs Rule 17 `vertical_coordinate_of_x_point`. | Change P8 to `vertical_coordinate_of_x_point`. |
| `_grammar_reference.md` | 115–125 (transformations table) | 4-token allow-list contradicts `transformations.yml` 30+ tokens. | Delete the table; render the live enum via `prompt_loader` at compose time. |
| `_grammar_reference.md` | generic_bases table | Lists 12 tokens; `generic_physical_bases.yml` lists 12 (matches). OK. | No change. |
| `compose_dd.md`, `compose_signals.md` | — | Not reviewed in this pass. Next rotation should check they mirror `compose_system.md` wording. | Audit during rc14. |
| `_controlled_tags.md` | — | Not reviewed; flag `spatial-profile` vs description content (see §2.14 `magnetic_shear`). | Enrichment rotation fix. |
| `_scoring_rubric.md` | — | Not reviewed. Confirm it scores documentation length ≥1200 chars per plan 11 GAP-1. | Enrichment rotation fix. |

---

## 4. Prioritized recommendation register

### 4.1 imas-codex (CLS / CMP / AUD)

| Pri | ID | Action | Owner | Dep |
|---|---|---|---|---|
| P0 | CLS-01 | Land plan 30 (NodeCategory `fit_artifact`, `representation`). | CLS | — |
| P0 | CLS-02 | Extend plan 30 with `solver_artifact` (or fold into `fit_artifact`) covering `transport_solver_numerics/boundary_conditions_*`. | CLS | CLS-01 |
| P0 | CLS-03 | Exclude `pulse_schedule/.*/reference(/.*)?` and `pulse_schedule/.*/reference_waveform(/.*)?$`. Alternative: classify as `representation` with documentation that explains. | CLS | CLS-01 |
| P0 | CLS-04 | Veto `/diamagnetic` leaf paths under vector containers (`/velocity/diamagnetic`, `/e_field/diamagnetic`, `/a_field/diamagnetic`, `/j_tot/diamagnetic`). | CLS | — |
| P0 | CMP-01 | Rewrite exemplars P3, P5, P8 in `_exemplars_name_only.md` to match compose_system Rules 17/18/21. | CMP | — |
| P0 | CMP-02 | Replace the 4-token transformation allow-list in `_grammar_reference.md` with a live-rendered enum from `transformations.yml`. | CMP | — |
| P1 | AUD-01 | Fix `multi_subject_check` to prefer longest compound-subject match (e.g. `deuterium_tritium` > `deuterium` + `tritium`). | AUD | — |
| P1 | AUD-02 | Add `pulse_schedule_reference_check`: any DD path matching the pattern is quarantined regardless of unit/name. | AUD | CLS-03 interim |
| P1 | AUD-03 | Extend `representation_artifact_check` to match `_coefficients_(imaginary\|real)_part`, `_ggd_coefficients$`, `_on_ggd$`, `_interpolation_coefficient`. | AUD | — |
| P1 | AUD-04 | Add `coordinate_prefix_check` (interim to ISN-F6) to whitelist `major_radius_of_/vertical_coordinate_of_/toroidal_angle_of_` preceding a Position or Object. | AUD | — |
| P1 | AUD-05 | Fix unit-whitespace acceptance in `unit_validity` check to reject `Elementary Charge Unit`, `m^dimension`, and similar. | AUD | — |
| P2 | AUD-06 | Add near-duplicate detector (semantic-cluster + same-unit pairs). Emit as advisory, not block. | AUD | ENR |
| P2 | CMP-03 | Rewrite anti-pattern A-family in `_exemplars_name_only.md` to cover `_on_ggd` suffix, `_reference_waveform` suffix, `_ggd_coefficients` suffix. | CMP | — |
| P2 | CMP-04 | Add prompt cue on compound-subject canonicalization — "use `deuterium_tritium` as a single Subject token, never `deuterium_*_tritium_*`." | CMP | — |

### 4.2 iter-standard-names (ISN)

| Pri | ID | Action | Dep |
|---|---|---|---|
| P0 | ISN-01 | Add `e_cross_b_drift` to `processes.yml`. | — |
| P0 | ISN-02 | Add `thermal_fusion` to `processes.yml`; document as parallel to `beam_beam_fusion`/`beam_thermal_fusion`. | — |
| P0 | ISN-03 | Add `thermalization_of_fast_particles` to `processes.yml` or broaden `thermalization` template to accept `thermalization_of_<subject>`. | — |
| P1 | ISN-04 | Add `suprathermal_electrons` to `subjects.yml` (Tier 3) and document population tier between `thermal` and `fast`. | — |
| P1 | ISN-05 | Add `ferritic_element_centroid` to `positions.yml` (synonym or canonicalize `insert` vs `element`). | — |
| P1 | ISN-06 | Add `neutron_detector` to `positions.yml` (promote Objects→Position mirror). | — |
| P1 | ISN-07 | Add `normalized_parallel`, `normalized_perpendicular` to `components.yml` (for next rotation's wave refractive-index names). | — |
| P1 | ISN-08 | Drop `translation` from Component vocabulary (confirmed misplaced) and re-route via Process/Transformation. | — |
| P1 | ISN-09 | Add `electron_equivalent` to `transformations.yml` (for gas-injection counts). | — |
| P1 | ISN-10 | Add canonical `ratio_of_<A>_to_<B>` pattern via `binary_operators.yml` extension or a transformation `ratio_of`. | — |
| P2 | ISN-11 | Split Component vocabulary into **axis** vs **coordinate_prefix** (R1-F6 finally landing). | — |
| P2 | ISN-12 | Improve parser error message on `diamagnetic_component_of_X`: suggest `<axis>_component_of_X_diamagnetic_drift_velocity`. | — |
| P3 | ISN-13 | Investigate tensor-component grammar segment (R1-D4 deferred) for `metric_tensor_g{ij}` consolidation. | — |

### 4.3 IMAS DD — upstream filings (Jira / Bitbucket / GGUS)

| Pri | ID | Filing | Scope |
|---|---|---|---|
| P0 | DD-01 | Rename `/diamagnetic` siblings under vector containers (see table §2.13). | edge_profiles, core_profiles, plasma_profiles |
| P0 | DD-02 | Sanitize unit strings: `Elementary Charge Unit` → `e`; `m^dimension` → per-object unit. | summary/particles, equilibrium/ggd |
| P1 | DD-03 | Clarify `pulse_schedule/*/reference` units — publish underlying quantity's unit, or add explicit dimensionless-reference annotation. | pulse_schedule |
| P1 | DD-04 | Deprecate one of `neutron_fluxes`/`neutron_rates` duplicate subtrees. | summary/fusion |
| P2 | DD-05 | Converge British/American spelling: `thermalisation` ↔ `thermalization`. | distributions |
| P2 | DD-06 | Fold `ferritic/element/*` → `ferritic/object/*` or add `RENAMED_TO`. | ferritic |

### 4.4 Graph remediation (GRF) — after CLS/CMP/ISN land

| Pri | ID | Action |
|---|---|---|
| P0 | GRF-01 | After plan 30: purge all quarantined names flagged `representation_artifact`, `fit_artifact`, or under pulse_schedule/reference. Expected deletion: ~60 nodes. |
| P0 | GRF-02 | Re-run compose over rc13 quarantines; expect <5% residual quarantine rate. |
| P1 | GRF-03 | Consolidate `wave_absorbed_power` family (§2.10 Family A) into a vector parent + axes. |
| P1 | GRF-04 | Consolidate `metric_tensor_g{ij}` family (§2.10 Family C) — parent + `PART_OF`. |
| P1 | GRF-05 | Normalize segment order in `*_gas_injection_accumulated_*` family (§2.10 Family G). |
| P2 | GRF-06 | Add `NEAR_DUPLICATE_OF` relationship (semantic cluster + unit); populate with §2.12 pairs. |
| P2 | GRF-07 | Run enrichment rotation (plan 29 ENRICH) targeting 0/84 coverage gap; enforce documentation length ≥ 1200 chars. |

---

## 5. Success criteria (for rc14 bootstrap)

- **Quarantine rate:** ≤ 5% (from 17.2%). Driven by CLS-01..04.
- **Valid-name count:** 580 ± 30 (after consolidation might drop;
  acceptable).
- **Description coverage:** ≥ 90% (ENR rotation).
- **Description length:** median ≥ 1200 chars.
- **Reviewer score:** avg ≥ 0.80 per plan 11.
- **No prompt/exemplar contradictions** (CMP-01/02).
- **No pydantic coordinate_prefix errors** (AUD-04 or ISN-11).
- **≥ 10 grammar vocabulary tokens** added (ISN-01..09).
- **≥ 4 DD issues filed** (DD-01..04).

---

## 6. Out of scope / deferred

- Tensor grammar (R1-D4 / ISN-13) — needs ADR and parser work.
- `basis_frame` segment (R1-F5) — deferred.
- New physics_domain values — continue to use existing 24.
- Migration of downstream consumers of `review_status='drafted'`
  (plan 29 one-release alias handles this).
- Rebuilding the 6,000 facility-signal MEASURES links — separate
  rotation.

---

*— End of 12-full-graph-review.md*
