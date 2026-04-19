# DD-04 — `pulse_schedule/*/reference` nodes carry sentinel `unit=1` regardless of physical quantity

## DD path(s) / pattern

```
pulse_schedule/*/reference
pulse_schedule/*/reference/data
pulse_schedule/*/reference_waveform/data
```

Affected IDS: `pulse_schedule` (all controlled-quantity subtrees).

Concrete paths (sample):

- `pulse_schedule/ec/beam/power_launched/reference/data` (unit: 1 — should be W)
- `pulse_schedule/ec/beam/frequency/reference/data` (unit: 1 — should be Hz)
- `pulse_schedule/ec/beam/steering_angle_pol/reference/data` (unit: 1 — should be rad)
- `pulse_schedule/ic/antenna/power/reference/data` (unit: 1 — should be W)
- `pulse_schedule/ic/antenna/frequency/reference/data` (unit: 1 — should be Hz)
- `pulse_schedule/lh/antenna/power/reference/data` (unit: 1 — should be W)
- `pulse_schedule/nbi/unit/power/reference/data` (unit: 1 — should be W)
- `pulse_schedule/position_control/*/reference/data` (unit: 1 — should be m)
- `pulse_schedule/flux_control/loop_voltage/reference/data` (unit: 1 — should be V)

## Evidence

Rc13 standard names that quarantined because of this issue:

- `electron_cyclotron_beam_frequency_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — name contains 'frequency' but unit is dimensionless; expected Hz
- `electron_cyclotron_beam_launched_power_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — name contains 'power' but unit is dimensionless; expected W
- `electron_cyclotron_beam_poloidal_steering_angle_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — name contains 'angle' but unit is dimensionless; expected rad
- `electron_cyclotron_beam_toroidal_steering_angle_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — same
- `electron_cyclotron_launched_power_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — same
- `ion_cyclotron_heating_antenna_reference_frequency` (audit: name_unit_consistency_check; unit: 1) — same
- `ion_cyclotron_heating_antenna_reference_power` (audit: name_unit_consistency_check; unit: 1) — same
- `ion_cyclotron_heating_reference_power` (audit: name_unit_consistency_check; unit: 1) — same
- `loop_voltage_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — name contains 'voltage' but unit is dimensionless; expected V
- `lower_hybrid_antenna_reference_power` (audit: name_unit_consistency_check; unit: 1) — same
- `neutral_beam_injection_power_reference` (audit: name_unit_consistency_check; unit: 1) — same
- `neutral_beam_injection_unit_energy_reference_waveform` (audit: name_unit_consistency_check; unit: 1) — name contains 'energy' but unit is dimensionless; expected eV or J
- `line_integrated_electron_density_reference_waveform` (audit: cumulative_prefix_check; unit: 1) — also caught by cumulative prefix check
- `line_integrated_electron_density_inside_flux_surface_reference_waveform` (audit: cumulative_prefix_check; unit: 1) — same
- `poloidal_field_coil_current_reference_waveform` (audit: implicit_field_check; unit: 1) — also caught by implicit field check

Total: **16 quarantined names** (the largest single-issue cluster in rc13),
plus an additional **~5 valid names** that escaped because they happen to be
dimensionless quantities where `unit=1` is accidentally correct (e.g.,
`effective_charge_reference_waveform`).

This issue alone accounts for **13% of all quarantines** in the rc13 corpus.

## Proposed fix

Option A (preferred): **Publish the controlled quantity's unit.** Each
`reference/data` node should carry the same unit as the corresponding
physics quantity. For example, `pulse_schedule/ec/beam/power_launched/reference/data`
should have `unit=W`, mirroring the `power_launched` physics path.

Option B: **Add explicit documentation.** Tag each `reference` subtree with
`description: "controller-reference waveform; dimensionless normalized"`
and add `lifecycle=alpha`. This allows codex to filter them at the classifier
level, but does not fix the root cause.

Option C: **Add a `reference_unit` attribute** that points to the controlled
quantity's unit path, allowing tools to resolve the correct unit indirectly.

## Downstream impact

- **imas-codex SN extractor:** unblocks 16 quarantined names — the single
  largest quarantine cluster. With correct units, these names pass the
  `name_unit_consistency_check` directly. Cuts `plasma_control` domain
  quarantine from ~51% to ≤10%.
- **pydantic audits:** `name_unit_consistency_check` is correct and should
  remain — the bug is in the DD, not in the audit. No audit bypass needed
  after DD fix.
- **imas-standard-names:** no vocab change needed.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §0 (blocker B3)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 5)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-03)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
