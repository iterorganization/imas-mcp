# DD-05 — Duplicate `neutron_fluxes` / `neutron_rates` subtrees in `summary/fusion`

## DD path(s) / pattern

```
summary/fusion/neutron_fluxes/dd/{thermal,beam_thermal,beam_beam,total}/value
summary/fusion/neutron_rates/dd/{thermal,beam_thermal,beam_beam,total}/value
summary/fusion/neutron_fluxes/dt/{thermal,beam_thermal,beam_beam,total}/value
summary/fusion/neutron_rates/dt/{thermal,beam_thermal,beam_beam,total}/value
summary/fusion/neutron_fluxes/tt/{thermal,beam_thermal,beam_beam,total}/value
summary/fusion/neutron_rates/tt/{thermal,beam_thermal,beam_beam,total}/value
```

Affected IDS: `summary`.

Both subtrees have **identical structure** and **identical units** (`Hz`).

## Evidence

The SN extractor generates names from both subtrees, producing standard names
that differ only in whether the source DD path says `neutron_fluxes` or
`neutron_rates`. The rc13 corpus shows naming inconsistency:

- `deuterium_deuterium_neutron_flux_due_to_thermal_fusion` (unit: Hz; vs: quarantined) — sourced from `neutron_fluxes/dd/thermal`
- `deuterium_tritium_neutron_flux_due_to_thermal_fusion` (unit: Hz; vs: quarantined) — from `neutron_fluxes/dt/thermal`
- `deuterium_tritium_total_neutron_flux` (unit: Hz; vs: quarantined) — from `neutron_fluxes/dt/total`
- `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` (unit: Hz; vs: quarantined) — from `neutron_fluxes/dt/beam_beam`
- `deuterium_tritium_neutron_flux_due_to_beam_thermal_fusion` (unit: Hz; vs: quarantined) — from `neutron_fluxes/dt/beam_thermal`
- `tritium_tritium_neutron_flux_due_to_thermal_fusion` (unit: Hz; vs: quarantined) — from `neutron_fluxes/tt/thermal`
- `tritium_tritium_beam_thermal_neutron_rate` (unit: Hz; vs: valid) — from `neutron_rates/tt/beam_thermal`
- `tritium_tritium_thermal_neutron_rate` (unit: Hz; vs: valid) — from `neutron_rates/tt/thermal`
- `tritium_tritium_total_neutron_rate` (unit: Hz; vs: valid) — from `neutron_rates/tt/total`

The quarantines on `neutron_flux` names are from other audit checks (e.g.,
`thermalization`-related grammar gaps), but the **duplication itself** creates
naming ambiguity: are DD and DT reaction products "fluxes" or "rates"? Both terms
map to unit Hz (count per second). The SN vocabulary must pick one canonical term.

Total: **~6 quarantined** and **~12 valid** names across the two subtrees; the
duplication doubles the SN count for no additional physics content.

## Proposed fix

Option A (preferred): **Deprecate `neutron_rates` → mark as `obsolescent`** with
a `RENAMED_TO` record pointing to `neutron_fluxes`. In nuclear/fusion physics,
"neutron flux" (or "neutron source rate") is the standard term for total neutron
production rate in Hz. "Rate" without qualification is ambiguous (rate of what?).

Option B: **Deprecate `neutron_fluxes`** instead, if the DD committee prefers
"rate" as the canonical term. Note that `neutron_fluxes` has broader adoption in
existing tools and documentation.

Option C: **Distinguish semantics.** If the two subtrees are meant to carry
different quantities (e.g., `neutron_fluxes` = surface-integrated flux [n/m²/s],
`neutron_rates` = total volumetric rate [n/s]), then change the units to reflect
the distinction and add documentation. Currently both carry `unit=Hz`, making
them indistinguishable.

## Downstream impact

- **imas-codex SN extractor:** halves the number of neutron-related names from
  ~18 to ~9 after deduplication. Eliminates the "flux vs rate" naming ambiguity
  that forces the LLM to choose inconsistently.
- **pydantic audits:** no audit changes needed — the quarantines on these names
  are from unrelated checks (grammar gaps, `due_to_` vocabulary).
- **imas-standard-names:** can standardize on either `neutron_flux` or
  `neutron_rate` as the canonical Subject/Object token; currently both exist
  in the vocabulary.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 6)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-04)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
