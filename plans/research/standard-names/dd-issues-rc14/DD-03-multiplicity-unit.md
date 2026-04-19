# DD-03 — `atom_multiplicity_of_isotope` declared with invalid unit `Elementary Charge Unit`

## DD path(s) / pattern

```
*/element/multiplicity
```

Affected IDSs: `core_profiles`, `edge_profiles`, `waves`, `wall`, `pellets`,
`turbulence`, and all IDSs containing `ion/element/multiplicity` or
`neutral/element/multiplicity`.

Concrete paths (sample):

- `core_profiles/profiles_1d/ion/element/multiplicity` (unit: Elementary Charge Unit)
- `edge_profiles/ggd/ion/element/multiplicity` (unit: Elementary Charge Unit)
- `waves/coherent_wave/global_quantities/ion/element/multiplicity` (unit: Elementary Charge Unit)
- `waves/coherent_wave/beam_tracing/beam/ion/element/multiplicity` (unit: Elementary Charge Unit)
- `wall/description_ggd/ggd/energy_fluxes/recombination/ion/element/multiplicity` (unit: Elementary Charge Unit)
- `pellets/time_slice/pellet/propellant_gas/element/multiplicity` (unit: Elementary Charge Unit)

## Evidence

Rc13 standard names that quarantined because of this issue:

- `atom_multiplicity_of_isotope` (audit: pydantic:scalar.unit; unit: Elementary Charge Unit) — unit string contains whitespace, which is not a valid UDUNITS expression

Total: **1 quarantined name**, failing `pydantic:scalar.unit` ("Unit must not
contain whitespace").

Two issues compound here:

1. **Invalid UDUNITS string:** `Elementary Charge Unit` contains spaces — not
   parseable by any UDUNITS library. The correct UDUNITS symbol for elementary
   charge is `e`.
2. **Semantically wrong unit:** `multiplicity` is the number of atoms of a given
   element in a molecule/compound (e.g., 2 for O in H₂O). It is a dimensionless
   integer count, not a charge. The DD likely confused `multiplicity` with `z_n`
   (nuclear charge number), which correctly carries `unit=e` elsewhere (e.g.,
   `pellets/time_slice/pellet/species/z_n`).

## Proposed fix

Option A (preferred — semantics fix): **Change unit to `1` (dimensionless).**
Multiplicity is an integer count — it has no physical unit. The sibling field
`z_n` already carries `unit=e` for charge-related quantities.

Option B (UDUNITS fix only): **Change unit to `e`** if the DD truly intends this
field to represent a charge-like quantity. However, the field name `multiplicity`
and the DD documentation ("Multiplicity of the atom") strongly suggest
dimensionless.

In either case, the whitespace in the unit string must be fixed.

## Downstream impact

- **imas-codex SN extractor:** unblocks 1 quarantined name; with correct unit the
  name `atom_multiplicity_of_isotope` (unit: 1) would pass validation.
- **pydantic audits:** the `pydantic:scalar.unit` whitespace rejection is generic
  and correct — no audit changes needed.
- **imas-standard-names:** no vocab change needed.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 3)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-02, grouped with grid_object_measure as unit sanitization)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
