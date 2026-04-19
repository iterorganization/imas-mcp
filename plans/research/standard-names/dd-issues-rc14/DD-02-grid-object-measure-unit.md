# DD-02 — `grid_object_measure` carries placeholder unit `m^dimension`

## DD path(s) / pattern

```
*/space/objects_per_dimension/object/measure
```

Affected IDSs (every IDS with a GGD grid): `equilibrium`, `edge_profiles`,
`edge_sources`, `edge_transport`, `em_coupling`, `ferritic`, `mhd`, `tf`,
`transport_solver_numerics`, `waves`, `distribution_sources`, `distributions`.

Concrete paths (sample):

- `equilibrium/time_slice/ggd/grid/space/objects_per_dimension/object/measure` (unit: m^dimension)
- `edge_profiles/grid_ggd/space/objects_per_dimension/object/measure` (unit: m^dimension)
- `mhd/grid_ggd/space/objects_per_dimension/object/measure` (unit: m^dimension)
- `tf/field_map/grid/space/objects_per_dimension/object/measure` (unit: m^dimension)
- `waves/coherent_wave/full_wave/grid/space/objects_per_dimension/object/measure` (unit: m^dimension)
- `ferritic/grid_ggd/space/objects_per_dimension/object/measure` (unit: m^dimension)

## Evidence

Rc13 standard names that quarantined because of this issue:

- `grid_object_measure` (audit: pydantic:scalar.unit, unit_validity_check; unit: m^dimension) — unit string `m^dimension` is not a valid UDUNITS expression; the `^dimension` exponent is a compile-time placeholder that was never resolved

Total: **1 quarantined name**, failing `pydantic:scalar.unit` ("Invalid unit token
'm^dimension'") and `unit_validity_check` ("unit 'm^dimension' contains non-unit
token 'dimension'").

The issue affects **12+ DD paths** across all GGD-equipped IDSs, but the SN
extractor correctly deduplicates them into one standard name.

## Proposed fix

Option A (preferred): **Split into dimension-specific leaves.** Replace the single
`measure` leaf with four concrete leaves:

| Leaf | Unit | Semantics |
|------|------|-----------|
| `measure_count` or `number_of_objects` | `1` | Count of grid objects (0D) |
| `measure_length` | `m` | Total length (1D objects) |
| `measure_area` | `m^2` | Total area (2D objects) |
| `measure_volume` | `m^3` | Total volume (3D objects) |

Each `objects_per_dimension` array element already carries a `dimension` integer
(0–3), so the correct leaf is unambiguous at runtime.

Option B: **Parametric unit via `units_template`.** Introduce a `units_template`
attribute that resolves `m^{dimension}` at DD compile/validation time using the
parent `dimension` value. This preserves the single-leaf structure but requires DD
tooling support for template units.

Option C (minimal): **Document the unit as `mixed`** and add a `description` tag
explaining the dimension dependency. This unblocks the SN extractor (which can
skip `mixed`-unit paths) but does not fix the underlying ambiguity.

## Downstream impact

- **imas-codex SN extractor:** unblocks 1 quarantined name; if split into
  per-dimension leaves, would produce 3–4 specific names (`grid_object_length`,
  `grid_object_area`, `grid_object_volume`) with correct units.
- **pydantic audits:** `unit_validity_check` exception for `m^dimension` can be
  removed — the unit will be a real UDUNITS string.
- **imas-standard-names:** no vocab change needed.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 2)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-02)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
