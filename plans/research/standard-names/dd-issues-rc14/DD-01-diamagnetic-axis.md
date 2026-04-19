# DD-01 — `/diamagnetic` used as spatial axis under vector containers

## DD path(s) / pattern

```
*/ggd/*/diamagnetic          (under a_field, b_field, e_field, j_tot)
*/profiles_1d/*/velocity/diamagnetic
*/profiles_1d/e_field/diamagnetic
*/profiles_2d/*/velocity/diamagnetic
```

Affected IDSs: `edge_profiles`, `core_profiles`, `core_instant_changes`,
`plasma_profiles`.

Concrete paths (sample):

- `edge_profiles/ggd/a_field/diamagnetic` (unit: T.m)
- `edge_profiles/ggd/b_field/diamagnetic` (unit: T)
- `edge_profiles/ggd/e_field/diamagnetic` (unit: V.m^-1)
- `core_profiles/profiles_1d/electrons/velocity/diamagnetic` (unit: m.s^-1)
- `core_profiles/profiles_1d/ion/velocity/diamagnetic` (unit: m.s^-1)
- `core_profiles/profiles_1d/e_field/diamagnetic` (unit: V.m^-1)

## Evidence

Rc13 standard names that quarantined because of this issue:

- `diamagnetic_component_of_anomalous_current_density` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: A.m^-2) — 'diamagnetic' is not a spatial projection axis; it labels a drift (v_dia = B × ∇p / (qnB²))
- `diamagnetic_component_of_electric_field` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: V.m^-1) — same: not a coordinate axis
- `diamagnetic_component_of_ion_exb_drift_velocity` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: m.s^-1) — same
- `diamagnetic_component_of_ion_velocity_per_magnetic_field_magnitude` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: m.s^-1) — same
- `diamagnetic_component_of_magnetic_field` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: T) — same
- `diamagnetic_component_of_magnetic_vector_potential` (audit: diamagnetic_component_check, pydantic:scalar.name; unit: m.T) — same

Total: **6 quarantined names**, all failing both `pydantic:scalar.name` (missing
from Component vocabulary) and `diamagnetic_component_check`.

## Proposed fix

Rename `/diamagnetic` leaf nodes in the DD:

| Container | Current leaf | Proposed rename |
|-----------|-------------|-----------------|
| `*/velocity/` | `diamagnetic` | `diamagnetic_drift` |
| `*/e_field/` | `diamagnetic` | `diamagnetic_drift` |
| `*/a_field/` | `diamagnetic` | `diamagnetic_drift` |
| `*/b_field/` | `diamagnetic` | `diamagnetic_drift` |
| `*/j_tot/` | `diamagnetic` | `diamagnetic_drift` |

The `/radial`, `/poloidal`, `/toroidal` siblings are true coordinate axes.
`/diamagnetic` is a drift mechanism — physically a vector quantity, not a spatial
projection. Renaming to `diamagnetic_drift` clarifies that this is a drift-type
contribution, consistent with the DD's existing `e_x_b` (ExB drift) siblings.

Alternatively, if the DD prefers to keep the path name: add explicit documentation
tagging these leaves as "drift-type contribution projected onto the local radial
direction" rather than a coordinate axis.

## Downstream impact

- **imas-codex SN extractor:** unblocks 6 quarantined paths; the LLM will
  generate `diamagnetic_drift_velocity` (or similar) instead of the invalid
  `diamagnetic_component_of_*` pattern.
- **pydantic audits:** `diamagnetic_component_check` audit can be removed entirely
  — no more false-positive `_component_of_` construction.
- **imas-standard-names:** no vocab change needed; `diamagnetic_drift` already
  exists as a Process token. If DD renames, codex produces valid names without
  grammar additions.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 1)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-01)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
