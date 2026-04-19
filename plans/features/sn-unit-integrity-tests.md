# SN Unit Integrity Tests + Backfill

## Problem

A one-shot audit over the graph (335 valid StandardNames) revealed 4 SN nodes whose
declared `unit` disagrees with every DD path they source from:

| Name | SN unit | DD unit(s) | Root cause |
|---|---|---|---|
| `ion_average_charge_of_ion_state` | `1` | `e` | DD-side bug (elementary charge ≠ dimensionless) |
| `ion_average_square_charge_of_ion_state` | `1` | `e` | same DD-side bug |
| `electron_temperature_peaking_factor` | `1` | `eV` | DD-side bug |
| `plasma_current` | `1` | `A` | SN-side bug; poisoned before session-10 narrowing fix |

Unit flow is now correct in code (DD HAS_UNIT → compose_worker overrides LLM → HAS_UNIT
edge in graph), but there is no **test** asserting the invariant, and no **backfill** for
these 4 pre-existing nodes.

Additionally, 173 SN source_paths point to DD nodes without a `HAS_UNIT` edge — a
DD-schema cleanliness gap (dimensionless quantities lacking explicit `1` unit).

## Scope

1. Add `tests/graph/test_sn_unit_integrity.py` (schema-driven, parametrised from graph).
2. One-time backfill CLI subcommand or one-off migration Cypher to fix the 4 mismatches.
3. Report DD-side unit gaps as a JSON artefact under `plans/research/dd/` for upstream
   triage; these are NOT fixed by this plan (DD rebuild territory).

## Non-goals

- Fixing DD `ion_atomic_number [e]`, `ion_vibrational_level [e]`, or
  `neutral_emitted_kinetic_energy_flux [m^-2.s^-1]` — these are DD-rebuild scope.
- Re-running rotation (that is `sn-coverage-closure` territory).

## Implementation phases

### Phase 1: Schema-driven test

File: `tests/graph/test_sn_unit_integrity.py`

```python
def test_sn_unit_matches_linked_dd_path_unit(graph_client):
    """Every StandardName.unit must equal (or be contained in) the
    units of the DD paths in its source_paths list, modulo dimensionless gaps.
    """
    rows = graph_client.query("""
        MATCH (sn:StandardName {validation_status:'valid'})
        WHERE sn.unit IS NOT NULL
        UNWIND sn.source_paths AS sp_raw
        WITH sn, replace(sp_raw, 'dd:', '') AS sp
        OPTIONAL MATCH (:IMASNode {id: sp})-[:HAS_UNIT]->(u:Unit)
        WITH sn.id AS name, sn.unit AS sn_unit,
             collect(DISTINCT u.id) AS raw
        WITH name, sn_unit, [x IN raw WHERE x IS NOT NULL] AS dd_units
        WHERE size(dd_units) > 0 AND NOT sn_unit IN dd_units
        RETURN name, sn_unit, dd_units
    """)
    assert rows == [], f"SN units disagree with DD: {rows}"
```

A second test asserts `sn.unit IS NOT NULL` for all valid SNs (every SN must have a
unit declared, even if `1`).

A third test, marked `xfail` for now, tallies `no_unit_edge` DD-side gaps and writes
them to `plans/research/dd/sn_unit_gaps.json` for DD-upstream triage. Not a build
failure.

### Phase 2: Backfill

One-off migration Cypher via `imas-codex graph shell`:

```cypher
// plasma_current — SN wrong, DD correct
MATCH (sn:StandardName {id: 'plasma_current'})
OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit)
DELETE r
WITH sn
MERGE (u:Unit {id: 'A'})
MERGE (sn)-[:HAS_UNIT]->(u)
SET sn.unit = 'A';

// electron_temperature_peaking_factor — SN correct (dimensionless); DD tagged eV incorrectly
// Leave SN as '1'; log DD gap.

// ion_average_(square_)charge_of_ion_state — DD wrong (`e`), SN '1' is physically correct
// Leave SN as '1'; log DD gap.
```

Resulting invariant: the new test passes. 3 of the 4 were actually DD-side bugs and the
test is now LHS-of-the-equality with the DD being the violator. Document them in
`plans/research/dd/sn_unit_gaps.json` and update the test to use a small per-name
allow-list of known DD-side issues (each with a DD-rebuild follow-up note).

### Phase 3: DD gap report

Emit `plans/research/dd/sn_unit_gaps.json` (not committed as code, but checked-in as
research artefact):

```json
{
  "dd_side_unit_bugs": [
    {"dd_path": "...", "sn_that_surfaces_it": "...", "dd_unit": "e",  "proposed_unit": "1"},
    {"dd_path": "...", "sn_that_surfaces_it": "...", "dd_unit": "eV", "proposed_unit": "1"}
  ],
  "missing_has_unit_edges": [
    {"dd_path": "...", "expected_unit": "1"}
  ]
}
```

Agents working on `dd-rebuild.md` read this file to close the loop.

## Success criteria

- `tests/graph/test_sn_unit_integrity.py` passes in CI.
- The 4 mismatches are zero after backfill (or documented in a tight allow-list with
  DD-rebuild follow-up tickets).
- `plans/research/dd/sn_unit_gaps.json` exists with the DD-upstream gap list.

## Dispatch

Single engineer agent (opus-4.6). Tight scope, one test file, one migration, one
research artefact.
