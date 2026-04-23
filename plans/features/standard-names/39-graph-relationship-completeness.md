# Plan 39 — Standard Name Graph Relationship Completeness

## Goal

Every `StandardName` node written to the graph — whether by the generation
pipeline or by catalog import — carries its full set of semantic
relationships.

## Context

- Graph is actively cleared and regenerated. No migration, no backwards
  compatibility guards, no one-shot CLI repair tools.
- Grammar v0.7 uses suffix form for complex parts (`X_real_part`) and
  postfix components (`X_toroidal_component`). See plan 38 §A12 for the
  canonical corpus and §A11 for the ISN release gate.
- ISN ships a validated parser + IR. We **consume** it; we do not roll
  our own peeler. Names ISN cannot yet parse simply produce no derived
  edges until plan 38 lands.

## Schema audit (done before proposing new edges)

Enumerated every `relationship_type` across `imas_codex/schemas/*.yaml`.
Four existing edges already cover proposed semantics — reuse them:

| Need                     | Existing edge      | Reuse as                                                  |
| ------------------------ | ------------------ | --------------------------------------------------------- |
| uncertainty siblings     | `HAS_ERROR`        | `StandardName -[:HAS_ERROR {error_type}]-> StandardName`  |
| deprecation lineage prev | `HAS_PREDECESSOR`  | `StandardName -[:HAS_PREDECESSOR]-> StandardName`         |
| deprecation lineage next | `HAS_SUCCESSOR`    | `StandardName -[:HAS_SUCCESSOR]-> StandardName`           |
| cluster membership       | `IN_CLUSTER`       | `StandardName -[:IN_CLUSTER]-> IMASSemanticCluster`       |

Two net-new edges are required:

| Edge                  | Direction                                    | Purpose                                                                 |
| --------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| `HAS_ARGUMENT`        | derived `StandardName` → parent `StandardName` | Single generalised edge for every derivation layer (prefix / postfix / projection / binary). Replaces all proposed hand-rolled component / real-part / imaginary-part / moment edges. |
| `HAS_PHYSICS_DOMAIN`  | `StandardName` / `FacilitySignal` → `PhysicsDomain` | Promote physics domain from a dangling enum value to a first-class vocabulary node. Mirrors `HAS_COCOS` / `HAS_UNIT`. |

Rejected (duplicates of existing edges):

- `UNCERTAINTY_OF` — `HAS_ERROR` already carries `error_type ∈ {upper, lower, index}`.
- `DEPRECATES` / `SUPERSEDED_BY` — `HAS_PREDECESSOR` / `HAS_SUCCESSOR` already exist for version-chain and signal-epoch lineage with identical semantics.

## `HAS_ARGUMENT` — single generalised edge, driven by ISN IR

One edge type. One rule. Per `StandardName N`, run
`parser.parse(N.id).ir` and look at the **outermost** operator only:

| IR shape                                                | Emit                                                                            |
| ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ops[0].kind == unary_prefix` and op ∈ {`upper_uncertainty`, `lower_uncertainty`, `uncertainty_index`} | `HAS_ERROR`: inner_name → N, with `error_type = upper \| lower \| index`. No HAS_ARGUMENT. |
| `ops[0].kind == unary_prefix` (any other op)            | `HAS_ARGUMENT`: N → inner_name, `operator=op.op`, `operator_kind="unary_prefix"`. |
| `ops[0].kind == unary_postfix`                          | `HAS_ARGUMENT`: N → inner_name, `operator=op.op`, `operator_kind="unary_postfix"`. |
| `ops[0].kind == binary`                                 | Two `HAS_ARGUMENT` edges: N → compose(args[0]) `{role:"a"}`, N → compose(args[1]) `{role:"b"}`, `operator=op.op`, `operator_kind="binary"`, `separator=op.separator`. |
| `projection` set (and no operator)                      | `HAS_ARGUMENT`: N → inner_name, `operator="component"`, `operator_kind="projection"`, `axis=projection.axis`. |
| none of the above                                       | no edge (leaf name). |

`inner_name` is produced by `parser.compose(stripped_ir)` where
`stripped_ir` is `N`'s IR with the outermost operator/projection
removed. Recursion happens naturally: when the inner `StandardName` is
itself written to the graph, its own derivation runs and adds its own
HAS_ARGUMENT / HAS_ERROR edge. We never peel more than one layer at a
time — each node owns its single outermost fact.

### Algorithm

```python
# imas_codex/standard_names/derivation.py  (pure logic, no graph access)

from dataclasses import dataclass
from imas_standard_names.grammar import parser, ir as isn_ir

@dataclass(frozen=True)
class DerivedEdge:
    edge_type: str                 # "HAS_ARGUMENT" or "HAS_ERROR"
    from_name: str                 # source StandardName id
    to_name: str                   # target StandardName id
    props: dict                    # edge properties

def derive_edges(name: str) -> list[DerivedEdge]:
    """Return derived edges for a single StandardName id.

    Pure function. ISN parser is the sole source of structural truth.
    """
    try:
        result = parser.parse(name)
    except Exception:
        return []                  # Unparseable → leaf (no edges)
    ir = result.ir
    # Outermost operator
    if ir.operators:
        op = ir.operators[0]
        if op.kind == isn_ir.OperatorKind.BINARY:
            a = parser.compose(op.args[0])
            b = parser.compose(op.args[1])
            return [
                DerivedEdge("HAS_ARGUMENT", name, a, {
                    "operator": op.op, "operator_kind": "binary",
                    "role": "a", "separator": op.separator,
                }),
                DerivedEdge("HAS_ARGUMENT", name, b, {
                    "operator": op.op, "operator_kind": "binary",
                    "role": "b", "separator": op.separator,
                }),
            ]
        stripped = _strip_outer(ir)
        inner = parser.compose(stripped)
        if op.kind == isn_ir.OperatorKind.UNARY_PREFIX and op.op in _UNCERTAINTY_OPS:
            # HAS_ERROR inverts direction: parent -> sibling
            return [DerivedEdge("HAS_ERROR", inner, name, {
                "error_type": _UNCERTAINTY_OPS[op.op],
            })]
        return [DerivedEdge("HAS_ARGUMENT", name, inner, {
            "operator": op.op, "operator_kind": op.kind.value,
        })]
    # Projection (component) without operator
    if ir.projection is not None:
        stripped = _strip_projection(ir)
        inner = parser.compose(stripped)
        return [DerivedEdge("HAS_ARGUMENT", name, inner, {
            "operator": "component", "operator_kind": "projection",
            "axis": ir.projection.axis,
        })]
    return []

_UNCERTAINTY_OPS = {
    "upper_uncertainty": "upper",
    "lower_uncertainty": "lower",
    "uncertainty_index": "index",
}
```

Helpers `_strip_outer` and `_strip_projection` build a new
`StandardNameIR` with `operators[1:]` or `projection=None` and all
other fields unchanged.

### Edge property summary

| Prop            | Meaning                                                    | Present for                              |
| --------------- | ---------------------------------------------------------- | ---------------------------------------- |
| `operator`      | ISN operator token (`maximum`, `magnitude`, `ratio`, …)    | always (HAS_ARGUMENT)                    |
| `operator_kind` | `unary_prefix` \| `unary_postfix` \| `binary` \| `projection` | always (HAS_ARGUMENT)                  |
| `role`          | `a` \| `b`                                                  | binary only                              |
| `separator`     | ISN binary separator (`to`, `and`)                         | binary only                              |
| `axis`          | `projection.axis` (`parallel`, `toroidal`, …)              | projection only                          |
| `error_type`    | `upper` \| `lower` \| `index`                              | HAS_ERROR only                           |

## Forward reference within a batch

`inner_name` may not yet exist in the graph when its parent is first
persisted. Follow the same pattern as `HAS_PREDECESSOR`/`HAS_SUCCESSOR`:

1. Primary pass writes all `StandardName` nodes in the batch.
2. Tail pass runs `derive_edges()` for every name and `MERGE`s the
   edges. The target node is `MERGE`d by id as a bare `StandardName`
   placeholder so the edge can be created before the target's own
   payload lands — its full properties arrive later in the same batch,
   in a subsequent batch, or from catalog import.

This mirrors the existing pattern for predecessor/successor edges;
nothing new is invented here.

## `HAS_PHYSICS_DOMAIN` — promote enum to vocabulary singleton

Today `PhysicsDomain` is an enum; `physics_domain` is a scalar string on
`StandardName` and `IMASNode`. Promote to a singleton vocabulary class
mirroring `COCOS`:

- New class `PhysicsDomain` in `common.yaml` with `id` (slug),
  `label`, `description`.
- New relationship slots `physics_domain_ref` on `StandardName` (and
  `FacilitySignal`, for parity with `HAS_COCOS`) that emit
  `HAS_PHYSICS_DOMAIN`.
- Scalar `physics_domain` stays on the node (it is DD-authoritative and
  already populated). The edge is derived from it at write time.
- Seed all enum values as `PhysicsDomain` nodes at graph init.

Rationale: enables graph queries like "all standard names in the
`equilibrium` domain" without string-matching, and aligns with the
established `HAS_COCOS` / `HAS_UNIT` pattern.

## Catalog-import parity

`catalog_import.py::_write_import_entries` currently writes `HAS_UNIT`
and grammar scalar properties. After this plan it must match the
pipeline write path exactly:

- `HAS_UNIT` (already there)
- `HAS_COCOS` (missing today)
- `HAS_SEGMENT` (missing today)
- `HAS_ARGUMENT` (new)
- `HAS_ERROR` (new — for uncertainty siblings)
- `HAS_PREDECESSOR` / `HAS_SUCCESSOR` (populated from catalog
  `deprecates` / `superseded_by` YAML fields)
- `IN_CLUSTER` (from `primary_cluster_id` scalar already on the node)
- `HAS_PHYSICS_DOMAIN`

Extract a shared helper `_write_standard_name_edges(tx, entries)` in
`graph_ops.py` so the pipeline writer and the catalog importer call
identical code. This replaces both the existing `HAS_UNIT` blocks and
the per-call COCOS writes.

## Unit tests

A12 rows from plan 38 are the canonical fixtures. Two test modules:

### `tests/standard_names/test_derivation.py` (pure logic, no graph)

| #   | Input                                                                              | Expected edges                                                                     |
| --- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| D1  | `temperature` (A12 row 1 class — leaf)                                             | none                                                                               |
| D2  | `maximum_of_temperature`                                                           | HAS_ARGUMENT → `temperature` `{op:maximum, kind:unary_prefix}`                    |
| D3  | `time_derivative_of_temperature`                                                   | HAS_ARGUMENT → `temperature` `{op:time_derivative, kind:unary_prefix}`           |
| D4  | `time_average_of_maximum_of_temperature` (A12 row 14 class)                        | HAS_ARGUMENT → `maximum_of_temperature` `{op:time_average, kind:unary_prefix}`   |
| D5  | `temperature_magnitude`                                                            | HAS_ARGUMENT → `temperature` `{op:magnitude, kind:unary_postfix}`                |
| D6  | `temperature_moment`                                                               | HAS_ARGUMENT → `temperature` `{op:moment, kind:unary_postfix}`                   |
| D7  | `temperature_reference_waveform` (A12 rows 18, 19)                                 | HAS_ARGUMENT → `temperature` `{op:reference_waveform, kind:unary_postfix}`       |
| D8  | `temperature_bessel_0` (A12 row 27)                                                | HAS_ARGUMENT → `temperature` `{op:bessel_0, kind:unary_postfix}`                 |
| D9  | `ratio_of_temperature_to_pressure`                                                 | two HAS_ARGUMENT: → `temperature` `{role:a}` and → `pressure` `{role:b}`, `{op:ratio, kind:binary, separator:to}` |
| D10 | `upper_uncertainty_of_temperature`                                                 | HAS_ERROR from `temperature` → `upper_uncertainty_of_temperature` `{error_type:upper}`, no HAS_ARGUMENT |
| D11 | `lower_uncertainty_of_temperature`                                                 | HAS_ERROR `{error_type:lower}`                                                     |
| D12 | `uncertainty_index_of_temperature`                                                 | HAS_ERROR `{error_type:index}`                                                     |
| D13 | `maximum_of_temperature_at_plasma_boundary` (A12 row 22)                           | HAS_ARGUMENT → `temperature_at_plasma_boundary` `{op:maximum, kind:unary_prefix}` — locus preserved |
| D14 | `elongation_of_plasma_boundary` (A12 row 1)                                        | none (leaf with locus only)                                                        |
| D15 | garbage string `not_a_name`                                                        | none (parser raises, caught, no edges)                                             |
| D16 | ISN parser returns projection (stub via monkeypatch until plan 38 lands postfix component) | HAS_ARGUMENT `{op:component, kind:projection, axis:parallel}` — readiness test |

D16 uses monkeypatching to simulate the IR shape plan 38 will produce
for `magnetic_field_toroidal_component`, so that when ISN rc21+ lands
the grammar no codex code change is needed — only the test shifts from
monkeypatch to direct parse.

### `tests/standard_names/test_graph_edge_writers.py` (graph integration)

Uses the existing SN test graph fixture.

| #   | Scenario                                                                                  | Assert                                                                                    |
| --- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| G1  | Write `temperature` and `maximum_of_temperature` in one batch.                            | `(maximum_of_temperature)-[:HAS_ARGUMENT {operator:'maximum'}]->(temperature)` exists.    |
| G2  | Write `maximum_of_temperature` first, `temperature` in a later batch.                     | Edge exists after second batch; target merged as placeholder then enriched.               |
| G3  | Write `upper_uncertainty_of_temperature` alone.                                           | `(temperature)-[:HAS_ERROR {error_type:'upper'}]->(upper_uncertainty_of_temperature)`.   |
| G4  | Write `ratio_of_temperature_to_pressure`.                                                 | Two HAS_ARGUMENT edges with `role` ∈ {a, b}.                                              |
| G5  | Write same batch twice.                                                                   | Edge counts unchanged — MERGE idempotent.                                                  |
| G6  | Catalog import of a YAML file containing the same names as G1.                            | Same edges as G1 — import path has pipeline parity.                                        |
| G7  | Write `StandardName` with `deprecates` YAML field.                                        | `HAS_PREDECESSOR` edge written.                                                            |
| G8  | Write `StandardName` with `superseded_by` YAML field.                                     | `HAS_SUCCESSOR` edge written.                                                              |
| G9  | Write `StandardName` with `primary_cluster_id` set.                                       | `IN_CLUSTER` edge to the cluster node.                                                     |
| G10 | Write `StandardName` with `physics_domain='equilibrium'`.                                 | `HAS_PHYSICS_DOMAIN` edge to the singleton `PhysicsDomain {id:'equilibrium'}`.            |

## Implementation order

1. Schemas: promote `PhysicsDomain` enum → singleton class; add
   relationship slots on `StandardName` for `arguments`, `error_siblings`,
   `predecessor`, `successor`, `primary_cluster_ref`,
   `physics_domain_ref`. Add `physics_domain_ref` on `FacilitySignal`
   too (parity with `HAS_COCOS`).
2. `uv run build-models --force`.
3. `imas_codex/standard_names/derivation.py` — pure logic, ISN
   parser + compose only. No I/O.
4. `tests/standard_names/test_derivation.py` — 16 cases. **Red first.**
5. Run derivation tests → green.
6. `_write_standard_name_edges` helper in `graph_ops.py`. Wire into
   `write_standard_names` and `persist_composed_batch` (forward-ref
   tail pass).
7. Catalog parity: call the same helper from
   `catalog_import.py::_write_import_entries`.
8. PhysicsDomain singleton seeding: idempotent `MERGE` for all enum
   values at graph init (same path that seeds COCOS conventions).
9. `tests/standard_names/test_graph_edge_writers.py` — 10 cases.
10. Run full SN test suite + `tests/graph/test_schema_compliance.py`
    + `test_referential_integrity.py`.
11. Docs: update `AGENTS.md` edge table and `plans/README.md`. Delete
    this plan file (code is the documentation per project policy) or
    keep a short one-liner pointing at the code.
12. Single atomic commit with conventional message.

## Out of scope

- Migrating existing graph data (graph is cleared before next cycle).
- Any CLI command for one-shot repair (e.g. `sn graph
  reconcile-structural`) — user-rejected.
- Rolling a codex-local operator-registry peeler — user-rejected.
- Grammar v0.7 postfix component / complex-part parsing — lives in
  plan 38 / ISN rc21+. This plan is forward-compatible: when those
  names parse, they will produce edges automatically with no codex
  change.

## Risks

- ISN parser exceptions on malformed names must be swallowed
  (derive no edges, emit a single debug log line per failure). Covered
  by D15.
- Binary separator `from` (used by `difference`) currently raises a
  Pydantic literal error in rc24. Not a blocker — the derivation
  function catches and returns no edges. Tracked via plan 38.
- ISN rc24 does not yet parse several A12 vNext canonical forms
  (postfix component, `_real_part`). Derivation silently emits no
  edges for those names. Acceptable — they land automatically when
  plan 38 ships.
