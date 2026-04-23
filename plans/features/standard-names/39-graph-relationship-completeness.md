# Plan 39 — Standard-Name Graph Relationship Completeness

> Revised after three-reviewer critique (Sonnet 4.6, Opus 4.6, GPT-5.4).
> Blocking findings incorporated; outstanding concerns explicitly resolved.

## Problem

The `StandardName` node in our Neo4j graph has solid relationships for its
immediate "metadata" context (`HAS_UNIT`, `HAS_COCOS`, `HAS_SEGMENT`,
`REFERENCES`, `REVIEWS`, source-side `FROM_DD_PATH` / `FROM_SIGNAL` /
`PRODUCED_NAME`, `EVIDENCED_BY`, `HAS_STANDARD_NAME_VOCAB_GAP`) but is
missing nearly every natural **name-to-name structural edge**. Audit
evidence in session notes `files/codex-audit.md`:

- Vector parents and their scalar components are only lexically related
  via the `_component_of_` substring in the name id (no edge).
- Complex parents and their real/imaginary parts are detected only by
  `kind_derivation.py` (no edge). The live catalog uses the
  **suffix** form (`<parent>_real_part`, `<parent>_imaginary_part`) —
  there are 6 such names in `gyrokinetics/` today and 0 prefix-form
  names. The original heuristic in the audit assumed prefix-only and
  would have produced zero edges. Both forms must be handled.
- Error/uncertainty siblings are minted via `mint_error_siblings` and
  even re-walked by `reconcile_error_siblings` for orphan detection —
  but the parent link is never persisted as a graph edge.
- `deprecates` and `superseded_by` are stored as **string scalar
  properties**, not `DEPRECATES` / `SUPERSEDED_BY` edges, and not as
  an inverse pair.
- `physics_domain`, `tags`, `species`, `population` are plain string
  properties. `FacilitySignal` already declares a `MENTIONS_DOMAIN ->
  PhysicsDomain` relationship in `facility.yaml`, but a code search
  confirms that edge is never actually written to the graph —
  migration is schema-only.
- `IMASSemanticCluster` membership is used transiently in
  `enrichment.py` for batching but is never persisted as a
  `StandardName → Cluster` edge.
- `catalog_import._write_import_entries` writes grammar segment fields
  as plain `grammar_*` properties but never creates `HAS_SEGMENT`
  edges, and never creates `HAS_COCOS` — so catalog-imported names
  end up structurally inconsistent with pipeline-generated names.

### Known coverage gap — accept explicitly

The catalog has 88 names starting with `{r,z,phi,radial,vertical,
poloidal,toroidal,…}_component_of_<X>` but only 8 of those 88 target
parent names `<X>` exist as standard names in the catalog today
(91% orphan rate). This plan **will not** manufacture missing vector
parents. The `COMPONENT_OF` edge is created only where the parent
exists. Creating the missing parents is a **separate vocabulary
completeness pass** (future plan) and is documented here as an open
gap that plan 40's inline `parent:` emission will inherit.

## Approach

Add every missing relationship in a **single atomic schema + writer
commit** plus a deterministic idempotent reconcile pass. All edges are
derivable from data we already have (name grammar, `grammar_*`
properties, `model='deterministic:dd_error_modifier'`,
`cocos_transformation_type`, `primary_cluster_id`, existing string
properties) — no new LLM calls are required.

The work is split into three tiers:

- **Tier 1 (must-have):** edges that fix known correctness bugs and
  unlock hierarchy export for plan 40.
- **Tier 2 (should-have):** domain-node promotion + `IN_CLUSTER` edge
  that materially improve graph-search organisation. Includes explicit
  backfill for existing nodes.
- **Tier 3 (opportunistic):** species / population node vocabulary —
  schema-declared today but not written anywhere. Deferred until
  there is a concrete consumer.

Plan 40 depends on **Tier 1 only**; Tier 2 is independent and can slip
without blocking plan 40.

## Scope

### 1. Schema additions (`imas_codex/schemas/standard_name.yaml`)

Tier 1 edges: `COMPONENT_OF`, `UNCERTAINTY_OF`, `REAL_PART_OF`,
`IMAGINARY_PART_OF`, `DEPRECATES`, `SUPERSEDED_BY`. Each with
`range: StandardName` and explicit
`annotations.relationship_type` per project convention.

Tier 2 edges: `IN_CLUSTER` (`range: IMASSemanticCluster`) and
`AT_PHYSICS_DOMAIN` (`range: PhysicsDomain`).

Promote `PhysicsDomain` from enum-only to a singleton vocabulary class
in `imas_codex/schemas/common.yaml` with `id` equal to the enum value.
Pre-populate 32 `PhysicsDomain` nodes at graph-bootstrap
(`imas-codex graph bootstrap-domains` — also run automatically by
`imas-codex graph start` when the node count is zero).

Rename `FacilitySignal.MENTIONS_DOMAIN → AT_PHYSICS_DOMAIN` for
project-wide uniformity. Verification: `grep -r MENTIONS_DOMAIN` finds
only the schema docstring and the generated model file — **no code
writes this edge today** — so the rename is schema-only with no data
migration. Confirmed by pre-revision grep; recorded here for
auditability. The plan will re-verify against the live graph with
`MATCH ()-[r:MENTIONS_DOMAIN]->() RETURN count(r)` before the schema
commit.

### 2. Structural post-pass — `imas_codex/standard_names/structural_edges.py` (new)

One idempotent function `write_structural_edges(gc, sn_ids, *,
impact_closure=True) -> StructuralReport`.

**Impact closure.** If `impact_closure=True`, before performing any
writes the function expands `sn_ids` to include every SN whose
structural edges could target or originate from a name in `sn_ids`.
The expansion runs in two passes so that **deprecation retargeting**
(where the *old* counterpart still holds a stale inverse edge)
always gets reprocessed:

1. **Name-pattern closure:** every SN with `_component_of_<x>`,
   `<x>_real_part`, `<x>_imaginary_part`, `real_part_of_<x>`,
   `imaginary_part_of_<x>`, or uncertainty prefix pointing at any
   `x ∈ sn_ids`, and every SN with `deprecates IN sn_ids` or
   `superseded_by IN sn_ids`.
2. **Edge-reachability closure:** `MATCH (s:StandardName)
   -[:DEPRECATES|SUPERSEDED_BY|COMPONENT_OF|REAL_PART_OF|
   IMAGINARY_PART_OF|UNCERTAINTY_OF]-(n) WHERE s.id IN $sn_ids
   RETURN n.id` (note the **undirected** `-[]-` arrow). This captures
   the old counterpart B after A's `deprecates` is retargeted from B
   to C: B has a stale `SUPERSEDED_BY -> A` edge that closure (1)
   cannot find because B's scalar property already points elsewhere.

The closed set is the union of both passes. This guarantees that a
rename or retargeting of a parent reconciles all dependents — old
and new — in the same pass.

**Per-edge Cypher** — delete-then-rewrite, with guards fixed per GPT
review. Each query is wrapped with `@retry_on_deadlock()` and ordered
by `rand()` per project convention:

`COMPONENT_OF` (requires `grammar_component` guard to eliminate
false positives on names that happen to contain `_component_of_` but
are not grammatical components):

```cypher
UNWIND $ids AS id
MATCH (sn:StandardName {id: id})
OPTIONAL MATCH (sn)-[old:COMPONENT_OF]->()
DELETE old
WITH sn, split(sn.id, '_component_of_') AS parts
WHERE size(parts) = 2
  AND sn.grammar_component IS NOT NULL
  AND parts[0] = sn.grammar_component
MATCH (parent:StandardName {id: parts[1]})
WHERE parent.id <> sn.id
MERGE (sn)-[:COMPONENT_OF]->(parent);
```

`UNCERTAINTY_OF`:

```cypher
UNWIND $ids AS id
MATCH (sn:StandardName {id: id})
OPTIONAL MATCH (sn)-[old:UNCERTAINTY_OF]->()
DELETE old
WITH sn,
  CASE
    WHEN sn.model = 'deterministic:dd_error_modifier'
         AND sn.id STARTS WITH 'upper_uncertainty_of_'
      THEN substring(sn.id, size('upper_uncertainty_of_'))
    WHEN sn.model = 'deterministic:dd_error_modifier'
         AND sn.id STARTS WITH 'lower_uncertainty_of_'
      THEN substring(sn.id, size('lower_uncertainty_of_'))
    WHEN sn.model = 'deterministic:dd_error_modifier'
         AND sn.id STARTS WITH 'uncertainty_index_of_'
      THEN substring(sn.id, size('uncertainty_index_of_'))
    ELSE NULL
  END AS parent_id
WHERE parent_id IS NOT NULL
  AND coalesce(sn.pipeline_status, '') <> 'skipped'
MATCH (parent:StandardName {id: parent_id})
WHERE parent.id <> sn.id
  AND coalesce(parent.pipeline_status, '') <> 'skipped'
MERGE (sn)-[:UNCERTAINTY_OF]->(parent);
```

`REAL_PART_OF` / `IMAGINARY_PART_OF` — handle **both** suffix form
(live catalog today) and prefix form (future-grammar form):

```cypher
UNWIND $ids AS id
MATCH (sn:StandardName {id: id}) WHERE sn.kind = 'complex'
OPTIONAL MATCH (sn)-[old:REAL_PART_OF|IMAGINARY_PART_OF]->()
DELETE old
WITH sn,
  CASE
    WHEN sn.id ENDS WITH '_real_part'
      THEN substring(sn.id, 0, size(sn.id) - size('_real_part'))
    WHEN sn.id STARTS WITH 'real_part_of_'
      THEN substring(sn.id, size('real_part_of_'))
    ELSE NULL
  END AS real_parent,
  CASE
    WHEN sn.id ENDS WITH '_imaginary_part'
      THEN substring(sn.id, 0, size(sn.id) - size('_imaginary_part'))
    WHEN sn.id STARTS WITH 'imaginary_part_of_'
      THEN substring(sn.id, size('imaginary_part_of_'))
    ELSE NULL
  END AS imag_parent
FOREACH (rp IN CASE WHEN real_parent IS NOT NULL THEN [real_parent] ELSE [] END |
  MERGE (parent:StandardName {id: rp})
  MERGE (sn)-[:REAL_PART_OF]->(parent)
)
FOREACH (ip IN CASE WHEN imag_parent IS NOT NULL THEN [imag_parent] ELSE [] END |
  MERGE (parent:StandardName {id: ip})
  MERGE (sn)-[:IMAGINARY_PART_OF]->(parent)
);
```

Note: the `MERGE (parent:StandardName {id: rp})` **creates** the
parent node if absent — this is intentional to surface the
vocabulary gap. Stub nodes are tagged with
`pipeline_status = 'stub'` on creation (via
`SET parent.pipeline_status = coalesce(parent.pipeline_status, 'stub')`
inside the FOREACH), which:

- Excludes them from the export query (filters on
  `pipeline_status IN ['published','accepted','reviewed','enriched']`).
- Excludes them from the reviewer/pipeline path (which reads `drafted`).
- Gives the vocabulary-gap report a stable predicate
  (`WHERE sn.pipeline_status = 'stub'`).
- Prevents stubs from polluting Tier-2 backfill (IN_CLUSTER and
  AT_PHYSICS_DOMAIN backfill queries gate on the respective
  non-null properties, both NULL on stubs).

A later `write_standard_names()` call that writes the full entry for
that parent naturally overwrites `pipeline_status` via its normal
`SET sn.pipeline_status = coalesce(b.pipeline_status, sn.pipeline_status)`
semantics, so a stub is a transient artefact that upgrades in place
once the parent is properly generated.

`DEPRECATES` / `SUPERSEDED_BY` — written as inverse pairs; either
side filling the scalar property creates both edges:

The delete step is **undirected** so that both outgoing *and*
incoming deprecation edges incident to `sn` are removed before the
rewrite — otherwise a stale inverse pair from a previous
counterpart would survive the reconcile:

```cypher
UNWIND $ids AS id
MATCH (sn:StandardName {id: id})
OPTIONAL MATCH (sn)-[old:DEPRECATES|SUPERSEDED_BY]-()
DELETE old
WITH sn
OPTIONAL MATCH (dep:StandardName {id: sn.deprecates})
FOREACH (_ IN CASE WHEN dep IS NOT NULL AND dep.id <> sn.id THEN [1] ELSE [] END |
  MERGE (sn)-[:DEPRECATES]->(dep)
  MERGE (dep)-[:SUPERSEDED_BY]->(sn)
)
WITH sn
OPTIONAL MATCH (sup:StandardName {id: sn.superseded_by})
FOREACH (_ IN CASE WHEN sup IS NOT NULL AND sup.id <> sn.id THEN [1] ELSE [] END |
  MERGE (sn)-[:SUPERSEDED_BY]->(sup)
  MERGE (sup)-[:DEPRECATES]->(sn)
);
```

An invariant test asserts that for every `A-[:DEPRECATES]->B` there
is a matching `B-[:SUPERSEDED_BY]->A` in both directions.

**Return value.** `StructuralReport` — dataclass with per-edge
counts (created / deleted / stub-parents-created / errors) and the
list of orphan component parents (for the vocabulary completeness
follow-up).

### 3. Pipeline integration

- `persist_composed_batch` (and `_write_import_entries` for the
  catalog-import path) calls `write_structural_edges(gc, batch_ids,
  impact_closure=False)` at batch tail — per-batch writes keep the
  graph consistent during long runs.
- `sn export` calls `write_structural_edges(gc, all_valid_ids,
  impact_closure=False)` **as a mandatory pre-export gate** so
  catalog emission never sees stale or missing structural edges.
- The new `sn graph reconcile-structural` CLI (see §4) runs the same
  function across the full graph with `impact_closure=True` for
  backfill and on-demand repair.

### 4. CLI

New sub-command — deliberately scoped outside the `sn run` phase
namespace to avoid collision with `sn run --only reconcile` (which
already means *source-staleness reconciliation*):

```
imas-codex sn graph reconcile-structural \
  [--tier {1,2,all}] \
  [--domain <physics_domain>] \
  [--ids <id1> <id2> ...] \
  [--dry-run] [--report path.json]
```

Flags:
- `--tier` selects which edge families to reconcile (default `all`).
- `--dry-run` reports what would change without writing.
- `--report` emits the `StructuralReport` as JSON for CI integration.
- Scope flags narrow the impact-closure seed set.

### 5. Cluster + domain edge wiring

In `write_standard_names` and `_write_import_entries`, after writing
scalar properties:

```cypher
FOREACH (_ IN CASE WHEN b.cluster_id IS NOT NULL THEN [1] ELSE [] END |
  MERGE (c:IMASSemanticCluster {id: b.cluster_id})
  MERGE (sn)-[:IN_CLUSTER]->(c)
)
FOREACH (_ IN CASE WHEN b.physics_domain IS NOT NULL THEN [1] ELSE [] END |
  MERGE (d:PhysicsDomain {id: b.physics_domain})
  MERGE (sn)-[:AT_PHYSICS_DOMAIN]->(d)
)
```

### 6. Tier-2 backfill for existing nodes

A separate one-shot sub-command
`imas-codex sn graph backfill-tier2` MERGEs `IN_CLUSTER` and
`AT_PHYSICS_DOMAIN` edges for every existing `StandardName` node from
its `primary_cluster_id` and `physics_domain` scalar properties.
Scheduled to run exactly once immediately after the schema + writer
commit lands. Writes batches of 100, `@retry_on_deadlock()`,
deterministic id order.

### 7. Catalog-import parity fixes

In `_write_import_entries` (`catalog_import.py`):
- After the main MERGE block, call `_write_segment_edges(gc,
  [e["id"] for e in entries])` so imported names have `HAS_SEGMENT`
  parity with pipeline-generated names.
- Add the same `HAS_COCOS` MERGE that `write_standard_names` does,
  gated on `b.cocos_transformation_type IS NOT NULL`.
- Call `write_structural_edges(gc, [e["id"] for e in entries],
  impact_closure=True)` — catalog PRs may touch any edge source or
  target, so closure is mandatory here.

### 8. Tests (new in `tests/standard_names/`)

Behavioural tests — each run against a fresh in-memory fixture graph:

1. **Component edge** — composed vector + components produces
   `COMPONENT_OF` edges when parents exist; no edges when parents
   absent; `grammar_component` mismatch produces no edge.
2. **Uncertainty edge** — `dd_error_modifier` children produce
   `UNCERTAINTY_OF`; skipped siblings do not.
3. **Complex both forms** — suffix-form and prefix-form real/imag
   names both produce correct edges.
4. **Deprecation symmetry** — setting `deprecates` on A produces
   both directions of the edge pair; clearing it removes both.
5. **Retarget** — changing `superseded_by` from X to Y replaces the
   old edge with the new one.
6. **Rename impact closure** — renaming parent P to P' with
   `impact_closure=True` repairs all `COMPONENT_OF` edges targeting
   it in a single reconcile call.
7. **Batch-ordering recovery** — child composed before parent leaves
   a missing edge; reconcile at run-end or pre-export creates it.
8. **Tier-2 backfill** — existing nodes (no edges pre-backfill) gain
   `IN_CLUSTER` and `AT_PHYSICS_DOMAIN` edges after one backfill
   run; idempotent on re-run.
9. **Catalog-import parity** — a catalog-imported entry ends up with
   the same edge set (modulo source edges) as a pipeline-composed
   one.
10. **Dry-run** — `--dry-run` produces a non-empty report but writes
    no edges (pre/post Cypher counts identical).

Schema-compliance tests (`tests/graph/test_schema_compliance.py`,
`test_referential_integrity.py`) are re-run after the schema +
writer commit and must pass — this is guaranteed by making the
schema and writer a **single atomic commit** (see §10).

### 9. Out of scope

- `HAS_TAG` → `Tag` node migration. Tag strings are adequate for
  `WHERE 'equilibrium' IN sn.tags` filtering today; node promotion
  carries cost without proportional query benefit until Tag nodes
  gain metadata. **Deferred.**
- `species` / `population` / `toroidal_mode` / `flux_surface_average`
  vocabulary nodes. These slots are schema-declared but have zero
  writers today. Wait for a concrete consumer. **Deferred.**
- Creating missing vector parent names. The 80/88 orphan
  `COMPONENT_OF` targets are a vocabulary-completeness follow-up,
  not part of this plan. A `StructuralReport` section lists them
  so the follow-up plan has a concrete seed list.
- LLM-driven relationship inference. Every Tier 1/2 edge is
  deterministic from existing data.

### 10. Rollout

1. **Schema + writers + CLI = one atomic commit.** Update
   `standard_name.yaml`, `common.yaml`, `facility.yaml`
   (`MENTIONS_DOMAIN` → `AT_PHYSICS_DOMAIN` rename), rebuild models
   via `uv run build-models --force`, add `structural_edges.py`,
   update `write_standard_names`, `persist_composed_batch`,
   `_write_import_entries`, add the `PhysicsDomain` bootstrap
   command, add `sn graph reconcile-structural` CLI. All tests
   green. This is a single commit because an intermediate
   schema-only state would leave
   `tests/graph/test_schema_compliance.py` red in CI for the
   inter-commit interval.
2. **Pre-flight verify** (before running backfill).
   ```
   MATCH ()-[r:MENTIONS_DOMAIN]->() RETURN count(r) AS n
   ```
   If `n > 0`, add a data migration step to the backfill command;
   otherwise the rename is a pure schema change. Pre-revision grep
   confirms `n` is expected to be 0.
3. **Tier-1 backfill.** Run
   `imas-codex sn graph reconcile-structural --tier 1 --report
   tier1-backfill.json`. Review report for unexpected counts and
   orphan parents.
4. **Tier-2 backfill.** Run
   `imas-codex sn graph backfill-tier2`. Verify edge counts with a
   spot-check Cypher query.
5. **Docs commit.** Update `AGENTS.md`; verify auto-regenerated
   `agents/schema-reference.md` reflects all new relationships.
6. **Release.** Graph schema version bump
   (`standard_name.yaml` schema version `0.9.0 → 0.10.0`); follow
   release-CLI workflow to push an rc tag to GHCR.

### 11. Observability

`sn graph reconcile-structural --report` emits a JSON summary:

```json
{
  "tier1": {
    "component_of": {"created": 7, "deleted": 0, "stub_parents": 0, "errors": 0},
    "uncertainty_of": {"created": 220, "deleted": 0, "errors": 0},
    "real_part_of": {"created": 4, "deleted": 0, "errors": 0},
    "imaginary_part_of": {"created": 2, "deleted": 0, "errors": 0},
    "deprecates": {"created": 0, "deleted": 0, "errors": 0},
    "superseded_by": {"created": 0, "deleted": 0, "errors": 0}
  },
  "tier2": {
    "in_cluster": {"created": 690, "errors": 0},
    "at_physics_domain": {"created": 696, "errors": 0}
  },
  "orphan_component_parents": ["current_density_due_to_non_inductive_current_drive", "..."]
}
```

A test in CI asserts the report structure and that `errors` is
always 0.

## Documentation updates

- `AGENTS.md` — Standard Names → Schema: update the relationships
  table; add `write_structural_edges` to the module table; document
  `sn graph reconcile-structural` and `sn graph backfill-tier2` in
  the CLI table; document the pre-export reconcile gate.
- `agents/schema-reference.md` — auto-regenerated by `uv run
  build-models`.
- `plans/README.md` — add plan 39.
- `docs/architecture/standard-names.md` — new "Structural edges"
  section.

## Revision notes

Changes since first draft — all driven by reviewer feedback:

- Suffix-form real/imag handling (Sonnet #1a, verified against live
  catalog: 6 suffix names, 0 prefix names).
- 91% orphan `COMPONENT_OF` parents declared as known gap (Sonnet
  #1b, verified: 80/88 orphans). Stub parent nodes surface the
  vocabulary gap.
- `grammar_component` guard added to `COMPONENT_OF` Cypher (GPT #2).
- `DEPRECATES` / `SUPERSEDED_BY` now written as mandatory inverse
  pair (Sonnet #1c, GPT #2).
- Impact closure on reconcile — handles rename/retarget of edge
  targets (GPT #1).
- `split('_component_of_')` uses `size(parts) = 2` guard rather
  than arbitrary index (GPT #1d / Sonnet #1d).
- `MENTIONS_DOMAIN` rename Cypher corrected from the invalid
  `CREATE ()-[:...]->()` form; verified zero live writers so no
  data migration is needed (GPT #3).
- `@retry_on_deadlock()` on all reconcile writers (GPT #4).
- Tier-2 backfill for existing nodes added as explicit step (GPT
  #5).
- Pre-export reconcile gate made mandatory (all three reviewers).
- Schema + writer combined into one atomic commit (Sonnet #9).
- CLI renamed from `sn reconcile-structural` to `sn graph
  reconcile-structural` to avoid collision with `sn run --only
  reconcile` (GPT #11).
- `--dry-run` + `--report` flags added (Opus #7a).

**Round-2 revisions:**

- Undirected DELETE in `DEPRECATES` / `SUPERSEDED_BY` Cypher — removes
  both outgoing and incoming deprecation edges incident to `sn`, so a
  stale inverse edge from a previous counterpart is cleaned up when
  the old counterpart is reprocessed (GPT r2 #1).
- Impact closure extended with a second **edge-reachability** pass
  (undirected `MATCH`) so retargeting a `deprecates` from B to C now
  pulls B into the closed set and cleans its stale
  `SUPERSEDED_BY -> A` edge (GPT r2 #2).
- Stub parent nodes from `REAL_PART_OF` / `IMAGINARY_PART_OF` /
  `COMPONENT_OF` now tagged `pipeline_status = 'stub'` on creation.
  Upgraded in place by `write_standard_names` via coalesce semantics
  (Opus r2 #1).
- `StructuralReport` observability (Opus #7a).
- Plan 40 dependency tightened to Tier 1 only (Opus #1).
