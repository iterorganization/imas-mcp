# Plan 40 — Domain-Grouped Catalog Layout with Graph-Hierarchy Preservation

> **Rewritten** after plan 39 shipped as commit `0d5674a3` with a different
> edge model from the original plan-39 draft.  The shipped edge set is
> `HAS_ARGUMENT`, `HAS_ERROR`, `HAS_PREDECESSOR`, `HAS_SUCCESSOR`,
> `IN_CLUSTER`, `HAS_PHYSICS_DOMAIN` — driven by the ISN grammar parser
> (`imas_standard_names.grammar.parser`).  See
> `AGENTS.md` → "StandardName Graph Edges" for the canonical table and
> `imas_codex/standard_names/derivation.py` for the derivation logic.

## Problem

The IMAS Standard Names Catalog (ISNC) stores every standard name as a
single YAML file at `standard_names/<physics_domain>/<name>.yml` (~480
files today, 23 domains).  Two concrete pains (audit evidence in
`files/isn-audit.md`):

1. **Pipeline PRs do not scale.**  A 400-name documentation-quality
   pass generates a 400-file PR.  GitHub diff review is functionally
   unusable at that scale.  A 50-name bulk run is still 50 isolated
   file diffs with no semantic grouping.  `publish.py` does a wholesale
   `shutil.rmtree` + `shutil.copytree`, so there is no surgical
   per-name patching.
2. **Graph hierarchy is lost at the catalog boundary.**  The Neo4j
   graph now encodes structural relationships via `HAS_ARGUMENT` /
   `HAS_ERROR` edges (plan 39, shipped).  The exported YAML is flat:
   a reviewer cannot see `x_component_of_magnetic_field`,
   `y_component_of_magnetic_field` and their base `magnetic_field`
   in adjacent diff hunks — each lives in its own file, alphabetically
   scattered across the domain directory.

## Approach

Restructure the catalog from **one-file-per-name** to
**one-file-per-physics-domain** as a YAML sequence of entries.  Within
each file, order entries by a deterministic **graph traversal** over
the shipped plan-39 edges (roots first, then descendants pre-order DFS,
alphabetic tie-break at each sibling set).  Emit the structural edges
as two inline YAML fields — `arguments` and `error_variants` — so
related names are adjacent in the diff and downstream consumers (plan
41) can reconstruct a local graph from the YAML alone.

`arguments` and `error_variants` are **computed fields**: always
re-derived from graph edges on export; silently ignored on import
(never written from YAML to graph).  Deprecation scalars
(`deprecates` / `superseded_by`) remain editorial — they have
corresponding graph edges, but the catalog is the editorial source of
truth for them (consistent with the shipped `_write_standard_name_edges`
write path which accepts either the pipeline-side `predecessor`/
`successor` properties or the catalog-side `deprecates` /
`superseded_by` properties).

Byte-stable round-trip is guaranteed by an exhaustive
`CANONICAL_KEY_ORDER`.  Partial-export data loss is prevented by an
`export_scope` manifest that `sn publish` consults before any
destructive file operation.

NetworkX local-graph, MCP traversal tools, and catalog-site renderer
upgrades (Mermaid, structured nav, hyperlink resolution) are
**out of scope** for this plan and ship as plan 41.

**Dependency:** plan 39 commit `0d5674a3` must be in the production
graph before `sn export` runs (export reads `HAS_ARGUMENT` and
`HAS_ERROR` edges).  It is — plan 39 shipped — so plan 40 may rollout
immediately.

## Scope

### A — imas-codex repo

#### 1. Computed-field contract

Introduce module-level constants in
`imas_codex/standard_names/catalog_import.py`:

```python
COMPUTED_FIELDS = frozenset({
    "arguments",
    "error_variants",
})
```

Import-time semantics:

- `catalog_import` reads YAML entries normally.
- When diffing against the graph and applying edits, `COMPUTED_FIELDS`
  entries are **silently ignored** — never written to node properties,
  never trigger the `_protected_fields_differ` branch.  A human who
  edits them in a catalog PR sees their edit discarded on the next
  export round-trip (logged at INFO level, not as an error).
- `PROTECTED_FIELDS` is unchanged.  Computed fields are explicitly not
  protected; they are not editorial.

**Curator-facing warning.**  `check_catalog` (pre-merge CI hook) emits
a **WARNING** for every PR diff that touches a `COMPUTED_FIELD`:

> `{field} is computed from HAS_ARGUMENT / HAS_ERROR graph edges and
> will be overwritten on next export — edit has no effect.  See plan
> 40 / COMPUTED_FIELDS.`

Export-time semantics: `_graph_node_to_entry_dict` always re-derives
`COMPUTED_FIELDS` from the graph.  YAML is a view of graph truth.

**Editorial scalars that mirror graph edges but stay editorial:**

- `deprecates` → `HAS_PREDECESSOR` edge target.  Catalog-writable.
- `superseded_by` → `HAS_SUCCESSOR` edge target.  Catalog-writable.
- `primary_cluster_id` → `IN_CLUSTER` edge target.  Pipeline-only;
  not emitted inline (cluster ids are unstable and introduce churn).
- `physics_domain` → `HAS_PHYSICS_DOMAIN` edge target.  Already a
  first-class editorial scalar on the entry (unchanged by this plan).

#### 2. Export format — `imas_codex/standard_names/export.py`

Replace `_write_entry_yaml` with
`_write_domain_yaml(staging, domain, entries)`.  Output path:
`<staging>/standard_names/<domain>.yml`.  Format: YAML sequence of
entry mappings.

Per-domain file header (YAML block comment, prepended to
`yaml.safe_dump` output):

```yaml
# Domain: <domain>
# Catalog sha: <codex sha>
# Entries: <count>
# Ordering: structural traversal
#   (HAS_ARGUMENT-incoming + HAS_ERROR-outgoing, pre-order DFS,
#    alphabetic tie-break)
```

**Inline computed fields** (re-derived on every export):

- **`arguments`** — emitted when the node has outgoing `HAS_ARGUMENT`
  edges.  Format depends on the outer operator shape (mirrors the
  ISN IR one-layer-peel):

  Unary prefix / postfix (1 edge):

  ```yaml
  arguments:
    - name: temperature
      operator: maximum
      operator_kind: unary_prefix
  ```

  Binary (2 edges, emitted in role order `a`, `b`):

  ```yaml
  arguments:
    - name: pressure
      operator: ratio
      operator_kind: binary
      role: a
      separator: to
    - name: density
      operator: ratio
      operator_kind: binary
      role: b
      separator: to
  ```

  Projection / component (1 edge):

  ```yaml
  arguments:
    - name: magnetic_field
      operator: component
      operator_kind: projection
      axis: x
      shape: component
  ```

  Edge property keys emitted when present: `operator`, `operator_kind`,
  `role`, `separator`, `axis`, `shape`.  `operator_kind` values are the
  literal ISN `OperatorKind` string values (`unary_prefix`,
  `unary_postfix`, `binary`, `projection`).

- **`error_variants`** — emitted when the node has outgoing
  `HAS_ERROR` edges.  Sparse mapping keyed by `error_type`:

  ```yaml
  error_variants:
    upper: upper_uncertainty_of_temperature
    lower: lower_uncertainty_of_temperature
    index: uncertainty_index_of_temperature
  ```

  Ordering of keys: `upper`, `lower`, `index` (fixed).

**Not emitted inline** (reverse index, can be reconstructed by
consumers from `arguments` across the catalog):

- `wrapped_by` / inverse `HAS_ARGUMENT`.  Emitting it on every base
  entry would add a line every time a new wrapping is introduced;
  plan 41's NetworkX module builds this index in-memory from the
  forward edges at load time.

**Ordering — deterministic topological traversal.**  New function
`order_entries_by_hierarchy(entries, edges) -> list[entry]`
(in `imas_codex/standard_names/catalog_ordering.py` — new module).

The ordering DAG is built from all `HAS_ARGUMENT` and `HAS_ERROR`
edges **between entries in the domain** (in-domain edges only).  For
ordering purposes the **ordering-parent** of a node `v` is the node
closer to the base:

- If `v -[:HAS_ARGUMENT]-> u`, then `u` is the ordering-parent of `v`.
- If `u -[:HAS_ERROR]-> v`, then `u` is the ordering-parent of `v`.

This unifies the two edge types into a single directed
"ordering-parent" relation whose direction is always "closer to
base" → "further from base".

Algorithm — **Kahn's topological sort with alphabetic tie-break**:

1. **Compute in-degree** for every in-domain entry under the unified
   ordering-parent relation.  Also compute the **full-graph**
   in-degree (including cross-domain edges) to distinguish true roots
   from cross-domain-orphans.
2. **Seed the ready set** with every entry whose **in-domain**
   in-degree is zero.  Split the ready set into two queues preserved
   through the whole sort:
   - **clean-roots queue** — entries whose **full-graph** in-degree
     is also zero (the purest bases, e.g.  `temperature`).
   - **orphan queue** — entries with zero in-domain in-degree but
     non-zero full-graph in-degree (their ordering-parent lives in a
     different domain — e.g.  a wrapped form whose base is in
     another domain).
3. **Drain in priority order.**  Repeatedly pop the alphabetically
   smallest entry from the **clean-roots queue** if non-empty, else
   from the **orphan queue**.  Emit it, decrement the in-domain
   in-degree of every in-domain child, and when a child's in-degree
   drops to zero append it to the **clean-roots queue** (children
   always inherit clean-root status once their in-domain parents
   have all been emitted — cross-domain in-degree is a property of
   the node itself, not inherited).
4. **Every entry is emitted exactly once** by construction.  No
   visited set, no de-dup branch.
5. **Cycle detection.**  If the queues are empty but unemitted
   entries remain (i.e.  their in-domain in-degree never reached
   zero), raise `OrderingError` listing those entries.  The DAG is
   acyclic by construction of the grammar parser, but the guard is
   defensive.

**Properties of this algorithm:**

- A node is emitted **only after all its in-domain ordering-parents
  have been emitted**.  This is the topological guarantee.  In
  particular a binary wrapping (e.g.  `ratio_of_pressure_to_density`)
  appears after both its arguments in the same domain.
- The clean-roots-first ordering ensures a base's own uncertainty
  siblings and wrappings immediately follow it (they become ready
  as soon as the base is emitted, inheriting clean-root queue
  status).  A domain's "true bases" cluster together at the top,
  each followed immediately by its variants, before any
  cross-domain orphans are appended.
- Pure function of `(entry-ids, in-domain-edges, cross-domain-edge
  presence)`.  Stable across cluster re-assignments (cluster not
  consulted), across Neo4j property permutations (only id and edge
  topology), and across unrelated additions to other domains — a
  cross-domain edge gaining or losing a target does not change
  any node's position unless that target leaves/enters the domain
  being ordered.

**Cross-domain orphan semantics.**  A wrapped form whose base is in
a different domain has a non-empty cross-domain ordering-parent set
but zero in-domain in-degree.  It is emitted from the **orphan queue
after all clean-roots are drained**, alphabetically.  This preserves
the intuition "bases first, orphans last" while still producing a
total order.

**Canonical key order.**  Add
`imas_codex/standard_names/canonical.py::CANONICAL_KEY_ORDER` as an
**exhaustive** tuple listing every allowed top-level YAML key in
emission order:

```python
CANONICAL_KEY_ORDER = (
    "id", "name", "kind", "status",
    "description", "documentation",
    "unit", "cocos_transformation_type", "cocos",
    "physics_domain", "tags", "links",
    "validity_domain", "constraints",
    "arguments", "error_variants",
    "deprecates", "superseded_by",
    "provenance",
)
```

`_write_domain_yaml` constructs each entry dict in this order before
passing to `yaml.safe_dump(..., sort_keys=False)`.  Any entry key not
present in `CANONICAL_KEY_ORDER` raises `UnknownCatalogKeyError` —
**hard fail, not fallthrough**.  This guarantees byte-stable
round-trip regardless of Neo4j property-return order and forces every
new field to be added to the tuple explicitly.  A unit test drives
the full-graph export and asserts no `UnknownCatalogKeyError` is
raised.

#### 3. Export-scope manifest

Add a new key to the exported `catalog.yml` manifest at the staging
root:

```yaml
export_scope: full            # or "domain_subset"
domains_included: [transport, magnetics, …]
catalog_commit_sha: <sha>
exported_at: <iso8601>
edge_model_version: plan_39_v1
```

`edge_model_version` pins the structural-edge schema the manifest was
produced under.  If plan 39 is superseded by a future edge-model
revision, manifests from the old model must be regenerated — the
publish step checks this value and refuses incompatible versions.

`sn export` without `--domain` sets `export_scope: full` and lists all
23 domains.  `sn export --domain X` sets `export_scope: domain_subset`
and lists only the exported domain(s).

#### 4. Publish safety — `imas_codex/standard_names/publish.py`

Refactor to consult the export-scope manifest.  All file operations
run under a filesystem lock on the ISNC checkout to serialise
concurrent publishes:

```python
from filelock import FileLock

with FileLock(isnc / ".sn-publish.lock", timeout=30):
    ...  # all validation, rmtree/copy, git operations
```

**Pre-flight validation (runs under lock, before any write):**

1. Load `<staging>/catalog.yml`.  If missing or unparseable, abort.
2. Assert `edge_model_version == "plan_39_v1"`; abort otherwise.
3. Compute `staged_domains = {basename(p).removesuffix('.yml')
   for p in (staging/standard_names).glob('*.yml')}`.
4. Assert `set(domains_included) == staged_domains`.  Mismatch in
   either direction (listed but missing file, or unlisted file
   present) is a fatal manifest inconsistency — abort.
5. If `export_scope == "full"`, additionally assert
   `set(domains_included) == EXPECTED_DOMAIN_SET` where
   `EXPECTED_DOMAIN_SET` is loaded from the live codex graph:
   `MATCH (sn:StandardName) WHERE sn.pipeline_status IN [...]
   RETURN DISTINCT sn.physics_domain`.  Prevents a malformed manifest
   from triggering an rmtree that silently deletes a real domain.
6. Assert ISNC working tree is clean.

**Full-scope path** (`export_scope: full`): after pre-flight passes,
`shutil.rmtree(<isnc>/standard_names)` +
`shutil.copytree(staging/standard_names, isnc/standard_names)`.

**Domain-subset path** (`export_scope: domain_subset`): replace
**only** the listed domain files.  For each domain in
`domains_included`, `shutil.copy2(staging/standard_names/<d>.yml,
isnc/standard_names/<d>.yml)`.  Other domain files untouched.

**Post-copy validation:** re-run `check_catalog(isnc)` and
`load_catalog(isnc)`.  On failure, the lock holder issues
`git checkout -- standard_names/` to revert, then raises.

Commit-message upgrade: list modified domains explicitly, e.g.
`sn: update transport, magnetics (12 entries)`.

#### 5. Import adaptation — `imas_codex/standard_names/catalog_import.py`

- `_derive_domain_from_path` regex:
  `standard_names/([^/]+)\.ya?ml$` (file basename is the domain).
  Reject names containing `/`.
- Loader: both `run_import()` and `check_catalog()` iterate
  `yaml.safe_load(path)` and handle the list-root case.  A top-level
  dict (legacy per-file) is rejected with a clear migration error —
  greenfield per project convention.
- Per-entry processing unchanged: validate / diff / merge.
- `_protected_fields_differ` / `filter_protected` unchanged.
  `COMPUTED_FIELDS` silently ignored (see §1).
- At the end of each import batch, `_write_standard_name_edges(gc,
  imported_entries)` is already wired (plan 39); hierarchy edges
  stay in sync with catalog edits automatically.  **No new code path
  required** — plan 40 leverages the shipped edge writer.

#### 6. Tests — `tests/standard_names/`

1. **Round-trip byte stability.**  Fixture graph → export → parse →
   re-emit through `CANONICAL_KEY_ORDER` → byte-identical.
2. **Round-trip idempotence.**  Export → import → export again.
   Graph state identical (all properties + all edges).
3. **Ordering fixture — unary prefix.**  Domain with `temperature`,
   `maximum_of_temperature`, `minimum_of_temperature`.  Expect order
   `[temperature, maximum_of_temperature, minimum_of_temperature]`.
4. **Ordering fixture — projection.**  Domain with `magnetic_field`,
   `x_component_of_magnetic_field`, `y_component_of_magnetic_field`,
   `z_component_of_magnetic_field`.  Components sorted alphabetically
   under the base.
5. **Ordering fixture — binary.**  Domain with `pressure`, `density`,
   `ratio_of_pressure_to_density`.  Expected order:
   `[density, pressure, ratio_of_pressure_to_density]`.  Both
   `density` and `pressure` have in-degree 0 (clean roots); alpha
   tie-break puts `density` first.  `ratio_of_pressure_to_density`
   has in-degree 2 and only becomes ready after both ordering-parents
   have been emitted, so it always appears last.
6. **Ordering fixture — uncertainty.**  Domain with `temperature` +
   its three uncertainty siblings.  Expect the four names in
   `[temperature, lower_uncertainty_of_temperature,
   uncertainty_index_of_temperature,
   upper_uncertainty_of_temperature]` (alphabetic tie-break among
   `HAS_ERROR` children).
7. **Ordering fixture — mixed.**  Domain with a base that has both
   uncertainty siblings AND components (unlikely but legal).  The
   base appears first (clean root); its uncertainty variants and
   components become ready together once the base is emitted and
   are drained alphabetically from the clean-roots queue.
8. **Ordering fixture — orphan.**  A wrapped form whose base lives
   in a different domain has zero in-domain in-degree but non-zero
   full-graph in-degree; it seeds the orphan queue, drained after
   the clean-roots queue empties.  Asserts the entry lands **after**
   all in-domain clean-root traversals.
9. **Ordering stability under cluster re-assignment.**  Reassign
   `primary_cluster_id` for one entry; re-export; assert file
   positions unchanged.
10. **Ordering stability under property permutation.**  Permute
    Neo4j property return order; re-export; assert byte-identical.
11. **Computed-field ignored on import.**  Catalog PR edits
    `arguments: …` on an entry.  Import runs.  Graph
    `HAS_ARGUMENT` edges unchanged.  INFO log emitted.  Curator
    warning surfaces in `check_catalog`.
12. **Partial-export publish safety.**  `sn export --domain transport`
    + `sn publish` touches only `transport.yml`; other 22 domain
    files untouched.  If the staging manifest claims
    `export_scope: full` but lists only one domain, abort.
13. **`check_catalog` + list-root parity.**  `check_catalog` runs
    against a list-root fixture and emits the same error set as
    `run_import`.
14. **Legacy per-file rejection.**  A top-level-dict YAML file in
    `standard_names/X.yml` is rejected with a clear migration error.
15. **Edge-model version guard.**  A manifest with
    `edge_model_version: plan_39_v0` is rejected by publish.

#### 7. Documentation (codex side)

- `AGENTS.md` — Standard Names → Catalog round-trip: updated file
  layout, inline `arguments` / `error_variants` fields, computed-field
  contract, `export_scope` manifest, ordering algorithm.
  Cross-reference the existing "StandardName Graph Edges" table.
- `plans/README.md` — strike P1e (plan 39 shipped); update P1f to
  link to this plan.
- `docs/architecture/standard-names.md` — extend the round-trip
  diagram with the new file-layout node.

### B — imas-standard-names library

#### 8. Loader refactor — `imas_standard_names/yaml_store.py`, `imas_standard_names/rendering/catalog.py`

- `YamlStore.load()` and `CatalogRenderer.load_names()` iterate
  `yaml.safe_load(path)` as `list[dict]`.  Top-level dict (legacy
  per-file) rejected with a migration error.
- No multi-document (`---`-separated) streams: `yaml.safe_load`
  remains the sole loader API.

#### 9. Pydantic model additions — `imas_standard_names/schemas/`

Two optional fields added to the entry model:

```python
class ArgumentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    operator: str
    operator_kind: Literal[
        "unary_prefix", "unary_postfix", "binary", "projection"
    ]
    role: Literal["a", "b"] | None = None
    separator: Literal["and", "to"] | None = None
    axis: str | None = None
    shape: Literal["component", "coordinate"] | None = None


class StandardNameEntry(BaseModel):
    # … existing fields …
    arguments: list[ArgumentRef] | None = None
    error_variants: dict[
        Literal["upper", "lower", "index"], str
    ] | None = None
```

Cross-reference validator (referential integrity, not structural):

- For every `arguments[].name`, warn (not error) if the target is
  absent from the loaded catalog — missing-target policy matches
  the plan 39 known-gap treatment.
- For every `error_variants[k]`, warn if absent.

`cluster` / `primary_cluster_id` fields are **not** added to the
entry model — cluster is graph-only; emitting it inline introduces
unstable ids into catalog diffs.

#### 10. Topological load ordering — `imas_standard_names/ordering.py`

Extend the existing `TopologicalSorter` edges to include all
`arguments[].name` references as load-ordering dependencies so
forward-referencing entries load after their targets.  Load ordering
is independent of emit ordering (the per-domain file emits via plan
40's hierarchy traversal, but the library's load must be robust to
any emit order).

#### 11. Tests (ISN side)

- Round-trip: load a domain-file fixture, re-serialise, match
  byte-for-byte.
- Missing cross-reference warning fires but does not fail
  validation.
- Topological load handles `arguments[].name` edges.
- Legacy per-file YAML produces a clean migration error.
- `ArgumentRef` validates `operator_kind` enum strictly;
  `role`/`separator` required for `binary`, `axis`/`shape` required
  for `projection`, forbidden otherwise.

#### 12. Documentation (ISN side)

- ISN `AGENTS.md` — catalog file layout section; ISN edge vocabulary
  (argument / error_variant) mirrors codex `HAS_ARGUMENT` /
  `HAS_ERROR`.
- `CONTRIBUTING.md` (ISN / ISNC) — how to edit a domain file, how to
  resolve merge conflicts (rebase-then-merge for concurrent PRs on
  the same domain file).
- `mkdocs.yml` — no renderer changes in plan 40; deferred to plan 41.

### C — imas-standard-names-catalog repo

#### 13. One-shot migration

Migration PR steps (coordinated with the codex publish commit):

1. Codex runs `sn export` against the production graph into an empty
   staging dir.  Staging contains 23 `<domain>.yml` files plus
   `catalog.yml` with `export_scope: full` and
   `edge_model_version: plan_39_v1`.
2. `sn publish` does a full `rmtree` + `copytree` into ISNC (wiping
   all ~480 per-name files, writing 23 domain files).
3. Commit message: `sn: migrate to per-domain file layout (plan 40)`.
4. PR reviewed domain-by-domain (23 files, one commit, one author
   per domain comment thread).

Rollback sequencing (must respect codex-side ISN pin):

- If ISN library commit breaks: revert before migration PR.
- If codex export breaks: revert codex commit; ISNC unchanged.
- If migration PR surfaces issues: revert in this **exact order**:
  (a) open a codex PR pinning back to the pre-rc ISN tag and deploy
  that codex rc, then (b) revert the ISNC migration commit.
  Skipping (a) leaves the deployed codex unable to parse reverted
  ISNC.
- ISN pin uses the rc tag initially; after stabilisation the pin
  moves to an immutable commit SHA.  The codex release carrying the
  SHA pin is the "stable" cut-over.

## Out of scope

- **NetworkX local graph + MCP tools + catalog-site renderer
  upgrades.**  All deferred to plan 41 (Mermaid hierarchy blocks,
  structured nav, `links:` hyperlink resolution, per-entry sibling
  navigation, `cocos_transformation_type` rendering).
- **`wrapped_by` reverse index inline in YAML.**  Plan 41's NetworkX
  loader builds it from forward `arguments` edges.
- **Cluster / physics-domain `IN_CLUSTER` / `HAS_PHYSICS_DOMAIN`
  fields inline.**  Cluster is graph-only.  Physics domain is already
  a first-class scalar on the entry.
- **Markdown-with-frontmatter or TOML formats.**  YAML sequence wins
  on reader simplicity.
- **Per-commit provenance attribution** (one commit per pipeline
  name).  Pipeline PRs remain one-commit-per-run.
- **Legacy per-file loader shim.**  Greenfield: legacy YAML rejected.

## Rollout

Plan 39 has shipped (commit `0d5674a3` — `HAS_ARGUMENT` / `HAS_ERROR`
edges live in production graph).  Plan 40 may begin immediately.

Sequence:

1. **ISN library release.**  Add `arguments` / `error_variants`
   fields to the Pydantic entry model; loader refactor; version bump
   (`v0.7.x → v0.8.0rc1`).
2. **Codex ISN pin bump.**  Update `pyproject.toml`; verify
   `uv sync` + unit tests green.
3. **Codex export + import + publish commit.**  Switch to per-domain
   files; introduce `CANONICAL_KEY_ORDER`, `export_scope`, publish
   branching, `FileLock`.  Update `check_catalog` with
   `COMPUTED_FIELDS` warning.  All tests green.
4. **Deploy codex.**  Standard release-CLI rc tag; GHCR push.
5. **One-shot migration PR.**  Codex runs `sn export` + `sn publish`
   against production graph into ISNC repo.  PR opened with
   domain-by-domain review thread.
6. **Docs commits** (ISN, codex, ISNC).
7. **ISN final release** once migration is stable; move codex pin
   from rc tag to commit SHA.

## Observability

- `sn export` logs `export_scope`, `domains_included`, entry counts
  per domain, `edge_model_version`.
- `sn publish` logs full-scope vs domain-subset path, exact list of
  files written, lock acquire/release.
- `sn import` logs any `COMPUTED_FIELDS` overrides silently ignored
  (name + field, INFO level).

## Revision notes

**Rewrite from the Round-2 plan** driven by the shipped plan-39
implementation differing from the drafted edge model:

- **Edge vocabulary replaced throughout.**  Drafted
  `COMPONENT_OF` / `REAL_PART_OF` / `IMAGINARY_PART_OF` /
  `UNCERTAINTY_OF` is not what shipped.  Shipped edges are
  `HAS_ARGUMENT` (parametrised by `operator` / `operator_kind` /
  `role` / `axis` / `shape` / `separator`) and `HAS_ERROR`
  (parametrised by `error_type`).
- **Computed-field surface collapsed.**  Was `{parent, components,
  real_part, imag_part, uncertainty_siblings}` (5 fields).  Now
  `{arguments, error_variants}` (2 fields) — the single
  `arguments` list carries every outer-operator shape.
- **Ordering algorithm replaced** with Kahn's topological sort
  (alphabetic tie-break, two-queue priority) after the original
  pre-order DFS was flagged by reviewer as inconsistent with test
  #5 and the orphan-tail semantics.  Kahn guarantees a node is
  emitted only after all its in-domain ordering-parents, removes
  the visited-set/cycle-detection conflict, and gives a clean
  clean-roots/orphan-queue split for cross-domain edges.
- **No separate edge-writer wiring required.**  Plan 39's shipped
  `_write_standard_name_edges` already handles both pipeline and
  import paths; plan 40 is pure catalog-layout work plus a
  COMPUTED_FIELDS ignore-pass on import.
- **`CANONICAL_KEY_ORDER` updated** to the two new field names.
- **`edge_model_version` manifest pin added** to prevent a future
  edge-model revision from being silently re-published under an old
  reader.

## Documentation updates

- `AGENTS.md` — Standard Names → Catalog round-trip section (updated
  layout, computed fields, export_scope, ordering).
- `docs/architecture/standard-names.md` — round-trip diagram.
- ISN `AGENTS.md` — catalog file layout + entry model additions.
- ISN `CONTRIBUTING.md` — per-domain editing + merge-conflict
  workflow.
- `plans/README.md` — P1e → shipped; P1f linked to this plan.
