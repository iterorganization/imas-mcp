# Plan 40 — Domain-Grouped Catalog Layout with Graph-Hierarchy Preservation

> Revised after three-reviewer critique (Sonnet 4.6, Opus 4.6, GPT-5.4).
> NetworkX local graph + catalog-site renderer upgrades split out to
> a follow-up plan 41.

## Problem

The IMAS Standard Names Catalog (ISNC) stores every standard name as a
single YAML file at `standard_names/<physics_domain>/<name>.yml` (480
files today, 23 domains). Audit evidence in session notes
`files/isn-audit.md`:

1. **Pipeline PRs do not scale.** A 400-name documentation-quality
   pass generates a 400-file PR. GitHub diff review is functionally
   unusable at that scale. A 50-name bulk run is still 50 isolated
   file diffs with no semantic grouping. `publish.py` does a
   wholesale `shutil.rmtree` + `shutil.copytree`, so there is no
   surgical per-name patching.
2. **Graph hierarchy is lost at the catalog boundary.** Our Neo4j
   graph encodes many structural relationships (plan 39:
   vector ↔ components, complex ↔ real/imag, parent ↔ uncertainty
   siblings, deprecation chains, cross-references). The exported
   YAML is flat — every structural relationship must be re-derived
   by the consumer from name strings. Related names are not
   adjacent in the file, so a reviewer cannot see a vector plus
   all its components in a single diff hunk.

## Approach

Restructure the catalog from **one-file-per-name** to
**one-file-per-physics-domain** as a YAML sequence of entries. Order
entries within each file by an explicit **graph traversal** over
structural edges from plan 39 (roots first, then descendants
depth-first, alphabetic tie-break at each sibling set). Ordering is
stable across cluster re-assignments because cluster membership is
**not** in the sort key — cluster is graph-only.

Emit the structural edges from plan 39 as inline YAML fields (`parent`,
`components`, `real_part`, `imag_part`, `uncertainty_siblings`) so
related names are adjacent in the diff and downstream consumers can
build a local graph from the YAML alone. These fields are declared as
**COMPUTED_FIELDS** in the round-trip model — always re-derived from
graph edges on export, ignored by `sn import` (never written from YAML
to the graph). This preserves the editorial-vs-computed boundary:
humans edit `description`, `documentation`, `tags`; the pipeline owns
hierarchy.

Ensure byte-stable round-trip by declaring an explicit
`CANONICAL_KEY_ORDER` and emitting entries through it. Protect against
partial-export data loss by introducing an `export_scope` manifest
field that `sn publish` consults before any destructive file
operation.

NetworkX local-graph module, MCP tools, and catalog-site renderer
upgrades (Mermaid, structured nav, hyperlink resolution) are
**out of scope** for this plan and ship as plan 41.

Plan 40 depends on **Plan 39 Tier 1 only**. Tier 2 (`IN_CLUSTER`,
`AT_PHYSICS_DOMAIN` edges) is independent and can slip without
blocking this work.

## Scope

### A — imas-codex repo

#### 1. Computed-field contract

Introduce module-level constants in
`imas_codex/standard_names/catalog_import.py`:

```python
COMPUTED_FIELDS = frozenset({
    "parent",
    "components",
    "real_part",
    "imag_part",
    "uncertainty_siblings",
})
```

Import-time semantics:
- `catalog_import` reads YAML entries normally.
- When diffing against the graph and applying edits,
  `COMPUTED_FIELDS` entries are **silently ignored** — they are
  never written to node properties and never trigger the
  `_protected_fields_differ` branch. A human who edits them in a
  catalog PR sees their edit discarded on the next export round-trip
  (logged at INFO level, not as an error).
- `PROTECTED_FIELDS` is unchanged. Computed fields are explicitly
  not protected; they are not editorial.

**Curator-facing warning.** `check_catalog` (the pre-merge CI hook)
emits a **WARNING** for every catalog PR diff that touches a
`COMPUTED_FIELD`, with message:
`"{field} is computed from graph edges and will be overwritten on
next export — edit has no effect. See plan 40 / COMPUTED_FIELDS."`
This surfaces the silent-drop semantics to the PR author before
merge, so they can either revert the edit or request the corresponding
graph change via a codex-side PR.

Export-time semantics:
- `_graph_node_to_entry_dict` always re-derives `COMPUTED_FIELDS`
  from graph structural edges. The YAML is a view of graph truth.

#### 2. Export format — `imas_codex/standard_names/export.py`

- Replace `_write_entry_yaml` with
  `_write_domain_yaml(staging, domain, entries)`. Output path:
  `<staging>/standard_names/<domain>.yml`. Format: YAML sequence of
  entry mappings.
- Per-domain file header (YAML block comment, emitted via a small
  pre-write concatenation — `ruamel.yaml` not required because the
  header is a single multi-line comment prepended to
  `yaml.safe_dump` output):

  ```yaml
  # Domain: <domain>
  # Catalog sha: <codex sha>
  # Ordering: structural traversal (roots first, children DFS,
  #           alphabetic tie-break)
  ```

- **Ordering — explicit graph traversal.** New function
  `order_entries_by_hierarchy(entries, graph_edges) -> list[entry]`
  (in `imas_codex/standard_names/catalog_ordering.py` — new
  module). Algorithm:

  1. Construct a DAG `G` over domain entries using only
     `COMPONENT_OF`, `REAL_PART_OF`, `IMAGINARY_PART_OF`,
     `UNCERTAINTY_OF` edges from plan 39. `DEPRECATES` /
     `SUPERSEDED_BY` are **excluded** from the ordering DAG because
     a bidirectional pair would create a 2-cycle by construction.
  2. Identify roots: entries with no **outgoing** ordering edge
     within the domain. The structural edges all point from child
     to parent (`child -[:COMPONENT_OF]-> parent` etc.), so a root
     in the hierarchy is a node that has no parent of its own —
     i.e. no outgoing edge within the ordering DAG. It may have any
     number of incoming edges from children.
  3. Sort roots alphabetically by `id`.
  4. For each root, **pre-order DFS** its descendants via
     **incoming** ordering edges (children point at their parent);
     at each sibling-set within the DFS, sort alphabetically by
     `id`. Pre-order means an entry is emitted before its own
     children, so a complex parent that is also an uncertainty
     child is visited under its uncertainty parent (outer
     grouping) and then its real/imag children follow.
  5. Cycle detection: if the DFS revisits a node via a non-tree
     edge, `order_entries_by_hierarchy` raises `OrderingError` with
     the offending cycle. The pre-export reconcile gate (plan 39)
     and the structural-edge invariants make this physically
     impossible, but the guard is defensive.
  6. Append orphan entries (not reachable from any root, and not
     emitted by DFS) at the end in alphabetic order.

  This ordering is a **pure function of (entry-ids, ordering-edges)**.
  It is stable across cluster re-assignments because cluster is not
  used. It is stable across graph property perturbations because
  only id and edge topology are consulted.

- **Canonical key order.** Add
  `imas_codex/standard_names/canonical.py::CANONICAL_KEY_ORDER` as
  an **exhaustive** tuple listing every allowed top-level YAML key
  in emission order:

  ```python
  CANONICAL_KEY_ORDER = (
      "id", "name", "kind", "status",
      "description", "documentation",
      "unit", "cocos_transformation_type", "cocos",
      "physics_domain", "tags", "links",
      "validity_domain", "constraints",
      "parent", "components", "real_part", "imag_part",
      "uncertainty_siblings",
      "deprecates", "superseded_by",
      "provenance",
  )
  ```

  `_write_domain_yaml` constructs each entry dict in this order
  before passing to `yaml.safe_dump(..., sort_keys=False)`. Any
  entry key not present in `CANONICAL_KEY_ORDER` raises
  `UnknownCatalogKeyError` — **hard fail, not fallthrough**. This
  guarantees byte-stable round-trip regardless of Neo4j
  property-return order and forces every new field to be added to
  the tuple explicitly. A unit test drives the full-graph export
  and asserts no `UnknownCatalogKeyError` is raised.

- **Per-entry computed fields** (re-derived on every export):
  - `parent: <vector-name>` when the SN has a `COMPONENT_OF` edge.
  - `components: {<grammar_component>: <child-name>, …}` when the
    SN has `COMPONENT_OF` incoming edges — keyed by the child's
    `grammar_component`.
  - `real_part:` / `imag_part:` when the SN has `REAL_PART_OF` /
    `IMAGINARY_PART_OF` incoming edges.
  - `uncertainty_siblings: [upper, lower, index]` (sparse — only
    siblings that exist) when `UNCERTAINTY_OF` incoming edges
    exist.
  - `deprecates` / `superseded_by` — scalar from node property
    (not an inline-computed field because they are editorial).

#### 3. Export-scope manifest

Add a new key to the exported `catalog.yml` manifest at the root of
the staging directory:

```yaml
export_scope: full            # or "domain_subset"
domains_included: [transport, magnetics, …]
catalog_commit_sha: <sha>
exported_at: <iso8601>
```

`sn export` without `--domain` sets `export_scope: full` and lists
all 23 domains. `sn export --domain X` sets `export_scope:
domain_subset` and lists only the exported domain(s).

#### 4. Publish safety — `imas_codex/standard_names/publish.py`

Refactor to consult the export-scope manifest. All file operations
run under a filesystem lock on the ISNC checkout to serialise
concurrent publishes:

```python
with FileLock(isnc / ".sn-publish.lock", timeout=30):
    ...  # all validation, rmtree/copy, git operations
```

(Implementation via the `filelock` package, already an indirect
dependency. The lock file is gitignored.)

- **Pre-flight validation (runs under lock, before any write):**

  1. Load `<staging>/catalog.yml`. If missing or unparseable, abort.
  2. Compute `staged_domains = {basename(p).removesuffix('.yml')
     for p in (staging/standard_names).glob('*.yml')}`.
  3. Assert `set(domains_included) == staged_domains`. Mismatch in
     either direction (listed but missing file, or unlisted file
     present) is a fatal manifest inconsistency — abort.
  4. If `export_scope == "full"`, additionally assert
     `set(domains_included) == EXPECTED_DOMAIN_SET` where
     `EXPECTED_DOMAIN_SET` is loaded from the live codex graph
     (`MATCH (sn:StandardName) WHERE sn.pipeline_status IN [...]
     RETURN DISTINCT sn.physics_domain`). If the manifest claims
     `full` but is missing a domain that exists in the graph, abort
     — this prevents a malformed manifest from triggering an rmtree
     that silently deletes a real domain.
  5. Assert ISNC working tree is clean (`git status --porcelain`
     empty). Refuse to publish on top of uncommitted changes.

- **Full-scope path** (`export_scope: full`): after pre-flight
  passes, `shutil.rmtree(<isnc>/standard_names)` +
  `shutil.copytree(staging/standard_names, isnc/standard_names)`.
- **Domain-subset path** (`export_scope: domain_subset`): replace
  **only** the listed domain files. For each domain in
  `domains_included`, `shutil.copy2(staging/standard_names/<d>.yml,
  isnc/standard_names/<d>.yml)`. Other domain files untouched.
- **Post-copy validation:** re-run `check_catalog(isnc)` and
  `load_catalog(isnc)` against the new layout. If either fails,
  the lock holder issues `git checkout -- standard_names/` to
  revert the file operations before releasing the lock.
- Commit-message upgrade: list modified domains explicitly, e.g.
  `sn: update transport, magnetics (12 entries)`.

#### 5. Import adaptation — `imas_codex/standard_names/catalog_import.py`

- `_derive_domain_from_path` regex: `standard_names/([^/]+)\.ya?ml$`
  (the file basename is the domain). Reject names containing `/`.
- Loader: both `run_import()` and `check_catalog()` iterate
  `yaml.safe_load(path)` and handle the list-root case. A top-level
  dict (legacy per-file) is rejected with a clear error message —
  this is a greenfield migration per project convention.
- Per-entry processing unchanged: validate / diff / merge.
- `_protected_fields_differ` / `filter_protected` unchanged —
  field-level protection is per-entry. `COMPUTED_FIELDS` silently
  ignored (see §1).
- At the end of each import, call
  `write_structural_edges(gc, imported_ids, impact_closure=True)`
  so hierarchy edges reflect catalog edits (plan 39 integration).

#### 6. Tests — `tests/standard_names/`

1. **Round-trip byte stability.** Fixture graph → export → parse →
   re-emit with `CANONICAL_KEY_ORDER` → byte-identical to first
   emission.
2. **Round-trip idempotence.** Export → import → export again. Graph
   state identical on re-run (all properties + all edges).
3. **Ordering fixture.** Domain with one vector, 3 components, one
   complex base with real+imag, one uncertainty triplet, two
   orphans. Assert exact output order.
4. **Ordering stability under cluster re-assignment.** Reassign
   `primary_cluster_id` for one entry; re-export; assert file
   positions unchanged (no spurious diff).
5. **Ordering stability under property permutation.** Permute
   Neo4j property return order; re-export; assert byte-identical
   output.
6. **Computed-field ignored on import.** Catalog PR edits
   `parent: X` to `parent: Y` on a component entry. Import runs.
   Graph `COMPONENT_OF` edge unchanged (still points at the name
   derived from grammar). Info-level log emitted.
7. **Partial-export publish safety.** `sn export --domain
   transport` + `sn publish` touches only `transport.yml`; other
   22 domain files untouched. If the staging manifest claims
   `export_scope: full` but lists only one domain, abort.
8. **`check_catalog` + list-root parity.** `check_catalog` runs
   against a list-root fixture and emits the same error set as
   `run_import` would.
9. **Legacy per-file rejection.** A top-level-dict YAML file in
   `standard_names/X.yml` is rejected with a clear migration-error
   message.
10. **Orphan-component export.** A domain with an orphan component
    (parent missing — see plan 39's known gap) exports with
    `parent: <missing-name>` present but no `components` back-link
    on the target; the importer does not fail.

#### 7. Documentation (codex side)

- `AGENTS.md` — Standard Names → Catalog round-trip:
  updated file layout, inline hierarchy fields, computed-field
  contract, `export_scope` manifest, ordering algorithm.
- `plans/README.md` — add plan 40.
- `docs/architecture/standard-names.md` — extend the round-trip
  diagram.

### B — imas-standard-names library

#### 8. Loader refactor — `imas_standard_names/yaml_store.py`, `imas_standard_names/rendering/catalog.py`

- `YamlStore.load()` and `CatalogRenderer.load_names()` iterate
  `yaml.safe_load(path)` as `list[dict]`. Top-level dict (legacy
  per-file) rejected with a migration error.
- No multi-document (`---`-separated) streams: `yaml.safe_load`
  remains the sole loader API.

#### 9. Pydantic model additions — `imas_standard_names/schemas/`

Optional fields added to the entry model:

```python
parent: str | None = None
components: dict[str, str] | None = None
real_part: str | None = None
imag_part: str | None = None
uncertainty_siblings: list[str] | None = None
```

Cross-reference validator (referential integrity, not structural):
if `parent: X` is present, X must exist in the loaded catalog at
validation time. Validation is a **warning, not an error**, for
missing targets — matches the plan 39 known-gap policy (91% of
component parents are absent).

`cluster` field is **not** added — cluster is graph-only; emitting
it inline introduces unstable ids into catalog diffs.

#### 10. Topological load ordering — `imas_standard_names/ordering.py`

Extend the existing `TopologicalSorter` edges to include
`parent` and `real_part` / `imag_part` so catalog entries with
forward-references load in the correct order. Emit a single
deterministic order even when the YAML list order is different
(catalog files are ordered by plan 40's hierarchy traversal, but
the ISN loader should be independent of emit order).

#### 11. Tests (ISN side)

- Round-trip: load a domain-file fixture, re-serialise, match
  byte-for-byte.
- Missing cross-reference warning fires but does not fail
  validation.
- Topological load handles `parent` edges.
- Legacy per-file YAML produces a clean migration error.

#### 12. Documentation (ISN side)

- ISN `AGENTS.md` — catalog file layout section.
- `CONTRIBUTING.md` (ISN / ISNC) — how to edit a domain file, how
  to resolve merge conflicts (rebase-then-merge workflow for
  concurrent PRs on the same domain file).
- `mkdocs.yml` — no renderer changes in plan 40; deferred to plan
  41.

### C — imas-standard-names-catalog repo

#### 13. One-shot migration

Migration PR steps (exactly once, coordinated with the codex
publish commit):

1. Codex runs `sn export` against the production graph into an
   empty staging dir. Staging contains 23 `<domain>.yml` files plus
   `catalog.yml` with `export_scope: full`.
2. `sn publish` does a full `rmtree` + `copytree` into ISNC (wiping
   all 480 per-name files, writing 23 domain files).
3. Commit message: `sn: migrate to per-domain file layout (plan 40)`.
4. PR reviewed domain-by-domain (23 files, one commit, one author
   per domain comment thread).

Rollback plan (if the ISN library or codex export turns out to be
broken) must **account for the codex deployment pin** — a partial
rollback that only reverts ISNC but leaves codex pinned to the rc
ISN library will break catalog workflows because the rc loader
rejects the reverted legacy layout. The commits are sequenced so
rollback is possible from any step:
- If ISN library commit breaks: revert before migration PR.
- If codex export breaks: revert codex commit; ISNC unchanged.
- If migration PR surfaces issues: revert in this **exact order**:
  (a) open a codex PR pinning back to the pre-rc ISN tag and deploy
  that codex rc, then (b) revert the ISNC migration commit. Skipping
  (a) leaves the deployed codex unable to parse reverted ISNC.
- ISN pin uses the **rc tag** initially; after stabilisation the
  pin is moved to an immutable commit SHA to defend against tag
  mutation. The codex release that carries the SHA pin is the
  "stable" cut-over.

## Out of scope

- **NetworkX local-graph module and MCP tools** (moved to plan 41).
- **Catalog-site renderer upgrades**: Mermaid blocks, hyperlink
  resolution, structured nav, `cocos_transformation_type`
  rendering. All moved to plan 41.
- **Cluster field** (`cluster: …`) in inline YAML. Cluster
  membership is graph-only; ordering does not depend on it; the
  NetworkX module in plan 41 can read cluster membership from a
  separate manifest if needed.
- **Markdown-with-frontmatter or TOML formats.** YAML sequence wins
  on reader simplicity and parser availability (audit §B8).
- **Per-commit provenance attribution** (one commit per
  pipeline-touched name). Pipeline PRs remain one-commit-per-run;
  per-name provenance lives in node properties
  (`last_run_id`, etc.).
- **Full merge-conflict avoidance engineering.** Conflicts within a
  domain file are an accepted cost. `CONTRIBUTING.md` prescribes a
  rebase-then-merge policy for concurrent PRs on the same domain
  file.
- **Legacy per-file loader shim.** The project convention is
  greenfield: no backwards compat. Legacy YAML is rejected with a
  migration error.

## Rollout

Plan 40 must land **after** plan 39 Tier 1 edges are present in the
production graph (export reads those edges). Tier 2 of plan 39 is
independent; this plan does not wait on it.

Sequence:

1. **ISN library release (optional fields + loader).** Bump ISN
   version (e.g., `v0.7.0 → v0.8.0rc1`). Published to GitHub
   tag. Loader accepts list-root YAML; rejects legacy per-file.
2. **Codex ISN pin bump.** Update `pyproject.toml`:
   `imas-standard-names @ git+https://github.com/iterorganization/imas-standard-names@v0.8.0rc1`.
   Verify `uv sync` and unit tests green locally.
3. **Codex export + import + publish commit.** Switch to per-domain
   files; introduce `CANONICAL_KEY_ORDER`, `export_scope`,
   publish-mode branching. Update `check_catalog`. All tests green.
4. **Deploy codex.** Standard release-CLI rc tag; GHCR push.
5. **One-shot migration PR.** Codex runs `sn export` + `sn publish`
   against production graph into ISNC repo. PR opened with
   domain-by-domain review thread.
6. **Docs commits** (ISN, codex, ISNC).
7. **ISN final release** once migration is stable.

## Observability

- `sn export` logs `export_scope`, `domains_included`, entry
  counts per domain.
- `sn publish` logs whether it took the full-scope or
  domain-subset code path, and the exact list of files written.
- `sn import` logs any computed-field overrides that were
  silently ignored (with the name and field, at INFO level) so a
  catalog curator can see that their edit was not persisted.

## Revision notes

Changes since first draft — driven by reviewer feedback:

- **Editorial boundary fixed** (Opus #2 — blocking): introduced
  `COMPUTED_FIELDS` contract so the import path does not write
  hierarchy fields from YAML; export always re-derives from graph
  edges.
- **Cluster removed from inline YAML** (Sonnet #4, Opus #2, GPT
  #7). Unstable ids and editorial ambiguity both eliminated.
- **Ordering algorithm is explicit graph traversal** (GPT #8), not
  a comparator, and does **not** use cluster as the primary sort
  key (Sonnet #6). Stable across cluster re-assignments and Neo4j
  property permutations.
- **Byte-stable round-trip** via explicit `CANONICAL_KEY_ORDER`
  (Sonnet #7).
- **Partial-publish data-loss hole closed** with `export_scope`
  manifest and publish-mode branching (GPT #6 — blocking).
- **Importer parity**: `check_catalog` also updated (GPT #10).
- **ISN topological load** extended for `parent` edges (Sonnet #8).
- **NetworkX + renderer upgrades split out to plan 41** (Opus #5).
- **Dependency tightened** to plan 39 Tier 1 only (Opus #1).
- **Rollback plan** added for each rollout step (Opus #7b).
- **Version choreography explicit**: ISN release → codex pin bump
  → codex release → migration (GPT #12).
- **Known coverage gap acknowledged**: 91% of components have no
  parent name; inline `parent:` fields will be missing for those
  until a vocabulary completeness pass (separate plan) runs.

**Round-2 revisions:**

- **Ordering root detection fixed** (GPT r2 #6): roots are entries
  with no **outgoing** ordering edge (top of hierarchy), not no
  incoming. Corrected direction comment in step 4. Added explicit
  pre-order DFS semantics. Added cycle-detection guard.
- **`CANONICAL_KEY_ORDER` made exhaustive** (GPT r2 #5): full tuple
  enumerated; unknown keys raise `UnknownCatalogKeyError` (hard
  fail, not fallthrough). Unit test drives full-graph export.
- **Publish pre-flight hardened** (GPT r2 #4): both-direction
  manifest check (`set(domains_included) == set(staged_domains)`),
  full-scope completeness check against graph
  (`EXPECTED_DOMAIN_SET`), concurrent-publish `FileLock`, clean
  worktree assertion, post-copy rollback on `check_catalog`
  failure.
- **Rollback accounts for deployed codex pin** (GPT r2 #8):
  explicit ordering — revert codex pin first, then revert ISNC.
  Pin moves from rc tag to immutable commit SHA at stabilisation.
- **Curator-facing COMPUTED_FIELDS warning in `check_catalog`**
  (Opus r2 #2): WARNING on PR diff touching a computed field so
  the author sees the silent-drop semantics before merge.

## Documentation updates

- `AGENTS.md` (codex) — Catalog round-trip section: new file
  layout, `COMPUTED_FIELDS` contract, `CANONICAL_KEY_ORDER`,
  `export_scope` manifest, ordering algorithm.
- `AGENTS.md` (ISN) — catalog file layout section.
- `CONTRIBUTING.md` (ISN / ISNC) — per-domain edit workflow,
  rebase-then-merge for concurrent PRs, `sn preview` local
  workflow.
- `plans/README.md` — add plan 40; mark it as depending on plan
  39 Tier 1; reference plan 41 for downstream consumers.
- `docs/architecture/standard-names.md` — round-trip diagram
  update.
