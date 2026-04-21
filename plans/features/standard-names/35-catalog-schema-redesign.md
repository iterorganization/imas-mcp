# 35 — Catalog Schema Redesign & Gated Publish/Import Round-Trip

**Status:** PLANNING (pre-implementation, rubber-duck v2 pending)
**Depends on:** ISN package (we own, coordinated release); imas-standard-names-catalog repo

## Problem

The graph ↔ catalog round-trip is broken on multiple axes:

1. `sn publish` writes ad-hoc YAML that does not validate against any
   ISN Pydantic model. Extension (`.yaml` vs `.yml`), status enum
   (`drafted` vs `draft`), link encoding (`{name: x}` vs `name:x`
   string), and a spurious `provenance` block that collides with ISN's
   grammatical provenance slot.
2. The graph carries pipeline metadata that should never be in the
   public catalog (LLM scores, reviewer history, enrichment timings,
   worker coordination).
3. The graph carries fields that duplicate grammar (`species`,
   `population`, `toroidal_mode`, `flux_surface_average`, plus legacy
   `physical_base`/`subject`/`component`/`coordinate`/`position`/`process`
   now subsumed by `grammar_*`). These over-specify the name.
4. `cocos` (FK to COCOS singleton) is tracked per-name. Useful
   **graph-side** (find all COCOS-dependent quantities) but redundant
   in the catalog — the catalog supports a single convention, which
   belongs in a manifest, not repeated per-entry.
5. `review_status` (pipeline state) and ISN `status` (vocabulary
   lifecycle) are conflated under one field name.
6. There is no gate between "I ran the pipeline" and "I opened the
   export": publish silently emits whatever the graph holds, including
   drifted or unvalidated rows.

## Goals

- **Graph is authoritative**: the catalog repo is a thin sink of
  data exported from the graph. No migration of the existing catalog
  is needed — on first publish with the new schema, the catalog is
  wiped and regenerated from the graph. Rollback is a git revert on
  the catalog repo.
- **Responsibility split** (hard boundaries):
  - **ISN** — grammar, vocabulary, shared enums (`PhysicsDomain`),
    `StandardNameEntry` Pydantic model (catalog schema), catalog-site
    rendering machinery (`catalog-site serve` for local preview,
    `catalog-site deploy` for MkDocs→GitHub-Pages). Dependency-free
    of ISNC and the graph: ISN does not know ISNC exists as a git
    repo, and has no Neo4j driver.
  - **imas-codex** — graph is source of truth; owns pipeline
    metadata (`pipeline_status`, `reviewer_score`, `model`, …); owns
    graph↔disk transport: `sn export`, `sn publish`, `sn import`.
    Depends on ISN for schema + local preview (via `sn preview` thin
    wrapper).
  - **ISNC** (imas-standard-names-catalog) — thin repo that
    receives published data; no logic; just versioned YAML + git
    history. A GitHub Action on ISNC pushes triggers ISN
    `catalog-site deploy` to publish the rendered docs site,
    decoupled from imas-codex.
  - **Rationale for publish in codex, not ISN**: (a) symmetry with
    `sn import` which must live in codex since the graph is the
    destination; (b) codex already has the staging tree, gate
    results, and graph provenance at hand, so the commit message
    writes itself; (c) ISN stays free of any repo-writing logic.

    history.
- **Two-step export → preview → publish**:
  1. `sn export` — graph → local staging directory of YAML files +
     `catalog.yml` manifest. Runs the full publish gate. Output is a
     ready-to-publish tree, but nothing is pushed anywhere.
  2. **Preview** — user runs `imas-standard-names catalog-site serve
     <staging-dir>` to render and browse the proposed catalog in a
     browser before committing to publish. Iterate on the graph side
     and re-export as needed.
  3. `sn publish` — local staging tree → ISNC repo (git commit +
     push or PR). No regeneration at this step; publish is a
     transport operation only, so what the user previewed is exactly
     what lands in the catalog.
- **Lossless round-trip for catalog-owned fields**: `export → clear →
  import` reproduces the YAML's catalog subset exactly on a
  re-export.
- **Graph-only fields preserved on re-import**: coalesce pattern
  protects pipeline metadata when importing onto an existing node.
  A `clear`-then-`import` flow is explicitly destructive by design —
  not in scope to preserve ephemeral pipeline state across
  destructive clears.
- **Graph-retain ≠ catalog-emit**: fields excluded from the catalog
  stay in the graph schema (user rule). Only truly dead fields
  (0/925 populated today) may be schema-dropped, and only after an
  explicit audit.
- **Export is gated**: the gate runs in `sn export` (not `sn
  publish`), because the gate validates the graph state that
  produced the YAML. The YAML on disk between export and publish is
  assumed trustworthy. `sn publish` only verifies that the staging
  directory is well-formed (manifest present, structure valid) and
  that its manifest matches the `catalog.yml` already in the ISNC
  repo (compatibility check).
- **Single catalog COCOS**: catalog manifest declares one
  convention; graph keeps per-name `cocos` FK and `HAS_COCOS` edges
  (useful for queries). Export gate asserts every non-null graph
  `cocos` matches the manifest value.
- **Clean pipeline/vocabulary lifecycle separation**: rename graph
  `review_status` → `pipeline_status` with CLI/MCP alias for one
  release cycle; add catalog-authoritative `status` matching ISN
  enum.
- **Source-agnostic provenance**: `source_paths` (already in graph,
  prefix-encoded `dd:` / `<facility>:`) is the one authoritative
  list; `dd_paths` is retired from the catalog model; import
  reconstructs both `IMASNode` and `FacilitySignal` relationships
  from prefixes.
- **PR-driven round-trip with divergence protection**: the catalog is
  edited by humans via GitHub PRs against ISNC. Each graph node carries
  an `origin` flag (`pipeline` | `catalog_edit`) identifying which side
  last wrote catalog-owned fields. When `origin=catalog_edit`, the
  pipeline is structurally prevented from overwriting those fields
  until an explicit `--override-edits` flag is passed. PR provenance
  (`catalog_pr_number`, `catalog_pr_url`) is extracted from the merge
  commit on import for auditability. See §PR-driven round-trip.

## Scope

In:
- ISN `StandardNameEntry` model rewrite, new
  `StandardNameCatalogManifest` model; coordinated ISN release (new
  rc).
- imas-codex schema delta: rename `review_status` →
  `pipeline_status`; add `status` / `deprecates` / `superseded_by`;
  drop only verified-dead fields after audit.
- **New `sn export` command**: graph → local staging directory
  (YAML + manifest), pre-export gate (graph suites + catalog
  integrity checks), `--min-score` filter.
- **`sn publish` rewrite**: staging dir → ISNC repo git operation
  (commit/push or PR). No graph reads; transport only.
- `sn import` rewrite: validate against new ISN model; derive
  `physics_domain` from relative path; partition `source_paths` by
  prefix to recreate `SOURCE_DD_PATH` / `SOURCE_SIGNAL`
  relationships; recompute `grammar_*` from the name using the same
  helper as the pipeline writes.
- Fold `reconcile` + `resolve-links` into `sn run`; delete
  `sn reconcile`, `sn resolve-links`, `sn seed` standalone commands.
- **ISNC clean break**: on first `sn publish` with the new schema,
  existing ISNC contents are wiped in the publish commit and
  replaced with the freshly exported tree. No in-place migration
  script is written. Rollback is a git revert in ISNC.
- Test suite: unit tests (model/normalizer), plus
  `@pytest.mark.graph` integration tests that reuse the existing
  live-Neo4j gating pattern, plus an export → preview → import
  round-trip test against a test catalog repo.

Out:
- Parametrised-name design (species / population / toroidal_mode /
  flux_surface_average first-class grammar operator) — future plan.
  This plan strips the dead fields; it doesn't redesign grammar.
- Any change to COCOS nodes or `COCOS`-singleton semantics. Graph
  keeps them untouched.
- Vocabulary lifecycle promotion (`draft → active` in the catalog
  repo) — that's a human PR workflow in ISNC, not an automated
  codex transition.
- Preserving the current ISNC directory names (`ic-heating`,
  `thomson-scattering`, …). They will be replaced by ISN
  `PhysicsDomain` enum values on the first new-schema publish; no
  migration tooling is written.

---

## Catalog schema (new)

### Per-name file: `standard_names/<physics_domain>/<name>.yml`

```yaml
# Required identity & semantics
name: electron_temperature
description: Electron kinetic temperature across the plasma.
documentation: |-
  Governing equation: ...
kind: scalar                    # scalar | vector | metadata
unit: eV                        # '' for metadata entries

# Vocabulary lifecycle (ISN-authoritative)
status: draft                   # draft | active | deprecated | superseded
deprecates: null
superseded_by: null

# Classification / cross-references
tags: [core-physics, kinetics]
links:
  - name:electron_density

# Physics semantics
source_paths:                   # source-agnostic, prefix-encoded
  - dd:core_profiles/profiles_1d/electrons/temperature
  - tcv:thomson/temperature_e
validity_domain: null
constraints: []
cocos_transformation_type: null # psi_like | ip_like | b0_like | ...

# Grammatical derivation (derived / composite names only)
provenance:                     # ISN grammatical provenance; NOT pipeline
  mode: operator
  operators: [gradient_of]
  base: electron_temperature
  operator_id: null
```

**Not in per-name YAML** (derived on import):
- `physics_domain` — from relative path `standard_names/<domain>/<name>.yml`
- `grammar_*` (13 fields) — from ISN parser on name string
- Everything else listed under "graph-only" below.

### Catalog-level manifest: `catalog.yml` at **repo root**

```yaml
catalog_name: imas-standard-names-catalog
cocos_convention: 17            # project-wide COCOS (current graph has 867/925 at 17)
grammar_version: 0.8.0          # ISN package version used for emission
isn_model_version: 0.8.0        # StandardNameEntry schema version
dd_version_lineage:             # DD versions this catalog was derived from
  - 4.0.0
generated_by: imas-codex sn publish
generated_at: 2026-04-21T...
# Publish-gate provenance — records the filter applied to the candidate set
min_score_applied: 0.65         # reviewer_score >= this was required for inclusion
min_description_score_applied: null
include_unreviewed: false
candidate_count: 412            # number of SNs that entered the gate
published_count: 389            # number emitted (post gate failures excluded)
excluded_below_score_count: 97  # filtered before gate
excluded_unreviewed_count: 17   # filtered before gate
```

Manifest placed at **repo root** (not inside `standard_names/`) so the
importer's recursive scan of `standard_names/**/*.yml` never mis-parses
it as an entry.

---

## Disposition table (CORRECTED: catalog-drop ≠ schema-drop)

| Graph field | Catalog | Graph schema | Notes |
|---|---|---|---|
| `id` | ✓ (as `name`) | keep | Canonical identifier |
| `description` | ✓ | keep | |
| `documentation` | ✓ | keep | |
| `kind` | ✓ | keep | |
| `unit` | ✓ | keep | |
| `tags` | ✓ | keep | |
| `links` | ✓ | keep | Emit as `name:x` strings |
| `source_paths` | ✓ | keep | Replaces `dd_paths` in catalog |
| `source_types` | ✗ | **keep** | Used by stats/filters; derivable but cached |
| `validity_domain` | ✓ | keep | |
| `constraints` | ✓ | keep | |
| `cocos_transformation_type` | ✓ | keep | Per-name physics semantics |
| `cocos` (FK) + `HAS_COCOS` edge | ✗ | **keep** | Graph-side queries; gated equal to manifest |
| `physics_domain` | derived from path | keep | Authoritative path = `standard_names/<domain>/<name>.yml` |
| `review_status` | ✗ | **rename → `pipeline_status`** | With CLI/MCP alias for one release |
| `status` (new) | ✓ | **add** | ISN lifecycle; catalog-authoritative |
| `deprecates` (new) | ✓ | **add** | Nullable transport field |
| `superseded_by` (new) | ✓ | **add** | Nullable transport field |
| `species`, `population`, `toroidal_mode`, `flux_surface_average` | ✗ | **KEEP (schema contract)** | 0/925 populated today; contract remains |
| Legacy `physical_base`, `subject`, `component`, `coordinate`, `position`, `process` | ✗ | audit first | Drop from schema ONLY if 0/925 populated; otherwise keep graph-side |
| `grammar_*` (13) | ✗ | keep | Recomputed on import via shared helper |
| `confidence`, `model`, `generated_at`, `dd_version` | ✗ | keep | LLM provenance |
| `embedding`, `embedded_at` | ✗ | keep | Recomputable |
| `reviewer_*`, `reviewed_at`, `review_mode`, `review_count`, `review_mean_score`, `review_disagreement` | ✗ | keep | QA state |
| `reviewer_model_secondary`, `reviewer_score_secondary`, `reviewer_scores_secondary`, `reviewer_disagreement` | ✗ | **drop** | Already deprecated in schema |
| `enriched_at`, `enrich_tokens`, `enrich_batch_id` | ✗ | keep | |
| `validation_issues`, `validation_layer_summary`, `validation_status`, `validated_at`, `consolidated_at` | ✗ | keep | Pipeline state |
| `vocab_gap_detail` | ✗ | keep | |
| `link_status`, `link_retry_count`, `link_checked_at` | ✗ | keep | |
| `review_input_hash` | ✗ | keep | |
| `claimed_at`, `claim_token` | ✗ | keep | Worker coordination |
| `last_run_id`, `last_run_at`, `last_turn_number` | ✗ | keep | Run audit |
| `regen_count`, `regen_reason` | ✗ | keep | |
| `imported_at`, `catalog_commit_sha` | ✗ | keep (graph-only) | Set by `sn import`; NOT on ISN entry model (avoids noisy YAML diffs) |
| `catalog_pr_number`, `catalog_pr_url` (new) | ✗ | **add graph-only** | Set by `sn import`; null if not from a merge commit |
| `exported_at` (new) | ✗ | **add graph-only** | Set by `sn export`; NOT on ISN entry model — manifest-level `exported_at` carries run-level provenance |
| `origin` (new) | ✗ | **add graph-only** | `pipeline` \| `catalog_edit`; diff-based flip on import; never serialised to YAML |
| `created_at` | ✗ | keep | |

**Net schema delta (to be audited before finalising):**
- **Rename**: `review_status` → `pipeline_status`.
- **Add**: `status`, `deprecates`, `superseded_by`, `origin`,
  `catalog_pr_number`, `catalog_pr_url`, `exported_at`.
- **Drop** (pending audit, only if 0/925 populated): `reviewer_*_secondary`
  (already marked deprecated), legacy grammar fields if confirmed dead.
- **Keep schema-side (catalog-excluded)**: `cocos` + `HAS_COCOS`,
  `species`, `population`, `toroidal_mode`, `flux_surface_average`,
  `source_types`, all pipeline/QA/worker metadata.

---

## Export gate

Before writing any YAML to the staging tree, `sn export` runs the
following gate. Any failure blocks the export unless `--force` is
passed, in which case failures are written to
`<staging>/.export_gate_report.json` alongside the catalog for
post-hoc review. `sn publish` is not gated (transport only) — it
only runs a structural check that the staging tree is well-formed
(manifest present, no stray files, manifest schema valid).

### A. Reused existing test suites (already live-graph-gated)

| Suite | Marker | Purpose |
|---|---|---|
| `tests/graph/test_sn_unit_integrity.py` | `graph` | SN unit ↔ DD unit agreement |
| `tests/graph/test_grammar_graph_compliance.py` | `graph`, `integration` | Grammar graph matches ISN `SEGMENT_ORDER` |
| `tests/standard_names/test_corpus_health.py` | `corpus_health` | Corpus health gates (dup names, orphan links, unit coverage) |

Export invokes these via an in-process pytest runner with the
`graph or corpus_health` marker expression.

### B. New catalog-integrity checks (added in this plan)

Applied to the **candidate set** (post `--min-score` filter).

| Check | Failure mode |
|---|---|
| Every non-null `sn.cocos` equals `manifest.cocos_convention` | Lists offending `(name, cocos)` pairs |
| Every name with `cocos_transformation_type` set has non-null `cocos`; and vice-versa (XOR is suspicious) | Lists inconsistent rows |
| Every SN node referenced by `links` resolves to another SN node in the graph | Lists dangling `name:x` links |
| Every `deprecates` / `superseded_by` target exists | Lists dangling references |
| Every SN with `validation_status='valid'` has non-null `description`, `documentation`, `unit`, `kind` | Lists incomplete rows |
| No SN has `pipeline_status='drafted'` (only `enriched`, `published`, `accepted` publishable) | Lists unpublished drafts |
| Every candidate has a non-null `reviewer_score` (unless `--include-unreviewed`) | Lists unreviewed candidates |
| Every candidate's `reviewer_score >= --min-score` | Should be empty after filter; sanity check |
| Every SN has at least one `source_path` OR `provenance` (derived names) OR explicit exemption | Lists anchor-less names |
| Schema compliance: no SN node has a property not declared in the LinkML schema | Lists undeclared properties |

### C. Domain scoping

When `sn export` is called with `--domain <d>`, gate **B** scopes
to the named domain; gate **A** runs globally unless
`--gate-scope=domain` is passed. Rationale: partial exports can
still benefit from global integrity checks, but users iterating on
one domain should be able to bypass corpus-wide gates with an
explicit flag.

### D. Gate flags (`sn export`)

- `--force`: write staging tree despite gate failures; emit
  `.export_gate_report.json`.
- `--skip-gate`: skip gate entirely (requires `--force`).
- `--gate-only`: run the gate and report, do not emit.
- `--gate-scope {global,domain}`: default `global`.
- `--min-score <float>`: minimum `reviewer_score` a name must meet to
  be included in the export. Default `0.65` (= tier ≥ `adequate` on
  the 6×0–20 rubric). Names below threshold are excluded from the
  candidate set **before** gate B runs, so the gate only sees the
  names that will actually be published. Names with no `reviewer_score`
  (never reviewed) are excluded unless `--include-unreviewed` is
  passed. `--include-unreviewed` is mutually exclusive with
  `--min-score >0`. The emitted manifest records
  `min_score_applied: <float>` and `candidate_count: <int>` for
  audit. Rationale: the catalog is the public review surface and
  should not expose names whose authors (the pipeline + reviewer)
  already flagged as poor or adequate-only. Running the review loop
  before publish becomes mandatory, enforced by the gate itself
  failing fast if any candidate has `reviewer_score IS NULL`.
- `--min-description-score <float>`: optional secondary threshold on
  the `documentation` sub-score from the 6-dim reviewer rubric (0–20
  scale, default off). Use when a name is acceptable but its
  description is thin — keeps the author in the regeneration loop
  instead of shipping thin docs.

---

## ISN entry model rewrite

```python
# imas_standard_names.models
class StandardNameEntryBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: Name
    description: Description
    documentation: Documentation
    kind: Literal['scalar', 'vector', 'metadata']
    unit: Unit

    status: Literal['draft', 'active', 'deprecated', 'superseded'] = 'draft'
    deprecates: Name | None = None
    superseded_by: Name | None = None

    tags: list[str] = []
    links: list[Link] = []

    source_paths: list[str] = []
    validity_domain: str | None = None
    constraints: list[str] = []
    cocos_transformation_type: str | None = None

    provenance: OperatorProvenance | ExpressionProvenance | ReductionProvenance | None = None


class StandardNameCatalogManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    catalog_name: str
    cocos_convention: int
    grammar_version: str
    isn_model_version: str
    dd_version_lineage: list[str]
    generated_by: str
    generated_at: datetime
```

**Breaking changes vs current ISN 0.7.x:**
- `dd_paths` → `source_paths` (rename + semantics broaden)
- `physics_domain` field removed (derived from path on load)
- `provenance` meaning unchanged (grammatical); add `StandardNameCatalogManifest`

**Coordinated release strategy** (BLOCKER 10 from RD v1):
Dual-loader in ISN 0.8.0 reads both `dd_paths` (legacy) and
`source_paths` (new), normalising to `source_paths` internally. ISN
0.9.0 drops `dd_paths` support entirely. This gives codex and
imas-standard-names-catalog independent migration windows.

---

## Pipeline / vocabulary status consistency

### Graph field: `pipeline_status` (renamed from `review_status`)

Values (from current live graph): `named`, `enriched`, `drafted` —
and the broader documented set `published`, `accepted`, `vocab_gap`.
This is **pipeline state** only.

**CLI/MCP compat layer**: for one release window, `review_status` is
accepted as input (mapped to `pipeline_status`) and reported under
both names in output. Removal scheduled for the following release.
Every call site is enumerated in the migration checklist (phase 2c).

### Graph + catalog field: `status` (new)

Values: `draft | active | deprecated | superseded` (ISN-authoritative).

**Rules** (BLOCKER 3 from RD v1):
- **Catalog owns `status`.** Codex never promotes vocabulary lifecycle.
- On publish: emit graph `status` directly. If null, default `'draft'`.
- On import: catalog `status` overwrites graph `status`.
- Backfill once at migration time: `NULL → 'draft'` for all existing
  graph rows. **No mapping from `review_status` to `status`.** Pipeline
  state and vocabulary lifecycle are orthogonal.
- Promotion from `draft → active` is a human PR in imas-standard-names-catalog,
  not an automated transition.

---

## PR-driven round-trip and provenance

### Canonical round-trip flow

```
imas-codex graph ──► sn export ──► staging dir ──► sn publish ──► ISNC main
                                                                   │
                                                                   │ (human fork,
                                                                   │  edits foo.yml,
                                                                   │  opens PR)
                                                                   ▼
                                                          ISNC PR reviewed + merged
                                                                   │
                                                                   ▼
imas-codex graph ◄── sn import ◄──────────────────────── ISNC main (post-merge)
```

Two authorship channels feed the graph:
1. **Pipeline** — `sn run` writes new or regenerates existing names from DD/signals.
2. **Catalog edit** — human PRs to ISNC (typo fixes, description polish, status promotions). `sn import` pulls the merged edits back.

Without provenance tracking, the pipeline will silently overwrite human edits on the next `sn export`. This must not happen.

### `origin` state machine

New graph-only field, enum `OriginType` (NOT serialised to catalog YAML — the YAML is by definition the `catalog_edit` side):

```
           ┌─────────────┐
 created ─►│  pipeline   │
           └──────┬──────┘
                  │  sn import, AND at least one catalog-owned field
                  │  differs from the current graph value after normalisation
                  ▼
           ┌─────────────┐
           │catalog_edit │◄── sn import (diff still non-empty)
           └──────┬──────┘
                  │
                  │  sn run --override-edits <name>  (explicit re-take)
                  │  OR sn clear followed by sn run
                  ▼
            pipeline
```

**Diff-based flip rule (HIGH fix from RD v2)**: `sn import` flips `origin` to `catalog_edit` **only when the imported catalog-owned field set differs from the current graph values after normalisation** (whitespace-trim, list-sort for unordered lists, YAML round-trip). No-op imports (e.g. the first import after a clean-break export where YAML matches graph exactly) preserve `origin=pipeline`. This makes Phase 8 acceptance consistent: regenerating ISNC from the graph and immediately re-importing is a no-op that does not lock the catalog.

Semantics:
- `origin=pipeline` — pipeline owns **all** protected fields; regeneration is free.
- `origin=catalog_edit` — **protected fields are read-only** for all pipeline writers (see §Pipeline protection enforcement). Graph-only fields flow freely.

### PR provenance extraction

On `sn import`, extract merge-commit metadata from the ISNC checkout.

**First-parent walk (HIGH fix from RD v2)** — squash and rebase merges are NOT merge commits, so `git log --merges` misses them entirely. Use a first-parent walk of commits reachable from HEAD but newer than the previous watermark SHA:

```
git log --first-parent --format='%H%x00%s%x00%b' <watermark_sha>..HEAD -- <file>
```

For each commit, try in order:
1. Subject regex `r"Merge pull request #(\d+) from"` — classic merge commit.
2. Subject regex `r"\(#(\d+)\)(?:\s|$)"` — GitHub squash merge default (`<PR title> (#123)`).
3. Body regex `r"^(?:Closes|Fixes|Refs|Related|PR)\s*:?\s*#(\d+)"` — rebase merges via trailers.
4. **Fallback**: if a `GITHUB_TOKEN` is in the env and `.git` is present, query `GET /repos/{owner}/{repo}/commits/{sha}/pulls` for each first-parent commit in the range. This handles any merge strategy correctly.

If none of the above yields a PR number, set `catalog_pr_number`/`catalog_pr_url` to null and log a `PR provenance unavailable` warning — but still stamp `origin=catalog_edit` (if diff non-empty) and `catalog_commit_sha`.

**Tarball import** (`.git` absent): skip PR extraction, set only `imported_at` and `catalog_commit_sha` (if user passes `--commit-sha`).

### Import concurrency: lock + CAS watermark (HIGH fix from RD v2)

Two agents on different machines can both run `sn import` concurrently against the same graph. Without synchronisation they would each diff from the same old watermark and race-update it. Mitigation:

1. **Lock**: singleton `(l:ImportLock {id: 'singleton'})` node. Acquire with:
   ```cypher
   MERGE (l:ImportLock {id: 'singleton'})
   ON CREATE SET l.held = true, l.holder = $holder, l.acquired_at = datetime()
   ON MATCH SET l.held = CASE WHEN NOT l.held THEN true ELSE l.held END,
                l.holder = CASE WHEN NOT l.held THEN $holder ELSE l.holder END,
                l.acquired_at = CASE WHEN NOT l.held THEN datetime() ELSE l.acquired_at END
   RETURN l.holder = $holder AS acquired
   ```
   If `acquired=false`, the other holder is still running; abort with a friendly message.
2. **Compare-and-set watermark update**: on import completion, advance `ImportWatermark.last_commit_sha` only if it still equals the SHA observed at import start:
   ```cypher
   MATCH (w:ImportWatermark {id: 'singleton'})
   WHERE w.last_commit_sha = $expected_prev_sha
   SET w.last_commit_sha = $new_sha, w.last_imported_at = datetime()
   RETURN w.last_commit_sha
   ```
   If the match returns no rows, another importer moved the watermark between our start and finish — abort with clear diagnostic, release the lock, do not stamp any nodes. Caller retries.
3. **Stale-lock recovery**: if `l.acquired_at` is older than a configurable timeout (default 30 min), a subsequent `sn import` may break the lock after printing a warning.
4. **Partial failure**: watermark advances only after all names are stamped AND the transaction commits. If the import fails mid-way, watermark stays at previous SHA — next import re-diffs the same range, idempotent by diff-based `origin` flip.

### Divergence detection in `sn export`

Before writing the staging tree, `sn export` computes an invariant-check report: for each graph SN with `origin=catalog_edit`, compare the graph's catalog-owned fields against the catalog YAML at `imported_catalog_commit_sha` (if ISNC checkout is git and that SHA is reachable). Any difference is an INVARIANT_VIOLATION — it means some pipeline write path bypassed protection. This is a hard-fail (not maskable by `--force` or `--skip-gate`) because it indicates a bug, not a policy choice. The `--override-edits <name>` flag resets affected names to `pipeline` and re-exports them cleanly.

### Pipeline protection enforcement (HIGH fix from RD v2: not a single choke-point)

Audit of current writers shows catalog-owned fields are written from multiple call sites:

| Writer | File:line | Touches |
|---|---|---|
| `write_standard_names()` | `graph_ops.py` (build path) | all fields (coalesce) |
| `persist_enriched_batch()` | `graph_ops.py:1222` | `description`, `documentation`, `tags` |
| `write_enrichment_results()` | `graph_ops.py:2406` | `description`, `documentation`, `tags`, `links`, `validity_domain`, `constraints` |
| `resolve_links_batch()` | `graph_ops.py:2233` | `links` |

Protection is implemented once in a shared helper, applied at every writer:

```python
# imas_codex/standard_names/graph_ops.py
PROTECTED_FIELDS: frozenset[str] = frozenset({
    "description", "documentation", "kind", "tags", "links",
    "status", "deprecates", "superseded_by",
    "validity_domain", "constraints",
})

def filter_protected(items: list[dict], *, override: bool) -> tuple[list[dict], list[str]]:
    """Strip PROTECTED_FIELDS from each item whose graph origin=catalog_edit.
    Returns (filtered_items, skipped_name_list). Override bypasses.
    """
    if override:
        return items, []
    # Single Cypher fetch of origin per name in batch:
    names = [it["name"] for it in items]
    protected = set(query("MATCH (sn:StandardName) "
                          "WHERE sn.name IN $names AND sn.origin = 'catalog_edit' "
                          "RETURN sn.name AS name", names=names))
    filtered = []
    skipped = []
    for it in items:
        if it["name"] in protected:
            stripped = {k: v for k, v in it.items() if k not in PROTECTED_FIELDS}
            if stripped != it:
                skipped.append(it["name"])
            filtered.append(stripped)
        else:
            filtered.append(it)
    return filtered, skipped
```

Applied at entry of **every writer listed above** (not a single choke-point). Each writer accepts `override: bool = False` kwarg; `sn run --override-edits <name>` propagates this to the active batch.

### Catalog-owned field classification (HIGH fix from RD v2)

Reclassified per RD finding #5:

| Field | Classification | Reason | On catalog edit |
|---|---|---|---|
| `description` | **catalog-owned** | Editorial prose; humans polish | Protected; pipeline cannot overwrite |
| `documentation` | **catalog-owned** | Editorial prose | Protected |
| `kind` | **catalog-owned** | Controlled vocab; review decision | Protected |
| `tags` | **catalog-owned** | Editorial grouping | Protected |
| `links` | **catalog-owned** | Curation layer | Protected |
| `status` | **catalog-owned** | Lifecycle promotion is a review decision | Protected |
| `deprecates`, `superseded_by` | **catalog-owned** | Curation relationship | Protected |
| `validity_domain`, `constraints` | **catalog-owned** | Editorial annotation | Protected |
| `unit` | **validated on import** | DD/physics-authoritative; LLM never writes it anyway | Import rejects mismatch with DD unless `--accept-unit-override` is passed; not on the `origin` protection list |
| `cocos_transformation_type` | **validated on import** | Physics-authoritative; manifest-level COCOS is already gated | Import rejects mismatch with graph COCOS FK unless override |
| `source_paths` | **pipeline-owned** | Provenance of where this name was extracted from; tracking DD/signal reality, not editorial | Import UNIONs pipeline values with catalog values (humans may add manual refs); no protection needed |
| `cocos` (FK) | **graph-only** | Not serialised to catalog | N/A |

### Should we track this metadata?

**Yes**, for three reasons:
1. **Correctness** — without `origin`, pipeline silently clobbers human edits. Primary use-case of the catalog; must work.
2. **Auditability** — `(origin, catalog_pr_url)` answers "who last wrote this and in which PR?" in one line.
3. **Cost trivial** — 4 nullable fields per node (~3700 cells); no index needed.

**Non-goals**:
- No GitHub actor/author identity (GDPR risk; it's in the PR URL).
- No full commit chain per name (most recent merge commit only; older provenance is in git history).

---

## Source-agnostic relationship writes

BLOCKER 4 from RD v1. On import, `catalog_import._import_relationships`
partitions `source_paths` by prefix:

```python
for sp in source_paths:
    if sp.startswith('dd:'):
        # MERGE (sn)-[:SOURCE_DD_PATH]->(dd:IMASNode {id: sp[3:]})
    elif ':' in sp:
        facility, signal_id = sp.split(':', 1)
        # MERGE (sn)-[:SOURCE_SIGNAL]->(fs:FacilitySignal {id: sp})
    else:
        # log warning; strict mode: fail
```

Publish emits `source_paths` verbatim from the graph property (no
relationship reconstruction needed — the property is the source of
truth for ordering and completeness).

---

## Grammar recomputation on import

SERIOUS 5 from RD v1. On import, `catalog_import` calls the same
`grammar_ops.decompose_name(name)` helper that the pipeline write-path
uses (`graph_ops.py:50-78`). This populates all 13 `grammar_*` fields
from the canonical ISN parser.

**Parse-failure policy**: hard-fail the import for that name, record
the failure in a batch error report, continue with other names. An
unparseable name in a catalog file is a bug upstream — the catalog
should not silently land malformed entries.

---

## CLI surface after this plan

| Tool | Change |
|------|--------|
| `sn run` | **Folds in reconcile (pre-extract) + resolve-links (post-persist, pre-review).** `--skip-reconcile`, `--skip-resolve-links`, `--only {reconcile,resolve-links,...}` flags. |
| `sn review` | Unchanged. |
| `sn clear` | Unchanged. |
| `sn status` | Unchanged. |
| `sn export` | **New**. Graph → local staging dir. Runs gate A+B. Flags: `--output <dir>` (default `./build/catalog/`), `--min-score <float>` (default 0.65), `--include-unreviewed`, `--min-description-score <float>`, `--domain <d>`, `--force`, `--skip-gate`, `--gate-only`, `--gate-scope {global,domain}`. Writes `<output>/standard_names/<domain>/<name>.yml` + `<output>/catalog.yml`. |
| `sn preview` | **New, thin wrapper**. Delegates to `imas-standard-names catalog-site serve <staging-dir>` with the staging dir as the default argument. Exists so users don't have to switch CLIs; ISN owns the rendering. Flags pass-through. |
| `sn publish` | **Rewritten as transport-only**: staging dir → ISNC repo. No graph reads, no gate (gate ran at export). Flags: `--staging <dir>` (default `./build/catalog/`), `--catalog-repo <path-or-url>`, `--mode {commit,pr}` (default `commit` for a local checkout, `pr` for a remote URL), `--branch <name>`, `--message <str>`, `--dry-run`. Structural check: manifest present + parseable, no stray files outside `standard_names/` and `catalog.yml`, manifest schema valid. |
| `sn import` | **Rewritten**: strict ISN validation, path-derived `physics_domain`, prefix-partitioned relationship writes, shared grammar helper, `--check` validates without writing. |
| `sn benchmark` | Unchanged. |
| `sn gaps` | Unchanged. |
| `sn seed` | **Deleted** (throwback from early dev). |
| `sn reconcile` | **Deleted** (folded into `run`). |
| `sn resolve-links` | **Deleted** (folded into `run`). |

Final surface: **10 commands** (was 11; split publish into export + publish + preview-wrapper, deleted 3 legacy commands, net −1).

### Two-step flow in practice

```
# 1. Export (graph → local, gated)
uv run imas-codex sn export --output ./build/catalog/

# 2. Preview (local → browser, via ISN machinery)
uv run imas-codex sn preview ./build/catalog/
#   ↳ equivalent to: uv run imas-standard-names catalog-site serve ./build/catalog/

# 3. Publish (local → ISNC git repo)
uv run imas-codex sn publish \
    --staging ./build/catalog/ \
    --catalog-repo ~/Code/imas-standard-names-catalog \
    --mode commit \
    --message "catalog: republish from graph @ $(cd ~/Code/imas-codex && git rev-parse --short HEAD)"
```

The two-step split guarantees that what the user saw in the preview
is exactly what lands in the catalog repo. The export tree is a
committable artefact; `sn publish` does not re-read the graph.

---

## Round-trip test strategy

SERIOUS 8 from RD v1. Split into unit + gated integration, normalised
comparisons (not byte-for-byte).

### Unit tests (no Neo4j; CI-default)
- `tests/standard_names/test_publish_yaml_shape.py`:
  - Given a mocked graph record, `generate_yaml_entry` produces a dict
    that validates against `StandardNameEntry` (strict, `extra='forbid'`).
  - Every catalog-owned field maps 1:1 from graph record.
  - No graph-only field leaks into the emitted dict.
- `tests/standard_names/test_publish_manifest.py`:
  - Manifest model validates; all required fields set; path is
    `catalog.yml` at repo root.
- `tests/standard_names/test_catalog_import_normalizer.py`:
  - Given an ISN `StandardNameEntry` instance, the normalizer produces
    the graph merge payload: catalog fields overwrite, graph-only
    keys absent (coalesce-friendly).
  - `physics_domain` derived from path input.
  - Prefix partition produces correct relationship writes.
  - Grammar helper is called with the name.

### Integration tests (`@pytest.mark.graph`; gated on live Neo4j)
- `tests/graph/test_catalog_roundtrip.py`:
  - Seed 10 hand-curated SN nodes (all three kinds, with
    `source_paths`, `cocos_transformation_type`, `deprecates`, `links`,
    `validity_domain`, `constraints`, nontrivial documentation with
    `name:x` links, populated graph-only fields like `reviewer_score`).
  - Run publish → tmpdir. Validate every YAML against `StandardNameEntry`.
  - Re-publish from an existing node (update path): assert
    graph-only fields unchanged, catalog-owned fields match YAML input.
  - Delete only `StandardName` nodes, preserving `COCOS`, `Unit`,
    `IMASNode`, `FacilitySignal`.
  - Run import from tmpdir. Re-read nodes: assert catalog-owned fields
    match original dump (via `model_dump(mode='json')` comparison).
  - Run import a second time (idempotence): same catalog fields,
    updated `imported_at` / `catalog_commit_sha` only.
  - Seed a single node with populated graph-only fields, run import
    of the same name, assert all graph-only fields unchanged.
- `tests/graph/test_publish_gate.py`:
  - Inject a non-matching `cocos` → gate blocks with clear diagnostic.
  - Inject a `name:ghost` dangling link → gate blocks.
  - Healthy graph → gate passes.
  - `--force` bypasses; `.publish_gate_report.json` written.

---

## Phases

### Multi-agent coordination

This plan spans **three repositories**. Phase ordering reflects cross-repo dependencies.

**Dependency graph (corrected per RD v2)**:

```
Phase 0 (codex audit)               Phase 1 (ISN rewrite + rc cut)
    │                                       │
    ▼                                       ▼
    └──────────────► Phase 2 ◄── Phase 1.5 (codex dep bump)
                    (schema + protection
                     helpers + lock)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Phase 3 backend   Phase 4 backend   Phase 5 backend
   (export.py +      (catalog_import    (turn.py changes)
   new publish.py)    rewrite)
        │                 │                 │
        └─────────────────┴─────────────────┘
                          ▼
               Phase 3/4/5 CLI integration
              (serialised — single cli/sn.py agent)
                          │
                          ▼
                    Phase 6 (tests)
                          │
                          ▼
                    Phase 7 (docs)
                          │
                          ▼
                Phase 8 (ISNC clean-break regen)
                 — preview site BEFORE open PR
```

**Key corrections from RD v2**:

- **Phase 2 blocks Phase 3/4/5** (not just Phase 1.5). Phase 3/4/5 backends need
  the schema rename, new fields, lock/watermark helpers, AND the shared
  `filter_protected()` helper.
- **CLI integration is serialised**, not parallel. Phases 3/4/5 edit the
  same file (`imas_codex/cli/sn.py`) and would produce repeated merge
  conflicts. Split is:
  - Backend module work runs in parallel (three agents on three different
    files: `export.py`/`publish.py`, `catalog_import.py`, `turn.py`).
  - After all three backend branches land, a single follow-up agent
    integrates all CLI changes in one pass on `cli/sn.py`.
  - Alternatively (optional pre-work): split `cli/sn.py` into per-command
    modules first, then backend + CLI can proceed per-command in parallel.
- **Phase 2 is hard-blocked on Phase 0** — 2a/2c need the written audit
  report. Preparatory non-dropping edits can start early; schema-drop
  Cypher waits.

**Repo ownership per phase**:

| Phase | Repo | Agent type | Parallel-with |
|---|---|---|---|
| 0 | imas-codex | @engineer (audit, read-only queries) | Phase 1 |
| 1 | imas-standard-names | @architect (schema rewrite) | Phase 0 |
| 1.5 | imas-codex | @engineer (pyproject pin + build-models + sync) | — (blocks on 1) |
| 2 | imas-codex | @architect (schema migration + protection helpers + lock; spans 10+ files) | — (blocks on 0 + 1.5) |
| 3 backend | imas-codex | @architect (export + gate + divergence) | Phase 4, 5 backends |
| 4 backend | imas-codex | @architect (import + PR extraction + diff-flip + lock) | Phase 3, 5 backends |
| 5 backend | imas-codex | @engineer (turn.py fold) | Phase 3, 4 backends |
| 3/4/5 CLI | imas-codex | @engineer (cli/sn.py integration in one pass) | — (serial after all backends) |
| 6 | imas-codex | @engineer (round-trip + concurrency tests) | — |
| 7 | 3 repos | @engineer (docs) | — |
| 8 | ISNC | @engineer (clean-break regen + PR, after preview verified) | — |

### Phase 0 — Codex-side field audit (parallel with Phase 1)

Purpose: decide which fields to schema-drop vs keep (dead-but-retained).

0a. Query graph for population counts on every candidate-drop field:
    `physical_base`, `subject`, `component`, `coordinate`, `position`, `process`, `reviewer_*_secondary`, plus the 4 explicitly listed dead fields (`species`, `population`, `toroidal_mode`, `flux_surface_average`).
0b. Produce `files/schema-drop-audit.md` with counts + recommendation per field.
0c. **Acceptance**: report committed to session files; ready for Phase 2 consumer.

### Phase 1 — ISN schema rewrite + rc cut (ISN repo)

Files: `~/Code/imas-standard-names/src/imas_standard_names/models.py`, related validators/tests.

1a. Rewrite `StandardNameEntry{Scalar,Vector,Metadata}` per §ISN rewrite:
    - Replace `dd_paths` with `source_paths`.
    - Add `status`, `deprecates`, `superseded_by`, `cocos_transformation_type`.
    - **Do NOT** add per-entry provenance fields (`exported_at`, `imported_at`,
      `catalog_commit_sha`, `catalog_pr_number`, `catalog_pr_url`, `origin`) to
      the ISN model — these are all graph-only (MED fix from RD v2). Per-entry
      timestamps in catalog YAML create noisy diffs on every export. Generation
      metadata belongs **only** in the manifest (see 1b).
    - Drop: `dd_paths` (renamed), species/population/toroidal_mode/flux_surface_average (never were on the ISN model; confirm absent).
1b. Add `StandardNameCatalogManifest` model. Manifest carries ONLY run-level
    metadata: `schema_version`, `cocos`, `grammar_version`, `exported_at`,
    `min_score_applied`, `published_count`, `candidate_count`,
    `excluded_below_score_count`, `excluded_unreviewed_count`, `source_repo`,
    `source_commit_sha`.
1c. Loader: accept `source_paths` only (clean break; no dual-loader — ISNC will be wiped and regenerated in Phase 8).
1d. ISN tests: round-trip model dump/load; manifest construction.
1e. Docs: update ISN README with new schema.
1f. **Release**: `cd ~/Code/imas-standard-names && uv run standard-names release --bump minor -m "feat: rewrite StandardNameEntry for catalog schema v2"`. This cuts an rc (e.g. `v0.8.0rc1`) on origin (fork).
1g. **RC rollback (LOW fix from RD v2)**: if Phase 1.5 smoke test fails, patch
    ISN on `main` and cut `v0.8.0rc2` via `standard-names release -m "..."`.
    Do NOT retag rc1. Codex `pyproject.toml` ISN dep pin advances to the
    latest passing rc. RCs are disposable on the fork.
1h. **Acceptance**: rc tag visible on ISN origin; ISN tests green on fork CI.

### Phase 1.5 — Codex ISN dep bump + model regen (imas-codex)

**Blocking on Phase 1 rc being available on PyPI or via direct git ref.**

1.5a. In `pyproject.toml`: pin ISN dep to the new rc version.
1.5b. `uv sync --extra test`.
1.5c. `uv run imas-codex build-models --force`.
1.5d. Quick smoke: `uv run python -c "from imas_standard_names.models import StandardNameEntryScalar; print(StandardNameEntryScalar.model_fields.keys())"` — confirm new fields.
1.5e. Commit: `chore(deps): bump imas-standard-names to 0.8.0rcN for catalog schema v2`.
1.5f. **Acceptance**: import succeeds; `source_paths` present; `dd_paths` absent; green worktree ready for Phase 2 dispatch.

### Phase 2 — Graph schema migration (imas-codex)

Files: `imas_codex/schemas/standard_name.yaml`, `imas_codex/standard_names/graph_ops.py`, `imas_codex/standard_names/models.py`, call-sites of `review_status`.

**BLOCKED on Phase 0 audit (MED fix from RD v2)** — 2a/2c require the audit
report to know which fields to drop. Preparatory, non-dropping work (field
renames, new-field additions) may start early, but schema-drop edits wait on
Phase 0's written recommendation.

2a. Edit LinkML schema:
    - Rename `review_status` → `pipeline_status`.
    - Add `status` (enum: draft/active/deprecated/superseded; default draft).
    - Add `deprecates` (list[str]), `superseded_by` (str|null).
    - Add `origin` (enum: pipeline/catalog_edit; default pipeline).
    - Add `catalog_pr_number` (int|null), `catalog_pr_url` (str|null), `exported_at` (datetime|null).
    - Drop fields confirmed dead by Phase 0 audit (including `reviewer_*_secondary`).
    - Keep per §disposition: `cocos`, `HAS_COCOS`, `species`, `population`, `toroidal_mode`, `flux_surface_average`, `source_types`.
    - Add `ImportWatermark` node type (singleton; `last_commit_sha`, `last_imported_at`, `repo_url`).
    - Add `ImportLock` node type (singleton; `held`, `holder`, `acquired_at`).
2b. `uv run build-models --force`.
2c. Inline Cypher migration via `graph shell`:
    ```cypher
    MATCH (sn:StandardName)
    SET sn.pipeline_status = sn.review_status
    REMOVE sn.review_status
    SET sn.status = coalesce(sn.status, 'draft')
    SET sn.origin = coalesce(sn.origin, 'pipeline')
    // schema-drop REMOVEs per Phase 0 audit
    ```
2d. Update all call sites: `grep -rn "review_status" imas_codex/` → rename.
2e. CLI + MCP parameter alias: `--review-status` emits DeprecationWarning but still accepts old value during one release window.
2f. Implement shared `PROTECTED_FIELDS` + `filter_protected()` helper in
    `graph_ops.py` (see §Pipeline protection enforcement). Apply at entry of
    **every** writer listed in the audit table (HIGH fix from RD v2):
    - `write_standard_names()` (build path)
    - `persist_enriched_batch()` (enrichment path)
    - `write_enrichment_results()` (enrichment batch write)
    - `resolve_links_batch()` (link resolution)
    Each writer accepts `override: bool = False`; CLI surfaces via
    `sn run --override-edits`.
2g. Implement graph-backed `acquire_import_lock()` / `release_import_lock()`
    and CAS-based `advance_watermark()` helpers in a new
    `imas_codex/standard_names/import_sync.py`.
2h. Tests:
    - `test_origin_protection_write.py` — `write_standard_names` drops protected fields when `origin=catalog_edit`.
    - `test_origin_protection_enrich.py` — `persist_enriched_batch` + `write_enrichment_results` both respect protection (HIGH fix from RD v2).
    - `test_origin_protection_links.py` — `resolve_links_batch` respects `links` protection.
    - `test_origin_protection_override.py` — `override=True` bypasses all four writers.
    - `test_import_lock.py` — concurrent acquire-attempt correctness; stale-lock break.
    - `test_import_watermark_cas.py` — CAS rejects stale-SHA update (HIGH fix from RD v2).
2i. **Acceptance**: `uv run pytest tests/graph/` green; migration Cypher shows expected row counts.

### Phase 3 — `sn export` + publish gate + divergence report (imas-codex)

Files: new `imas_codex/standard_names/export.py`, rewritten `imas_codex/standard_names/publish.py` (transport), `imas_codex/cli/sn.py` (new `export`, new `preview`, rewritten `publish`).

**BLOCKED on Phase 2 (MED fix from RD v2)** — depends on schema migration
AND on the shared protection helper being available. CLI integration is
serialised with Phase 4 and Phase 5 to avoid `cli/sn.py` merge conflicts
(see §Multi-agent coordination update below).

3a. Implement `sn export` (replaces current `publish` yaml generation):
    - Reads graph → writes staging dir `<staging>/standard_names/<domain>/<name>.yml` + `<staging>/catalog.yml`.
    - Emits `.yml` (not `.yaml`).
    - Applies gate A (graph tests) + B (cross-field consistency: COCOS, grammar-version, source_paths resolve) + C (`--min-score`, `--include-unreviewed`, `--min-description-score`) + D (divergence detection — see §PR-driven round-trip).
    - Flags: `--staging <dir>` (required), `--min-score <float>` (default 0.65), `--include-unreviewed`, `--min-description-score <float>`, `--force`, `--skip-gate`, `--gate-only`, `--gate-scope`, `--domain`, `--override-edits <name>...` (per-name; or `--override-edits all` explicit opt-in).
    - Writes `<staging>/.export_report.json` with gate results, divergence report, name counts per filter.
3b. Implement `sn preview` as a thin wrapper that calls ISN `catalog-site serve` on the staging dir.
3c. Rewrite `sn publish` to transport only:
    - Input: staging dir from 3a.
    - Action: mirror into ISNC checkout, commit, push to origin (fork) or upstream per existing release conventions.
    - NO gate logic (already run at export).
    - Flags: `--isnc <path>`, `--push`, `--dry-run`.
3d. Tests: gate pass/fail for each of A/B/C/D; divergence-detection unit test; export→staging→publish round trip (staging intermediates assertable); `test_source_paths_roundtrip.py` — mixed DD + signal refs preserve ordering and partitioning across export→import.
3e. **Acceptance**: `sn export`, `sn preview`, `sn publish` all functional independently; gate blocks catastrophic states; divergence report included in `.export_report.json`.

### Phase 4 — `sn import` rewrite + PR extraction (imas-codex)

Files: `imas_codex/standard_names/catalog_import.py`, `imas_codex/cli/sn.py` (`import` command).

**BLOCKED on Phase 2** (needs schema + lock/watermark helpers).
CLI integration serialised with Phase 3 and Phase 5.

4a. Accept new schema (no `dd_paths` fallback — clean break; Phase 8 regenerates ISNC).
4b. Derive `physics_domain` from file path `standard_names/<domain>/<name>.yml`; refuse mismatches with loud error.
4c. Call shared `decompose_name(name)` to repopulate `grammar_*`.
4d. Partition `source_paths` by prefix (UNION with existing graph values per
    §Catalog-owned field classification — `source_paths` is pipeline-owned
    and merges rather than overwrites):
    - `dd:<path>` → re-link via `SOURCE_DD_PATH` to `IMASNode`.
    - `signal:<facility>:<id>` → `SOURCE_SIGNAL` to `FacilitySignal`.
    - Unknown prefixes logged + skipped (not fatal).
4e. **Diff-based `origin` flip** (HIGH fix from RD v2): after normalising
    catalog-owned fields in the incoming entry (whitespace-trim, list-sort),
    compare against current graph values. Stamp `origin=catalog_edit` only
    when at least one protected field differs. No-op imports preserve
    `origin=pipeline`.
4f. Validate `unit` against DD value for DD-sourced names; reject mismatch
    unless `--accept-unit-override` is passed (refers to §Catalog-owned
    field classification reclassification).
4g. Validate `cocos_transformation_type` against graph COCOS FK; reject
    mismatch unless `--accept-cocos-override`.
4h. Acquire import lock + read expected-prev-SHA from `ImportWatermark` at
    start. Walk first-parent commits from `<watermark>..HEAD` to extract PR
    metadata per §PR provenance extraction (squash + rebase merges covered).
    On completion, CAS-advance the watermark. Release lock in a `try/finally`.
4i. Tests:
    - `test_import_stamps_origin_diff.py` — diff-based flip (no-op preserves `pipeline`; edit flips to `catalog_edit`).
    - `test_import_pr_extraction_merge.py` — classic merge commit.
    - `test_import_pr_extraction_squash.py` — squash merge `(#123)`.
    - `test_import_pr_extraction_rebase.py` — rebase merge via trailer (HIGH fix from RD v2).
    - `test_import_pr_extraction_null.py` — direct push with no PR reference.
    - `test_import_watermark_lock.py` — two concurrent imports; second aborts cleanly.
    - `test_import_partial_failure.py` — import fails mid-way; watermark does NOT advance; re-run is idempotent (HIGH fix from RD v2).
    - `test_import_unit_mismatch.py` / `test_import_cocos_mismatch.py` — rejection unless override flag.
4j. **Acceptance**: importing a seeded ISNC checkout flips `origin` correctly by diff; pipeline-only fields untouched; PR numbers extracted for all three merge styles; concurrent imports safe.

### Phase 5 — Fold `reconcile` + `resolve-links` into `sn run`, delete `seed` (imas-codex)

Files: `imas_codex/standard_names/turn.py`, `imas_codex/cli/sn.py`, delete `imas_codex/standard_names/seed.py`.

**BLOCKED on Phase 2** (needs protection-aware writers). CLI integration
serialised with Phase 3 and Phase 4.

5a. `turn.py`: add `reconcile` pre-extract phase honouring `--source {dd,signals}` scope.
5b. `turn.py`: add `resolve-links` after persist, before review; multi-round; scoped to this turn's touched names (not global sweep).
5c. `sn run` flags: `--skip-reconcile`, `--skip-resolve-links`, `--only {extract,compose,validate,consolidate,persist,review,resolve-links,reconcile}`, `--override-edits <name>...` (plumbed through to protected writers).
5d. Delete `sn reconcile`, `sn resolve-links`, `sn seed` CLI verbs.
5e. Delete `imas_codex/standard_names/seed.py`.
5f. Tests: `sn run --only resolve-links` touches only named-this-turn nodes; `--skip-reconcile` skips cleanly; `--override-edits` bypasses protection on the listed names only.
5g. **Acceptance**: `sn --help` shows no `reconcile`/`resolve-links`/`seed`; functionality reachable via `sn run`; CI green.

### Phase 6 — Consolidated test pass (imas-codex)

Runs after Phases 2–5 all merge.

6a. Full round-trip: seed small fixture graph → `sn export <staging>` → manual YAML tweak in staging → `sn publish` to mock ISNC → `sn import` from mock ISNC → assert `origin=catalog_edit` + edit preserved.
6b. Regression: rerun pipeline (`sn run`) after import; assert edited `description` unchanged (protection working); assert embedding refreshed (graph-only update passes through).
6c. Divergence injection: manually bypass protection → assert `sn export` divergence report flags it.
6d. Gate matrix: all permutations of `--min-score`, `--include-unreviewed`, `--min-description-score`, `--force`.
6e. PR extraction: fixture git repo with merge commit, squash commit, direct push; assert extraction correctness on each.
6f. **Acceptance**: full `pytest tests/standard_names/ tests/graph/` green.

### Phase 7 — Documentation

7a. `AGENTS.md`:
    - CLI table: `sn export`, `sn preview`, `sn publish`, `sn import`. Drop `seed`, `reconcile`, `resolve-links`.
    - Schema: note `pipeline_status` vs `status`; introduce `origin`.
    - §PR-driven round-trip section summarising the protection model and override mechanism.
7b. ISN README: new `StandardNameEntry` schema + manifest.
7c. ISNC README: clean-break notice; layout (`standard_names/<domain>/<name>.yml`); PR workflow; `--min-score` gate context; link to preview site.
7d. `plans/README.md`: move plan 35 to pending/ or delete on full implementation.

### Phase 8 — ISNC clean-break regeneration (imas-standard-names-catalog)

**Blocking on Phases 2–7 merged.**

8a. Bump ISNC's ISN dep to 0.8.0 (or final of rc series after RC bake-out).
8b. Remove all existing `.yml` entries in ISNC (clean break — graph is authoritative).
8c. `sn export --staging /tmp/isnc-staging/ --min-score 0.65` from imas-codex.
8d. **Preview before PR (MED fix from RD v2)**: `sn preview /tmp/isnc-staging/`
    (thin wrapper → `catalog-site serve`). Smoke-check rendered pages locally;
    iterate on graph side + re-export if any category renders incorrectly.
    ONLY after preview looks good, proceed to 8e.
8e. `sn publish --staging /tmp/isnc-staging/ --isnc ~/Code/imas-standard-names-catalog --push`.
8f. Open ISNC PR: "feat!: catalog schema v2 — clean-break regeneration from imas-codex graph at schema v2".
8g. Verify GitHub Pages deploy of preview via ISN `catalog-site deploy` (triggered by ISNC CI on merge).
8h. **Acceptance**: ISNC main reflects new schema; preview site renders; first `sn import` from ISNC main is a no-op by diff-based `origin` flip rule (files match graph verbatim → no flips to `catalog_edit`).

---

---

## Catalog file layout (clean break)

### Layout

- **Source of truth**: `standard_names/<physics_domain>/<name>.yml`
  — one file per name, directory matches **ISN `PhysicsDomain` enum
  verbatim** (underscores, not dashes).
  - `imas_standard_names.grammar.tag_types.PhysicsDomain` is the
    authoritative shared taxonomy; imas-codex re-exports it via
    `imas_codex.core.physics_domain`; graph `physics_domain` is
    populated from it. The catalog follows the same enum.
  - 34 values: `equilibrium`, `transport`, `core_plasma_physics`,
    `turbulence`, `magnetohydrodynamics`, `auxiliary_heating`,
    `current_drive`, `waves`, `edge_plasma_physics`,
    `plasma_wall_interactions`, `divertor_physics`, `fueling`,
    `fast_particles`, `runaway_electrons`,
    `particle_measurement_diagnostics`,
    `electromagnetic_wave_diagnostics`,
    `radiation_measurement_diagnostics`,
    `magnetic_field_diagnostics`,
    `mechanical_measurement_diagnostics`,
    `plasma_measurement_diagnostics`, `spectroscopy`, `neutronics`,
    `plasma_control`, `machine_operations`, `plasma_initiation`,
    `magnetic_field_systems`, `structural_components`,
    `plant_systems`, `data_management`, `computational_workflow`,
    `general`, `gyrokinetics` (+ any additions in the ISN release
    cut for this plan).
  - Diagnostic sub-taxonomies (thomson-scattering vs
    interferometry) move into `tags` on the individual entry — they
    were never proper physics domains.
- **Manifest**: `catalog.yml` at repo root (NOT under
  `standard_names/`, so importer's recursive scan never mis-parses
  it).
- **Generated bundle**: `dist/standard_names.bundle.yml` produced
  by `imas-standard-names catalog bundle` in CI on tag push.
  Consumers pin to a release and load the single bundle.

### Clean-break strategy (no migration tooling)

The catalog repo is a thin sink of data exported from the
authoritative graph. There is no in-place migration script. On
first `sn publish` with the new ISN model:

1. `sn export` writes a fresh staging tree using ISN-enum domain
   names.
2. `sn publish` commits to ISNC by:
   - `rm -rf standard_names/ dist/` in the working tree
   - copying the staging tree into `standard_names/` +
     `catalog.yml` at root
   - `git add -A && git commit -m "catalog: republish from graph @
     <short-sha>"` or open a PR with that commit
3. Rollback is a `git revert` of that single commit. Old catalog
   contents remain reachable in git history.

This is a **one-time wipe+regen**, not a recurring pattern:
subsequent publishes write a fresh staging tree from the graph, so
the graph always wins. Hand-edits in ISNC are either (a) import-ed
back into the graph first (round-trip path), or (b) lost on next
publish. This must be documented in ISNC README.

### Rationale for one-file-per-name

| Dimension | One-file-per-name (chosen) | One-file-per-domain | Single super-file |
|---|---|---|---|
| PR diff per name add | 1 new file, +25 lines | 1 file, +25 lines | 1 file, +25 lines |
| PR merge-conflict risk | near zero | medium (concurrent PRs on same domain file) | very high |
| GitHub review UX | excellent — file-level diffs, one reviewable unit per name | acceptable for small domains, poor at 2000+ lines | collapsed diffs, painful |
| Rename / move domain | `git mv` preserves history | edit in place | edit in place |
| Cross-name consistency checks | filesystem scan | in-file grep | in-file grep |
| Consumer bundle | separate generated artefact (`dist/`) | already bundled but coarse | already bundled |

GitHub review is the primary consumer of ISNC. One-file-per-name
gives each review a bounded surface. Per-domain punishes concurrent
authorship via file-level conflicts. Super-file makes every publish
land on the same file and guarantees conflicts.

The "consumer ergonomics" argument for a single file is solved by
the generated bundle in `dist/` — source-of-truth and release
artefact can differ.

### `git mv` discipline on re-publish

When `sn publish` writes the fresh tree, it can detect renames
(graph name unchanged, `physics_domain` changed) and use `git mv`
instead of delete+add to preserve blame history. Implementation:
before the wipe, read the existing tree into memory, compute the
rename set from `(old_domain, name) → (new_domain, name)` pairs,
issue `git mv` for each, then overwrite remaining files. This is a
nice-to-have; the basic wipe+regen is correct without it.

---

## Open questions (for final rubber-duck round)

1. **Migration dual-read window**: ISN 0.8.0 dual-loader supports
   both `dd_paths` and `source_paths`. When does the window close?
   Proposal: ISN 0.9.0 (one release later).
2. **Publish gate performance**: running full corpus-health suite on
   every publish may be slow (the existing `corpus_health` marker is
   documented as "NOT part of default CI"). Do we default to a fast
   subset + opt-in full suite?
3. **Grammar-version mismatch on import**: if `catalog.yml`
   `grammar_version` differs from the installed ISN version, is that
   a hard fail or a warning? Proposal: warning only; importing an
   older catalog under a newer ISN is fine as long as individual
   entries still validate.
4. **`source_types` deprecation timeline**: keep graph-side as a
   cached derived field for now. Schedule a follow-up plan to migrate
   call sites to prefix queries on `source_paths`, then drop
   `source_types` in a later cleanup.
