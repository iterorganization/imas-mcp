# Plan 30 — DD Semantic Node Categories (fit_artifact, representation)

## Motivation

After plans 28/29 rotation, the SN pipeline retained 4 semantic filters on top
of the DD graph's own `node_category` classification:

1. `*_fit` diagnostics (chi², residual, covariance, fitting_weight, fit_type)
2. `*_fit` provenance children (reconstructed, measured, weight,
   time_measurement, rho_tor_norm)
3. GGD / basis-function representation artefacts (grids_ggd/*,
   grid_subset/*, spline / fourier_mode / finite_element coefficients)
4. `core_instant_changes/*` event-delta IDS

Items 1–3 are **DD-wide semantic facts** — they apply identically to every
downstream consumer (search, embeddings, IMAS mapping, standard names,
future enrichment). Keeping them as an SN-local Python filter means every
other pipeline that touches these paths silently sees them as `quantity`
and has to rediscover the same rejection logic.

The DD classifier is already designed as the single source of truth
(`imas_codex/core/node_classifier.py`, two-pass, shared by build and
migration CLI). The fix is to extend its enum and rules so the semantic
distinctions live **once**, in the graph, queryable from Cypher.

Item 4 stays in the SN classifier — `core_instant_changes` leaves are
legitimate `quantity` nodes. The SN policy is *de-duplication against
peer IDSs*, not a DD classification fact. DD would be wrong to label
them anything else.

## Proposed enum extension

Two new `NodeCategory` permissible values in `imas_codex/schemas/imas_dd.yaml`:

```yaml
fit_artifact:
  description: >-
    Artefact of a fitting or reconstruction process — not an independent
    physics concept. The underlying physical quantity has its own path
    outside the `*_fit` subtree. Includes fit diagnostics
    (chi_squared, residual, covariance, fitting_weight, fit_type) and
    per-fit provenance children under *_fit parents (reconstructed,
    measured, weight, time_measurement, rho_tor_norm). Neither
    enriched, embedded, nor SN-extracted.

representation:
  description: >-
    Discretisation / basis-function storage that encodes a physical
    field rather than the field itself. GGD subtrees, grid_subset
    nodes, spline coefficients, Fourier-mode harmonics, finite-element
    interpolation coefficients. The represented quantity has its own
    path; this path stores its numerical representation. Neither
    enriched, embedded, nor SN-extracted.
```

Pydantic model regenerated via `uv run build-models --force`.

## Classification rules

### Pass 1 (`classify_node_pass1` — path-pattern only)

Add checks *before* the existing `structural`/`quantity` fall-through:

**fit_artifact (pass 1):**
- `last_segment in {"chi_squared", "residual", "covariance",
  "fitting_weight", "fit_type"}`
- `last_segment` matches `_(chi_squared|residual|covariance)(_|$)`
  to catch compound names

**representation (pass 1):**
- path contains `grids_ggd/` or `/grid_subset/`
- last or parent segment matches
  `_(coefficients?|spline|fourier_modes?|finite_element|
  interpolation|basis|harmonics_coefficients|grid_object|
  jacobian|metric)(_|$)`

Both rules run before the physics-leaf fallthrough so they outrank
`quantity` / `geometry`.

### Pass 2 (`classify_node_pass2` — graph-relationship evidence)

**fit_artifact (pass 2):**
- For nodes where `last_segment in {"reconstructed", "measured",
  "weight", "time_measurement", "time_measurement_slice",
  "rho_tor_norm"}`:
  - Look up parent path; if parent segment matches `[A-Za-z0-9_]*_fit$`,
    set `node_category = fit_artifact`.

Pass 2 is required because the rule depends on parent identity, which
is only available after `HAS_PARENT` edges are built.

## Graph migration

The `dd migrate-categories` CLI already exists (invokes the same shared
classifier). Add migration with progressive rollout:

1. **Test on a small subset first.** Spin a scratch graph with equilibrium
   + edge_profiles only, run migration, snapshot the diff
   (`MATCH (n) WHERE n.node_category IN ['fit_artifact','representation']
   RETURN n.id, n.node_category`), spot-check 20 paths.
2. **Full migration.** Run against `codex` graph after SN nodes are
   cleared (so we don't have stale SN edges pointing to re-classified
   paths during the transition).
3. **Verify invariants:**
   - No existing `quantity` re-categorised as `structural` / `metadata`
     (would break other pipelines). Log and flag any.
   - No path transitions `coordinate → fit_artifact` (semantically
     wrong — coords are never fit outputs).
4. **Reindex vector indexes** if any re-categorised paths had
   embeddings — they should be deleted (lose their embeddings) since
   they no longer qualify for the embedding pipeline.

## Downstream updates (single atomic commit with schema)

1. **`imas_codex/core/node_categories.py`** — add new categories to the
   appropriate sets:
   - `SN_SOURCE_CATEGORIES` (`quantity`, `geometry`) — **unchanged**
   - Any existing `EMBEDDABLE_CATEGORIES` / `ENRICHABLE_CATEGORIES` sets
     get `fit_artifact` and `representation` explicitly excluded.

2. **`imas_codex/standard_names/classifier.py`** — remove rules S3, S4, S5
   (fit_diagnostic, fit_child, representation). Keep only S1
   (`core_instant_changes`) and S2 (`_is_error_field` — defensive). This
   collapses the SN classifier to ~20 lines.

3. **`imas_codex/standard_names/sources/dd.py`** — the extractor's
   `node_category IN $sn_categories` filter automatically excludes the
   new categories (`SN_SOURCE_CATEGORIES` only lists quantity/geometry).
   No change required beyond verifying the filter still works.

4. **Tests** — extend `tests/core/test_node_classifier.py` with:
   - ≥10 fit_artifact positive cases (one per fit subtree hit in the
     rotation: `equilibrium_fit/chi_squared`,
     `pressure_fit/reconstructed`, `q_profile_fit/weight`, etc.)
   - ≥10 representation positive cases (`grids_ggd/grid/space`,
     `coefficients` under a Fourier block, etc.)
   - ≥5 negative cases proving `measured` / `reconstructed` on a
     non-fit parent still classify as `quantity`
   - ≥3 negative cases proving `equilibrium/time_slice/boundary/outline/r`
     (geometry) is untouched.

5. **Agent docs** — update `agents/schema-reference.md` is auto-generated;
   add one line to `AGENTS.md` under "Schema System" listing the two new
   categories.

## Success criteria

- `MATCH (n:IMASNode) WHERE n.node_category = 'fit_artifact' RETURN count(n)`
  returns > 200 nodes (exact number measured post-migration, not
  hardcoded).
- `MATCH (n:IMASNode) WHERE n.node_category = 'representation' RETURN
  count(n)` returns > 500 nodes.
- Zero SN extractions from `*_fit/chi_squared` or `grids_ggd/*` paths
  in a subsequent `sn generate --domain equilibrium -c 0.5 --dry-run`.
- No `quantity` node demoted to `metadata`/`structural`/`coordinate`
  (regression test in migration script).
- All 117 SN audit tests still pass.
- `uv run pytest tests/core/test_node_classifier.py` green with new
  test cases.

## Documentation updates

- [ ] `AGENTS.md` — add `fit_artifact` and `representation` to the
  Schema System section's node category list.
- [ ] `agents/schema-reference.md` — auto-regenerated on `uv sync`.
- [ ] `plans/README.md` — register Plan 30; mark complete on
  implementation.
- [ ] Delete this plan file on full landing (code is the documentation).

## Out of scope (deferred)

- `core_instant_changes` handling — stays SN-local.
- Any further DD enum expansion (e.g. `diagnostic_setting`,
  `calibration_artefact`). Revisit only if rotation surfaces a
  distinct recurring class.
- Pass 2 relationship patterns beyond the `*_fit` parent lookup.
