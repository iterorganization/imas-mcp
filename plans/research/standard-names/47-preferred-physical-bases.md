# Preferred `physical_base` Vocabulary — resolving ordering ambiguity without closing the open segment

**Date:** 2026-04-21
**Status:** Implemented (codex-side, no ISN change required)
**Feature branch:** `main` (committed as `feat(sn): ...`)

## 1. Problem

The ISN grammar defines `physical_base` as an **open** vocabulary — any
lowercase snake_case compound is admissible. Keeping it open is correct
(new physics quantities must be representable without blocking on a
vocabulary PR), but it creates a downstream consistency problem:

> Two grammatically-valid forms can compete for the same concept.
> `plasma_boundary_gap_angle` and `angle_of_plasma_boundary_gap` both
> parse. Reviewers score both highly, so the catalog silently
> accumulates synonym pairs that break downstream search and reduce
> cross-name coherence.

Plan `33-benchmark-evolution.md` flagged the same failure mode
("decomposition disagreement between near-synonym bases") but deferred
the structural fix. Plan `44-open-physical-base-audit.md` established
that `physical_base` is the only truly open segment in rc18 and
proposed decomposition audits — those catch *absorbed closed-vocab
tokens* but do not address open-vocab ordering ties.

## 2. Design

Introduce a **curated preferred-anchor list** that the composer and
reviewer use as a tiebreaker. The anchor list is seeded from the
graph's own emergent vocabulary so it reflects the catalog's current
consensus rather than a top-down prescription.

### 2.1 Selection criteria

A `physical_base` token qualifies as a preferred anchor when:

- It appears as the `grammar_physical_base` of **≥ 2 distinct**
  `StandardName` nodes.
- Every qualifying node has `review_mean_score ≥ 0.75`
  (`named` / `drafted` / `published` / `accepted`).

Cypher:

```cypher
MATCH (sn:StandardName)
WHERE sn.grammar_physical_base IS NOT NULL
  AND sn.review_mean_score IS NOT NULL
  AND sn.review_mean_score >= 0.75
WITH sn.grammar_physical_base AS token,
     sn.physics_domain        AS domain,
     sn.id                    AS name
WITH token, domain, collect(DISTINCT name) AS names
WITH token,
     collect({domain: domain, names: names}) AS per_domain,
     sum(size(names)) AS usage_count
WHERE usage_count >= 2
RETURN token, usage_count, per_domain
ORDER BY usage_count DESC, token ASC
```

### 2.2 Deliverables

| Artefact | Purpose |
|---|---|
| `imas_codex/llm/config/preferred_physical_bases.yaml` | committed seed list (67 anchors) |
| `imas_codex/standard_names/preferred_bases.py` | loader, mining, soft-suggestion helper |
| `imas_codex/llm/prompts/sn/_preferred_bases.md` | Jinja partial (included by compose + review) |
| `imas_codex/cli/sn.py` | `sn anchors list` / `sn anchors mine` subgroup |
| `imas_codex/llm/server.py` | `list_preferred_bases()` MCP tool |
| `tests/standard_names/test_preferred_bases.py` | unit + integration tests |

### 2.3 Prompt injection

`build_compose_context()` now sets `ctx["preferred_bases"]` by calling
`get_preferred_anchors_for_prompt()`. The `_preferred_bases.md` partial
renders the anchor list under a heading titled "Preferred
`physical_base` Anchors (ordering tiebreaker)" and spells out four
rewrite rules. The review prompt pulls the same context dict (see
`review/pipeline.py::_review_single_batch`) so the same list appears
for both composer and reviewer.

On the reviewer side, the convention dimension gained an explicit
tiebreaker: a candidate scoring `<X>_<anchor>` when the anchor-led
form `<anchor>_of_<X>` would parse loses **3 points on convention**
and is nudged to propose the anchor-led rewrite via `revised_name`.

### 2.4 Soft-suggestion helper

`suggest_anchor(physical_base)` returns an `AnchorSuggestion` when the
input ends with `_<anchor>` for some anchor on the list, and `None`
otherwise. The helper is intentionally non-mandatory so callers can
surface it as a lint hint (e.g. from a future `--lint` flag on
compose) rather than a hard reject. Longest-match wins so that
`number_density` beats `density` when both are anchors.

### 2.5 Open segment stays open

Critically, the partial **explicitly states**:

> IMPORTANT: This is **NOT** a closed list. `physical_base` remains
> an open vocabulary — new compound tokens may still be coined freely.

The anchors are advice, not gates. When a DD path surfaces a new
compound quantity with no anchor suffix that naturally applies,
composers are encouraged to coin a new base.

## 3. Initial anchor seed (67 anchors)

Mined from the graph on **2026-04-21** against the current catalog
(907 StandardNames with a `grammar_physical_base` property, 377 of
which meet the score threshold). The 67-anchor seed spans 20 physics
domains:

```
auxiliary_heating, divertor_physics, edge_plasma_physics,
electromagnetic_wave_diagnostics, equilibrium, fast_particles,
general, gyrokinetics, machine_operations, magnetic_field_systems,
magnetohydrodynamics, particle_measurement_diagnostics,
plant_systems, plasma_control, plasma_measurement_diagnostics,
plasma_wall_interactions, radiation_measurement_diagnostics,
transport, turbulence, waves
```

**Top-20 by usage** (matches the list in the feature plan):

| Rank | Token | Usage | Primary domain |
|---:|---|---:|---|
| 1 | `major_radius` | 8 | equilibrium |
| 1 | `temperature` | 8 | general |
| 3 | `coordinate` | 7 | equilibrium |
| 3 | `energy_flux` | 7 | transport |
| 3 | `magnetic_field` | 7 | magnetic_field_systems |
| 6 | `momentum_flux` | 6 | transport |
| 6 | `particle_flux` | 6 | transport |
| 6 | `momentum_convective_velocity` | 6 | transport |
| 9 | `velocity` | 5 | transport |
| 9 | `effective_charge` | 5 | general |
| 9 | `wave_absorbed_power_inside_flux_surface_per_toroidal_mode` | 5 | waves |
| 12 | `electric_field` | 4 | transport |
| 12 | `center_of_mass_velocity` | 4 | edge_plasma_physics |
| 12 | `heating_power` | 4 | general |
| 12 | `magnetic_vector_potential` | 4 | transport |
| 12 | `momentum_source` | 4 | transport |
| 12 | `number_density` | 4 | general |
| 12 | `momentum_flux_limiter` | 4 | transport |
| 12 | `momentum_flux_due_to_diamagnetic_drift` | 4 | transport |
| 12 | `wave_absorbed_power_per_toroidal_mode` | 4 | waves |
| 12 | `wave_absorbed_power_density_per_toroidal_mode` | 4 | waves |

The remaining 46 anchors cover transport-domain derivatives
(`particle_radial_diffusivity`, `energy_radial_diffusivity`,
`current_density_due_to_*`), magnetohydrodynamics (`halo_current`,
`magnetic_flux_of_halo_boundary`), equilibrium
(`contravariant_metric_tensor`, `toroidal_flux_coordinate`), and
cross-domain shape/geometry tokens (`angle`, `minor_radius`,
`pressure`, `density`, `plasma_current`). Full list in
`preferred_physical_bases.yaml`.

### Observations

- Transport dominates (≈ 60% of anchors), reflecting the current
  catalog bias. This will self-correct as enrichment pushes into
  under-represented domains — re-mining rebalances automatically.
- `coordinate` emerges as an anchor with 7 uses, confirming the
  codex `vertical_coordinate_of_X` vs `vertical_position_of_X`
  convention battle (Rule 17) is worth enforcing structurally.
- Compound tokens (`momentum_flux_due_to_diamagnetic_drift`) make
  the list because their usage reflects a coherent family; these are
  not ordering tiebreakers but rather "use this exact compound rather
  than a re-ordered variant" hints.

## 4. Growing the list

The anchor set is intended to evolve:

1. **Automatic refresh** — run `uv run imas-codex sn anchors mine
   --write` after every major rotation cycle. The CLI prints a diff
   against the committed YAML; reviewers approve the delta.
2. **Threshold tuning** — `min_usage_count` can be raised as the
   catalog matures. At 5 000 StandardNames with the current scoring
   distribution, `min_usage_count ≥ 3` would keep the list at a
   comparable size.
3. **Manual additions** — editors can add `note:` annotations to
   explain subtle rules (e.g. *"prefer `angle_of_X` over `X_angle`
   to keep the base canonical"*) and insert anchors that the graph
   has not yet seen ≥ 2 uses of but that a reviewer judges worth
   pre-emptively enforcing. Hand-added entries should mark
   `usage_count: 0` and a `note` so future mining knows not to
   silently drop them.

### Future: quarantine cycle

A subsequent cycle should audit existing names of the form
`<X>_<anchor>` and propose renames to `<anchor>_of_<X>` where
round-trip and kind/unit stability permit. This turns the soft
suggestion into a retroactive cleanup pass — tracked separately so we
can monitor review signal before shipping rename migrations.

## 5. Validation

- `tests/standard_names/test_preferred_bases.py` exercises:
  - YAML load + schema,
  - anchor-count and domain-coverage invariants (40–80 anchors,
    ≥ 15 domains),
  - soft-suggestion semantics (including longest-match and anchor-led
    no-op cases),
  - Jinja partial rendering (non-empty tokens, "NOT closed" disclaimer),
  - CLI `sn anchors list [--domain]`,
  - MCP shape check.

Full suite command: `uv run pytest tests/standard_names/test_preferred_bases.py -v`.

## 6. Non-goals / explicit exclusions

- **No ISN change.** The anchor list is entirely codex-side. ISN
  continues to treat `physical_base` as open.
- **No hard reject** on non-anchor bases. The partial, the convention
  rubric (−3 points), and `suggest_anchor()` are all advisory.
- **No per-anchor unit constraint.** Two names sharing an anchor may
  still have different units — anchors resolve *ordering*, not
  *dimensional* ambiguity.
- **No migration of existing names.** Rewriting existing
  `<X>_<anchor>` names is deferred.
