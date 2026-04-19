# SN Enrichment Rotation (documentation phase)

## Context

The naming rotation has produced 335 valid StandardNames. Zero of them have
`description` or `documentation` fields populated — the `sn enrich` pipeline
(landed in Plan 29) has never been exercised at scale. Enrichment is the
documentation phase and must happen before the downstream MCP `search_standard_names`
tool can return meaningful prose for consumers.

## Why a rotation (vs one big run)?

The naming rotation taught us that per-domain $2-capped batches with a senior-physicist
review between runs surfaces architectural and prompt issues that bulk generation hides.
Enrichment has more degrees of freedom than naming (free prose, reference links, tags)
and therefore more quality risk. Same rotation pattern applies.

## Pipeline precondition

`sn enrich` CLI already exists with:

- `--domain D` filter (repeatable)
- `--status STATE` (default `named`)
- `-c/--cost-limit USD`
- `--limit N`
- `--batch-size N` (default 8)
- `--dry-run`
- `--force` to re-enrich
- `--model` override

We use these as-is. No new CLI surface.

## Rotation protocol

For each domain D (ordered by coverage density):

1. `uv run imas-codex sn enrich --domain D -c 2 --limit 40` — run, observe cost & cost/name.
2. Review via MCP:
   ```
   MATCH (sn:StandardName {physics_domain: 'D', validation_status: 'valid'})
   WHERE sn.description IS NOT NULL
   RETURN sn.id, sn.description, sn.documentation, sn.source_paths
   ```
3. Senior-physicist critique (done by me, the operator):
   - semantic accuracy against DD path documentation,
   - consistency of cross-name terminology within the domain,
   - tautologies (`radiated power due to impurity radiation`),
   - latex symbol hygiene,
   - links (`links` property) — pointing to real SN ids or broken?
4. File issues by kind:
   - prompt patches → `sn/enrich_*.md` edits (commit to codex).
   - grammar / vocab gaps → ISN PR.
   - audit additions → new `audits.py` rules.
5. `sn generate --reset-to drafted --reset-only` then re-run step 1. Continue until either:
   - mean reviewer score improves by ≥ 0.05, or
   - two consecutive iterations plateau.

## Domain order

Smallest to largest, to get fast feedback. Also avoids burning budget on domains where
naming itself still has gaps.

1. `turbulence` (10)
2. `plasma_control` (12)
3. `plasma_wall_interactions` (24)
4. `magnetohydrodynamics` (31)
5. `radiation_measurement_diagnostics` (31)
6. `edge_plasma_physics` (47)
7. `equilibrium` (63)
8. `transport` (113)

## Context packaging (POSTLINK)

The enrich worker must inject, per name:

- All DD paths in `source_paths` with their `documentation` text.
- 3–5 nearest-neighbour SNs (by description-embedding) **that are themselves already enriched** — few-shot exemplars.
- The SN's `cocos_transformation_type` and `unit` as read-only context.
- The SN's domain-level preferred-vocabulary list (same as naming-rotation L1 lever).

Verify `enrich_workers.py` currently packages all of these. Gap: reference-exemplar
retrieval from already-enriched names requires at least one domain to be enriched
before bootstrapping; cold start uses ISN's `core/` canonical SN descriptions as seeds.

## Cost budget

$2 per domain × 8 domains = $16 total for Pass 1.  Expected cost/name at sonnet-4.6:
$0.02-0.05 (names are shorter than compose prompts but docs are longer). Limit of 40/run
keeps us well inside the cap.

## Success metrics

| Metric | Target |
|---|---|
| Description coverage | ≥ 95 % of valid names after P1 rotation |
| Documentation coverage | ≥ 95 % of valid names after P1 rotation |
| Links field non-empty | ≥ 70 % (most names have siblings) |
| Reviewer mean score | ≥ 0.80 per domain post-iter-2 |
| Cost per domain | ≤ $2 |
| No new `validation_status=quarantined` flips | hard gate |

## Risks

- **Link resolution**: docs reference other SNs by name; if target SN doesn't exist
  the link is a dangling string. Plan 23 Phase 4 (`async_link_resolution`) addresses
  this but is unlanded. Mitigation: for P1 rotation, accept free-text links; add
  resolver in P2.
- **Over-long documentation**: enrich may over-generate. Add a soft-cap audit
  `documentation_length_check` (warn > 800 chars, quarantine > 1500).
- **Model drift**: sonnet-4.6 may generate different style vs naming phase. Lock the
  enrich model via `[tool.imas-codex.sn-enrich]` pyproject section.

## Dispatch

Operator-driven (interactive rotation, same pattern as naming). No background agent.
