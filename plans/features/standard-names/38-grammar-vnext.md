# Plan 38 — Grammar vNext (ISN rc21) + Codex Schema & Prompt Overhaul

> **Status**: DRAFT — awaiting RD review then fleet dispatch
> **Supersedes**: plan 37 (grammar-identity-prefix)
> **Scope**: ISN grammar + vocabulary rework (`imas-standard-names` repo) and
> downstream codex integration (schema simplification, prompt rewrite,
> exemplar rewrite, full regeneration validation).

---

## Executive Summary

Two independent reviews (rubber-duck `rd-plan-37`, self-authored full
grammar review `files/grammar-review-copilot.md`) converged on rejecting
plan 37's preposition-stripping approach, and identified a deeper, shared
set of defects in the ISN rc20 grammar:

| Defect | Severity | Source |
|---|---|---|
| `_of_` overloaded across 7 distinct roles | Critical | Both |
| `component` and `coordinate` share Component vocab | High | Both |
| `device` and `object` share Object vocab | High | Both |
| `physical_base` is open-vocab (fallback-any) | High | Copilot |
| Reductions not wired to main parser | High | Copilot |
| Transformation/decomposition/reduction overlap | Medium | Both |
| Geometric_base half-closed and inconsistent | Medium | Copilot |
| Plan 37's prep-less identity zone costs traceability | Critical | Both |

Plan 38 executes the **maximal redesign**: a from-the-ground-up grammar
reshape in ISN (v0.7.0rc21) combining the best of both reviews, followed
by a codex-side restructure that (1) drops the 13 `grammar_*` graph
properties in favour of a derived/recoverable model, (2) rewrites prompts
and exemplars around the new grammar, and (3) bootstraps a new
StandardName corpus from scratch to prove the gains.

**Key design insight on storage** (answers user's question):

Once the parser is round-trip-safe (`compose(parse(name)) == name` for
every valid name), grammar decomposition is **pure derived data** from
the name string. Storing 13 `grammar_*` fields on every StandardName is
schema bloat and a staleness risk whenever the parser evolves. The plan
drops the `grammar_*` prefix and replaces it with a single
`grammar_parse_version` stamp plus on-demand parsing at query time. This
removes ~150 lines of graph-ops, simplifies the LinkML schema, and makes
the ISN parser the single source of truth.

---

## Goals

| # | Goal | Success criterion |
|---|---|---|
| G1 | `_of_` appears in exactly 3 unambiguous roles | Parser rejects or diagnoses any other usage |
| G2 | Every segment has a closed, uniquely-vocabularied domain | No duplicate enums across segments |
| G3 | Round-trip safe — `compose(parse(name)) == name` | 100% of valid names; tested against ≥ 500-name corpus |
| G4 | `physical_base` is closed | Parser rejects unknown bases with actionable diagnostic |
| G5 | Operators unified into one registry with documented precedence | `magnitude_of` exists in exactly one place |
| G6 | Plan-37's Option E (prep-less clustering) rejected | No identity prefix without a grammatical marker |
| G7 | Graph schema dropped `grammar_*` properties | LinkML schema and models.py no longer expose them |
| G8 | LLM compose quality improves on real corpus | ≥ 10% uplift in reviewer-composite vs rc20 baseline on matched batch |

---

## Part A — ISN Grammar vNext (rc21)

Executed in the `imas-standard-names` repo. Greenfield breaking change;
no backwards compatibility.

### A0. Canonical-form philosophy (documentation pre-work)

Before code lands, write `docs/architecture/grammar-vnext.md` in ISN
codifying:

- **Liberal parser, strict generator**: parser accepts legacy and colloquial
  variants but emits diagnostics; generator emits exactly one canonical
  form per concept.
- **Relations keep prepositions** — `_of_`, `_at_`, `_over_`, `_due_to_`
  all preserved because they carry boundary and semantic information.
- **Prepositions are scarce resources**: each preposition marks exactly
  one semantic role.
- **Operators carry explicit scope markers**: all operator templates
  end in `_of_` inside the operator registry (not baked into token strings).
- **Closed vocabularies everywhere**: any permissive-fallback is a latent
  bug.

### A1. New 5-group internal IR (from RD's maximal sketch)

Replace the 12 declared segments with a 5-group IR:

```
StandardNameIR :=
  {
    operators:    [ OperatorApplication ],           # outer-to-inner stack
    projection:   AxisProjection | None,             # <axis>_component_of
    qualifiers:   [ species | source_entity ],       # plain prefix, closed-vocab
    base:         QuantityBase | GeometryCarrier,    # closed vocab
    locus:        LocusRef | None,                   # _of_E / _at_L / _over_R
    mechanism:    Process | None                      # _due_to_P
  }
```

Where:

- **OperatorApplication** = `{ kind: "unary_prefix" | "unary_postfix" | "binary", op: Token, args: [StandardNameIR] }`
  Recursive; the operator stack is a tree, not a flat list.
- **AxisProjection** = `{ axis: CoordinateAxis, shape: "component" | "coordinate" }`
- **QuantityBase** ∈ closed `physical_bases.yml` (new, ~200+ tokens)
- **GeometryCarrier** ∈ closed `geometry_carriers.yml` (position, outline,
  centroid, trajectory, …) — distinct from *quantities about geometry*
  like `area_of_flux_surface` (which remains a QuantityBase)
- **LocusRef** = `{ relation: "of" | "at" | "over", token: LocusToken, type: Entity | Position | Region | Geometry }`
  — single typed registry

### A2. Canonical rendering templates

One canonical form per IR; concrete templates:

```
canonical(ir) :=
    render_operators(ir.operators, fn)
        where fn(base_name) :=
            [render_projection(ir.projection) + "_component_of_"]?
            [render_qualifiers(ir.qualifiers)]?
            base_name
            [render_locus(ir.locus)]?
            [render_mechanism(ir.mechanism)]?
```

Specific renderings:
- `render_projection(p) = p.axis + "_component"` when `shape=component`
- `render_projection(p) = p.axis + "_coordinate"` when `shape=coordinate`
  (only valid when base is a GeometryCarrier)
- `render_locus(l)` = `"_" + l.relation + "_" + l.token` — relation chosen
  from `LocusRef.type` via a compatibility table (not free choice)
- `render_operators(ops, fn)` applies operators outer-to-inner:
  - unary_prefix: `op + "_of_" + inner`
  - unary_postfix: `inner + "_" + op`
  - binary: `op + "_of_" + a + "_and_" + b` (or `_to_` for ratios)

Example: `root_mean_square_of_radial_component_of_electron_pressure_at_plasma_boundary`
- operators = [unary_prefix: root_mean_square]
- projection = radial / component
- qualifiers = [electron]
- base = pressure
- locus = (at, plasma_boundary, Position)
- mechanism = None

### A3. `_of_` disambiguation (Copilot R1 + codification)

After vNext, `_of_` appears in exactly these locations:

| Role | Template | Markers |
|---|---|---|
| Unary prefix operator | `op_of_` + inner | longest-match-first against operator registry; always followed by recursive IR |
| Binary operator | `op_of_` + A + `_and_`/`_to_` + B | `_and_`/`_to_` disambiguator mandatory |
| Locus (entity/geometry) | `_of_` + LocusToken | always the **last** `_of_` in the name (trailing-position rule) |

All other `_of_` usages in rc20 are eliminated:
- ~~component template `X_component_of_`~~ → `X_component_of_` is now the
  projection-prefix template *inside* operator rendering; it is structurally
  distinguishable because it is preceded by a closed CoordinateAxis token
- ~~decomposition tokens with `_of` baked in~~ → all operator tokens in
  `transformations.yml`, `decomposition.yml`, `reductions.yml` (unified)
  store **bare** tokens (`magnitude`, `fourier_coefficient`, `time_average`);
  the `_of_` is rendered by the template engine
- ~~reduction implicit prefixes~~ → wired through the unified operator
  registry; no separate path

### A4. Axis model fix (RD minimal fix)

- New vocabulary file `coordinate_axes.yml` (distinct from `components.yml`)
- Kept: `components.yml` (18 tokens) — used only with `QuantityBase`
- New: `coordinate_axes.yml` — used only with `GeometryCarrier`; initial
  set = {radial, vertical, toroidal, poloidal, parallel, perpendicular, r, z, phi, x, y, z}
  (can overlap values with components; different *type*)
- Model validators:
  - projection with `shape=component` requires base.type == QuantityBase
  - projection with `shape=coordinate` requires base.type == GeometryCarrier

### A5. Typed locus registry (RD R4)

Replace the rc20 `object`, `geometry`, `position`, `device` segments (4
segments, 3 overlapping enums) with a single `locus_registry.yml`:

```yaml
loci:
  plasma_boundary:
    type: entity         # allowed relations: of
  magnetic_axis:
    type: position       # allowed relations: at, of
  active_limiter_point:
    type: position       # allowed relations: at, of
  flux_loop:
    type: entity         # allowed relations: of
  separatrix:
    type: entity         # allowed relations: of
  # ...
```

- `device` and `object` merge into `type: entity`
- `position` retained as distinct type (tokens that answer "where")
- `geometry` retained as distinct type (shape carriers)
- `region` kept as separate segment (`over_`) because regions are extended,
  not point/entity

Parser compatibility matrix enforced at validation time: applying `_at_` to
an entity token is a diagnostic; applying `_of_` to a region token is a
diagnostic.

### A6. Closed `physical_base` vocabulary (Copilot R3)

Create `physical_bases.yml` (new, closed-vocab) to replace the current
"open fallback" behaviour in the parser:

- Seed from the ~19 distinct `grammar_physical_base` values currently in
  the codex graph plus all physics-domain standard quantities from the
  calibration dataset and CF convention reference.
- Expected size: ~200–400 tokens after initial curation.
- Token rule: a physical_base is a noun phrase describing a physical
  quantity (temperature, pressure, density, magnetic_field, …) but is
  **never** a generic one — `generic_physical_bases.yml` lists those that
  require a qualifier (current, power, voltage, …).
- Parser behaviour: if the residue after all suffix/prefix peeling does
  not match a closed `physical_bases` or `geometry_carriers` token, the
  name is **rejected** with a diagnostic pointing to the nearest
  match(es).

### A7. Operator unification (RD R1, Copilot R4)

Merge `transformations.yml`, `decomposition.yml`, and (currently-unused)
`reductions.py` patterns into a single `operators.yml`:

```yaml
operators:
  magnitude:
    kind: unary_postfix      # rendered as <inner>_magnitude
    precedence: 10
    returns: scalar
    arg_types: [vector, complex]
  time_derivative:
    kind: unary_prefix       # rendered as time_derivative_of_<inner>
    precedence: 20
    returns: rate
  time_average:
    kind: unary_prefix
    precedence: 30           # higher number = applied outer
    returns: scalar_or_vector
  root_mean_square:
    kind: unary_prefix
    precedence: 30
  fourier_coefficient:
    kind: unary_prefix
    precedence: 40
    indexed: true            # emits fourier_coefficient_of_X_m_<m>_n_<n>
  ratio:
    kind: binary
    separator: "_to_"
    precedence: 5
  product:
    kind: binary
    separator: "_and_"
  # …
```

- One file, one lookup, explicit precedence, one parser path.
- `reductions.py` deleted; patterns fold into this registry.
- Validators enforce `arg_types` (e.g. `magnitude` requires vector or
  complex input).

### A8. Parser rewrite (liberal, diagnostic, round-trip safe)

Replace `grammar/support.py::parse_standard_name` with a new
`grammar/parser.py` implementing a staged parse:

```
def parse(name: str) -> ParseResult:
    diagnostics = []
    # 1. Strip _due_to_<process> from end
    # 2. Strip one of _over_<region>, _at_<position>, _of_<locus> from end
    # 3. Strip _<axis>_component or _<axis>_coordinate (postfix projection) OR
    #    peel <axis>_component_of_ from front (prefix projection, rendered form)
    # 4. Peel outer operators right-to-outermost:
    #    - suffix operator (longest match from operators where kind=unary_postfix)
    #    - prefix operator (longest match from operators where kind=unary_prefix)
    #    - detect binary via _and_/_to_ split
    #    recursive on inner
    # 5. Strip closed qualifier prefixes (species, source_entity)
    # 6. Residue must match physical_bases OR geometry_carriers exactly.
    #    If not, return ParseError with suggestions (edit-distance on closed vocab).
    return ParseResult(ir=…, diagnostics=diagnostics)
```

- **Liberal**: accepts known legacy forms, annotates them with
  `Diagnostic(category="non_canonical", hint="use <canonical>")` but still
  returns a valid IR.
- **Strict generator** (`compose(ir) -> str`) has no fallbacks.
- `validate_round_trip(name)` = `compose(parse(name).ir) == name` or
  records a diagnostic.
- Round-trip gate enforced in CI for the curated corpus.

### A9. Vocabulary curation pass

Real-corpus driven expansion:

- Extract distinct tokens from:
  - Current codex graph (38 names × ~13 segments)
  - Existing rc20 vocabularies
  - IMAS DD paths surfaced by EXTRACT pass (subset of DD IDSes)
  - Calibration dataset (`benchmark_calibration.yaml` — before it's removed per user's prior decision)
- Manual curation by Copilot (senior reviewer role) against the new IR
- Specifically close:
  - `physical_bases.yml` — new, ~200 tokens
  - `geometry_carriers.yml` — new, ~30 tokens
  - `locus_registry.yml` — new, merges object+position+geometry+device
  - `coordinate_axes.yml` — new, separate from components
- Deprecate/remove rc20 files: `objects.yml`, `positions.yml`,
  `geometric_bases.yml` (partial — carriers split out), old
  `transformations.yml`, `decomposition.yml`
- Update `subjects.yml` (species list) and `processes.yml` (mechanism list)
  only additively.

### A10. Test suite

Expand ISN tests:

- `tests/grammar/test_roundtrip.py` — exhaustive round-trip on cross-product
  of IR combinations (operators × projections × qualifiers × bases × loci ×
  mechanisms); target ≥ 5000 synthetic names.
- `tests/grammar/test_ambiguity_harness.py` — 50 curated pairs that must
  parse distinctly (e.g. `radial_magnetic_field` vs `radial_component_of_magnetic_field`)
- `tests/grammar/test_rejection.py` — known-bad names that must fail with
  helpful diagnostic (open-vocab fallbacks that rc20 accepted).
- `tests/grammar/test_legacy_acceptance.py` — rc20 canonical names must
  parse (possibly with non-canonical diagnostic) so pre-existing corpora
  survive.

### A11. ISN release

- Version: `v0.7.0rc21`
- Tag `rc21-grammar-vnext` on ISN main
- PyPI pre-release
- Changelog: breaking change list, migration guide from rc20
- Bump `imas-codex` pin to `imas-standard-names>=0.7.0rc21,<0.8`

---

## Part B — Codex Graph Schema Simplification

### B1. Drop `grammar_*` properties (answering the user's question)

**Current state** (as of session-scoped inspection):
- 13 `grammar_*` properties on StandardName (LinkML schema lines 617–692)
- Written by `graph_ops.py::write_standard_names` via coalesce pattern
- Read by `llm/sn_tools.py` and `llm/server.py` for MCP search post-filters
- CLI `sn` commands use some for audit queries (graph_ops.py lines 1529, 2434)
- Current population: 38 nodes, most segments sparse (subject=0, object=0,
  device=0, binary_operator=0, secondary_base=0)

**Decision**: drop all 13 `grammar_*` properties from the LinkML schema.

**Rationale**:
1. Once vNext is round-trip-safe, grammar is **derived from the name
   string**. Storing it duplicates truth.
2. Staleness risk: if the ISN parser version on disk differs from the
   parser that wrote the fields, they go stale silently.
3. Schema bloat: 13 fields × every node; compose writes 13 coalesce
   expressions per name.
4. Sparse usage: most segments are 0 in practice; they're noise.
5. The few legitimate use cases (MCP search facet filtering, CLI audits)
   are better served by on-demand parsing in a single helper:
   `imas_codex.standard_names.parse.get_segment(name, "physical_base")`.

**Replacement design**:
- Drop all 13 `grammar_*` fields.
- Add single new field `grammar_parse_version: string` — stamped at write
  time with `imas_standard_names.__version__`; enables staleness checks.
- Add single new field `validation_diagnostics_json: string` — structured
  JSON of non-canonical or edge-case diagnostics from the parser. Replaces
  the ad-hoc `validation_issues` list.
- Keep `validation_status` (pending/valid/quarantined).
- MCP search and CLI audits that filter on grammar segments:
  - **Small-result path**: parse on demand after the vector/keyword
    pre-filter returns ≤ 1000 results; fast enough.
  - **Hot path (if needed)**: add a dedicated derived materialized
    property only for the ONE segment that proves hot (likely
    `physical_base`) — but only after profiling shows necessity. Default
    no such field.

**Migration**:
- Clear StandardName nodes (`sn clear --all`) before applying schema change —
  consistent with user's "clearing between turns" bootstrap workflow.
- No forward-migration of grammar_* data needed; new corpus regenerated
  from scratch (Part D).

### B2. `validation_issues` → `validation_diagnostics_json`

Current `validation_issues` is a list of tagged strings. Replace with a
structured diagnostic schema:

```json
[
  {
    "category": "non_canonical",
    "layer": "parser",
    "message": "legacy form _component_of_ accepted; canonical postfix is _magnetic_field_radial_component",
    "suggestion": "magnetic_field_radial_component",
    "severity": "info"
  },
  {
    "category": "vocab_gap",
    "layer": "compose",
    "message": "needed geometry token not in locus_registry",
    "token": "gap_reference_point",
    "severity": "warning"
  }
]
```

Enables richer reviewer context and targeted vocab expansion workflows.

### B3. MCP API surface update

`imas_codex/llm/sn_tools.py::search_standard_names` currently accepts 13
`grammar_*` post-filter kwargs. Simplify to:

- Drop all 13 `grammar_*` kwargs.
- Add `segment_filter: dict[str, str] | None` — general-purpose facet
  dictionary applied by parsing results on the fly. Example:
  `{"physical_base": "temperature", "locus_relation": "at"}`.
- Internal implementation calls ISN `parse()` post vector hit to compute
  segments.

### B4. Graph operations simplification

- `graph_ops.py::write_standard_names` — remove all 13 coalesce
  expressions for `grammar_*`; reduce write payload.
- Any queries projecting `grammar_*` at lines 1529, 2384, 2434 rewritten
  to parse on demand or removed if stale dashboards.

### B5. Schema regeneration

`uv run build-models --force` after LinkML edits. Verify `models.py`,
`agents/schema-reference.md`, `schema_context_data.py` regenerated.

---

## Part C — Codex Prompt & Exemplar Overhaul

### C1. Grammar reference rewrite

File: `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`
- Rewritten end-to-end against vNext IR and rendering templates.
- Sourced from `imas_standard_names.grammar.get_grammar_context()` (vNext
  API returns new IR + template definitions).
- Includes the full 5-group IR, canonical template, closed vocabularies
  (physical_bases, geometry_carriers, operators, coordinate_axes,
  locus_registry, subjects, processes, regions), and disambiguation rules
  for `_of_`.
- Target size: ≤ 4k tokens (current is ~3k).

### C2. Compose prompt restructure

File: `imas_codex/llm/prompts/sn/compose_system.md`
- Shift emphasis: "name from IR" not "name from pieces".
- Make the compose LLM output an IR JSON (or structured Pydantic model),
  not a raw name string — then let codex call `compose()` to render.
  (Optional; full rewrite assesses if worth the complexity.)
- Remove all `_of_` guidance that contradicted vNext (rc20 had
  `<component>_of_<base>` templates; vNext postfixes component).
- Exemplar placement: exemplars at bottom of system prompt, user prompt
  contains only the batch.

### C3. Exemplar corpus

File: new `imas_codex/llm/prompts/shared/sn/_exemplars.md`
- Curated by Copilot in senior-reviewer mode against vNext grammar.
- Tiered:
  - **Outstanding (score 1.0)** — 10 exemplars, one per primary physics
    domain, each with a 2-sentence "why this is canonical" note.
  - **Anti-patterns** — 10 named-and-shamed examples showing:
    `_of_` overloading, open-base fallbacks, ambiguous projection,
    vocab mixing, prep-stripping (what plan 37 got wrong).
- Injected into compose prompt via `{% include "shared/sn/_exemplars.md" %}`.

### C4. Scoring rubric update

Files: `imas_codex/llm/prompts/sn/review.md`,
`imas_codex/llm/config/sn_review_criteria.yaml`
- 6-dimensional scoring preserved (grammar, semantic, documentation,
  convention, completeness, compliance).
- "Grammar" dimension criteria rewritten against vNext — penalise:
  - non-canonical `_of_` usages
  - projection on wrong base type
  - unknown physical_base (would fail parse in vNext)
  - prep-stripped identity zones
  - operator-token with `_of_` baked in
- Outstanding target still 1.0 (20/20 across all six); publish threshold
  still 0.65 (inadequate boundary); no change to aggregate.

### C5. Remove benchmark_calibration.yaml (user prior decision)

- Delete `imas_codex/standard_names/calibration.py` (cached loader)
- Delete `benchmark_calibration.yaml` fixture
- Migrate any remaining exemplar references into `_exemplars.md`

### C6. Unit injection preserved (user earlier requirement)

- Compose prompt still treats `unit` as read-only; it's injected post-LLM
  from the DD `HAS_UNIT` relationship.
- COCOS injection likewise preserved.

---

## Part D — Validation & Bootstrap

### D1. Clear graph

```bash
uv run imas-codex sn clear --all --include-sources --include-accepted
```

Confirms zero StandardName / StandardNameSource nodes.

### D2. Regenerate in small batches

Per-physics-domain, small batches, $2 cost cap each, to match the
iterative-bootstrap workflow:

```bash
uv run imas-codex sn generate --source dd --domain equilibrium \
    --limit 40 --cost-limit 2 --name-only
uv run imas-codex sn generate --source dd --domain core_profiles \
    --limit 40 --cost-limit 2 --name-only
# … repeat across ~10 domains
```

### D3. Independent review

Per batch:
1. Query graph for generated names + descriptions.
2. Copilot (senior-reviewer role) enumerates anti-patterns, inconsistencies,
   vocab gaps surfaced, grammar violations.
3. Compare against rc20 baseline (existing 38 names as control).

### D4. Quantitative gates

- **Round-trip**: ≥ 99% of generated names round-trip on first parse
  (target 100%).
- **Reviewer composite**: ≥ 10% uplift vs rc20 baseline matched batch.
- **Vocab-gap rate**: ≤ 15% of names (down from current ~30% implied by
  open-fallback).
- **Quarantine rate**: ≤ 5% (down from current ~25% in recent generations).
- **Outstanding tier rate**: ≥ 25% (up from current ~8%).

### D5. Iterate

If a batch surfaces grammar gaps (missing physical_base, missing locus,
missing operator), file them as VocabGap nodes and expand the ISN closed
vocabularies. Issue `rc22` (and further) as needed. The bootstrap loop
continues until quality plateaus.

---

## Part E — Documentation Updates

Every affected doc on both repos, per plan-lifecycle rules:

### ISN repo (`imas-standard-names`)
- `docs/architecture/grammar-vnext.md` (NEW) — canonical-form philosophy
  and 5-group IR spec (A0 deliverable).
- `README.md` — update grammar overview.
- `CHANGELOG.md` — rc21 breaking-change note.
- `docs/migration-rc20-to-rc21.md` (NEW) — migration guide.

### Codex repo (`imas-codex`)
- `AGENTS.md` § Standard Names — remove `grammar_*` MCP-filter mentions;
  describe vNext IR and on-demand parse pattern; updated search tool
  API.
- `docs/architecture/standard-names.md` — update IR diagram, update
  write/read patterns.
- `docs/architecture/standard-names-decisions.md` — new ADR: "Grammar
  is derived; drop grammar_* fields"; another: "Reject plan 37
  prep-stripping".
- `plans/README.md` — mark plan 37 SUPERSEDED, add plan 38 entry.
- `plans/features/standard-names/37-grammar-identity-prefix.md` —
  already marked superseded (see Part F).
- `agents/schema-reference.md` — auto-regenerated by build pipeline.

---

## Part F — Plan Hygiene

- Plan 37 left in place, marked **SUPERSEDED by plan 38** at the top
  (done in preparatory commit).
- On plan 38 completion, delete plan 38 (code is the doc) and optionally
  move plan 37 to `pending/` as historical reference.

---

## Phase Ordering & Dependencies

```
Phase 0 (this plan) — RD review → user approval
   ↓
Phase 1 — ISN A0 (docs)           ] parallel
Phase 2 — ISN A1..A7 (grammar)    ]
Phase 3 — ISN A8 (parser rewrite)
Phase 4 — ISN A9 (vocab curation) — blocks Phase 5 and 6
Phase 5 — ISN A10 (tests)         ] parallel with codex Phase 6, 7
Phase 6 — ISN A11 (rc21 release)
   ↓
Phase 7 — Codex B1..B5 (schema simplification) ] parallel
Phase 8 — Codex C1..C6 (prompt overhaul)       ]
   ↓
Phase 9 — Codex D1..D5 (validation & bootstrap)
   ↓
Phase 10 — Part E (documentation)
```

Critical path: ISN parser rewrite and vocab curation (A8 + A9) are the
pacing items. Codex Part B + C can start once ISN rc21 is pinned.

---

## Fleet Dispatch Taxonomy

Recommended agent split:

| Task | Agent | Rationale |
|---|---|---|
| A0 docs | engineer | Writing prose from spec |
| A1–A2 IR + templates | architect | Design decisions |
| A3 `_of_` disambiguation | architect | Cross-cutting grammar invariant |
| A4 Axis model | engineer | Spec explicit |
| A5 Locus registry | architect | Merging 4 segments into typed registry |
| A6 physical_base closure | architect | Vocabulary curation with judgment |
| A7 Operator unification | engineer | Mechanical merge per spec |
| A8 Parser rewrite | architect | Core algorithm, round-trip invariant |
| A9 Vocab curation | architect | Semantic judgment required |
| A10 Tests | engineer | Harness + fixtures |
| A11 Release | engineer | Mechanical |
| B1 Schema drop grammar_* | engineer | Spec explicit |
| B2 Diagnostics JSON | engineer | Schema edit |
| B3 MCP API update | engineer | Spec explicit |
| B4 Graph ops simplification | engineer | Mechanical |
| B5 Build regen | engineer | Mechanical |
| C1 Grammar reference | architect | Prose tied to new IR |
| C2 Compose prompt | architect | Prompt engineering |
| C3 Exemplars | architect | Senior reviewer role, physics judgment |
| C4 Scoring rubric | architect | Criteria design |
| C5 Calibration removal | engineer | Mechanical |
| D1–D5 Validation | (user + Copilot serial) | Iterative review loop |
| E1 Docs | engineer | Prose per spec |

All `architect` tasks use `claude-opus-4.6` (or latest opus) per user's
prior requirement.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `physical_base` closure misses needed tokens | Medium | High | Seed from real corpus + iterate via VocabGap workflow (D5); liberal parser accepts with diagnostic first |
| IR refactor blows out parser complexity | Low | Medium | Recursive IR is simpler than current 12-segment parser; prove on test suite before shipping |
| Round-trip gate rejects legitimate names | Medium | Medium | Liberal parser allows non-canonical with diagnostic; gate only failure-of-parse not non-canonical |
| Dropping `grammar_*` breaks downstream tool | Low | Medium | Search all call sites pre-cut; no persistent external consumers (fresh greenfield) |
| Regression in reviewer composite | Low | High | Quantitative gates (D4) explicit; rollback = re-pin rc20 |
| User loses per-segment Cypher filtering | Low | Low | On-demand parse helper provided; materialized field can be re-added if profiled |

---

## Open Questions for User / RD

1. **physical_base closure scope** — 200 or 400 tokens? Leaning 200 with
   iterative expansion via VocabGap. Confirm acceptable to user.
2. **Compose LLM output format** — raw name vs IR JSON? IR JSON is
   slightly more robust (codex renders the name) but adds a structured
   output step to the prompt. Recommend raw name for v1; IR JSON as
   follow-up if quality plateaus.
3. **Legacy acceptance policy** — rc20 canonical names should parse under
   vNext with non-canonical diagnostic (A10 test_legacy_acceptance). Hard
   requirement or best-effort? Recommend hard requirement.
4. **Regeneration corpus breadth** — how many physics domains in D2? Plan
   scoped at ~10. Confirm.
5. **ISN upstream acceptance** — does the ISN maintainer accept this
   volume of breaking change in rc21, or should some moves defer to rc22?
   Plan assumes rc21 carries it all; adjust if not.

---

## Acceptance Criteria (Overall)

- [ ] Plan 37 marked SUPERSEDED (done)
- [ ] ISN rc21 tagged and released
- [ ] Codex pinned to rc21
- [ ] Zero `grammar_*` properties on StandardName schema
- [ ] `validation_diagnostics_json` in use
- [ ] Grammar reference & compose prompt rewritten
- [ ] Exemplar corpus curated (10 outstanding + 10 anti-patterns)
- [ ] Graph cleared and regenerated across ≥ 5 physics domains
- [ ] All D4 quantitative gates passed
- [ ] All Part E documentation updated
- [ ] Plan 38 deleted (code is the doc)
