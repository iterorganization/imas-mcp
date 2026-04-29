# Plan 38 — Grammar vNext (ISN rc21) + Codex Schema & Prompt Overhaul

> **Status**: DRAFT — awaiting RD review then fleet dispatch
> **Supersedes**: plan 37 (grammar-identity-prefix)
> **Scope**: ISN grammar + vocabulary rework (`imas-standard-names` repo),
> ISNC corpus preservation (`imas-standard-names-catalog` fork), and
> downstream codex integration (schema simplification, prompt rewrite,
> exemplar rewrite, full regeneration validation).

---

## Terminology

| Term | Expansion | Meaning in this plan |
|---|---|---|
| **IR** | Intermediate Representation | A structured in-memory form of a standard name, sitting between the raw string and the rendered canonical string. The parser produces an IR, the generator consumes an IR. Round-trip safety is defined against the IR: `compose(parse(name)).string == name`. See §A1 for the 5-group IR structure. |
| **ISN** | `imas-standard-names` | The upstream grammar & vocabulary Python package (`imas_standard_names`). |
| **ISNC** | `imas-standard-names-catalog` | The YAML catalog repo holding published `StandardNameEntry` files exported from codex. Current contents (fork `main`, tag `v0.2.0-rc20-corpus`): 479 names across 23 physics domains. |
| **rc20 corpus** | The 479 names in ISNC tag `v0.2.0-rc20-corpus` | The frozen, reviewed pre-vNext corpus — used to mine vocab gaps and test round-trip behaviour. |
| **Round-trip safe** | `compose(parse(name)).string == name` | A name whose parser-IR renders back byte-for-byte to the same string. Goal G3. |
| **Canonical form** | Single unique rendering | The one string the strict generator produces for a given IR. Multiple non-canonical inputs may parse to the same IR; the generator emits exactly one canonical output. |

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

### A-1. ISNC rc20 corpus preservation (prerequisite — COMPLETED)

The imas-standard-names-catalog fork (`origin`,
`github.com:Simon-McIntosh/imas-standard-names-catalog`) has been tagged
with **`v0.2.0-rc20-corpus`** capturing the 479-name, 23-domain published
corpus exported at codex commit `0bb79adf` under ISN `v0.7.0rc20`. This
tag is the frozen mining substrate for §A9 vocabulary curation and the
baseline against which grammar vNext improvements are measured (§D4).

Fork-only — **not pushed to upstream (`iterorganization`)**. Upstream
promotion happens separately after grammar vNext stabilises, so the
upstream catalog is not published with names we intend to supersede.

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

### A9. Vocabulary curation pass — driven by ISNC rc20 corpus

Real-corpus driven expansion. **Primary source**: ISNC tag
`v0.2.0-rc20-corpus` (479 names across 23 physics domains — see §A-1).
Secondary sources fill in what the corpus does not yet cover.

Sources, in priority order:

1. **ISNC rc20 corpus** — 479 curated, published names; the main mining
   substrate. Every token that appears in a published name is a candidate
   for inclusion in one of the vNext closed vocabularies, classified by
   its grammatical role.
2. **rc20 ISN vocabularies** — seed for well-known tokens
   (subjects, processes, components, regions, etc.).
3. **IMAS DD paths** surfaced by the current EXTRACT pass.
4. **Physics convention references** (CF metadata conventions, plasma
   physics glossaries) for gap-filling where the corpus and DD don't
   surface a canonical term.

#### A9.1 Corpus-derived vocabulary gap analysis

Preliminary token mining on the rc20 corpus already shows **130 tokens
with frequency ≥ 3** that are **not in any rc20 ISN vocabulary**. Top
examples (frequency in parentheses):

```
momentum (31)     gyrokinetic (17)    eigenmode (16)    diffusivity (15)
radiated (14)     angle (12)          waveform (11)     moment (11)
potential (11)    torque (9)          rate (9)          weight (9)
effective (8)     gradient (8)        flag (8)          convective (7)
exact (7)         width (6)           decay (6)         length (6)
count (6)         angular (5)         electrostatic (5) reconstruction (5)
accumulated (4)   prefill (4)         magnetization (4) simulation (4)
collisional (3)   beta (3)            launched (3)      flow (3)
sputtering (3)    emissivity (3)      fraction (3)      stored (3)
```

These belong in one of: `physical_bases.yml` (new, closed),
`operators.yml` (new unified), `subjects.yml` (additive), or
`locus_registry.yml` (new typed). Classification is a deliverable of A9.

#### A9.2 Curation deliverables

Specifically close, populated from the corpus:

- **`physical_bases.yml`** (NEW, closed) — every unique `physical_base`
  token that appears as a name's residue after IR decomposition. From
  the corpus: ~180 distinct physical bases (temperature, pressure,
  number_density, current_density, magnetic_flux, safety_factor,
  effective_charge, radiated_power, torque_density, …).
- **`geometry_carriers.yml`** (NEW, closed) — position, outline, centroid,
  trajectory, line_of_sight, etc.; ~30 tokens from corpus.
- **`locus_registry.yml`** (NEW, typed) — merge of rc20
  `objects.yml` + `positions.yml` + `geometry` tokens + `device` enums,
  each tagged with its allowed relations. ~230 tokens from corpus
  (plasma_boundary, magnetic_axis, separatrix, wall, divertor_target,
  flux_loop, x_point, ion_cyclotron_heating_antenna, …).
- **`coordinate_axes.yml`** (NEW) — separate from components; ~12 tokens.
- **`operators.yml`** (NEW, unified) — merges rc20 transformations +
  decomposition + reductions + binary operators, with precedence,
  arg-type, and rendering metadata. Expected ~45 operators including
  corpus-surfaced ones like `moment` (postfix), `gyroaveraged` (postfix),
  `maximum_of` (prefix), `derivative_of_X_with_respect_to_Y` (binary-like
  with coordinate qualifier), `line_integrated`, `flux_surface_averaged`,
  `accumulated`, `prefill`.

Deprecate/remove rc20 files: `objects.yml`, `positions.yml`,
`geometric_bases.yml` (partial — carriers split out), old
`transformations.yml`, `decomposition.yml`.

Update additively: `subjects.yml` (species list; add `counter_passing`,
`co_passing`, `thermal`, `fast` if not already), `processes.yml`
(mechanism list; add `coulomb_collisions`, `thermalization`, `impurity_radiation`,
`diamagnetic_drift`, `perpendicular_viscosity`), `regions.yml` (add
`pedestal`, `separatrix` as region-typed where applicable).

#### A9.3 Curation workflow

1. Extract distinct tokens from all 479 corpus names (grouping by position
   in rc20 parse IR).
2. Cross-reference with rc20 vocabulary files.
3. Copilot (senior-reviewer role, Opus 4.6) classifies each
   unknown/ambiguous token into the correct vNext segment.
4. Generate draft vocab files; parse every corpus name through the new
   parser; any `ParseError` triggers a vocab review cycle.
5. Target: ≥ 95% of corpus names parse cleanly on first pass (the
   remaining ≤ 5% are expected to be legitimate diagnostics where the
   name itself is non-canonical and will be regenerated).

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

### A11. ISN release — rc21 gate for codex work

This is the **hard gate** between ISN work and codex work. Codex Part B/C/D
cannot start until ISN rc21 is released to PyPI pre-releases and the codex
`pyproject.toml` pin is updated. The gate ensures codex teams `uv sync`
against a stable grammar surface before rewriting prompts and schema.

Steps:

1. All §A10 tests pass.
2. Run full round-trip parse against ISNC rc20 corpus (§A-1, 479 names);
   ≥ 95% parse cleanly, remaining ≤ 5% fail with actionable diagnostics.
3. Bump ISN version: `pyproject.toml` → `0.7.0rc21`
4. Tag ISN: `git tag v0.7.0rc21 && git push origin v0.7.0rc21`
5. Publish pre-release to PyPI (ISN project uses trusted-publisher CI
   triggered on tag; verify release page shows 0.7.0rc21).
6. In imas-codex: update `pyproject.toml` pin to
   `imas-standard-names>=0.7.0rc21,<0.8`; run `uv sync --refresh`;
   commit the bump.
7. **Only after step 6 completes, Part B/C work begins.**

Changelog content for rc21:
- Grammar vNext 5-group IR (breaking)
- Vocabulary closure: `physical_bases`, `geometry_carriers`,
  `locus_registry`, `coordinate_axes` (new, closed)
- Operator unification (transformations + decomposition + reductions +
  binary → single `operators.yml`)
- `_of_` disambiguation
- Round-trip gate enforced in CI
- Deprecation list: `objects.yml`, `positions.yml`, partial
  `geometric_bases.yml`, old `transformations.yml`, `decomposition.yml`,
  `reductions.py`
- Public API: `parse()`, `compose()`, `get_grammar_context()`,
  `validate_round_trip()`

### A12. Current-vs-proposed name table — demonstrating the uplift

This table is the concrete evidence that vNext is demonstrably better.
Every row is a real corpus name from ISNC tag `v0.2.0-rc20-corpus`, shown
in both the rc20 form (as published today) and the vNext canonical form
it would parse to or be rewritten as.

Each row's commentary flags the specific defect it exposes in rc20 and
what invariant vNext restores. The vNext column shows the strict
canonical rendering.

| # | Domain | Current (rc20) | vNext canonical | Uplift commentary |
|---|---|---|---|---|
| 1 | equilibrium | `elongation_of_plasma_boundary` | `elongation_of_plasma_boundary` | Unchanged. `_of_` is the *single* permitted locus suffix pointing to an entity-typed LocusToken. Canonical under both grammars. Demonstrates vNext preserves clean names. |
| 2 | equilibrium | `minor_radius_of_plasma_boundary` | `minor_radius_of_plasma_boundary` | Unchanged for the same reason as #1. Rejects plan 37's proposed `plasma_boundary_minor_radius` — the `_of_` form is grammatically clearer because `_of_` is reserved for this one role. |
| 3 | equilibrium | `major_radius_of_x_point` | `major_radius_of_x_point` | Unchanged. `x_point` now typed as `position` in `locus_registry`; the `_of` relation is valid for position-typed loci. Under rc20 `x_point` collided between `objects` and `positions` — vNext resolves. |
| 4 | equilibrium | `flux_surface_averaged_inverse_major_radius` | `flux_surface_averaged_major_radius_inverse` OR keep as `flux_surface_averaged_inverse_of_major_radius` | Reveals rc20 reductions-not-wired: `flux_surface_averaged` is a reduction operator but rc20 parser treated the whole string as `physical_base`. vNext routes it through unified operators and can render either form unambiguously (postfix `_inverse` or prefix `inverse_of_`). **Decision needed**: whether `inverse` is a unary postfix op. Recommendation: yes, same treatment as `magnitude`. |
| 5 | equilibrium | `derivative_of_flux_surface_cross_sectional_area_with_respect_to_radial_coordinate` | `derivative_with_respect_to_radial_coordinate_of_flux_surface_cross_sectional_area` OR retain `derivative_of_X_with_respect_to_Y` | In rc20 the `derivative_of_X_with_respect_to_Y` form is implicit — not parsed as an operator. vNext promotes `derivative_with_respect_to_<Y>` to a **parametric unary prefix operator** where `Y` is a CoordinateAxis or known independent variable. Parser recognises the pattern. Recommended canonical: `derivative_of_<X>_with_respect_to_<Y>` kept, but now *structurally decomposed* rather than parsed as one big base. |
| 6 | equilibrium | `contravariant_metric_tensor` | `contravariant_metric_tensor` | Unchanged. Listed as a closed `physical_base`. rc20 accepts it via open-fallback; vNext accepts it via closed-vocab match — same string, much stronger guarantee. |
| 7 | edge_plasma_physics | `parallel_component_of_current_density_due_to_perpendicular_viscosity` | `current_density_parallel_component_due_to_perpendicular_viscosity` | Component moves from prefix `_of_` form to postfix. `perpendicular_viscosity` is a `process` (already in rc20). The `_due_to_` suffix remains. Result: `_of_` is freed for locus use only; component boundary marked by closed-vocab postfix `_parallel_component`. |
| 8 | edge_plasma_physics | `parallel_component_of_electron_momentum_convective_velocity` | `electron_momentum_convective_velocity_parallel_component` | Same component inversion as #7. Reads left-to-right as "electron momentum convective velocity, parallel component". |
| 9 | edge_plasma_physics | `ion_momentum_flux_due_to_diamagnetic_drift` | `ion_momentum_flux_due_to_diamagnetic_drift` | Unchanged. `_due_to_` is a unique mechanism marker in both grammars. `diamagnetic_drift` added to `processes.yml`. |
| 10 | fast_particles | `fast_particle_toroidal_torque_density_due_to_coulomb_collisions_with_electrons` | `fast_particle_toroidal_torque_density_due_to_coulomb_collisions_with_electrons` | Unchanged, but `coulomb_collisions_with_electrons` becomes a *compound process token* in `processes.yml` rather than a string the open-fallback swallowed. `with_<subject>` is a process modifier. |
| 11 | fast_particles | `ion_collisional_toroidal_torque_density_on_thermal_population_due_to_fast_ions` | `collisional_toroidal_torque_density_on_thermal_ion_population_due_to_fast_ions` | Clarifies where `thermal` / `fast_ions` / `ion` apply. vNext: qualifier `thermal_ion` (compound subject), process `coulomb_collisions` implicit in `collisional`, agent `fast_ions` via `due_to_`. Removes the ambiguity of "is `ion` a species or qualifier?" from rc20. |
| 12 | transport | `derivative_of_ion_poloidal_velocity_with_respect_to_normalized_toroidal_flux_coordinate` | `derivative_of_ion_poloidal_velocity_with_respect_to_normalized_toroidal_flux_coordinate` | Unchanged surface form; vNext *structurally decomposes* as `{op: derivative, wrt: normalized_toroidal_flux_coordinate}` around `{subject: ion, base: poloidal_velocity}`. The rc20 parser treats this as one opaque string. **Demonstrably better**: vNext can answer "what are all derivatives with respect to normalized_toroidal_flux_coordinate?" from IR without substring search. |
| 13 | transport | `counter_passing_particle_number_density` | `counter_passing_particle_number_density` | Unchanged. `counter_passing` promoted to an additive `subjects.yml` entry (pitch-angle species qualifier). rc20 parsed via open-fallback. |
| 14 | general | `maximum_of_derivative_of_electron_pressure_with_respect_to_normalized_poloidal_flux_at_pedestal` | `maximum_of_derivative_of_electron_pressure_with_respect_to_normalized_poloidal_flux_at_pedestal` | Unchanged surface form; vNext decomposes as nested operators: `maximum_of(derivative_wrt(normalized_poloidal_flux, electron_pressure))_at_pedestal`. Parser correctly peels two prefix operators AND an `_at_` locus suffix. rc20 parser cannot represent this composition. **Big uplift**: operator stack recursion is a vNext first-class feature. |
| 15 | gyrokinetics | `gyrokinetic_eigenmode_normalized_gyrocenter_parallel_current_density_moment_gyroaveraged` | `gyroaveraged_moment_of_parallel_gyrocenter_current_density_of_gyrokinetic_eigenmode` OR canonical shorter form TBD | Exposes rc20's flat-parse failure on nested operators. vNext decomposes: operators [`gyroaveraged` postfix, `moment` postfix], projection `parallel/component`, qualifier `gyrocenter`, base `current_density`, locus `gyrokinetic_eigenmode` (entity). Canonical ordering puts operators outer-first. **Demonstrably better**: legibility improves and parser answers "all `gyroaveraged` quantities" from IR. |
| 16 | magnetohydrodynamics | `neoclassical_tearing_mode_seed_island_width` | `seed_island_width_of_neoclassical_tearing_mode` | In rc20, `neoclassical_tearing_mode` prefixed as source_entity and `seed_island_width` fell to open physical_base. vNext: `seed_island_width` is a closed `physical_base`; `neoclassical_tearing_mode` is an entity-typed locus. The relation `_of_` is now required — makes the subject-object relation explicit and parseable. |
| 17 | magnetohydrodynamics | `normalized_toroidal_flux_coordinate_at_sawtooth_inversion_radius` | `normalized_toroidal_flux_coordinate_at_sawtooth_inversion_radius` | Unchanged. `normalized_toroidal_flux_coordinate` is a `geometry_carrier` (new closed vocab, §A9); `sawtooth_inversion_radius` is a position-typed locus. `_at_` relation valid for position loci. |
| 18 | plasma_control | `electron_cyclotron_beam_toroidal_steering_angle_reference_waveform` | `electron_cyclotron_beam_toroidal_steering_angle_reference_waveform` | Unchanged surface form, but `reference_waveform` becomes a closed *postfix operator* (kind=unary_postfix) in `operators.yml`. rc20 treats it as part of an open physical_base. vNext uplift: "all reference waveforms" queryable from IR; semantic type tracked (time series of reference values). |
| 19 | plasma_control | `lower_hybrid_antenna_parallel_refractive_index_reference_waveform` | `lower_hybrid_antenna_parallel_refractive_index_reference_waveform` | Same as #18 — postfix operator `_reference_waveform`. Demonstrates vNext's uniform operator treatment across diverse bases. |
| 20 | gyrokinetics | `gyrokinetic_eigenmode_normalized_parallel_temperature_moment_gyroaveraged_real_part` | `gyroaveraged_real_part_of_moment_of_parallel_temperature_of_gyrokinetic_eigenmode_normalized` OR shorter canonical | Triple-operator: `real_part` postfix, `gyroaveraged` postfix, `moment` postfix. rc20 open-fallback. vNext IR captures full structure. **Demonstrably better**: the distinction between `_real_part` and `_imaginary_part` is an operator, not a magic string — composers cannot mint novel `_<complex_op>_part` forms. |
| 21 | particle_measurement_diagnostics | `neutron_detector_line_integrated_emissivity` | `line_integrated_emissivity_of_neutron_detector` | `neutron_detector` becomes entity-typed locus; `line_integrated` becomes a closed reduction operator (currently in rc20 as an unused pattern). Generator emits canonical order: operator outer, locus last. |
| 22 | plasma_wall_interactions | `maximum_of_power_flux_density_at_inner_divertor_target` | `maximum_of_power_flux_density_at_inner_divertor_target` | Unchanged surface form. IR: operator stack = [`maximum` unary_prefix], base = `power_flux_density`, locus = `inner_divertor_target` (position). Correct decomposition enables reduction-aware queries. |
| 23 | auxiliary_heating | `vertical_coordinate_of_ion_cyclotron_heating_antenna` | `vertical_coordinate_of_ion_cyclotron_heating_antenna` | Unchanged. `vertical_coordinate` = CoordinateAxis(vertical) + GeometryCarrier(coordinate). This row demonstrates the vNext projection/carrier model (A4). rc20 parses `vertical` as coordinate but leaves `coordinate` alone as a meaningless base. vNext binds `vertical_coordinate` to the typed carrier. |
| 24 | magnetic_field_systems | `toroidal_component_of_magnetic_field_at_ferritic_element_centroid` | `magnetic_field_toroidal_component_at_ferritic_element_centroid` | Component inversion (postfix) + `_at_` locus preserved. `ferritic_element_centroid` is a new entity-typed locus. Reads left-to-right as "magnetic field, toroidal component, at ferritic-element centroid". |
| 25 | magnetic_field_systems | `toroidal_component_of_magnetic_moment_of_ferritic_element_centroid` | `magnetic_moment_toroidal_component_of_ferritic_element_centroid` | **Critical**: rc20 has two `_of_` in this name with different meanings — component relation and locus relation. vNext eliminates this ambiguity: component is postfix, only `_of_` remaining is the locus relation. This is the single most compelling per-name demonstration of `_of_` disambiguation. |
| 26 | structural_components | `coolant_outlet_temperature_of_breeding_blanket_module` | `coolant_outlet_temperature_of_breeding_blanket_module` | Unchanged. `coolant_outlet_temperature` is a compound physical_base (closed), `breeding_blanket_module` is an entity-typed locus, `_of_` is the sole locus relation. Demonstrates vNext can accept engineering-domain compound bases without reaching for open-fallback. |
| 27 | turbulence | `gyrokinetic_eigenmode_normalized_gyrocenter_parallel_current_density_moment_bessel_0` | `bessel_0_moment_of_parallel_gyrocenter_current_density_of_gyrokinetic_eigenmode_normalized` OR canonical TBD | `bessel_0`, `bessel_1` are indexed postfix operators (parametric, like `fourier_coefficient_<m>_<n>`). vNext operators registry allows indexed operators. rc20 open-fallback. |
| 28 | plant_systems | `toroidal_angle_of_soft_xray_detector_line_of_sight_second_point` | `toroidal_angle_of_soft_xray_detector_line_of_sight_second_point` | Unchanged. `toroidal_angle` is a GeometryCarrier (angle with a direction), `soft_xray_detector_line_of_sight_second_point` is a compound entity-typed locus. **Note**: the `line_of_sight_second_point` pattern should be factored into `locus_registry` as a position-on-line-of-sight composite. |

---

#### A12.1 Summary of demonstrable uplift

**Invariants vNext establishes that rc20 does not**:

1. **Each `_of_` in a name has exactly one role** — row 25 alone would be
   ambiguous without vNext's disambiguation.
2. **Operators are first-class** — rows 12, 14, 15, 20, 21, 22 were flat
   strings in rc20; vNext exposes nested IR enabling queries like "all
   derivatives with respect to X", "all gyroaveraged quantities", "all
   reference waveforms".
3. **Loci are typed** — row 3 resolves the `x_point` object-vs-position
   collision. Relations `_of_`, `_at_`, `_over_` are validated against
   locus type.
4. **Projection is typed** — row 23 shows the coordinate-vs-component
   distinction enforced by base type (QuantityBase vs GeometryCarrier).
5. **Physical bases are closed** — rows 6, 10, 16, 18, 26 expose the rc20
   open-fallback; vNext either accepts them into the closed vocabulary
   (with provenance) or rejects with an actionable diagnostic.
6. **Canonical generator is deterministic** — rows 7, 8, 11, 15, 16, 24,
   25 show rewrites where the vNext canonical form is different from rc20.
   The parser accepts both (liberal) during rc21 transition; the generator
   emits only canonical (strict).
7. **Round-trip safety** — every row passes `compose(parse(name)).string
   == name` for its canonical form. Goal G3 measurable per-name.

**Counter-examples (deliberately unchanged)**:

Rows 1, 2, 9, 13, 17 demonstrate vNext preserves clean rc20 names
byte-for-byte. The redesign is not cosmetic churn — it targets only the
structural defects.

**Macro-metric targets** (quantified in D4, evaluated against the full
479-name corpus):

- Round-trip success ≥ 99% (stretch 100%)
- Names requiring non-canonical rewrite: 30–40% (rows 4, 7, 8, 11, 15, 16,
  18, 19, 20, 21, 24, 25, 27 class)
- Names passing unchanged: 60–70% (row 1, 2, 3, 6, 9, 10, 12, 13, 14, 17,
  22, 23, 26, 28 class)
- Zero ambiguous `_of_` usages (row 25 class eliminated)
- Zero open-fallback physical_base hits after §A9 curation completes

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
Phase 1  — ISNC rc20 corpus tag (DONE: v0.2.0-rc20-corpus on fork)
   ↓
Phase 2  — ISN A0 (docs)           ] parallel
Phase 3  — ISN A1..A8 (grammar,    ] parallel (no corpus dep yet)
             parser, IR, ops)      ]
   ↓
Phase 4  — ISN A9 (vocab curation DRIVEN BY ISNC rc20 corpus)
   ↓
Phase 5  — ISN A10 (tests incl. round-trip on 479-name corpus)
   ↓
Phase 6  — ┌──────────────────────────────────────────────────┐
           │  HARD GATE: ISN v0.7.0rc21 released to PyPI      │
           │  imas-codex pyproject.toml pin bumped & uv sync  │
           └──────────────────────────────────────────────────┘
   ↓
Phase 7  — Codex B1..B5 (schema simplification) ] parallel
Phase 8  — Codex C1..C6 (prompt overhaul)       ]
   ↓
Phase 9  — Codex D1..D5 (validation & bootstrap)
   ↓
Phase 10 — Part E (documentation, both repos)
```

**Critical path**: A9 vocab curation (driven by ISNC corpus) → A10 tests
→ A11 release gate. Codex Part B/C cannot begin until the rc21 release
completes and codex is pinned.

**Why the gate matters**: without the release, codex prompt and schema
changes reference a moving target. The gate ensures all downstream work
builds against a stable, tagged grammar surface installed via `uv sync`.

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
