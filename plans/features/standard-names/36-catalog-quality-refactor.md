# 36 — Catalog Quality Refactor: Field Rationalisation, Link Generation, Grounded Examples, Complex Parent/Part

**Status:** PLANNING (RD round 2 pending — incorporates round 1 findings + complex parent/part design)
**Depends on:** Plan 35 complete; ISNC PR #1 open and green; ISN v0.7.0rc20 pinned
**Precedes:** Re-enrichment + re-export cycle; ISNC PR #1 update
**RD round 1 findings addressed:** sanitiser `dd:` blocker fixed in 2a; Phase 1 ordering corrected (KEEP `complex` removes the rc21-vs-Cypher hazard); grounding rule replaced with prohibit-unless-grounded (presence-check, not truth-check); cost cap proportional to per-domain count; selective YAML rollback primary; Phase 2.5 rubric calibration added; `PROTECTED_FIELDS` pruned; complete blind-spot list addressed.

## Problem

Review of 480 catalog entries published via ISNC PR #1 surfaces six concerns
spanning schema design, LLM prompting, reviewer grounding, and taxonomy:

1. `cocos_transformation_type` is uniform in every catalog entry (`one_like`
   or similar) but is only physically meaningful for sign-convention-dependent
   quantities (522 / 927 graph nodes carry it non-trivially). Putting it on
   every catalog row — including those where it is irrelevant — is noise.
2. Catalog entries have **no `see also` / links content**. Graph coverage:
   `links = 0/927`. The enrichment prompt asks for them "optional but
   encouraged"; the LLM obliges by skipping. The link phase (renamed from
   resolve-links) runs but finds nothing to resolve.
3. `validity_domain` and `constraints` are **empty on every entry**
   (`0/927`). Same prompt-weakness as links, plus the fields themselves
   have poorly-defined semantics and overlap with `tags` + `documentation`.
4. Documentation contains **ungrounded facility claims**
   (e.g. "in ITER elongation is typically …") generated without verification.
   The reviewer has no explicit mandate to check these.
5. Documentation mixes units in numeric ranges ("a few milliradians to
   several degrees"). No prompt rule forbids this.
6. ISN `Kind` was extended to `{scalar, vector, tensor, complex, metadata}`
   in v0.7.0rc20. The question is whether we also need
   `complex_scalar` / `complex_vector` / `complex_tensor` to carry rank
   orthogonally to "is it complex?". Current data: all 6 `complex` names
   are `*_real_part` / `*_imaginary_part` of gyrokinetic eigenmodes — i.e.,
   already reduced to scalars by storage convention.

The composite effect: the catalog ships dead fields, misleading uniform
metadata, cross-reference gaps, and occasional physically-dubious prose —
all the things we want the catalog NOT to ship.

## Critical review of field design

Before deciding how to fix, we must decide whether the current
`StandardNameEntryBase` schema shape is correct.

### `validity_domain: str = ""`

- **Semantics as-is**: "Physical region or regime where the quantity is
  meaningful" — free-form string (`""`, `"core plasma"`, `"SOL"`, …).
- **Overlap**: `tags` already carries secondary classification;
  `physics_domain` (graph-side, not emitted) carries primary. Free-form
  `validity_domain` with no controlled vocabulary duplicates these poorly.
- **Enforceability**: unconstrained strings defeat any consumer that wants
  to filter ("show me all SOL-only quantities").
- **Verdict**: Current design is **wrong**. Either
  - **A1 Delete**. Move any meaningful regime info into `tags` (with a
    controlled `regime-*` namespace) or into `documentation` prose.
  - **A2 Convert to controlled enum**: `validity_domain: Regime | None`
    with `Regime = Literal["core", "edge", "sol", "pedestal", "divertor",
    "scrape_off_layer", "global", "boundary", …]`. Enforced, filterable.
- **Recommendation**: **A1 (delete)**. `tags` already supports secondary
  classification and has a controlled vocabulary. A separate regime field
  is redundant; if we want enforced regime filtering we add it as a tag
  namespace (`regime-core`, `regime-sol`). Deletion is the cleaner path.

### `constraints: list[str] = []`

- **Semantics as-is**: "Physical constraints on the quantity" — free-form
  list of prose strings.
- **Real-world coverage**: `0/927`. Nobody has emitted one.
- **Enforceability**: zero — free-form list of strings is untestable.
- **Prose alternative**: physical constraints (positivity, boundedness,
  normalisation, conservation laws) belong in `documentation` where they
  can be written mathematically (`$0 \\le \\rho_{\\mathrm{tor,n}} \\le 1$`)
  and interpreted by humans.
- **Verdict**: Current design is **wrong**. Delete.

### `links: list[str] = []`

- **Semantics as-is**: "Related standard names by `name:x` prefix or
  external URLs".
- **Coverage**: `0/927`. The link resolution machinery exists
  (`dd:` placeholder → `name:` after link phase) but the enrichment LLM
  emits nothing.
- **Structural design**: a list of typed references (`name:`, `dd:`, `url:`)
  is the right shape — distinct from prose, renderable as a `See also`
  block, queryable by the graph for cross-references.
- **Verdict**: Current design is **correct**; only the prompt and
  enforcement are broken. Make links required.

### `cocos_transformation_type: str | None`

- **Semantics as-is**: COCOS sign-flip class (`psi_like`, `ip_like`, …,
  or `one_like`/`null` for sign-invariant quantities).
- **Physically meaningful**: yes, for ~522 of 927 names.
- **Catalog relevance**: the catalog supports a single COCOS convention
  (picked by the consumer). Per-entry COCOS type is graph-side
  bookkeeping, not consumer-facing. For COCOS-invariant entries the field
  is actively misleading.
- **Verdict**: Move to **graph-only**. Exclude from catalog YAML export.
  Keep on graph `StandardName` node; keep the `HAS_COCOS` relationship
  and the `cocos_transformation_type` property. Add to the export skip-set
  alongside `source_paths`, reviewer scores, etc.

### Complex kind — mirror the vector parent/component pattern

Current state (6 names, all `complex`): every entry is a `*_real_part`
or `*_imaginary_part` **derived part** with **no canonical parent** —
they are orphans. Compare with vectors:

| Convention | Parent (carries docs) | Derived part |
|---|---|---|
| Vector | `plasma_velocity` (`vector`) | `radial_component_of_plasma_velocity` (`vector`) |
| Vector | `plasma_velocity` (`vector`) | `magnitude_of_plasma_velocity` (`vector`, via `.magnitude` @property) |
| **Complex (target)** | `perturbed_electrostatic_potential` (`complex`) | `perturbed_electrostatic_potential_real_part` (`complex`) |
| **Complex (target)** | `perturbed_electrostatic_potential` (`complex`) | `perturbed_electrostatic_potential_imaginary_part` (`complex`) |

The ISN `Kind` docstring (models.py:164–167) already declares that `vector`
covers both the parent and named single components — a flat-discriminator
design choice. We apply exactly the same rule to `complex`: a canonical
complex parent AND its real/imaginary/amplitude/phase/modulus parts all
carry `kind: complex`. Parent ↔ part relationships are purely
name-structural (stem + suffix), mirroring vector: no new graph edges,
no `PART_OF` relationship type.

- **Decision: KEEP `complex`** in the Kind enum (**B1**, overriding
  the earlier B2 recommendation after RD review). Rationale:
  - Storage-vs-provenance argument (B2) misses the point: the catalog
    encodes physics semantics, not storage format. A complex-valued
    quantity split into real and imaginary halves is one physical
    object, and the kind discriminator must convey that.
  - Removing `complex` destroys the only structured signal that
    `X_real_part` and `X_imaginary_part` are paired; naming-convention-
    only inference has no enforcement mechanism.
- **Corollary — fix the root cause**: the 6 current entries are
  orphan parts because the pipeline never generates their canonical
  parents. Phase 1c (new) mirrors `StandardNameVectorEntry.magnitude`
  with part-derivation properties on `StandardNameComplexEntry`, and
  Phase 2d (new) adds a compose rule + audit that require parent-first
  emission whenever a derived-part name is produced.
- **Complex-variant kinds (`complex_scalar`, `complex_vector`, `complex_tensor`)**:
  **Do not add**. Rank orthogonality is handled by the existing
  suffix convention (`<parent>_real_part`, …) within each kind.
  Native complex vector/tensor quantities can be represented by
  extending parts into component space (e.g.
  `radial_component_of_<parent>_real_part`) when they arise. Revisit
  only if catalog demand surfaces.

### Summary of field decisions

| Field | Current | Decision |
|---|---|---|
| `validity_domain` | free-form str, 0/927 | **Delete** |
| `constraints` | list[str], 0/927 | **Delete** |
| `links` | list[str], 0/927 | **Keep, make required** (min_length=1 pre-sanitise) |
| `cocos_transformation_type` | str, 522/927 | **Exclude from catalog** (graph-only) |
| `complex` Kind | enum value, 6/927 | **KEEP**; generate missing canonical parents; mirror vector pattern |

## Link design: inline prose vs `see also` list

User question: "should links be in text or should we manage a list of
links that could be displayed with a `see also` line when rendered?"

### Recommendation: structured `links: list[str]` (keep current), render as `See also`

- **Why not inline prose**: inline prose cross-references are useful for
  flow, but they are:
  - ambiguous to parse (is `name:x` in a sentence a reference or a
    literal?),
  - fragile (typos kill the link silently),
  - not queryable from the graph (can't ask "what names reference X?").
- **Why structured list wins**:
  - Explicit `name:X` / `dd:Y` / `https://...` prefix tags each link's
    nature.
  - Pydantic validates targets before accepting.
  - `dd:` placeholders auto-upgrade to `name:` when the referenced DD
    path gets its own standard name (link phase).
  - Catalog site renders a clean `See also` block at the bottom of each
    entry.
  - Graph traversal: `MATCH (sn)-[:REFERENCES]->(other)` works.
- **Inline is still allowed** in prose when context demands it — prose
  can reference `electron_temperature` by name. But the canonical
  cross-reference record is the `links` list.

### Link generation strategy

Enrichment LLM sees, per name being enriched:
- the target name + its grammar decomposition,
- DD path(s) that anchor it,
- nearby standard names (vector-similarity over embeddings),
- sibling standard names (same physics_domain / same grammar base).

From this context, the LLM must emit 1–5 `links`, each either:
- `name:<existing_standard_name>` — verified against the graph at sanitize
  time; invalid → dropped with a warning,
- `dd:<ids>/<path>` — a placeholder for a DD path whose standard name
  doesn't exist yet. The link phase rewrites these to `name:X` when
  the target acquires a standard name.
- `https://...` — external reference (textbook, paper, ITER docs).

**Minimum-link enforcement**: the enrichment Pydantic response model
changes `links: list[str] = []` → `links: list[str] = Field(min_length=1)`.
Downstream graph coverage must rise from 0/927 to >90%.

## Grounding facility claims

User concern: "Great care needs to be taken when providing tokamak
examples — we need to ensure that the generating model is properly
grounded."

### Policy

- **Prefer regime-level ranges over machine-specific values**. "Central
  electron temperatures in fusion-relevant tokamak plasmas span 1–20 keV"
  is universal and safe. "In ITER, $T_e(0) \\approx 25\\,\\mathrm{keV}$
  at full performance" is machine-specific and **prohibited by default**.
- **Machine-specific numeric claims are PROHIBITED** unless the prompt
  context for that specific entry supplies grounding data (a wiki
  snippet, DD path annotation, or operator-domain handbook excerpt
  already injected into the enrichment prompt). Rationale from RD
  review: "must cite" rules are unenforceable against LLMs — they
  hallucinate citations. A presence-check (facility_name + numeric
  value co-occurrence) is reliable; a truth-check is not. Reject the
  presence pattern outright unless grounding material is in-context.
- **Hedge-phrase escape hatch**: if the LLM wishes to convey a
  characteristic regime value, it must use a hedge phrase
  (`typically`, `on the order of`, `in the range of X–Y`) with no
  facility name attached.
- **Uncertain → drop**. If the LLM is not confident in any number,
  it must omit it rather than invent.

### Review enforcement

`sn review` (or a dedicated docs-review pass) adds a check:
- Scan documentation for ITER|JET|DIII-D|TCV|WEST|AUG|MAST|KSTAR etc.
  mentions.
- For each, confirm a nearby citation OR a hedge ("typical values"
  without a specific number) OR flag as ungrounded.
- Score dimension: `grounding` (0-20) added to the 6-dim rubric →
  now 7-dim rubric, aggregate normalised to `sum / 140`.

## Range / unit consistency

User concern: "I see the following for faraday rotation `from a few
milliradians to several degrees depending on wavelength` we should never
mix units when describing ranges."

### Rule (new entry in `_enrich_style_guide.md`)

> **Range-unit consistency.** Numeric ranges in documentation must use
> the same unit on both bounds. Write `$10^{-3}$ rad to $0.1$ rad`
> (scientific notation if the span is wide) or `1 mrad to 100 mrad`,
> never `milliradians to degrees`. Convert to a single unit or quote
> bounds at each end with explicit unit labels in the same family.
>
> Exception: if the natural unit changes across regimes (e.g. "1 eV
> in SOL, 10 keV in core"), use the same family of units (eV) and
> scale as a prefix.

### Review enforcement

Add reviewer check: flag any documentation containing two distinct units
of the same quantity (rad & deg, m & cm, Pa & bar, eV & keV, etc.)
*within* a range construct ("from X to Y", "between X and Y", "X–Y",
"ranges from X to Y").

## Regeneration strategy

The key operational question: what do we re-run?

### Classification of changes

| Change | Requires LLM regen? | Cost scale |
|---|---|---|
| Drop `cocos_transformation_type` from export | No (filter-only) | Free |
| Drop `validity_domain` from schema | No (already 0/927) | Free |
| Drop `constraints` from schema | No (already 0/927) | Free |
| Reclassify 6 complex → scalar | No (Cypher UPDATE) | Free |
| Remove `complex` from ISN Kind enum | No (no complex entries after reclass) | Free |
| **Links required (min_length=1)** | **Yes — full enrich re-run for all 927** | moderate |
| **Grounded facility examples** | **Yes — re-enrich + re-review** | moderate |
| **Range-unit consistency** | **Yes — re-enrich + re-review** | moderate |

### Two-path proposal

Choose one:

- **Path L (light)**: schema-only drops + complex reclass + catalog
  re-export without LLM re-run. Update ISNC PR #1 with cleaned entries.
  **Links stay empty, facility grounding unchanged, range-unit issues
  persist.** Addresses ~40% of concerns with zero LLM cost.

- **Path F (full)**: schema drops + complex reclass + re-enrichment of all
  927 names with hardened prompts (required links, grounding rule,
  range-unit rule) + re-review with 7-dim rubric + re-export.
  Addresses 100% of concerns. Cost estimate:
  - Enrich: 927 × avg enrich cost ≈ $X (need to check `sn status` /
    historical per-name cost).
  - Review: 927 × avg review cost ≈ $Y.
  - Links resolution: sweeping pass, in-graph, free.

**Recommendation**: **Path F**, executed in small batches with cost caps
per physics domain (mirrors the original bootstrap loop). This is the
corrective cycle the user explicitly asked for at the start of the
session — "iterative bootstrap quality of names and descriptions".

## Implementation phases

### Phase 1 — Schema cleanup (ISN + codex, no LLM cost)

**Ordering note (from RD review):** the decision to KEEP `complex` in
the Kind enum removes the previously-flagged rc21-before-Cypher
ordering hazard. No graph reclassification is needed. However, a NEW
ordering hazard applies to Phase 1c (complex-part derivation): the
ISN changes (properties + validators) must ship in rc21 *before*
codex starts consuming them. Phase 1a → 1b → 1c is the correct order.

**1a. ISN schema changes** (`~/Code/imas-standard-names`):

- `imas_standard_names/models.py`:
  - Delete `validity_domain` field from `StandardNameEntryBase`.
  - Delete `constraints` field from `StandardNameEntryBase`.
  - Delete the corresponding `Constraints`, `Domain` type aliases if
    unused elsewhere.
  - Mark `cocos_transformation_type` as catalog-excluded (add
    `exclude=True` in model_config, or similar serialisation hint), OR
    delete entirely if graph can re-annotate downstream. Decision:
    **delete from `StandardNameBase`**. The field is codex-graph-only.
    If ISN wants it for rendering, re-add as a rendered-only computed
    field sourced from the graph sidecar — but not stored in YAML.
  - **KEEP** `complex` in the `Kind` enum (override of earlier B2
    recommendation). No changes to `ComplexEntry` / `ComplexNameOnly`
    beyond additions in Phase 1c.
  - Update docstrings to remove references to validity_domain /
    constraints. Retain complex-kind documentation.
- ISN tests: delete tests for dropped fields; tighten tests for
  remaining fields.
- ISN docs (`docs/`): scrub mentions of dropped fields. Retain
  complex-kind sections (will be extended in 1c).
- Cut release `v0.7.0rc21`.

**1b. Codex schema + graph** (`/home/ITER/mcintos/Code/imas-codex`):

- Bump ISN pin: `v0.7.0rc20` → `v0.7.0rc21`. `uv sync`.
- `imas_codex/standard_names/models.py`:
  - Delete `validity_domain` and `constraints` from the enrichment
    response models (`EnrichResponseItem`, etc.).
  - Delete references in audits.py, canonical.py, graph_ops.py (writers
    and readers).
- `imas_codex/schemas/standard_name.yaml`: drop
  `validity_domain`, `constraints` slots. `cocos_transformation_type`
  stays on the graph node but is added to an `is_private: true` or
  equivalent catalog-exclusion mechanism.
- Regenerate models: `uv run build-models --force`. Do not stage generated files.
- `imas_codex/standard_names/protection.py`: **remove**
  `validity_domain` and `constraints` from the `PROTECTED_FIELDS` set
  (currently at line 21). Post-deletion these fields do not exist to
  protect.
- Graph migration (Cypher, in place; no node deletion):
  ```cypher
  // Drop validity_domain / constraints (were already empty)
  MATCH (sn:StandardName)
  REMOVE sn.validity_domain, sn.constraints
  RETURN count(sn);
  ```
  No `complex`-reclassification migration needed (decision B1 keeps complex).
- Update `imas_codex/standard_names/export.py`:
  - Add `cocos_transformation_type` to the catalog-excluded set.
- Update `imas_codex/standard_names/canonical.py`:
  - Ensure `canonicalise_entry` does not emit excluded fields.
- Update `imas_codex/standard_names/catalog_import.py`:
  - Add `validity_domain`, `constraints`, `cocos_transformation_type` to
    the forbidden-key list (reject if catalog PR tries to reintroduce).
- All `tests/standard_names/` must stay green.
- Commit.

**1c. Complex parent/part infrastructure — mirror vector** (ISN rc21 + codex):

This phase encodes the design: a canonical complex parent is the
documentation anchor; `_real_part`, `_imaginary_part`, `_amplitude`,
`_phase`, `_magnitude`, `_modulus` are derived parts. Mirror the
existing `StandardNameVectorEntry.magnitude` property pattern exactly.

- **ISN additions** in `imas_standard_names/models.py`
  `StandardNameComplexEntry` (next to vector's `.magnitude`):
  ```python
  @property
  def real_part(self) -> str:
      return f"{self.name}_real_part"

  @property
  def imaginary_part(self) -> str:
      return f"{self.name}_imaginary_part"

  @property
  def amplitude(self) -> str:
      return f"{self.name}_amplitude"

  @property
  def phase(self) -> str:
      return f"{self.name}_phase"

  @property
  def modulus(self) -> str:
      return f"{self.name}_modulus"
  ```
  Apply the same properties to `StandardNameComplexNameOnly`.
- **ISN name-validation** (soft warning, not rejection): if a
  complex-kind entry's name ends in any of
  `{_real_part, _imaginary_part, _amplitude, _phase, _magnitude, _modulus}`,
  it is a **derived part**. A sibling parent (the stem with suffix
  stripped) should exist in the catalog. Warn-only in ISN; codex adds
  a hard audit below.
- **ISN test**: new `tests/test_complex_parent_pattern.py` verifies
  the five derived-name properties match the expected suffix strings,
  analogous to existing vector-magnitude tests.
- **ISN release**: all of 1a + 1c ship together as v0.7.0rc21.
- **Codex DD pair detection** in
  `imas_codex/standard_names/sources/dd.py`:
  - New `pair_complex_siblings(paths)`: groups sibling DD paths with
    matching stems and recognised pair suffixes
    (`{real, imaginary}`, `{r, i}`, `{amplitude, phase}`,
    `{magnitude, phase}`). Returns a list of `ComplexPairSource`
    dataclasses, each carrying the common stem and the two member
    path objects.
  - Extraction emits a **canonical-parent** `StandardNameSource`
    (kind=`complex`, no suffix) with `provenance.source_paths` linking
    BOTH part DD paths, plus one `StandardNameSource` per part (kind=
    `complex`, with suffix).
- **Codex compose-time rule** (`llm/prompts/sn/compose_system.md`,
  new rule **NC-20a** adjacent to the existing NC-20 suffix-form
  hard-prohibition):
  > **NC-20a — Canonical complex parent MUST precede its parts.**
  > When a source corresponds to one half of a complex pair (DD
  > siblings `.../real`+`.../imaginary` or suffix-pair names),
  > you MUST emit the canonical parent name first (the stem, no
  > suffix) as a `complex`-kind entry carrying the full physics
  > documentation. Then emit each part as a `complex`-kind entry
  > with an abbreviated description that references the parent by
  > `name:<parent>` link. Mirrors the vector convention: the parent
  > quantity (`plasma_velocity`) carries the documentation, and
  > `radial_component_of_plasma_velocity` references it implicitly.
- **Codex audit** — new `complex_parent_presence_check` in
  `imas_codex/standard_names/audits.py`:
  - Trigger: `kind == "complex"` AND name ends in one of the six
    recognised suffixes.
  - Check: stem (suffix stripped) exists as a `complex`-kind
    StandardName node in the graph (same `SNRun` scope).
  - Failure: append `complex_parent_missing: stem=<X>` to
    `validation_issues`; set `pipeline_status = needs_revision`.
    Runs AFTER compose so the parent can be emitted in the same
    batch.
- **Codex `derive_kind` refinement** in
  `imas_codex/standard_names/kind_derivation.py`:
  - Existing rule 5 (`real_part` / `imaginary_part` → `complex`)
    stays.
  - New rule 5a (before rule 5): if the source is a
    `ComplexPairSource` (extractor-supplied hint) AND the name has
    no suffix in the recognised set → `complex` (canonical parent).
- **Bootstrap the 6 orphan parts**: one-shot Cypher/CLI pass to
  regenerate canonical parents for the 6 existing orphans. Details
  in Phase 3 bootstrap (3a).

**1d. Acceptance**:

- `uv run pytest tests/standard_names/ tests/graph/` green in codex.
- `uv run pytest` green in ISN.
- `sn export --staging /tmp/isnc-staging-light/` produces entries with
  NO `validity_domain`, `constraints`, or `cocos_transformation_type`
  fields.
- All existing `kind: complex` entries retain `kind: complex`.
- `derive_kind("perturbed_electrostatic_potential_real_part")` returns
  `"complex"`.
- New ISN test `test_complex_parent_pattern.py` passes.

### Phase 2 — Prompt + response-model hardening (codex)

**2a. Link sanitizer fix (BLOCKING — must ship before 2b).**

Current `_sanitize_links` in `imas_codex/standard_names/enrich_workers.py`
(lines ~120–122) strips `dd:` prefixes as junk, rendering the plan's
`dd:<path>` placeholder strategy inoperative. Fix:

- Add `"dd:"` to `_LINK_PREFIXES` (the whitelist of recognised link
  schemes). Remove `"dd:"` from the junk-strip list.
- Extend the regex that validates post-strip content so `dd:<path>`
  with slashes is accepted (DD paths contain `/`). The existing
  `_SN_ID_RE` only allows `^[a-z][a-z0-9_]*$` — it must be joined
  with a DD-path regex for the `dd:` branch.
- Remove any "no `dd:` prefixes" instructions from the current
  `_enrich_style_guide.md`; those instructions contradict the new
  required-links policy.
- Unit test at `tests/standard_names/test_link_sanitize.py`: assert
  `dd:equilibrium/time_slice/profiles_1d/psi` survives sanitisation;
  `name:electron_temperature` survives; invalid `name:…` targets are
  dropped as today.

**2b. Prompt fragments**:

- `imas_codex/llm/prompts/shared/sn/_enrich_style_guide.md`: add
  - § Range-unit consistency (see sketched rule below).
  - § Facility grounding (regime-preferred; machine-specific numeric
    claims prohibited unless prompt context supplies grounding data).
- `imas_codex/llm/prompts/sn/enrich_system.md`:
  - Elevate `links` from "optional but encouraged" to **required**: "You
    MUST emit at least one `links` entry per name. Prefer `name:<x>`
    for existing standard names. Use `dd:<ids>/<path>` when referencing
    a DD path that has no standard name yet. External URLs
    (`https://...`) allowed for textbooks and peer-reviewed technical
    reports only."
  - Remove `validity_domain` and `constraints` from the output schema
    and field-constraints section.
  - Add: "Do NOT mix units inside a range construct. See
    range-unit-consistency rule."
  - Add: "Machine-specific numeric claims (e.g. 'in ITER, X ≈ Y keV')
    are PROHIBITED unless grounding data is present in your prompt
    context. Use hedge phrases (`typically`, `on the order of`,
    `in the range of`) with no facility name attached when describing
    regime-characteristic values."
- `imas_codex/llm/prompts/sn/review_docs.md`:
  - Add new review dimension: `grounding` (0-20).
  - Add scoring criteria: full score when all numeric claims are
    regime-level OR grounded via prompt-context citation; partial
    when one claim is machine-specific without grounding; fail when
    multiple machine-specific claims are ungrounded. The reviewer
    performs a **presence check** only (facility name +
    number co-occurrence), not a truth check.
  - Add range-unit-consistency check to scoring rubric.
  - Add link-quality guidance to `completeness` rubric: "Penalise
    `name:X` links to physically unrelated quantities."
- `imas_codex/llm/config/sn_review_criteria.yaml`:
  - Extend from 6-dim to 7-dim rubric (add `grounding`).
  - Update aggregate formula: `sum / 140.0` instead of `sum / 120.0`.
  - Recalibrate tier thresholds. `sn_review_criteria.yaml` currently
    uses absolute-sum cutoffs; translate each via `old * 140/120`
    and round to the nearest integer.
  - Update compose_system prompt's references to 6-dim → 7-dim.
- `imas_codex/llm/prompts/sn/compose_system.md`: add rule **NC-20a**
  (see Phase 1c; compose-time canonical-parent-first rule).

**2c. Range-unit-consistency regex — best-effort, warn-only**:

- Regex sketch (documentation scan; best-effort):
  ```
  # Catch: "<num> <unit_a> to <num> <unit_b>" where unit_a ≠ unit_b
  (\d+(?:\.\d+)?)\s*(\w+)\s+(?:to|–|—)\s*(\d+(?:\.\d+)?)\s*(\w+)
  ```
  Extract unit_a, unit_b; compare via `pint.Unit` equality. If
  distinct and dimensionally related (e.g. milliradians ↔ degrees),
  flag. Otherwise pass (e.g. "1 m to 10 s" is typo but genuinely
  unrelated — will produce false positives on these).
- **False-positive risk acknowledged**: mixed units CAN be legitimate
  (e.g. measurement uncertainty expressed in different units). Rule
  is warn-only, not reject. Reviewer scoring catches the
  physically-wrong cases.

**2d. Pydantic response model updates**:

- `imas_codex/standard_names/models.py`:
  - `EnrichmentResponseItem` (or equivalent): remove
    `validity_domain: str | None`, remove `constraints: list[str]`.
  - `links: list[str] = Field(min_length=1, ...)`.
    **Enforcement timing**: `min_length=1` is enforced at Pydantic
    parse time, BEFORE `_sanitize_links` runs. Rationale: we gate the
    LLM's raw output so retry logic triggers when the model emits no
    links. Post-sanitisation link count MAY be 0 for niche entries
    (all LLM-emitted links dropped as invalid). Those are tracked as
    `post_sanitization_empty_links` metric, not re-gated.
  - Add a validator that runs the range-unit-consistency regex
    against `documentation`. Flag as warning, not reject.

**2e. Graph coverage tests** (new):

- `tests/standard_names/test_enrich_fields.py`: after enrichment of a
  fixture cohort, assert
  - `links` coverage (at least one link post-sanitisation) ≥ 90%,
  - `validity_domain` and `constraints` properties do not exist on
    any StandardName node,
  - every `complex`-kind node with a recognised suffix has a stem
    sibling (complex-parent audit).

**2f. Rubric calibration — run BEFORE mass re-enrichment (Phase 2.5)**:

Per RD review: validate the new 7-dim rubric on the existing 480
exported entries BEFORE committing to Phase 3 regeneration.

- Re-run `sn review --dry-run` against current graph state (no
  writes).
- Produce a baseline score histogram across the 7 dimensions.
- Verify tier cutoffs behave sensibly (no mass-tier-flip from the
  120→140 formula change).
- Save baseline to `plans/features/standard-names/36-baseline-rubric.json`
  for Phase 3 comparison.
- If >10% of entries tier-flip unexpectedly, re-calibrate thresholds
  before proceeding.

**2g. Acceptance**:

- `pytest tests/standard_names/` green.
- `tests/standard_names/test_link_sanitize.py` passes (dd: links
  survive).
- `sn benchmark --models claude-sonnet-4.6,claude-opus-4.6 --max-candidates 5`
  scores the hardened prompts within ±5% of pre-change baseline on
  the 6 retained dimensions; `grounding` dimension scores calibrate
  around the median of existing entries.
- Phase 2.5 baseline captured.

### Phase 3 — Graph regeneration (cost-gated)

**3a. Pre-flight**:

- `sn status` to confirm baseline coverage: 927 names, expected
  breakdown by pipeline_status.
- **Selective-rollback baseline (primary)**: run Phase 1
  light-export to
  `/tmp/isnc-staging-baseline-$(date +%Y%m%d)/`.
  This YAML snapshot is the selective-rollback anchor —
  if a single domain degrades, `sn import` from this staging dir
  restores that domain's entries only, without touching the rest of
  the graph (per RD review: `sn import` with per-domain filter
  accomplishes this more surgically than a full-graph restore).
- **Full-graph backup (backstop)**: `uv run imas-codex graph backup`
  is the catastrophic-failure backstop, not the primary rollback.
- **Freeze ISNC PR #1** (per RD review): announce on PR #1 that it
  is frozen for the duration of Phase 3 — no merges, no branch
  updates. Phase 4 replaces the branch contents. Closing + reopening
  the PR loses review history; freezing preserves it.

**3a.1. Complex-parent bootstrap** (one-shot, before per-domain loop):

The 6 existing `complex` orphan parts (`*_real_part`,
`*_imaginary_part` of gyrokinetic eigenmodes) have no canonical
parents. Bootstrap them:

```bash
# Compute stems: strip recognised suffixes from each complex name
uv run imas-codex sn complex-bootstrap --dry-run
# Inspect the proposed 3 canonical parent names (stems shared across
# real+imaginary pairs), confirm, then:
uv run imas-codex sn complex-bootstrap --apply --cost-limit 1.00
```

This one-shot CLI (new, added in Phase 2) computes
`stems = {strip_suffix(n) for n in orphan_parts}`, emits a
compose+enrich+review cycle for each stem as a canonical `complex`
parent, and writes `links: [name:<part1>, name:<part2>]` on the
parent. The 6 orphan parts are updated with `links:
[name:<parent>]` in the link phase (3d).

**3b. Cost-calibrated re-enrichment loop** (domain-gated):

Per RD review: a flat `--cost-limit 2.00` across all domains
under-budgets large domains. At opus-4.6 pricing (~$15/M input,
~$75/M output, ~2000 tokens input + ~500 tokens output per
enrichment) a single enrichment costs ~$0.067; so `$2` caps a
domain at ~30 entries. Large domains (`equilibrium`, `core_profiles`,
`transport`) carry 80–150 names each and need $6–10.

Procedure:

```bash
# Get per-domain name counts
uv run imas-codex sn status --group-by physics_domain > /tmp/domain-counts.txt
```

Per-domain cost cap: `cap = max(2.0, count * 0.10)` (giving
10¢/name headroom including retries). Explicitly enumerate the
caps in this plan once `sn status` output is available; no
surprise budget overruns.

For each physics_domain, in priority order
(core_profiles → equilibrium → transport → magnetohydrodynamics →
auxiliary_heating → edge_plasma_physics → … 20 domains with names):

```bash
uv run imas-codex sn run \
  --only enrich \
  --domain <domain> \
  --cost-limit <per_domain_cap> \
  --force
```

`--force` bypasses the `PROTECTED_FIELDS` protection on
existing-enrichment fields because this is intentional regeneration.
ISNC PR #1 is frozen (see 3a) so there are no catalog edits to
overwrite.

**3c. Re-review loop** (7-dim rubric):

```bash
uv run imas-codex sn run \
  --only review \
  --domain <domain> \
  --cost-limit <per_domain_cap>
```

Quarantine rules:
- Aggregate score < 0.5 → quarantine + re-enrichment attempt (1
  retry).
- `grounding` dimension < 10/20 → quarantine + manual review.
- `completeness` < 10/20 AND `post_sanitization_empty_links` metric
  true → quarantine + re-enrichment attempt.

**3d. Link phase**:

```bash
uv run imas-codex sn run --only link
```

Rewrites all `dd:<path>` placeholders to `name:<target>` where the
target standard name now exists. Fully complete only after all
domains are re-enriched.

**3e. Per-domain quality audit (quantified)**:

After each domain's re-enrich + re-review pass, compare against
baseline (captured in Phase 2.5 + 3a):

| Metric | Threshold for ACCEPT | Else |
|---|---|---|
| Mean aggregate score | ≥ baseline_mean − 0.05 | ROLLBACK this domain |
| % entries with ≥1 post-sanitisation link | ≥ 90% | re-enrich this domain, 1 retry |
| `grounding` dim mean | ≥ 14/20 | re-enrich this domain, 1 retry |
| `complex_parent_missing` audit hits | 0 | manual parent-generation fix |

Rollback mechanism (per-domain):

```bash
uv run imas-codex sn import \
  --staging /tmp/isnc-staging-baseline-$(date +%Y%m%d)/ \
  --domain <failed_domain> \
  --force
```

This selectively restores the failing domain's entries from the
pre-regen YAML snapshot without touching other domains.

**3f. Manual spot-audit**: 5 names from each of top-10 domains = 50
entries, ~2 h human time. Confirm links are physically relevant
(not syntactically-valid-but-shallow), documentation has no
unit-mixing, no ungrounded machine-specific numerics.

### Phase 4 — Re-export and catalog update

**4a. Fresh export**:
```bash
rm -rf /tmp/isnc-staging/
uv run imas-codex sn export --staging /tmp/isnc-staging/ --min-score 0.65
```

**4b. Preview**:
```bash
uv run imas-codex sn preview --staging /tmp/isnc-staging --port 8765
```

Human spot-check: walk through 5 domains; confirm `See also` blocks
render with `name:` links (NOT `dd:` placeholders — those should all
have been resolved in Phase 3d); confirm no unit-mixing; confirm no
ungrounded facility claims; confirm complex-parent entries render
correctly alongside their parts.

**4c. Publish**:
```bash
uv run imas-codex sn publish --staging /tmp/isnc-staging \
  --isnc ~/Code/imas-standard-names-catalog --push
```

Force-updates the frozen ISNC PR #1 branch with the regenerated
entries. CI must stay green. Unfreeze PR #1 and request re-review.

**4d. Acceptance**:

- ISNC PR #1 CI green (validate + catalog-site).
- Round-trip idempotent: `sn import --dry-run` shows 0 `catalog_edit`
  flips.
- Catalog preview renders `See also` blocks with real cross-references;
  ≥ 95% of visible links are `name:` (not `dd:` placeholders).
- No entry contains `validity_domain`, `constraints`, or
  `cocos_transformation_type`.
- **Complex-kind entries retained**; every `*_real_part` /
  `*_imaginary_part` has a canonical parent sibling; their `links`
  contain the parent via `name:<stem>`.
- Grounding-score ≥ 14/20 on ≥ 90% of entries.
- `post_sanitization_empty_links` ≤ 10%.

### Phase 5 — Documentation

**5a. AGENTS.md updates**:
- Schema table: remove validity_domain, constraints row.
- Add links-required note + `dd:` placeholder convention.
- Add range-unit-consistency rule reference.
- Add complex parent/part convention (mirrors vector
  parent/component).
- Grounding scoring dimension: update rubric count 6 → 7.

**5b. ISN docs**: scrub dropped fields. Add complex parent/part
section alongside vector parent/component.

**5c. ISNC README**: note "validity_domain and constraints fields
removed in schema v2.1; rely on tags + documentation for regime and
constraint information. Complex quantities now follow a
parent/part convention: the canonical complex name (no suffix)
carries the full documentation; its `_real_part` / `_imaginary_part` /
`_amplitude` / `_phase` / `_magnitude` / `_modulus` derivatives
reference it via `See also`."

**5d. Plan cleanup**:
- Move plan 36 to `plans/features/standard-names/pending/` after Phase
  4 merges, or delete if fully implemented.

## Open questions for RD review (v2 — post-critique)

1. **Link minimum count — hard gate pre-sanitise, soft-track
   post-sanitise**. RESOLVED. `min_length=1` at Pydantic parse time;
   post-sanitisation count may be 0, tracked as metric.
2. **Grounding enforcement — presence-check, not truth-check**.
   RESOLVED. Reviewer flags facility+number co-occurrence without
   grounding-material in prompt context; does not attempt to verify
   truth of claims.
3. **`complex` retention — KEEP + mirror vector pattern**. RESOLVED
   per user steer. 6 orphans get canonical parents generated in 3a.1.
4. **Regeneration path — Path F with validated calibration**.
   RESOLVED. Phase 2.5 validates 7-dim rubric on existing 480 entries
   before committing to mass re-enrichment; tiers recalibrated if
   >10% flip.
5. **Cost caps — per-domain, proportional**. RESOLVED.
   `cap = max(2.0, count * 0.10)`. Enumerate per-domain caps once
   `sn status` output is captured.
6. **Rollback — selective via `sn import` from YAML**. RESOLVED.
   Full graph dump is backstop only; per-domain regression restored
   from Phase 3a baseline YAML.
7. **Tag namespacing for regime info** (deferred). Introduce
   `regime-*` namespace only if filter demand emerges post-regen.
   Non-blocking.

## Non-goals

- Re-generating **names** themselves. This plan touches enrichment
  only. Name minting stays frozen.
- Changing the `tags` vocabulary or scoring rubric beyond adding
  `grounding`.
- Removing `complex` from the `Kind` enum (override of earlier B2
  recommendation).
- Adding a `PART_OF` graph relationship for complex parts (mirrors
  vector: relationship is purely name-structural).
- Adding rank-aware complex variants (`complex_scalar`, etc.).

## Rollback

**Primary (per-domain, surgical)**: `sn import` from the Phase 3a
baseline YAML snapshot restores a single failing domain's entries
without touching others. Invoke on any domain that fails the Phase
3e quality gate.

**Backstop (catastrophic failure)**: the full `imas-codex graph
backup` taken in Phase 3a restores the entire pre-regen state.

Phase 1 (schema cleanup) and Phase 2 (prompt hardening) can stay in
place independently of a Phase 3 rollback — they contain no LLM
output that would need to be unwound.
