# Plan 36 — Catalog Quality Refactor (v4+round4)

> **Status**: READY FOR FLEET DISPATCH (RD round 4 complete, 6 findings folded at end of file)
> **Supersedes**: v1 (complex parent/part addition), v2 (complex-SUFFIX asymmetric), v3 (postfix + linking)
> **v4 scope additions**: graph-backed DD context amplification in SN prompts (Deltas A–J);
> target-anchored dynamic example library split by consumer with per-dimension reviewer
> reasoning (Delta K); rubric unification (publish threshold = `good` tier floor = 0.65;
> `adequate` → `inadequate`); removal of static `benchmark_calibration.yaml`.
> **RD round 2** findings (on v2) are either resolved by postfix (#1) or merged into this
> plan (#4, #6, #7, #8a/b, #1b/d/e, #5). **RD round 3** findings folded into Phase-specific
> fix tables.

---

## Problem

Plan 35 shipped a working bootstrap; the 480+ catalog produced from it exposed three
classes of defect that are not addressable by prompt nudges alone:

1. **Field design flaws** — `validity_domain` / `constraints` are empty for most entries,
   `cocos_transformation_type` is a graph concern leaking into the catalog, and the
   current `links` workflow is broken end-to-end (0 links on all 927 names).
2. **Grammar inconsistency** — vector parent/component uses PREFIX form
   (`radial_component_of_X`), reduction tokens (`magnitude_of_`) are also PREFIX, but
   5 of 7 other SEGMENT_RULES are already POSTFIX. This forces the grammar to carry two
   contradictory composition directions.
3. **Complex-number gap** — Fourier / FFT / reflectometry quantities have no first-class
   parent/part schema, producing 6 ad-hoc orphans with made-up suffix form that the
   grammar can't parse.

**v3's single reframe**: fix the grammar first (postfix inversion unifies everything),
then rebuild the linking workflow, then regenerate under a small cost-bound loop that
proves link resolution actually works before opening the taps.

---

## Decisions resolved before RD round 3

### D1 — POSTFIX grammar inversion (user greenlit)

Under postfix, there is ONE rule: modifiers strip right-to-left from a closed reserved
vocabulary appended as suffixes. This subsumes both the vector-component pattern
(`plasma_velocity_radial_component`) and the complex real/imaginary/amplitude/phase
parts (`density_fluctuation_real_part`) under the same mechanism.

| Aspect | Prefix (v2) | Postfix (v3) |
|---|---|---|
| Rules | 2 (prefix for vector+reductions, postfix for obj/geom/pos/region/process) | 1 (all postfix) |
| Lexicographic grouping | Components scatter around the alphabet | Parent + all its parts sort adjacently |
| Parser disambiguation | Forward + backward scan needed | Right-to-left strip only |
| LLM composability | Must know two composition directions | Single rule |
| Search anchoring | `STARTS WITH` gives the base | `STARTS WITH` gives the base + all parts (prefix = common) |
| Operator order | `magnitude_of_plasma_velocity` ⟂ math convention | `plasma_velocity_magnitude` = `‖v‖` order |

**Closed reserved postfix vocabulary** (user accepted):

- Component axes: `_radial_component`, `_poloidal_component`, `_toroidal_component`,
  `_parallel_component`, `_perpendicular_component`, `_vertical_component`,
  `_diamagnetic_component`
- Scalar reductions (kind ∈ {vector}): `_magnitude`, `_flux_surface_average`,
  `_volume_integral`, `_root_mean_square`, `_time_average`
- Complex parts (kind = complex): `_real_part`, `_imaginary_part`, `_amplitude`,
  `_phase`, `_modulus`
- Derivatives: `_radial_derivative`, `_time_derivative`
- Per-mode: `_per_<mode>` (e.g. `_per_toroidal_mode`)

**Open questions for RD round 3**: is `_amplitude` safe alongside `_magnitude`
(`_amplitude` on vector → ambiguous with magnitude)? Should `_amplitude` be scoped to
`kind=complex` only?

### D2 — Remove `_magnitude` from complex suffix set (RD #1d)

Complex `|z|` canonical suffix is `_modulus`. `_magnitude` is reserved for the vector
reduction. Avoids `density_fluctuation_magnitude` and `density_fluctuation_modulus`
being treated as synonyms.

### D3 — Export excludes `dd:` links (RD #4 BLOCKING)

The catalog exported to the ISN repository is the normative artifact. Any unresolved
`dd:` link in the graph is internal state, not a catalog fact. `export.py` gains a
field-exclusion mechanism (which does not exist today) that strips any `dd:`-prefixed
link from exported entries. If `link_status == 'unresolved'`, emit a warning and
exclude the whole entry from the export so consumers never see half-resolved catalog
data.

### D4 — Documentation-text `dd:` placeholders use `doc_resolution_status`

User asked: "do we need a separate status flag for when documentation/links still
resolves to dd: links instead of SN ones?" — YES, parallel field.

- `link_status`: already exists for the `links` field (resolved/unresolved/None)
- `doc_resolution_status`: NEW field, same enum, scans `description` +
  `documentation` for `dd:<path>` placeholders.

Two fields, not one unified — because they resolve at different times (links during
enrich turn, docs may need a separate doc-resolve pass) and catalog export needs to
know per-field whether to emit, reject, or strip.

### D5 — Sibling auto-populate for vector/complex parents (user requested)

After a batch completes compose, a post-step queries all names matching
`{parent}_<reserved_suffix>` and appends them as `name:` links to the parent. Each
component reciprocally gets `name:{parent}` in its links. Runs after link resolution
phase so missing siblings are skipped safely (will be picked up on later turn when
claim_unresolved_links re-sweeps).

### D6 — Range-unit consistency validator + reviewer routing (RD #7)

User-reported case: Faraday rotation documentation says *"from a few milliradians to
several degrees depending on wavelength"* — mixes units in a single range. New
validator scans description/documentation for range patterns with mixed units; flags
as `validation_issues` and the reviewer rubric gains a −2 point penalty on
completeness when the validator tagged.

### D7 — Tokamak example grounding (user requested)

Reviewer prompt gains explicit instruction: if the entry cites a facility parameter
(`"in ITER elongation is typically..."`), the reviewer must confirm the parameter
against its own knowledge. Wrong-by-knowledge numbers are treated as factual errors
(completeness deduction).

---

## Critical review of field design (inherited from v2, trimmed)

### `validity_domain: str = ""` — REMOVE

`validity_domain` is empty for most entries. Its signal belongs in `documentation`
prose ("valid in the core plasma", "valid above the X-point"). Keeping it as a field
invites empty strings and parse ambiguity. Remove from the schema.

### `constraints: list[str] = []` — REMOVE

Same reasoning. Constraints belong in `documentation` prose when physical
("non-negative"), or are algorithmic/computational (belong in calling code, not the
catalog).

### `links: list[str] = []` — KEEP, FIX WORKFLOW

Structural. This is the field whose generation pipeline is broken (11 bugs in v3
Phase 2).

### `cocos_transformation_type: str | None` — REMOVE from catalog entries

COCOS transformation is a graph relationship to a COCOS node (`(StandardName)-[:HAS_COCOS]->(COCOS)`).
It's applied selectively (not all quantities COCOS-transform). Exporting a string
field that is null for most entries adds noise without utility. Graph-only concern.

### `kind` enum — ADD `complex`, `vector`, `tensor`

User directive. Closed enum: `scalar`, `vector`, `tensor`, `complex`,
`metadata`. `complex` is a shape descriptor independent of rank (scalar vs vector
complex-valued quantities exist — e.g. Fourier coefficient is complex;
polarization tensor is complex-tensor).

**RD round 3 question**: do we need distinct `complex-vector` / `complex-tensor`
kinds, or is `kind=complex` + grammar structure (`*_real_part` etc) sufficient?
Current recommendation: start with `complex` only; `complex-vector` /
`complex-tensor` added on demand when first DD-sourced occurrence lands.

### Summary of field decisions (v3)

| Field | Action | Rationale |
|---|---|---|
| `validity_domain` | REMOVE | Empty for most; belongs in docs prose |
| `constraints` | REMOVE | Empty for most; belongs in docs prose |
| `cocos_transformation_type` | REMOVE from catalog | Graph-only concern |
| `kind` | EXTEND to {scalar, vector, tensor, complex, metadata} | User-requested |
| `links` | KEEP, REBUILD WORKFLOW | 11 bugs in Phase 2 |
| `link_status` | KEEP | Already exists; wire correctly |
| `doc_resolution_status` | ADD | Parallel to link_status for doc-text placeholders |

---

## Link workflow — 11 bugs (Phase 2 fixes)

Empirical baseline: **ALL 927 names in graph have 0 links and `link_status=None`**.
The workflow is fully non-functional. Cause: stacked bugs B1+B2+B3+B5 form a closed
null loop; B7 closes any back-door survival.

| # | Bug | Location | Severity | Fix |
|---|---|---|---|---|
| B1 | `_sanitize_links` strips `dd:` as junk | `enrich_workers.py:120-122` | BLOCKING | Add `dd:` to `_LINK_PREFIXES`; remove from junk-strip list |
| B2 | Contradictory LLM instructions: system prompt says `"MUST use name:"`, style-guide says `"No dd:, no URLs, no name:"` | `enrich_system.md:66` vs `_enrich_style_guide.md:60` | BLOCKING | Rewrite `_enrich_style_guide.md:57-63` to permit `name:`, `dd:`, `https://` |
| B3 | Zero links across 927 names (consequence of B1+B2) | Empirical | BLOCKING | Resolved by B1+B2+B8 |
| B4 | No sibling auto-population for vector/complex parents | Missing feature | BLOCKING (user req) | New `_populate_sibling_links` pass in `enrich_workers.py` |
| B5 | `link_status` never reaches `unresolved` (no `dd:` ever written) | `graph_ops.py:81-91` | HIGH | Resolved by B1+B2+B8 |
| B6 | `missing_reverse` audit requires A→B ⇒ B→A (over-strict for parent/component) | `review/audits.py:408-420` | MEDIUM | Scope check: skip symmetry assertion when source is parent and target is its component (or vice versa) |
| B7 | `_sanitize_links` silently drops `name:X` when X ∉ `valid_names` — kills bootstrap | `enrich_workers.py:114-117` | HIGH | If X ∉ `valid_names` AND origin has known DD path, rewrite as `dd:<path>`; else drop with WARN |
| B8 | LLM never receives the DD-path list of the current batch — no grounding signal | Missing prompt context | HIGH | New Jinja `{% for path in batch_dd_paths %}` block in `enrich_user.md` enumerating DD paths the LLM may emit as `dd:` placeholders |
| B9 | No re-queue when `failed` names later become resolvable | `graph_ops.py:2327-2328` | MEDIUM | On new SN persist, run `reactivate_failed_links(new_name_id)` to flip any failed name whose unresolved target matches the new name back to `unresolved` with retry_count reset |
| B10 | `#` in junk-strip collides with URL fragments | `enrich_workers.py:120` | LOW | Remove `#` from junk prefix list |
| B11 | DD semantic-cluster siblings / identifier-schema relatives not used as candidate links | Missing feature | **MEDIUM → MERGED INTO 2c** | Phase 2c.2 (graph-backed cluster-peer injection) |

### Link generation strategy (v3)

1. **Compose/enrich emit `dd:<path>` placeholders** freely, drawn from the
   prompt-provided DD-path context (B8). No longer gated on `valid_names` (B7).
2. **Persist writes with `link_status='unresolved'`** whenever any `dd:` link is
   present.
3. **Resolution phase** (`_run_link_phase`) sweeps unresolved names, rewrites
   `dd:<path>` → `name:<id>` when `(IMASNode {id: path})-[:HAS_STANDARD_NAME]->(sn)`
   exists.
4. **Sibling auto-populate** (D5): after batch commit, for each `kind ∈ {vector, complex}`
   parent, query `MATCH (sn:StandardName) WHERE sn.id STARTS WITH parent.id + '_'` (postfix
   prefix-scan) + reserved suffix membership, append as `name:` links; reciprocate on each
   part.
5. **Failed re-queue** (B9): on any new SN write, check if any `failed` name now has a
   resolvable target; flip to `unresolved`, reset retry counter.
6. **Export** (D3, RD #4): any name still `unresolved` is excluded from catalog
   output; `dd:` links stripped from all surviving entries' `links` arrays.

---

## Range / unit consistency (RD #7, D6)

### Validator rule (new `range_unit_consistency_check`)

Scan `description` and `documentation` for patterns matching range descriptions
(`"from X to Y"`, `"between X and Y"`, `"X–Y"`). For each match, extract units
(`milliradians`, `degrees`, `keV`, `eV`, etc.). If a range mixes units from different
families (angle, energy, length, time), flag as `range_unit_mixed` in
`validation_issues`.

### Reviewer rubric deduction

`review_docs.md` gains: *"If validation_issues contains `range_unit_mixed`, deduct 2
points from Completeness."*

---

## Tokamak example grounding (D7)

### Policy

Any description or documentation text that cites a facility parameter (regex:
`in (ITER|JET|TCV|DIII-D|WEST|EAST|KSTAR|NSTX) .* (typical|roughly|approximately|about)`)
gets flagged for reviewer scrutiny. Reviewer prompt addendum:

> When the entry cites a facility parameter, validate the cited value against your
> own physics knowledge of that device. If the cited value is wrong (e.g. claims ITER
> elongation is 2.5 when it is 1.85), treat it as a factual error and deduct 2 points
> from Completeness. If you are not confident in the parameter, flag it in comments.

---

## Regeneration strategy

### Classification of changes (v3)

| Change | Regeneration trigger |
|---|---|
| Postfix grammar inversion (Phase 0) | **Full clear** of vector + complex + anything with component/reduction suffix |
| Field removals (validity_domain, constraints, cocos_transformation_type) | Property drop + export regen (no LLM cost) |
| `kind` enum extension | Graph-side reclass of existing complex orphans (no LLM cost) |
| Link workflow fixes | **Regenerate all 927 names' links** via revised enrich pass (links-only enrichment, skips name+desc regen) |
| Sibling auto-populate | Graph-side post-step (no LLM cost) |

### Two-path proposal

**Path A (SELECTED — user directive)**: clear SN graph, regenerate domain-by-domain
under small cost-bound validation loop. Cheapest moment to flip postfix convention
and rebuild links; zero rename script needed.

**Path B** (rejected): hot-migrate 927 existing names via grammar round-trip +
rename table. Too brittle; postfix invalidates all component-bearing names anyway.

### Cost-bound validation loop (Phase 4)

First domain is a **small verifier run** (≤20 names, cap $2) with ALL fixes enabled.
Gate: verify links are populated, `link_status` transitions correctly, sibling
auto-links appear on vector parents, complex parent/part round-trip parses. Only
after the verifier run passes does the full domain rotation resume.

---

## Implementation phases

### Phase 0 — ISN postfix grammar inversion (NEW, BLOCKING for Phase 1+)

#### Phase 0a — ISN upstream changes (rc22)

Files in `~/Code/imas-standard-names`:

1. `imas_standard_names/grammar/constants.py`:
   - Invert `component` SEGMENT_RULE from PREFIX template `{token}_component_of` to
     POSTFIX template `_{token}_component` (SEGMENT_TEMPLATES entry updated).
   - Add postfix reduction tokens to SEGMENT_TEMPLATES: `_magnitude`, `_time_average`,
     `_volume_integral`, `_root_mean_square`, `_flux_surface_average`.
   - Add complex-part postfix segment: `_real_part`, `_imaginary_part`, `_amplitude`,
     `_phase`, `_modulus`. Each with closed-vocab single token.
   - Add derivative postfix segment: `_radial_derivative`, `_time_derivative`.
   - Add `_per_<mode>` postfix segment with vocabulary {`toroidal_mode`, `poloidal_mode`,
     `radial_mode`, `mode`}.

2. `imas_standard_names/reductions.py`:
   - Rewrite `REDUCTION_PATTERNS` from prefix (`magnitude_of_`) to suffix (`_magnitude`).
   - Parser: change from left-to-right "strip prefix" to right-to-left "strip suffix".

3. `imas_standard_names/models.py`:
   - `StandardNameVectorEntry.magnitude` property: return `f"{self.name}_magnitude"` (was `f"magnitude_of_{self.name}"`).
   - Add `StandardNameComplexEntry` class with `.real_part`, `.imaginary_part`,
     `.amplitude`, `.phase`, `.modulus` postfix-form properties. `kind = 'complex'`.
   - All 18 existing validators reviewed for prefix assumptions; no functional change
     expected (validators operate on decomposed segments, not raw strings).

4. `imas_standard_names/grammar/parser.py`:
   - `parse_standard_name`: invert scan direction. Start from end of string, strip
     reserved postfix tokens greedily, yield segments in (base, suffix-list) form.

5. `imas_standard_names/tests/`:
   - Round-trip tests covering all 7 segment types.
   - Parse tests for `plasma_velocity_radial_component`, `density_fluctuation_real_part`,
     `electron_temperature_volume_integral`, etc.
   - `test_grammar_specification.yaml` regenerated.

6. `imas_standard_names/schemas/specification.yaml`:
   - Update prose to describe the single postfix rule.

**Acceptance**:
- All existing ISN tests pass under postfix.
- Round-trip `compose → parse → compose` is identity for all 7 segment types.
- `imas_standard_names/examples/` catalog regenerated (if any vector examples exist).
- Release as rc22, push to PyPI.

#### Phase 0b — Codex ISN pin bump

1. `pyproject.toml`: bump `imas-standard-names` to rc22.
2. `uv lock --upgrade-package imas-standard-names && uv sync`.
3. Run `uv run pytest tests/standard_names/` — expect grammar-related test breakage in
   codex test suite (to be fixed in 0c).

#### Phase 0c — Codex grammar / prompt / audit updates

1. `imas_codex/llm/prompts/sn/compose_system.md`:
   - **NC-19 INVERTED**: was *"COMPONENT PRECEDES BASE"*, now *"COMPONENT FOLLOWS BASE"*.
     Example block rewritten: `radial_component_of_plasma_velocity` → `plasma_velocity_radial_component`.
   - **NC-20 DELETED**: complex suffix mandate becomes redundant under unified postfix
     grammar (NC-19 now covers it).
   - Add new "POSTFIX MODIFIER LIBRARY" section listing all closed-vocab suffix tokens
     with one-line semantics.

2. `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`:
   - Replace prefix examples with postfix examples throughout.
   - Add segment-order table showing subject → base → position → geometry → region →
     object → process → component/reduction/derivative → per-mode.

3. `imas_codex/standard_names/audits.py`:
   - `amplitude_of_prefix_check` INVERTED to `amplitude_prefix_reject` — rejects
     `amplitude_of_*` and accepts `*_amplitude`.
   - NEW `derived_part_parent_presence_check` (unified for vector + complex): if name
     ends in any reserved suffix token, verify the stripped base exists as a parent
     `kind ∈ {vector, complex}` StandardName.
   - `kind_derivation` suffix rules rewritten: strip from right, match closed vocab,
     derive `(kind, base)`.

4. `imas_codex/standard_names/models.py`:
   - Any prefix-based regex or string-contains in Pydantic validators inverted to
     suffix-based.

**Acceptance**:
- `uv run pytest tests/standard_names/` passes.
- Existing prompts render cleanly (Jinja).
- `sn generate --dry-run` on a small test sample emits postfix names.

---

### Phase 1 — Schema cleanup + export exclusion mechanism

#### Phase 1a — Remove fields from schemas

1. `imas_codex/schemas/standard_name.yaml`:
   - REMOVE `validity_domain`, `constraints`, `cocos_transformation_type` slots.
2. `imas-standard-names` mirror: same removals in rc22 (Phase 0a).
3. `uv run build-models --force` → regenerate `models.py`, `dd_models.py`.
4. Graph migration (inline Cypher, not script):
   ```cypher
   MATCH (sn:StandardName)
   REMOVE sn.validity_domain, sn.constraints, sn.cocos_transformation_type
   RETURN count(sn) AS updated;
   ```

#### Phase 1b — Export field-exclusion mechanism (RD #4 BLOCKING)

`imas_codex/standard_names/export.py` currently has no exclusion mechanism; uses
`model_dump(mode="json")` with None-filter only.

1. Add `EXCLUDED_FIELDS: set[str] = {"cocos_transformation_type", "link_status",
   "doc_resolution_status", "embedding", "embedding_dim", ...}` (graph-only fields).
2. After `model_dump`, strip keys in `EXCLUDED_FIELDS`.
3. Add `LINK_PREFIX_EXCLUSIONS: set[str] = {"dd:"}`. When serialising `links`, drop
   any entry matching an excluded prefix. Record via `export_warnings` counter.
4. If `link_status == 'unresolved'` OR `doc_resolution_status == 'unresolved'`,
   EXCLUDE the entire entry from export; log at WARNING; surface in export summary
   report.

**Acceptance**:
- Zero `dd:` in exported catalog.
- Zero entries with unresolved status in exported catalog.
- Export summary reports counts.

#### Phase 1c — Add `doc_resolution_status` field (D4)

1. `imas_codex/schemas/standard_name.yaml`: add `doc_resolution_status` enum field
   (values: `resolved`, `unresolved`, `failed`, None).
2. `graph_ops.py`: `_compute_doc_resolution_status(description, documentation)`
   scans for `dd:<path>` placeholders in prose; returns `unresolved` if any present,
   else None.
3. Wire into write paths alongside `_compute_link_status`.
4. `_run_doc_resolve_phase` in `turn.py` (new, sibling to `_run_link_phase`): claim
   unresolved, re-scan prose text, replace `dd:<path>` with `name:<id>` where
   `(IMASNode {id})-[:HAS_STANDARD_NAME]->(sn)` resolves. Multi-round with
   `_MAX_RESOLVE_ROUNDS`.

#### Phase 1d — Non-DD canonical parent hand-tag policy (RD #1e)

1. Add `origin: Literal["dd", "signal", "manual"]` to StandardName schema.
2. `sn hand-tag` CLI subcommand: create a single StandardName manually with
   `origin='manual'`; exempt from `derived_part_parent_presence_check` only if at
   least one derived part exists as a downstream entry.

#### Phase 1e — Delete stale B2 rows (RD #8b)

Remove the complex-kind rows from `constants.py` KIND enum that were auto-generated
by plan 35 as ad-hoc entries; handled wholesale by Phase 0c's unified kind
derivation.

---

### Phase 2 — Linking workflow rebuild (11 bugs)

#### Phase 2a — `_sanitize_links` fix (B1, B10)

`imas_codex/standard_names/enrich_workers.py`:

```python
_LINK_PREFIXES: tuple[str, ...] = ("name:", "dd:", "http://", "https://")
_LINK_JUNK_PREFIXES: tuple[str, ...] = ()  # was ("dd:", "standard_name:", "sn:", "#")
```

Remove the entire junk-prefix strip; the valid-prefix allow-list is sufficient.

#### Phase 2b — Prompt-contradiction fix (B2)

1. `imas_codex/llm/prompts/shared/sn/_enrich_style_guide.md` lines 57–63: rewrite to:

   > **Links**: cross-reference related quantities via the `links` field. Permitted
   > prefixes:
   > - `name:<existing_standard_name_id>` when the referent is already a minted
   >   StandardName (visible in the "KNOWN STANDARD NAMES" prompt block).
   > - `dd:<imas_path>` when the referent is a DD path listed in "AVAILABLE DD
   >   PATHS" but not yet minted — will be resolved to `name:` by the pipeline.
   > - `https://…` for external references (e.g. ITER physics basis chapter).
   > Never invent a `name:` that isn't in the known list. Prefer `dd:` over
   > omission.

2. `imas_codex/llm/prompts/sn/enrich_system.md` line 66: align verbiage; replace
   *"MUST use the `name:foo_bar` prefix"* with the tri-prefix rule from the style
   guide.

#### Phase 2c — Graph-backed link-context injection (B8 + B11 unified)

The graph already provides far richer link-candidate infrastructure than a flat DD
path list. `IN_CLUSTER` (22,841 edges) groups semantically-related IMASNodes;
`HAS_STANDARD_NAME` gives free `dd:` → `name:` pre-resolution;
`standard_name_desc_embedding` vector-searches minted SN descriptions. Phase 2c
replaces the flat-list approach with three graph-backed context blocks, absorbing
bug **B11** (previously marked optional) as mandatory.

##### 2c.1 Source-path context (already in batch data)

Each source DD path's full description, unit, physics_domain, keywords. Already
loaded by the extract phase; no new code.

##### 2c.2 Cluster-peer block (traversal-backed)

New `imas_codex/standard_names/link_context.py` :: `build_cluster_peer_context(
batch_source_paths, cluster_peer_cap=8, cluster_scope_priority=('domain','global'),
exclude_cluster_types=('cocos',)) -> list[LinkCandidate]`.

Verified Cypher (tested in REPL):

```cypher
UNWIND $paths AS src_id
MATCH (src:IMASNode {id: src_id})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN $scope_priority
  AND NOT any(t IN $exclude_types WHERE c.id STARTS WITH t)
WITH src, c
ORDER BY CASE c.scope WHEN 'domain' THEN 0 ELSE 1 END
WITH src, collect(DISTINCT c)[..3] AS clusters
UNWIND clusters AS cluster
MATCH (peer:IMASNode)-[:IN_CLUSTER]->(cluster)
WHERE peer.id <> src.id
  AND peer.node_category = 'quantity'
  AND NOT (peer)-[:DEPRECATED_IN]->(:DDVersion)
OPTIONAL MATCH (peer)-[:HAS_STANDARD_NAME]->(sn:StandardName)
WITH src, cluster, peer, sn
LIMIT $cap
RETURN src.id AS source, cluster.description AS cluster_context,
       peer.id AS peer_path, peer.unit AS unit,
       substring(peer.description, 0, 140) AS peer_doc,
       sn.id AS minted_name
```

Each peer emitted as either `name:<minted>` (pre-resolved when `HAS_STANDARD_NAME`
edge exists) or `dd:<peer_path>` (placeholder). Cluster-scope priority defaults to
**domain > global** for domain-scoped compose runs (`--physics-domain X`), and
**global > domain** for cross-domain bootstrap runs. `cocos_*` clusters excluded
by default (transformation-shared noise, not physics-shared peers) — can be
re-enabled via CLI flag for COCOS-specific compose runs.

##### 2c.3 Semantic-neighbour block (vector-index-backed)

For each source DD path with a non-empty description, vector-search
`standard_name_desc_embedding` with top-5 minted-SN neighbours (similarity ≥ 0.75).
Offered as `name:` candidates. Implementation: reuses `semantic_search()` helper
from `imas_codex/graph/vector_search.py`. Skipped on early domains when few minted
SNs exist.

##### 2c.4 Prompt template integration

New `compose_user.md` and `enrich_user.md` Jinja block:

```jinja
{% if link_candidates %}
## RELATED REFERENCES (link to these via the `links` field)

The following are semantically related to the current batch. Prefer `name:` when
shown; use `dd:` for DD paths not yet minted (pipeline will resolve later).

{% for src in link_candidates %}
### For source `{{ src.path }}`:

**Cluster peers** ({{ src.cluster_context }}):
{% for p in src.peers %}
- `{{ p.tag }}` [{{ p.unit }}]: {{ p.doc }}
{% endfor %}

{% if src.semantic_neighbours %}
**Similar minted names**:
{% for sn in src.semantic_neighbours %}
- `name:{{ sn.id }}`: {{ sn.description_short }}
{% endfor %}
{% endif %}
{% endfor %}
{% endif %}
```

##### 2c.5 Token-budget analysis

Per-source cost: (8 peers × ~130 chars) + (5 SN-neighbours × ~130 chars) ≈ 1.7 kB.
Per-batch (N=5-10): 8-17 kB. Static system section (~4-5 kB) remains cacheable.
Target split: ~40% cacheable / ~60% dynamic. Peer cap tuneable via
`[tool.imas-codex.sn.linking]` in pyproject.toml.

##### 2c.6 Pre-resolution correctness

Pre-resolving `dd:X` → `name:Y` at prompt-construction time carries zero additional
staleness risk: (a) the graph read is the same source of truth the resolution loop
uses later, (b) if `Y` is deleted/renamed between prompt construction and response
write, the B7 graceful-fallback `_sanitize_links` path handles it — `name:Y` that
doesn't exist at write time gets logged and dropped or (if the origin DD path is
recoverable) rewritten back to `dd:X` for the resolution loop to retry.

##### 2c.7 CLI tuning surface

`pyproject.toml` additions:

```toml
[tool.imas-codex.sn.linking]
cluster_peer_cap = 8
semantic_neighbour_cap = 5
semantic_neighbour_min_score = 0.75
exclude_cluster_types = ["cocos"]
cluster_scope_priority_domain = ["domain", "global"]
cluster_scope_priority_global = ["global", "domain"]
```

#### Phase 2d — `_sanitize_links` valid_names graceful fallback (B7)

Behaviour change:

```python
def _sanitize_links(raw, valid_names=None, origin_dd_paths=None):
    out = []
    for link in raw:
        if not link.startswith(_LINK_PREFIXES):
            continue
        prefix, _, body = link.partition(":")
        if prefix == "name" and valid_names is not None and body not in valid_names:
            # graceful fallback — emit as dd: if we can recover a path
            if origin_dd_paths and body in origin_dd_paths:
                link = f"dd:{origin_dd_paths[body]}"
            else:
                log.warning("Dropping unknown name-link: %s", body)
                continue
        out.append(link)
    return out
```

#### Phase 2e — `link_status` wiring audit (B5)

Wire-up is already correct (`_compute_link_status` called at every StandardName
write). Verification only: after Phase 2a–d, confirm `dd:` prefixes now survive to
graph, `link_status == 'unresolved'` for any name with such a link.

#### Phase 2f — Scope `missing_reverse` audit (B6)

`review/audits.py:408-420`:

```python
if target and target in all_ids:
    # skip symmetry check if target is a derived part of name_id or vice versa
    if _is_derived_part_pair(name_id, target):
        continue
    target_linkers = reverse_links.get(name_id, set())
    ...
```

`_is_derived_part_pair(a, b)`: true if `a == parent_of(b)` or `b == parent_of(a)` via
reserved-suffix parse.

#### Phase 2g — Sibling auto-populate (B4, D5)

New `imas_codex/standard_names/enrich_workers.py` function:

```python
def populate_sibling_links(batch_item_ids: set[str]) -> int:
    """After compose commit, for each kind=vector/complex parent in batch,
    query its derived parts from graph and update both sides' links."""
    # 1. Find parents in batch
    # 2. For each parent, MATCH (sn) WHERE sn.id STARTS WITH parent.id + '_'
    #    AND <suffix-token-check>
    # 3. Append name:<child> to parent.links (if not present)
    # 4. Append name:<parent> to each child.links (if not present)
    # 5. Recompute link_status on all touched names
```

Called from `turn.py` after compose-commit phase, before enrich.

#### Phase 2h — Failed re-queue (B9)

`graph_ops.py` new `reactivate_failed_links(new_name_ids: list[str]) -> int`:

```cypher
MATCH (sn:StandardName)
WHERE sn.link_status = 'failed'
  AND any(link IN sn.links WHERE
      link STARTS WITH 'dd:' AND
      exists { MATCH (n:IMASNode {id: substring(link, 3)})-[:HAS_STANDARD_NAME]->(new)
               WHERE new.id IN $new_name_ids })
SET sn.link_status = 'unresolved', sn.link_retry_count = 0
RETURN count(sn) AS reactivated
```

Called after every persist-new-SN commit.

#### Phase 2i — (merged into 2c)

Originally scoped as optional DD semantic-cluster candidate link injection.
**Absorbed into Phase 2c.2** (cluster-peer context block) and promoted to
mandatory — the graph infrastructure makes this the right default, not an
add-on.

---

### Phase 3 — Prompt + response model hardening

#### Phase 3a — Pydantic response models

1. `imas_codex/standard_names/models.py`:
   - Remove `validity_domain`, `constraints`, `cocos_transformation_type` from
     `StandardNameDocumentation` / `StandardNameCompose*` response models.
   - Add `kind: Literal[...]` with `complex` support.

#### Phase 3b — Enrich style guide full rewrite (RD #8a)

Rewrite `_enrich_style_guide.md` in full under Phase 2b; also update:
- Range/unit consistency section (D6).
- Tokamak example grounding caveat (D7).
- Postfix examples throughout (Phase 0c consequence).

#### Phase 3c — Unit override (always copy from DD)

Reaffirm: `unit` is never an LLM-generated field. `imas_codex/standard_names/workers.py`
`_compose_batch` already enforces this. Audit: add `unit_source_assertion` that
logs ERROR if the compose response includes a `unit` key (should not be in
response model).

#### Phase 3d — Review rubric calibration (RD #6)

Current tier cutoffs were calibrated against a 140-pt rubric; post-refactor rubric
is 120-pt (6 dims × 20). Procedure:

1. **Mechanical translation**: `new_cutoff = old_cutoff × 120/140`.
   - outstanding ≥ 112 (was 126) — scales to 96/120
   - good ≥ 98 (was 112) — scales to 84/120
   - adequate ≥ 84 (was 98) — scales to 72/120
2. **Validation**: re-tier the existing 480+ entries under new cutoffs.
3. **Binary-search refinement**: if >10% of entries flip tier vs. pre-refactor
   distribution, adjust cutoffs ±2 points until flip rate ≤10%.

---

### Phase 3.5 — Rubric + validator smoke test

Before regenerating any new names, run:

1. `range_unit_consistency_check` on existing 480 entries — expect ~10-20 hits
   (Faraday rotation, several wave quantities).
2. New `derived_part_parent_presence_check` on existing complex orphans — expect all
   6 to fail (parents don't exist under postfix form).
3. Calibration translation applied; sample 50 entries for manual sanity-check.

---

### Phase 4 — Cost-bound validation loop + regeneration

#### Phase 4a — Clear SN nodes only

User directive: preserve DD enrichment, facility signals, IMASNodes, COCOS — clear
ONLY:

1. `StandardName` nodes (927 entries)
2. `StandardNameSource` nodes (1983 entries)
3. Detach all `HAS_STANDARD_NAME`, `HAS_UNIT` (from StandardName), `HAS_COCOS`
   (from StandardName).

CLI: `sn clear --all --include-sources` (existing command, no new work).

#### Phase 4b — Small verifier run

Target domain: `equilibrium` (cleanest prior results, well-bounded vocabulary).

1. `sn generate --source dd --physics-domain equilibrium --limit 20 -c 2.0 --name-only`
2. `sn enrich --domain equilibrium -c 2.0`
3. `sn review --domain equilibrium -c 2.0`

**Gate criteria (must ALL pass before Phase 4c)**:

- ≥ 80% of new names have at least one link (refutes B3).
- ≥ 20% of links are initially `dd:` (refutes B1+B2+B8).
- `link_status` transitions: `unresolved` → `resolved` on ≥ 50% of `dd:` links after
  `_run_link_phase` sweep (validates B5).
- Any vector parent in the batch has all minted components in its `links` (validates
  B4+D5).
- Any complex parent has all minted parts in its `links` (validates D5).
- `missing_reverse` audit produces 0 findings for parent/component pairs
  (validates B6).
- `range_unit_mixed` validator fires on synthetic-test entry containing
  "milliradians to degrees" (validates D6).

#### Phase 4c — Domain-by-domain rotation

Under $2 cap per domain (RD #5: 10¢/name covers enrich only; total ≈ 2× cap):

```bash
for domain in magnetics equilibrium transport core_profiles \
              auxiliary_heating edge_plasma_physics \
              electromagnetic_wave_diagnostics fast_particles \
              plasma_wall_interactions structural_components \
              turbulence gyrokinetics wave_physics; do
    sn generate --source dd --physics-domain $domain -c 2.0 --name-only
    sn enrich --domain $domain -c 2.0
    sn review --domain $domain -c 2.0
done
```

#### Phase 4d — Bootstrap derived parents

Under postfix, parent/part derivation is unified. Rename
`sn complex-bootstrap` → `sn bootstrap-derived-parents`. Operates on both
`kind=vector` (if any parts exist without a parent) and `kind=complex`.

Cap: `max(2.0, count * 0.10)`. `--max-stems` safety limit = 50.

---

### Phase 5 — Catalog export + ISN PR

1. `sn publish --output-dir standard_names_catalog/` → emits per-domain YAMLs.
2. Verification: zero `dd:` in output; zero unresolved-status entries; all
   `derived_part_parent_presence` satisfied.
3. PR to `imas-standard-names` with refreshed catalog.
4. Trigger ISN's `test serve` to render the updated catalog via its web renderer;
   spot-check rendered pages for "See also" sibling lists.

---

### Phase 6 — Documentation

1. `AGENTS.md`:
   - Postfix grammar note in the Standard Names section.
   - New CLI `sn bootstrap-derived-parents` documented.
   - Link workflow strategy: `name:` / `dd:` / `https://` prefixes + resolution loop.
   - Query pattern for parent/part discovery: suffix-strip approach, not
     `STARTS WITH` alone (RD #1b).
2. `docs/architecture/standard-names.md`:
   - Postfix grammar migration note.
   - Field removal rationale (validity_domain, constraints, cocos_transformation_type).
3. `plans/README.md`: mark plan 36 status.

---

## Open questions for RD round 3

1. **Postfix grammar inversion correctness**: any blind spots in SEGMENT_TEMPLATES
   inversion, parser round-trip, reduction suffix-strip? Particularly: what happens
   to existing names with a natural `_of_` infix that isn't a grammar segment (e.g.
   `_rate_of_change_of_`)?
2. **`_amplitude` ambiguity**: safe to allow both `_amplitude` (complex) and
   `_magnitude` (vector) as closed-vocab suffixes? Or scope `_amplitude` to
   `kind=complex` only and forbid on vectors?
3. **`doc_resolution_status` parallel vs unified**: two status fields (links + docs)
   or one unified `resolution_status`? Trade-off: parallel is simpler to reason
   about per-field; unified reduces field count.
4. **Sibling auto-link race**: what if a parent renames after parts are created? Or
   a new part lands after parent was already linked? Does the sibling populate run
   on every batch commit or only for batches containing the parent?
5. **Small verifier run gate**: correct first domain choice? Should the gate criteria
   include quantitative reviewer score threshold, or only structural checks?
6. **Link ordering / deduplication**: `links` is a list; preserve emission order or
   canonicalize alphabetically? Affects diff stability on re-export.
7. **`complex-vector` / `complex-tensor` kinds**: defer until first DD-sourced
   occurrence lands, or pre-specify the grammar now? Risk: late-add forces a second
   prompt update.
8. **B11 cluster-peer link injection**: include in Phase 2 or defer to Phase 3+? Risk
   of over-linking noisy peers vs signal from strong cluster.

---

## Non-goals

- Migrating prefix-form names via rename table (Path B rejected).
- Reviving `validity_domain` / `constraints` as catalog fields.
- Adding `_amplitude` to the vector reduction set unless RD round 3 approves.
- DD-upstream changes to fix unit tagging (`ion_atomic_number [e]` etc.) — tracked
  separately.

---

## Rollback

If Phase 0 (ISN postfix) fails tests and rc22 cannot ship:

1. Pin ISN back to rc15 in `pyproject.toml`.
2. Keep Phase 1+2 changes; they are grammar-neutral.
3. Regeneration loop proceeds under prefix grammar with linking workflow fixed.
4. Postfix inversion deferred to plan 37.

All other phases are independently rollback-able via git revert.

---

## RD round 2 findings — reconciliation table

| # | Finding | Resolution in v3 |
|---|---|---|
| #1 | Vector/complex false-symmetry | RESOLVED by Phase 0 postfix unification |
| #1b | Query pattern docs (suffix-strip) | Phase 6 docs |
| #1d | `_magnitude` vs `_modulus` redundancy | D2: `_magnitude` reserved for vector, `_modulus` for complex |
| #1e | Non-DD hand-tag policy | Phase 1d |
| #4 | `dd:` export contradiction + no exclusion mechanism | Phase 1b (D3) |
| #5 | Cost-cap 10¢/name covers enrich only | Phase 4c: `max(2.0, count * 0.10)`, documented 2× total |
| #6 | "Recalibrate" procedure undefined | Phase 3d: mechanical translation + binary-search |
| #7 | Range-unit warnings not routed to reviewer | D6 + Phase 3b |
| #8a | `_enrich_style_guide.md` links section rewrite | Phase 2b + 3b |
| #8b | Stale B2 rows | Phase 1e |
| #1c | NC-20a self-healing | ✅ CONFIRMED SOUND |
| #2 | Phase 1c numbering | ✅ RESTRUCTURED |
| #3 | Bootstrap CLI + $1.00 cap | ✅ GENERALISED as Phase 4d |

---

# v4 deltas (RD round 4 input)

> The sections below extend Plan 36 with amplification-focused deltas A–K.
> They modify/extend the v3 body above but do not rewrite it; merge conflicts
> with the v3 phases are flagged inline per delta.

# Plan 36 v4 Deltas — graph-context amplification (staged for RD round 3)

**Status**: drafted; to be merged into plan 36 immediately after RD round 3
completes (currently at 2918s / 71 tool calls / "Analyzing graph
infrastructure context"). This file converts the prompt audit findings plus
user guidance into concrete plan edits.

**User directives this turn**:

1. IMAS-VEC is **parallel** injection, not fallback.
2. Use `search_dd_paths` **backing function directly** — hybrid (60/40
   vector/text + path-segment tiebreaker + accessor de-ranking + IDS-preference
   boost + full enrichment), not naive vector.
3. Prompt-inject far richer context than we currently do.
4. Build **iteration loops** into the plan — this is a significant multi-area
   change that will not land zero-shot; verification + tight retry loops are
   mandatory.

---

## Delta A — Refactor `search_dd_paths` for pipeline reuse

**Problem**: the hybrid search lives inside a stateful MCP tool class
(`HybridDDSearch._gc`, `_embed_query`) — the pipeline cannot currently call it
directly. Plan 36 needs a pure function.

**Change**:

1. Extract core hybrid into `imas_codex/graph/dd_search.py`:

   ```python
   def hybrid_dd_search(
       query: str,
       *,
       ids_filter: str | list[str] | None = None,
       physics_domain: str | None = None,
       node_category: str | None = None,
       cocos_transformation_type: str | None = None,
       dd_version: int | None = None,
       max_results: int = 20,
       gc: GraphClient | None = None,
       encoder: Encoder | None = None,
   ) -> list[DDSearchHit]:
       ...
   ```

2. `DDSearchHit` dataclass exposing **every enrichment field** the MCP tool
   surfaces: id, ids, documentation, data_type, units, physics_domain,
   lifecycle, coordinates, structure_reference, cocos_label,
   cocos_expression, identifier_schema, timebase, score.

3. `HybridDDSearch.search_dd_paths` (MCP tool) refactored to delegate to
   `hybrid_dd_search()` + format.

4. Acceptance: MCP tool output byte-identical before/after refactor; new
   function callable from the compose/enrich workers without MCP plumbing.

**Why a refactor and not a wrapper**: the hybrid logic (tiebreaker, de-ranking,
IDS-preference) is 200 lines — we do not want to duplicate or freeze it. One
source of truth shared by MCP tool and pipeline.

---

## Delta B — Phase 2c.2 upgrade: parallel hybrid injection (supersedes v3's Phase 2c.2)

### 2c.2a — Cluster-peer block (unchanged from v3)

As already documented in v3: `IN_CLUSTER` traversal with `HAS_STANDARD_NAME`
pre-resolution. Scope priority: **domain → global → ids**. Excludes `cocos_*`
cluster types by default. Cap: 8 peers per source.

### 2c.2b — Hybrid-search block (NEW — parallel, not fallback)

For each source DD path in the compose/enrich batch, issue **two parallel
hybrid queries** via `hybrid_dd_search()`:

| Query text | Purpose |
|---|---|
| `path.description` (first 200 chars) | Physics-concept nearest neighbours |
| `path.id` (path-like, text-only mode) | Structural cousins (same segment patterns) |

Both queries receive:
- `physics_domain` filter matching the source's domain when set
- `ids_filter=None` (cross-IDS is the point)
- `node_category="quantity"` (filters out structures/coordinates)
- `max_results=10`

Deduplicate across the two result sets; union capped at 15 per source path.

**Pre-resolution**: after hybrid results return, issue a single graph query to
batch-fetch `HAS_STANDARD_NAME` for every returned `IMASNode.id`. Emit as
`name:<sn>` where minted, `dd:<path>` otherwise — same tagging scheme as
cluster-peer block.

**Difference from cluster-peer block**: hybrid search catches relatedness that
the cluster authorship missed (authorship is imperfect), **and** captures
text-match signal (literal segment overlap, e.g. `*_temperature_*` searches
catch ALL temperature paths regardless of clustering). The two blocks are
complementary — never substitute, always parallel.

### 2c.2c — Pre-resolved SN-VEC block (unchanged from v3 but refined)

For each source path's **description** (not the cluster label), issue
`search_similar_sns_with_full_docs` top-5 with `min_score=0.75`.
Returns full documentation and tags. Offered as authoritative examples.

### 2c.2d — Identifier-schema peers (NEW, free)

When a source path has `HAS_IDENTIFIER_SCHEMA`, fetch the other paths sharing
the same schema with their minted SN where applicable. This catches the
"same enum type across IDSs" case that cluster membership frequently misses.

Cypher (verified pattern from existing queries):

```cypher
UNWIND $paths AS src_id
MATCH (src:IMASNode {id: src_id})-[:HAS_IDENTIFIER_SCHEMA]->(schema)
MATCH (peer:IMASNode)-[:HAS_IDENTIFIER_SCHEMA]->(schema)
WHERE peer.id <> src.id AND peer.node_category = 'quantity'
OPTIONAL MATCH (peer)-[:HAS_STANDARD_NAME]->(sn:StandardName)
RETURN src.id, peer.id, peer.description, sn.id AS minted_name
LIMIT 6
```

### 2c.2e — Version-history block (NEW, free)

Attach `IMASNodeChange` entries (renames, unit changes, COCOS transforms) per
source path. Already implemented in `prompt_tools.fetch_version_history` —
just wire it into `sources/dd.py::_enrich_row` so it flows through
compose_dd, enrich_user, and review prompts.

### 2c.2f — Unified Jinja template block

```jinja
## RELATED REFERENCES

{% for src in sources %}
### For `{{ src.path }}` — {{ src.unit }}

{% if src.cluster_peers %}
**Cluster peers** ({{ src.cluster_context }}):
{% for p in src.cluster_peers %}- `{{ p.tag }}` — {{ p.doc_short }}
{% endfor %}{% endif %}

{% if src.hybrid_neighbours %}
**Hybrid-search neighbours** (physics-concept + structural):
{% for p in src.hybrid_neighbours %}- `{{ p.tag }}` [{{ p.unit }}, {{ p.physics_domain }}] — {{ p.doc_short }}{% if p.cocos_label %} (COCOS {{ p.cocos_label }}){% endif %}
{% endfor %}{% endif %}

{% if src.identifier_peers %}
**Shared identifier schema** (`{{ src.identifier_schema }}`):
{% for p in src.identifier_peers %}- `{{ p.tag }}` — {{ p.doc_short }}
{% endfor %}{% endif %}

{% if src.sn_neighbours %}
**Similar minted standard names**:
{% for sn in src.sn_neighbours %}- `name:{{ sn.id }}` ({{ sn.unit }}, kind={{ sn.kind }}): {{ sn.description_short }}
{% endfor %}{% endif %}

{% if src.version_notes %}
**Version history** ({{ src.path }} has changes):
{% for v in src.version_notes %}- {{ v.change_type }} in DD {{ v.to_version }}: {{ v.detail }}
{% endfor %}{% endif %}
{% endfor %}
```

### 2c.2g — Token budget (revised)

Per-source cost (worst case):
- Cluster peers: 8 × ~130 chars = 1.0 kB
- Hybrid neighbours: 15 × ~150 chars = 2.2 kB (richer — includes unit, domain, COCOS)
- Identifier peers: 6 × ~100 chars = 0.6 kB
- SN neighbours: 5 × ~180 chars = 0.9 kB
- Version notes: avg 2 × ~80 chars = 0.2 kB
- **Total: ~5 kB per source path × (batch N=5-10) = 25-50 kB**

Plus static prefix (~4-5 kB) → total prompt ~30-55 kB. Well within context
windows (128k+) and ~40-60% dynamic / 40-60% static cacheable.

All caps tuneable via `[tool.imas-codex.sn.linking]` in pyproject.toml.

---

## Delta C — Phase 2c.3: extend ALL amplification blocks to enrich_user

**Problem**: enrich prompt tells LLM to populate `links` but gives no
candidates — root cause of B5 (0 / 927 names have links).

**Change**: extend `enrich_workers._build_item_context` to compute
identically-structured `cluster_peers`, `hybrid_neighbours`, `identifier_peers`,
`sn_neighbours`, `version_notes` lists — one per DD source path attached to
the SN being enriched (a single SN may have multiple DD paths; merge
across them).

New Jinja block in `enrich_user.md`:

```jinja
{% if item.link_candidates %}
### Candidate cross-references (for `links` field)

Prefer `name:` when the target is already minted. Use `dd:` for paths not
yet named — the pipeline resolves them after this round.

{% for c in item.link_candidates %}- `{{ c.tag }}` [{{ c.kind_hint }}] — {{ c.doc_short }}
{% endfor %}{% endif %}
```

Where `link_candidates` merges all five blocks from 2c.2 and deduplicates.
Cap at 20 to prevent prompt bloat.

---

## Delta D — Phase 2c.4: Signals compose parity (NEW — absorbs G3)

`compose_signals.md` currently has **no graph context at all**. Upgrade to
match DD compose's amplification:

### For each signal in the batch:

1. **SN-VEC against signal description** — top-5 minted SNs. Instructs
   LLM to **reuse** when applicable rather than minting a duplicate.

2. **Hybrid DD-search against signal description** (via `hybrid_dd_search`) —
   finds nearest DD paths. For each returned path, pre-resolve
   `HAS_STANDARD_NAME`; if present, offer the minted SN as the primary
   reuse candidate. Critical: facility signals frequently map to a DD path
   that has already been named.

3. **Cross-facility same-signal** — other facilities' signals sharing
   description similarity that already resolved to an SN.

4. **Cluster peers via DD bridge** — when the best-matching DD path has
   `IN_CLUSTER` edges, surface those peers too.

New Jinja block in `compose_signals.md`:

```jinja
{% for item in items %}
### Signal: {{ item.signal_id }}
- Description: {{ item.description }}
...

{% if item.sn_reuse_candidates %}
**Candidate standard names to reuse** (by description similarity):
{% for sn in item.sn_reuse_candidates %}- `name:{{ sn.id }}` ({{ sn.unit }}): {{ sn.description_short }}
{% endfor %}{% endif %}

{% if item.dd_path_candidates %}
**Nearest DD paths** (via hybrid search):
{% for p in item.dd_path_candidates %}- `{{ p.tag }}` ({{ p.ids }}, {{ p.unit }}): {{ p.doc_short }}
{% endfor %}{% endif %}

{% if item.sibling_facilities %}
**Other facilities' equivalents**:
{% for s in item.sibling_facilities %}- `{{ s.facility_id }}:{{ s.signal_id }}` → `name:{{ s.standard_name }}`
{% endfor %}{% endif %}
{% endfor %}
```

**Severity**: HIGH — this is the largest missing context pool in the pipeline.

---

## Delta E — Phase 2c.5: Reviewer gets DD context (NEW — absorbs G4, new bug B12)

**Problem**: reviewer scores "Semantic Accuracy" (20 pts) but doesn't see the
DD path documentation the composer used. Can't catch semantic mismatches.

**New bug B12**: Reviewer cannot validate semantic fidelity to source DD docs
without seeing them.

**Change**: `review/pipeline.py` build path — for each candidate in a review
batch, fetch via a single batched Cypher:
- `source_paths[0..n].documentation` and `.description` (the full DD text the
  composer saw)
- cluster_context + cluster_peers for cross-check
- hybrid_neighbours (limited — top 5 sufficient for reviewer)

Add to `review.md`:

```jinja
### Candidate {{ loop.index }}: `{{ item.standard_name }}`
...
**Source DD paths** (primary truth for semantic accuracy):
{% for p in item.source_paths %}- `{{ p.id }}` [{{ p.unit }}]: {{ p.documentation or p.description }}
{% endfor %}

**Nearest minted peers** (semantic-accuracy sanity check):
{% for n in item.nearest_peers %}- `name:{{ n.id }}`: {{ n.description_short }}
{% endfor %}
```

Same block added to `review_docs.md` and `review_name_only.md`.

Update review scoring rubric: D6a **"Description must match DD path documentation intent"** — reviewer penalised for approving name when DD docs say something materially different.

---

## Delta F — Phase 2c.6: Version-history wiring (Delta into existing + cleanup)

Trivial wiring. `prompt_tools.fetch_version_history` exists. Edit
`sources/dd.py::_enrich_row` to bulk-fetch `IMASNodeChange` per batch and
attach as `item.version_notes`. Used by 2c.2e above.

---

## Delta G — Phase 2c.7: Retire the tool-calling variant

Delete `compose_dd_tool_calling.md` and `prompt_tools.py` after B-G land.
Pre-injection + prompt cache dominates tool-calling for this workload.

---

## Delta H — **Iteration loop design (mandatory)**

All prior phases in v3 assumed zero-shot implementation correctness. With the
volume of context amplification in Deltas B-F, this is unrealistic. Plan v4
adds explicit iteration loops at every boundary.

### Phase 4b verifier loop — expanded

v3 already has a 20-name verifier run on equilibrium with 7 gate criteria.
v4 adds:

1. **Per-block verification** — each context block (cluster-peer /
   hybrid / identifier / sn-vec / version-notes) must show measurable
   contribution. Measurement: ablation study during verifier run.

   Procedure: after first full verifier run, re-run the same batch with
   each block **disabled in turn** (5 additional runs, ~$10 total). For each
   block, measure:
   - Links populated (absolute count + % with ≥1 link)
   - Reviewer score delta (should be positive when block is ON)
   - Unique-concept-coverage delta (block ON should surface references
     the batch-only context missed)

   Gate: each block must show **≥ 5 % positive delta on ≥ 1 metric**, else
   deprecate it.

2. **Link-resolution loop** — after compose + enrich + link phase, re-examine
   unresolved `dd:` links. Gate: **≥ 60 % of `dd:` links resolve to a
   minted `name:` within 3 rounds** (current system achieves 0 %).

3. **Regression retry** — if any gate criterion fails, auto-iterate:

   | Gate failure | Retry action |
   |---|---|
   | Low link count | Inspect block outputs, tighten Jinja template (often issue #1) |
   | Low reviewer scores | Re-prompt with failed reasons injected as antipatterns |
   | Block contributes nothing | Raise its cap (was too small) or deprecate |
   | Cluster-peer pre-resolution stale | Force re-resolve and re-embed |

   Cap: **3 retries per batch** at $0.50 each before human escalation.

### Phase 4c domain-rotation loop — expanded

After equilibrium lands gate-green, rotate through `core_profiles`,
`magnetics`, `wall`, `transport` — each as a single verifier-gated batch.
Before each domain:

1. Re-run Phase 2c ablation on **prior domain's cluster** to confirm the
   block mix still contributes.
2. If any block's contribution dropped below 5 %, escalate before proceeding.

### Phase 3.5 pre-flight loop — new

Before ANY compose run against the refactored prompts:

1. Dry-run against 5 already-minted names (equilibrium). Compare the new
   prompt's token budget vs. v3. **Gate: ≤ 2x token growth**. If exceeded,
   reduce caps (typically hybrid neighbours 15 → 10 and identifier peers
   6 → 4).

2. Smoke test: hybrid-search call for 5 known paths returns >= 3 results each.
   If any returns 0, the hybrid function or embedding index is broken — halt.

3. Cluster-peer pre-resolution: for 5 paths known to have `HAS_STANDARD_NAME`
   peers, confirm at least 1 `name:` tag appears in their resolved candidate
   list. If 0 for all 5 → pre-resolution is broken — halt.

### Phase 2 itself (dev) — TDD retries

Each of Deltas A-F gets:
- **Unit tests** for the new data flow (hybrid_dd_search returns ≥ 1 hit,
  pre-resolution correctness, Jinja block renders without errors on
  empty/filled inputs)
- **Integration tests** (one-source-path end-to-end: goes in as path id,
  comes out as rendered prompt block with all 5 context types present)
- **Fixture set**: 10 equilibrium paths covering edge cases (unclustered,
  deprecated, with/without identifier_schema, with/without version history).
  All integration tests run against these.
- **Retry budget**: 3 implementation iterations per delta before escalating
  to RD.

---

## Delta I — Documentation updates

Add to Phase 6 documentation updates:

1. `AGENTS.md` § "SN pipeline" — document the 5-block amplification model.
2. `plans/features/standard-names/` — add
   `prompt-amplification-architecture.md` explaining what each block
   contributes, when it fires, and the token-cost accounting.
3. `docs/architecture/standard-names.md` if it exists — include the
   amplification section.
4. `imas_codex/standard_names/README.md` if it exists — brief signpost.

---

## Revised plan structure summary

```
Phase 0  — ISN postfix rc22 [BLOCKING predecessor]
Phase 1  — Schema cleanup + export exclusion + doc_resolution_status
Phase 2  — Linking rebuild (B1-B11 + B12)
  2a-2i  — bug fixes as in v3
  2c     — GRAPH AMPLIFICATION (expanded):
           2c.1 Source context (existing)
           2c.2 Cluster-peer block (v3)
           2c.3 Enrich extension (Delta C)       ← new
           2c.4 Signals compose parity (Delta D) ← new
           2c.5 Reviewer DD context (Delta E)    ← new
           2c.6 Version-history wiring (Delta F) ← new
           2c.7 Retire tool-calling variant      ← new
Phase 3  — Prompt/model/rubric hardening
Phase 3.5 — Pre-flight loop (Delta H) + smoke tests on existing 480
Phase 4  — CLEAR + verifier loop
  4a    — clear SN nodes
  4b    — VERIFIER LOOP with ablation (Delta H)  ← expanded
  4c    — domain-rotation with per-domain ablation regression check
  4d    — bootstrap-derived-parents
Phase 5  — Export + ISN PR
Phase 6  — Docs (Delta I additions)              ← expanded
```

---

## Outstanding questions to cross-check with RD round 3 response

When RD round 3 returns, verify my Deltas against its findings:

1. Does RD flag hybrid-search per-source token cost? (my estimate: 5 kB/source,
   25-50 kB/batch — under budget but non-trivial)
2. Does RD raise the pre-resolution staleness concern? (already addressed by
   B7 graceful fallback in v3)
3. Does RD propose ablation other than per-block toggle?
4. Does RD flag Phase 4b cost (5 ablation reruns @ ~$2 each = ~$10 extra per
   domain = ~$60 over 6 domains)? Worth it for gated gate criteria.
5. Will the architect/engineer agents implementing this have enough context
   from the plan alone, or should we spin up a supplementary design doc?

Once RD round 3 returns, merge this file into plan 36 as v4 under a single
commit and write_agent to RD for round 4 focused on the amplification design
only.

---

## Delta J — Expand prompt-injection tool surface (tools audit)

### Inventory: all DD / SN search+fetch tools in imas-codex

Traced in `imas_codex/tools/graph_search.py`, `version_tool.py`, `migration_guide.py`:

| Tool | Status in plan v3 | Delta |
|------|--------------------|-------|
| `search_dd_paths` (graph_search.py:367, `HybridDDSearch.search_dd_paths`) | Covered via Delta A refactor | — |
| `check_dd_paths` (799) | Not useful at compose time | — |
| `fetch_dd_paths` (981) | Implicit via source enrichment | — |
| `fetch_error_fields` (1159) | **Not used** | **ADD J1** |
| `list_dd_paths` (1227) | Not useful at compose time | — |
| `get_dd_catalog` (1432) | Too coarse | — |
| `search_dd_clusters` (1567) | Indirect via cluster-peer Cypher | — |
| `get_dd_identifiers` (1920) | **Partially used** | **ADD J2** |
| `find_related_dd_paths` (2163) | **NOT USED — biggest gap** | **ADD J3 (primary)** |
| `get_ids_summary` (2329) | Too coarse | — |
| `get_dd_cocos_fields` (2482) | Covered via source enrichment | — |
| `get_dd_version_context` (version_tool.py:88) | Covered in Delta F | — |
| `get_dd_changelog` (305) | Too noisy | — |
| `get_dd_migration_guide` (migration_guide.py) | Out of scope | — |
| `search_standard_names` (SN MCP) | Covered via SN-VEC | — |
| `fetch_standard_names` | Covered via cluster-peer pre-resolution | — |
| `list_standard_names` | Covered via Delta K tiered-exemplars | — |

### J1 — Inject uncertainty companion fields (`fetch_error_fields`)

**Problem**: a DD path with `_error_upper`/`_error_lower` children represents a *measured* quantity with quantified uncertainty. The LLM currently gets zero signal about this. Correct documentation should mention uncertainty bounds for such quantities.

**Design**:
- In `sources/dd._enrich_row`, call `fetch_error_fields(path)` per source (cached).
- Emit new context field `has_error_fields: bool` and `error_field_paths: list[str]`.
- Jinja block (compose & enrich):
  ```
  {% if source.has_error_fields %}
  **Uncertainty**: source has companion error fields ({{ source.error_field_paths | join(', ') }}).
  → Documentation MUST describe whether this quantity is measured/reconstructed/fitted
    and reference the uncertainty channel rather than burying it in prose.
  {% endif %}
  ```
- Cost: 1 Cypher per distinct source; cachable via source enrichment memoization.

### J2 — Inject identifier-schema allowed values (`get_dd_identifiers`)

**Problem**: when a source path has `identifier_schema` (e.g. `probe_type`, `coordinate_system`, `grid_type`), the LLM knows only the schema name. The *allowed values* are crucial grounding for naming and docs.

**Design**:
- Extend `sources/dd._enrich_row` to fetch allowed values via `get_dd_identifiers(query=schema_name)` when `identifier_schema` is present.
- Emit `identifier_values: list[{name, index, description}]`.
- Jinja block:
  ```
  {% if source.identifier_values %}
  **Identifier schema** ({{ source.identifier_schema }}) — allowed values:
  {% for v in source.identifier_values[:10] %}
    - `{{ v.name }}` (index {{ v.index }}): {{ v.description }}
  {% endfor %}
  → If the name or description implies one specific value, state it explicitly.
  {% endif %}
  ```
- Cost: 1 cached call per unique schema (typically <20 distinct schemas across a batch).

### J3 — Swap naive cluster-peer Cypher for `find_related_dd_paths` (PRIMARY)

**Problem**: the plan v3 Phase 2c.2 re-implements cluster-peer traversal in raw Cypher. This duplicates `find_related_dd_paths` (graph_search.py:2163) which already delivers FIVE relationship types in one call:

1. **cluster_siblings** — IN_CLUSTER peers (with cross-IDS filter `sibling.ids <> p.ids` + noise exclusion for `error`/`metadata` categories)
2. **coordinate_partners** — HAS_COORDINATE siblings (cross-IDS)
3. **unit_companions** — HAS_UNIT siblings (cross-IDS, cross-domain)
4. **identifier_links** — HAS_IDENTIFIER_SCHEMA siblings
5. **cocos_kin** — COCOS-transformation peers (unioned from both property + `cocos_*` cluster sources)

All with noise-category filtering baked in, all deterministic-ordered, already tested in production MCP tool.

**Action**: Replace Delta B block 2c.2a (naive Cypher) with a call to `find_related_dd_paths`, filter relationship_types by config (`enable_cluster, enable_coordinate, enable_unit, enable_identifier, enable_cocos`), then do the `HAS_STANDARD_NAME` pre-resolution pass on each returned `path` in a single batched query:

```cypher
UNWIND $candidate_ids AS cid
OPTIONAL MATCH (n:IMASNode {id: cid})-[:HAS_STANDARD_NAME]->(sn:StandardName)
WHERE sn.validation_status = 'valid'
RETURN cid, sn.id AS minted_name
```

Then the Jinja template emits `name:X` or `dd:X` per the pre-resolution result.

**Refactor target**: like Delta A, extract a `related_dd_paths()` pure function into `imas_codex/graph/dd_search.py` alongside `hybrid_dd_search()`. MCP tool `find_related_dd_paths` delegates to it. Pipeline uses the pure function.

**Token budget impact**:
- 5 relationship types × ~4 peers each = ~20 candidate references per source (dedup across sections drops to ~12-15 unique).
- At ~80 bytes per reference line → ~1.2 kB per source × 5-10 sources = 6-12 kB for the related-peers block.
- Under the 25-50 kB batch budget; matches Delta B's original estimate.

**Config**:
```toml
[tool.imas-codex.sn.linking]
related_enable_cluster = true
related_enable_coordinate = true
related_enable_unit = true
related_enable_identifier = true
related_enable_cocos = false   # noisy for physics cross-referencing (Q4 RD guidance)
related_max_per_type = 4
related_unique_peers_per_source = 12
```

### J4 — Bonus: extend Delta D (`compose_signals.md`) with `search_signals` peers

**Problem**: for facility-signals compose, the SN graph has cross-facility siblings (same concept different facility) that are invaluable exemplars.

**Design**: inject top-3 `FacilitySignal` peers matched by description similarity + same `physics_domain`, with their `HAS_STANDARD_NAME` pre-resolution. Uses existing `search_signals` backend via `imas_codex/tools/graph_search.py` facility-signal path.

Added to Phase 2c.4 (Delta D) spec.

---

## Delta K — Target-anchored examples, split by consumer, per-dimension reviewer context (NEW PHASE 2c.3)

### K-1 — Delete `benchmark_calibration.yaml` and its loaders (dead path)

Static calibration fixtures predate the review pipeline producing graph-stored reviewer output. Dynamic examples (K1-onwards) are strictly higher-signal: they reflect the current prompt regime, match the current graph's physics-domain distribution, and carry live per-dimension scores. The static fixtures cannot track prompt evolution and are now obsolete.

**Surface to delete (atomic)**:

| Path | Scope |
|---|---|
| `imas_codex/standard_names/benchmark_calibration.yaml` | Entire file |
| `imas_codex/standard_names/calibration.py` | Entire file (22-line cached loader) |
| `imas_codex/standard_names/benchmark.py:253-...` | `load_calibration_entries()` function |
| `imas_codex/standard_names/benchmark.py:478` | Call site in benchmark runner |
| `imas_codex/standard_names/review/pipeline.py:1251-...` | `_load_calibration_entries()` function |
| `imas_codex/standard_names/review/pipeline.py:444` | Call site in review pipeline |
| `tests/standard_names/test_benchmark.py` | Remove ~12 calibration-related test blocks |
| `AGENTS.md` | Remove any reference to `benchmark_calibration.yaml` and `imas_codex/standard_names/calibration.py` |

Verify post-delete: `rg "calibration_entries\|benchmark_calibration\|standard_names/calibration" imas_codex tests` returns zero hits.

### K0 — Rubric alignment (prerequisite sub-delta)

Three coupled rubric changes shipping as one atomic commit so schema, rubric, models, prompts, and fixtures stay consistent.

**K0.a — Boundary move: `good` tier minimum 0.60 → 0.65**

| Threshold | Current | New |
|---|---|---|
| `good` tier minimum | 0.60 | **0.65** |
| Reviewer `accept` gate | 0.60 | **0.65** |
| `sn publish --min-score` default | 0.65 | 0.65 (unchanged) |

One threshold: `0.65`. Tier `good` ⇔ publishable; below `good` ⇔ fails publish.

**K0.b — Rename `adequate` → `inadequate`**

A name below the publish threshold is not adequate. Rename the enum value throughout the codebase.

| Tier | Band after K0.a | Semantics |
|---|---|---|
| `outstanding` | ≥ 0.85 | Top-quality, publishable |
| `good` | [0.65, 0.85) | Publishable baseline |
| `inadequate` | [0.40, 0.65) | Below publish threshold (was "adequate") |
| `poor` | [0.00, 0.40) | Clearly failing |

Rename scope:

| Location | Change |
|---|---|
| `imas_codex/schemas/standard_name.yaml:412-415, 947` | Enum value + descriptions |
| `imas_codex/llm/config/sn_review_criteria.yaml:34` | Key rename |
| `imas_codex/standard_names/models.py:215, 278, 362` | Tier-assignment return literal (3 sites) |
| `imas_codex/standard_names/benchmark.py:823` | Histogram key |
| `imas_codex/standard_names/graph_ops.py:443` | Docstring |
| `imas_codex/llm/prompts/sn/review.md:180` | Rubric prose |
| `imas_codex/llm/prompts/sn/review_docs.md:54` | Rubric prose |
| `imas_codex/llm/prompts/sn/review_name_only.md:112` | Rubric prose |
| `imas_codex/llm/prompts/shared/sn/_scoring_rubric.md:74` | Rubric prose |
| Regression test | `grep '"adequate"'` in `imas_codex/standard_names/` must be empty |

Because Phase 4 starts with `sn clear --all`, no data migration needed.

**K0.c — Schema extension: per-dimension reviewer comments**

Current state (verified):

- `reviewer_scores` stores a JSON blob of 6 per-dimension integers — already usable.
- `reviewer_comments` stores a **single aggregate** `reasoning` string from the Pydantic model — cannot be decomposed by dimension at render time.
- `issues` (Pydantic field) and `verdict` are **not persisted** to the graph.

To render per-dimension anchors to the review LLM ("a 15/20 on grammar looks like this because X; a 6/20 on documentation looks like this because Y"), the reviewer must *produce* per-dimension reasoning, and the graph must *store* it.

**Schema additions to `standard_name.yaml`**:

```yaml
# New slot on StandardName
reviewer_dimension_comments:
  description: >-
    JSON-encoded per-dimension reviewer reasoning, one string per dimension.
    Keys mirror reviewer_scores: grammar, semantic, documentation, convention,
    completeness, compliance (or the 4-dimension name-only subset).
    Distinct from reviewer_comments, which is a single aggregate string.

# New slot on StandardName
reviewer_issues:
  description: >-
    JSON-encoded list of specific issues flagged by the reviewer during the
    most recent review. Persisted for audit, example rendering, and reset
    decision logic.

# New slot on StandardName
reviewer_verdict:
  description: >-
    Reviewer's verdict for the most recent review (accept, revise, reject).
    Persisted alongside score so downstream tooling can filter on verdict
    without recomputing from thresholds.
```

**Pydantic model additions to `standard_names/models.py`**:

Introduce a parallel dimensional-comments container keyed by the same six (or four) dimensions as `StandardNameQualityScore`:

```python
class StandardNameQualityComments(BaseModel):
    """Per-dimension reasoning for each rubric dimension (matches StandardNameQualityScore)."""
    grammar: str = Field(description="Why grammar scored as it did (1-3 sentences)")
    semantic: str = Field(description="Why semantic scored as it did (1-3 sentences)")
    documentation: str = Field(description="Why documentation scored as it did (1-3 sentences)")
    convention: str = Field(description="Why convention scored as it did (1-3 sentences)")
    completeness: str = Field(description="Why completeness scored as it did (1-3 sentences)")
    compliance: str = Field(description="Why compliance scored as it did (1-3 sentences)")


class StandardNameQualityReview(BaseModel):
    # ... existing fields ...
    scores: StandardNameQualityScore
    comments: StandardNameQualityComments    # NEW — required, one sentence per dimension
    verdict: StandardNameReviewVerdict
    reasoning: str                           # kept as aggregate summary
    issues: list[str] = Field(default_factory=list)
    # ... existing fields unchanged ...
```

Parallel name-only variant (`StandardNameQualityCommentsNameOnly`) and docs variant (`StandardNameQualityCommentsDocs`) subset the fields to match their 4-dimension rubrics.

**Persistence update in `review/pipeline.py:965-969`**:

```python
original["reviewer_score"] = review.scores.score
original["reviewer_scores"] = json.dumps(review.scores.model_dump())
original["reviewer_dimension_comments"] = json.dumps(review.comments.model_dump())  # NEW
original["reviewer_comments"] = review.reasoning
original["reviewer_issues"] = json.dumps(review.issues)                               # NEW
original["reviewer_verdict"] = review.verdict.value                                    # NEW
original["review_tier"] = review.scores.tier
```

**Prompt update to `review.md`, `review_name_only.md`, `review_docs.md`**:

Add per-dimension comment requirement to the response schema example, with a strict length guidance: each dimension's comment must be 1-3 sentences explaining *why this specific score*.

**Parser for retrieval (`graph_ops.py`)**:

`load_examples_for_review()` (K3 below) parses `reviewer_dimension_comments` JSON and delivers it to the Jinja template as a nested dict.

### K1 — Four targets, one per tier, anchored at tier midpoints

| Slot | Tier band after K0 | Target score |
|---|---|---|
| S1 | outstanding [0.85, 1.00] | **1.00** |
| S2 | good [0.65, 0.85] | **0.75** |
| S3 | inadequate [0.40, 0.65] | **0.52** |
| S4 | poor [0.00, 0.40] | **0.20** |

Tolerance ±0.05 for S2/S3/S4. S1 is "highest reviewed score with comments" — no lower bound.

### K2 — Split library by consumer

Different LLM tasks benefit from different example subsets. One `load_examples()` module, two rendering paths.

| Consumer | Prompt | Slots consumed | Rationale |
|---|---|---|---|
| Generator | `compose_system.md`, `compose_dd.md`, `compose_signals.md` | **outstanding + good** (2) | Positive demonstration. Failing examples risk priming and eat tokens; validation + review catch failures downstream. |
| Enricher | `enrich_system.md` | **outstanding + good** (2) | Emulation task. Same rationale. |
| Reviewer | `review.md`, `review_name_only.md`, `review_docs.md` | **all four** (outstanding + good + inadequate + poor) with full per-dimension breakdown | Grader task. Full-range anchors stabilise the 0-20 integer scale across models and across batches; per-dimension breakdown mirrors the exact output shape the reviewer must produce. |

**API**:

```python
# standard_names/examples.py
TARGETS = [
    ("outstanding", 1.00, 0.15),  # tolerance effectively unused; S1 takes max
    ("good",        0.75, 0.05),
    ("inadequate",  0.52, 0.05),
    ("poor",        0.20, 0.05),
]

def load_examples_for_compose(sources: list[SourceRecord]) -> dict[str, Example]:
    """Returns only {outstanding, good} slots."""
    return _load_slots(sources, slot_subset={"outstanding", "good"})

def load_examples_for_review(sources: list[SourceRecord]) -> dict[str, ExampleWithDimensions]:
    """Returns all four slots, each carrying reviewer_scores + reviewer_dimension_comments."""
    return _load_slots(sources, slot_subset={"outstanding", "good", "inadequate", "poor"},
                       with_dimensions=True)
```

### K3 — Batch-context-aware physics-domain scoping

Same as before — batch's surfaced physics_domains first; fall back to all-domains per-slot if empty. No CLI flag.

Compose/enrich Cypher returns the same fields as in K3 of the prior draft. **Review Cypher additionally returns**:

```cypher
// Add to review-side query
RETURN ..., sn.reviewer_scores AS reviewer_scores_json,
       sn.reviewer_dimension_comments AS dimension_comments_json,
       sn.reviewer_issues AS issues_json,
       sn.reviewer_verdict AS verdict
```

`load_examples_for_review` parses all three JSON blobs into typed dicts before handing to Jinja.

### K4 — Jinja rendering — two fragments, one per consumer type

#### K4.a — Compose / enrich template: `shared/sn/_compose_scored_examples.md`

Outstanding + good only. No failing block. Renders empty on cold graph.

```jinja
{% if examples and (examples.outstanding or examples.good) %}
## REVIEW-SCORED EXAMPLES

Reviewed standard names selected from the graph as reference patterns to
emulate. Entries are deterministically chosen by closeness to target score
and remain stable across runs once the graph settles.

{% if examples.outstanding %}
### Outstanding (score {{ "%.2f"|format(examples.outstanding.score) }}, {{ examples.outstanding.domain }})
- **`{{ examples.outstanding.id }}`** [{{ examples.outstanding.unit or 'dimensionless' }}, kind={{ examples.outstanding.kind }}]
  - Description: {{ examples.outstanding.description }}
  - Documentation: {{ examples.outstanding.documentation | truncate(400, True, "…") }}
  - Reviewer note: *{{ examples.outstanding.comments | truncate(200, True, "…") }}*
{% endif %}

{% if examples.good %}
### Good (score {{ "%.2f"|format(examples.good.score) }}, {{ examples.good.domain }})
- **`{{ examples.good.id }}`** [{{ examples.good.unit or 'dimensionless' }}, kind={{ examples.good.kind }}]
  - {{ examples.good.description }}
  - Reviewer note: *{{ examples.good.comments | truncate(200, True, "…") }}*
{% endif %}
{% endif %}
```

Included from: `compose_system.md`, `compose_dd.md`, `compose_signals.md`, `enrich_system.md`.

#### K4.b — Review template: `shared/sn/_review_scored_examples.md`

All four slots, each showing per-dimension score + per-dimension comment. Mirrors the reviewer's target output shape.

```jinja
{% if examples and (examples.outstanding or examples.good or
                    examples.inadequate or examples.poor) %}
## REVIEWER CALIBRATION EXAMPLES

Previously reviewed standard names spanning the full score range. Each
example shows the per-dimension 0-20 score you must produce and the
reasoning tied to each dimension. Use these to anchor your own scores to
a consistent absolute scale across batches.

{% for slot in ['outstanding', 'good', 'inadequate', 'poor'] %}
{% set ex = examples[slot] %}
{% if ex %}
### {{ slot | capitalize }} — aggregate score {{ "%.2f"|format(ex.score) }} ({{ ex.domain }})

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
Description: {{ ex.description }}

Per-dimension scores and reasoning:
{% for dim in ['grammar', 'semantic', 'documentation', 'convention', 'completeness', 'compliance'] %}
{% if dim in ex.scores %}
- **{{ dim }}: {{ ex.scores[dim] }}/20** — {{ ex.dimension_comments[dim] | default('(no per-dimension comment recorded)') }}
{% endif %}
{% endfor %}

{% if ex.issues %}Reviewer-flagged issues: {{ ex.issues | join('; ') }}{% endif %}
{% if ex.verdict %}Verdict: **{{ ex.verdict }}**{% endif %}
{% endif %}
{% endfor %}
{% endif %}
```

Included from: `review.md`, `review_name_only.md`, `review_docs.md`. The `for dim in ...` loop filters by dimensions present in the example's `scores` dict so the same template handles 6-dim full review and 4-dim name-only review.

### K5 — Automatic, no toggle

Both fragments render unconditionally. Cold graph → empty block → zero tokens. Warm graph → block populates as reviews land. Phase 4b ablation keeps internal test-only kwargs (`_force_disable_examples_compose`, `_force_disable_examples_review`) for measurement.

### K6 — Configuration

```toml
[tool.imas-codex.sn.examples]
outstanding = { target = 1.00, tolerance = 0.15 }
good        = { target = 0.75, tolerance = 0.05 }
inadequate  = { target = 0.52, tolerance = 0.05 }
poor        = { target = 0.20, tolerance = 0.05 }

# Per-consumer slot filtering (not user-facing; documented for transparency).
compose_slots = ["outstanding", "good"]
review_slots  = ["outstanding", "good", "inadequate", "poor"]

min_comment_chars = 40          # aggregate reviewer comment
min_dim_comment_chars = 20      # each per-dimension comment
```

### K7 — Stability contract

Both fragments cache-stable on identical inputs (graph snapshot + rubric YAML + batch physics_domain set). Expected prefix-cache lift: +5-10 pp on compose prompts, +10-15 pp on review prompts (review benefits more because its prompt has historically been shorter and more variable).

### K8 — Lifecycle

| Phase | Graph state | Compose block | Review block |
|---|---|---|---|
| Cold | 0 reviewed | Absent | Absent |
| Thawing | First review pass completes | Partial (outstanding often fills first) | Partial (outstanding + good likely; inadequate + poor rare) |
| Warm | Several review passes; inadequate + poor filled via revise-path + reject-path reviews | Full (2 slots) | Full (4 slots) |
| Hot | Stable | Full, cache-stable | Full, cache-stable |

### K9 — What explicitly NOT in Delta K

- No `benchmark_calibration.yaml` / `calibration.py` (deleted in K-1)
- No static bootstrap fixtures
- No provisional / unreviewed fallback
- No CLI flag (`--exemplar-domain` removed)
- No new score bands or vocabulary beyond the `adequate`→`inadequate` rename

### K10 — Dependencies

- **K-1 (delete calibration)** runs first — no prerequisites, pure removal.
- **K0.a, K0.b, K0.c (rubric + schema)** ship atomically in Phase 0 / Phase 2 pre-flight. K0.c requires `uv run build-models --force` and updates to reviewer Pydantic response model + 3 review prompts.
- **K1-K8** depend on the review pipeline having produced at least some reviewed+commented entries. First meaningful fill happens after the first Phase 3 review pass.

### K11 — Risks

- **Schema extension (K0.c)**: extends the reviewer's response contract. Existing review prompts omit per-dimension reasoning, so the Pydantic parser will reject old response shapes. Mitigation: K0.c ships with the prompt updates in the same commit; `sn clear --all` ensures no stale graph state assumes the old shape.
- **Increased reviewer token output**: +6 short reasoning strings per name ≈ +150-250 output tokens per review. Cost increase is measurable but small relative to reviewer model cost; offset by improved score calibration reducing rework. Track in Phase 4b ablation.
- **Per-dimension reasoning may be rote**: reviewer LLMs may produce formulaic per-dimension strings. Mitigation: enforce `min_dim_comment_chars = 20` in example selection so only substantive anchors make it into the prompt; formulaic cases simply aren't chosen as examples.
- **Sparse failing-slot review fills**: `inadequate` and `poor` slots rarely populate because publish-intent names rarely get reviewed to low scores. No-op render handles sparsity. Deliberate fills come from the revise-loop and reject pipeline.

### K12 — Todos (SQL)

- `k-1-calibration-remove` — delete `calibration.py`, `benchmark_calibration.yaml`, 2 loader call sites, ~12 test blocks, `AGENTS.md` references; verify `rg "calibration_entries|benchmark_calibration"` returns zero
- `k0a-rubric-boundary` — `sn_review_criteria.yaml` good.min → 0.65, accept → 0.65
- `k0b-tier-rename` — atomic `adequate`→`inadequate` across schema, rubric, models.py, benchmark.py, graph_ops.py, 4 prompt files; regression test asserts no `"adequate"` literal
- `k0c-schema-dimension-comments` — add `reviewer_dimension_comments`, `reviewer_issues`, `reviewer_verdict` slots to `standard_name.yaml`; add `StandardNameQualityComments` Pydantic model (+ name-only + docs variants); update `review/pipeline.py` persistence; update `review.md`, `review_name_only.md`, `review_docs.md` response-schema sections; `uv run build-models --force`
- `k1-examples-module` — implement `load_examples_for_compose` + `load_examples_for_review` in `standard_names/examples.py` with 4 targets + batch-domain → all-domain fallback
- `k2-compose-fragment` — create `shared/sn/_compose_scored_examples.md`; include from compose + enrich system prompts
- `k3-review-fragment` — create `shared/sn/_review_scored_examples.md`; include from 3 review prompts; loop dimensions adaptive to 6-dim or 4-dim rubric
- `k4-config` — add `[tool.imas-codex.sn.examples]` + settings accessor
- `k5-safeguard` — enforce `size(reviewer_comments) >= 40` and `size(dim_comment) >= 20` in selection Cypher; unit-test empty/short cases
- `k6-ablation-hook` — test-only kwargs `_force_disable_examples_compose|review` on loaders; wire into Phase 4b matrix
- `k7-plan-integration` — merge Delta K as Phase 2c.3 in `plans/features/standard-names/36-catalog-quality-refactor.md`
- `k8-cache-metrics` — extend benchmark cache-% logging to track prefix-cache delta by consumer type (compose vs review)
- `k9-docs` — update `AGENTS.md` SN section: unified 0.65 threshold, `inadequate` tier, dimension-comments schema, split example consumers; delete calibration references

---

# v4 deltas — RD round 4 reconciliation

RD round 4 (rubber-duck opus-4.6) returned **6 findings, no BLOCKING**. All fold-ins are concrete scope expansions that would otherwise cause test failures or silent data loss during implementation. No user decisions required — reconciliation is mechanical.

## H-1 (HIGH) — K0.b rename scope expansion

**Symptom**: Plan verification grep `grep '"adequate"' in imas_codex/standard_names/` is too narrow. Implementing K0.b as specified leaves ~20 broken test assertions and one CLI help string.

**Expanded K0.b rename scope** (additions in **bold**):

- `imas_codex/schemas/standard_name.yaml:412-415, 947` — enum + description
- `imas_codex/llm/config/sn_review_criteria.yaml:34` — rubric key
- `imas_codex/standard_names/models.py:215, 278, 362` — 3 literal returns
- `imas_codex/standard_names/models.py:210-216` — hardcoded tier threshold property (confirm all 3 tier property methods: `StandardNameQualityScore.tier`, `StandardNameQualityScoreNameOnly.tier`, `StandardNameQualityScoreDocs.tier`)
- `imas_codex/standard_names/benchmark.py:823` — tier bucket key
- `imas_codex/llm/prompts/sn/review.md:180`, `review_docs.md:54`, `review_name_only.md:112`, `shared/sn/_scoring_rubric.md:74`
- **`imas_codex/cli/sn.py:261`** — `--tier` flag help text `'poor,adequate'` → `'poor,inadequate'`
- **`tests/cli/test_sn_generate_cli.py:136, 144`** — CLI assertion
- **`tests/standard_names/test_review.py:198, 199, 211`** — rename `test_quality_score_tier_adequate` → `test_quality_score_tier_inadequate`, update 2 literal assertions
- **`tests/standard_names/test_scoring.py:96, 104, 105`** — variable `adequate` → `inadequate`, 2 literal assertions
- **`tests/standard_names/test_review_pipeline.py:908, 941`** — 2 fixture literals
- **`tests/standard_names/test_fetch_review_feedback_for_sources.py:91, 114, 138`** — 3 fixture literals
- **`tests/standard_names/test_compose_feedback_injection.py:92`** — 1 fixture literal
- **`tests/standard_names/test_review_name_only.py:50, 53`** — 4-dim tier test
- **`tests/standard_names/test_benchmark.py:793, 862`** — tier distribution + tier-range assertions
- **`tests/standard_names/test_review_rubrics.py:55, 61`** — docs tier test
- **`tests/standard_names/test_graph_ops.py:747, 751`** — tier filter literal
- **`docs/architecture/standard-names.md`** — grep for `adequate`, update all tier references

**Verification grep** (replaces narrow grep in K0.b):
```bash
rg '"?\badequate\b"?' imas_codex tests docs scripts --glob '!*.yaml'
# Must return zero matches after K0.b lands.
```

## H-2 (HIGH) — K-1 calibration deletion is a signature refactor

**Symptom**: Plan describes K-1 as "atomic deletion" but `calibration_entries` is a threaded function parameter and Jinja template variable. Simple file deletion leaves broken signatures and dangling Jinja block.

**Expanded K-1 scope** (additions in **bold**):

Files to DELETE:
- `imas_codex/standard_names/benchmark_calibration.yaml`
- `imas_codex/standard_names/calibration.py`
- **`tests/standard_names/test_calibration.py`** (entire file — imports from deleted module)

Call sites to EDIT:
- `imas_codex/standard_names/benchmark.py` — remove import, remove `load_calibration_entries()` definition (~line 253), remove call at line 478
- **`imas_codex/standard_names/benchmark.py:273`** — remove `calibration_entries` from `score_with_reviewer()` signature (required positional parameter)
- **`imas_codex/standard_names/benchmark.py:334, 370`** — remove `"calibration_entries": cal` from template context dict
- `imas_codex/standard_names/review/pipeline.py` — remove `_load_calibration_entries()` definition (~line 1251), remove call at line 444
- **`imas_codex/standard_names/review/pipeline.py:1029`** — remove `calibration_entries` kwarg from `_review_single_batch()` signature
- **`imas_codex/standard_names/review/pipeline.py:490, 571, 1166`** — remove `calibration_entries=calibration_entries` from 3 call sites
- **`imas_codex/standard_names/review/pipeline.py:1083, 1100, 1112`** — remove template-context injection at 3 sites
- **`imas_codex/llm/prompts/sn/review.md:192-199`** — remove `{% for entry in calibration_entries %}` Jinja block (see M-3 for pairing with K4)
- **`scripts/model_comparison_study.py:277, 361`** — remove 2 references (the script is not under `standard_names/` so not caught by narrow grep)
- **`docs/architecture/standard-names.md:529`** — remove reference

AGENTS.md references to remove:
- `imas_codex/standard_names/calibration.py` row in Key Modules table
- `benchmark_calibration.yaml` mentions in Benchmark section

**Verification grep**:
```bash
rg "calibration|_load_calibration_entries|load_calibration_entries" imas_codex tests docs scripts AGENTS.md
# Must return zero matches after K-1 lands.
```

## M-1 (MEDIUM) — Docs rubric has independent dimension names, not subset

**Symptom**: Docs rubric uses `description_quality / documentation_quality / completeness / physics_accuracy` — NOT a subset of the full rubric's `grammar / semantic / documentation / convention / completeness / compliance`. K4.b hardcoded dimension loop produces zero output for docs-reviewed examples.

**Fix in K0.c**:
- Define `StandardNameQualityCommentsDocs` with **independent fields** matching `StandardNameQualityScoreDocs`: `description_quality`, `documentation_quality`, `completeness`, `physics_accuracy` (1-3 sentences each, all required).
- Full/name-only retain subset relationship as originally specified.

**Fix in K4.b** — replace hardcoded dimension list at plan line 1713:
```jinja
{# BEFORE #}
{% for dim in ['grammar', 'semantic', 'documentation', 'convention', 'completeness', 'compliance'] %}
{% if dim in ex.scores %}
{{ dim }}: {{ ex.scores[dim] }}/20 — {{ ex.dimension_comments[dim] }}
{% endif %}
{% endfor %}

{# AFTER — iterate over actual dimension keys present on the example #}
{% for dim, score in ex.scores.items() if dim in ex.dimension_comments %}
{{ dim }}: {{ score }}/20 — {{ ex.dimension_comments[dim] }}
{% endfor %}
```

Also: the load path in K3 must return `dimension_comments` as a dict keyed by the dimension names actually stored — do not assume 6-dim shape. The `{% for dim, score in ex.scores.items() %}` pattern handles all three rubric variants uniformly.

## M-2 (MEDIUM) — graph_ops.py persistence path incomplete in K0.c

**Symptom**: K0.c sets 3 new properties on `original` dict in pipeline.py but `write_standard_names()` has a fixed Cypher SET block that explicitly names every persisted property. Without extending it, the 3 new properties are silently dropped.

**Additional files in K0.c scope**:

1. **`imas_codex/standard_names/graph_ops.py:637-648`** — extend MERGE SET block for `write_standard_names()`:
   ```cypher
   sn.reviewer_dimension_comments = coalesce(b.reviewer_dimension_comments, sn.reviewer_dimension_comments),
   sn.reviewer_issues = coalesce(b.reviewer_issues, sn.reviewer_issues),
   sn.reviewer_verdict = coalesce(b.reviewer_verdict, sn.reviewer_verdict),
   ```

2. **`imas_codex/standard_names/graph_ops.py:~693`** — extend batch-construction dict:
   ```python
   "reviewer_dimension_comments": _ensure_json(n.get("reviewer_dimension_comments")),
   "reviewer_issues": _ensure_json(n.get("reviewer_issues")),
   "reviewer_verdict": n.get("reviewer_verdict"),
   ```

3. **`imas_codex/standard_names/graph_ops.py:821-895`** — `write_reviews()` persists Review nodes separately. If K3 reads per-dimension comments via Review nodes (plan line ~1648), add the same 3 properties to the Review MERGE block + batch dict. If K3 reads directly from StandardName node, skip this step but document the decision in K3.

**Verification**:
```bash
# After K0.c lands, run:
uv run pytest tests/standard_names/test_graph_ops.py -k reviewer -xvs
# And post-persistence sanity check:
uv run imas-codex graph shell <<< "MATCH (sn:StandardName) WHERE sn.reviewer_verdict IS NOT NULL RETURN count(sn) LIMIT 1"
```

## M-3 (MEDIUM) — K-1/K4 coupling in `review.md`

**Symptom**: K-1 removes calibration Jinja block from `review.md`; K4 adds dynamic-examples include. If K-1 ships before K4, the review prompt has no calibration anchor during the window.

**Fix**: K-1 and K4.b's `review.md` edit must ship in the **same commit** (or K-1 stubs the include path while K4 fills it in). Update K10 dependencies section:

- K-1 **may ship in the same commit as** K4.b (the `review.md` Jinja edit is pure swap: remove block, add include).
- Concretely: K-1's verification grep above is satisfied immediately after both land.
- Compose/enrich paths in K-1 are independent — they can ship before K4.a lands (the include line will be a no-op against empty graph, matching K4.a's zero-lookup behavior).

Alternative: K-1 replaces the Jinja block with a comment placeholder `{# K4 dynamic examples injected here #}` which K4.b then fills with `{% include ... %}`. Both sequences are valid; pick per-implementer convenience.

## L-1 (LOW) — Delta J3 `find_related_dd_paths` is larger than Delta A

**Symptom**: Plan describes J3 as "like Delta A" but extraction involves async→sync change, 5-query fan-out, and a `noise_clause.replace("sibling", "prop_sib")` string-manipulation pattern.

**Fix**: Note in J3 that the extraction is larger than Delta A:
- Remove `async` and replace with sync signature taking `gc: GraphClient`; all J3 callers (compose, enrich workers) already run sync or inside `asyncio.to_thread`
- Clean up the `noise_clause` string-replace pattern during extraction — use parameterized Cypher with both parameter names directly, not runtime string substitution
- 5-query fan-out is fine as-is; do not collapse

No correctness risk; implementer will encounter during development.

---

## Round-4 verdict

All 6 findings are mechanical scope expansions. No design changes. No user decisions. Plan 36 v4+round4 is **ready for fleet dispatch**.

**Fleet ordering** (unchanged from K10, minor amendment from M-3):
1. K-1 + K4.b `review.md` edit — same commit
2. K0.a, K0.b, K0.c — atomic commit each (three separate commits)
3. Delta A (hybrid_dd_search extraction) — blocking prerequisite for B-F
4. Deltas B-F — parallelizable after A
5. Deltas G, H, I, J — parallelizable with B-F
6. K4.a, K5-K8 — after K0.c persists per-dim data
7. K11 iteration loop — last

