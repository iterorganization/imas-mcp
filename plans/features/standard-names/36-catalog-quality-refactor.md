# Plan 36 — Catalog Quality Refactor (v3)

> **Status**: PLANNING (RD round 3 pending on this refactor)
> **Supersedes**: v1 (complex parent/part addition) and v2 (complex-SUFFIX asymmetric design)
> **Scope pivot (v3)**: Greenfield postfix grammar inversion (unifies vector + complex
> under ONE rule); complete linking-workflow rebuild (11 bugs); field design cleanup.
> **RD round 2 findings** (on v2) are either resolved by postfix (#1) or merged into this
> plan (#4, #6, #7, #8a/b, #1b/d/e, #5).

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

### `kind` enum — ADD `complex`, `complex-scalar`, `vector`, `tensor`

User directive. Closed enum: `scalar`, `vector`, `tensor`, `complex-scalar`,
`metadata`. `complex` is a shape descriptor independent of rank (scalar vs vector
complex-valued quantities exist — e.g. Fourier coefficient is complex-scalar;
polarization tensor is complex-tensor).

**RD round 3 question**: do we need distinct `complex-vector` / `complex-tensor`
kinds, or is `kind=complex-scalar` + grammar structure (`*_real_part` etc) sufficient?
Current recommendation: start with `complex-scalar` only; `complex-vector` /
`complex-tensor` added on demand when first DD-sourced occurrence lands.

### Summary of field decisions (v3)

| Field | Action | Rationale |
|---|---|---|
| `validity_domain` | REMOVE | Empty for most; belongs in docs prose |
| `constraints` | REMOVE | Empty for most; belongs in docs prose |
| `cocos_transformation_type` | REMOVE from catalog | Graph-only concern |
| `kind` | EXTEND to {scalar, vector, tensor, complex-scalar, metadata} | User-requested |
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
4. **Sibling auto-populate** (D5): after batch commit, for each `kind ∈ {vector, complex-scalar}`
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
     `.amplitude`, `.phase`, `.modulus` postfix-form properties. `kind = 'complex-scalar'`.
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
     `kind ∈ {vector, complex-scalar}` StandardName.
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
    """After compose commit, for each kind=vector/complex-scalar parent in batch,
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
   - Add `kind: Literal[...]` with `complex-scalar` support.

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
- Any complex-scalar parent has all minted parts in its `links` (validates D5).
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
`kind=vector` (if any parts exist without a parent) and `kind=complex-scalar`.

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
   `kind=complex-scalar` only and forbid on vectors?
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
