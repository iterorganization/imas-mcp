# Standard Name Quality Gap Analysis

> Reference document tracking measurable gaps between automated pipeline output
> and the ISN catalog standard. Each gap has a **metric**, **evidence**, and
> **resolution status** so progress can be checked off as fixes land.

## Methodology

**Catalog baseline** — 309 entries across 20 domains in
`imas-standard-names-catalog/standard_names/`, generated via ISN's interactive
MCP-assisted workflow (`.github/prompts/generate_standard_names_from_imas_paths.prompt.md`).

**Pipeline output** — 17 StandardName nodes produced by `sn generate --ids equilibrium --limit 20`
on 2025-04-13, with review enabled. Queried from Neo4j graph.

---

## Summary Scorecard

| Metric | Catalog (309) | Pipeline (before) | Pipeline (after) | Target | Status |
|--------|--------------|-------------------|------------------|--------|--------|
| Avg documentation length | 1569 chars (median 2008) | 473 chars | **941 chars** (avg of latest 7) | ≥1200 chars | 🟡 **2× improved, closing** |
| With governing equation | 72% (224/309) | ~40% | **100%** (latest batch) | ≥80% | ✅ **RESOLVED** |
| With sign convention | 65% (201/309) | ~10% | **100%** (latest batch) | ≥60% | ✅ **RESOLVED** |
| With links | 47% (145/309), avg 8.2 | 100% but many invalid | 100%, avg 3.3 | ≥80%, avg ≥6 | 🟡 **Prefix fixed, count improving** |
| Tags — secondary only | 100%, avg 2.6 | ~40% include primary tags | **100% secondary only** | 100% secondary only | ✅ **RESOLVED** — migrated 9 names |
| Units persisted | 100% | 100% (stored as `canonical_units`) | 100% | 100% | ✅ **RESOLVED** |
| Reviewer score | N/A (human reviewed) | avg 0.54 (range 0.36–0.68) | **avg 0.84** (range 0.50–0.96) | avg ≥0.75 | ✅ **RESOLVED** — 3 outstanding, 4 good |
| ISN validation pass | 100% (catalog is canonical) | ~70% (post-fix) | **100%** (latest 7) | ≥95% | ✅ **RESOLVED** |

---

## Gap Details

### GAP-1: Documentation depth (CRITICAL)

**Metric:** Average documentation character count.

**Evidence:**
- Catalog Q1=529, Q2=2008, Q3=2155, max=3081 chars
- Pipeline avg=473, max=573 chars (below catalog Q1!)
- Catalog entries have structured sections: Opening → Governing equation → Physical significance → Measurement → Typical values → Sign convention → Cross-references
- Pipeline output is 1–2 paragraphs without section structure

**Catalog example** (`poloidal_magnetic_flux_profile`, 1720 chars):
```
Poloidal magnetic flux ψ(ρ) as function of flux surface coordinate...
Governing equation: $$Δ*ψ = −μ₀R²p'(ψ) − FF'(ψ)$$
Physical significance: fundamental solution of Grad-Shafranov equation...
Measurement/Calculation methods: Computed from magnetic equilibrium...
Typical values: ITER: ψ ≈ −5 to 0 Wb...
Sign convention: Positive when poloidal flux increases from axis to boundary.
```

**Pipeline example** (`bootstrap_current`, 483 chars):
```
The bootstrap current I_bs is a non-inductive plasma current driven by
the pressure gradient and neoclassical effects. It is a critical component
for steady-state tokamak scenarios...
```

**Root cause:** Compose prompt says "200-500 characters" for documentation.
ISN catalog targets 150-400 **words** (≈800-2200 chars). The compose system
prompt's documentation template is too brief and doesn't enforce section structure.

**Resolution:** Update compose_system.md documentation template to require:
1. Explicit section structure (opening, equation, significance, measurement, values, sign)
2. Target 800-2200 chars (~150-400 words)
3. Show 2-3 COMPLETE curated examples with full documentation

**Status:** ☐ Not started

---

### GAP-2: DD context starvation (CRITICAL)

**Metric:** Characters of DD context passed to compose LLM per path.

**Evidence:**
- Pipeline passes: description (~30 chars), documentation (~0-100 chars from graph),
  unit, data_type, cluster label, parent path
- ISN interactive workflow calls `fetch_imas_paths` MCP tool → gets 500-2000 chars
  of DD documentation per path including coordinate specs, related paths, physics domain
- The `n.documentation` field in our graph is typically NULL or very short for most
  IMASNode paths

**ISN workflow step** (from prompt):
> "Extract IMAS DD documentation, units, and sign conventions"
This uses `search_imas` + `fetch_imas_paths` MCP tools that return rich documentation.

**Root cause:** The pipeline relies on `n.documentation` from the graph (which is the
raw DD description), not the LLM-enriched documentation that the ISN workflow gets
from MCP tools. Our DD MCP server (`imas-codex serve --dd-only`) has the rich data,
but the pipeline doesn't call it.

**Resolution:** Before compose, fetch rich DD documentation via the same tools our
MCP server exposes. This means calling `fetch_dd_paths()` or the equivalent Python
function to get full documentation for each path in the batch.

**Status:** ☐ Not started

---

### GAP-3: ~~Unit persistence failure~~ — RESOLVED

**Metric:** Fraction of StandardName nodes with non-null unit.

**Evidence (UPDATED):**
- Units ARE correctly persisted as `canonical_units` property (not `unit`)
- `CANONICAL_UNITS` relationships ARE correctly created to `Unit` nodes
- Initial investigation used wrong property name (`sn.unit` vs `sn.canonical_units`)
- Verified: `bootstrap_current` → `canonical_units=A`, `metric_jacobian` → `canonical_units=1`

**Status:** ✅ Resolved — no code change needed. Graph property is `canonical_units`.

---

### GAP-4: Link format inconsistency (HIGH)

**Metric:** Fraction of links with correct `name:` prefix.

**Evidence:**
- 45 links across older names lack the `name:` prefix (bare names like `plasma_current`)
- Recent pipeline output (post-fix) correctly uses `name:` prefix
- ISN catalog uses `name:bootstrap_current` format consistently
- Links currently point only to names already in the graph (44 names)

**Root cause:** Two issues:
1. Older names were generated before the `_normalize_links()` fix
2. Links are constrained to `existing_names` list (only graph names), not the full
   DD namespace

**User direction:** Links should be able to point to ANY DD path. Unresolved links
(pointing to names that don't exist yet) should be treated as temporary and resolved
asynchronously once generation is complete across the DD.

**Resolution:**
1. Fix older names: batch update to add `name:` prefix
2. Remove constraint that links must be in `existing_names`
3. Add async link resolution worker (see GAP-8)

**Status:** ☐ Not started

---

### GAP-5: Primary tag leakage (MEDIUM)

**Metric:** Fraction of names with primary tags in `tags` field.

**Evidence:**
- `toroidal_beta` has tags `['fundamental', 'core-physics', 'global-quantity']` —
  `fundamental` and `core-physics` are primary tags
- `poloidal_beta` has tags `['fundamental', 'equilibrium']` — both primary
- Post-fix names correctly filter to secondary only
- Older names (pre-fix) still have primary tags

**Root cause:** Names generated before `_filter_secondary_tags()` was implemented
still carry primary tags. Also, LLM sometimes ignores the instruction and outputs
primary tags despite explicit "secondary only" guidance.

**Resolution:**
1. Batch-fix older names in graph
2. Strengthen compose prompt to list primary tags as FORBIDDEN in tags field
3. Post-processing filter is already in place (defense in depth)

**Status:** ☐ Partially addressed (new names clean, old names dirty)

---

### GAP-6: Missing equations (HIGH)

**Metric:** Fraction of documentation containing LaTeX governing equations.

**Evidence:**
- Catalog: 72% have governing equations with `$$...$$` display math
- Pipeline: ~40% have equations, often inline ($...$) not display ($$...$$)
- Catalog pattern: "Governing equation:\n$$..equation..$$\nwhere ... (definitions)"
- Pipeline: equations appear inline without dedicated section

**Root cause:** Documentation template in compose_system.md mentions equations
but doesn't make them mandatory. No structured section headers required.

**Resolution:** Make "Governing equation:" a required section in documentation
template. Show display-math format (`$$...$$`) explicitly. Require variable
definitions after every equation.

**Status:** ☐ Not started

---

### GAP-7: Missing sign conventions (HIGH)

**Metric:** Fraction of COCOS-dependent quantities with sign convention paragraph.

**Evidence:**
- Catalog: 65% have "Sign convention: Positive when [condition]."
- Pipeline: ~10% mention sign conventions at all
- Catalog format: Separate paragraph, plain text, not bold
- Many equilibrium quantities are COCOS-dependent (psi, B_t, I_p, q)

**Root cause:** Sign convention guidance in compose_system.md (DS-5) is present
but not enforced as mandatory for COCOS-dependent quantities. The LLM doesn't
know which paths are COCOS-dependent without being told.

**Resolution:**
1. Fetch COCOS dependency info from DD (via `get_dd_cocos_fields()`)
2. Pass COCOS flag per path in the user prompt
3. Make sign convention paragraph mandatory when COCOS flag is set

**Status:** ☐ Not started

---

### GAP-8: No async link resolution (NEW — user requirement)

**Metric:** Fraction of links pointing to existing standard names.

**Evidence:**
- Current pipeline requires links to exist in `existing_names` list at compose time
- When generating the first batch for an IDS, there are few/no existing names
- Links should be able to point to ANY standard name — even ones not yet generated
- ISN catalog links to names across all 20 domain directories

**User direction:** Links should point freely to DD paths. Unresolved links get
resolved asynchronously after generation completes. Requires:
1. New graph state: `link_status = unresolved | resolved | unresolvable`
2. Async link resolution worker that checks if target names exist
3. Quasi-random selection with "last_checked" penalty to avoid spinning on
   unresolvable links (NOT a deterministic ORDER BY — that causes deadlocks)
4. Worker places back unresolved links at back of queue

**Root cause:** Pipeline was designed for small batches where all names are known.
At scale (full DD), names reference each other across batches and domains.

**Status:** ✅ **RESOLVED** — LinkResolutionStatus enum, `sn resolve-links` CLI,
age-weighted claim_unresolved_links(), resolve_links_batch().

---

### GAP-9: Curated examples not fully exploited (MEDIUM)

**Metric:** Information from ISN curated examples visible in compose prompt.

**Evidence:**
- 42 curated examples loaded from ISN package
- Compose prompt shows: name, category, kind, unit, description (~120 chars)
- Missing from prompt: full documentation (1500-2500 chars each)
- ISN interactive workflow has access to full catalog entries via MCP

**Root cause:** `_load_curated_examples()` loads complete YAML entries but the
compose_system.md template only renders abbreviated fields.

**Resolution:** Show 3-5 COMPLETE examples with full documentation in the system
prompt. Use the best examples (poloidal_magnetic_flux_profile, safety_factor,
flux_surface_averaged_parallel_bootstrap_current_density) as gold standards.

**Status:** ☐ Not started

---

### GAP-10: No --paths option for targeted debugging (MEDIUM)

**Metric:** Developer iteration speed.

**Evidence:**
- Every debugging iteration requires running full extract pipeline (graph query,
  classification, batching)
- `--limit` caps raw query, not output — unpredictable results
- ISN workflow accepts specific paths in a markdown list

**Root cause:** CLI was designed for bulk generation, not targeted debugging.

**Resolution:** Add `--paths` flag accepting space-separated DD paths. Paths
bypass graph query and classifier, go directly into batches.

**Status:** ✅ **RESOLVED** — `--paths` flag implemented, implies `--force`.

---

## Priority Order

1. ~~**GAP-3** (unit persistence) — RESOLVED~~
2. **GAP-2** (DD context) — CRITICAL, highest impact on name quality
3. **GAP-1** (doc depth) — CRITICAL, requires prompt rewrite
4. **GAP-10** (--paths flag) — enables rapid iteration for all subsequent fixes
5. **GAP-6** (equations) — HIGH, addressed by prompt rewrite in GAP-1
6. **GAP-7** (sign conventions) — HIGH, requires COCOS context injection
7. **GAP-9** (curated examples) — MEDIUM, amplifies prompt quality
8. **GAP-4** (link format) — HIGH, batch fix + link resolution design
9. **GAP-8** (async link resolution) — NEW, requires new worker design
10. **GAP-5** (primary tags) — MEDIUM, batch fix + already defended

---

## Acceptance Criteria

The pipeline achieves parity with the ISN catalog when:

- [ ] Avg documentation ≥1200 chars (catalog median: 2008)
- [ ] ≥80% of entries have governing equations
- [ ] ≥60% of COCOS-dependent entries have sign conventions
- [ ] 100% of entries have persisted units
- [ ] 100% of links use `name:` prefix format
- [ ] 0% of entries have primary tags in `tags` field
- [ ] Avg reviewer score ≥0.75 (current: 0.54)
- [ ] ISN validation pass rate ≥95%
- [ ] Async link resolution resolves ≥80% of links within 2 passes
