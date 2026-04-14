# Standard Name Quality Parity — Implementation Plan

> Multi-phase plan to close quality gaps between automated `sn generate`
> pipeline output and the ISN catalog standard (309 entries, avg 2008 char
> documentation, 72% with equations, 65% with sign conventions).
>
> Gaps tracked in: `plans/research/standard-names/11-quality-gap-analysis.md`
>
> Reviewed by rubber-duck agent. Key changes from critique:
> - Phase 0 added: benchmark harness FIRST for measurable progress
> - Unit persistence (GAP-3) confirmed resolved — property is `canonical_units`
> - DD context: use library layer (`GraphPathTool`), NOT MCP server
> - Doc length: soft target with content gates, not hard char minimum
> - Link priority formula: fixed inversion (was boosting recent, not stale)
> - Phase 1C: update publish.py before tag migration
> - Link resolution: added `retry_count` and terminal `failed` state

## Phase 0: Benchmark harness

**Rationale (from critique):** Without a baseline measurement framework,
we can't tell which phase helps or regresses. Build this first.

### 0A: Holdout path set

Select 10-15 DD paths that have corresponding ISN catalog entries.
These are the A/B comparison targets — NEVER use them as prompt examples.

**Candidate holdout paths:**
- `equilibrium/time_slice/profiles_1d/psi` → `poloidal_magnetic_flux_profile`
- `equilibrium/time_slice/profiles_1d/q` → `safety_factor`
- `equilibrium/time_slice/global_quantities/beta_pol` → `poloidal_beta`
- `equilibrium/time_slice/global_quantities/beta_tor` → `toroidal_beta`
- `equilibrium/time_slice/global_quantities/li_3` → `internal_inductance`
- `equilibrium/time_slice/global_quantities/ip` → `plasma_current`
- `core_profiles/profiles_1d/electrons/temperature` → `electron_temperature`
- `core_profiles/profiles_1d/electrons/density` → `electron_density`
- `core_profiles/profiles_1d/ion/temperature` → `ion_temperature`
- `equilibrium/time_slice/profiles_1d/j_bootstrap` → `flux_surface_averaged_parallel_bootstrap_current_density`

### 0B: Scoring framework

**Metrics (automated):**
1. Documentation length (chars)
2. Has governing equation ($$...$$)
3. Has sign convention paragraph
4. Has typical values section
5. Has measurement/calculation section
6. Link count and `name:` prefix compliance
7. Tags — secondary only, no primary leakage
8. ISN validation pass/fail (via `create_standard_name_entry()`)
9. Reviewer score (from review worker)

**Comparison method:**
- Run `sn generate --paths <holdout_set>` before and after each phase
- Compare against catalog entries loaded from YAML
- Store results in session SQL for tracking

**Files:** New `benchmarks/sn_quality.py` or session-only script

---

## Phase 1: Foundation fixes (no prompt changes)

### ~~1A: Unit persistence fix (GAP-3)~~ — RESOLVED

Units ARE correctly persisted as `canonical_units` property and
`CANONICAL_UNITS` relationship. Earlier investigation used wrong property
name (`sn.unit` instead of `sn.canonical_units`). No code change needed.

### 1B: --paths CLI flag (GAP-10)

**Problem:** No way to test specific DD paths without running full pipeline.

**Implementation:**
1. Add `--paths` option to `sn generate` CLI accepting space-separated DD paths
2. When `--paths` is set, bypass `extract_dd_candidates()` graph query entirely
3. Fetch each path directly with full context (unit, cluster, parent, COCOS)
4. **Skip classifier** — user explicitly chose these paths
5. **Skip already-named check** — targeted mode is for debugging, always process
6. **Ignore `--ids/--domain/--limit`** when `--paths` is set (strict semantics)
7. Group into batches by IDS

**Files:** `cli/sn.py`, `sources/dd.py` (add `extract_specific_paths()`),
`workers.py` (extract_worker dispatch)

### 1C: Batch fix older names + publish alignment (GAP-4 + GAP-5)

**Problem:** 45 links missing `name:` prefix, primary tags in older names.

**CRITICAL (from critique):** `publish.py` groups by `tags[0]` (line 110, 154).
Removing primary tags will push entries into `unscoped`. Must update publish
to prefer `physics_domain` BEFORE running the tag migration.

**Implementation:**
1. **First:** Update `publish.py` to group by `physics_domain` when available,
   fall back to `tags[0]` only when `physics_domain` is null:
   ```python
   subdir = entry.physics_domain or (entry.tags[0] if entry.tags else "unscoped")
   ```
2. **Then:** Run Cypher migration to fix links and tags:
   ```cypher
   // Fix links: add name: prefix to bare names
   MATCH (sn:StandardName)
   WHERE sn.links IS NOT NULL
   WITH sn, [l IN sn.links |
     CASE WHEN l STARTS WITH 'name:' OR l STARTS WITH 'http'
       THEN l ELSE 'name:' + l END
   ] AS fixed_links
   SET sn.links = fixed_links
   ```
   ```cypher
   // Fix tags: remove primary tags
   MATCH (sn:StandardName)
   WHERE sn.tags IS NOT NULL
   WITH sn, [t IN sn.tags WHERE NOT (t IN
     ['fundamental', 'core-physics', 'equilibrium', 'transport',
      'edge-physics', 'mhd', 'heating', 'diagnostics', 'engineering',
      'plasma-wall', 'energetic-particles', 'radiation'])] AS clean_tags
   SET sn.tags = clean_tags
   ```

**Files:** `publish.py` (grouping logic), graph shell (migration)

---

## Phase 2: DD context enrichment (GAP-2)

**Problem:** Pipeline sends ~30-char descriptions to compose LLM. The ISN
interactive workflow gets 500-2000 chars via MCP tools. Our graph has
avg 81 chars of `documentation` for equilibrium paths — starvation confirmed.

**Key decision (from critique):** Use the shared library layer
(`GraphPathTool.fetch_imas_paths`) directly, NOT the MCP server endpoint.
This avoids formatting/parsing overhead and coupling to chat interface.

### 2A: Enrich extraction with rich DD context

**Measured baseline:** `n.documentation` avg 81 chars for equilibrium.
The `description` field (LLM-enriched) avg ~120 chars. Neither is close to
the 500-2000 chars the ISN workflow provides.

**Design:** Extend `_ENRICHED_QUERY` to surface ALL relevant fields that
`GraphPathTool.fetch_imas_paths` returns (line 507-543 of `graph_search.py`):
- `p.cocos_label_transformation` — COCOS dependency type
- `p.cocos_transformation_expression` — sign convention expression
- `p.coordinate1_same_as`, `p.coordinate2_same_as` — coordinate specs
- `p.timebasepath` — time coordinate
- `p.keywords` — already fetched but verify rendering
- Coordinate spec details via `HAS_COORDINATE`

**Implementation:**
1. Extend `_ENRICHED_QUERY` Cypher to fetch COCOS and coordinate fields
2. For each path, build a structured context block (300-500 chars) combining:
   - Full description + documentation from graph
   - COCOS info: "COCOS psi_like: sign depends on convention"
   - Coordinate specs: "Coordinate 1: rho_tor_norm (dimensionless)"
   - Related paths from parent structure
3. Render this in `compose_dd.md` with dedicated sections per path

**Files:** `sources/dd.py` (extend query), `compose_dd.md` (new template sections)

### 2B: COCOS context injection (GAP-7)

**Implementation:**
1. `_ENRICHED_QUERY` already gets `p.cocos_label_transformation` after 2A
2. In `compose_dd.md`, add per-path COCOS section:
   ```
   {% if item.cocos_label %}- **COCOS dependent** ({{ item.cocos_label }}):
     You MUST include a "Sign convention: Positive when [condition]." paragraph.{% endif %}
   ```
3. No separate COCOS bulk query needed — individual path info sufficient

**Files:** `sources/dd.py`, `compose_dd.md`

---

## Phase 3: Prompt quality upgrade (GAP-1, GAP-6, GAP-9)

**Problem:** Documentation template targets 200-500 chars. Catalog averages
2008 chars with structured sections.

**Key decision (from critique):** Use SOFT length target with CONTENT GATES,
not a hard character minimum. Gate on: equation present, variables defined,
typical values, sign convention when COCOS-dependent.

### 3A: Documentation template rewrite

**Implementation in compose_system.md:**

1. Replace "200-500 characters" with soft guidance:
   > Write comprehensive documentation following the section structure below.
   > Simple dimensionless quantities (e.g., safety_factor) may be 800-1200 chars.
   > Complex profile quantities (e.g., poloidal_magnetic_flux_profile) should
   > be 1500-2200 chars. Documentation under 500 chars is almost always insufficient.

2. Make section structure REQUIRED (not suggested):
   ```
   1. Opening definition — what the quantity is (1-2 sentences)
   2. Governing equation — $$...equation...$$ with ALL variables defined
      immediately after (MANDATORY for physics quantities)
   3. Physical significance — why it matters (2-3 sentences)
   4. Measurement/Calculation methods — how obtained (1-2 sentences)
   5. Typical values — use ranges from tokamak parameter data
   6. Sign convention — "Sign convention: Positive when [condition]."
      (MANDATORY when COCOS dependent flag is set for this path)
   7. Cross-references — [name](#name) links to related quantities
   ```

3. Add content gates (not just length):
   - Physics quantities MUST have at least one display equation ($$...$$)
   - Every variable in an equation MUST be defined with units
   - Typical values MUST reference specific machines (ITER, JET, DIII-D)

### 3B: Full curated examples (GAP-9)

**Implementation:**
1. Show 3 COMPLETE examples with full documentation in system prompt:
   - `poloidal_magnetic_flux_profile` (equilibrium, 1720 chars, 7 links)
   - `safety_factor` (fundamental, 1500 chars, 7 links)
   - `flux_surface_averaged_parallel_bootstrap_current_density` (equilibrium, 1400 chars, 10 links)
2. These MUST NOT overlap with holdout paths (Phase 0A)
3. Keep abbreviated list for remaining 39 examples (vocabulary coverage)
4. In `context.py`, split into:
   - `_load_showcase_examples()` — 3-5 complete entries with full docs
   - `_load_vocabulary_examples()` — all 42 abbreviated for reference

**Files:** `compose_system.md`, `context.py`

---

## Phase 4: Async link resolution (GAP-8)

**Problem:** Links constrained to existing names. At scale, names reference
each other across batches. User requires: links point to any standard name,
unresolved links resolved asynchronously.

### 4A: Schema changes

**Implementation:**
Add to `schemas/standard_name.yaml`:
```yaml
link_status:
  description: Status of link resolution for this name's outgoing links
  range: LinkResolutionStatus
link_checked_at:
  description: When links were last checked for resolution
  range: datetime
unresolved_links:
  description: Links not yet resolved to existing StandardName nodes
  multivalued: true
  range: string
link_retry_count:
  description: Number of resolution attempts for this name
  range: integer
```

Add `LinkResolutionStatus` enum (from critique — needs terminal state):
```yaml
LinkResolutionStatus:
  permissible_values:
    unresolved: { description: "Links not yet checked" }
    partially_resolved: { description: "Some links resolved, some pending" }
    resolved: { description: "All links point to existing StandardName nodes" }
    failed: { description: "Terminal: unresolvable links after max retries" }
```

Rebuild models: `uv run build-models --force`

### 4B: Compose changes — free link generation

**Implementation:**
1. Remove constraint that links must be in `existing_names` list
2. Update compose_system.md links guidance:
   ```
   Reference 4-8 related standard names using `name:` prefix. You may link to:
   - Names from the existing_names list (these are known to exist)
   - Names you expect to exist for related physics quantities
   - Use physics concept names (e.g., `name:electron_temperature`)
   Links are validated and resolved asynchronously after generation.
   ```
3. In persist_worker, set `link_status=unresolved` for all new names
4. Unresolved links MUST NOT appear in published output — publish.py must
   filter or warn on `link_status != resolved`

### 4C: Link resolution worker

**Design:** Separate `sn resolve-links` CLI command (from critique: keeps
`sn generate` deterministic, easier retries/idempotency). Optional auto-trigger
after generate.

**Queue ordering — age-weighted random (FIXED from critique):**

The original formula was inverted — it gave highest priority to recently-checked
items. Fixed: priority INCREASES with staleness.

```cypher
MATCH (sn:StandardName)
WHERE sn.link_status IN ['unresolved', 'partially_resolved']
  AND (sn.link_checked_at IS NULL
       OR sn.link_checked_at < datetime() - duration('PT5M'))
  AND coalesce(sn.link_retry_count, 0) < $max_retries
WITH sn,
     // Priority INCREASES with age — stale items get picked first
     CASE WHEN sn.link_checked_at IS NULL THEN 10.0
          ELSE duration.between(sn.link_checked_at, datetime()).minutes + 1.0
     END AS priority
// Randomize within eligible cohort, weighted by staleness
ORDER BY rand() * priority DESC
LIMIT $batch_size
SET sn.link_checked_at = datetime(), sn.claim_token = $token
```

**Resolution logic:**
```python
for link in name.links:
    target = link.replace('name:', '')
    exists = gc.query(
        "MATCH (sn:StandardName {id: $target}) RETURN sn.id",
        target=target
    )
    if exists:
        resolved.append(link)
    else:
        unresolved.append(link)

if not unresolved:
    SET link_status = 'resolved', unresolved_links = null
elif unresolved != previous_unresolved:
    # Progress made — some links newly resolved
    SET link_status = 'partially_resolved', unresolved_links = unresolved
else:
    # No progress — increment retry count
    SET link_retry_count = coalesce(sn.link_retry_count, 0) + 1
    if link_retry_count >= max_retries:
        SET link_status = 'failed'  # Terminal — stop retrying
    else:
        SET link_status = 'unresolved', unresolved_links = unresolved
```

**Worker lifecycle:**
1. Separate CLI: `sn resolve-links [--max-retries 5] [--batch-size 50]`
2. Iterate until all resolved/failed or max iterations reached
3. `@retry_on_deadlock()` + `claim_token` pattern
4. Publish must check: only output names where `link_status IN ['resolved', null]`

**Files:** `schemas/standard_name.yaml`, `graph_ops.py`, `workers.py`,
`cli/sn.py` (add `sn resolve-links`), `publish.py` (filter unresolved)

---

## Phase 5: A/B validation

### 5A: Targeted path comparison (holdout set)

Run holdout paths through pipeline after EACH phase, compare to catalog.

### 5B: Bulk generation test

Run `sn generate --ids equilibrium --limit 100` with all fixes applied.
Compare full scorecard against catalog baselines.

---

## Documentation Updates

| Target | Update needed |
|--------|---------------|
| `AGENTS.md` | `--paths` flag, `sn resolve-links` command, link resolution |
| `plans/research/standard-names/11-quality-gap-analysis.md` | Status checkboxes |
| Schema reference | Auto-generated after `link_status` schema addition |
| `publish.py` docstrings | Updated grouping logic |

## Implementation Order

```
Phase 0 (benchmark harness)   ── FIRST: enables measurement

Phase 1B (--paths flag)       ─┐
Phase 1C (publish + migration) ─┘── Foundation

Phase 2A (DD context)         ─┐── Context enrichment
Phase 2B (COCOS injection)    ─┘   (test via --paths)

Phase 3A (doc template)       ─┐── Prompt rewrite
Phase 3B (full examples)      ─┘   (test via --paths)

Phase 4A (schema changes)     ─┐
Phase 4B (free links)         ─┤── Link resolution
Phase 4C (resolution worker)  ─┘

Phase 5 (A/B validation)      ── After all phases
```
