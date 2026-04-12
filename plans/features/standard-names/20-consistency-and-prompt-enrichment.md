# 20: DD-Enriched Standard Name Generation

**Status:** Revised after dual-frontier review (Opus 4.6 + GPT 5.4)
**Depends on:** Plan 19 (benchmark parity — DONE)
**Agent type:** Architect (research-driven design + implementation)

## Problem Statement

A standard name is not just a name — it is a **name + documentation + unit +
IMAS path links**. The current pipeline treats these as independent LLM outputs
when they should be derived from authoritative DD context.

### Core gaps

1. **No DD enrichment at compose time.** The pipeline extracts thin path
   records from the graph (path, description, cluster_label) and asks the LLM
   to generate names, units, and documentation from scratch. Meanwhile, the
   DD MCP tools (`fetch_dd_paths`, `search_dd_clusters`, `find_related_dd_paths`)
   surface authoritative units, rich documentation, cluster membership, cross-IDS
   relationships, and coordinate specifications. None of this context is used.

2. **No unit safety.** It is currently possible to group DD paths with
   different units under the same StandardName, assign names to paths with
   undefined units, and let the LLM generate units that disagree with the DD.
   The only "validation" is `len(unit) < 50`. The graph has 6,406 paths with
   `HAS_UNIT->Unit` relationships (418 unique units) — all ignored by the
   compose pipeline. 348 semantic clusters have mixed units across members.

3. **Shallow context produces inconsistent output.** Without per-path DD
   context, the same concept generates different names depending on model and
   batch composition. The partner project (imas-standard-names) achieves
   consistency via MCP tool calls that build rich per-concept context before
   generation. Our pipeline provides a static system prompt + thin batch.

4. **Naming scope is undefined.** The DD has 61,366 nodes. Which ones get
   standard names? The pipeline currently targets dynamic leaf data nodes,
   but doesn't distinguish physics quantities from metadata (time bases,
   validity flags, error bounds, data arrays under structure quantities).

## DD Graph Landscape

### Node population (IMASNode)

| Category | Count | Naming scope |
|----------|-------|-------------|
| **Physics quantity leaves** (FLT_0D/1D/2D, INT_0D, not /data /time /validity /error) | 9,964 | **Primary target** — each gets a StandardName |
| Structure quantities (STRUCTURE/STRUCT_ARRAY, dynamic, with description) | 108 | **Concept containers** — name inherited by children |
| `/data` leaves (data arrays under structures) | 522 | **Metadata** — inherit name from parent structure |
| `/time` leaves (time bases) | 694 | **Metadata modifier** — coordinate, not a concept |
| `/validity` + `/validity_timed` | 153 | **Metadata modifier** — quality flag |
| Error fields (`_error_upper/lower/index`) | excluded | Already excluded from search via `HAS_ERROR` |

### Naming scope model (CF conventions pattern)

Following CF conventions, only **independent physics concepts** get standard
names. Metadata about how the quantity is stored, measured, or validated
is captured via modifiers and relationships, not new names:

```
barometry/gauge/pressure              (STRUCTURE, unit=Pa)
  pressure/data                     -> metadata: "data array of gauge_pressure"
  pressure/time                     -> metadata: coordinate (time base)
  pressure/data_error_upper         -> metadata: error bound
  pressure/validity                 -> metadata: quality flag

-> ONE StandardName: gauge_pressure (unit=Pa)
-> /data, /time, /error, /validity are metadata of that name
```

**Key principles:**
- **Method independence**: Same quantity measured differently -> same name.
  `electron_temperature` whether from Thomson scattering or ECE.
- **Processing as metadata**: `filtered_`, `smoothed_` are not name segments
  but processing metadata.
- **Structure = concept, children = storage**: When a STRUCTURE node has
  `HAS_UNIT` and a description, it IS the physics concept. Its `/data`,
  `/time`, `/validity` children are storage artifacts.

### Graph relationships for enrichment

| Relationship | Direction | What it tells us |
|-------------|-----------|-----------------|
| `HAS_UNIT` | `(IMASNode)->(Unit)` | **Authoritative unit** — source of truth |
| `IN_CLUSTER` | `(IMASNode)->(IMASSemanticCluster)` | **Concept identity** — same physics across IDSs |
| `HAS_PARENT` | `(IMASNode)->(IMASNode)` | **Structural context** — parent structure, sibling quantities |
| `HAS_COORDINATE` | `(IMASNode)->(IMASNode)` | **Coordinate context** — what grid/axis this data lives on |
| `IN_IDS` | `(IMASNode)->(IDS)` | **IDS membership** — batching scope |
| `COORDINATE_SAME_AS` | `(IMASNode)->(IMASNode)` | **Coordinate equivalence** — shared grids |

**Key stats:**
- 8,943 physics leaves with cluster membership (94.8% of 9,964)
- 2,535 unique clusters -> approximate number of unique StandardNames
- Average 7.1 paths per cluster (cross-IDS replication)
- 1,465 clusters are unit-homogeneous, 348 have mixed units
- Average 2.0 clusters per path (many-to-many)

### Mixed-unit cluster examples

These clusters contain paths with genuinely different units — they represent
related but distinct quantities that must not share a StandardName:

| Cluster | Units | Why mixed |
|---------|-------|-----------|
| pressure | Pa, J/m3 | Pressure vs energy density (dimensionally equivalent) |
| psi_like COCOS fields | Wb, Wb/m | Flux vs flux derivative |
| Coherent Wave Field Profiles | 13 units | Wave amplitude, frequency, power — different quantities |
| toroidal field | T, s, m.T, V | Field strength, time, flux, voltage |

## Architecture

### Core insight

**Rich authoritative context from DD tools enables cheaper models to produce
gold-standard output.** Instead of relying on expensive reasoning models to
infer physics, we programmatically navigate the DD graph to surface the exact
context each path needs, then let the LLM focus on naming and documentation.

### Current flow (thin context, LLM hallucinates)

```
graph query (path, desc, cluster_label)
    -> static system prompt
    -> batch user prompt
    -> LLM generates name + unit + docs + links
```

### Proposed flow (DD-enriched, two-pass with consolidation)

```
graph navigation (path -> unit, siblings, coordinates, parent structure)
    |
    v
classify: physics_quantity | metadata_modifier | skip
    |
    v
select PRIMARY cluster per path (narrowest scope, highest similarity)
    |
    v
group by (primary_cluster x unit) — GLOBAL, not IDS-scoped
    |
    v
split oversized groups (token-budget estimator, ~20-30 concepts/batch)
    |
    v
PASS 1: GENERATE — parallel LLM compose (enriched context, unit GIVEN)
    |
    v
PASS 2: CONSOLIDATE — cross-batch dedup, name harmonization, conflict detect
    |
    v
validate: imas_standard_names Pydantic + grammar roundtrip + unit match
    |
    v
persist: conflict-detecting writes (NOT coalesce — fail on mismatch)
```

### DD tool integration

| Tool | Data surfaced | Pipeline use |
|------|---------------|-------------|
| `fetch_dd_paths()` | Full description, data_type, coordinates, cluster labels | Per-path documentation grounding |
| `search_dd_clusters()` | All paths sharing concept across IDSs | Concept identity, `ids_paths` auto-population |
| `find_related_dd_paths()` | Cross-IDS siblings by similarity, coordinates, units | Collision detection before generation |
| `check_dd_paths()` | Path existence validation, typo suggestions | Link validation before persistence |
| **Custom graph queries** | `HAS_UNIT->Unit`, `HAS_PARENT->Structure`, `HAS_COORDINATE->Grid` | Unit source of truth, structural context, coordinate context |

All DD tools are fast graph queries (not remote API calls). The enrichment
layer combines tool calls with custom Cypher for structural navigation:

```cypher
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (sibling:IMASNode)-[:IN_CLUSTER]->(c) WHERE sibling.id <> n.id
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coord:IMASNode)
OPTIONAL MATCH (coord)-[:HAS_UNIT]->(cu:Unit)
OPTIONAL MATCH (n)-[:IN_IDS]->(ids:IDS)
RETURN n, u.id AS unit, c.label AS cluster,
       collect(DISTINCT sibling.id) AS siblings,
       parent.id AS parent_path, parent.data_type AS parent_type,
       coord.id AS coord_path, cu.id AS coord_unit, ids.id AS ids_name
```

## Unit Safety Model

### Invariants (enforced by pipeline, not LLM)

1. **Unit comes FROM the DD.** `canonical_units` is set from
   `IMASNode.HAS_UNIT->Unit`, never from LLM output. The prompt includes
   the authoritative unit as a given fact.

2. **Same StandardName => same unit.** If multiple DD paths share a
   StandardName, all MUST have the same unit. Different units =>
   different quantities => different names.

3. **No unit => classify first.** Paths without `HAS_UNIT`:
   - Genuinely dimensionless (beta, safety_factor, elongation) -> `unit=null`, `kind=scalar`
   - Metadata (indices, flags, identifiers) -> `kind=metadata`, skip naming
   - Missing data -> flag for manual review, do not generate name

4. **Mixed-unit clusters are split.** Group by `(cluster x unit)` not just
   cluster. The "pressure" cluster (Pa vs J/m3) produces two separate
   batches with two separate names.

## Phases

### Phase 1: Naming scope classification

Define which IMASNode paths receive standard names and which are metadata.

**Path classifier** (`imas_codex/sn/classifier.py`):

```python
def classify_path(node: dict) -> Literal["quantity", "metadata", "skip"]:
    """Classify a DD path for standard name generation.

    quantity: Independent physics concept -> gets a StandardName
    metadata: Storage artifact (data, time, validity, error) -> modifier
    skip:     Not a nameable concept (container, identifier, etc.)
    """
```

**Classification rules:**
- `/data` under STRUCTURE parent with `HAS_UNIT` -> `metadata` (parent owns the name)
- `/time` -> `metadata` (coordinate, not concept)
- `/validity`, `/validity_timed` -> `metadata` (quality flag)
- `_error_upper`, `_error_lower`, `_error_index` -> already excluded via HAS_ERROR
- `STR_0D` (string fields: names, identifiers, descriptions) -> `skip`
- `INT_0D` without `HAS_UNIT` AND description matches index/flag pattern
  (`"index"`, `"flag"`, `"identifier"`, `"number of"`, `"count"`) -> `skip`
- `INT_0D` without `HAS_UNIT` but with physics description (e.g., `"toroidal
  mode number"`) -> `quantity` (unit=null, genuinely dimensionless)
- Physics leaf with `HAS_UNIT` and description -> `quantity`
- Physics leaf without `HAS_UNIT`, genuinely dimensionless -> `quantity` (unit=null)
- STRUCTURE with `HAS_UNIT` and physics description -> `quantity`
  (children /data, /time etc are metadata of this concept)

**Hardening (from review):** Use `node_category` field to disambiguate
edge cases. Maintain a reviewed gold set of ~50 edge-case paths with
expected classifications. Classifier tests parametrized from this gold set.

### Phase 2: DD enrichment layer (`imas_codex/sn/enrichment.py`)

New module that navigates the DD graph to build rich context for each path.

**Functions:**

```python
def select_primary_cluster(path: str, clusters: list[dict]) -> dict | None:
    """Choose ONE primary cluster per path from its many-to-many memberships.

    Resolution order:
    1. IDS-scope cluster (most specific)
    2. Domain-scope cluster
    3. Global-scope cluster
    Within same scope: highest embedding similarity score.

    This eliminates the cartesian product: a path in 2 clusters
    generates 1 batch assignment, not 2.
    """

async def enrich_dd_batch(batch: ExtractionBatch, gc: GraphClient) -> EnrichedBatch:
    """Enrich a batch with authoritative DD context via graph navigation.

    For each path:
    1. Custom Cypher: unit, cluster, parent, coordinates (single query)
    2. Primary cluster selection (deduplicate multi-cluster paths)
    3. Cluster siblings for concept context (informational, not authority)

    Returns EnrichedBatch with unit-validated, context-rich items.
    """

def group_by_concept_and_unit(
    items: list[dict],
    max_batch_size: int = 25,
) -> list[ExtractionBatch]:
    """Group by (primary_cluster x unit) GLOBALLY, not per-IDS.

    Critical design decisions (from dual-frontier review):
    - GLOBAL grouping: same concept across IDSs → same batch → same name
    - Primary cluster: each path appears in exactly ONE batch (no cartesian product)
    - Mixed-unit clusters split into separate batches
    - Oversized groups split by token budget estimator
    - Unclustered paths: sub-group by parent_path, not catch-all bucket
    """
```

**Why global grouping, not IDS-scoped (review finding #1):**

IDS-scoped batching violates method independence. `electron_temperature`
in `core_profiles` and `edge_profiles` would land in different batches,
generating different names. Global (cluster × unit) grouping ensures all
paths for the same concept enter the same LLM call, so the LLM sees the
full cross-IDS context and produces one canonical name.

**Why primary cluster selection (review finding #2):**

Paths average 2.0 cluster memberships (many-to-many). Without primary
cluster selection, a path in 2 clusters generates 2 rows → 2 batch
assignments → 2 independent name generations. The primary cluster
eliminates this cartesian product: each path appears in exactly one batch.

**Enriched item structure** (what the user prompt receives):

```python
{
    "path": "core_profiles/profiles_1d/electrons/temperature",
    "description": "Electron temperature (Te) 1D radial profile...",
    "unit": "eV",                         # AUTHORITATIVE from HAS_UNIT
    "data_type": "FLT_1D",
    "coordinates": ["core_profiles/profiles_1d/grid/rho_tor_norm"],
    "coordinate_units": ["dimensionless"],
    "cluster_label": "electron temperature",
    "cluster_siblings": [                  # same concept, other IDSs
        "edge_profiles/profiles_1d/electrons/temperature (eV)",
        "plasma_profiles/profiles_1d/electrons/temperature (eV)",
        "turbulence/profiles_2d/electrons/temperature (eV)",
    ],
    "parent_path": "core_profiles/profiles_1d/electrons",
    "parent_type": "STRUCTURE",
    "physics_domain": "transport",
    "ids_name": "core_profiles",
}
```

## Unclustered Path Handling

~500 paths (5.2%) have no cluster membership. Without special handling,
these form giant catch-all batches per IDS with no concept context.

**Strategy:**
1. Sub-group unclustered paths by `parent_path` — structural siblings
   are likely related quantities (e.g., all children of `electrons/`)
2. If parent sub-group has >1 path with same unit, batch together
3. Singleton unclustered paths: process individually or flag for manual review
4. Never lump unrelated unclustered paths into a single catch-all batch

**Rationale:** Parent-based sub-grouping provides structural context even
without semantic clusters. A path like `wall/temperature` unclustered is
still meaningfully grouped with `wall/heat_flux` if they share a parent.

## Concept Registry (Cross-Run Consistency)

**Problem:** Even with consolidation within a single run, subsequent runs
can generate different names for the same concept (LLM non-determinism).

**Solution:** Maintain a concept registry — a mapping from concept
signature to approved standard name. Before generation:

1. **Lookup:** Check if concept already has an approved name in graph
   (`StandardName` with `review_status = 'accepted'`)
2. **Reuse:** If found, skip generation — reuse existing name
3. **Generate:** Only generate names for concepts not yet in registry
4. **Register:** After consolidation + validation, new names become
   candidates (status `'drafted'`) available for reuse in future runs

**Concept signature:** `(primary_cluster_id, unit)` — the same key used
for batching. This ensures deterministic concept identity across runs.

## Dual-Frontier Review Findings

Plan 20 was reviewed by **Claude Opus 4.6** and **GPT 5.4** independently.
Both reviews converged on the same critical issues, increasing confidence
in the findings. All blocking issues have been addressed in this revision.

### Convergent findings (both reviewers agreed)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | 250-path batch exceeds all model output limits | **Blocking** | Revised to 20-30 concepts. Token-budget estimator added. |
| 2 | No cross-batch dedup/consistency mechanism | **Blocking** | Added Pass 2 consolidation before persistence. |
| 3 | Multi-cluster cartesian product (2.0 clusters/path → 2× rows) | **Blocking** | Primary cluster selection per path. |
| 4 | `coalesce()` silently masks unit/metadata conflicts | **Blocking** | Conflict-detecting writes replace coalesce. |
| 5 | INT_0D classification ambiguity (index vs dimensionless) | High | Keyword heuristics + gold set of edge cases. |
| 6 | Unclustered paths form giant catch-all batches | Medium | Parent-based sub-grouping. |

### Opus 4.6 unique findings

| Finding | Resolution |
|---------|------------|
| Plan conflates done vs proposed work | Added "Current state" annotations to each phase. |
| Unit validation still `len(unit)<50`, not wired to Pint | Phase 4 now includes unit match validation against DD source. |
| CF "cell methods" equivalent missing | Deferred — noted as future work for distinguishing time-averaged vs instantaneous. |

### GPT 5.4 unique findings

| Finding | Resolution |
|---------|------------|
| `ids_paths` auto-population from clusters unsafe | Changed to validated equivalence, not raw cluster membership. |
| CF contradiction: device segment puts method back in name | Fixed — device is for physical objects, not measurement methods. |
| If unit is authoritative, why ask LLM to output it? | Removed unit from LLM output model. Injected from DD at persistence. |
| Need concept registry/cache for reuse-first policy | Added Concept Registry section. |
| Need coverage accounting | Added to consolidation: every path → mapped \| skipped \| review. |

### Deferred items (out of scope, noted for future plans)

- **CF "cell methods" equivalent** — distinguishing time-averaged, flux-surface-averaged,
  line-integrated variants of the same quantity. Requires DD enrichment with coordinate
  semantics that goes beyond current cluster/unit grouping.
- **DD version/rename tracking** — stability of `ids_paths` across DD versions.
  Graph has all versions but names should reference a specific DD version range.
- **COCOS/sign-convention enrichment** — COCOS-dependent quantities may need
  convention metadata on StandardName nodes.

### Phase 3: Enriched prompt templates

Update `compose_dd.md` to render DD-enriched context. Unit is presented as
an authoritative given fact, cluster siblings establish concept identity, and
coordinate context grounds the LLM in the path's dimensionality.

**Key prompt changes:**
- Unit presented as GIVEN FACT: "The `unit` field is pre-filled from the IMAS
  Data Dictionary. Copy it exactly. Do NOT substitute, convert, or omit it."
- Cluster siblings provide concept identity across IDSs
- Coordinate context grounds dimensionality
- Anti-pattern examples for common naming mistakes
- **Unit removed from LLM output model** (review finding): unit is authoritative,
  not a generation target. LLM composes name + documentation + tags. Unit is
  injected from DD at persistence time, bypassing LLM entirely.

**Current state:** `compose_dd.md` already renders enriched fields (description,
documentation, unit, data_type, physics_domain, keywords, ndim, cluster info,
siblings, parent structure, coordinates). Remaining work: remove unit from
SNCandidate output model, add anti-pattern examples.

### Phase 4: Two-pass persistence with consolidation

**Pass 1 → Pass 2 architecture (review finding #5):**

LLM compose runs in parallel across batches. Before persistence, a
consolidation pass detects and resolves cross-batch conflicts.

#### Pass 1: Generate (parallel)
Compose workers generate candidates in parallel. Each batch produces
`list[SNCandidate]`. Results accumulate in `state.validated`.

#### Pass 2: Consolidate (serial, before any graph writes)

```python
def consolidate_candidates(candidates: list[dict]) -> ConsolidationResult:
    """Cross-batch dedup and conflict detection.

    Checks:
    1. No duplicate standard_name with different units
    2. No duplicate standard_name with different kind
    3. No source path claimed by multiple candidates
    4. Coverage: every non-skipped source path mapped exactly once
    5. Concept registry lookup: if name already exists in graph, reuse it

    Returns ConsolidationResult with:
    - approved: list[dict] — ready for persistence
    - conflicts: list[ConflictRecord] — need resolution
    - coverage_gaps: list[str] — unmapped paths
    """
```

#### Conflict-detecting persistence (review finding #3)

Replace `coalesce(b.field, sn.field)` with conflict-detecting writes:

```python
# Instead of: SET sn.canonical_units = coalesce(b.unit, sn.canonical_units)
# Use: fail-on-mismatch pattern
"""
MATCH (sn:StandardName {id: $name})
WHERE sn.canonical_units IS NOT NULL AND sn.canonical_units <> $unit
RETURN sn.id AS conflict, sn.canonical_units AS existing, $unit AS incoming
"""
```

If any conflicts are returned, raise `UnitConflictError` with details.
Only after conflict check passes do we MERGE with SET (not coalesce).

#### `ids_paths` population (review finding: clusters ≠ authority)

`ids_paths` is populated from **validated equivalence**, not raw cluster
membership. A path qualifies for `ids_paths` only if:
1. It shares the same primary cluster AND same unit
2. The classifier marks it as `quantity` (not metadata/skip)
3. It has been named in the same consolidation pass

Cluster siblings are **context for the LLM** (informational), not
**authority for path linking** (which requires validation).

### Phase 5: Benchmark approach profiles

Extend `sn benchmark` to test enrichment vs model choice:

| Profile | Description | Tests |
|---------|-------------|-------|
| `baseline` | Current thin context | Control |
| `dd-enriched` | Full DD graph navigation enrichment | Context quality impact |
| `dd-enriched-small-batch` | Enrichment + batch=5 | Batch size x context interaction |

Run each with Claude Sonnet (expensive) AND Gemini Flash Lite (cheap).
**If enrichment closes the quality gap, we use cheap models for production** —
the context does the heavy lifting, not the model's reasoning.

### Phase 6: Documentation

Write architectural decisions to `docs/` as they crystallize.

| Document | Content |
|----------|---------|
| `docs/architecture/standard-names.md` | **Architecture doc**: naming scope model, unit safety invariants, DD enrichment flow, metadata modifier pattern, graph navigation queries. The canonical reference for how SN generation works. |
| `docs/architecture/standard-names-decisions.md` | **Decision log**: CF conventions alignment, cluster-based concept identity, unit-as-given pattern, metadata vs quantity classification rules, mixed-unit splitting strategy. Records *why* each design choice was made. |
| `AGENTS.md` | Updated SN section: enrichment layer, unit model, benchmark profiles |

These documents must be written **as each phase completes**, not deferred.
Phase 1 -> document naming scope classification rules.
Phase 2 -> document enrichment architecture and graph queries.
Phase 4 -> document unit safety invariants.
Phase 5 -> document benchmark evidence for model/approach selection.

## Implementation Order

1. **Phase 1** — Naming scope classification (defines WHAT gets named)
2. **Phase 2** — DD enrichment layer (primary cluster selection + global grouping)
3. **Phase 3** — Enriched prompt templates (unit removed from LLM output)
4. **Phase 4** — Two-pass persistence (consolidation + conflict-detecting writes)
5. **Phase 5** — Benchmark profiles (measures impact, validates model + batch size)
6. **Phase 6** — Documentation (writes decisions as they're made — interleaved)

**Current state (already implemented):**
- `sources/dd.py`: enriched Cypher query with HAS_UNIT, IN_CLUSTER, HAS_PARENT,
  HAS_COORDINATE, sibling lookup. Groups by (IDS × cluster × unit) — **needs
  revision to global grouping with primary cluster selection**.
- `compose_dd.md`: renders authoritative unit, cluster context, siblings, parent
  structure, coordinates, keywords, physics domain. **Needs: remove unit from
  output model, add anti-pattern examples.**
- `units`→`unit` property fix across entire pipeline (commit `ded4870c`).
- SN benchmark config with 7 compose-models (commit `6d78a0e5`).

## Success Criteria

- [ ] Path classification separates 9,964 physics quantities from metadata
- [ ] Every StandardName has `canonical_units` sourced from DD `HAS_UNIT`
- [ ] Paths with different units CANNOT share a StandardName (enforced)
- [ ] Paths with missing units are classified (dimensionless vs flagged)
- [ ] Mixed-unit clusters split into separate batches before generation
- [ ] Each path appears in exactly ONE batch (primary cluster selection)
- [ ] Cross-batch consolidation detects name/unit conflicts before persistence
- [ ] Conflict-detecting writes replace coalesce pattern
- [ ] `ids_paths` populated from validated equivalence, not raw cluster membership
- [ ] Unclustered paths handled explicitly (parent-based sub-groups or manual review)
- [ ] Batch size capped by token-budget estimator (~20-30 concepts)
- [ ] Coverage accounting: every target path → mapped | skipped | review
- [ ] Benchmark: enriched + cheap model >= thin + expensive model
- [ ] `docs/architecture/standard-names.md` written and reviewed
- [ ] Enrichment adds <2s per batch (graph queries, not remote calls)

## Resolved Questions

1. **Dimensional equivalence** — Units are already pint-normalized on ingest
   via `normalize_unit_symbol()` from `imas_codex.units`. Group by exact
   normalized unit string. Pa and J.m^-3 produce separate batches despite
   same dimensionality. Could add optional re-normalization pass for
   consistency but not required — the graph stores pint-canonical forms.
   **Note:** Graph units use `.0` fractional exponents (e.g., `m^2.0.s^-1`)
   that fail `imas_standard_names` regex validation. Strip `.0` suffixes
   before passing to validation: `re.sub(r'\^(-?\d+)\.0', r'^\1', unit)`.

2. **Structure quantity naming** — Name comes from the **concept**, not the
   method. Method independence is a core principle — same quantity measured
   differently gets the same name. Example:
   `langmuir_probes/embedded/t_e` → `electron_temperature` (not
   `langmuir_probe_electron_temperature`).

   **Clarification (from review):** The `device` grammar segment is for
   cases where the device defines a **distinct physical quantity**, not
   a measurement method. `langmuir_probe` as a modifier violates method
   independence. Valid uses: `tokamak_wall_temperature` (the wall IS the
   object), `divertor_heat_flux` (the divertor IS the location).
   Measurement method belongs in metadata/tags, not in the name.

3. **Batch size with enrichment** — ~~Target 250-path batches.~~ **Revised
   after review:** 250 paths × ~500 output tokens/name = ~125K tokens,
   exceeding all model output limits (Claude 8K default, Gemini 8K). Both
   reviewers flagged this as blocking. **Revised target: 20-30 concepts
   per batch.** Implement a token-budget estimator that serializes one
   `SNCandidate` to JSON, counts tokens, and caps batch size accordingly.
   Benchmark will validate optimal size. Note: "concepts" not "paths" —
   a concept may map to 7+ cross-IDS paths but produces one output entry.

4. **Graph vs DD completeness** — The graph matches the DD exactly (built
   from all DD versions via `build_dd.py`). All unit information in the DD
   is already in the graph via `HAS_UNIT` relationships and `unit` property
   on IMASNode. No gap to fill via `fetch_dd_paths`.

5. **Signals source enrichment** — Future work. Not in scope for this plan.

## Pre-Persistence Validation

### imas_standard_names validation layer

Every proposed StandardName MUST be validated against the `imas_standard_names`
package before graph persistence. The package provides:

1. **Pydantic model validation** — `StandardNameScalarEntry` enforces:
   - `name`: snake_case pattern `^[a-z][a-z0-9_]*$`
   - `unit`: dot-exponent pattern `^[A-Za-z0-9]+(\^[+-]?\d+)?(\.[A-Za-z0-9]+(\^[+-]?\d+)?)*$|^$`
   - `description`: required, max 180 chars
   - `documentation`: required
   - `physics_domain`: required
   - `tags`: must be from controlled vocabulary (29 primary + 56 secondary)
   - `kind`: literal `"scalar"` | `"vector"` | `"metadata"`

2. **Grammar parsing** — `parse_standard_name()` decomposes into segments,
   `compose_standard_name()` reassembles. If roundtrip != original, the
   name has invalid grammar structure.

3. **Quality checks** — `run_quality_checks()` checks domain-specific
   rules, prose quality (proselint), and semantic consistency.

### Invalidation strategy: Selective per-name

Names within a batch are semi-independent — each maps to specific IMAS
paths. Rejecting one valid name because another in the batch failed
wastes LLM output.

**Strategy:**
1. LLM generates batch of N candidates
2. Each candidate validated individually:
   a. Construct `StandardNameScalarEntry` (or Vector/Metadata by kind)
   b. Run `parse_standard_name()` → `compose_standard_name()` roundtrip
   c. Check unit consistency: all source paths must have same unit
3. **Valid names** → proceed to graph persistence
4. **Invalid names** → logged with validation errors, status set to
   `"validation_failed"`, not persisted to graph
5. **Retry opportunity** — failed names can be retried in a subsequent pass
   with the validation error as additional LLM context

**Why selective, not batch-level:**
- Names are not interdependent within a batch (no collision avoidance)
- Unit-safe grouping (primary_cluster × unit) prevents cross-contamination
- Rejecting an entire batch of 20-30 concepts because 1 name fails is wasteful
- Failed names get explicit error messages for targeted retry

### Enriched item structure (what the user prompt receives)

```python
{
    "path": "core_profiles/profiles_1d/electrons/temperature",
    "description": "Electron temperature (Te) 1D radial profile...",  # LLM-enriched
    "documentation": "Temperature",                                    # raw DD doc
    "unit": "eV",                                  # AUTHORITATIVE from HAS_UNIT
    "data_type": "FLT_1D",
    "physics_domain": "transport",
    "keywords": ["Te", "core profiles", "transport", "thermal energy"],
    "ndim": 1,
    "coordinates": ["core_profiles/profiles_1d/grid/rho_tor_norm"],
    "coordinate_units": ["dimensionless"],
    "cluster_label": "electron temperature",
    "cluster_description": "Electron thermal energy in the core plasma",
    "cluster_siblings": [                           # same concept, other IDSs
        {"path": "edge_profiles/.../temperature", "unit": "eV"},
        {"path": "turbulence/.../temperature", "unit": "eV"},
    ],
    "parent_path": "core_profiles/profiles_1d/electrons",
    "parent_description": "Electron fluid properties...",
    "parent_type": "STRUCTURE",
    "ids_name": "core_profiles",
}
