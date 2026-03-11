# Signal Enrichment Pipeline v2

> **Status**: Planning  
> **Priority**: Critical — blocks mapping quality and IDS assembly  
> **Scope**: Discover → Enrich → Map pipeline for FacilitySignal and SignalGroup  
> **Principle**: LLM cost is not a constraint. Quality of descriptions and mappings is the sole objective. Deterministic steps must use function calls, not LLM inference.

## Problem Statement

The current signal enrichment pipeline produces descriptions that are too
generic for high-quality IMAS field mapping. Five critical gaps degrade
enrichment context, and the downstream mapping pipeline receives signal group
descriptions that lack the specificity needed for correct transform generation
and unit handling.

The pipeline must be redesigned as a chained LLM process:
**discover → enrich → map**, where each stage builds on the previous stage's
output and deterministic steps (context fetches, field validation, unit checks,
COCOS lookups) are executed as programmatic function calls rather than LLM
inference.

---

## Architecture Overview

```
┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────┐
│  Seed   │───▶│  Scan   │───▶│   Discover   │───▶│  Enrich   │───▶│   Map   │
│ (trees) │    │ (nodes) │    │  (signals)   │    │  (LLM)    │    │  (LLM)  │
└─────────┘    └─────────┘    │              │    │           │    │         │
                              │ ✦ Normalize  │    │ ✦ Context │    │ ✦ Sect. │
                              │   filters    │    │   inject  │    │   assign│
                              │ ✦ Signal     │    │ ✦ Quality │    │ ✦ Field │
                              │   groups     │    │   assess  │    │   map   │
                              │              │    │ ✦ Post-   │    │ ✦ Valid │
                              │              │    │   prop     │    │   (det.)│
                              └──────────────┘    └───────────┘    └─────────┘
                                    │                   │                │
                                    └───────────────────┴────────────────┘
                                      All share normalized schema fields
```

---

## Phase 1: Normalize Discovery Filters and Schema

**Goal**: Every FacilitySignal has consistent, queryable metadata regardless
of discovery source. Normalization happens at discover time, not enrichment
time.

### 1.1 Define Canonical Filter Fields

Currently, signals from different discovery sources populate different fields
inconsistently:

| Field | JET device_xml | TCV tree_traversal | TCV tdi_scan | TCV wiki_extraction |
|-------|---------------|-------------------|-------------|-------------------|
| `data_source_node` (property) | ✅ set | ❌ null | ❌ null | ❌ null |
| `HAS_DATA_SOURCE_NODE` (edge) | ✅ | ✅ (28,690) | ❌ | ❌ |
| `data_source_name` | ✅ (tree name) | ✅ (tree name) | ✅ (function name) | ✅ (wiki page) |
| `data_source_path` | ✅ (MDSplus path) | ❌ null | ❌ null | ❌ null |
| `discovery_source` | `device_xml` | `tree_traversal` | `tdi_scan` | `wiki_extraction` |
| `tdi_quantity` | ❌ | ❌ | ✅ | ❌ |
| `tdi_function` | ❌ | ❌ | ✅ | ❌ |

**Action**: Add a normalization step at discovery time that ensures:

1. **`data_source_path`**: Always set. For `tree_traversal` signals, derive
   from the `HAS_DATA_SOURCE_NODE→SignalNode.path`. For `tdi_scan`, use the
   TDI function file path. For `wiki_extraction`, use the wiki page URL.

2. **`data_source_node`** (property): Deprecate as a filter field. The
   `HAS_DATA_SOURCE_NODE` edge is the authoritative link to `SignalNode`. The
   property was a convenience that only JET's device_xml scanner populates.
   Remove the property filter in `enrich_worker` (line 2950) and replace with
   an edge-existence check in the claim query or a batch edge-lookup.

3. **`discovery_source`**: Already consistent. No changes needed.

4. **`accessor`**: Already normalized. No changes needed.

### 1.2 Fix Tree Context Filter (Critical Bug)

**File**: `imas_codex/discovery/signals/parallel.py` line 2950  
**Bug**: `tree_signal_ids = [s["id"] for s in signals if s.get("data_source_node")]`

This filters on the `data_source_node` *property*, which is `null` for all
TCV signals (0/32,369). The `HAS_DATA_SOURCE_NODE` *edge* exists for 28,690
TCV signals. Result: tree context (parent, siblings, TDI source, epochs) is
**never injected** for TCV enrichment.

**Fix**: Replace the property-based filter with an edge-existence approach:

**Option A** (preferred): Query edge existence during claim. Add to
`claim_signals_for_enrichment` return clause:
```cypher
OPTIONAL MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
RETURN ... , sn IS NOT NULL AS has_tree_node
```
Then filter: `tree_signal_ids = [s["id"] for s in signals if s.get("has_tree_node")]`

**Option B**: Send all signal IDs to `fetch_tree_context()` and let the Cypher
`MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)` naturally filter. Only
signals with the edge will return results. This is simpler but queries more IDs.

**Impact**: 28,690 TCV signals gain tree context (parent path, sibling paths,
TDI source code, epoch ranges).

### 1.3 Backfill `data_source_path` for Existing Signals

Run a one-time migration to set `data_source_path` for signals that have
`HAS_DATA_SOURCE_NODE` edges:
```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_path IS NULL
SET s.data_source_path = sn.path
```

### 1.4 Normalize enrichment_source Values

Currently `mark_signals_enriched()` does not set `enrichment_source`, leaving
it null for directly-enriched signals. Only `propagate_signal_group_enrichment`
sets `signal_group_propagation`. Add `enrichment_source = 'direct'` to
`mark_signals_enriched()` and `enrichment_source = 'direct_underspecified'`
to `mark_signals_underspecified()`.

---

## Phase 2: Fix Context Injection Gaps

**Goal**: Every enrichment context source reaches the LLM prompt with
maximum available information.

### 2.1 Re-run TDI Scanner for `source_code`

**Problem**: All 189 TDI functions in the graph have empty `source_code`
fields. The graph predates the scanner fix (commit `603604fd`) that corrected
TDI source code serialization. 338 signals reference these functions.

**Action**: Re-scan TDI functions for the affected facilities. This is a
deterministic step — no LLM calls. The scanner fix is already in the codebase.
```bash
uv run imas-codex discover signals tcv --scanner tdi --rescan
```

### 2.2 Inject Deterministic Code References

**Problem**: 2,208 signals are reachable via the graph path:
```
CodeChunk →[CONTAINS_REF]→ DataReference →[RESOLVES_TO_NODE]→ SignalNode
   ←[HAS_DATA_SOURCE_NODE]← FacilitySignal
```
These code references contain real usage examples (how the signal is read,
units used, transformations applied) but are never injected into the
enrichment prompt. Currently only `_fetch_code_context()` (vector search)
is used — it matches by group-level semantic similarity, missing specific
signal-level code references.

**Action**: Add `fetch_signal_code_refs()` — a deterministic graph traversal
that fetches code chunks directly linked to each signal via the path above:
```cypher
MATCH (s:FacilitySignal {id: $signal_id})
      -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
      <-[:RESOLVES_TO_NODE]-(dr:DataReference)
      <-[:CONTAINS_REF]-(cc:CodeChunk)
RETURN cc.text AS code, cc.language AS language,
       cc.source_file AS file LIMIT 3
```
Inject these as `## Direct Code References` in the per-signal prompt section,
before the existing group-level `## Source Code Context`.

### 2.3 Fix Wiki Path Matching for TCV

**Problem**: `_find_wiki_context()` looks up `data_source_path` from the
signal dict, but TCV signals have `data_source_path = null`. Wiki chunks
with `mdsplus_paths_mentioned` (1,829 chunks, 6,955 path references) are
never matched.

**Fix**: After Phase 1.3 backfills `data_source_path`, this resolves itself.
Additionally, add a fallback in `_find_wiki_context()`:
```python
path = signal.get("data_source_path") or signal.get("accessor")
```

### 2.4 Enrich SignalGroup Nodes

**Problem**: `SignalGroup` nodes have `description`, `keywords`, and
`physics_domain` fields in the schema but these are currently only populated
by propagation from the representative signal. The group-level description
is identical to the representative's — it doesn't describe the *group pattern*
(e.g., "Array of poloidal magnetic field measurements from N probes").

**Action**: After enriching the representative, generate a separate
group-level description that captures the pattern semantics:
- What the group represents (e.g., "Indexed set of poloidal magnetic field measurements")
- How members differ (e.g., "Each member corresponds to a different probe/channel index")
- The array structure this maps to in IMAS (e.g., "maps to `magnetics/flux_loop[:]/flux/data`")

This group-level description feeds directly into the mapping pipeline's
`signal_group_detail` context.

---

## Phase 3: Post-Propagation Individualization

**Goal**: After group enrichment is propagated, generate individualized
descriptions for member signals that capture their specific identity within
the group.

### 3.1 Design

Currently, `propagate_signal_group_enrichment()` copies the representative's
description verbatim to all members. For a group of 181 `MAGB` probes, all
get the identical description "Measured magnetic field from probe MAGB_NNN".

**New step**: After propagation, make a second LLM call per group that
takes:

**Input (deterministic, via function calls)**:
- The group description (from Phase 2.4)
- The representative's full enrichment
- List of all member signal IDs, accessors, and names
- Any member-specific tree context (parent path, position info)

**LLM task**: Generate a one-line individualized suffix for each member:
- Representative: "Measured magnetic field from poloidal field probe array"
- Member MAGB_001: "Magnetic field measurement from probe 1 — lower inboard midplane"
- Member MAGB_002: "Magnetic field measurement from probe 2 — upper inboard midplane"

**Output model**:
```python
class IndividualizedDescription(BaseModel):
    signal_id: str
    description: str  # Full individualized description
    position_context: str | None = None  # Location/channel info if available
```

### 3.2 When to Run

This runs as a new pipeline stage **after** `propagate_signal_group_enrichment`
completes for a group. The flow becomes:

1. Enrich representative → `mark_signals_enriched([representative])`
2. Propagate to members → `propagate_signal_group_enrichment(rep_id, enrichment)`
3. Individualize members → `individualize_group_members(group_id)` *(new)*

Step 3 can be batched — process all recently-propagated groups in one pass.

### 3.3 Unwind Descriptions

The user requires that we "unwind the descriptions back to the individual
signals in addition to describing the common pattern node". This means:

- **SignalGroup.description**: Describes the group pattern (from Phase 2.4)
- **FacilitySignal.description**: Each member gets its own individualized
  description (from Phase 3.1) that includes the group context plus
  member-specific details
- **enrichment_source**: Set to `'individualized'` for signals that received
  post-propagation individualization

---

## Phase 4: Enrichment Prompt Refinements

**Goal**: Maximize description quality and context utilization for both
direct enrichment and post-propagation individualization.

### 4.1 Current Enrichment Prompt Assessment

The enrichment prompt (`imas_codex/agentic/prompts/signals/enrichment.md`)
is well-structured with:
- Physics domain classification (enum-constrained)
- Context quality assessment (high/medium/low)
- Anti-hallucination rules for low-context signals
- Diagnostic category matching
- Sign convention extraction hierarchy

**Gaps identified**:
1. No guidance on generating descriptions useful for *downstream mapping* —
   the prompt optimizes for human readability, not IMAS field matching
2. No instruction to include units, coordinate systems, or array dimensions
   in descriptions — these are critical for mapping transforms
3. No mention of COCOS conventions in the enrichment prompt (only in mapping)
4. Group-level pattern description generation is not addressed

### 4.2 Enrichment Prompt Updates

Add a new section: **"Description Requirements for IMAS Mapping"**:

```markdown
## Description Requirements for IMAS Mapping

Your descriptions will be used downstream to map signals to IMAS data dictionary
fields. To support accurate mapping, descriptions MUST include when available:

1. **Physical quantity and measurement type**: What is measured (e.g., "poloidal
   magnetic field", "electron temperature", "plasma current")
2. **Units**: If known from context, state the measurement units
3. **Coordinate system**: Whether data is in R,Z (cylindrical) or ρ,θ (flux)
   coordinates
4. **Array structure**: Whether this is a scalar, 1D profile, 2D array, or
   time-dependent signal
5. **Diagnostic context**: The diagnostic system and measurement technique
6. **Sign convention**: How positive values are defined (especially for
   currents, fields, and fluxes)
```

### 4.3 Context Quality Thresholds

Refine the context_quality assessment to be more actionable:

| Quality | Criteria | Downstream Action |
|---------|----------|-------------------|
| `high` | TDI source code + wiki OR 3+ context sources | Direct to enriched |
| `medium` | Tree context + 1 other source, no source code | Direct to enriched, flag for review |
| `low` | Only accessor/path, no wiki/code/tree context | Mark underspecified, queue for re-enrichment |

---

## Phase 5: Mapping Pipeline Context Injection

**Goal**: The mapping pipeline receives maximum context from enrichment to
produce accurate field mappings with correct transforms.

### 5.1 Current Mapping Pipeline Assessment

The mapping pipeline (`imas_codex/ids/mapping.py`) has a well-designed
4-step architecture:

| Step | Type | Input | Output |
|------|------|-------|--------|
| 0 | Deterministic | Graph queries | Signal groups, IMAS subtree, semantic matches, existing mappings, COCOS paths |
| 1 | LLM | All groups + IMAS tree | Section assignments |
| 2 | LLM (per section) | Group detail + IMAS fields + units + COCOS | Field mapping entries |
| 3 | Deterministic | Proposed mappings | Validated results (source/target existence, transform execution, unit compat) |

**Assessment**: The mapping prompt is lean and well-structured. The
`signal_group_detail` is injected as raw JSON from `query_signal_groups()`,
which returns:
- `id`, `group_key`, `description`, `keywords`, `physics_domain`, `status`
- `member_count`, `sample_members` (first 5 member IDs)
- `imas_mappings` (existing MAPS_TO_IMAS relationships)

### 5.2 Context Injection: Enrichment vs Mapping

**Decision**: Inject context at **both** stages, but with different purposes:

| Context Type | Enrichment Stage | Mapping Stage |
|-------------|-----------------|---------------|
| Signal description | Generated here | Consumed here (via `signal_group_detail`) |
| Units | Extracted & stored | Consumed for transform generation |
| Sign convention | Extracted & stored | Consumed for COCOS handling |
| Tree context (parent, siblings) | Injected for physics understanding | Not needed — description captures this |
| Wiki context | Injected for description quality | Not needed — description captures this |
| Code references | Injected for units/signs/coordinates | Could inject for transform examples |
| COCOS paths | Not relevant at enrichment | Injected here (deterministic) |
| IMAS field structure | Not relevant at enrichment | Injected here (deterministic) |
| Unit compatibility analysis | Not relevant at enrichment | Injected here (deterministic) |

**Key insight**: The enrichment prompt should produce descriptions that
*encode* the context the mapping prompt needs. The mapping prompt then
combines descriptions with deterministic IMAS field metadata. Adding raw
tree/wiki context to the mapping prompt would bloat it without benefit
— the description should already capture the physics meaning.

**Exception — Code references for transforms**: For groups where the
representative has direct code references (via `RESOLVES_TO_NODE`), inject
a `code_context` field into `signal_group_detail` showing how the signal is
read/transformed in existing code. This helps the LLM generate accurate
`transform_expression` values:

```python
# In query_signal_groups() or _step0_gather_context():
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
    -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
    <-[:RESOLVES_TO_NODE]-(dr:DataReference)
    <-[:CONTAINS_REF]-(cc:CodeChunk)
WITH sg, ..., collect(cc.text)[..2] AS code_refs
```

### 5.3 Extend `query_signal_groups()` for Richer Context

Currently the query returns minimal fields. Extend to include:

```python
# Add to tools.py query_signal_groups():
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
RETURN ...
       rep.description AS representative_description,
       rep.sign_convention AS sign_convention,
       rep.unit AS unit,
       rep.diagnostic AS diagnostic,
       rep.discovery_source AS discovery_source,
       rep.accessor AS representative_accessor
```

This gives the mapping LLM:
- The full enriched description (not just group_key)
- Units for transform generation
- Sign convention for COCOS handling
- Diagnostic context for IDS section assignment

### 5.4 Add Member Signal Details for Array Mappings

For groups that map to IMAS arrays (e.g., `magnetics/flux_loop[:]/`), the
mapping needs to know the array dimension. Add member count and sample
accessors to the signal group detail.

This is already partially done (`member_count`, `sample_members`), but
`sample_members` only returns IDs, not accessors. Extend:

```cypher
collect(DISTINCT {id: m.id, accessor: m.accessor, name: m.name})[..5]
    AS sample_members
```

---

## Phase 6: Align Naming to Signal Group Schema

**Goal**: All functions, queries, variables, and prompts use the `SignalGroup`
vocabulary from the schema rename (commit `bce1e696`).

### 6.1 Remaining Old Naming

The schema rename was thorough but some areas need alignment:

| Location | Old Name | New Name | Status |
|----------|----------|----------|--------|
| `parallel.py` L2950 comment | "signals with HAS_DATA_SOURCE_NODE edges" | (keep — accurate) | ✅ |
| `enrichment.md` examples | References to "pattern" in worked examples | Update to "signal group" language | ⬜ |
| `enrichment_source` values in graph data | `pattern_propagation` (historic data) | `signal_group_propagation` (code already updated) | ⚠️ data migration |
| `_signal_context_key()` | Internal — already uses generic keys | (keep) | ✅ |
| `_accessor_to_pattern()` | Describes regex pattern extraction | Rename to `_accessor_to_group_key()` | ⬜ |

### 6.2 Graph Data Migration

Existing enriched signals in the graph may have `enrichment_source = 'pattern_propagation'`
from before the rename. Run migration:
```cypher
MATCH (s:FacilitySignal)
WHERE s.enrichment_source = 'pattern_propagation'
SET s.enrichment_source = 'signal_group_propagation'
```

Also clean up any remaining `pattern_representative_id` or `pattern_template`
properties from before the MEMBER_OF relationship model:
```cypher
MATCH (s:FacilitySignal)
WHERE s.pattern_representative_id IS NOT NULL
REMOVE s.pattern_representative_id, s.pattern_template
```

---

## Phase 7: Data Reset and Re-Discovery

**Goal**: Once the enrichment pipeline is stable and tested, clear existing
signal data and re-run discovery for maximum quality.

### 7.1 Pre-Conditions (Must Complete Before Reset)

All of the following must be implemented, tested, and verified:

- [ ] Phase 1: Filter normalization + tree context fix
- [ ] Phase 2: Context injection gaps filled (TDI source_code, code refs, wiki paths)
- [ ] Phase 3: Post-propagation individualization working
- [ ] Phase 4: Enrichment prompt updated for mapping-quality descriptions
- [ ] Phase 5: Mapping pipeline context injection extended
- [ ] Phase 6: Naming alignment complete
- [ ] Integration tests pass: enrichment → propagation → individualization → mapping
- [ ] Dry run on a small subset (~100 signals) verifying improved descriptions

### 7.2 Reset Procedure

```bash
# 1. Clear all FacilitySignal data for both facilities
uv run imas-codex graph migrate clear-signals --facility tcv
uv run imas-codex graph migrate clear-signals --facility jet

# 2. Clear SignalGroup nodes (they'll be re-detected)
uv run imas-codex graph migrate clear-signal-groups --facility tcv
uv run imas-codex graph migrate clear-signal-groups --facility jet

# 3. Re-discover signals
uv run imas-codex discover signals tcv
uv run imas-codex discover signals jet

# 4. Re-enrich (includes group detection, enrichment, propagation, individualization)
uv run imas-codex discover signals tcv --enrich
uv run imas-codex discover signals jet --enrich

# 5. Re-check
uv run imas-codex discover signals tcv --check
uv run imas-codex discover signals jet --check

# 6. Re-embed
uv run imas-codex embed index --labels FacilitySignal,SignalGroup
```

### 7.3 Assessment: Code Bug vs Outdated Data

The tree context gap (Phase 1.2) is a **code bug** — line 2950 filters on a
property that TCV never populates, regardless of data freshness. Even after
a full re-discovery, the bug would persist. This must be fixed in code.

The TDI source_code gap (Phase 2.1) is **outdated data** — the scanner fix
exists but was applied after the graph was built. A re-scan would fix this
without code changes.

The wiki path matching gap (Phase 2.3) is **both** — `data_source_path` is
not populated at discover time for TCV (data issue), AND `_find_wiki_context()`
doesn't fall back to the accessor (code issue).

**Recommendation**: Fix all code bugs first (Phases 1-6), verify on a small
subset, then do the full data reset (Phase 7).

---

## Implementation Order

```
Phase 1 ──────────────────────────────────────────── Normalize
  1.2  Fix tree context filter              [1 line change, critical]
  1.1  Add has_tree_node to claim query     [claim query update]
  1.3  Backfill data_source_path            [migration script]
  1.4  Normalize enrichment_source          [2 mark functions]

Phase 2 ──────────────────────────────────────────── Context
  2.1  Re-scan TDI functions                [CLI command]
  2.2  Add fetch_signal_code_refs()         [new function + prompt injection]
  2.3  Fix wiki path matching               [1 line change + Phase 1.3]
  2.4  Generate SignalGroup descriptions    [new LLM step]

Phase 3 ──────────────────────────────────────────── Individualize
  3.1  Design individualization model       [new Pydantic model]
  3.2  Implement individualize_group_members [new function]
  3.3  Wire into enrich_worker pipeline     [pipeline integration]

Phase 4 ──────────────────────────────────────────── Enrichment Prompt
  4.1  Add IMAS mapping requirements        [prompt update]
  4.2  Refine context quality thresholds    [prompt update]

Phase 5 ──────────────────────────────────────────── Mapping Context
  5.1  Extend query_signal_groups()         [query update]
  5.2  Add code references to groups        [query + format]
  5.3  Extend sample_members with accessors [query update]

Phase 6 ──────────────────────────────────────────── Naming
  6.1  Rename _accessor_to_pattern          [function rename]
  6.2  Run graph data migrations            [migration script]

Phase 7 ──────────────────────────────────────────── Reset + Re-Run
  7.1  Verify pre-conditions                [checklist]
  7.2  Clear and re-discover all signals    [CLI commands]
  7.3  Validate quality improvement         [comparison metrics]
```

---

## Deterministic vs LLM Steps

A core principle of the chained pipeline is that **deterministic steps use
function calls**, not LLM inference. This table maps each operation:

| Operation | Type | Implementation |
|-----------|------|----------------|
| Fetch tree context (parent, siblings, epochs) | Deterministic | `fetch_tree_context()` — Cypher query |
| Fetch code references for signal | Deterministic | `fetch_signal_code_refs()` — Cypher query (new) |
| Fetch wiki context by path | Deterministic | `_find_wiki_context()` — dict lookup |
| Fetch wiki context by semantics | Deterministic | `_fetch_group_wiki_context()` — vector search |
| Fetch code context by semantics | Deterministic | `_fetch_code_context()` — vector search |
| Fetch TDI source code | Deterministic | `get_tdi_source()` — graph query |
| Detect signal groups | Deterministic | `detect_signal_groups()` — accessor pattern matching |
| Generate signal description | **LLM** | `enrich_worker()` — structured output |
| Assess context quality | **LLM** | Part of enrichment structured output |
| Generate group description | **LLM** | New step in Phase 2.4 |
| Individualize member descriptions | **LLM** | New step in Phase 3 |
| Assign groups to IMAS sections | **LLM** | `_step1_assign_sections()` |
| Generate field mappings | **LLM** | `_step2_field_mappings()` |
| Fetch IMAS subtree structure | Deterministic | `fetch_imas_subtree()` — Cypher query |
| Fetch IMAS field details | Deterministic | `fetch_imas_fields()` — Cypher query |
| Semantic search IMAS paths | Deterministic | `search_imas_semantic()` — vector search |
| Get COCOS sign-flip paths | Deterministic | `get_sign_flip_paths()` — coded tables |
| Analyze unit compatibility | Deterministic | `analyze_units()` — pint library |
| Validate mapping paths exist | Deterministic | `check_imas_paths()` — Cypher query |
| Validate transform execution | Deterministic | `validate_mapping()` — Python eval |
| Check duplicate mappings | Deterministic | `validate_mapping()` — set comparison |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| TCV signals with tree context | 0 / 28,690 | 28,690 / 28,690 |
| Signals with TDI source code | 0 / 338 | 338 / 338 |
| Signals with code references | 0 / 2,208 (unused) | 2,208 / 2,208 |
| Signals with wiki context (path match) | ~0 TCV | ~6,955 path matches |
| Unique descriptions (vs templated) | ~60% | >95% |
| context_quality = high | unknown | >40% |
| context_quality = low (underspecified) | unknown | <10% |
| SignalGroup descriptions (group-specific) | 0 | All groups |
| Mapping field accuracy (manual review on 50 mappings) | baseline TBD | >90% |
| Sign convention coverage | TCV 3.9%, JET 18% | >50% |
