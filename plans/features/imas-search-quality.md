# IMAS Search Quality Improvement Plan

## Problem Statement

Vector semantic search for IMAS Data Dictionary paths is severely degraded:
**P@1 = 0%** across all embedding dimensions (256→1024). The root cause is
the **LLM enrichment text design** — descriptions are optimised for human
reading, not for vector retrieval — compounded by embedding ALL 20K nodes
including 7K structural accessors that carry no unique physics semantics.

### Evidence (3 benchmark rounds, 20 queries each)

**Benchmark 1 — Embedding dimension (256→1024, enriched text):**

| Dimension | P@1 | MRR | Conclusion |
|-----------|-----|-----|------------|
| 256 | 0% | 0.048 | Dimension is NOT the problem |
| 512 | 0% | 0.049 | |
| 1024 | 0% | 0.052 | |

**Benchmark 2 — Text strategy (20K corpus, dim=256):**

| Strategy | Avg chars | P@1 | P@5 | MRR | Conclusion |
|----------|-----------|-----|-----|-----|------------|
| embed_text (current) | 486 | 0% | 0% | 0.016 | Catastrophic — wrapping kills signal |
| desc_kw (desc + keywords) | 335 | 0% | 0% | 0.013 | Worse than current |
| desc_only (LLM enriched) | 263 | 0% | 10% | 0.066 | Long LLM text dilutes signal |
| desc_name (desc + path) | 134 | 0% | 20% | 0.099 | Shorter helps |
| first_sent (1st sentence) | 104 | 10% | 25% | 0.171 | 10× better |
| **doc_only (raw DD text)** | **82** | **15%** | **15%** | **0.175** | **Best — shortest wins** |

**Benchmark 3 — Hybrid strategies (20K corpus, dim=256):**

| Strategy | P@1 | MRR | Conclusion |
|----------|-----|-----|------------|
| doc_only | 15% | 0.175 | Confirmed best |
| first_sent | 10% | 0.171 | Close second |
| hybrid_25 (doc≥25 else desc) | 0% | 0.051 | Mixing styles destroys embedding space |
| hybrid_parent (parent desc) | 0% | 0.040 | Parent context HURTS leaf retrieval |
| doc_kw3 (doc + 3 keywords) | 5% | 0.104 | Keywords add noise |
| kw_only (just keywords) | 0% | 0.050 | Not enough signal |
| optimal (parent-inject) | 0% | 0.029 | Worst — inconsistent texts |

**Benchmark 4 — Corpus filtering (desc_only text, dim=256):**

| Corpus | Nodes | P@1 | MRR | Conclusion |
|--------|-------|-----|-----|------------|
| all | 20,037 | 0% | 0.066 | |
| no-accessor | 16,175 | 0% | 0.066 | Marginal improvement |
| focused | 14,157 | 5% | 0.092 | Small improvement |
| struct-only | 4,764 | 0% | 0.037 | Too coarse |

### Root Causes (validated)

1. **LLM descriptions are too long for vector retrieval**: The enriched
   descriptions average 263 chars — packed with contextual physics detail
   that is excellent for reading but spreads the semantic signal too thin
   for a 256-dim embedding. Raw DD docs (82 chars avg) outperform them by
   2.7× MRR.

2. **The `generate_embedding_text()` wrapper is catastrophic**: Adding path
   context, units, coordinates, data type, keywords to the embedding text
   inflates it to 486 chars and produces P@1=0%, MRR=0.016 — the worst of
   all strategies tested.

3. **Mixing text styles destroys embedding space coherence**: Hybrid strategies
   that use raw docs for some nodes and LLM descriptions for others produce
   WORSE results than either alone (MRR drops 60–80%). The embedding space
   needs consistent document style.

4. **Parent context injection hurts leaf retrieval**: Using parent descriptions
   for sparse-doc nodes ("value: The total stored thermal energy...") produces
   MRR=0.040 — worse than any single-strategy approach. Parent descriptions
   describe the CONTAINER, not the specific quantity.

5. **7,029 accessor terminals pollute the corpus**: Nodes like `grid_index`
   (797×), `coefficients` (642×), `time` (572×), and `value` (853× with
   doc="Value") carry no unique physics meaning — their semantics come
   entirely from their parent. They compete for top-k positions with genuine
   physics nodes.

6. **The enrichment pipeline has a context gap**: Children see raw
   `documentation` (avg 78 chars) from parents, NOT the enriched `description`
   (avg 278 chars, 8.6× expansion). A child enriching under `electrons`
   sees "Quantities related to the electrons" instead of the rich enriched
   description.

7. **BM25 score inflation**: A 0.7 floor on normalized BM25 scores means ANY
   keyword match scores ≥0.7, drowning out vector results that correctly rank
   at 0.82–0.88.

### Key Constraint

**No single text strategy works for all query types:**

- "radiated power" → rank 1 with raw docs, rank 2529 with first_sent
- "electron temperature" → rank 1388 with raw docs, rank 50 with first_sent
- "effective charge" → rank 1 with raw docs, rank 14 with first_sent

Raw docs win when they're specific ("Plasma current. Positive sign means
anti-clockwise...") but fail for sparse docs ("Temperature", "Value", "Data").
LLM first-sentence wins for conceptual queries but fails for technical terms.

This means we cannot simply switch text strategies. We need a **structural
redesign** of what gets described, how it gets described, and what gets
embedded.

---

## Design: Three-Layer Architecture

The solution requires changes at three layers. Each must be validated in
isolation before integration.

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1: ENRICHMENT REDESIGN (what text gets generated)           │
│                                                                    │
│  LLM prompt → generates TWO fields per node:                       │
│    search_text (30-80 chars) — optimized for vector retrieval      │
│    description (100-800 chars) — optimized for human/LLM reading   │
│                                                                    │
│  Hierarchical ordering: parents enriched first                     │
│  Children see enriched parent descriptions, not raw DD docs        │
│  Template enrichment for 7K accessor terminals (no LLM)            │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│  Layer 2: EMBEDDING SELECTION (which nodes get embedded)           │
│                                                                    │
│  Only embed nodes where search_text carries UNIQUE physics         │
│  Embed: physics leaf nodes + concept STRUCTURE containers          │
│  Don't embed: accessor terminals (grid_index, coefficients, time,  │
│    validity, generic value/data/values with sparse docs)           │
│  Accessor terminals found via parent traversal, not vector search  │
│  Estimated corpus: ~13K nodes (from 20K)                           │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│  Layer 3: SEARCH PIPELINE (how results are found & ranked)         │
│                                                                    │
│  Vector search: on search_text embeddings (short, consistent)      │
│  Keyword search: BM25 on documentation, name, description, kw     │
│     Fix: remove 0.7 floor, compress CONTAINS scores               │
│  Path lookup: short-circuit for path-like queries                  │
│  Child traversal: return children of matched nodes via HAS_PARENT  │
│  Score merge: RRF instead of max+bonus                             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Enrichment Redesign

**Goal**: Fix the root cause — generate text that is suitable for both
vector retrieval AND human consumption by separating the two concerns.

### Phase 1A: Add `search_text` field to IMASNode schema

**File**: `imas_codex/schemas/imas_dd.yaml`

Add a new field to IMASNode:

```yaml
search_text:
  description: >-
    Short (30-80 char) text optimized for vector embedding and semantic
    search retrieval. Contains the core physics identity of this node
    without contextual padding. Generated by LLM during enrichment.
    Separate from 'description' which is optimized for reading.
  range: string
```

This field is what gets embedded. The existing `description` field remains
for display. The `embedding_text` field (currently the wrapped 486-char text)
is replaced by `search_text` as the embedding source.

**Rationale**: The benchmarks proved that mixing text styles in the embedding
space destroys coherence (hybrid strategies all scored MRR < 0.05). A
dedicated field ensures ALL nodes have text of consistent style and length.

### Phase 1B: Redesign the LLM enrichment prompt

**File**: `imas_codex/llm/prompts/imas/enrichment.md`

**Current prompt** asks for:
- `description`: "Explain what this quantity measures, its physical
  significance, how it relates to other quantities in the IDS, and its
  role in fusion plasma analysis workflows."
- `keywords`: Up to 5 searchable terms

**Proposed prompt** asks for THREE fields:

1. **`search_text`** (NEW): A concise phrase (30-80 chars) that captures the
   unique physics identity of this node. Must be distinct from sibling nodes.
   Should match what a physicist would type to find this quantity.
   Examples:
   - "1D radial profile of the electron temperature"
   - "Total plasma current (positive = anti-clockwise from above)"
   - "Safety factor profile q = d(toroidal flux)/d(poloidal flux)"
   - "Line-integrated electron density from interferometry"

2. **`description`** (EXISTING, refined): Physics-aware description for display.
   Keep current guidelines but add: "The first sentence must be a self-
   contained summary suitable for search result snippets."

3. **`keywords`** (EXISTING): Up to 5 searchable terms not in the path or docs.

**Key prompt additions**:
- "The `search_text` field will be used for vector embedding. It must be
  SHORT (30-80 chars), DISTINCTIVE (different from sibling nodes), and
  COMPLETE (a physicist should recognize the quantity from this text alone)."
- "Do NOT include units, data types, coordinates, or path context in
  `search_text` — these are handled by other search mechanisms."
- "For STRUCTURE containers, the `search_text` should describe what physics
  concept the container groups (e.g., 'Electron kinetic profiles in the core
  plasma')."

### Phase 1C: Expand template enrichment coverage

**File**: `imas_codex/graph/dd_enrichment.py`

**Current**: Only 17 nodes get template enrichment (validity flags).

**Proposed**: Extend to ~7,029 accessor terminal nodes:

| Pattern | Count | Template search_text |
|---------|-------|---------------------|
| `grid_index` | 797 | "Grid index for {parent_doc}" |
| `grid_subset_index` | 797 | "Grid subset index for {parent_doc}" |
| `coefficients` | 642 | "Interpolation coefficients for {parent_doc}" |
| `*_coefficients` | 857 | "{component} interpolation coefficients" |
| `time` (doc='Time') | 572 | "Timebase for {parent_doc}" |
| `data` (doc='Data') | 523 | "{parent_doc} data array" |
| `value` (doc='Value') | 853 | "{parent_doc}" |
| `values` | 667 | "{parent_doc} values" |
| `validity` | 127 | "Validity flag for {sibling_quantity}" |
| `validity_timed` | 120 | "Time-dependent validity for {sibling_quantity}" |
| `index` (enumeration) | 414 | "Type identifier for {parent_doc}" |
| `label` | 179 | "Label for {parent_doc}" |
| `r/z/phi/x/y` (geometric) | 1562 | "{component} coordinate of {parent_doc}" |

These templates derive `search_text` from parent context. The `description`
field gets a template description (already done for validity). The nodes are
NOT embedded — they are found via parent traversal.

**Implementation**: Add `is_accessor_terminal()` predicate based on
`(name, documentation)` patterns. Called before LLM enrichment to split
batch into LLM vs template paths.

### Phase 1D: Implement hierarchical enrichment ordering

**File**: `imas_codex/graph/dd_graph_ops.py` → `claim_paths_for_enrichment()`

**Current**: Paths claimed via `ORDER BY rand()` — no depth ordering. Children
may be enriched before their parents. Children see only raw `documentation`
from ancestors, not enriched `description`.

**Proposed**: Two-pass enrichment:

**Pass 1 — Shallow first**: Claim paths ordered by depth (ascending).
STRUCTURE/STRUCT_ARRAY nodes at depth ≤ 4 enriched first.

**Pass 2 — Deep nodes**: Leaf nodes enriched with access to parent enriched
descriptions via the ancestor query.

**Modify ancestor context query** in `dd_enrichment.py` (line ~553):
```cypher
MATCH (p)-[:HAS_PARENT*]->(ancestor:IMASNode)
-- Change: return enriched description, not just documentation
RETURN ancestor.id, ancestor.name,
       coalesce(ancestor.description, ancestor.documentation) AS documentation
```

This single change — using `coalesce(description, documentation)` — means
children automatically see enriched parent descriptions once parents are
enriched first.

### Phase 1E: Validate enrichment redesign

Before re-enriching all 20K nodes:

1. Re-enrich a sample of 500 nodes (50 per IDS, 10 IDS) with the new prompt
2. Embed the `search_text` field at dim=256
3. Run the 20-query benchmark
4. Compare against current `doc_only` (best baseline, MRR=0.175)

**Success criteria**: MRR ≥ 0.25 with search_text on 500-node sample
(projected MRR on full corpus accounting for reduced competition).

**Gate**: If MRR < 0.25, iterate on prompt design before proceeding.

---

## Phase 2: Embedding Selection

**Goal**: Reduce the embedded corpus to nodes that carry unique physics
semantics. Accessor terminals found via parent traversal.

**Dependency**: Phase 1C (template enrichment) must define which nodes are
accessor terminals.

### Phase 2A: Define `should_embed()` predicate

**File**: `imas_codex/graph/build_dd.py` or `dd_graph_ops.py`

```python
def should_embed_node(name: str, documentation: str, data_type: str) -> bool:
    """Node carries unique physics semantics worth embedding."""
    # Always embed structures (concept containers)
    if data_type in ("STRUCTURE", "STRUCT_ARRAY"):
        return True
    # Skip accessor terminals — found via parent traversal
    if is_accessor_terminal(name, documentation):
        return False
    # Everything else gets embedded
    return True
```

The `is_accessor_terminal()` function reuses the same patterns from Phase 1C
template enrichment — a single source of truth for what's an accessor.

### Phase 2B: Update embedding claim query

**File**: `imas_codex/graph/dd_graph_ops.py` → `claim_paths_for_embedding()`

Add the predicate to the Cypher WHERE clause or apply it in Python after
claiming (simpler, avoids complex Cypher conditions).

### Phase 2C: Validate corpus reduction

Run the 20-query benchmark with the reduced corpus (~13K nodes) using
`search_text` embeddings. Compare against full corpus.

**Success criteria**: MRR improvement of ≥ 0.03 from corpus filtering
(on top of the search_text improvement from Phase 1E).

---

## Phase 3: Search Pipeline Fixes

**Goal**: Fix keyword search and path lookup independently of vector search.
Each component must perform well alone before hybrid integration.

**Dependency**: None. Can run in parallel with Phase 1 and 2.

### Phase 3A: Remove BM25 score floor

**File**: `imas_codex/tools/graph_search.py` (~line 1954)

Remove `max(raw, 0.7)` floor. BM25 scores should reflect actual relevance.

### Phase 3B: Fix CONTAINS fallback scoring

Compress CONTAINS scores so they don't outrank confident vector matches:

| Match type | Current | Proposed |
|-----------|---------|----------|
| Documentation + leaf | 0.95 | 0.75 |
| Name + leaf | 0.93 | 0.70 |
| Path/ID match | 0.90 | 0.65 |
| Documentation (non-leaf) | 0.88 | 0.60 |
| Else | 0.85 | 0.55 |

### Phase 3C: Path detection short-circuit

Add path detection at the top of `search_imas_paths()`. When query contains
"/" and no spaces, short-circuit to direct lookup → prefix expansion →
RENAMED_TO fallback → normal search.

### Phase 3D: Validate keyword search

Run keyword-only benchmark (no vector) on the standard 20 queries. Measure
P@1, MRR for BM25 alone.

**Success criteria**: Keyword-only MRR ≥ 0.30.

---

## Phase 4: Child Traversal

**Goal**: Return children of matched concept nodes, giving LLM agents a
complete picture of available data.

**Dependency**: Can be developed in parallel. Validated after Phase 1+2.

### Phase 4A: Implement batched child traversal

After score-merge, for all result nodes, run ONE batched Cypher query:

```cypher
UNWIND $concept_ids AS cid
MATCH (child:IMASNode)-[:HAS_PARENT]->(concept:IMASNode {id: cid})
WHERE child.node_category IN ['data', 'error']
RETURN concept.id AS parent_id,
       collect({name: child.name, data_type: child.data_type,
                category: child.node_category,
                doc: left(coalesce(child.documentation, ''), 60)}) AS children
```

Benchmarked at 2.3ms per traversal. No schema change needed — `HAS_PARENT`
already supports reverse traversal.

### Phase 4B: Format children by role

Group children into: data, fit, error, validity, metadata. Collapse groups
with >5 items. Show max 10 children per result.

### Phase 4C: Show siblings for leaf nodes

When a leaf node is matched, show its siblings under the same parent:

```
## core_profiles/profiles_1d/electrons/temperature [FLT_1D]
   Electron temperature profile.
   Siblings: density, pressure, pressure_thermal, collisionality_norm, ...
   Error fields: temperature_error_upper, temperature_error_lower
```

---

## Phase 5: Hybrid Score Integration

**Dependency**: Phase 1E (vector validated) AND Phase 3D (keyword validated).

### Phase 5A: Implement Reciprocal Rank Fusion (RRF)

Replace `max(vector, text) + 0.05` with RRF:

```python
def reciprocal_rank_fusion(vector_ranked, text_ranked, k=60):
    scores = {}
    for rank, r in enumerate(vector_ranked):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(text_ranked):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Phase 5B: End-to-end benchmark

Run full 20-query benchmark comparing vector-only, keyword-only, and
hybrid RRF. Include non-English queries.

**Success criteria**: Hybrid MRR ≥ max(vector, keyword) + 0.05.

---

## Phase 6: Testing and Regression Prevention

### Phase 6A: Embedding quality tests (with embed server guard)
### Phase 6B: BM25 scoring tests (no server needed)
### Phase 6C: Integration tests (full pipeline)

---

## Sequencing and Parallelism

```
  ═══ PARALLEL TRACK A: Enrichment Redesign ═══

  Phase 1A: Add search_text to schema
      │
  Phase 1B: Redesign LLM prompt
      │
  Phase 1C: Expand template enrichment (7K accessor nodes)
      │
  Phase 1D: Hierarchical enrichment ordering
      │
  Phase 1E: Validate on 500-node sample
      │
      ▼ GATE: MRR ≥ 0.25?
      │
  Phase 2A-C: Embedding selection + validation
      │
      ▼ GATE: MRR improvement ≥ 0.03?

  ═══ PARALLEL TRACK B: Search Pipeline ═══

  Phase 3A: Remove BM25 floor      ─┐
  Phase 3B: Fix CONTAINS scores     ├─ All parallel
  Phase 3C: Path detection          ─┘
      │
  Phase 3D: Validate keyword search
      │
      ▼ GATE: Keyword MRR ≥ 0.30?

  ═══ PARALLEL TRACK C: Traversal ═══

  Phase 4A: Batched child traversal
  Phase 4B: Format children by role
  Phase 4C: Show siblings for leaf nodes

  ═══ INTEGRATION (after Track A + B gates pass) ═══

  Phase 5A: RRF hybrid merge
  Phase 5B: End-to-end benchmark
      │
      ▼ GATE: Hybrid MRR ≥ max(vector, keyword) + 0.05?

  Phase 6A-C: Test suite
```

### Agent Assignment

| Phase | Parallel Group | Agent Type |
|-------|---------------|------------|
| 1A (schema) | Group 1 | Develop |
| 1B (prompt) | Group 1 | Develop |
| 1C (templates) | Group 1 | Develop |
| 1D (ordering) | Group 1 | Develop |
| 1E (validate) | After Group 1 | Task |
| 2A-C | After 1E gate | Develop |
| 3A-C | Group 1 (parallel with 1*) | Develop |
| 3D | After 3A-C | Task |
| 4A-C | Group 1 (parallel with 1*) | Develop |
| 5A-B | After gates | Develop |
| 6A-C | Final | Develop |

---

## Key Files Modified

| File | Phases | Changes |
|------|--------|---------|
| `imas_codex/schemas/imas_dd.yaml` | 1A | Add `search_text` field |
| `imas_codex/llm/prompts/imas/enrichment.md` | 1B | Add search_text to prompt |
| `imas_codex/graph/dd_enrichment.py` | 1B, 1C, 1D | Prompt changes, template expansion, hierarchy ordering |
| `imas_codex/graph/dd_graph_ops.py` | 1D, 2B | Enrichment claim ordering, embed filtering |
| `imas_codex/graph/build_dd.py` | 1A, 2A | search_text in generate_embedding_text, should_embed predicate |
| `imas_codex/tools/graph_search.py` | 3A-C, 4A-C, 5A | BM25 fix, path detect, traversal, RRF |
| `tests/tools/test_*.py` | 6A-C | New test files |

## Key Design Decisions

1. **Separate search_text from description**: The fundamental insight from
   benchmarking is that text optimized for reading is harmful for retrieval.
   Two fields serve two purposes. This is the root fix.

2. **Consistent text style in embedding space**: All embedded search_text
   must be 30-80 chars of the same style (physics identity phrase). Mixing
   raw docs with LLM descriptions destroys the embedding space (hybrid
   strategies all scored MRR < 0.05).

3. **Template enrichment for accessor terminals**: 7K nodes get template
   descriptions derived from parent context. No LLM cost. Not embedded.
   Found via parent traversal.

4. **Hierarchical enrichment ordering**: Parents enriched first. One-line
   fix (`coalesce(description, documentation)`) gives children rich parent
   context.

5. **Dimension stays at 256**: Benchmark proved no benefit from 384-1024.

6. **No schema relationship changes**: `HAS_PARENT` reverse traversal at
   2.3ms is sufficient. No `HAS_CHILD` needed.

7. **Gate-driven progression**: Each phase has a measurable success criterion.
   Iteration happens within phases, not across them.
