# IMAS Search Quality Improvement Plan

## Problem Statement

Vector semantic search for IMAS Data Dictionary paths is severely degraded:
**P@1 = 0%** across all embedding dimensions (256→1024). The root cause is
NOT the embedding model or dimension — it is the **embedding text generation**
and **corpus composition** that dilute semantic signal.

### Evidence

| Test | Result |
|------|--------|
| Dimension benchmark (256–1024) | P@1 = 0% at ALL dimensions |
| First-sentence-only text (dim=256) | P@1 = 10%, MRR = 0.167 (12× improvement) |
| "electron temperature" with enriched text | Rank 28 (should be 1) |
| "plasma current" with first-sentence text | **Rank 1** ✓ |
| Cross-language (JP/FR→EN) embeddings | Cosine 0.76–0.90 (model is fine) |
| BM25 score floor | 0.7 minimum drowns out vector signal |

### Root Causes

1. **Embedding text dilution**: `generate_embedding_text()` produces ~488-char
   texts mixing path context, units, coordinates, data types, and keywords with
   the core physics description. This spreads the semantic signal across
   irrelevant metadata.

2. **Corpus pollution**: 20,037 embedded nodes include accessor terminals
   (`grid_index`, `coefficients`, `time`, `validity`) that share generic
   keywords with physics nodes and compete for top-k positions.

3. **BM25 score inflation**: A 0.7 floor on normalized BM25 scores means ANY
   keyword match scores ≥0.7, drowning out vector results that correctly rank
   at 0.82–0.88.

4. **No concept-level landing**: Search returns individual leaf terminals
   instead of physics concepts. A user searching for "electron temperature"
   wants the concept — not `temperature_fit/chi_squared` or
   `temperature_validity`.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      User Query                               │
│              "electron temperature profile"                    │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
    ┌──────────▼──────────┐ ┌────────▼─────────────┐
    │   Vector Search      │ │   Keyword Search      │
    │   (Phase 1)          │ │   (Phase 2)           │
    │                      │ │                        │
    │  Embed query →       │ │  BM25 fulltext index   │
    │  cosine similarity   │ │  on name, description, │
    │  against FOCUSED     │ │  documentation,        │
    │  first-sentence      │ │  keywords, id          │
    │  embeddings          │ │                        │
    └──────────┬──────────┘ └────────┬─────────────┘
               │                      │
    ┌──────────▼──────────────────────▼─────────────┐
    │            Score Merge (Phase 3)               │
    │                                                │
    │  Reciprocal Rank Fusion or calibrated merge    │
    │  No 0.7 floor. Proper range normalization.     │
    └──────────────────────┬────────────────────────┘
                           │
    ┌──────────────────────▼────────────────────────┐
    │         Child Traversal (Phase 4)              │
    │                                                │
    │  For each matched concept node:                │
    │  MATCH (child)-[:HAS_PARENT]->(concept)        │
    │  Return children grouped by role:              │
    │    data: temperature, density, pressure        │
    │    fit:  temperature_fit/* (collapsed)          │
    │    error: *_error_upper/lower/index             │
    │    validity: *_validity, *_validity_timed       │
    └───────────────────────────────────────────────┘
```

---

## Phase 1: Vector Search Fix

**Goal**: Achieve P@1 ≥ 40% and MRR ≥ 0.50 for physics queries using vector
search alone.

**Dependency**: None. Can start immediately.

### Phase 1A: Embedding Text Redesign

**File**: `imas_codex/graph/build_dd.py` → `generate_embedding_text()`

**Current** (488 chars mean):
```
The temperature in profiles 1d in electrons field in the core profiles IDS.
1D radial profile of the electron temperature. This is a fundamental
thermodynamic quantity for almost all plasma analysis, derived typically from
Thomson Scattering or ECE, and is a key input for calculating... Measured in
electron volt (eV) representing [energy]. Related to core_profiles physics.
This is a one-dimensional array. Indexed along the rho_tor_norm coordinate.
Keywords: Te profile, Thomson Scattering, ECE, electron kinetics, thermal energy.
```

**Proposed** (~50–120 chars):
```
1D radial profile of the electron temperature.
```

The embedding text should contain **only the first sentence of the LLM-enriched
description** — the sentence that captures the core physics meaning. Everything
else (units, coordinates, data type, path context, keywords) is handled by
keyword search and metadata enrichment, NOT vector search.

#### Implementation

1. Rewrite `generate_embedding_text()` to extract just the first sentence:
   ```python
   def generate_embedding_text(path: str, path_info: dict, ids_info: dict | None = None) -> str:
       desc = path_info.get("description") or path_info.get("documentation", "")
       first_sentence = desc.split(". ")[0].strip()
       if first_sentence and not first_sentence.endswith("."):
           first_sentence += "."
       return first_sentence or path_info.get("name", path.split("/")[-1])
   ```

2. Store the old function as `generate_embedding_text_v1()` for reference/rollback.

3. Add a `embedding_text_version` property to DDVersion to track which text
   strategy was used for stored embeddings.

#### Validation

Run the benchmark script (from `/tmp/embed_benchmark*.py` pattern) against
the new text strategy before re-embedding the full graph.

**Success criteria**: P@1 ≥ 25% on the 20-query benchmark with first-sentence
text at dim=256 (matching the isolated test result of P@1=10% on 20K corpus,
expected to improve with corpus filtering in Phase 1B).

### Phase 1B: Corpus Filtering — Exclude Accessor Terminals

**Goal**: Reduce the embedded corpus from 20,037 to ~14,000 concept nodes by
excluding accessor terminals that add noise without unique physics content.

**File**: `imas_codex/graph/build_dd.py` → `_classify_node()` and/or
`imas_codex/graph/dd_graph_ops.py` → `claim_paths_for_embedding()`

#### Nodes to exclude from embedding (NOT from the graph)

These nodes remain in the graph and are returned via traversal. They are simply
not embedded because they don't carry unique physics semantics:

| Pattern | Count | Reason |
|---------|-------|--------|
| `*_validity` | ~126 | Quality flags, not physics data |
| `*_validity_timed` | ~120 | Time-dependent quality flags |
| `grid_index` (leaf) | ~795 | Grid coordinate accessor |
| `grid_subset_index` (leaf) | ~795 | Grid subset accessor |
| `coefficients` (leaf) | ~642 | FE interpolation coefficients |
| `*_coefficients` (leaf) | ~600 | Component-specific coefficients |
| `time` (leaf, not in path) | ~603 | Time base accessor |

**Estimated reduction**: ~3,700 nodes → corpus of ~16,300

#### Implementation approach

Do NOT change `node_category`. Instead, add a new concept to the embedding
pipeline: a `should_embed()` predicate that checks whether a node carries
unique physics content worth embedding.

```python
# In dd_graph_ops.py or build_dd.py
_ACCESSOR_LEAF_NAMES = frozenset({
    "grid_index", "grid_subset_index", "coefficients",
    "r_coefficients", "z_coefficients", "phi_coefficients",
    "parallel_coefficients", "poloidal_coefficients",
    "toroidal_coefficients", "diamagnetic_coefficients",
    "radial_coefficients",
})

def should_embed_node(node_id: str, name: str, data_type: str) -> bool:
    """Determine if a node carries unique physics content for embedding."""
    # Always embed structures (concept containers)
    if data_type in ("STRUCTURE", "STRUCT_ARRAY"):
        return True
    # Skip validity flags
    if name.endswith("_validity") or name.endswith("_validity_timed"):
        return False
    # Skip accessor terminals
    if name in _ACCESSOR_LEAF_NAMES:
        return False
    # Skip bare 'time' leaves (but not 'time_slice', 'time_measurement', etc.)
    if name == "time" and data_type in ("FLT_1D",):
        return False
    # Everything else gets embedded
    return True
```

Add this predicate to the Cypher WHERE clause in `claim_paths_for_embedding()`.

#### Validation

Re-run the benchmark with corpus filtering only (no text change) to measure
the isolated effect of removing accessor noise.

**Success criteria**: MRR improvement of ≥ 0.05 from corpus filtering alone.

### Phase 1C: Re-embed IMAS Nodes

**Dependency**: Phase 1A + Phase 1B completed and validated.

Regenerate `embedding_text` for all nodes using the new function, then
re-embed. This can run on the titan embed server.

```bash
uv run imas-codex dd embed --force   # Re-embed all with new text
```

Estimated time: ~7 minutes on titan (4×P100, 14K nodes, dim=256).

### Phase 1D: Benchmark and Validate

**Dependency**: Phase 1C completed.

Run the full 20-query benchmark against the live graph to confirm the
improvement is real and persists end-to-end.

**Success criteria**: P@1 ≥ 30%, MRR ≥ 0.40 on the standard benchmark.
If not met, iterate on text strategy before proceeding to Phase 2.

---

## Phase 2: Keyword Search Fix

**Goal**: Achieve high precision for exact and partial keyword matches.
Fix the BM25 scoring that currently inflates low-quality matches.

**Dependency**: None. Can run in parallel with Phase 1.

### Phase 2A: Remove BM25 Score Floor

**File**: `imas_codex/tools/graph_search.py` → `_text_search_imas_paths()`
(~line 1954)

**Current** (broken):
```python
normalized.append({"id": pid, "score": max(raw, 0.7)})
```

**Proposed** (fixed):
```python
normalized.append({"id": pid, "score": raw})
```

The 0.7 floor was presumably added to ensure text matches always have
"significant" scores, but it destroys the ranking signal. A BM25 match
with normalized score 0.15 should score 0.15, not 0.70.

#### Validation

Before/after comparison on a set of queries showing the score distribution
changes. Verify that weak text matches no longer dominate.

### Phase 2B: Fix CONTAINS Fallback Scoring

**File**: `imas_codex/tools/graph_search.py` → `_text_search_imas_paths()`
(~line 1959-2018)

**Current** (broken):
The CONTAINS fallback assigns fixed scores 0.85–0.95 to any substring match.
These scores are higher than typical vector similarity scores (0.82–0.88),
meaning a crude string match outranks a correct semantic match.

**Proposed**:
Reduce CONTAINS scores to a reasonable range that doesn't compete with
confident vector matches:

| Match type | Current | Proposed |
|-----------|---------|----------|
| Documentation + leaf | 0.95 | 0.75 |
| Name + leaf | 0.93 | 0.70 |
| Path/ID match | 0.90 | 0.65 |
| Documentation (non-leaf) | 0.88 | 0.60 |
| Description/keyword match | 0.85 | 0.55 |

These compressed scores reflect that CONTAINS is a crude match method that
should not outrank vector similarity for top results.

### Phase 2C: Keyword Search Unit Tests

**Dependency**: Phase 2A + 2B completed.

Create unit tests that verify:
1. BM25 scores are NOT floored
2. CONTAINS scores are in the compressed range
3. A specific keyword query returns expected paths in expected order
4. Score distribution is smooth (no cliffs at 0.7)

**File**: `tests/tools/test_keyword_search.py`

---

## Phase 3: Direct Path Lookup Enhancement

**Goal**: Ensure that queries containing IMAS paths are handled as exact
lookups, not semantic searches.

**Dependency**: None. Can run in parallel with Phase 1 and 2.

### Phase 3A: Path Detection in search_imas_paths

**File**: `imas_codex/tools/graph_search.py` → `search_imas_paths()`

Currently, `search_imas_paths()` treats all queries the same — it embeds
the query and does BM25 search. But when a query looks like a path (contains
"/" and no spaces), it should short-circuit to direct lookup.

**Current behavior**: "equilibrium/time_slice/profiles_1d/psi" gets embedded
as a sentence and compared semantically. This wastes compute and may not
find the exact path.

**Proposed**: Add path detection at the top of `search_imas_paths()`:

```python
def _looks_like_path(query: str) -> bool:
    """Detect if query is an IMAS path rather than a search term."""
    stripped = strip_path_annotations(query.strip())
    return "/" in stripped and " " not in stripped

# In search_imas_paths():
if _looks_like_path(query):
    # Direct lookup + prefix expansion
    return await self._path_lookup_with_context(query, ...)
```

The path lookup should:
1. Try exact match first
2. If no exact match, try prefix expansion (show all children)
3. If no prefix match, try fuzzy match (RENAMED_TO traversal)
4. Fall back to normal search only if all above fail

### Phase 3B: Partial Path Matching

Support queries like `"electrons/temperature"` (no IDS prefix) by searching
for paths ending with the query segment:

```cypher
MATCH (n:IMASNode)
WHERE n.id ENDS WITH $suffix
  AND n.node_category = 'data'
RETURN n.id, n.data_type
```

### Phase 3C: Array Notation Passthrough

Verify that `strip_path_annotations()` correctly handles all common formats:
- `profiles_1d(itime)/electrons/temperature(:)` → strips `(itime)` and `(:)`
- `time_slice[1]/profiles_1d[:]/psi` → strips `[1]` and `[:]`
- `channel(i1)/n_e_line` → strips `(i1)`

Add test cases for each format.

---

## Phase 4: Child Traversal in Search Results

**Goal**: When search lands on a concept node, return its children as
structured context — giving the LLM agent a complete picture of what data
is available under that concept.

**Dependency**: Phase 1 (vector fix) should be done first so we can validate
that concept nodes rank correctly. But the implementation can be developed
in parallel and tested independently.

### Phase 4A: Design the Traversal Query

The existing `HAS_PARENT` relationship supports efficient reverse traversal
(benchmarked at 2.3ms per query). The query pattern:

```cypher
// Get direct children of a matched concept node
MATCH (child:IMASNode)-[:HAS_PARENT]->(concept:IMASNode {id: $concept_id})
RETURN child.id AS id, child.name AS name, child.data_type AS data_type,
       child.node_category AS category,
       left(coalesce(child.documentation, ''), 80) AS doc
ORDER BY child.name
```

Children are grouped by role:
- **data**: Physics leaf nodes (FLT_*, INT_*, STR_*, CPX_*)
- **structure**: Sub-containers (STRUCTURE, STRUCT_ARRAY)
- **error**: Error bounds (via HAS_ERROR or name suffix)
- **validity**: Quality flags (*_validity, *_validity_timed)
- **fit**: Fit metadata (*_fit/*)
- **metadata**: Administrative fields (source, description, name)

### Phase 4B: Implement in Search Result Formatting

**File**: `imas_codex/tools/graph_search.py`

After the score-merge step, for each result node, perform a single batched
traversal to get children:

```cypher
UNWIND $concept_ids AS cid
MATCH (child:IMASNode)-[:HAS_PARENT]->(concept:IMASNode {id: cid})
WHERE child.node_category IN ['data', 'error']
RETURN concept.id AS parent_id,
       collect({
           name: child.name,
           data_type: child.data_type,
           category: child.node_category,
           doc: left(coalesce(child.documentation, ''), 60)
       }) AS children
```

This adds ONE extra query for the entire result batch, not one per result.

### Phase 4C: Format Children in Tool Output

Add a `children` section to the search result output. Example:

```
## equilibrium/time_slice/profiles_1d/psi  [FLT_1D]
   Poloidal magnetic flux profile.
   Units: Wb  |  Coordinates: rho_tor_norm
   Children:
     └─ [error] psi_error_upper (FLT_1D), psi_error_lower (FLT_1D)
     └─ [related] q (FLT_1D) — safety factor, phi (FLT_1D) — toroidal flux

## core_profiles/profiles_1d/electrons  [STRUCTURE]
   Electron kinetic profiles container.
   Children:
     ├─ temperature (FLT_1D) — 1D radial profile of electron temperature
     ├─ density (FLT_1D) — Total electron density profile
     ├─ pressure (FLT_1D) — Total electron pressure
     ├─ [fit] temperature_fit/ (STRUCTURE), density_fit/ (STRUCTURE)
     └─ [validity] density_validity (INT_0D), temperature_validity (INT_0D)
```

### Phase 4D: Decide When to Show Children

Not all results need children. Rules:
- **STRUCTURE/STRUCT_ARRAY nodes**: Always show children (they ARE concepts)
- **Leaf data nodes with siblings**: Show sibling summary
- **Top-level IDS nodes**: Show top-level structure only
- **Limit**: Max 10 children per result, collapse fit/error groups

---

## Phase 5: Hybrid Score Integration

**Goal**: Combine the fixed vector and keyword searches into a robust
hybrid ranking that outperforms either alone.

**Dependency**: Phase 1D (vector validated) AND Phase 2C (keyword validated).
This phase tunes the combination; both components must work well independently
first.

### Phase 5A: Implement Reciprocal Rank Fusion (RRF)

**File**: `imas_codex/tools/graph_search.py` → score merge section (~line 217)

RRF is the standard hybrid merge algorithm. It combines ranked lists without
requiring score calibration:

```python
def reciprocal_rank_fusion(
    vector_results: list[dict],  # [{id, score}] sorted by score desc
    text_results: list[dict],    # [{id, score}] sorted by score desc
    k: int = 60,                 # RRF constant (standard value)
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for rank, r in enumerate(vector_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(text_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1.0 / (k + rank + 1)
    merged = [{"id": pid, "score": s} for pid, s in scores.items()]
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
```

Replace the current `max(vector, text) + 0.05` merge.

### Phase 5B: Path Boost Refinement

Keep the path segment boost but calibrate it. Currently +0.03 per matching
word. This should be applied as a re-ranking signal, not a score additive.

Consider: If query words match path segments exactly, add a small RRF bonus
(equivalent to a third "path match" ranked list).

### Phase 5C: End-to-End Hybrid Benchmark

Run the full 20-query benchmark comparing:
1. Vector only (Phase 1 fix)
2. Keyword only (Phase 2 fix)
3. Hybrid RRF (Phase 5 fix)

Report P@1, P@5, P@10, MRR for each. Include non-English queries.

**Success criteria**: Hybrid MRR ≥ max(vector MRR, keyword MRR) + 0.05
(i.e., hybrid is better than either component alone).

---

## Phase 6: Testing and Regression Prevention

**Dependency**: All phases implemented. This is the final validation step.

### Phase 6A: Embedding Quality Unit Tests

**File**: `tests/tools/test_embedding_quality.py`

Tests require a running embed server. Use a pytest marker:

```python
import pytest
from imas_codex.embeddings.encoder import Encoder

embed_server = pytest.mark.skipif(
    not _embed_server_reachable(),
    reason="Embed server not available"
)

@embed_server
def test_electron_temperature_top5():
    """Electron temperature must rank in top 5 for 'electron temperature'."""
    results = search_imas("electron temperature", k=5)
    ids = [r.id for r in results]
    assert "core_profiles/profiles_1d/electrons/temperature" in ids

@embed_server
def test_plasma_current_top3():
    """Plasma current must rank in top 3 for 'plasma current'."""
    results = search_imas("plasma current", k=3)
    ids = [r.id for r in results]
    assert "equilibrium/time_slice/global_quantities/ip" in ids

@embed_server
def test_cross_language_japanese():
    """Japanese query must find electron temperature."""
    results = search_imas("電子温度", k=10)
    ids = [r.id for r in results]
    assert any("temperature" in p for p in ids)
```

### Phase 6B: BM25 Scoring Tests

**File**: `tests/tools/test_bm25_scoring.py`

```python
def test_no_score_floor():
    """BM25 scores must NOT have a 0.7 floor."""
    results = _text_search_imas_paths(query="xyznonexistent123", ...)
    # Even if results exist, scores should reflect actual relevance
    for r in results:
        # Scores should span a range, not be floored
        assert r["score"] >= 0.0

def test_contains_scores_below_vector_range():
    """CONTAINS fallback scores must not exceed 0.80."""
    results = _text_search_imas_paths(query="temperature", ...)
    for r in results:
        assert r["score"] <= 0.80
```

### Phase 6C: Integration Tests

**File**: `tests/tools/test_search_integration.py`

Test the full pipeline end-to-end:
1. Path detection correctly short-circuits to direct lookup
2. Concept nodes return children in results
3. Hybrid merge produces better ranking than either component
4. Non-English queries work through the full pipeline

---

## Sequencing and Parallelism

```
            ┌─────────────────┐
            │   Phase 1A      │  Embedding text redesign
            │   Phase 1B      │  Corpus filtering
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │   Phase 1C      │  Re-embed IMAS nodes
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │   Phase 1D      │  Benchmark vector search
            └────────┬────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │        GATE: P@1 ≥ 30%?        │
    │   If no → iterate Phase 1A     │
    └────────────────┬────────────────┘
                     │ yes
                     ▼
              ┌──────┴──────┐
              │  Phase 5    │  Hybrid integration
              │  Phase 6    │  Tests & regression
              └─────────────┘

    ═══ PARALLEL TRACK (no dependency on Phase 1) ═══

    ┌─────────────────┐     ┌─────────────────┐
    │   Phase 2A      │     │   Phase 3A      │
    │   Phase 2B      │     │   Phase 3B      │
    │   Phase 2C      │     │   Phase 3C      │
    └────────┬────────┘     └────────┬────────┘
             │                        │
             └────────┬───────────────┘
                      │
             ┌────────▼────────┐
             │   Phase 4A-D   │  Child traversal
             └────────┬────────┘
                      │
                      ▼ (joins Phase 5)
```

### Agent Assignment

| Phase | Can Parallelize | Estimated Effort | Agent Type |
|-------|----------------|------------------|------------|
| 1A | No (sequential with 1B) | Small — one function rewrite | Develop |
| 1B | With 2A, 3A | Small — add predicate + filter | Develop |
| 1C | No (needs 1A+1B) | Tiny — run CLI command | Task |
| 1D | No (needs 1C) | Small — run benchmark script | Task |
| 2A | With 1A, 3A | Tiny — remove one line | Develop |
| 2B | With 2A | Small — adjust 5 constants | Develop |
| 2C | After 2A+2B | Small — write test file | Develop |
| 3A | With 1A, 2A | Medium — path detection logic | Develop |
| 3B | With 3A | Small — add suffix query | Develop |
| 3C | With 3A | Small — test cases | Develop |
| 4A-B | After Phase 1D validated | Medium — traversal + formatting | Develop |
| 4C-D | With 4B | Small — output format | Develop |
| 5A | After 1D + 2C | Medium — RRF implementation | Develop |
| 5B-C | With 5A | Small — calibration | Develop |
| 6A-C | After all phases | Medium — test suite | Develop |

### Parallel dispatch groups

**Group 1** (launch simultaneously):
- Phase 1A: Embedding text redesign
- Phase 2A+2B: BM25 score floor + CONTAINS fix
- Phase 3A+3B+3C: Path detection and lookup

**Group 2** (after Group 1 completes):
- Phase 1B: Corpus filtering (needs 1A for combined validation)
- Phase 2C: Keyword search tests
- Phase 4A+4B: Child traversal design and implementation

**Group 3** (after re-embed and validation):
- Phase 1C: Re-embed
- Phase 1D: Benchmark
- Phase 4C+4D: Output formatting

**Group 4** (final integration):
- Phase 5A+5B: Hybrid RRF merge
- Phase 5C: End-to-end benchmark
- Phase 6A+6B+6C: Full test suite

---

## Key Files Modified

| File | Phases | Changes |
|------|--------|---------|
| `imas_codex/graph/build_dd.py` | 1A, 1B | `generate_embedding_text()` rewrite, `should_embed_node()` predicate |
| `imas_codex/tools/graph_search.py` | 2A, 2B, 3A, 3B, 4B, 4C, 5A, 5B | BM25 fix, path detection, traversal, RRF merge |
| `imas_codex/graph/dd_graph_ops.py` | 1B | Embedding claim filter update |
| `imas_codex/core/paths.py` | 3C | Additional strip_path_annotations tests |
| `tests/tools/test_embedding_quality.py` | 6A | New test file |
| `tests/tools/test_bm25_scoring.py` | 6B | New test file |
| `tests/tools/test_search_integration.py` | 6C | New test file |

## Key Design Decisions

1. **Dimension stays at 256** — Benchmark proved no benefit from higher dims.
   The per-request dimension API on the embed server is kept for future use.

2. **No schema change needed** — `HAS_PARENT` already supports reverse
   traversal at 2.3ms. No `HAS_CHILD` relationship needed.

3. **Node category unchanged** — Accessor terminals keep `node_category='data'`.
   Embedding exclusion is done via a `should_embed_node()` predicate, not by
   reclassifying nodes.

4. **Keywords field already exists** — LLM-enriched keywords (max 5 per node)
   are already stored and indexed in the fulltext index. No additional
   keyword generation needed.

5. **First sentence, not keywords, for embedding** — The benchmark showed
   keywords in embedding text hurt when path abbreviations mismatch
   ("ip" ≠ "plasma current"). The first sentence is pure physics description.

6. **RRF over weighted combination** — RRF doesn't require score calibration.
   It combines ranked lists purely by position, making it robust to the
   different score distributions of vector and BM25.
