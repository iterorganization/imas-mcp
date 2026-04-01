# IMAS DD Search Quality Fix — Plan

## Problem Statement

Vector search for IMAS paths is broken: P@1=0% across all dimensions (256→1024). Root cause confirmed through 4 benchmark rounds (80 experiments): the embedding text is too long, and the corpus is polluted by ~9,500 accessor terminal nodes.

## Evidence Summary

| Benchmark | Finding |
|-----------|---------|
| Dim sweep (256-1024) | P@1=0% at ALL dimensions — not a capacity issue |
| 6 text strategies × 20K | Raw DD docs (82 chars): MRR=0.175; LLM enriched (486 chars): MRR=0.016 |
| 7 hybrid strategies | ALL hybrids WORSE than either pure strategy — mixing destroys embedding space |
| 4 corpus sizes | Filtering alone: MRR 0.066→0.092 — marginal without text fix |

## Root Causes (confirmed)

1. **Embedding text dilution**: `generate_embedding_text()` produces 486-char texts (8 sentences: path context + IDS desc + LLM description + units + domain + data type + coordinates + keywords). Shorter = dramatically better: 82-char raw docs give 10× better MRR.

2. **Corpus pollution**: 9,467 accessor terminals (47% of embedded corpus) dilute the vector space:
   - `value` (917), `time` (695), `data` (535), `values` (667), `coefficients` (644)
   - `grid_index` (797), `grid_subset_index` (797), `index` (458), `label` (179)
   - `r` (429), `z` (586), `phi` (309), `x` (159), `y` (159)
   - `*_coefficients` (857), geometric components (523)
   - Total: 9,467 nodes with generic docs (65.5% of all true leaf nodes)

3. **Boilerplate filter bug**: `BOILERPLATE_PATTERNS` regex `_validity$` catches `density_validity` (17 nodes) but misses standalone `validity` (127) and `validity_timed` (127) = 254 nodes LLM-enriched instead of template-enriched.

4. **Template enrichment gap**: Only 17 of ~9,500 accessor terminals are template-enriched. The rest get expensive LLM descriptions that dilute the physics signal.

5. **BM25 score floor**: Line ~1954 in `graph_search.py`: `max(raw, 0.7)` means ANY text match scores ≥0.7, drowning vector results (0.82-0.88).

## Graph Structure Analysis

```
IMASNode (61,366 total)
├── error     31,281  (NOT embedded — correct)
├── data      20,037  (embedded — too many)
│   ├── STRUCTURE/STRUCT_ARRAY   4,764  (concept containers)
│   │   ├── 913 have 'value' child
│   │   ├── 604 have 'time' child  
│   │   └── avg 4.1 children each
│   └── Leaf data nodes         15,273
│       ├── Unique names (≤5 occ):    1,450  ← true physics concepts
│       ├── Borderline (6-19 occ):    1,200  ← mixed
│       └── Frequent (≥20 occ):       9,989  ← 65.5% are accessor terminals
└── metadata  10,048  (NOT embedded — correct)
```

Key relationship: `HAS_PARENT` (60,334 edges, child→parent). Reverse traversal to get children: 2.3ms. No HAS_CHILD needed.

## Architecture: Concept-Node Search with Child Traversal

### Concept Node Definition

A concept node is the **landing page** for a physics quantity. Users search for concepts; accessor children are surfaced via traversal.

**Two types of concept nodes:**
1. **Container concepts** (STRUCTURE/STRUCT_ARRAY with data children): `summary/global_quantities/q_95` → children: `value`, `source`, `value_error_upper`
2. **Direct leaf concepts** (named physics quantities with no children): `equilibrium/time_slice/profiles_1d/psi`, `core_profiles/profiles_1d/electrons/density_thermal`

**NOT concept nodes** (accessor terminals — meaning comes from parent):
- Generic data: `value` (917), `data` (535), `values` (667)
- Time bases: `time` (695)
- Grid refs: `grid_index` (797), `grid_subset_index` (797), `index` (458)
- Interpolation: `coefficients` (644), `*_coefficients` (857)
- Geometric vectors: `r`, `z`, `phi`, `x`, `y` + directional components (2,258 total)
- Flags: `validity` (127), `validity_timed` (127), `*_flag` (59)
- Labels: `label` (179)

### Embedding Text: Raw Documentation Only

Replace the 486-char `generate_embedding_text()` output with the raw `documentation` field (~82 chars avg). This is the single highest-impact change — benchmark showed 10× MRR improvement.

**Why raw docs, not first-sentence-of-description?** 
- Raw docs are written by domain experts in consistent, terse style
- LLM descriptions are 3.4× longer with variable style, hurting embedding consistency
- Raw docs already contain the key physics terms that matter for retrieval
- BM25/fulltext already searches `description` and `keywords` — no need to duplicate in embeddings

### Corpus: ~10,500 Concept Nodes (47% Reduction)

| Keep | Count | Criteria |
|------|-------|----------|
| STRUCTURE/STRUCT_ARRAY parents with children | 4,764 | Concept containers |
| Unique-name leaves (≤5 occurrences) | 1,450 | True physics concepts |
| Borderline physics leaves (6-19 occ, docs > 30 chars) | ~800 | `b0`, `velocity_tor`, `momentum_phi`, etc. |
| Select frequent leaves with meaningful docs (>30 chars) | ~3,500 | Physics quantities that happen to be common |
| **Total concept corpus** | **~10,500** | |

| Remove | Count | Criteria |
|--------|-------|----------|
| Accessor terminal names (predefined list) | 8,610 | `value`, `time`, `data`, `coefficients`, etc. |
| `*_coefficients` pattern | 857 | Interpolation coefficients |
| **Total removed** | **~9,467** | Template-enriched, not embedded |

### Search Result Enrichment

When a concept node is found, return its children grouped by role:

```
summary/global_quantities/q_95  (STRUCTURE)
  "Safety factor at the 95% poloidal flux surface"
  ├── value (FLT_1D) — Scalar value
  ├── source (STR_0D) — Data source identifier
  └── [error: value_error_upper, value_error_lower, value_error_index]
```

Children retrieved via: `MATCH (child)-[:HAS_PARENT]->(concept) RETURN child`

### BM25 Scoring Fix

1. **Remove 0.7 floor** on normalized BM25 scores (line ~1954)
2. **Compress CONTAINS fallback** scores from 0.85-0.95 → 0.60-0.75
3. **Cap total score** at 1.0
4. **Medium-term**: RRF (Reciprocal Rank Fusion) instead of score merging

## Implementation Phases

### Phase 1: Expand Template Enrichment (fixes enrichment gap)

**Files:** `dd_enrichment.py`

1A. Fix `BOILERPLATE_PATTERNS` to catch standalone `validity`/`validity_timed`:
```python
BOILERPLATE_PATTERNS = [
    re.compile(r"_error_index$"),
    re.compile(r"_error_lower$"),
    re.compile(r"_error_upper$"),
    re.compile(r"_validity$"),
    re.compile(r"_validity_timed$"),
    re.compile(r"^validity$"),        # ADD
    re.compile(r"^validity_timed$"),  # ADD
]
```

1B. Add `ACCESSOR_TERMINAL_NAMES` set for template enrichment:
```python
ACCESSOR_TERMINAL_NAMES = {
    'value', 'time', 'data', 'values', 'coefficients',
    'grid_index', 'grid_subset_index', 'index', 'label',
    'validity', 'validity_timed',
    'measured', 'reconstructed', 'chi_squared',
    'neighbours', 'measure', 'space', 'nodes',
    'geometry', 'geometry_2d', 'dim1', 'dim2', 'closed',
}
ACCESSOR_TERMINAL_SUFFIXES = ('_coefficients',)
```

1C. Expand `is_boilerplate_path()` to use new sets.

1D. Expand `generate_template_description()` with templates for accessor terminals:
- `value`: "Data value for {parent_name}. {parent_doc_first_sentence}"
- `time`: "Time base for {parent_name}."
- `data`: "Data array for {parent_name}. {parent_doc_first_sentence}"
- `coefficients`/`*_coefficients`: "Interpolation coefficients for {component} of {parent_name}."
- `grid_index`/`grid_subset_index`: "Grid reference index for {parent_name}."
- `r`/`z`/`phi`/`x`/`y`: "Geometric {component} coordinate of {parent_name}."
- `index`: "Integer identifier for {parent_name}."
- `label`: "String label for {parent_name}."

**Gate: ≥9,000 nodes template-enriched (up from 17)**

### Phase 2: Embedding Text Redesign (fixes text dilution)

**Files:** `build_dd.py`

2A. Rewrite `generate_embedding_text()` to return raw `documentation` only:
```python
def generate_embedding_text(path, path_info, ids_info=None):
    doc = path_info.get("documentation", "")
    return doc.strip() if doc else ""
```

2B. Add `is_concept_node()` predicate using accessor terminal classification from Phase 1.

2C. Update embedding claim query in `dd_graph_ops.py` to skip accessor terminals.

**Gate: Re-embed 500-node sample, measure MRR ≥ 0.20 (up from 0.016)**

### Phase 3: BM25 Scoring Fix (parallel with Phase 1)

**Files:** `graph_search.py`

3A. Remove `max(raw, 0.7)` floor on BM25 scores (line ~1954).

3B. Compress CONTAINS fallback scores: exact_name 0.95→0.80, partial 0.90→0.70, doc_contains 0.85→0.60.

3C. Add path-detection short-circuit: if query looks like a path (contains `/`), skip vector search entirely.

**Gate: Keyword-only MRR ≥ 0.30 for path-like queries**

### Phase 4: Child Traversal in Search Results (parallel with Phase 1)

**Files:** `graph_search.py`

4A. After vector+text search finds concept nodes, batch-fetch children:
```cypher
MATCH (child:IMASNode)-[:HAS_PARENT]->(concept)
WHERE concept.id IN $concept_ids AND child.node_category = 'data'
RETURN concept.id AS parent_id, collect({
    name: child.name, id: child.id, data_type: child.data_type,
    documentation: child.documentation
}) AS children
```

4B. Group children by role: data (value/data/values), time, fit (measured/reconstructed/chi_squared), error, validity, coordinates (r/z/phi), other.

4C. Format in tool output as nested tree under each result.

### Phase 5: Re-embed and Validate

5A. Re-run enrichment pipeline with expanded template patterns → ~9,500 template-enriched.

5B. Re-embed only concept nodes (~10,500) with raw documentation text.

5C. Full benchmark: target P@1 ≥ 15%, MRR ≥ 0.20 for vector search alone.

5D. End-to-end hybrid search benchmark: target MRR ≥ max(vector, keyword) + 0.05.

### Phase 6: Tests

6A. Embedding quality unit tests (with embed server guard).

6B. BM25 scoring tests (no server needed).

6C. Child traversal integration tests.

## Storage Impact

Stay at dim=256. Corpus reduction saves ~37 MB (9,467 × 256 × 4 bytes × 4 = ~37 MB freed).

## Parallel Dispatch Plan

Group 1 (independent, launch together):
- Phase 1 (template enrichment expansion)
- Phase 3 (BM25 fix)
- Phase 4 (child traversal)

Group 2 (after Phase 1 gate):
- Phase 2 (embedding text redesign)

Group 3 (after Phase 2 + 3 gates):
- Phase 5 (re-embed and validate)

Group 4 (after Phase 5 validates):
- Phase 6 (tests)
