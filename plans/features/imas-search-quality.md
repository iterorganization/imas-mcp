# IMAS DD Search Quality — Comprehensive Fix Plan

## Problem Statement

Vector/semantic search for IMAS DD paths is broken: **P@1=0% across all embedding dimensions** (256→1024). Confirmed through 4 benchmark rounds (80 experiments). Root causes are embedding text dilution and corpus pollution, compounded by BM25 scoring distortion and missing child traversal.

## Evidence Summary

| Benchmark | Finding |
|-----------|---------|
| Dimension sweep (256→1024) | P@1=0% at ALL dimensions — not a model capacity issue |
| 6 text strategies × 20K corpus | Raw DD docs (82 chars avg): **MRR=0.175**; LLM enriched (486 chars avg): **MRR=0.016** |
| 7 hybrid text strategies | ALL hybrids WORSE than either pure strategy — mixing destroys the embedding space |
| 4 corpus size experiments | Filtering alone: MRR 0.066→0.092 — marginal without fixing the text |

**Key insight**: Shorter, more focused text = dramatically better vector search. The 486-char LLM descriptions are 10× worse than 82-char raw docs. But raw docs are often poor ("Value", "Time", "Data"). The fix is **concise, disambiguating LLM descriptions** (~100-150 chars) that improve on both.

## Root Causes (confirmed with evidence)

### 1. Embedding Text Dilution (PRIMARY)
`generate_embedding_text()` in `build_dd.py:143-280` concatenates **8 components** into a single prose paragraph averaging 486 chars:

1. Path identity with parent context ("The psi in time_slice profiles_1d field in the equilibrium IDS")
2. IDS-level context ("The equilibrium IDS contains...")
3. LLM description (avg 263 chars — verbose, explanatory)
4. Unit expansion ("Measured in electron volt (eV) representing [energy]")
5. Physics domain context ("Related to plasma physics")
6. Data type in natural language ("This is a one-dimensional array")
7. Coordinate system ("Indexed along the rho_tor and theta coordinates")
8. Keywords ("Keywords: poloidal flux, equilibrium, profiles")

**Components 1, 2, 4-8 are metadata noise.** They dilute the physics signal. The LLM description (component 3) is the only semantic content but is itself too verbose.

**Evidence**: Current LLM descriptions average 263 chars with explanatory style:
```
# ece/channel/t_e (doc: "Electron temperature", 20 chars)
DESC (232 chars): "Local electron temperature measured by a specific channel of an
Electron Cyclotron Emission (ECE) radiometer. This value is derived from the
blackbody radiation intensity at the local electron cyclotron frequency (or
its harmonics)."
```
This should be: `"Electron temperature from ECE diagnostic per channel."` (52 chars)

### 2. Corpus Pollution (47% of embedded nodes)
9,467+ accessor terminal nodes pollute the vector space. These are structurally repetitive leaves whose meaning comes entirely from their parent:

| Name Pattern | Count | Example Doc |
|---|---|---|
| `value` | 905 | "Value" |
| `grid_subset_index` | 797 | (varies) |
| `grid_index` | 797 | (varies) |
| `time` | 694 | "Time" or "Generic time" |
| `values` | 667 | (varies) |
| `coefficients` | 644 | (varies) |
| `z` | 571 | (varies) |
| `data` | 535 | "Data" |
| `index` | 458 | (varies) |
| `r` | 414 | (varies) |
| `phi` | 300 | (varies) |
| `*_coefficients` | 857 | (varies) |
| `label`, `x`, `y`, directional | ~1,200+ | (varies) |
| **Total accessor terminals** | **~9,500** | 62% of all leaf nodes |

These nodes all receive verbose, unique LLM descriptions (avg 263 chars each) that cost money but hurt search because 917 `value` nodes create 917 different embedding vectors all competing with each other.

### 3. Template Enrichment Gap
Only **17 of ~9,500 accessor terminals** are template-enriched. The 5 `BOILERPLATE_PATTERNS` regexes only catch `*_error_index`, `*_error_lower`, `*_error_upper`, `*_validity`, `*_validity_timed` — and even miss standalone `validity` (127 nodes) and `validity_timed` (120 nodes).

### 4. Enrichment Context Bug
`gather_path_context()` at `dd_enrichment.py:401-412` queries ancestor chain using `node.documentation` (raw DD docs) instead of `node.description` (enriched text). When a parent has been enriched with a good description, that context is lost for child enrichment. This means children can't inherit the disambiguating context from their already-enriched parents.

### 5. BM25 Score Distortion
In `graph_search.py`:
- **Score floor** at line ~1905: `max(raw, 0.7)` means ANY BM25 match scores ≥0.7
- **CONTAINS fallback** scores range 0.85-0.95 (5-tier cascade)
- **Score merging** at line ~208: `max(vector, text) + 0.05` for co-occurrence
- Net effect: text search scores (0.7-0.95) drown vector results (0.82-0.88), and the 0.05 boost is too small to differentiate

## Current Graph State (live census)

```
IMASNode (61,366 total)
├── error     31,281  (NOT embedded — correct)
├── data      20,037  (ALL embedded — too many)
│   ├── Enrichment: 20,020 LLM (99.9%), 17 template (0.08%)
│   ├── Description: min=118, avg=263, max=804 chars
│   ├── Embedding text: min=233, avg=486, max=951 chars
│   ├── Documentation: min=4, avg=82, max=1262 chars
│   ├── Keywords: min=2, avg=4.6, max=5 per node (100% coverage)
│   │
│   ├── STRUCTURE/STRUCT_ARRAY    4,762  (concept containers)
│   │   └── avg 4.1 children each, max 53
│   │
│   └── True leaf nodes          15,247
│       ├── Unique names (≤5 occ):    927 names →  1,445 nodes  ← physics concepts
│       ├── Borderline (6-19 occ):    118 names →  1,179 nodes  ← mixed
│       └── Frequent (≥20 occ):       114 names → 12,623 nodes  ← mostly accessor terminals
│
│   Non-leaf data nodes: 4,790 (≈ STRUCTURE + STRUCT_ARRAY + root nodes)
│
└── metadata  10,048  (NOT embedded — correct)

Relationships: 60,334 HAS_PARENT edges. 584 root data nodes (no parent).
```

## Architecture: Concise Descriptions + Concept-Node Embedding + Child Traversal

### Design Principle

The fix has three layers, each addressing a different root cause:

1. **Better text** → concise LLM descriptions (fix dilution)
2. **Smaller corpus** → embed only concept nodes (fix pollution)
3. **Better scoring** → fix BM25 distortion + add child traversal (fix result quality)

### Layer 1: Concise LLM Descriptions

**Goal**: Replace verbose 263-char descriptions with focused 1-2 sentence descriptions (~80-150 chars) that name the specific physical quantity and what distinguishes this node from similar nodes elsewhere in the DD.

**Why not raw docs?** Raw docs are better for embedding (82 chars → MRR=0.175) but many are terrible: 2,671 nodes have docs ≤10 chars ("Value", "Time", "Data"). A concise LLM description can match the brevity of raw docs while adding the disambiguating context that makes vector search precise.

**Target quality** (examples from stashed prompt redesign):
```
core_profiles/profiles_1d/electrons/temperature →
  "Electron temperature radial profile in the core plasma." (56 chars)

thomson_scattering/channel/t_e →
  "Electron temperature measured by Thomson scattering per channel." (64 chars)

ece/channel/t_e →
  "Electron temperature from ECE diagnostic per frequency channel." (63 chars)

summary/local/pedestal/t_e →
  "Electron temperature at the pedestal, a summary scalar value." (61 chars)

equilibrium/time_slice/profiles_1d/psi →
  "Poloidal magnetic flux radial profile from equilibrium reconstruction." (70 chars)
```

**Current quality** (what we're replacing):
```
ece/channel/t_e →
  "Local electron temperature measured by a specific channel of an
  Electron Cyclotron Emission (ECE) radiometer. This value is derived
  from the blackbody radiation intensity at the local electron
  cyclotron frequency (or its harmonics)." (232 chars)
```

### Layer 2: Concept Node Classification

A **concept node** is the landing page for a physics quantity — what a user searches for. Accessor children are surfaced via graph traversal after the concept is found.

**Two types of concept nodes:**

| Type | Count | Example |
|------|-------|---------|
| Container concepts (STRUCTURE/STRUCT_ARRAY with data children) | ~4,762 | `summary/global_quantities/q_95` → children: value, source |
| Direct leaf concepts (named physics quantities, no children) | ~5,600 | `equilibrium/time_slice/profiles_1d/psi` |
| **Total concept corpus** | **~10,400** | |

**NOT concept nodes** (accessor terminals — template-enriched, not embedded):

| Pattern | Count | Why Remove |
|---|---|---|
| `value`, `data`, `values` | 2,107 | Generic data containers, meaning from parent |
| `time` | 694 | Time bases, always contextual |
| `grid_index`, `grid_subset_index`, `index` | 2,052 | Grid references |
| `coefficients`, `*_coefficients` | 1,501 | Interpolation coefficients |
| `r`, `z`, `phi`, `x`, `y` | 1,944+ | Geometric components |
| `label` | 178 | String labels |
| `validity`, `validity_timed` | 247 | Already partial boilerplate |
| directional (`parallel`, `poloidal`, `radial`, `toroidal`, `diamagnetic`) | 655 | Vector components (meaning from parent) |
| `*_n` (z_n, r_n, etc.) | ~500 | Normalized components |
| Other accessor names | ~600 | `measured`, `reconstructed`, `chi_squared`, `a`, etc. |
| **Total accessor terminals** | **~9,600** | |

**Classification algorithm**: A leaf node is an accessor terminal if:
1. Its `name` is in `ACCESSOR_TERMINAL_NAMES` set (predefined ~40 names), OR
2. Its `name` ends with a suffix in `ACCESSOR_TERMINAL_SUFFIXES` (`_coefficients`, `_n`), OR
3. Its `name` is a single letter (`a`-`z`) that occurs ≥20 times

Concept nodes = all data nodes that are NOT accessor terminals. This includes:
- All STRUCTURE/STRUCT_ARRAY nodes (they are parents by definition)
- All non-accessor leaf nodes (unique physics quantities)
- Borderline leaves like `density`, `pressure`, `psi`, `temperature` (real physics even if common)

### Layer 3: Embedding Text = Concise Description Directly

Replace the 8-component `generate_embedding_text()` with the concise `description` field directly:

```python
def generate_embedding_text(path: str, path_info: dict, ids_info: dict | None = None) -> str:
    desc = path_info.get("description", "")
    if desc:
        return desc.strip()
    doc = path_info.get("documentation", "")
    return doc.strip() if doc else ""
```

**Why this works**: With the redesigned prompt producing 80-150 char descriptions, we get:
- **Shorter than current** (150 vs 486) → proven 10× MRR improvement from benchmarks
- **Better than raw docs** (disambiguating context vs "Value") → handles the 2,671 poor-doc nodes
- **Consistent style** (LLM-normalized vs mixed DD writing) → tighter embedding clusters

### Layer 4: Search Result Child Traversal

When a concept node is found by search, retrieve and display its children grouped by role:

```
summary/global_quantities/q_95  (STRUCTURE)
  "Safety factor at the 95% poloidal flux surface"
  ├── Data: value (FLT_1D)
  ├── Source: source (STR_0D)
  ├── Time: (none — inherits parent timebase)
  └── Error: value_error_upper, value_error_lower, value_error_index
```

Uses existing `HAS_PARENT` relationships (60,334 edges, reverse traversal ~2.3ms). No schema changes needed.

## Implementation Phases

### Phase 1: Prompt Redesign and Validation

**Goal**: Redesign the enrichment prompt to produce concise, disambiguating descriptions. Validate that the new descriptions are better for embedding before proceeding.

**Files**: `imas_codex/llm/prompts/imas/enrichment.md`, `imas_codex/graph/dd_enrichment.py`

#### 1A. Rewrite enrichment prompt

The stashed prompt redesign is the starting point. Key changes:

**System prompt** (`enrichment.md`):
- Change task from "Generate rich, physics-aware descriptions" to "Write a concise description (1-2 sentences, under 150 characters)"
- Add explicit good/bad examples showing the target style
- Remove instructions that encourage verbosity ("Write as much as the available context justifies")
- Keep COCOS awareness (brief mention only)
- Keep hierarchy context instructions (use ancestors to disambiguate)

**Pydantic model** (`dd_enrichment.py:45-55`):
- Change `description` field docstring from "Physics-aware description... Explain what the quantity measures, its physical significance, how it relates..." to "Concise description (1-2 sentences, under 150 characters) that names the physical quantity and what distinguishes this node."

#### 1B. Fix ancestor context to use enriched descriptions

In `gather_path_context()` at `dd_enrichment.py:401-412`, change:
```cypher
-- BEFORE (line 410):
documentation: node.documentation

-- AFTER:
documentation: coalesce(node.description, node.documentation)
```

This ensures children receive the enriched descriptions from their already-processed parents.

#### 1C. Add hierarchical enrichment ordering

In `claim_paths_for_enrichment()` at `dd_graph_ops.py:40-98`, change the claim query to process shallow nodes before deep nodes:

```cypher
-- BEFORE: ORDER BY rand()
-- AFTER: ORDER BY size(split(p.id, '/')) ASC, rand()
```

This ensures parents are enriched before their children, so the ancestor context query (1B) returns enriched descriptions.

**Note**: This changes the claim order but doesn't break deadlock avoidance — `rand()` is still used as tiebreaker within the same depth level.

#### 1D. Validate with A/B benchmark

Before full re-enrichment:
1. Select 200 representative concept nodes (mix of containers and leaf concepts across IDSs)
2. Re-enrich with the new prompt → get new concise descriptions
3. Embed with the new text (description only, no 8-component concatenation)
4. Measure MRR against the same benchmark queries used in prior experiments
5. Compare: new concise descriptions vs raw docs vs current verbose descriptions

**Gate**: New concise descriptions must achieve **MRR ≥ 0.20** (current: 0.016 with verbose, 0.175 with raw docs). Target: **MRR ≥ 0.25** (better than raw docs thanks to disambiguation).

**Agent instructions**: This phase requires sequential execution — prompt changes must be validated before proceeding. A single agent should: apply the stashed prompt changes, select 200 sample nodes, call the LLM for re-enrichment, embed the results, and run the benchmark. Report numbers before proceeding.

---

### Phase 2: Template Enrichment Expansion

**Goal**: Template-enrich all accessor terminal nodes so they don't consume LLM budget and don't pollute the embedding corpus.

**Files**: `imas_codex/graph/dd_enrichment.py`

**Depends on**: Nothing (can run in parallel with Phase 1 validation)

#### 2A. Fix boilerplate patterns

Add standalone validity/validity_timed to `BOILERPLATE_PATTERNS` at line ~86:
```python
BOILERPLATE_PATTERNS = [
    re.compile(r"_error_index$"),
    re.compile(r"_error_lower$"),
    re.compile(r"_error_upper$"),
    re.compile(r"_validity$"),
    re.compile(r"_validity_timed$"),
    re.compile(r"^validity$"),         # NEW: catches 127 nodes
    re.compile(r"^validity_timed$"),   # NEW: catches 120 nodes
]
```

#### 2B. Define accessor terminal classification

Add to `dd_enrichment.py` (near BOILERPLATE_PATTERNS):

```python
ACCESSOR_TERMINAL_NAMES = frozenset({
    # Generic data containers
    'value', 'data', 'values',
    # Time bases
    'time',
    # Grid references
    'grid_index', 'grid_subset_index', 'index',
    # Interpolation
    'coefficients',
    # Geometric components
    'r', 'z', 'phi', 'x', 'y',
    # Directional components
    'parallel', 'poloidal', 'radial', 'toroidal', 'diamagnetic',
    # Validity (also covered by BOILERPLATE but explicit here)
    'validity', 'validity_timed',
    # Fit results
    'measured', 'reconstructed', 'chi_squared',
    # Labels/identifiers
    'label',
    # GGD structure
    'neighbours', 'nodes', 'measure', 'space',
    'geometry', 'geometry_2d', 'dim1', 'dim2', 'closed',
    # Other common accessors
    'surface', 'weight', 'multiplicity',
    'a', 'd', 'v',
})

ACCESSOR_TERMINAL_SUFFIXES = ('_coefficients', '_n')

ACCESSOR_SINGLE_LETTER_THRESHOLD = 20  # single-letter names appearing ≥20 times
```

#### 2C. Expand `is_boilerplate_path()` → rename to `is_accessor_terminal()`

```python
def is_accessor_terminal(path_id: str, name: str) -> bool:
    """Check if a node is an accessor terminal (template-enriched, not embedded)."""
    # Existing boilerplate patterns (error/validity/metadata)
    if any(p.search(name) for p in BOILERPLATE_PATTERNS):
        return True
    # Structural metadata subtrees
    parts = path_id.split("/")
    if len(parts) >= 2 and parts[1] in ("ids_properties", "code"):
        return True
    # Accessor terminal names
    if name in ACCESSOR_TERMINAL_NAMES:
        return True
    # Suffix patterns
    if any(name.endswith(s) for s in ACCESSOR_TERMINAL_SUFFIXES):
        return True
    return False
```

#### 2D. Expand `generate_template_description()` for accessor terminals

Add template descriptions for each accessor terminal category. Templates must reference the parent node to provide context:

```python
# Template patterns (parent_name and parent_doc come from HAS_PARENT query)
ACCESSOR_TEMPLATES = {
    'value': "{parent_doc_short}",
    'data': "Data array for {parent_name_readable}.",
    'values': "Array of values for {parent_name_readable}.",
    'time': "Time base for {parent_name_readable}.",
    'coefficients': "Interpolation coefficients for {parent_name_readable}.",
    'grid_index': "Grid index reference for {parent_name_readable}.",
    'grid_subset_index': "Grid subset index for {parent_name_readable}.",
    'index': "Integer index for {parent_name_readable}.",
    'label': "String label for {parent_name_readable}.",
    'r': "Major radius (R) coordinate of {parent_name_readable}.",
    'z': "Vertical (Z) coordinate of {parent_name_readable}.",
    'phi': "Toroidal angle (φ) coordinate of {parent_name_readable}.",
    'x': "X coordinate of {parent_name_readable}.",
    'y': "Y coordinate of {parent_name_readable}.",
    'parallel': "Parallel component of {parent_name_readable}.",
    'poloidal': "Poloidal component of {parent_name_readable}.",
    'radial': "Radial component of {parent_name_readable}.",
    'toroidal': "Toroidal component of {parent_name_readable}.",
    'diamagnetic': "Diamagnetic component of {parent_name_readable}.",
    'validity': "Integer validity flag for {parent_name_readable}. 0=valid, negative=invalid, positive=valid with caveats.",
    'validity_timed': "Time-dependent validity array for {parent_name_readable}.",
    'measured': "Measured value of {parent_name_readable}.",
    'reconstructed': "Reconstructed value of {parent_name_readable}.",
    'chi_squared': "Chi-squared goodness of fit for {parent_name_readable}.",
}
# For `value` nodes, the template uses the parent's documentation as the description
# because `value` IS the parent's data — it needs no separate identity
```

The `generate_template_description()` function needs a parent query. Modify to accept parent context:
```python
def generate_template_description(path_id: str, path_info: dict, parent_info: dict | None = None) -> dict:
```

#### 2E. Modify enrichment pipeline to query parent for template generation

In `enrich_imas_paths()`, when processing accessor terminals, batch-query their parents:
```cypher
UNWIND $path_ids AS pid
MATCH (n:IMASNode {id: pid})-[:HAS_PARENT]->(parent:IMASNode)
RETURN pid AS path_id, parent.id AS parent_id, parent.name AS parent_name,
       coalesce(parent.description, parent.documentation) AS parent_doc
```

**Gate**: ≥8,000 accessor terminal nodes template-enriched (up from 17). Verify with:
```cypher
MATCH (n:IMASNode {node_category: 'data', enrichment_source: 'template'})
RETURN count(n)
```

**Agent instructions**: This phase is self-contained. One agent can implement 2A-2E, run the enrichment pipeline, and verify the gate. No dependency on Phase 1 prompt changes — template enrichment doesn't use the LLM prompt.

---

### Phase 3: BM25 Scoring Fix

**Goal**: Fix text search scoring so it doesn't drown vector results.

**Files**: `imas_codex/tools/graph_search.py`

**Depends on**: Nothing (fully independent)

#### 3A. Remove BM25 score floor

At `graph_search.py` line ~1905:
```python
# BEFORE:
normalized.append({"id": pid, "score": max(raw, 0.7)})

# AFTER:
normalized.append({"id": pid, "score": raw})
```

#### 3B. Compress CONTAINS fallback scores

At `graph_search.py` lines ~1923-1935, reduce the 5-tier cascade from 0.85-0.95 to 0.55-0.80:

| Tier | Before | After | Rationale |
|------|--------|-------|-----------|
| Doc match + leaf type | 0.95 | 0.80 | Strong signal but shouldn't beat top vector |
| Name match + leaf type | 0.93 | 0.75 | Good signal |
| ID contains query | 0.90 | 0.70 | Path match |
| Doc match only | 0.88 | 0.65 | Moderate signal |
| Keywords/description | 0.85 | 0.55 | Weakest text signal |

#### 3C. Cap merged scores at 1.0

At `graph_search.py` line ~208:
```python
# BEFORE:
scores[pid] = round(max(scores[pid], text_score) + 0.05, 4)

# AFTER:
scores[pid] = round(min(max(scores[pid], text_score) + 0.05, 1.0), 4)
```

#### 3D. Add path-detection short-circuit

If query contains `/`, treat it as a path lookup — skip vector search entirely:
```python
if "/" in query:
    # Direct path lookup mode — text search only
    text_results = _text_search_imas_paths(gc, query, limit=max_results, ...)
    # Skip vector search and score merging
```

**Gate**: Run the benchmark queries through text-only search. Verify that BM25 scores for strong matches are now 0.6-0.8 (not 0.7-0.95). Verify path queries like `equilibrium/time_slice/profiles_1d/psi` return exact matches at rank 1.

**Agent instructions**: This phase is purely scoring logic changes. One agent can implement 3A-3D and write scoring unit tests. No data pipeline needed.

---

### Phase 4: Embedding Text Redesign

**Goal**: Replace the 8-component embedding text with the concise description.

**Files**: `imas_codex/graph/build_dd.py`, `imas_codex/graph/dd_graph_ops.py`

**Depends on**: Phase 1 (prompt validation gate must pass)

#### 4A. Rewrite `generate_embedding_text()`

In `build_dd.py:143-280`, replace the entire function:

```python
def generate_embedding_text(path: str, path_info: dict, ids_info: dict | None = None) -> str:
    """Generate embedding text for an IMAS DD node.

    Uses the concise LLM-enriched description (preferred) or raw documentation
    as fallback. No metadata concatenation — the description alone is the
    embedding text for maximum vector search precision.
    """
    desc = path_info.get("description", "").strip()
    if desc:
        return desc
    doc = path_info.get("documentation", "").strip()
    return doc
```

#### 4B. Add concept-node filter to embedding claims

In `claim_paths_for_embedding()` at `dd_graph_ops.py:181-230`, add a filter to skip accessor terminals:

```cypher
MATCH (p:IMASNode)
WHERE p.status = $status
  AND p.node_category = 'data'
  AND p.enrichment_source <> 'template'  -- Skip accessor terminals
  AND (p.claimed_at IS NULL
       OR p.claimed_at < datetime() - duration($cutoff))
WITH p
ORDER BY rand()
LIMIT $limit
SET p.claimed_at = datetime(), p.claim_token = $token
```

This leverages the template enrichment from Phase 2: all accessor terminals now have `enrichment_source='template'`, so filtering by `<> 'template'` gives us only concept nodes.

#### 4C. Handle the status transition

Accessor terminals (template-enriched) should transition from `enriched` → `embedded` without actually generating embeddings. Add a bulk status update after Phase 2:

```cypher
MATCH (n:IMASNode {node_category: 'data', enrichment_source: 'template'})
WHERE n.status = 'enriched'
SET n.status = 'embedded',
    n.embedding = null,
    n.embedding_text = null,
    n.embedded_at = null
```

This ensures they don't block the pipeline and don't appear in the vector index.

**Gate**: After re-embedding concept nodes only:
- Verify corpus size: ~10,400 embedded nodes (down from 20,037)
- Verify avg embedding_text length: 80-150 chars (down from 486)
- Run benchmark: target **MRR ≥ 0.25**

**Agent instructions**: This phase modifies the embedding pipeline. Must run AFTER Phase 1 confirms the prompt quality and AFTER Phase 2 sets enrichment_source='template' on accessor terminals.

---

### Phase 5: Child Traversal in Search Results

**Goal**: When search finds a concept node, show its children grouped by role.

**Files**: `imas_codex/tools/graph_search.py`, `imas_codex/llm/search_formatters.py`

**Depends on**: Phase 2 (accessor terminal classification needed for grouping)

#### 5A. Add child fetch after search

In `search_imas_paths()`, after scoring and metadata enrichment, add a child traversal step for structure nodes:

```cypher
UNWIND $concept_ids AS pid
MATCH (concept:IMASNode {id: pid})
WHERE concept.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
OPTIONAL MATCH (child:IMASNode {node_category: 'data'})-[:HAS_PARENT]->(concept)
WITH pid, collect({
    name: child.name,
    id: child.id,
    data_type: child.data_type,
    documentation: child.documentation
}) AS children
WHERE size(children) > 0
RETURN pid AS parent_id, children
```

#### 5B. Group children by role

Use the accessor terminal classification to group children:

| Role | Names | Display |
|------|-------|---------|
| **data** | `value`, `data`, `values` | Primary data fields |
| **time** | `time` | Time base |
| **coordinates** | `r`, `z`, `phi`, `x`, `y` | Geometric coordinates |
| **components** | `parallel`, `poloidal`, `radial`, `toroidal`, `diamagnetic` | Directional components |
| **interpolation** | `coefficients`, `*_coefficients` | Interpolation data |
| **grid** | `grid_index`, `grid_subset_index`, `index` | Grid references |
| **quality** | `validity`, `validity_timed`, `chi_squared` | Quality indicators |
| **fit** | `measured`, `reconstructed` | Measurement/reconstruction |
| **other** | everything else | Named children (physics quantities) |

#### 5C. Add `children` field to `SearchHit`

In `search_strategy.py`, add to `SearchHit`:
```python
children: list[dict[str, Any]] | None = None  # Grouped child nodes for structure results
```

#### 5D. Update search result formatter

In `search_formatters.py`, render children as a compact tree:
```
### equilibrium/time_slice/boundary (STRUCTURE)
  "Boundary shape of the last closed flux surface."
  IDS: equilibrium | Physics: equilibrium
  Children:
    Data: outline (STRUCTURE), psi (FLT_0D), geometric_axis (STRUCTURE)
    Coordinates: r (FLT_1D), z (FLT_1D)
    Error: psi_error_upper, psi_error_lower
```

**Gate**: Run 10 queries that should return structure nodes. Verify children appear in output. Verify non-structure results don't show children.

**Agent instructions**: This phase is search result formatting. One agent can implement 5A-5D. It needs Phase 2's accessor terminal classification for grouping logic, but doesn't need the enrichment pipeline to have run.

---

### Phase 6: Full Re-enrichment and Re-embedding

**Goal**: Apply all changes to the full DD graph.

**Files**: Pipeline execution (no code changes)

**Depends on**: Phases 1-4 all complete and validated

#### 6A. Reset enrichment status for LLM re-enrichment

```cypher
-- Reset all LLM-enriched data nodes to 'built' for re-processing
MATCH (n:IMASNode {node_category: 'data', enrichment_source: 'llm'})
SET n.status = 'built',
    n.description = null,
    n.keywords = null,
    n.enrichment_source = null,
    n.enrichment_hash = null
```

#### 6B. Run enrichment pipeline

```bash
uv run imas-codex dd enrich --force
```

With hierarchical ordering (Phase 1C), parents are enriched first. Accessor terminals get template descriptions (Phase 2). Concept nodes get concise LLM descriptions (Phase 1).

Expected outcome:
- ~9,600 template-enriched (accessor terminals)
- ~10,400 LLM-enriched (concept nodes) with avg description ~100-150 chars
- Total cost: ~$2-5 (Google Flash Lite at ~$0.07/M tokens × ~50M tokens)

#### 6C. Clear existing embeddings and re-embed

```cypher
-- Clear all existing embeddings
MATCH (n:IMASNode)
WHERE n.embedding IS NOT NULL
SET n.embedding = null, n.embedding_text = null, n.embedded_at = null, n.embedding_hash = null
```

```bash
uv run imas-codex dd embed
```

Only concept nodes (~10,400) will be claimed for embedding (Phase 4B filter). Each gets its concise description as embedding text (Phase 4A).

Expected outcome:
- ~10,400 embedded nodes (down from 20,037)
- Avg embedding text: ~100-150 chars (down from 486)
- Embedding time: ~15 min with 4×P100 GPUs

#### 6D. Full benchmark

Run the complete benchmark suite:

| Metric | Target | Current |
|--------|--------|---------|
| Vector P@1 | ≥ 15% | 0% |
| Vector MRR | ≥ 0.25 | 0.016 |
| Keyword MRR | ≥ 0.30 | ~0.20 (estimated) |
| Hybrid MRR | ≥ max(vector, keyword) | < min(vector, keyword) |
| End-to-end MRR | ≥ 0.35 | ~0.15 |

**Agent instructions**: This phase is pipeline execution. One agent runs the full enrichment, embedding, and benchmark. Must wait for all code changes (Phases 1-5) to be committed and validated.

---

### Phase 7: Integration Tests

**Goal**: Ensure all changes are tested and maintainable.

**Files**: `tests/graph/`, `tests/tools/`

**Depends on**: Phases 1-5 code complete

#### 7A. Prompt quality tests

Test that the enrichment prompt produces descriptions matching target criteria:
- Length: 50-200 chars (reject >300)
- No metadata repetition (units, data types, coordinates)
- Starts with quantity name (not "The" or "This" or "Container for")

#### 7B. Accessor terminal classification tests

Test `is_accessor_terminal()`:
- Known accessor names return True
- Known concept names return False
- Suffix patterns match correctly
- Edge cases (empty name, unknown name)

#### 7C. Template enrichment tests

Test `generate_template_description()`:
- Each template category produces valid descriptions
- Parent context is correctly interpolated
- Descriptions include `enrichment_source: 'template'`

#### 7D. BM25 scoring tests (no server needed)

Test the scoring pipeline:
- No score floor (BM25 scores can be < 0.7)
- CONTAINS fallback scores in correct ranges
- Merged scores capped at 1.0
- Path queries skip vector search

#### 7E. Child traversal tests

Test child grouping and formatting:
- Structure nodes return grouped children
- Leaf nodes return no children
- Role classification works correctly

**Agent instructions**: This phase is test writing. Can be parallelized across agents — each agent takes one test file.

## Parallel Dispatch Plan

```
Timeline:
                                                    
Phase 1: Prompt redesign + validation    ████████░░░  (sequential — must validate)
Phase 2: Template enrichment expansion   ████████░░░  (parallel with 1)
Phase 3: BM25 scoring fix               █████░░░░░░  (parallel with 1, 2)
Phase 5: Child traversal                 ██████░░░░░  (parallel with 1, 2, 3)
                                         ↓ gate
Phase 4: Embedding text redesign         ░░░░░████░░  (after Phase 1 + 2 gates)
                                              ↓ gate
Phase 6: Full re-enrich + re-embed       ░░░░░░░████  (after Phase 4 gate)
Phase 7: Integration tests               ░░░░░██████  (after Phases 1-5 code)
```

**Group 1** (launch in parallel immediately):
- Agent A: Phase 1 (prompt redesign + A/B validation) — **blocking gate**
- Agent B: Phase 2 (template enrichment expansion)
- Agent C: Phase 3 (BM25 scoring fix)
- Agent D: Phase 5 (child traversal in search results)

**Group 2** (after Phase 1 + 2 gates pass):
- Agent E: Phase 4 (embedding text redesign)
- Agent F: Phase 7 (tests for Phases 1-3, 5 — can start immediately for completed phases)

**Group 3** (after Phase 4 gate passes):
- Agent G: Phase 6 (full re-enrichment, re-embedding, benchmark)

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Concise prompt produces worse descriptions than raw docs | Phase 1 A/B validation gate — if MRR < 0.20, iterate on prompt before proceeding |
| Template enrichment misclassifies concept nodes | Conservative accessor terminal list — only names with ≥20 occurrences and clearly generic semantics. Physics quantities like `density`, `pressure`, `psi` are NOT in the list |
| BM25 score changes break existing keyword search | Phase 3 gate tests keyword-only MRR. Score compression is conservative (0.55-0.80 range still provides clear ranking) |
| Re-enrichment costs too much | Google Flash Lite at $0.07/M tokens. Full re-enrichment of 10,400 nodes × 50 per batch = 208 calls. Estimated $2-5 total |
| Embed server capacity | Current 4×P100 handles 20K nodes in ~30 min. 10.4K will take ~15 min |

## Success Criteria

The plan succeeds when:
1. **Vector P@1 > 0%** (from 0% — any improvement proves the fix works)
2. **Vector MRR ≥ 0.20** (from 0.016 — 12× improvement)
3. **Hybrid MRR > max(vector, keyword)** (from "all hybrids worse" — proves merging works)
4. **Search results show children** for structure nodes
5. **Accessor terminals** are template-enriched, not embedded, and not returned as top results for physics queries
