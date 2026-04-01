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

**Note**: Phase 2 establishes the initial accessor terminal classification using explicit name lists and suffix patterns. Phase 6 replaces this with a layered classification pipeline that adds force-include physics concepts, regex patterns, and structural heuristics for future-proofing. Phase 2 code will be refactored in Phase 6 — the initial implementation here is deliberately simpler to unblock parallel work.

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

#### 4B. Keyword inheritance from children to parent descriptions

Concept nodes (STRUCTURE/STRUCT_ARRAY) should inherit relevant child accessor names into their keyword lists. This ensures queries like "radius of X-point" can match the parent through keywords even without the child being embedded:

```python
def inherit_child_keywords(gc: GraphClient):
    """Add child accessor names as keywords on parent concept nodes."""
    KEYWORD_WORTHY_CHILDREN = {
        'r': ['radius', 'major_radius', 'R'],
        'z': ['height', 'vertical', 'Z'],
        'phi': ['toroidal_angle', 'azimuthal'],
        'time': ['timebase', 'temporal'],
        'measured': ['measured', 'measurement'],
        'reconstructed': ['reconstructed', 'reconstruction'],
        'parallel': ['parallel_component'],
        'poloidal': ['poloidal_component'],
        'radial': ['radial_component'],
        'toroidal': ['toroidal_component'],
    }
    
    gc.query("""
        UNWIND $mappings AS m
        MATCH (child:IMASNode {name: m.name, node_category: 'data'})
              -[:HAS_PARENT]->(parent:IMASNode)
        WHERE parent.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        WITH parent, m.extra_keywords AS kws
        SET parent.keywords = apoc.coll.toSet(
            coalesce(parent.keywords, []) + kws
        )
    """, mappings=[
        {"name": k, "extra_keywords": v} for k, v in KEYWORD_WORTHY_CHILDREN.items()
    ])
```

This runs once after enrichment completes. The additional keywords make parent nodes discoverable through keyword/BM25 search for child-oriented queries.

#### 4C. Add concept-node filter to embedding claims

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

#### 4D. Handle the status transition

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

### Phase 6: Accessor Terminal Classification — Layered Defense

**Goal**: Future-proof classification of accessor terminals using patterns, not just a static list.

**Files**: `imas_codex/graph/dd_enrichment.py`

**Depends on**: Phase 2 (builds on the accessor terminal work)

The static `ACCESSOR_TERMINAL_NAMES` frozenset from Phase 2 is necessary but insufficient. A new DD version could introduce `*_validate`, `*_flag`, `*_uncertainty` patterns that slip through. This phase adds layered pattern-based guards.

#### 6A. Layered classification pipeline (short-circuit on first match)

```python
def classify_node(path_id: str, name: str, node_stats: dict | None = None) -> str:
    """Classify a data node as 'concept' or 'accessor'.
    
    Layers evaluated in order. First match wins (short-circuit).
    """
    # Layer 1: Error/metadata (absolute, no false positives)
    if _is_error_or_metadata(name, path_id):
        return 'accessor'
    
    # Layer 2: Force-include physics concepts (semantic veto)
    if name in FORCE_INCLUDE_CONCEPTS:
        return 'concept'
    
    # Layer 3: Explicit accessor names (conservative list from Phase 2)
    if name in ACCESSOR_TERMINAL_NAMES:
        return 'accessor'
    
    # Layer 4: Regex suffix/prefix patterns (future-proof)
    if _matches_accessor_pattern(name):
        return 'accessor'
    
    # Layer 5: Frequency + structural heuristic (data-driven)
    if node_stats:
        occ = node_stats.get('occurrence_count', 0)
        struct_ratio = node_stats.get('structure_parent_ratio', 0.0)
        if occ >= 20 and struct_ratio >= 0.95:
            logger.info(f"Accessor by heuristic: {name} (occ={occ}, ratio={struct_ratio})")
            return 'accessor'
    
    # Default: concept
    return 'concept'
```

#### 6B. Force-include physics concepts (Layer 2)

Protects real physics quantities from frequency-based misclassification:

```python
FORCE_INCLUDE_CONCEPTS = frozenset({
    'psi', 'density', 'temperature', 'pressure', 'flux',
    'current', 'voltage', 'power', 'energy', 'frequency',
    'velocity', 'momentum', 'conductivity', 'resistivity',
    'elongation', 'triangularity', 'b0', 'b_field_r', 'b_field_z',
    'q', 'rho_tor_norm', 'rho_pol_norm',
})
```

#### 6C. Regex patterns (Layer 4) — catches unknown future accessors

```python
ACCESSOR_REGEX_PATTERNS = [
    # Error/uncertainty bounds (current + future)
    re.compile(r"_(error|uncertainty|confidence|reliability)_(upper|lower|index|bound)$"),
    # Standalone validity variants
    re.compile(r"^(error|uncertainty|validity)(_timed)?$"),
    # Coefficients (interpolation)
    re.compile(r"_coefficients$"),
    # Normalized variants
    re.compile(r"_n$"),
    # Flag/status patterns (future-proof)
    re.compile(r"_(flag|status|validate|check)$"),
    # Scale/offset patterns
    re.compile(r"_(scale|offset|weight|bias)$"),
]
```

#### 6D. Structural heuristic (Layer 5) — data-driven from graph

Pre-compute per-name statistics at pipeline start:
```cypher
MATCH (n:IMASNode {node_category: 'data'})
WHERE NOT EXISTS { MATCH (c:IMASNode)-[:HAS_PARENT]->(n) }
WITH n.name AS name, count(n) AS occ,
     sum(CASE WHEN EXISTS {
         MATCH (n)-[:HAS_PARENT]->(p:IMASNode)
         WHERE p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
     } THEN 1 ELSE 0 END) * 1.0 / count(n) AS struct_ratio
WHERE occ >= 20
RETURN name, occ, struct_ratio
```

Any name with ≥20 occurrences AND ≥95% of instances having a STRUCTURE parent is classified as accessor. This catches new patterns automatically without code changes.

**Gate**: Classification must produce ~9,500 accessors and ~5,700 concepts (±500). Verify `psi`, `density`, `temperature`, `pressure` are ALL classified as concepts.

**Agent instructions**: One agent implements 6A-6D and writes classification unit tests. Tests must cover all layers including edge cases.

---

### Phase 7: Accessor Terminal Query Routing

**Goal**: Ensure child-oriented queries ("radius of X-point") surface parent concept nodes even though accessor terminals are not embedded.

**Files**: `imas_codex/tools/graph_search.py`, `imas_codex/search/search_strategy.py`

**Depends on**: Phase 5 (child traversal), Phase 6 (accessor classification)

#### Problem

When accessor terminals are excluded from the vector index, queries like "radius of X-point" won't find `equilibrium/time_slice/boundary/x_point` because the parent's description says "X-point location" not "radius". The `r` child (which means "radius") is not embedded.

#### 7A. Parent-promotion in vector search

Modify the vector search Cypher to check if a hit is an accessor terminal and promote its parent:

```cypher
CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
YIELD node AS path, score
WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
  AND path.node_category = 'data'
OPTIONAL MATCH (path)-[:HAS_PARENT]->(parent:IMASNode)
WHERE parent.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
WITH CASE
    WHEN path.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'] THEN path
    WHEN parent IS NOT NULL AND NOT (path.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
         THEN parent
    ELSE path
  END AS result_node,
  score,
  CASE WHEN parent IS NOT NULL THEN path.name ELSE null END AS matched_accessor
RETURN result_node.id AS id, max(score) AS score,
       collect(matched_accessor) AS matched_children
```

This way, if the vector search happens to match an accessor terminal that IS embedded (e.g., some borderline nodes), the parent is returned instead with the accessor context.

#### 7B. Post-retrieval child name matching

After the initial search returns concept nodes, do a cheap second pass checking if query terms match any child names:

```python
def _boost_by_child_match(scores: dict, query: str, gc: GraphClient) -> dict:
    """Boost concept nodes whose children match query terms."""
    query_words = {w.lower() for w in query.split() if len(w) > 2}
    if not query_words:
        return scores
    
    # Child name synonyms for common accessor terminals
    CHILD_SYNONYMS = {
        'r': {'radius', 'major_radius', 'radial'},
        'z': {'height', 'vertical', 'elevation'},
        'phi': {'toroidal_angle', 'azimuthal'},
        'time': {'timebase', 'temporal'},
    }
    
    path_ids = list(scores.keys())
    children = gc.query("""
        UNWIND $path_ids AS pid
        MATCH (child:IMASNode)-[:HAS_PARENT]->(parent:IMASNode {id: pid})
        WHERE child.node_category = 'data'
        RETURN pid AS parent_id, collect(child.name) AS child_names
    """, path_ids=path_ids)
    
    for row in children:
        pid = row['parent_id']
        child_names = set(row['child_names'])
        # Check if query words match child names or their synonyms
        expanded = set()
        for cn in child_names:
            expanded.add(cn)
            expanded.update(CHILD_SYNONYMS.get(cn, set()))
        
        matches = query_words & expanded
        if matches and pid in scores:
            scores[pid] = round(scores[pid] + 0.03 * len(matches), 4)
    
    return scores
```

#### 7C. Add `matched_children` field to SearchHit

```python
# In search_strategy.py SearchHit class:
matched_children: list[str] | None = None  # Accessor terminals that matched the query
```

Display in formatter: when `matched_children` is non-empty, show which children were relevant:
```
### equilibrium/time_slice/boundary/x_point (STRUCTURE, score: 0.87)
  "Location of the X-point on the separatrix boundary."
  Matched via children: r (radius), z (vertical position)
```

**Gate**: Query "radius of X-point" must return `equilibrium/time_slice/boundary/x_point` in top 5. Query "time base for electron temperature" must return `core_profiles/.../electrons/temperature` in top 5.

**Agent instructions**: One agent implements 7A-7C. This touches the core search Cypher so needs careful testing. Write integration tests with the specific gate queries.

---

### Phase 8: Full Re-enrichment and Re-embedding

**Goal**: Apply all changes to the full DD graph.

**Files**: Pipeline execution (no code changes)

**Depends on**: Phases 1-6 all complete and validated

#### 8A. Reset and re-enrich

```cypher
MATCH (n:IMASNode {node_category: 'data', enrichment_source: 'llm'})
SET n.status = 'built',
    n.description = null, n.keywords = null,
    n.enrichment_source = null, n.enrichment_hash = null
```

```bash
uv run imas-codex dd enrich --force
```

Expected: ~9,600 template-enriched, ~10,400 LLM-enriched with avg description ~100-150 chars. Cost: ~$2-5.

#### 8B. Clear and re-embed

```cypher
MATCH (n:IMASNode) WHERE n.embedding IS NOT NULL
SET n.embedding = null, n.embedding_text = null, n.embedded_at = null, n.embedding_hash = null
```

```bash
uv run imas-codex dd embed
```

Expected: ~10,400 embedded concept nodes, avg embedding text ~100-150 chars. Time: ~15 min.

#### 8C. Independent per-method benchmarks

Run each search method INDEPENDENTLY before combining:

| Method | Benchmark | MRR Target | Current |
|--------|-----------|------------|---------|
| Vector-only | 30 queries, vector search only | ≥ 0.40 | 0.016 |
| BM25-only | 30 queries, fulltext index only | ≥ 0.45 | ~0.20 |
| CONTAINS-only | 30 queries, string matching only | ≥ 0.30 | ~0.25 |
| Path lookup | 10 exact path queries | 1.00 | ~0.90 |

**Gate**: Each method must pass its individual MRR target before proceeding to hybrid tuning. If any method fails, debug that method in isolation — do NOT combine broken methods.

**Agent instructions**: One agent runs the full pipeline and benchmarks each method independently. Reports per-method MRR with query-level detail.

---

### Phase 9: Hybrid Tuning and Score Integration

**Goal**: Find the optimal combination of search methods through systematic experimentation.

**Files**: `imas_codex/tools/graph_search.py`

**Depends on**: Phase 8 (all individual methods validated)

#### 9A. Replace naive score merging with RRF

Replace the current `max(vector, text) + 0.05` formula at `graph_search.py` line ~208 with Reciprocal Rank Fusion:

```python
def reciprocal_rank_fusion(
    vector_hits: list[dict], text_hits: list[dict], k: int = 60
) -> dict[str, float]:
    """Combine ranked lists using RRF. Score-scale invariant."""
    rrf_scores = {}
    for rank, hit in enumerate(sorted(vector_hits, key=lambda x: x['score'], reverse=True), start=1):
        rrf_scores[hit['id']] = rrf_scores.get(hit['id'], 0) + 1 / (k + rank)
    for rank, hit in enumerate(sorted(text_hits, key=lambda x: x['score'], reverse=True), start=1):
        rrf_scores[hit['id']] = rrf_scores.get(hit['id'], 0) + 1 / (k + rank)
    return rrf_scores
```

**Why RRF over weighted linear**: RRF is score-scale invariant — it uses ranks, not scores. This eliminates the core problem where BM25 scores (0.7-0.95) drown vector scores (0.82-0.88). No normalization needed.

#### 9B. Add vector confidence gating

If the best vector score is below a threshold, don't include vector results in the merge — they're noise:

```python
vector_gate = 0.65  # Tunable
if vector_results and max(r['score'] for r in vector_results) < vector_gate:
    # Vector search returned low-confidence results — use text only
    return {r['id']: r['score'] for r in text_results}
```

#### 9C. Grid search experimentation

Run a systematic experiment over parameter space using the 30-query benchmark set:

| Parameter | Values to Test |
|-----------|----------------|
| RRF k | 40, 60, 100 |
| Vector gate threshold | 0.55, 0.65, 0.75, disabled |
| Path segment boost | 0.0, 0.02, 0.03, 0.05 |
| Heuristic rerank | enabled, disabled |

Total: 3 × 4 × 4 × 2 = 96 experiments. Evaluate with MRR on the 30-query set.

#### 9D. Heuristic reranking (zero-cost)

After RRF merge, apply metadata-based boosts:

```python
def heuristic_rerank(hits: list, query: str) -> list:
    """Zero-cost reranking based on metadata signals."""
    query_words = set(query.lower().split())
    for hit in hits:
        # Boost: IDS name appears in query
        if hit.ids_name and hit.ids_name.lower() in query.lower():
            hit.score += 0.02
        # Boost: exact path segment match
        segments = hit.path.lower().split('/')
        for seg in segments:
            if seg.replace('_', ' ') in query.lower() or seg in query_words:
                hit.score += 0.01
        # Boost: well-documented paths (richer docs = higher confidence)
        if len(hit.documentation or '') > 100:
            hit.score += 0.01
    return sorted(hits, key=lambda h: h.score, reverse=True)
```

No LLM cost. The MCP agent/subagent consuming the tools can do any further reranking itself based on its task context.

#### 9E. Validate hybrid > individual methods

**Gate**: Hybrid MRR must exceed the best individual method's MRR. If not, iterate on parameters. If after 3 iterations hybrid is still worse, ship the best individual method as default and make hybrid configurable.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Hybrid MRR | > max(vector MRR, text MRR) | Must improve, not degrade |
| Hybrid MRR | ≥ 0.50 | Correct answer in top 2 on average |
| No regression | Hybrid P@1 ≥ max(individual P@1) | Never lose a result that was #1 |

**Agent instructions**: One agent implements 9A-9D, runs the grid search, and reports the optimal configuration. Must run AFTER Phase 8 benchmarks confirm individual methods are healthy.

---

### Phase 10: Graph-Backed Benchmark Tests

**Goal**: Transform benchmarks into permanent pytest tests that gate on Neo4j availability and protect search quality over time.

**Files**: `tests/search/benchmark_data.py`, `tests/search/benchmark_helpers.py`, `tests/search/test_search_benchmarks.py`

**Depends on**: Phase 9 (hybrid tuning establishes baseline MRR values)

#### 10A. Benchmark query set (`tests/search/benchmark_data.py`)

30 queries across 6 categories, each with gold-standard expected paths:

| Category | Count | Example |
|----------|-------|---------|
| Exact concept match | 5 | "electron temperature" → `core_profiles/.../electrons/temperature` |
| Disambiguating | 5 | "ECE electron temperature" → `ece/channel/t_e` (NOT core_profiles) |
| Structural/path | 5 | `equilibrium/time_slice/profiles_1d/psi` → exact match |
| Abbreviation/synonym | 5 | "Ip" → `equilibrium/.../global_quantities/ip` |
| Accessor-oriented | 5 | "radius of X-point" → parent with `r` child |
| Cross-domain | 5 | "bootstrap current" → transport + equilibrium paths |

#### 10B. MRR test helpers (`tests/search/benchmark_helpers.py`)

```python
def calculate_mrr(results: list[tuple[str, list[str], list[str]]]) -> float:
    """Calculate MRR from (query, expected_paths, returned_paths) triples."""
    reciprocal_ranks = []
    for query, expected, returned in results:
        rr = 0.0
        for rank, path in enumerate(returned, start=1):
            if path in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def assert_mrr_above(results, threshold, method_name):
    """Assert with detailed failure message showing per-query breakdown."""
    mrr = calculate_mrr(results)
    if mrr < threshold:
        failures = [(q, exp, ret[:5]) for q, exp, ret in results
                     if not any(p in exp for p in ret[:5])]
        msg = f"{method_name} MRR={mrr:.3f} < {threshold}\nFailed queries:\n"
        for q, exp, ret in failures[:5]:
            msg += f"  '{q}': expected {exp[:2]}, got {ret[:3]}\n"
        raise AssertionError(msg)
```

#### 10C. Per-method benchmark tests (`tests/search/test_search_benchmarks.py`)

Each search method has an independent test with its own MRR threshold:

```python
import pytest
pytestmark = [pytest.mark.graph, pytest.mark.benchmark]

class TestVectorSearchBenchmark:
    """Vector/semantic search quality — gated on embed server + graph."""
    
    MRR_THRESHOLD = 0.40  # Baseline from Phase 8
    
    def test_vector_mrr(self, search_tool, benchmark_queries):
        results = []
        for q in benchmark_queries:
            hits = search_tool._vector_search_only(q.query_text, limit=50)
            results.append((q.query_text, q.expected_paths, [h['id'] for h in hits]))
        assert_mrr_above(results, self.MRR_THRESHOLD, "Vector")

class TestBM25SearchBenchmark:
    """BM25/fulltext search quality — gated on graph only."""
    
    MRR_THRESHOLD = 0.45
    
    def test_bm25_mrr(self, search_tool, benchmark_queries):
        # ... similar pattern, text search only ...

class TestHybridSearchBenchmark:
    """Combined hybrid search — must exceed both individual methods."""
    
    MRR_THRESHOLD = 0.50
    
    def test_hybrid_exceeds_individual(self, search_tool, benchmark_queries):
        # Run all three, verify hybrid > max(vector, text)

class TestAccessorRouting:
    """Accessor terminal queries must surface parent concept nodes."""
    
    def test_radius_of_xpoint(self, search_tool):
        hits = search_tool.search_imas_paths("radius of X-point", max_results=10)
        paths = [h.path for h in hits.hits]
        assert any('x_point' in p for p in paths[:5])
    
    def test_time_base_for_temperature(self, search_tool):
        hits = search_tool.search_imas_paths("time base for electron temperature", max_results=10)
        paths = [h.path for h in hits.hits]
        assert any('temperature' in p for p in paths[:5])
```

#### 10D. Regression detection

Tests pin MRR baselines. When a change improves search quality, update the threshold:

```python
# Update baselines after validated improvements:
# TestVectorSearchBenchmark.MRR_THRESHOLD = 0.45  # was 0.40
```

If a PR causes MRR to drop by >0.05 from the pinned baseline, the test fails and blocks merge.

#### 10E. Embed server gating

Vector benchmark tests need the embed server running. Add a marker:

```python
@pytest.fixture
def requires_embed_server():
    """Skip if embed server is not running."""
    try:
        from imas_codex.tools.graph_search import _get_encoder
        encoder = _get_encoder()
        encoder.encode(["test"])
    except Exception:
        pytest.skip("Embed server not available")
```

BM25 and CONTAINS tests do NOT need the embed server — they should always run when the graph is available.

**Gate**: All test classes pass. MRR thresholds match validated baselines from Phase 9.

**Agent instructions**: One agent creates the test files, benchmark data, and helpers. Tests must pass when the graph is available and skip gracefully when it's not.

---

### Phase 11: Integration Tests for Code Changes

**Goal**: Unit and integration tests for all code changes from Phases 1-7.

**Files**: `tests/graph/`, `tests/tools/`

**Depends on**: Phases 1-7 code complete

#### 11A. Prompt quality tests
- Description length: 50-200 chars (reject >300)
- No metadata repetition (units, data types, coordinates)
- Starts with quantity name (not "The" or "This" or "Container for")

#### 11B. Accessor classification tests
- All 6 layers tested independently
- Force-include concepts: `psi`, `density`, `temperature`, `pressure` → always concept
- Regex patterns catch: `_coefficients`, `_n`, `_flag`, `_validate`, `_uncertainty`
- Structural heuristic: mock node_stats for boundary cases
- Edge cases: empty name, single-char name, unknown name → default concept

#### 11C. Template enrichment tests
- Each template category produces valid descriptions
- Parent context correctly interpolated
- `enrichment_source: 'template'` always set

#### 11D. BM25 scoring tests (no server needed)
- No score floor (BM25 scores can be < 0.7)
- CONTAINS fallback scores in compressed ranges (0.55-0.80)
- Score cap at 1.0
- Path queries skip vector search

#### 11E. Child traversal tests
- Structure nodes return grouped children
- Leaf nodes return no children
- Role classification works correctly
- Formatter produces expected tree output

#### 11F. RRF merge tests
- Two ranked lists merge correctly
- Missing documents handled (partial credit)
- k parameter affects ranking
- Vector gating works (low-confidence vector excluded)

**Agent instructions**: Can be parallelized — each test file is independent.

---

## Parallel Dispatch Plan

```
Timeline:

Phase 1:  Prompt redesign + validation     ████████░░░░░░░  (blocking gate: MRR ≥ 0.40)
Phase 2:  Template enrichment expansion     ████████░░░░░░░  (parallel with 1)
Phase 3:  BM25 scoring fix                 █████░░░░░░░░░░  (parallel with 1, 2)
Phase 5:  Child traversal                  ██████░░░░░░░░░  (parallel with 1, 2, 3)
Phase 6:  Layered accessor classification  ░░██████░░░░░░░  (after Phase 2)
                                           ↓ gate
Phase 4:  Embedding text redesign          ░░░░░████░░░░░░  (after Phase 1 + 2)
Phase 7:  Accessor query routing           ░░░░░░████░░░░░  (after Phase 5 + 6)
                                                ↓ gate
Phase 8:  Re-enrich + re-embed + per-method benchmarks  ░░░░░░░░████░░░  (after 1-7)
                                                             ↓ gate (individual MRRs)
Phase 9:  Hybrid tuning + grid search      ░░░░░░░░░░░███░  (after Phase 8 gates)
                                                        ↓ gate (hybrid > individual)
Phase 10: Graph-backed benchmark tests     ░░░░░░░░░░░░███  (after Phase 9)
Phase 11: Integration tests                ░░░░░░░██████░░  (after Phases 1-7 code)
```

**Group 1** (launch in parallel immediately):
- Agent A: Phase 1 (prompt redesign + A/B validation) — **blocking gate**
- Agent B: Phase 2 (template enrichment expansion)
- Agent C: Phase 3 (BM25 scoring fix)
- Agent D: Phase 5 (child traversal in search results)

**Group 2** (after Group 1 completes):
- Agent E: Phase 4 (embedding text redesign — after Phase 1 + 2)
- Agent F: Phase 6 (layered accessor classification — after Phase 2)
- Agent G: Phase 7 (accessor query routing — after Phase 5 + 6)
- Agent H: Phase 11 (integration tests — after Phases 1-7 code)

**Group 3** (after all code phases):
- Agent I: Phase 8 (full re-enrichment, re-embedding, per-method benchmarks)

**Group 4** (after individual method gates pass):
- Agent J: Phase 9 (hybrid tuning grid search)

**Group 5** (after hybrid tuning validated):
- Agent K: Phase 10 (graph-backed benchmark tests)

## MRR Targets (revised upward)

| Metric | Target | Current | Rationale |
|--------|--------|---------|-----------|
| Vector P@1 | ≥ 25% | 0% | Correct answer at rank 1 for 1 in 4 queries |
| Vector MRR | ≥ 0.40 | 0.016 | Correct answer in top 2-3 on average |
| BM25 MRR | ≥ 0.45 | ~0.20 | Strong keyword matching |
| CONTAINS MRR | ≥ 0.30 | ~0.25 | Fallback, lower bar acceptable |
| Path lookup accuracy | 1.00 | ~0.90 | Exact path queries must always work |
| **Hybrid MRR** | **≥ 0.50** | **~0.15** | **Must exceed best individual method** |
| End-to-end MRR | ≥ 0.55 | ~0.15 | With accessor routing + child traversal |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Concise prompt produces worse descriptions than raw docs | Phase 1 A/B validation gate with iteration budget (3 prompt revisions) |
| Template enrichment misclassifies concept nodes | Layered defense (Phase 6) with force-include physics concepts list |
| Static accessor list misses future DD patterns | Layer 4 regex patterns + Layer 5 structural heuristic = future-proof |
| BM25 score changes break existing keyword search | Phase 3 gate tests keyword-only MRR before integration |
| Hybrid worse than individual methods | Phase 9 grid search with explicit gate: hybrid > max(individual). Fallback: ship best individual |
| RRF k value wrong | Grid search tests k=40,60,100. RRF is robust — performance varies <5% across k values |
| Overfitting benchmark queries | 30 queries is small but stratified across 6 categories. Phase 10 tests are regression guards, not the tuning set |
| Accessor routing adds latency | Post-retrieval child matching is one additional Cypher query. Budget: <50ms additional |
| Re-enrichment costs too much | Google Flash Lite at $0.07/M tokens. ~$2-5 total for full re-enrichment |

## Success Criteria

The plan succeeds when:
1. **Each search method passes its independent MRR gate** (vector ≥ 0.40, BM25 ≥ 0.45, etc.)
2. **Hybrid MRR > max(vector MRR, BM25 MRR)** — combination must add value, not destroy it
3. **Hybrid MRR ≥ 0.50** — correct answer in top 2 on average
4. **Accessor-oriented queries** surface parent concept nodes (verified by specific test cases)
5. **Search results show grouped children** for structure nodes
6. **Accessor classification is pattern-based** and catches future DD additions without code changes
7. **All benchmarks are codified as pytest tests** that protect quality on every commit
8. **No LLM cost in the search path** — only in the offline enrichment pipeline
