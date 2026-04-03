# Embedding Upgrade, Quantization & SEARCH Clause Migration

## Problem Statement

A critical bug in `RemoteEmbeddingClient._embed_single()` means the `dimension` parameter
is never sent to the embedding server. The server always returns its default (256-dim),
silently discarding 75% of Qwen3-Embedding-0.6B's native 1024-dim capacity. This was
discovered when the `TestDimensionComparison` harness returned identical MRR at all
dimensions (256/512/1024) — the server was returning 256-dim vectors regardless of request.

Fixing the bug is the prerequisite for all subsequent work. Once fixed, we can collect
**real MRR data** across dimensions using representative graph content — IMAS paths,
wiki chunks (including multi-language), code snippets, and facility signals — to make
an **evidence-based decision** on the target dimension.

Regardless of the final dimension choice, the graph will benefit from:

- **Quantization** — Neo4j 2026.01 supports `vector.quantization.enabled`, giving ~4×
  memory savings. Even at 1024-dim with quantization, storage matches current unquantized
  256-dim. This makes the storage argument for higher dimensions compelling.
- **Re-embedding** — All ~20K+ embedded nodes need fresh vectors. The current embeddings
  were generated with the bug active (256-dim from server, no per-request control).
  Re-embedding is required regardless of dimension change to ensure consistency.
- **SEARCH clause migration** — 27 call sites use the legacy
  `db.index.vector.queryNodes()` procedure. Neo4j 2026.01's native `SEARCH` clause
  supports in-index pre-filtering, eliminating the 3-5× k oversampling we currently
  use to compensate for post-filter loss.

## Approach

Six phases, ordered by dependency. Phases 0-1 are prerequisites that inform the
dimension decision. Phases 2-5 can proceed in parallel after the decision is made.

```
Phase 0 (bug fix) → Phase 1 (dimension evaluation) → DECISION POINT
                                                         ↓
                                              ┌──────────┼──────────┐
                                              ↓          ↓          ↓
                                           Phase 2    Phase 3    Phase 4
                                          (re-embed) (SEARCH)  (quantize)
                                              └──────────┼──────────┘
                                                         ↓
                                                      Phase 5
                                                   (embed text opt)
```

---

## Phase 0 — Fix Client Dimension Bug

**Goal:** Make `RemoteEmbeddingClient` send the `dimension` field in HTTP requests
so the server performs per-request Matryoshka truncation.

**Agent count:** 1

### 0.1 Add `dimension` parameter to client methods

**File:** `imas_codex/embeddings/client.py`

Add `dimension: int | None = None` to:
- `embed()` (line 184)
- `_embed_single()` (line 266)
- `_embed_chunked()` (line 219)

In `_embed_single()`, add to request body (after line 280):
```python
request_body: dict = {"texts": texts, "normalize": normalize}
if prompt_name is not None:
    request_body["prompt_name"] = prompt_name
if dimension is not None:
    request_body["dimension"] = dimension
```

### 0.2 Wire dimension through Encoder

**File:** `imas_codex/embeddings/encoder.py`

In `embed_texts()` (line 334), pass `dimension=get_embedding_dimension()` to
the remote client call (line 351):
```python
embeddings = self._remote_client.embed(
    texts,
    normalize=self.config.normalize_embeddings,
    prompt_name=prompt_name,
    dimension=get_embedding_dimension(),
)
```

Update the retry path (line 360) similarly.

The `_truncate_embeddings()` call (line 367) becomes a safety net — it should be a
no-op when the server respects the dimension request, but keeps the defensive
truncation for any edge cases.

Apply the same pattern in `embed_texts_with_result()` (line 413) and
`_generate_embeddings()` (line 683).

### 0.3 Add tests

**File:** `tests/embeddings/test_remote.py`

Add test verifying `dimension` is included in the HTTP request body:
```python
def test_embed_sends_dimension(self, mock_client_cls):
    """embed() includes dimension in request body when provided."""
    # ... mock setup ...
    client = RemoteEmbeddingClient("http://localhost:18765")
    client.embed(["text"], dimension=512)
    call_args = mock_instance.post.call_args
    assert call_args[1]["json"]["dimension"] == 512

def test_embed_omits_dimension_when_none(self, mock_client_cls):
    """embed() does not include dimension key when None."""
    # ... mock setup ...
    client = RemoteEmbeddingClient("http://localhost:18765")
    client.embed(["text"])
    call_args = mock_instance.post.call_args
    assert "dimension" not in call_args[1]["json"]
```

### 0.4 Verify end-to-end

Run `uv run pytest tests/embeddings/ -v` — all existing + new tests must pass.

---

## Phase 1 — Dimension Evaluation with Real Graph Data

**Goal:** Collect hard MRR numbers at 256, 512, and 1024 dimensions using content
representative of what is actually stored in the graph. This evidence drives the
dimension decision.

**Agent count:** 1 (requires live graph + embedding server)

### 1.1 Expand test corpus with graph-sourced examples

**File:** `tests/search/test_search_evaluation.py`

The current `TestDimensionComparison.sample_nodes` fixture samples only IMASNode
descriptions. Expand to include all content types that are embedded in the graph:

#### IMAS DD paths (existing — keep)
```python
# Already in sample_nodes fixture
# Format: "{path}. {enriched_description}"
# Example: "equilibrium/time_slice/profiles_1d/psi. Poloidal magnetic flux..."
```

#### Wiki chunks (NEW — add to corpus)
Sample 30 WikiChunk nodes from the graph, including multi-language content:
```python
wiki_chunks = graph_client.query("""
    MATCH (c:WikiChunk)
    WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL
    WITH c, rand() AS r
    ORDER BY r LIMIT 30
    RETURN c.id AS id, substring(c.text, 0, 500) AS text,
           c.facility_id AS facility_id
""")
```

Ensure multi-language coverage by also sampling language-tagged pages:
```python
# French/German wiki content (TCV has French, JET has some German)
multilang_chunks = graph_client.query("""
    MATCH (p:WikiPage)-[:HAS_CHUNK]->(c:WikiChunk)
    WHERE p.content_language IS NOT NULL
      AND p.content_language <> 'en'
      AND c.text IS NOT NULL
    WITH c, p.content_language AS lang, rand() AS r
    ORDER BY r LIMIT 10
    RETURN c.id AS id, substring(c.text, 0, 500) AS text,
           lang AS language
""")
```

#### Code chunks (NEW — add to corpus)
Sample 20 CodeChunk nodes:
```python
code_chunks = graph_client.query("""
    MATCH (cc:CodeChunk)
    WHERE cc.text IS NOT NULL AND cc.embedding IS NOT NULL
    WITH cc, rand() AS r
    ORDER BY r LIMIT 20
    RETURN cc.id AS id, substring(cc.text, 0, 500) AS text,
           cc.function_name AS function_name
""")
```

#### Signal descriptions (NEW — add to corpus)
Sample 20 FacilitySignal descriptions:
```python
signals = graph_client.query("""
    MATCH (s:FacilitySignal)
    WHERE s.description IS NOT NULL AND s.embedding IS NOT NULL
    WITH s, rand() AS r
    ORDER BY r LIMIT 20
    RETURN s.id AS id, s.description AS text
""")
```

### 1.2 Build multi-domain query set

Add cross-domain test queries that exercise the diversity of content types:

```python
DIMENSION_EVAL_QUERIES = [
    # IMAS DD queries (already covered by ALL_QUERIES benchmark)

    # Wiki-style natural language queries
    ("how to access equilibrium data in TCV", "wiki"),
    ("COCOS sign conventions for poloidal flux", "wiki"),
    ("Thomson scattering calibration procedure", "wiki"),

    # Multi-language queries (test cross-lingual retrieval)
    ("température électronique du plasma", "multilang"),  # French
    ("profil de densité électronique", "multilang"),       # French
    ("magnetische Flussdichte", "multilang"),              # German
    ("Plasmastrom Messung", "multilang"),                  # German
    ("プラズマ電流", "multilang"),                           # Japanese
    ("電子温度プロファイル", "multilang"),                    # Japanese

    # Code search queries
    ("read MDSplus tree node data", "code"),
    ("equilibrium reconstruction LIUQE", "code"),
    ("TDI function for magnetic field", "code"),

    # Abbreviation queries (stress test for low-dim)
    ("Ip", "abbreviation"),
    ("Te", "abbreviation"),
    ("ne profile", "abbreviation"),
    ("q95", "abbreviation"),
    ("Zeff", "abbreviation"),
]
```

### 1.3 Update dimension_embeddings fixture

After the Phase 0 bug fix, the fixture will correctly embed at each dimension.
Add dimension validation:
```python
# After embedding at each dim, assert actual dimension matches request
actual_dim = len(node_vecs[0])
assert actual_dim == dim, (
    f"Bug not fixed: requested dim {dim} but got {actual_dim}"
)
```

### 1.4 Add per-domain MRR breakdown

Extend `test_dimension_comparison_summary` to report MRR separately for each
content domain (IMAS DD, wiki, code, multilang, abbreviation). This reveals
which content types benefit most from higher dimensions.

### 1.5 Run evaluation and record results

```bash
uv run pytest tests/search/test_search_evaluation.py::TestDimensionComparison \
    -v -s --log-cli-level=INFO -m "graph and slow"
```

**Decision criteria:**
- If MRR gain from 256→1024 is ≥ 0.05 across domains: **use 1024**
  (quantization makes storage equivalent to current 256 unquantized)
- If MRR gain is < 0.03: **keep 256** but still enable quantization for
  memory savings
- If multilang/code domains show disproportionate gain: **use 1024** even
  if IMAS DD gain is marginal (the model's capacity benefits longer,
  more complex texts more than short path+description strings)

Record results in the test output log. The test already logs per-dimension MRR.

---

## Phase 2 — Re-embed All Graph Nodes

**Goal:** Re-generate all embeddings at the chosen dimension with consistent
server-side truncation. Update configuration to match.

**Agent count:** 1 (sequential — writes to graph)

**Depends on:** Phase 0 (bug fix), Phase 1 (dimension decision)

### 2.1 Update configuration

**File:** `pyproject.toml`

Update `[tool.imas-codex.embedding]` section:
```toml
[tool.imas-codex.embedding]
model = "Qwen/Qwen3-Embedding-0.6B"
dimension = <DECIDED_DIM>  # 256, 512, or 1024 based on Phase 1 results
location = "titan"
```

### 2.2 Update embedding text generation (conditional)

**File:** `imas_codex/graph/build_dd.py`

If the decision is to use ≥ 512 dimensions, update `generate_embedding_text()`:
```python
def generate_embedding_text(path, path_info, ids_info=None):
    desc = (path_info.get("description") or "").strip()
    doc = (path_info.get("documentation") or "").strip()
    primary = desc if desc else doc
    if not primary:
        return ""

    dim = get_embedding_dimension()
    if dim >= 512 and doc and desc:
        # At higher dims, the model can encode more semantic nuance
        # Include documentation excerpt for additional context
        doc_excerpt = doc[:200] if len(doc) > 200 else doc
        if doc_excerpt != desc:
            return f"{path}. {primary}. {doc_excerpt}"
    return f"{path}. {primary}"
```

Update the function docstring to remove the "optimized for dim-256" language.

### 2.3 Re-embed IMAS DD nodes

Use the existing reset mechanism — no one-off scripts needed:
```bash
# Step 1: Reset embeddings only (preserves enriched descriptions)
uv run imas-codex imas dd build --reset-to enriched

# Step 2: Rebuild embeddings at new dimension
uv run imas-codex imas dd build
```

This is the correct approach because:
- `--reset-to enriched` clears `embedding`, `embedding_hash`, `embedded_at`
- Sets status back to `enriched` (skips extract + build phases on next run)
- `embedding_hash` includes model name, so hash-based cache is invalidated
- Vector indexes are recreated with the new dimension via `ensure_vector_indexes()`
- The build pipeline is interrupt-safe — rerun continues from where it stopped

### 2.4 Re-embed facility nodes

For each facility with embedded wiki/code/signal nodes, reset and re-embed:
```bash
# Wiki chunks
uv run imas-codex discover wiki <facility> --reset-embeddings

# Code chunks
uv run imas-codex discover code <facility> --reset-embeddings

# Signals (description embeddings)
uv run imas-codex discover signals <facility> --reset-embeddings
```

If `--reset-embeddings` doesn't exist as a CLI flag, implement it:
- Clear `embedding`, `embedded_at` on all nodes of that type for the facility
- The next discovery run will pick up unembedded nodes via `_fetch_unembedded()`

Alternative approach using graph shell:
```cypher
// Clear wiki chunk embeddings for a facility
MATCH (c:WikiChunk)-[:AT_FACILITY]->(:Facility {id: 'tcv'})
WHERE c.embedding IS NOT NULL
SET c.embedding = null, c.embedded_at = null

// Clear code chunk embeddings
MATCH (cc:CodeChunk)-[:AT_FACILITY]->(:Facility {id: 'tcv'})
WHERE cc.embedding IS NOT NULL
SET cc.embedding = null, cc.embedded_at = null

// Clear signal description embeddings
MATCH (s:FacilitySignal {facility_id: 'tcv'})
WHERE s.embedding IS NOT NULL
SET s.embedding = null, s.embedded_at = null
```

Then run the embed workers:
```bash
uv run imas-codex discover embed <facility>
```

### 2.5 Verify embedding consistency

After re-embedding, verify all indexes are at the correct dimension:
```cypher
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
WHERE type = 'VECTOR'
RETURN name, labelsOrTypes, properties,
       options.indexConfig.`vector.dimensions` AS dim
```

All should report `dim = <DECIDED_DIM>`.

### 2.6 Drop and recreate vector indexes

Vector indexes in Neo4j are dimension-locked. If the dimension changes from 256,
existing indexes must be dropped and recreated:

```python
# In ensure_vector_indexes() — add logic to check existing dimension
existing = gc.query("SHOW INDEXES YIELD name, options WHERE type = 'VECTOR'")
for idx in existing:
    current_dim = idx["options"]["indexConfig"]["vector.dimensions"]
    if current_dim != get_embedding_dimension():
        gc.query(f"DROP INDEX {idx['name']}")
        # Will be recreated by ensure_vector_indexes() with new dimension
```

---

## Phase 3 — Migrate to SEARCH Clause

**Goal:** Replace all 27 `db.index.vector.queryNodes()` call sites with Neo4j's
native `SEARCH` clause, enabling in-index pre-filtering and eliminating k oversampling.

**Agent count:** 2 (split by file groups)

**Depends on:** Phase 0 (bug fix); can proceed in parallel with Phase 2

### 3.1 Understand SEARCH clause syntax

Neo4j 2026.01 SEARCH clause replaces the procedure call:

```cypher
-- LEGACY (current):
CALL db.index.vector.queryNodes('index_name', $k, $embedding)
YIELD node, score
WHERE node.facility_id = $facility  -- post-filtering (wasted k budget)

-- SEARCH (new) with in-index pre-filtering:
CALL () {
  SEARCH node:NodeLabel
  USING VECTOR INDEX index_name
  WHERE node.facility_id = $facility  -- filtered BEFORE vector scan
  WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
  ORDER BY score DESC
  LIMIT $k
}
-- Results already filtered — no wasted k budget
```

Key benefits:
- **In-index pre-filtering:** Property filters inside `SEARCH WHERE` are evaluated
  before the ANN scan, not after. No more 5× oversampling.
- **No YIELD:** Results come via `WITH` clause, standard Cypher
- **Composable:** Can chain SEARCH with MATCH, OPTIONAL MATCH, etc.

### 3.2 Migration strategy by priority

#### Priority 1 — High-impact sites with heavy post-filtering

These sites use 3-5× k oversampling because post-filters discard many results.
SEARCH pre-filtering eliminates this waste.

| File | Function | Index | Post-filters → Pre-filters |
|------|----------|-------|---------------------------|
| `tools/graph_search.py` | `search_imas_paths()` L468 | `imas_node_embedding` | `node_category='data'`, `ids IN`, DD version, deprecated |
| `graph/domain_queries.py` | `find_facility_signals()` L155 | `facility_signal_desc_embedding` | `facility`, `diagnostic`, `physics_domain`, `checked` |
| `graph/domain_queries.py` | `find_imas_paths()` L390 | `imas_node_embedding` | `node_category='data'`, `ids`, deprecated |
| `graph/domain_queries.py` | `find_data_nodes()` L564 | `signal_node_desc_embedding` | `facility`, `data_source_name`, `physics_domain`, `path STARTS WITH` |
| `graph/query_builder.py` | `graph_search()` L140 | dynamic | All dynamic WHERE conditions |
| `ingestion/search.py` | `search_code_chunks()` L68 | `code_chunk_embedding` | `facility_id`, `related_ids` |

**Conversion pattern for property filters:**
```python
# BEFORE (oversampled k):
f"""
CALL db.index.vector.queryNodes('{index}', $k, $embedding)
YIELD node AS n, score
WHERE n.facility_id = $facility AND n.physics_domain = $domain
RETURN n.id, score ORDER BY score DESC
"""

# AFTER (in-index pre-filtering):
f"""
CALL () {{
  SEARCH n:{label}
  USING VECTOR INDEX {index}
  WHERE n.facility_id = $facility AND n.physics_domain = $domain
  WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
  ORDER BY score DESC
  LIMIT $k
}}
RETURN n.id, score
"""
```

**Note on relationship filters:** Filters like `NOT (n)-[:DEPRECATED_IN]->(:DDVersion)`
cannot move inside SEARCH WHERE (no relationship traversal in index). These remain as
post-filters outside the CALL block, but property filters can still pre-filter to
greatly reduce the candidate set.

#### Priority 2 — Medium-impact sites with relationship filters

| File | Function | Index | Strategy |
|------|----------|-------|----------|
| `graph/domain_queries.py` | `find_wiki()` L283 | `wiki_chunk_embedding` | Pre-MATCH for facility, text CONTAINS stays post-filter |
| `graph/domain_queries.py` | `find_code()` L442 | `code_chunk_embedding` | Pre-MATCH for facility via CodeFile |
| `discovery/signals/parallel.py` | `get_code_context()` L3508 | `code_chunk_embedding` | Score threshold → pre-filter, facility → pre-MATCH |
| `tools/graph_search.py` | `_search_by_text()` L1566 | `cluster_embedding` | `scope`, `ids_names` → pre-filter |
| `tools/graph_search.py` | `_get_path_relationships()` L1977 | `imas_node_embedding` | `ids <>` exclusion, DD version |

#### Priority 3 — Clean semantic queries (minimal change)

| File | Function | Index | Change |
|------|----------|-------|--------|
| `tools/graph_search.py` | `list_all_imas_ids()` L1331 | `ids_embedding` | Direct syntax swap, no filters |
| `tools/graph_search.py` | `_search_identifiers()` L1766 | `identifier_schema_embedding` | Score threshold only |
| `ids/tools.py` | `search_source_documentation()` L841 | multiple | IMAS: `node_category` pre-filter; wiki/code: syntax swap only |

### 3.3 Build helper function

Create a query builder utility to reduce boilerplate:

**File:** `imas_codex/graph/vector_search.py` (new)

```python
def build_vector_search(
    index: str,
    label: str,
    embedding_property: str = "embedding",
    *,
    where_clauses: list[str] | None = None,
    k: int = 20,
    node_alias: str = "n",
    score_alias: str = "score",
) -> str:
    """Build a SEARCH clause for vector similarity queries.

    Generates Neo4j 2026.01 SEARCH syntax with in-index pre-filtering.
    Property-based WHERE clauses are pushed inside the SEARCH block
    for index-level evaluation. Relationship-based filters must be
    applied outside as post-filters.
    """
    where = ""
    if where_clauses:
        where = "WHERE " + " AND ".join(where_clauses) + "\n  "

    return (
        f"CALL () {{\n"
        f"  SEARCH {node_alias}:{label}\n"
        f"  USING VECTOR INDEX {index}\n"
        f"  {where}"
        f"  WITH {node_alias}, "
        f"vector.similarity.cosine({node_alias}.{embedding_property}, $embedding) "
        f"AS {score_alias}\n"
        f"  ORDER BY {score_alias} DESC\n"
        f"  LIMIT $k\n"
        f"}}\n"
    )
```

### 3.4 Reduce k oversampling

With in-index pre-filtering, the current 3-5× oversampling can be reduced:
- Sites with property-only pre-filters: reduce from `k * 5` to `k * 1.5`
- Sites with relationship post-filters: reduce from `k * 3` to `k * 2`
- Sites with no filtering: keep `k` as-is

### 3.5 Add integration tests

Test that SEARCH-based queries return equivalent results to legacy procedure calls.
Use a small set of known-good queries and verify top-5 results match.

### 3.6 Benchmark performance

Compare query latency before/after SEARCH migration on the most frequently called
queries (signal search, IMAS path search, wiki search).

---

## Phase 4 — Enable Quantization

**Goal:** Enable `vector.quantization.enabled` on all vector indexes for ~4×
memory savings.

**Agent count:** 1

**Depends on:** Phase 2 (re-embedding complete — indexes must be recreated)

### 4.1 Update index creation

**File:** `imas_codex/graph/client.py`

Update `ensure_vector_indexes()` to include quantization option:
```python
gc.query(f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{label}) ON n.{property}
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine',
            `vector.quantization.enabled`: true
        }}
    }}
""")
```

Also update all other index creation sites:
- `imas_codex/graph/build_dd.py` (cluster indexes, line ~4004)
- `imas_codex/discovery/wiki/pipeline.py` (wiki chunk index, line ~1343)
- Any other `CREATE VECTOR INDEX` statements

### 4.2 Verify quantization active

After index creation:
```cypher
SHOW INDEXES YIELD name, type, options
WHERE type = 'VECTOR'
RETURN name, options.indexConfig.`vector.quantization.enabled` AS quantized
```

### 4.3 Validate recall quality

Quantization introduces slight recall degradation (typically < 1%). Run the
benchmark suite before and after to verify acceptable quality:
```bash
uv run pytest tests/search/test_search_evaluation.py::TestSearchBenchmarkMRR -v
```

### 4.4 Measure memory savings

```cypher
CALL db.stats.retrieve("GRAPH COUNTS")
```

Compare index memory usage before and after quantization.

**Expected savings:**
| Dimension | Unquantized | Quantized | Savings |
|-----------|-------------|-----------|---------|
| 256-dim   | ~100 MB     | ~25 MB    | 75%     |
| 512-dim   | ~200 MB     | ~50 MB    | 75%     |
| 1024-dim  | ~400 MB     | ~100 MB   | 75%     |

Note: 1024-dim quantized (100 MB) equals current 256-dim unquantized (100 MB).
This is the key argument for upgrading to full native dimension — zero storage
cost increase with quantization.

---

## Phase 5 — Embedding Text Optimization (Conditional)

**Goal:** If Phase 1 shows ≥ 512-dim is beneficial, remove the dim-256 text
truncation constraints from embedding text generation.

**Agent count:** 1

**Depends on:** Phase 1 (dimension decision), Phase 2 (re-embed)

### 5.1 Update generate_embedding_text()

**File:** `imas_codex/graph/build_dd.py` (line 170)

At higher dimensions, the model can encode richer text without signal dilution.
Include documentation excerpts and keywords:
```python
def generate_embedding_text(path, path_info, ids_info=None):
    desc = (path_info.get("description") or "").strip()
    doc = (path_info.get("documentation") or "").strip()
    keywords = path_info.get("keywords", [])

    primary = desc if desc else doc
    if not primary:
        return ""

    parts = [f"{path}. {primary}"]

    dim = get_embedding_dimension()
    if dim >= 512:
        # Higher dims can encode more semantic nuance without signal dilution
        if doc and doc != desc:
            parts.append(doc[:300])
        if keywords:
            parts.append(", ".join(keywords[:8]))

    return ". ".join(parts)
```

### 5.2 Update TestEmbedTextQuality

**File:** `tests/search/test_search_evaluation.py` (line 1391)

Update the `TestEmbedTextQuality` class to handle dimension-dependent text format.

### 5.3 Re-run MRR evaluation

After embedding text optimization, re-run dimension evaluation to quantify the
improvement from richer embedding text at higher dimensions.

---

## Documentation Updates

| Target | Update Required |
|--------|----------------|
| `AGENTS.md` | Update embedding section: remove "dimension=256" references, document quantization, document SEARCH clause pattern |
| `README.md` | Update embedding configuration example |
| `plans/README.md` | Add this plan entry |
| `pyproject.toml` | Update dimension default (if changed) |
| Docstrings in `client.py`, `encoder.py`, `build_dd.py` | Remove dim-256 optimization language |

---

## Call Sites Reference

### All `db.index.vector.queryNodes()` locations (27 total)

| # | File | Line | Index | Pre-filter Opportunity |
|---|------|------|-------|----------------------|
| 1 | `tools/graph_search.py` | 468 | `imas_node_embedding` | HIGH |
| 2 | `tools/graph_search.py` | 1331 | `ids_embedding` | LOW |
| 3 | `tools/graph_search.py` | 1566 | `cluster_embedding` | HIGH |
| 4 | `tools/graph_search.py` | 1766 | `identifier_schema_embedding` | MODERATE |
| 5 | `tools/graph_search.py` | 1977 | `imas_node_embedding` | MODERATE |
| 6 | `graph/domain_queries.py` | 155 | `facility_signal_desc_embedding` | HIGH |
| 7 | `graph/domain_queries.py` | 283 | `wiki_chunk_embedding` | MODERATE |
| 8 | `graph/domain_queries.py` | 390 | `imas_node_embedding` | HIGH |
| 9 | `graph/domain_queries.py` | 442 | `code_chunk_embedding` | MODERATE |
| 10 | `graph/domain_queries.py` | 564 | `signal_node_desc_embedding` | HIGH |
| 11 | `graph/query_builder.py` | 140 | dynamic | HIGH |
| 12 | `ids/tools.py` | 841 | `imas_node_embedding` | HIGH |
| 13 | `ids/tools.py` | 886 | `wiki_chunk_embedding` | LOW |
| 14 | `ids/tools.py` | 915 | `code_chunk_embedding` | LOW |
| 15 | `ingestion/search.py` | 68 | `code_chunk_embedding` | HIGH |
| 16 | `discovery/signals/parallel.py` | 3508 | `code_chunk_embedding` | HIGH |
| 17-27 | `cli/imas_dd.py` + others | various | various | varies |

### Vector Indexes (all need quantization + dimension update)

| Index | Label | Property | Current Dim |
|-------|-------|----------|-------------|
| `imas_node_embedding` | `IMASNode` | `embedding` | 256 |
| `ids_embedding` | `IDS` | `embedding` | 256 |
| `identifier_schema_embedding` | `IdentifierSchema` | `embedding` | 256 |
| `cluster_label_embedding` | `IMASSemanticCluster` | `label_embedding` | 256 |
| `cluster_description_embedding` | `IMASSemanticCluster` | `description_embedding` | 256 |
| `wiki_chunk_embedding` | `WikiChunk` | `embedding` | 256 |
| `code_chunk_embedding` | `CodeChunk` | `embedding` | 256 |
| `facility_signal_desc_embedding` | `FacilitySignal` | `embedding` | 256 |
| `signal_node_desc_embedding` | `SignalNode` | `embedding` | 256 |
| `wiki_page_desc_embedding` | `WikiPage` | `embedding` | 256 |
| `code_example_desc_embedding` | `CodeExample` | `embedding` | 256 |
| `facility_path_desc_embedding` | `FacilityPath` | `embedding` | 256 |
