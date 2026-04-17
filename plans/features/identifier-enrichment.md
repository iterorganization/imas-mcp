# Identifier Node Enrichment & MCP Category Exploitation

## Problem Statement

238 identifier-category IMASNodes (e.g., `equilibrium/time_slice/profiles_2d/grid_type`,
`core_sources/source/identifier`) are invisible to vector search — they have no descriptions,
no embeddings, and `SEARCHABLE_CATEGORIES` excludes them. The 62 `IdentifierSchema` nodes
they reference ARE enriched and embedded, but the IMASNode→Schema link is not discoverable
via natural-language search.

These nodes classify their parent structures (e.g., `grid_type` tells `profiles_2d` which
coordinate system is used). Making them searchable lets users discover WHERE specific
classification schemes apply in the DD hierarchy.

## Approach

1. **Enrich identifier nodes** using a dedicated prompt that combines parent context + schema
   metadata — gated AFTER sibling enrichment completes to leverage their descriptions
2. **Introduce `IDENTIFIER_SEARCHABLE` constant** to avoid polluting clustering constants
3. **Auto-include identifier schema info** in `search_dd_paths` results for identifier hits
4. **Guard search results** against unenriched identifier nodes

## Design Decisions (from rubber-duck review)

### Split EMBEDDABLE_CATEGORIES — don't overload it

`EMBEDDABLE_CATEGORIES` is used by clustering code (`preprocessing.py`, `clusters.py`,
`build_dd.py`) to decide which nodes participate in semantic clustering. Adding `identifier`
there would pull structural classification nodes into physics clusters — clearly wrong.

**Solution**: New constant `IDENTIFIER_CATEGORIES = frozenset({"identifier"})` in
`node_categories.py`. Update `SEARCHABLE_CATEGORIES` to include it. Create a new constant
for embedding eligibility that combines quantity + identifier without affecting clustering:

```python
QUANTITY_CATEGORIES = frozenset({"quantity", "geometry"})
IDENTIFIER_CATEGORIES = frozenset({"identifier"})

# Clustering — quantity/geometry only (unchanged)
EMBEDDABLE_CATEGORIES = QUANTITY_CATEGORIES

# Vector search — quantity + geometry + coordinate + identifier
SEARCHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"} | IDENTIFIER_CATEGORIES

# Nodes that get vector embeddings (superset of EMBEDDABLE for non-clustering use)
VECTOR_INDEXED_CATEGORIES = QUANTITY_CATEGORIES | IDENTIFIER_CATEGORIES
```

Consumers that embed nodes for vector search use `VECTOR_INDEXED_CATEGORIES`.
Consumers that prepare clustering input continue using `EMBEDDABLE_CATEGORIES`.

### Second lifecycle variant for identifier nodes

Main pipeline lifecycle: `built → enriched → refined → embedded`
Identifier lifecycle: `built → enriched → embedded` (skip `refined` — no sibling-readiness barrier)

This is safe because:
- Main claim functions filter on `ENRICHABLE_CATEGORIES` (identifier not in this set)
- Embed worker claims `status='refined'` — identifier nodes at `enriched` won't be claimed
- Aux pipeline handles the full identifier lifecycle independently

### Progress display — don't mix aux into main worker bars

Identifier enrichment/embedding runs in the aux pipeline, not the main workers. Adding
identifier to the main embed progress bar would create a mismatch (238 "pending" until
aux runs). Instead, show identifier progress as a separate aux status line.

### Per-node cost tracking (atomic)

Identifier IMASNodes must persist `enrich_llm_cost` on each node atomically (same as
quantity/geometry nodes). The `_graph_refresh()` query already sums all `IMASNode.enrich_llm_cost`
regardless of category — so identifier costs automatically roll up into the total.

### Search result quality — avoid redundancy

The identifier IMASNode description should answer "what does this field control in its parent
context?" The IdentifierSchema description answers "what is this schema and what are its options?"
These are complementary, not redundant.

In search results, show a **compact schema summary** (schema name + 2-3 exemplar options).
Full option lists belong in `fetch_dd_paths` and `get_dd_identifiers`.

### Guard against unenriched search hits

Once `identifier` enters `SEARCHABLE_CATEGORIES`, text search can surface `built` nodes
(no description, no embedding). Add a `description IS NOT NULL` guard for identifier-category
hits in the search query.

## Implementation Phases

### Phase 1: Category Constants

File: `imas_codex/core/node_categories.py`

- Add `IDENTIFIER_CATEGORIES = frozenset({"identifier"})`
- Add `VECTOR_INDEXED_CATEGORIES = QUANTITY_CATEGORIES | IDENTIFIER_CATEGORIES`
- Update `SEARCHABLE_CATEGORIES = QUANTITY_CATEGORIES | {"coordinate"} | IDENTIFIER_CATEGORIES`
- Leave `EMBEDDABLE_CATEGORIES` unchanged (clustering use only)
- Update `__all__`

Audit all consumers of `EMBEDDABLE_CATEGORIES` — confirm none need `VECTOR_INDEXED_CATEGORIES`:
- `clusters/preprocessing.py:46` — clustering, keep `EMBEDDABLE_CATEGORIES` ✓
- `core/clusters.py:209` — coverage stats, keep `EMBEDDABLE_CATEGORIES` ✓
- `graph/build_dd.py:2237,2253` — embed path selection, **switch to `VECTOR_INDEXED_CATEGORIES`**
- `graph/dd_progress.py` — embed pending, keep `EMBEDDABLE_CATEGORIES` for main worker
- `graph/dd_workers.py` — embed worker, keep `EMBEDDABLE_CATEGORIES` for main worker
- `graph/dd_graph_ops.py` — claim embed, keep `EMBEDDABLE_CATEGORIES` for main worker

### Phase 2: Enrichment Pipeline

File: `imas_codex/graph/dd_identifier_enrichment.py` (extend existing)

New function `enrich_identifier_nodes(client, model, on_progress, on_items)`:
1. Query all 238 identifier-category IMASNodes with `status='built'`
2. For each, gather:
   - Parent node description (enriched by main pipeline)
   - Linked IdentifierSchema description + options
   - IDS context
   - Sibling names (for disambiguation)
3. Batch LLM call with dedicated prompt
4. Persist description + keywords + `enrich_llm_cost` atomically per node
5. Set `status = 'enriched'`

New function `embed_identifier_nodes(client, on_progress)`:
1. Query identifier nodes with `status='enriched'`
2. Generate embeddings using same text format as quantity nodes
3. Set `status = 'embedded'` with embedding

### Phase 3: Prompt Template

File: `imas_codex/llm/prompts/imas/identifier_node_enrichment.md`

Focus on: "What does this identifier classify in its parent structure?"
Include: parent description, schema description, exemplar valid values, IDS context.
Differ from `identifier_enrichment.md` (which enriches schemas, not path nodes).

### Phase 4: Worker Wiring

File: `imas_codex/graph/dd_workers.py`

In `_run_aux_enrichment()` — add as THIRD step (after IdentifierSchema + IDS):
```python
# Step 3: Identifier IMASNode enrichment (runs last — benefits from sibling descriptions)
ident_node_stats = enrich_identifier_nodes(client, model=model, ...)
```

In `_run_aux_embedding()` — add after existing aux embedding:
```python
# Step 3: Embed identifier IMASNodes
embed_identifier_nodes(client, ...)
```

Update progress stream items with styled labels (e.g., yellow for identifier nodes).

### Phase 5: Search Query Enhancement

File: `imas_codex/tools/graph_search.py`

For identifier-category search hits, extend the result query to fetch:
- Linked IdentifierSchema name and description
- Top 3 option values (name + description)

Add `description IS NOT NULL` guard for identifier-category nodes in search queries.

File: `imas_codex/llm/search_formatters.py`

When formatting identifier hits in `search_dd_paths` results, append:
```
  🏷️ Identifier Schema: grid_type
     Options: rectangular (1), inverse (2), ...
```

### Phase 6: Testing

File: `tests/core/test_node_categories.py` — update for new constants
File: `tests/graph/test_identifier_enrichment.py` — new:
- Test enrichment function with mock LLM response
- Test category set membership (identifier in SEARCHABLE/VECTOR_INDEXED, not in EMBEDDABLE)
- Test search guard excludes unenriched identifier nodes
- Test formatter includes schema info for identifier hits

File: `tests/graph/test_dd_progress.py` — add:
- Test non-enrichable categories (error/metadata/structural) excluded from all progress counts

### Phase 7: Integration Validation

- `imas dd build --limit 5` with identifier enrichment
- Verify `search_dd_paths("grid type")` returns identifier nodes with schema context
- Verify costs roll up in DDVersion metadata
- Verify progress display shows identifier aux progress correctly

## Cost Estimate

- 238 identifier nodes × ~$0.002/node ≈ $0.50
- Embedding: negligible (local server)
- No impact on existing enrichment/refinement costs

## Files to Modify

| File | Changes |
|------|---------|
| `imas_codex/core/node_categories.py` | Add `IDENTIFIER_CATEGORIES`, `VECTOR_INDEXED_CATEGORIES`; update `SEARCHABLE` |
| `imas_codex/graph/dd_identifier_enrichment.py` | Add `enrich_identifier_nodes()` + `embed_identifier_nodes()` |
| `imas_codex/llm/prompts/imas/identifier_node_enrichment.md` | New prompt template |
| `imas_codex/graph/dd_workers.py` | Wire identifier enrichment/embedding into aux steps |
| `imas_codex/graph/build_dd.py` | Switch embed path selection to `VECTOR_INDEXED_CATEGORIES` |
| `imas_codex/tools/graph_search.py` | Extend search query for identifier schema context; add status guard |
| `imas_codex/llm/search_formatters.py` | Format identifier hits with schema summary |
| `tests/core/test_node_categories.py` | Update for new constants |
| `tests/graph/test_identifier_enrichment.py` | New test file |
| `tests/graph/test_dd_progress.py` | Add non-enrichable exclusion test |

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update node_categories section with `IDENTIFIER_CATEGORIES`, `VECTOR_INDEXED_CATEGORIES` |
| `plans/README.md` | Add this plan |
