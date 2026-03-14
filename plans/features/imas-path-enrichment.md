# Plan: IMAS Path LLM Enrichment

**Goal:** Generate rich, physics-aware `description` fields for all IMASNode paths via LLM, producing embeddings that enable high-quality semantic matching for the IMAS mapping pipeline.

**Justification:** Currently 57% of IMASNode paths have only boilerplate documentation ("Data", "Lower error for…"). The `description` field is NULL for all 61,366 nodes. Embeddings exist for only 31.5% of nodes and are built from the raw `documentation` + auto-generated `embedding_text` which is formulaic. The mapping pipeline needs rich descriptions on both sides (signal + target) for effective semantic pre-filtering.

**Status:** ✅ IMPLEMENTED (2026-03-14)

---

## Phase 1: DD Build Pipeline Extension ✅

**Files:** `imas_codex/graph/build_dd.py`

### 1.1 Add `--enrich` Flag to DD Build ✅

Added `include_enrichment: bool = True` parameter to `build_dd_graph()` (defaults to true per user requirement). Added `--skip-enrichment` CLI flag. Enrichment runs after embeddings (Phase 3.5) and regenerates embeddings for enriched paths.

The enrichment is idempotent: each IMASNode gets an `enrichment_hash` (SHA256 of the concatenated context sent to the LLM + model name). On re-run, skip nodes whose `enrichment_hash` matches — same pattern as `embedding_hash`.

### 1.2 Add Schema Properties ✅

Added to `IMASNode` in `imas_codex/schemas/imas_dd.yaml`:
- `description`: LLM-generated physics-aware description
- `physics_summary`: One-sentence summary for compact display
- `keywords`: LLM-generated searchable keywords (max 5)
- `enrichment_hash`: Hash of context + model for idempotency
- `enrichment_model`: LLM model that generated the description
- `enrichment_source`: 'llm' or 'template'

Run `uv run build-models --force` after schema changes.

### 1.3 Expose DDVersion COCOS in Search Returns ✅

Added `cocos_label_transformation` and `cocos_transformation_expression` to:
- `SearchHit` model in `search_strategy.py`
- `IdsNode` model in `data_model.py`
- Return projections in `search_imas_paths()` and `fetch_imas_paths()`

---

## Phase 2: Enrichment Prompt Design ✅

**Files:** `imas_codex/llm/prompts/imas/enrichment.md` (new)

### 2.1 Prompt Architecture ✅

Created static-first prompt with:
1. Role definition: "IMAS Data Dictionary expert"
2. Physics domain enum (via `{% include "schema/physics-domains.md" %}`)
3. COCOS label transformation semantics
4. Output format guidelines
5. Anti-hallucination rules (don't repeat metadata like units, coordinates)

### 2.2-2.5 Context Gathering ✅

Implemented in `dd_enrichment.py:gather_path_context()`:
- Full parent chain from path ID
- Sibling paths via graph query
- Child summary for STRUCTURE/STRUCT_ARRAY nodes
- IDS-level context (description, physics_domain)
- Unit and coordinate information via graph queries
- Cluster membership for semantic grouping

### 2.3 Context Injection Strategy — Maximizing Tree Semantics

IMAS DD tree hierarchy is the primary source of semantic meaning. For each batch:

1. **Group by IDS section** — e.g., all paths under `pf_active/coil/` together
2. **Inject full parent chain** — `pf_active/coil/element/geometry/rectangle/r` → show all ancestors from IDS root
3. **Inject sibling context** — if enriching `r`, also show `z`, `width`, `height` as siblings
4. **Inject child summary** for STRUCTURE/STRUCT_ARRAY nodes — list immediate children
5. **Inject cluster context** — paths in the same semantic cluster share physics meaning

### 2.4 Scope: All Nodes Including Non-Leaf

Enrich STRUCTURE and STRUCT_ARRAY nodes too — their descriptions become critical for the mapping pipeline's section assignment step. Cost is marginal (~2K additional nodes) and the benefit is large: `pf_active/coil` getting a description like "Array of poloidal field coils with geometry, current, and electrical properties" dramatically helps section assignment.

### 2.5 Skip Boilerplate-Only Paths

For error/validity boilerplate paths (`*_error_index`, `*_error_lower`, `*_error_upper`, `*_validity`, `*_validity_timed`), generate a minimal template description (no LLM call): "Error index/lower/upper bound for {parent_field} in {section}". Mark with `enrichment_source = 'template'` to distinguish from LLM-enriched.

---

## Phase 3: Pydantic Models + Worker ✅

**Files:** `imas_codex/graph/dd_enrichment.py` (new)

### 3.1 Pydantic Response Model ✅

Implemented `IMASPathEnrichmentResult` and `IMASPathEnrichmentBatch` with:
- `path_index`: 1-based index
- `description`: 2-4 sentence physics description
- `physics_summary`: One-sentence summary
- `keywords`: Up to 5 searchable terms
- `physics_domain`: Optional domain override (per user requirement)

### 3.2 Enrichment Worker Function ✅

Implemented `enrich_imas_paths()` with:
1. Query unenriched paths (WHERE description IS NULL)
2. Separate boilerplate paths (template) from LLM paths
3. Batch LLM calls with rich context gathering
4. Graph updates via `_batch_update_enrichments()`
5. Cost and token tracking
6. Physics domain propagation back to graph (per user requirement)

### 3.3 Integration into build_dd_graph() ✅

Added enrichment as Phase 3.5, after embeddings:
```python
if include_enrichment and not dry_run:
    enrichment_stats = enrich_imas_paths(...)
```

After enrichment, regenerates embeddings for enriched paths with the new descriptions.

### 3.4 Update `generate_embedding_text()` to Use Description ✅

Modified to prefer LLM description, include physics_summary, and add keywords:
```python
enriched_desc = path_info.get("description")
if enriched_desc:
    sentences.append(enriched_desc)
    # Also include physics_summary and keywords
else:
    # Fall back to raw documentation
```

---

## Phase 4: Cost and Performance ✅

### 4.1 Estimates

- ~19,000 paths with useful documentation → ~380 LLM calls at batch_size=50
- ~31,000 boilerplate paths → template-only (no LLM cost)
- ~2,000 STRUCTURE/STRUCT_ARRAY nodes → ~40 LLM calls
- Total: ~420 LLM calls × ~$0.01/call (Gemini Flash) = **~$4-5**
- Per-version: only the first version is expensive; subsequent versions only enrich changed paths

### 4.2 Idempotency ✅

Each path gets `enrichment_hash = SHA256(context_text + model_name)[:16]`. On re-run:
- Same context + same model → skip (hash match)
- Changed documentation (new DD version) → re-enrich
- Changed model → re-enrich all

---

## Phase 5: Testing ✅

**Files:** `tests/graph/test_dd_enrichment.py` (new)

### 5.1 Unit Tests ✅

- `TestBoilerplateDetection`: Test pattern matching for error/validity paths
- `TestTemplateDescription`: Test template generation
- `TestEnrichmentHash`: Test hash computation and consistency
- `TestPydanticModels`: Test model validation
- `TestGenerateEmbeddingText`: Test embedding text with/without descriptions

### 5.2 Integration Test

To run:
```bash
imas-codex imas dd build --ids-filter "summary" --current-only
```

Verifies:
- All paths have descriptions
- STRUCTURE nodes have descriptions
- Embeddings regenerated post-enrichment
- Idempotent re-run skips all paths

---

## Implementation Notes

### User Requirements Incorporated

1. **Enrich flag defaults to true** — `include_enrichment=True` in `build_dd_graph()`
2. **Rich progress display** — Uses existing `create_build_monitor()` infrastructure
3. **LiteLLM proxy routing** — Uses `call_llm_structured()` which routes through proxy
4. **Metadata separation** — Prompt instructs LLM to NOT repeat units, coordinates, etc.
5. **Physics domain updates** — `IMASPathEnrichmentResult.physics_domain` propagated to graph

### Files Modified/Created

- `imas_codex/schemas/imas_dd.yaml` — Added enrichment fields to IMASNode
- `imas_codex/graph/build_dd.py` — Added enrichment phase, updated `generate_embedding_text()`
- `imas_codex/graph/dd_enrichment.py` — NEW: Enrichment worker module
- `imas_codex/cli/imas_dd.py` — Added `--skip-enrichment` and `--enrichment-model` flags
- `imas_codex/llm/prompts/imas/enrichment.md` — NEW: Enrichment prompt
- `imas_codex/llm/prompt_loader.py` — Added `imas_enrichment_schema` provider
- `imas_codex/search/search_strategy.py` — Added COCOS fields to SearchHit
- `imas_codex/core/data_model.py` — Added COCOS fields to IdsNode
- `imas_codex/tools/graph_search.py` — Added COCOS to queries and results
- `tests/graph/test_dd_enrichment.py` — NEW: Unit tests

---

## Dependency Chain

```
Phase 1 (schema + build flag) → Phase 2 (prompt) → Phase 3 (worker) → Phase 4 (cost validation) → Phase 5 (tests)
```

All phases completed 2026-03-14.
