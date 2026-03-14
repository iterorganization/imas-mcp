# Plan: IMAS Path LLM Enrichment

**Goal:** Generate rich, physics-aware `description` fields for all IMASNode paths via LLM, producing embeddings that enable high-quality semantic matching for the IMAS mapping pipeline.

**Justification:** Currently 57% of IMASNode paths have only boilerplate documentation ("Data", "Lower error for…"). The `description` field is NULL for all 61,366 nodes. Embeddings exist for only 31.5% of nodes and are built from the raw `documentation` + auto-generated `embedding_text` which is formulaic. The mapping pipeline needs rich descriptions on both sides (signal + target) for effective semantic pre-filtering.

---

## Phase 1: DD Build Pipeline Extension

**Files:** `imas_codex/graph/build_dd.py`

### 1.1 Add `--enrich` Flag to DD Build

Add an `include_enrichment: bool = False` parameter to `build_dd_graph()`. When enabled, run the enrichment step after embeddings are built (Phase 11.5 in the current build sequence).

The enrichment is idempotent: each IMASNode gets an `enrichment_hash` (SHA256 of the concatenated context sent to the LLM + model name). On re-run, skip nodes whose `enrichment_hash` matches — same pattern as `embedding_hash`.

### 1.2 Add Schema Properties

Add to `IMASNode` in `imas_codex/schemas/imas_dd.yaml`:
```yaml
description:
  description: >-
    LLM-generated physics-aware description of this IMAS path.
    Richer than the raw documentation field — explains what the
    quantity measures, its physical significance, and its role
    in the IDS structure.
enrichment_hash:
  description: Hash of context + model for idempotent re-enrichment.
enrichment_model:
  description: LLM model that generated the description.
```

Run `uv run build-models --force` after schema changes.

### 1.3 Expose DDVersion COCOS in Search Returns

Ensure that all MCP tool search/fetch results include the `cocos_label_transformation` and `cocos_transformation_expression` properties when present. Currently these exist on IMASNode but are not surfaced in `GraphSearchTool.search_imas_paths()` or `GraphPathTool.fetch_imas_paths()`. Add them to the return projections.

Verify that `DDVersion.cocos` (integer 11 or 17) is accessible via the `VersionTool` and returned by `get_dd_versions()`.

---

## Phase 2: Enrichment Prompt Design

**Files:** `imas_codex/llm/prompts/imas/enrichment.md` (new)

### 2.1 Prompt Architecture

Follow the static-first caching pattern. System prompt (static):
1. Role definition: "IMAS Data Dictionary expert"
2. Physics domain enum (include via `{% include "schema/physics-domains.md" %}`)
3. COCOS label transformation enum and semantics
4. Output format + schema (Pydantic injection via `schema_needs`)
5. Description guidelines: what makes a good IMAS path description
6. Anti-hallucination rules (don't invent units, don't fabricate paths)

User prompt (dynamic, per batch):
1. IDS-level context: IDS name, IDS description, IDS COCOS convention
2. Tree hierarchy context: parent path, sibling paths, child structure
3. The batch of paths to describe with their raw `documentation`
4. Unit information from `HAS_UNIT` relationships
5. Coordinate spec information from `HAS_COORDINATE`
6. COCOS label if present on that path
7. Cluster membership (which semantic cluster this path belongs to)
8. Identifier schema if applicable

### 2.2 Context Gathering — Use Existing Tool Functions

Use the backing functions from the MCP tools (not the MCP tools themselves) to gather rich context:

| Function | Source | Context Provided |
|----------|--------|-----------------|
| `GraphStructureTool.analyze_imas_structure()` | `imas_codex/tools/graph_search.py` | IDS-level stats, section breakdown |
| `GraphPathTool.fetch_imas_paths()` | `imas_codex/tools/graph_search.py` | Full metadata per path including coordinates, units, identifier schemas |
| `GraphClustersTool.search_imas_clusters()` | `imas_codex/tools/graph_search.py` | Semantic cluster membership for grouping related paths |
| `GraphPathContextTool.get_imas_path_context()` | `imas_codex/tools/graph_search.py` | Cross-IDS relationships, coordinate specs |
| `GraphListTool.list_imas_paths()` | `imas_codex/tools/graph_search.py` | Sibling/parent/child path enumeration |
| `_dd_version_clause()` | `imas_codex/tools/graph_search.py` | DD version scoping |

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

## Phase 3: Pydantic Models + Worker

**Files:** `imas_codex/graph/dd_enrichment.py` (new)

### 3.1 Pydantic Response Model

```python
class IMASPathEnrichmentResult(BaseModel):
    path_index: int  # 1-based
    description: str  # 2-4 sentences
    physics_summary: str  # 1-sentence summary for compact display
    keywords: list[str]  # max 5, for search
    
class IMASPathEnrichmentBatch(BaseModel):
    results: list[IMASPathEnrichmentResult]
```

### 3.2 Enrichment Worker Function

`enrich_imas_paths(gc, version, ids_filter, model, batch_size=50)`:

1. Query unenriched paths for the target version (WHERE description IS NULL)
2. Group by IDS → group by section within IDS
3. For each batch:
   a. Gather context using tool functions (hierarchy, siblings, clusters)
   b. Render prompt
   c. Call LLM via `call_llm_structured()`
   d. Update graph: SET description, enrichment_hash, enrichment_model
4. Track cost and token usage
5. Return stats dict

### 3.3 Integration into build_dd_graph()

After the embeddings phase (step 11), add:
```python
if include_enrichment:
    enriched_count = enrich_imas_paths(
        client, version=latest_version, 
        ids_filter=ids_filter, model=embedding_model,
        batch_size=50,
    )
```

After enrichment completes, regenerate embeddings for enriched paths (the description field should now be included in `generate_embedding_text()` for richer embeddings).

### 3.4 Update `generate_embedding_text()` to Use Description

Modify `generate_embedding_text()` to prefer `description` over raw `documentation` when available:
```python
# If LLM-enriched description exists, use it as primary content
desc = path_info.get("description")
if desc:
    sentences.append(desc)
else:
    doc = path_info.get("documentation", "")
    if doc:
        sentences.append(doc.strip())
```

---

## Phase 4: Cost and Performance

### 4.1 Estimates

- ~19,000 paths with useful documentation → ~380 LLM calls at batch_size=50
- ~31,000 boilerplate paths → template-only (no LLM cost)
- ~2,000 STRUCTURE/STRUCT_ARRAY nodes → ~40 LLM calls
- Total: ~420 LLM calls × ~$0.01/call (Gemini Flash) = **~$4-5**
- Per-version: only the first version is expensive; subsequent versions only enrich changed paths

### 4.2 Idempotency

Each path gets `enrichment_hash = SHA256(context_text + model_name)[:16]`. On re-run:
- Same context + same model → skip (hash match)
- Changed documentation (new DD version) → re-enrich
- Changed model → re-enrich all

---

## Phase 5: Testing

### 5.1 Unit Tests

- Test `generate_embedding_text()` with description present
- Test template generation for boilerplate paths
- Test enrichment hash computation and skip logic
- Test Pydantic model validation

### 5.2 Integration Test

- Build DD for a single small IDS (e.g., `summary`) with enrichment
- Verify all leaf nodes have descriptions
- Verify STRUCTURE nodes have descriptions
- Verify embeddings were regenerated post-enrichment
- Verify idempotent re-run skips all paths

---

## Dependency Chain

```
Phase 1 (schema + build flag) → Phase 2 (prompt) → Phase 3 (worker) → Phase 4 (cost validation) → Phase 5 (tests)
```

Phase 1 can be implemented and tested independently. Phases 2+3 are the core work. Phase 4 is a dry-run validation. Phase 5 follows.
