# IdentifierSchema & IDS Node Enrichment

## Status: Ready for Implementation

## Problem Statement

Two node types in the IMAS DD graph are fully defined in the schema and populated with structural data, but lack LLM-generated descriptions and vector embeddings. This creates blind spots in semantic search and degrades the quality of dynamically injected context in the mapping pipeline.

### IdentifierSchema (62 nodes, 607 total options)

**Current state**: Nodes exist with `{id, name, options: JSON[{index, name}], option_count, field_count, source}`. The enrichment and embedding pipeline is **fully scaffolded** (`dd_identifier_enrichment.py`, `identifier_enrichment.md` prompt, `identifier_schema_embedding` vector index) but has **never been executed** — zero nodes have `enriched_description` or `embedding`.

**Data gap**: The DD XML contains per-option `description` (100% coverage, 607/607) and `units` (11%, 68/607) and schema-level `header` text (100%, 63/63) via `imas_data_dictionaries.get_identifier_xml()`. We currently store only `{index, name}` per option — the full `{index, name, description, units}` is available but discarded during `_collect_identifier_schemas()`.

### IDS (87 nodes)

**Current state**: Nodes exist with `{id, name, description, physics_domain, path_count, leaf_count, lifecycle_status, ids_type}`. The `description` field contains the raw DD documentation (100% coverage), but there are **no enrichment fields** (`enriched_description`, `embedding`, `keywords`) in the schema or on the nodes. IDS nodes cannot participate in semantic search.

## Evidence Summary

### Graph State (queried 2026-03-15)

| Node Type | Count | Has Description | Has Enrichment | Has Embedding |
|-----------|-------|----------------|----------------|---------------|
| IdentifierSchema | 62 | 0 (0%) | 0 (0%) | 0 (0%) |
| IDS | 87 | 87 (100%) | 0 (0%) | 0 (0%) |
| IMASNode | ~90K | ~90K (enriched) | ~90K | ~60K (current ver) |

### IdentifierSchema Options — Data Left on the Floor

Current `options` JSON per node:
```json
[{"index": 0, "name": "unspecified"}, {"index": 1, "name": "x"}]
```

Available from `get_identifier_xml()`:
```json
[{"index": 0, "name": "unspecified", "description": "unspecified", "units": "m"},
 {"index": 1, "name": "x", "description": "First cartesian coordinate in the horizontal plane", "units": "m"}]
```

### Mapping Pipeline Identifier Usage

`_format_identifier_schemas()` in `mapping.py` is already called in both `map_signals` and `discover_assembly`. It renders per-option `{index, name, description}` for the LLM. However, it receives options from `fetch_imas_paths()` → `GraphPathTool` which reconstructs `IdentifierOption(name, index, description)` — but `description` is always empty because we never stored it.

### IDS with Heaviest Identifier Usage

| IDS | Identifier Fields | Distinct Schemas |
|-----|-------------------|-----------------|
| equilibrium | 14 | 10 |
| spectrometer_x_ray_crystal | 13 | 6 |
| distributions | 12 | 11 |
| wall | 12 | 10 |
| plasma_sources | 11 | 10 |

These IDS gain the most from rich identifier context during mapping.

---

## Implementation Plan

### Phase 1: Capture Per-Option Descriptions from XML (build pipeline fix)

**Goal**: Include `description` and `units` in the `options` JSON stored on IdentifierSchema nodes.

**Files to modify**:
- `imas_codex/graph/build_dd.py` → `_collect_identifier_schemas()`

**Design**: Replace the current path-metadata-based option extraction with direct XML parsing via `imas_data_dictionaries.get_identifier_xml()`. The current approach extracts options from `child.identifier_enum.__members__` which yields only `{name: int_value}`. The XML has `{name, description, units}` per `<int>` element plus a `<header>` text per schema.

**Implementation**:

```python
def _collect_identifier_schemas(paths: dict[str, dict]) -> dict[str, dict]:
    """Collect IdentifierSchema data with full option metadata from DD XML."""
    from imas_data_dictionaries import dd_identifiers, get_identifier_xml
    import xml.etree.ElementTree as ET

    # Enumerate schemas referenced by extracted paths
    referenced = {}
    for _path, info in paths.items():
        enum_name = info.get("identifier_enum_name")
        if enum_name:
            referenced.setdefault(enum_name, 0)
            referenced[enum_name] += 1

    schemas: dict[str, dict] = {}
    available = set(dd_identifiers())

    for enum_name, field_count in referenced.items():
        if enum_name not in available:
            continue
        xml_bytes = get_identifier_xml(enum_name)
        root = ET.fromstring(xml_bytes)

        # Extract header as schema description
        header = root.find("header")
        description = header.text.strip() if header is not None and header.text else ""

        # Extract per-option metadata
        options = []
        for child in root:
            if child.tag == "int" and child.text:
                options.append({
                    "index": int(child.text.strip()),
                    "name": child.attrib.get("name", ""),
                    "description": child.attrib.get("description", ""),
                    "units": child.attrib.get("units", ""),
                })
        options.sort(key=lambda o: o["index"])

        schemas[enum_name] = {
            "id": enum_name,
            "name": enum_name,
            "description": description,
            "options": json.dumps(options),
            "option_count": len(options),
            "field_count": field_count,
            "source": f"utilities/{enum_name}.xml",
        }

    return schemas
```

**Node creation update**: `_create_identifier_schema_nodes()` already stores `description` — no change needed since we're now populating it.

**Impact**: After `imas dd build --reset-to extracted`:
- 607 options gain `description` field (was empty)
- 68 options gain `units` field (was absent)
- 63 schemas gain `description` from XML `<header>`
- `_format_identifier_schemas()` in mapping prompts immediately gets richer context with no code changes

### Phase 2: Run Existing Enrichment + Embedding Pipeline

**Goal**: Execute the already-built `enrich_identifier_schemas()` and `embed_identifier_schemas()` functions that have never run.

**Root cause analysis**: The enrichment/embedding pipeline in `phase_enrich()` and `phase_embed()` already calls these functions. The reason they haven't produced results is that `enrich_identifier_schemas()` queries for `WHERE s.enriched_description IS NULL` — all 62 schemas match. The pipeline was likely never run end-to-end after the IdentifierSchema infrastructure was added.

**Action**: Run `imas-codex imas dd build --reset-to built` to trigger re-enrichment. This will:
1. Call `enrich_identifier_schemas()` with the `imas/identifier_enrichment.md` prompt
2. Generate `enriched_description` + `keywords` for all 62 schemas
3. Call `embed_identifier_schemas()` to create embeddings in `identifier_schema_embedding` vector index
4. The enrichment prompt should be updated to include the per-option descriptions we now capture (Phase 1)

**Prompt improvement**: Update `imas/identifier_enrichment.md` to pass `description` per-option (not just names). The current prompt builds `options_preview: "unspecified, x, y, z"` — it should include the physics descriptions:

```
Options:
- 0: unspecified — unspecified [m]
- 1: x — First cartesian coordinate in the horizontal plane [m]
- 2: y — Second cartesian coordinate in the horizontal plane [m]
```

This gives the LLM enough information to produce a high-quality schema-level description without hallucinating.

**Enrichment prompt context design for IdentifierSchema**:

The LLM needs:
1. Schema name and header description (from XML)
2. Full option list with per-option descriptions and units
3. Field count (how many DD paths reference this schema)
4. Source XML filename

The LLM should produce:
1. `description`: 2-4 sentence physics-aware explanation of what this enumeration controls — connecting the enum concept to plasma physics workflows
2. `keywords`: Up to 5 terms for semantic search discovery

**Cost estimate**: 62 schemas × ~200 input tokens × ~100 output tokens ≈ 18K tokens total. At $3/M tokens = ~$0.05. Negligible.

### Phase 3: IDS Node Enrichment + Embedding (schema extension)

**Goal**: Add LLM-enriched descriptions and embeddings to IDS nodes, enabling semantic search at the IDS level.

**Value proposition**:
- **Mapping pipeline**: The section_assignment prompt shows the full IDS subtree but never explains *what the IDS is for*. An LLM-enriched description of `equilibrium` would explain "contains MHD equilibrium reconstruction results including flux surfaces, safety factor profiles, and global quantities like plasma current and stored energy" — dramatically better context for section assignment.
- **MCP tool `get_imas_overview`**: Currently returns raw DD descriptions. Enriched descriptions would be more discoverable via semantic search.
- **IDS recommendation**: When an agent asks "where should I store electron density profile data?", embedding search across IDS nodes could return `core_profiles` with high confidence.

**Schema changes** (`imas_codex/schemas/imas_dd.yaml`):

Add to the `IDS` class:
```yaml
      enriched_description:
        description: >-
          LLM-generated physics-aware description of this IDS.
          Explains what physics the IDS captures, typical data sources,
          and relationships to other IDSs.
      keywords:
        description: LLM-generated searchable keywords
        multivalued: true
      enrichment_hash:
        description: Hash for enrichment idempotency
      enrichment_source:
        description: Source of enrichment (llm)
      embedding:
        description: Embedding vector for semantic search
        multivalued: true
        range: float
        annotations:
          vector_index_name: ids_embedding
      embedding_hash:
        description: Hash of text used for embedding
```

**Enrichment prompt design for IDS** (`imas/ids_enrichment.md`):

The IDS node has a unique enrichment challenge: it represents the *purpose and scope* of an entire data structure, not a specific field. The LLM needs:

1. **IDS name and raw description** (from DD documentation)
2. **Physics domain** (already assigned)
3. **Key structural sections** — the top-level struct-arrays (e.g., `equilibrium/time_slice`, `equilibrium/vacuum_toroidal_field`). These are the first 2 levels of the IDS tree, obtainable via `fetch_imas_subtree(ids_name, leaf_only=False, max_paths=30)`.
4. **Identifier schemas used** — which enumerated options this IDS employs. Query: `MATCH (p:IMASNode {ids: $ids})-[:HAS_IDENTIFIER_SCHEMA]->(s) RETURN DISTINCT s.name, s.enriched_description`
5. **Related IDS** — IDS that share physics domains or semantic clusters. Query semantic clusters spanning this IDS.
6. **Data cardinality** — path_count, leaf_count, max_depth (already on node)

The LLM should produce:
1. `description`: 3-5 sentence physics-aware overview. What physical phenomena does this IDS describe? What measurement systems or simulations produce this data? How does it relate to other IDSs in the physics workflow?
2. `keywords`: Up to 8 terms covering the physics domains, measurement types, and analysis methods associated with this IDS.

**Implementation**: Create `imas_codex/graph/dd_ids_enrichment.py` following the same pattern as `dd_identifier_enrichment.py`:
- Query all IDS nodes
- Gather context (subtree sections, identifier schemas, related IDS)
- Batch LLM calls with `call_llm_structured()`
- Hash-based idempotency
- Update graph with `enriched_description`, `keywords`, `enrichment_hash`

**Embedding**: After enrichment, generate embeddings using the same `Encoder` pattern. The embedding text should be: `"{ids_name}: {enriched_description} Keywords: {keywords}"`.

**Integration into `phase_enrich()` and `phase_embed()`**: Add calls after identifier enrichment, matching the existing pattern.

**Cost estimate**: 87 IDS × ~500 input tokens × ~200 output tokens ≈ 60K tokens. ~$0.20. Negligible.

### Phase 4: Surface Identifiers in Mapping Pipeline

**Goal**: Inject richer identifier context into mapping prompts, and use IDS descriptions for better section assignment.

#### 4a. Enrich mapping prompts with per-option descriptions

`_format_identifier_schemas()` already renders identifier schemas when `fetch_imas_fields` returns them. After Phase 1, the `description` field on each option will be populated, so this works automatically. No code changes needed — data quality improvement flows through the existing pipeline.

**Before** (current prompt output):
```
- **equilibrium/time_slice/profiles_1d/grid/identifier/index** (schema: coordinate_identifier)
  Valid values:
    - 0: unspecified
    - 1: x
    - 11: rho_tor
```

**After** (with Phase 1 data):
```
- **equilibrium/time_slice/profiles_1d/grid/identifier/index** (schema: coordinate_identifier)
  Translation table for coordinate_identifier_definitions.
  Valid values:
    - 0: unspecified — unspecified
    - 1: x — First cartesian coordinate in the horizontal plane
    - 11: rho_tor — The square root of the toroidal flux [m]
```

#### 4b. Add IDS description to section_assignment prompt

The `section_assignment.md` prompt currently receives `imas_subtree` (structural paths) and `signal_sources` but no IDS-level summary. After Phase 3, inject the IDS enriched description:

**Files to modify**:
- `imas_codex/llm/prompts/mapping/section_assignment.md` — add `{{ ids_description }}` block
- `imas_codex/ids/mapping.py` → `assign_sections()` — fetch IDS enriched description and pass to prompt

```python
# In gather_context() or assign_sections():
ids_info = gc.query("""
    MATCH (i:IDS {id: $ids_name})
    RETURN i.enriched_description AS desc, i.keywords AS keywords
""", ids_name=ids_name)
```

#### 4c. Identifier-aware assembly code generation

The assembly prompt already receives `{{ identifier_schemas }}` with valid values. After Phase 1, the LLM will see the full option descriptions, enabling it to:
- Select the correct identifier index for `coordinate_identifier` fields (e.g., `11` for `rho_tor` vs `12` for `rho_tor_norm`)
- Set material identifiers in wall/blanket assembly code
- Populate species reference identifiers in transport codes

No additional code changes needed — the data quality improvement is the implementation.

#### 4d. Identifier context in signal_mapping prompt (already present)

The `signal_mapping.md` prompt already has a `{{ identifier_schemas }}` section. After Phase 1, this section will include per-option descriptions, helping the LLM match source signal semantics to the correct identifier values.

### Phase 5: Enhance MCP Tools

#### 5a. `get_imas_identifiers` — already exists

The `GraphIdentifierTool` already supports listing and semantic search over IdentifierSchema nodes. After Phase 2 (enrichment + embedding), the semantic search path (`identifier_schema_embedding` vector index) will become functional.

**No code changes needed** — the tool is already built and waiting for data.

#### 5b. `get_imas_overview` — enhance with IDS embeddings

After Phase 3, `get_imas_overview` could use `ids_embedding` for IDS recommendation queries like "which IDS should I use for storing magnetic equilibrium data?".

**Files to modify**:
- `imas_codex/tools/graph_search.py` → `GraphOverviewTool.get_imas_overview()` — add optional `query` parameter semantic search path using `ids_embedding` vector index

#### 5c. `fetch_imas_paths` — enhance identifier schema rendering

Currently returns `identifier_schema_name`, `identifier_schema_description`, `identifier_schema_options`. After Phase 1, `identifier_schema_description` will be populated (from XML header), and options will include per-option descriptions.

**No code changes needed** — enriched data flows through existing API.

---

## Execution Order

| Phase | Scope | Effort | Dependencies | Cost |
|-------|-------|--------|-------------|------|
| 1 | Build pipeline: capture option descriptions | Small | None | $0 |
| 2 | Run enrichment + embedding | Small | Phase 1 | ~$0.05 |
| 3 | IDS schema + enrichment + embedding | Medium | Phase 2 (for IDS→identifier context) | ~$0.20 |
| 4 | Mapping pipeline improvements | Small | Phases 1-3 | $0 |
| 5 | MCP tool enhancements | Small | Phases 2-3 | $0 |

Phases 1-2 can be executed immediately. Phase 3 requires schema changes + model rebuild. Phases 4-5 are incremental improvements that flow naturally from the data quality gains.

## Risk Assessment

- **Low risk**: All proposed changes are additive. No existing schema fields are modified or removed.
- **Backward compatible**: The `options` JSON format change (adding `description` and `units` keys) is purely additive — existing consumers that read `index` and `name` continue to work.
- **Idempotent**: The enrichment pipeline uses hash-based caching — re-running is safe and only processes changed content.
- **Cost**: Total LLM cost for all enrichment is <$0.30. Embedding cost is negligible (local model).
