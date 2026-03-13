# Mapping Pipeline IMAS Context Enrichment

**Depends on: `imas-mcp-gap-closure.md` Phase 1 (shared tool enhancements) and
Phase 6 (ids/tools.py consolidation)**

**Priority: HIGH — directly impacts LLM mapping output quality**

Enrich the IMAS context injected into each LLM call of the `discover imas map`
pipeline so the LLM makes better-informed mapping decisions using graph data
that already exists but is not currently surfaced.

---

## Current State: What Each LLM Step Receives

### Step 1 — `assign_sections` (section_assignment.md)

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_sources` | `query_signal_sources()` | SignalSource nodes: id, group_key, description, physics_domain, rep_unit, rep_cocos, sample_accessors, existing MAPS_TO_IMAS |
| `imas_subtree` | `fetch_imas_subtree()` | IMASNode flat list: id, name, data_type, node_type, documentation, units |
| `semantic_results` | `search_imas_semantic()` | Vector-searched IMASNode: id, documentation, data_type, node_type, units, score |

**What the LLM does NOT see:**
- Cluster membership — which paths are semantically grouped
- Physics domain labels on individual paths
- Identifier schemas on struct-array sections (e.g., geometry types, coil types)
- DD version scoping — gets all paths including deprecated ones

### Step 2 — `map_signals` (signal_mapping.md) — per section

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_source_detail` | (formatted from groups) | Source metadata: description, physics_domain, units, sign_convention, cocos, members, accessors, existing mappings |
| `imas_fields` | `fetch_imas_fields()` or `fetch_imas_subtree(leaf_only=True)` | Path, data_type, units, documentation, ndim, physics_domain, cluster_labels, coordinates |
| `cocos_paths` | `get_sign_flip_paths()` | COCOS sign-flip paths for this IDS |
| `source_cocos` | (built from rep_cocos) | Per-source COCOS context |
| `existing_mappings` | `search_existing_mappings()` | Existing IMASMapping + POPULATES + MAPS_TO_IMAS state |
| `code_references` | `fetch_source_code_refs()` | Code snippets showing how signal is read |
| `unit_analysis` | `analyze_units()` | Pint-based unit compatibility checks |

**What the LLM does NOT see:**
- Identifier schema valid values for typed fields (geometry_type, grid_type, etc.)
- Version change history (sign convention changes, unit changes, renames)
- Cross-facility mapping precedent (what other facilities mapped to these paths)
- DD version scoping

### Step 3 — `discover_assembly` (assembly.md) — per section

| Context variable | Source function | Data provided |
|-----------------|----------------|---------------|
| `signal_mappings` | (formatted from Step 2) | Mapped source→target pairs with transforms |
| `imas_section_structure` | `fetch_imas_subtree()` | Section subtree paths |
| `source_metadata` | (formatted from groups) | Basic source info |

**What the LLM does NOT see:**
- Coordinate spec details — dimensionality, axis semantics, shared coordinates
- Identifier schema options — enumeration values needed by assembly code
- Array sizing constraints from coordinate specs

### Step 4 — `validate_mappings` — programmatic, no LLM

Uses `check_imas_paths()` and `get_sign_flip_paths()`. Does not DD-version-scope
its path existence checks.

---

## Gap Assessment

### HIGH impact — LLM currently lacks information that exists in the graph

| # | Gap | Affects step | Graph data available | Expected quality improvement |
|---|-----|-------------|---------------------|------------------------------|
| 1 | **DD version scoping** | All steps | `DDVersion`, `INTRODUCED_IN`, `DEPRECATED_IN` | Eliminates ghost paths from old DD versions that confuse section assignment and produce mappings to deprecated paths. Reduces escalations. |
| 2 | **Identifier schemas for typed fields** | Steps 2, 3 | `IdentifierSchema` via `HAS_IDENTIFIER_SCHEMA` | Fields like `geometry_type`, `grid_type`, `coordinate_system_type` have enumerated valid values. Without them the LLM guesses or omits these fields entirely. Assembly code also needs valid values for initialization. |
| 3 | **Version change history** | Step 2 | `IMASNodeChange` via `FOR_IMAS_PATH` | Fields with historical `sign_convention`, `coordinate_convention`, or `units` changes need version-aware transforms. Currently the LLM only sees COCOS labels — not whether a field's convention changed between DD versions. |
| 4 | **Cluster context in section assignment** | Step 1 | `IMASSemanticCluster` via `IN_CLUSTER` on subtree paths | The LLM sees a flat path list and has to infer which paths form physics-coherent groups. Cluster labels (e.g., "boundary geometry", "MHD stability") directly indicate section purpose, improving assignment accuracy for ambiguous sources. |

### MEDIUM impact — additional context improves quality but is not critical

| # | Gap | Affects step | Graph data available | Expected quality improvement |
|---|-----|-------------|---------------------|------------------------------|
| 5 | **Coordinate specs in assembly** | Step 3 | `IMASCoordinateSpec` via `HAS_COORDINATE` | Assembly patterns depend on understanding dimensions (time, space, channel). Coordinate spec data (axis semantics, sizing) directly informs `array_per_node` vs `concatenate` vs `matrix_assembly` selection. |
| 6 | **Cross-facility mapping precedent** | Steps 1, 2 | `MAPS_TO_IMAS` from other facilities' SignalSources | When another facility has already mapped similar signals to an IMAS section, this provides strong precedent for correct assignment. Currently only same-facility existing mappings are shown. |
| 7 | **Cross-IDS path context** | Step 2 | `get_imas_path_context` shared tool (Phase 3 of gap closure plan) | Shows how a target field connects to related fields in other IDS via shared clusters/coordinates/units. Helps the LLM understand dimensional conventions. |

---

## Relationship to `imas-mcp-gap-closure.md`

The gap closure plan addresses the **infrastructure** for these enrichments:

| Gap closure phase | What it provides | What this plan adds |
|-------------------|-----------------|---------------------|
| Phase 1.1 — DD version filter | `_dd_version_clause()` + parameter on all shared tools | Thread `dd_version` through `gather_context()` and all `ids/tools.py` calls |
| Phase 1.2 — Facility cross-refs | `facility` param on `GraphSearchTool` | Use cross-facility mappings as precedent context in prompts |
| Phase 1.3 — Version context | `include_version_context` on search | Inject `IMASNodeChange` data into `signal_mapping.md` prompt |
| Phase 1.4 — Raw dict return | `as_dicts()` on result models | Consumed by Phase 6 delegation in `ids/tools.py` |
| Phase 3 — `get_imas_path_context` | Cross-IDS relationship traversal | Optionally inject into signal_mapping prompt |
| Phase 6 — Consolidation | `ids/tools.py` delegates to shared tools | **Prerequisite** — after delegation, enriched data flows automatically |

**The gap closure plan consolidates the tools. This plan injects the richer data
they provide into the mapping prompts.** Without this plan, consolidation improves
code quality but does not improve mapping output quality.

---

## Implementation

All changes below assume Phase 1 and Phase 6 of the gap closure plan are complete.
The shared tools have been enhanced and `ids/tools.py` delegates to them.

### E1: Thread DD version through the pipeline

**Files:** `ids/mapping.py`, `ids/tools.py`

Pass `dd_version` from `generate_mapping()` through `gather_context()` and into
every IMAS query:

```python
def gather_context(facility, ids_name, *, gc, dd_version=None):
    groups = query_signal_sources(facility, gc=gc)
    subtree = fetch_imas_subtree(ids_name, gc=gc, dd_version=dd_version)
    semantic = search_imas_semantic(
        f"{facility} {ids_name}", ids_name, gc=gc, k=10, dd_version=dd_version
    )
    existing = search_existing_mappings(facility, ids_name, gc=gc)
    cocos_paths = get_sign_flip_paths(ids_name)
    return {
        "groups": groups,
        "subtree": subtree,
        "semantic": semantic,
        "existing": existing,
        "cocos_paths": cocos_paths,
        "dd_version": dd_version,
    }
```

Similarly thread into `fetch_imas_fields()` and `check_imas_paths()` calls
in `map_signals()` and `validate_mappings()`.

After Phase 6, these functions delegate to shared tools that already have
`dd_version` parameters (Phase 1.1). The `dd_version` just needs to be passed
through.

### E2: Inject cluster context into section assignment

**Files:** `ids/tools.py` (new function), `ids/mapping.py`, `llm/prompts/mapping/section_assignment.md`

Add a function to fetch cluster groupings for an IDS subtree:

```python
def fetch_section_clusters(ids_name: str, *, gc: GraphClient) -> list[dict]:
    """Return semantic clusters covering sections of this IDS.

    Groups IMAS paths by cluster membership, showing which paths share
    physics-coherent groupings.
    """
    cypher = """
        MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        WHERE p.ids = $ids_name
          AND p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        RETURN c.label AS cluster_label, c.description AS cluster_description,
               c.scope AS scope,
               collect(DISTINCT p.id) AS section_paths
        ORDER BY size(collect(DISTINCT p.id)) DESC
    """
    return gc.query(cypher, ids_name=ids_name)
```

Add to `gather_context()`:
```python
clusters = fetch_section_clusters(ids_name, gc=gc)
```

Format and inject into section assignment prompt:

```markdown
### Section Clusters

These semantic clusters group related IDS sections by physics concept:

{{ section_clusters }}
```

### E3: Inject identifier schemas into signal mapping

**Files:** `ids/tools.py` (new function), `ids/mapping.py`, `llm/prompts/mapping/signal_mapping.md`

When `fetch_imas_fields()` returns paths that have `HAS_IDENTIFIER_SCHEMA`
connections, fetch the schema options:

```python
def fetch_field_identifier_schemas(
    paths: list[str], *, gc: GraphClient
) -> dict[str, dict]:
    """Return identifier schemas for fields that have them.

    Returns {imas_path: {name, description, options: [{index, name, description}]}}.
    """
    cypher = """
        UNWIND $paths AS pid
        MATCH (p:IMASNode {id: pid})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
        RETURN p.id AS path, s.name AS schema_name,
               s.description AS schema_description, s.options AS options
    """
    rows = gc.query(cypher, paths=paths)
    result = {}
    for r in rows:
        options = []
        if r["options"]:
            try:
                import json
                options = json.loads(r["options"])
            except (json.JSONDecodeError, TypeError):
                pass
        result[r["path"]] = {
            "name": r["schema_name"],
            "description": r["schema_description"],
            "options": options,
        }
    return result
```

Inject into `map_signals()` after fetching fields:
```python
field_paths = [f["id"] for f in (subtree_fields or fields)]
identifier_schemas = fetch_field_identifier_schemas(field_paths, gc=gc)
```

Add to `signal_mapping.md` prompt:

```markdown
### Identifier Schemas

These target fields have enumerated valid values. Use these exact values
when populating identifier/type fields:

{{ identifier_schemas }}
```

Also inject into `assembly.md` prompt for assembly code generation.

### E4: Inject version change history into signal mapping

**Files:** `ids/mapping.py`, `llm/prompts/mapping/signal_mapping.md`

After Phase 1.3 of the gap closure plan, the shared tools support
`include_version_context=True`. Use this when the pipeline has a target
`dd_version`:

```python
# In map_signals(), after fetching fields:
version_context = {}
if context.get("dd_version"):
    from imas_codex.llm.search_tools import _get_version_context
    # After Phase 6 consolidation, use shared tool instead
    field_paths = [f["id"] for f in (subtree_fields or fields)]
    version_context = _get_version_context(gc, field_paths)
```

Add to `signal_mapping.md`:

```markdown
### Version History

Notable changes to target fields across DD versions. Check whether your
target DD version is before or after these changes:

{{ version_context }}
```

This is especially valuable for sign convention changes — the LLM can
see that a field's sign convention was standardized in DD 3.39 and
generate the correct transform for the target version.

### E5: Inject coordinate specs into assembly prompt

**Files:** `ids/tools.py` (new function), `ids/mapping.py`, `llm/prompts/mapping/assembly.md`

```python
def fetch_section_coordinate_specs(
    ids_name: str, section_path: str, *, gc: GraphClient
) -> list[dict]:
    """Return coordinate specs for fields in a section.

    Shows which fields share coordinates and their dimensional structure.
    """
    prefix = section_path if section_path.startswith(ids_name) else f"{ids_name}/{section_path}"
    cypher = """
        MATCH (p:IMASNode)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        WHERE p.id STARTS WITH $prefix
        RETURN coord.id AS coordinate_path,
               coord.name AS coordinate_name,
               collect(DISTINCT p.id) AS fields_using,
               count(DISTINCT p) AS field_count
        ORDER BY field_count DESC
    """
    return gc.query(cypher, prefix=prefix + "/")
```

Add to `assembly.md`:

```markdown
### Coordinate Specifications

These coordinate axes are used by fields in this section. Fields sharing
a coordinate should share that array dimension:

{{ coordinate_specs }}
```

### E6: Cross-facility mapping precedent (optional)

**Files:** `ids/tools.py` (new function), `ids/mapping.py`, `llm/prompts/mapping/section_assignment.md`

```python
def fetch_cross_facility_mappings(
    ids_name: str, exclude_facility: str, *, gc: GraphClient
) -> list[dict]:
    """Return mappings from other facilities to this IDS.

    Provides precedent for correct section assignment.
    """
    cypher = """
        MATCH (m:IMASMapping)-[:POPULATES]->(ip:IMASNode)
        WHERE m.ids_name = $ids_name
          AND m.facility_id <> $exclude
          AND m.status IN ['active', 'validated']
        RETURN m.facility_id AS facility, ip.id AS section_path,
               m.status AS status
        ORDER BY m.facility_id, ip.id
    """
    return gc.query(cypher, ids_name=ids_name, exclude=exclude_facility)
```

Inject into section assignment:

```markdown
### Cross-Facility Precedent

Other facilities have mapped signals to these IDS sections:

{{ cross_facility_mappings }}
```

---

## Prompt Template Changes Summary

| Template | New context variables | Purpose |
|----------|----------------------|---------|
| `section_assignment.md` | `section_clusters`, `cross_facility_mappings` (optional) | Better section-source matching via cluster semantics and precedent |
| `signal_mapping.md` | `identifier_schemas`, `version_context` | Correct typed field population, version-aware transforms |
| `assembly.md` | `coordinate_specs`, `identifier_schemas` | Dimensionally correct assembly patterns, valid enum initialization |
| `validation.md` | `dd_version` (already present) | No new variables — validation already receives dd_version |

---

## Implementation Order

```
E1  DD version threading            Prerequisite: gap closure Phase 1.1
    │
    ├── E2  Cluster context         New query + prompt update
    │
    ├── E3  Identifier schemas      New query + prompt updates (2 templates)
    │
    ├── E4  Version change history  Prerequisite: gap closure Phase 1.3
    │
    ├── E5  Coordinate specs        New query + prompt update
    │
    └── E6  Cross-facility          New query + prompt update (optional)
```

E1 is the prerequisite — all subsequent items are independent of each other
and can be implemented in any order. E2-E3 are highest impact. E6 is optional
and only useful once multiple facilities have active mappings.

---

## Testing

Each enrichment item should include:

1. **Unit test** — verify the new query function returns expected shape against
   the test graph (which has IMASSemanticCluster, IdentifierSchema, IMASCoordinateSpec,
   DDVersion nodes).
2. **Integration test** — run `generate_mapping()` for a test facility/IDS and
   verify the new context appears in the rendered prompts (mock the LLM call,
   assert prompt contains the new sections).
3. **Quality validation** — compare mapping output with and without each enrichment
   for at least one facility/IDS pair. Document quality delta.
