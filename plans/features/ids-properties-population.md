# IDS Properties & Code Metadata Population

**Status**: Plan  
**Created**: 2025-07-24  
**Scope**: `imas_codex/ids/mapping.py`, `imas_codex/ids/tools.py`, `imas_codex/llm/prompts/mapping/`

## Executive Summary

IMAS `ids_properties/*` (78 nodes per IDS) and `code/*` (13 nodes per IDS)
subtrees store per-IDS-occurrence metadata: who created the data, what software
produced it, which DD version was used, provenance information, and processing
notes. These 91 fields per IDS are fundamentally different from signal data:

- **Per-IDS, not per-signal**: A single set of values for each IDS occurrence
  (e.g., one `ids_properties/provider` for the entire `equilibrium` IDS)
- **Not physics data**: Version strings, commit hashes, URLs, free-text
  annotations — not measured or computed physical quantities
- **Deterministic for most fields**: DD version, access layer version, and
  pipeline identity are known at mapping time with no ambiguity
- **Pattern-based for remaining fields**: `provenance/*`, `comment`, and
  `code/parameters` require extracting information from facility documentation
  and configuration, not matching signals

## Field Inventory

### ids_properties/* (78 nodes, 56 leaf fields per IDS)

**Deterministic fields** (populatable without LLM):

| Field | Type | Source |
|-------|------|--------|
| `version_put/data_dictionary` | STR_0D | `DDVersion.id` from graph (known at build time) |
| `version_put/access_layer` | STR_0D | IMAS access layer version (from environment) |
| `version_put/access_layer_language` | STR_0D | Always "python" for imas-codex |
| `homogeneous_time` | INT_0D | Deterministic: analyze time structure of mapped signals |
| `creation_date` | STR_0D | Mapping run timestamp |

**Template fields** (populatable from facility config):

| Field | Type | Source |
|-------|------|--------|
| `provider` | STR_0D | Facility name or operator (from facility config) |
| `source` | STR_0D | imas-codex pipeline identifier |

**LLM-extractable fields** (require reading facility documentation):

| Field | Type | Source |
|-------|------|--------|
| `comment` | STR_0D | Free-text annotation — summarize mapping context |
| `occurrence_type/name` | STR_0D | Categorize occurrence type from context |
| `occurrence_type/index` | INT_0D | Look up in identifier schema |
| `occurrence_type/description` | STR_0D | Description of occurrence type |
| `provenance/node/path` | STR_0D | IDS path reference (structural) |
| `provenance/node/sources` | STR_0D | Data provenance chain description |

**Plugin fields** (13 per IDS — infrastructure metadata):

| Structure | Fields | Source |
|-----------|--------|--------|
| `plugins/infrastructure_get/*` | name, version, commit, repository, description | Read-access software info |
| `plugins/infrastructure_put/*` | name, version, commit, repository, description | Write-access software info |
| `plugins/node/*/readback` | 1 field per operation | Plugin node metadata |

These are highly structured and repetitive — the same software stack info
applies across all plugins for a given facility.

### code/* (13 leaf fields per IDS)

| Field | Type | Source |
|-------|------|--------|
| `code/name` | STR_0D | Pipeline name ("imas-codex") |
| `code/version` | STR_0D | Package version |
| `code/repository` | STR_0D | Git repo URL |
| `code/commit` | STR_0D | Current git commit hash |
| `code/description` | STR_0D | Pipeline description |
| `code/parameters` | STR_0D | Mapping configuration (JSON) |
| `code/output_flag` | INT_1D | Processing status flags |
| `code/library/*/name` | STR_0D | Dependency names |
| `code/library/*/version` | STR_0D | Dependency versions |
| `code/library/*/repository` | STR_0D | Dependency repo URLs |
| `code/library/*/description` | STR_0D | Dependency descriptions |
| `code/library/*/commit` | STR_0D | Dependency commit hashes |

## Implementation Options Analysis

### Option A: Fully Programmatic (No LLM)

**Approach**: Hard-code or template-fill all 91 fields from known sources.

**Pros**:
- Zero LLM cost, instant execution
- Deterministic, reproducible output
- No prompt engineering needed

**Cons**:
- `comment` and `provenance/sources` become generic boilerplate
- `occurrence_type` classification requires understanding context
- Plugin fields need facility-specific knowledge that may not be in config
- Misses the opportunity to generate meaningful free-text annotations

**Assessment**: Works for ~60% of fields (deterministic + template). The
remaining 40% would be generic placeholders. Acceptable for an MVP but
produces lower-quality metadata than a physicist would write.

### Option B: LLM-Only (One Call per IDS)

**Approach**: Send all facility context + field inventory to LLM, have it
populate all 91 fields in structured JSON output.

**Pros**:
- Highest quality for comment, provenance, occurrence_type
- LLM can infer plugin configurations from code context
- Single unified step

**Cons**:
- Expensive: 91 fields × 87 IDSs = 7,917 field populations → ~$2-5/run
- Slow: 87 LLM calls at ~5-10 seconds each
- Many fields are deterministic — wasted LLM capacity
- Risk of LLM hallucinating version numbers, commit hashes, URLs

**Assessment**: Overkill. The LLM would waste most of its output generating
values that could be computed directly. Worse, it may hallucinate concrete
values (version strings, URLs) that we know exactly.

### Option C: Hybrid — Programmatic + Targeted LLM (Recommended)

**Approach**: Fill deterministic fields programmatically, then make 1 LLM
call per IDS for the ~8 ambiguous fields that benefit from reasoning.

**Pros**:
- Optimal cost: 1 LLM call per IDS for 8 fields vs 91
- No hallucination risk for deterministic fields
- High quality for comment, occurrence_type, provenance
- ~$0.05/IDS → ~$4.35 total for all 87 IDSs

**Cons**:
- Slightly more complex orchestration (two substeps)
- Need to design a focused prompt for just the ambiguous fields

**Assessment**: Best trade-off. Deterministic fields get exact values,
ambiguous fields get LLM reasoning. The LLM call is focused on what it's
actually good at: reading facility documentation and generating structured
annotations.

## Recommended Design (Option C)

### Step 1: Programmatic Field Population

Build a `MetadataContext` from known sources:

```python
@dataclass
class MetadataContext:
    """Context for populating ids_properties and code fields."""
    dd_version: str           # from graph DDVersion
    access_layer_version: str # from environment/config
    creation_date: str        # ISO 8601 timestamp
    provider: str             # facility name
    source: str               # "imas-codex v{version}"
    pipeline_version: str     # package version
    pipeline_commit: str      # git commit hash
    pipeline_repo: str        # git repository URL
    pipeline_description: str # package description
    pipeline_config: dict     # mapping configuration used
    library_deps: list[dict]  # key dependencies with versions
```

This populates:
- `version_put/*` — from `dd_version`, `access_layer_version`
- `code/*` — from `pipeline_*` fields
- `creation_date`, `provider`, `source` — from context
- `code/library/*` — from `library_deps`

### Step 2: LLM Field Population

For each IDS, make one focused LLM call with:

**Input context**:
- Facility name and description
- IDS name and documentation
- List of mapped signals (from Stage 1)
- Facility wiki context (reuse from Stage 1)
- Available identifier schemas for `occurrence_type`

**Output**: Structured JSON for 8 fields:

```json
{
    "comment": "JET pf_active mapping covering N poloidal field coils...",
    "occurrence_type_name": "experimental",
    "occurrence_type_index": 0,
    "occurrence_type_description": "Experimental data from JET facility",
    "provenance_sources": "JET PPF/JPF signal database via MDSplus",
    "homogeneous_time": 1,
    "homogeneous_time_reasoning": "All mapped signals share common time base"
}
```

**Prompt design**: A new mapping prompt template pair:
- `metadata_population_system.md` — Static system prompt describing the task,
  IMAS ids_properties semantics, occurrence_type identifier schema
- `metadata_population.md` — Dynamic user prompt with facility context, mapped
  signals summary, specific fields to populate

### Step 3: Merge and Validate

Combine programmatic + LLM outputs into a complete metadata record. Validate:
- `occurrence_type/index` is a valid identifier schema entry
- `homogeneous_time` is 0, 1, or 2
- `comment` is non-empty and descriptive
- No version/URL hallucinations (programmatic fields override any LLM output)

## Implementation Phases

### Phase 1: MetadataContext Builder

Create `imas_codex/ids/metadata.py` with:
- `MetadataContext` dataclass
- `build_metadata_context()` — gathers known values from graph + environment
- `populate_deterministic_fields()` — fills the 60% of fields with exact values

**Tests**: Verify deterministic fields are correct for a known facility/IDS.

### Phase 2: Prompt Template

Create `imas_codex/llm/prompts/mapping/metadata_population_system.md` and
`metadata_population.md`:
- System prompt: ids_properties field semantics, occurrence_type schema,
  output format specification
- User prompt: facility context, IDS context, mapped signals summary

**Tests**: Verify prompt renders correctly with sample context.

### Phase 3: LLM Population Step

Add `populate_ids_metadata()` to `mapping.py`:
- Calls `build_metadata_context()` for programmatic fields
- Renders metadata population prompt with IDS-specific context
- Makes 1 LLM call, parses structured JSON response
- Merges with programmatic values (programmatic takes precedence for
  deterministic fields)

**Tests**: Mock LLM call, verify merge logic, validate output schema.

### Phase 4: Persistence

Add `persist_metadata()` to `models.py`:
- Writes `ids_properties/*` values to IMASNode properties or creates new
  relationship structure
- Writes `code/*` values similarly
- Tags with mapping run ID for provenance

**Tests**: Verify graph state after persistence.

### Phase 5: Orchestration

Wire Stage 3 into the `generate_mapping()` flow:
- Runs after Stage 1 (data mapping) and Stage 2 (error mapping)
- Uses Stage 1 mapped signals as input context
- Can run independently with `--stage metadata`

## Cost Analysis

| Component | Per-IDS Cost | Full Run (87 IDSs) |
|-----------|-------------|---------------------|
| Programmatic population | ~$0 | ~$0 |
| LLM metadata call | ~$0.05 | ~$4.35 |
| Graph queries | ~$0.001 | ~$0.09 |
| **Total** | **~$0.05** | **~$4.44** |

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| LLM hallucinates version strings | Medium | Low | Programmatic fields override LLM; only ambiguous fields use LLM |
| occurrence_type classification errors | Low | Medium | Validate against identifier schema; flag unknowns for review |
| Plugin fields vary by facility | High | Low | Start with common fields; facility-specific plugins added iteratively |
| Slow execution (87 LLM calls) | Medium | Low | Parallelize across IDSs; cache system prompt; ~7-15 min total |

## Dependencies

- **Requires**: Staged mapping pipeline (Stage 1 data mappings as input)
- **Requires**: Error-metadata-filtering (so metadata fields are excluded
  from Stage 1 search, avoiding circular contamination)
- **Optional**: Facility configuration system (for programmatic provider/source
  values; can hard-code initially)

---

## Priority & Dependencies

**Priority: P3 — Medium (blocked)**

| Depends On | Enables |
|-----------|---------|
| staged-mapping-pipeline (Stage 1 data mappings) | Complete mapping quality, full IDS metadata |
| error-metadata-filtering (✅ implemented) | — |

This plan cannot start until staged-mapping-pipeline Stage 1 is operational.

## Documentation Updates

When this work is complete, update:
- [ ] `AGENTS.md` — if new CLI commands or MCP tools are added for metadata population
- [ ] Prompt templates — new metadata population prompts
- [ ] `plans/README.md` — mark as complete or move to pending
- [ ] Schema reference — verify `ids_properties` fields are documented after schema changes
