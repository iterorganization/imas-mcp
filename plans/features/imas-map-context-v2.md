# IMAS Signal Mapping Context Enrichment v2

> **Status**: Planning  
> **Priority**: High — signal mapping quality depends on context depth at each step  
> **Scope**: `imas map` CLI — signal-source-to-IMAS-path signal mapping pipeline  
> **Dependency**: Requires signal enrichment v3 (companion plan) to be stable first  
> **Supersedes**: `imas-map-context-v1.md`  
> **Principle**: The mapping pipeline reads enriched signal metadata from the graph.
> Better signal enrichment ⇒ better mapping input ⇒ better mapping output. The two
> pipelines are decoupled by the graph.

## Key Changes from v1

1. **Terminology**: "field mapping" → "signal mapping" throughout. Signal Sources
   map to IMAS targets. IMAS is one target; the architecture supports future targets.
2. **Assembly step**: New LLM step discovers assembly patterns (concatenation,
   transposition, matrix assembly) as a separate pass after signal mapping.
3. **Function naming**: Removed `_step0_`, `_step1_` etc prefix convention.
   Functions are named by what they do.
4. **Multi-target support**: A single SignalSource can map to multiple IMAS paths.
   Prompts and pipeline explicitly support this.
5. **YAML recipes abandoned**: Assembly is graph-driven only. No YAML fallback.
6. **Rich CLI**: Adopts the common discover CLI progress infrastructure
   (`BaseProgressDisplay`, `StreamQueue`, `WorkerStats`, cost/time tracking).
7. **Unit handling**: Uses project dot-exponential notation via
   `normalize_unit_symbol()` / `validate_unit()`.

## LLM Routing & Caching

All LLM calls route through the LiteLLM proxy → OpenRouter. The `openrouter/`
prefix is required on model identifiers to preserve `cache_control` blocks.

Signal mapping calls use `call_llm_structured()` from
`imas_codex.discovery.base.llm` (returns `(parsed, cost_usd, tokens)` tuple).
Static system prompts (task description, transform rules, COCOS rules) are
cacheable across calls — dynamic context (per-section signal source details,
IMAS fields) varies per call.

The pipeline batches per-section: assign_sections is one LLM call for all
sources, map_signals is one LLM call per section, discover_assembly is one
LLM call per section. No additional batching changes needed.

## Separation from Signal Enrichment

The `discover signals` CLI enriches facility signals — it generates
descriptions, extracts units, classifies physics domains, and creates
SignalSource nodes. All of that work lives in the companion plan
(`signal-enrichment-v3.md`).

This plan covers **only** the `imas map` CLI and the signal mapping pipeline
(`imas_codex/ids/mapping.py`), which reads enriched signal data from the graph
and generates signal-level mappings + assembly patterns.

```
signals CLI (enrichment)      │    imas map CLI (signal mapping)
──────────────────────────────┼──────────────────────────────────
scan → enrich → check         │    gather_context
FacilitySignal properties     │    assign_sections (LLM)
SignalSource creation         │    map_signals (LLM, per section)
physics_domain assignment     │    discover_assembly (LLM, per section)
unit extraction               │    validate_mappings (programmatic)
code ref injection            │    persist → activate
──────────────────────────────┼──────────────────────────────────
        ▼ writes to graph     │    ▲ reads from graph
```

---

## Current Pipeline Architecture

The mapping pipeline in `imas_codex/ids/mapping.py` (`generate_mapping()`):

| Step | Current Name | Type | Purpose |
|------|-------------|------|---------|
| 0 | `_step0_gather_context()` | Deterministic | Fetch signal sources, IDS subtree, semantic search, existing mappings, COCOS paths |
| 1 | `_step1_assign_sections()` | LLM | Assign signal sources to IMAS struct-array sections |
| 2 | `_step2_field_mappings()` | LLM | Per section: generate signal-level source→target mappings with transforms |
| 3 | `_step3_validate()` | Programmatic | Source/target existence, transform execution, unit compatibility, duplicate detection |
| Persist | `persist_mapping_result()` | Programmatic | Write IMASMapping node, POPULATES, USES_SIGNAL_SOURCE, MAPS_TO_IMAS |

**New pipeline** (renamed + assembly step added):

| Step | New Name | Type | Purpose |
|------|----------|------|---------|
| — | `gather_context()` | Deterministic | Fetch signal sources with full metadata, IDS subtree, semantic search, existing mappings, COCOS paths |
| — | `assign_sections()` | LLM | Assign signal sources to IMAS struct-array sections |
| — | `map_signals()` | LLM | Per section: generate signal-level source→target mappings with transforms |
| — | `discover_assembly()` | LLM | Per section: discover how multiple signals compose into IMAS arrays/matrices |
| — | `validate_mappings()` | Programmatic | Extended validation with unit enforcement, domain cross-check, COCOS enforcement |
| — | `persist_mapping_result()` | Programmatic | Write all graph entities (unchanged) |

CLI commands (at `imas_codex/cli/map.py`) remain:
```
imas-codex imas map run FACILITY IDS_NAME [-m model] [--dd-version] [--cost-limit] [--no-persist] [--no-activate]
imas-codex imas map status FACILITY [IDS_NAME]
imas-codex imas map show FACILITY IDS_NAME
imas-codex imas map validate FACILITY IDS_NAME
imas-codex imas map clear FACILITY IDS_NAME
imas-codex imas map activate FACILITY IDS_NAME
```

---

## Phase 1: Rename Functions & Terminology

**Goal**: Align code with "signal mapping" terminology and remove step-number
prefixes from function names.

### 1.1 Rename Pipeline Functions

| Current | New |
|---------|-----|
| `_step0_gather_context()` | `gather_context()` |
| `_step1_assign_sections()` | `assign_sections()` |
| `_step2_field_mappings()` | `map_signals()` |
| `_step3_validate()` | `validate_mappings()` |
| `_format_groups()` | `_format_sources()` |

In `ids/tools.py`:

| Current | New |
|---------|-----|
| `query_signal_sources()` | `query_signal_sources()` (unchanged, name is already correct) |

### 1.2 Rename Prompt Templates

| Current | New |
|---------|-----|
| `field_mapping.md` | `signal_mapping.md` |
| `exploration.md` | `section_assignment.md` |

Update all frontmatter descriptions and template content:
- "field mapping" → "signal mapping"
- "signal group" → "signal source"
- "field-level" → "signal-level"

### 1.3 Update Model Descriptions

In `ids/models.py`, update docstrings and field descriptions:

| Model | Field | Current | New |
|-------|-------|---------|-----|
| `FieldMappingEntry` | class | "Field-level mapping entry" | "Signal mapping entry" |
| `FieldMappingBatch` | class | "Batch of field mappings" | "Batch of signal mappings" |
| `FieldMappingEntry` | `source_id` | "Source SignalGroup id" | "SignalSource node id" |
| `EscalationFlag` | `source_id` | "SignalGroup id" | "SignalSource node id" |

### 1.4 Update Cost Step Names

| Current | New |
|---------|-----|
| `"step1_sections"` | `"assign_sections"` |
| `"step2_fields_{section_path}"` | `"map_signals_{section_path}"` |

### 1.5 Update Module Docstring

Replace `mapping.py` docstring to reflect signal mapping terminology and the
new pipeline step names.

---

## Phase 2: Enrich Context at gather_context

**Goal**: The signal mapping LLM receives richer, more specific signal metadata.
Use deterministic graph queries to inject structured metadata including physics
domain, COCOS, code references, and representative signal details.

### 2.1 Extend `query_signal_sources()` Return Data

Currently returns: `id`, `group_key`, `description`, `keywords`,
`physics_domain`, `status`, `member_count`, `sample_members`, `imas_mappings`.

Add per-source:
- **Representative signal description**: The representative FacilitySignal's
  enriched `description` field — the most detailed physics description.
- **Representative units**: `unit` field from the representative signal (dot-exp format).
- **Representative sign_convention**: From the representative signal.
- **Representative cocos**: COCOS convention index if available.
- **Member accessor examples**: Return first 10 member accessors (not just IDs).
- **Physics domain**: Already returned, but verify it propagates correctly from
  `SignalSource.physics_domain`.

Updated Cypher:
```cypher
MATCH (sg:SignalSource)
WHERE sg.facility_id = $facility
OPTIONAL MATCH (m)-[:MEMBER_OF]->(sg)
WITH sg, count(m) AS member_count,
     collect(DISTINCT m.id)[..5] AS sample_members,
     collect(DISTINCT m.accessor)[..10] AS sample_accessors
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
OPTIONAL MATCH (sg)-[r:MAPS_TO_IMAS]->(ip:IMASNode)
RETURN sg.id AS id, sg.group_key AS group_key,
       sg.description AS description,
       sg.keywords AS keywords,
       sg.physics_domain AS physics_domain,
       sg.status AS status,
       member_count,
       sample_members,
       sample_accessors,
       rep.description AS rep_description,
       rep.unit AS rep_unit,
       rep.sign_convention AS rep_sign_convention,
       rep.cocos AS rep_cocos,
       collect(DISTINCT {
           target_id: ip.id,
           transform: r.transform_expression,
           source_units: r.source_units,
           target_units: r.target_units
       }) AS imas_mappings
ORDER BY sg.group_key
```

### 2.2 Inject Code References per Source

Add `fetch_source_code_refs()` to `ids/tools.py`:

```cypher
MATCH (sg:SignalSource {id: $source_id})
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
       -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
       -[:EXTRACTED_FROM]->(cc:CodeChunk)
RETURN cc.text AS code, cc.language AS language,
       cc.source_file AS file LIMIT 5
```

This returns actual code showing how the signal is read, what units it
expects, and what transforms are applied. This is critical for generating
correct `transform_expression` values.

Include in signal source detail when rendering `signal_mapping.md`.

### 2.3 Inject COCOS Context per Source

If the representative signal has a `cocos` field set, inject it into the
signal mapping prompt:

```markdown
### Source COCOS
Signal COCOS convention: {{ source_cocos }}
Target IMAS paths requiring sign flip: {{ cocos_paths }}
```

This gives the LLM concrete context to decide whether to include
`-value` in `transform_expression`.

### 2.4 Unit Analysis: Use Dot-Exponential Notation

The project uses a custom pint formatter (`"U"`) producing dot-exponential
notation (e.g., `m.s^-1`, `A.m^-2`). All unit references in prompts and
validation must use this format via `normalize_unit_symbol()` from
`imas_codex.units`.

Replace the keyword-based unit extraction in `_format_unit_analysis()`:

```python
def _format_unit_analysis(
    sources: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> str:
    """Run unit analysis between signal sources and IMAS fields.

    Uses the rep_unit field (dot-exp format from normalize_unit_symbol)
    instead of extracting units from keywords.
    """
    from imas_codex.units import normalize_unit_symbol

    lines: list[str] = []
    for src in sources:
        signal_unit = src.get("rep_unit")
        if not signal_unit:
            continue
        # Normalize to dot-exp
        signal_unit = normalize_unit_symbol(signal_unit) or signal_unit
        for f in fields:
            imas_unit = f.get("units")
            if imas_unit:
                imas_unit = normalize_unit_symbol(imas_unit) or imas_unit
                result = analyze_units(signal_unit, imas_unit)
                if result.get("compatible"):
                    factor = result.get("conversion_factor", 1.0)
                    lines.append(
                        f"  {signal_unit} → {imas_unit}: compatible"
                        + (f" (×{factor})" if factor != 1.0 else "")
                    )
                elif result.get("error"):
                    lines.append(f"  {signal_unit} → {imas_unit}: {result['error']}")
    return "\n".join(lines) if lines else "(no unit analysis needed)"
```

---

## Phase 3: Multi-Target Signal Mapping

**Goal**: A single SignalSource can map to multiple IMAS paths. The pipeline
must discover and persist all mappings, not just one-to-one.

### 3.1 Update Signal Mapping Prompt

The `signal_mapping.md` template must instruct the LLM that one signal source
can produce multiple IMAS path mappings. Examples:

- A coil position signal maps to both `pf_active/coil/element/geometry/rectangle/r`
  AND `pf_active/coil/element/geometry/outline/r`
- A magnetic probe signal maps to its measurement AND its position fields
- A plasma current signal maps to both `equilibrium/global_quantities/ip`
  AND `magnetics/ip/data`

The prompt should explicitly state:

```markdown
## Multi-Target Mapping

A single signal source may map to **multiple** IMAS fields. This is expected when:
- The same physical measurement appears in multiple IDS locations
- Position data feeds both geometry definitions and measurement contexts
- Derived quantities populate multiple output fields

Return ALL valid mappings for each source — do not limit to one-to-one.
```

### 3.2 Update Pydantic Models for Multi-Target

The `FieldMappingEntry` model already supports this — multiple entries can
share the same `source_id`. No model changes needed. The `SectionAssignment`
model allows the same `source_id` to appear in multiple assignments, enabling
cross-section multi-target mapping.

### 3.3 Validation: Handle Legitimate Duplicate Targets

The current `validate_mapping()` flags duplicate targets. With multi-target,
the same source mapping to different targets is expected. But the same
target receiving mappings from different sources IS still a concern.

Update validation:
- **Same source → multiple targets**: No warning (explicitly allowed)
- **Multiple sources → same target**: Warning escalation (potential conflict)
- **Same source → same target with different transforms**: Error (conflicting)

---

## Phase 4: Assembly Discovery Step

**Goal**: Add an LLM step after signal mapping that discovers how multiple
individually-mapped signals compose into IMAS array structures. This step
generates the assembly metadata stored on `POPULATES` relationship properties.

### 4.1 Why Assembly is a Separate Step

Signal mapping discovers the **individual** source→target relationships:
each SignalSource maps to one or more IMAS paths with a transform expression.
But IMAS struct-arrays often require **assembly** — combining multiple
individually-mapped signals into a coherent array structure.

Assembly patterns include:

| Pattern | Example | Description |
|---------|---------|-------------|
| `array_per_node` | PF coils | Each signal source becomes one struct-array entry — the array size equals the number of sources mapped to this section |
| `concatenate` | Magnetic probes across arrays | Multiple signal sources are concatenated into a single array, potentially with reindexing |
| `transpose` | Profile data | Array data from sources needs transposition to match IMAS dimension ordering |
| `matrix_assembly` | Coil circuit connections | Separate signals (e.g., circuit definitions stored individually) assembled into an interaction matrix at the IMAS target |
| `nested_array` | Wall limiter geometry | Sources populate nested sub-arrays within a parent structure (e.g., `wall/description_2d[0]/limiter/unit`) |

Signal mapping handles individual transforms. Assembly handles composition.

### 4.2 Assembly Discovery Prompt (`assembly.md`)

New prompt template at `imas_codex/agentic/prompts/mapping/assembly.md`:

```markdown
---
name: assembly
description: Discover assembly patterns for IMAS struct-array population
---

You are an IMAS assembly expert. Given signal mappings from the previous step,
determine how the mapped signals should be **assembled** into IMAS struct-array
entries.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}
- **Section**: {{ section_path }}

### Signal Mappings for This Section

These signal mappings have been generated. Each maps a SignalSource to one or
more IMAS fields with individual transform expressions:

{{ signal_mappings }}

### IMAS Section Structure

{{ imas_section_structure }}

### Signal Source Metadata

{{ source_metadata }}

## Assembly Patterns

Choose the assembly pattern that best describes how these signals compose into
the IMAS struct-array:

### `array_per_node` (Default)
Each signal source becomes one entry in the struct-array. Array size equals the
number of distinct sources. Most common pattern.

### `concatenate`
Multiple signal sources contribute to a single array field. Data from all sources
is concatenated along the first dimension. Order matters — specify the ordering
criterion if deterministic ordering is required.

### `concatenate_transpose`
Same as concatenate but the resulting array needs transposition to match IMAS
dimension ordering (e.g., [channel, time] → [time, channel]).

### `matrix_assembly`
Individual scalar or vector signals are assembled into a 2D matrix. The IMAS
target is a matrix (e.g., circuit connection matrix in pf_active) where each
source populates one row or column. This is common for interaction matrices
where individual circuit or component definitions are stored separately.

### `nested_array`
Sources populate nested sub-arrays within a parent container. The parent
structure is pre-sized (e.g., 1 entry for a single wall description) and
sources fill inner arrays (e.g., limiter units within a wall description).

## Task

For this section, determine:

1. **Assembly pattern**: Which pattern from above
2. **Array sizing**: How the struct-array should be dimensioned
3. **Init arrays**: Sub-arrays that need pre-initialization with fixed sizes
   (e.g., `{"position": 1}` for a single outline per coil)
4. **Element configuration**: If entries have sub-element arrays (e.g., individual
   turns within a coil), describe the element structure
5. **Ordering**: If order matters (e.g., coil index), specify the ordering field
6. **Source selection**: How to query signal data at assembly time:
   - `source_system`: The MDSplus/data system name
   - `source_data_source`: Specific data source within the system
   - `source_epoch_field`: Field linking to temporal epochs
   - `source_select_via`: Relationship-based selection (alternative to property match)

## Output Format

Return a JSON object matching the `AssemblyConfig` schema.
```

### 4.3 Assembly Pydantic Models

Add to `ids/models.py`:

```python
class AssemblyPattern(StrEnum):
    """How multiple signal sources compose into an IMAS struct-array."""
    ARRAY_PER_NODE = "array_per_node"
    CONCATENATE = "concatenate"
    CONCATENATE_TRANSPOSE = "concatenate_transpose"
    MATRIX_ASSEMBLY = "matrix_assembly"
    NESTED_ARRAY = "nested_array"


class AssemblyConfig(BaseModel):
    """Assembly configuration for one IMAS struct-array section."""
    section_path: str
    pattern: AssemblyPattern = AssemblyPattern.ARRAY_PER_NODE
    init_arrays: dict[str, int] | None = None
    elements_config: str | None = None  # JSON string
    nested_path: str | None = None
    parent_size: int | None = None
    source_system: str | None = None
    source_data_source: str | None = None
    source_epoch_field: str | None = None
    source_select_via: str | None = None
    ordering_field: str | None = None
    reasoning: str = ""
    confidence: float = 0.8


class AssemblyBatch(BaseModel):
    """Assembly configurations for all sections in an IDS."""
    ids_name: str
    configs: list[AssemblyConfig]
```

### 4.4 Pipeline Integration

The `discover_assembly()` function runs after `map_signals()`:

```python
def discover_assembly(
    facility: str,
    ids_name: str,
    sections: SectionAssignmentBatch,
    signal_batches: list[SignalMappingBatch],
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> AssemblyBatch:
    """Discover assembly patterns for each section."""
    configs: list[AssemblyConfig] = []

    for assignment, batch in zip(sections.assignments, signal_batches):
        section_path = assignment.imas_section_path

        # Get section structure for assembly context
        section_structure = fetch_imas_subtree(
            ids_name,
            section_path.removeprefix(f"{ids_name}/"),
            gc=gc,
        )

        prompt = _render_prompt(
            "assembly",
            facility=facility,
            ids_name=ids_name,
            section_path=section_path,
            signal_mappings=_format_signal_mappings(batch),
            imas_section_structure=_format_subtree(section_structure),
            source_metadata=_format_source_metadata(assignment, context),
        )

        messages = [
            {"role": "system", "content": "You are an IMAS assembly expert."},
            {"role": "user", "content": prompt},
        ]

        config = _call_llm(
            messages,
            AssemblyConfig,
            model=model,
            step_name=f"discover_assembly_{section_path}",
            cost=cost,
        )
        configs.append(config)

    return AssemblyBatch(ids_name=ids_name, configs=configs)
```

### 4.5 Assembly Persistence

The assembly config is persisted via the existing `POPULATES` relationship
properties. The `persist_mapping_result()` function already creates `POPULATES`
relationships — extend it to write assembly properties:

```python
# In persist_mapping_result():
for config in assembly_batch.configs:
    gc.query("""
        MATCH (m:IMASMapping {id: $mapping_id})
        MATCH (root:IMASNode {id: $section_path})
        MERGE (m)-[p:POPULATES]->(root)
        SET p.structure = $pattern,
            p.init_arrays = $init_arrays,
            p.elements_config = $elements_config,
            p.nested_path = $nested_path,
            p.parent_size = $parent_size,
            p.source_system = $source_system,
            p.source_data_source = $source_data_source,
            p.source_epoch_field = $source_epoch_field,
            p.source_select_via = $source_select_via
    """, ...)
```

This feeds directly into the existing `IDSAssembler` which reads these
properties via `load_sections()` in `graph_ops.py`.

### 4.6 Remove YAML Recipe Fallback

The `IDSAssembler` currently has two modes: graph-driven and YAML recipe.
The YAML recipe path is an abandoned pattern. Remove:

- The YAML recipe loading fallback in `assembler.py`
- The `_assemble_from_yaml()` method
- Any references to `recipes/` directory
- The constructor logic that searches for YAML files

The assembler should be purely graph-driven. If no `IMASMapping` exists,
it should raise a clear error directing the user to run `imas map run`.

---

## Phase 5: Prompt Improvements

**Goal**: Improve signal mapping prompts with better structure, richer context
injection, and static-first ordering for cache hits.

### 5.1 Signal Mapping Prompt: Structured Source Context

The `signal_mapping.md` template currently receives `signal_source_detail`
as a raw JSON dump. Restructure to highlight the most useful information:

```markdown
### Signal Source: {{ source_id }}

**Description**: {{ rep_description }}
**Physics Domain**: {{ physics_domain }}
**Units**: {{ rep_unit | default("unknown") }}
**Sign Convention**: {{ rep_sign_convention | default("unknown") }}
**COCOS**: {{ rep_cocos | default("not set") }}
**Members**: {{ member_count }} signals
**Accessor Pattern**: {{ group_key }}
**Sample Accessors**: {{ sample_accessors | join(", ") }}

{% if code_refs %}
### Code References

Real code showing how this signal is read and used:

{% for ref in code_refs %}
```{{ ref.language }}
// {{ ref.file }}
{{ ref.code }}
```
{% endfor %}
{% endif %}
```

### 5.2 Section Assignment Prompt: Physics Domain Emphasis

The section assignment prompt lists signal sources but doesn't emphasize
physics domain. Since enrichment assigns `physics_domain` from the
`PhysicsDomain` enum, surface it prominently:

```markdown
- {{ source.id }} (domain={{ source.physics_domain }}, members={{ source.member_count }}): {{ source.description }}
```

### 5.3 Static-First Prompt Ordering

Both mapping prompts must follow static-first ordering for cache hits.
The current `signal_mapping.md` has task/transform rules AFTER dynamic
context. Fix ordering:

```
1. System prompt (static): "You are an IMAS signal mapping expert"
2. User prompt:
   a. Task description + Transform Rules + COCOS Rules (static/quasi-static)
   b. COCOS sign-flip paths (quasi-static per IDS — move higher)
   c. Facility + IDS + section (quasi-static)
   d. Signal source detail with code refs (dynamic)
   e. IMAS fields (dynamic)
   f. Unit analysis (dynamic)
   g. Existing mappings (dynamic)
```

Move the "Task", "Transform Rules", and "COCOS sign-flip paths" sections
before the dynamic source/field context so the static instructions are part
of the cacheable prefix.

---

## Phase 6: Validation Improvements

**Goal**: Strengthen programmatic validation with checks leveraging richer
signal metadata.

### 6.1 Unit Compatibility as Hard Validation

After signal-enrichment-v3 ensures pint-validated units via
`normalize_unit_symbol()`, strengthen unit checks:

- If `source_units` and `target_units` are both set AND incompatible
  dimensionally → escalation severity = `error` (not `warning`)
- If `transform_expression = "value"` but `source_units ≠ target_units` →
  escalation: transform should include unit conversion
- All unit comparisons use dot-exponential notation from `normalize_unit_symbol()`

### 6.2 Physics Domain Cross-Check

Validate that the assigned IMAS section's physics domain is compatible
with the source's `physics_domain`. For example, a source with
`physics_domain = 'magnetics'` should not be mapped to
`core_profiles/profiles_1d/ion`.

Important: this is a **soft check** (warning) because:
- A single source can legitimately map to multiple IMAS paths across domains
- Cross-domain mappings exist (e.g., plasma current appears in both
  `magnetics` and `equilibrium`)
- The warning helps catch obvious misassignments without blocking valid mappings

### 6.3 COCOS Sign-Flip Enforcement

If a target path appears in `get_sign_flip_paths()` AND the source signal
has a `cocos` field set, validate that the `transform_expression` includes
sign handling. If `transform_expression = "value"` for a sign-flip path →
escalation.

### 6.4 Multi-Target Validation

Specific checks for multi-target mappings:
- Same source → multiple targets: allowed, no warning
- Multiple sources → same target: warning (potential conflict)
- Same source → same target with different transforms: error

---

## Phase 7: Rich CLI with Progress Monitoring

**Goal**: Adopt the common discover CLI infrastructure for rich progress
displays, cost tracking, and time management.

### 7.1 CLI Options

Add standard discover CLI options to `imas map run`:

```python
@map_cmd.command("run")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--model", "-m", help="Override LLM model identifier.")
@click.option("--dd-version", help="Override Data Dictionary version.")
@click.option("--cost-limit", "-c", type=float, default=5.0,
              help="Maximum LLM spend in USD (default: $5)")
@click.option("--no-persist", is_flag=True, help="Skip graph persistence.")
@click.option("--no-activate", is_flag=True,
              help="Persist as 'generated' without promoting.")
@click.option("--time", "time_limit", type=int, default=None,
              help="Maximum runtime in minutes.")
```

### 7.2 Progress Display

Create `MappingProgressDisplay` extending `BaseProgressDisplay` from
`imas_codex/discovery/base/progress.py`:

```python
class MappingProgressDisplay(BaseProgressDisplay):
    """Rich progress display for the signal mapping pipeline."""

    def __init__(self, facility, ids_name, cost_limit, model, console=None):
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            title_suffix=f"Signal Mapping — {ids_name}",
        )
        self.state = MappingProgressState(
            ids_name=ids_name,
            model=model,
        )
```

Pipeline stages displayed:

```
CONTEXT   ━━━━━━━━━━━━━━━━━━━━━━  Gathering signal sources + DD context
  42 signal sources, 156 IDS fields, 8 COCOS paths

SECTIONS  ━━━━━━━━━━━━━━━━━━━━━━  Assigning to 6 sections            $0.12
  jet:PF:r → pf_active/coil  (0.95)

MAPPING   ━━━━━━━━━━━━━━━━━━━━━━  3/6 sections mapped                $1.24
  pf_active/coil: 12 signal mappings  (2 escalations)

ASSEMBLY  ━━━━━━━━━━━━━━━━━━━━━━  2/6 sections analyzed              $0.45
  pf_active/coil: array_per_node (6 entries, init_arrays: position=1)

VALIDATE  ━━━━━━━━━━━━━━━━━━━━━━  48 bindings validated
  44 passed, 4 warnings, 0 errors
```

Each stage streams results via `StreamQueue` as the LLM produces them.

### 7.3 Cost and Time Tracking

Cost tracking uses the existing `PipelineCost` class, extended to integrate
with the display:

```python
@dataclass
class MappingProgressState:
    ids_name: str
    model: str
    # Pipeline progress
    sources_found: int = 0
    sections_assigned: int = 0
    sections_total: int = 0
    sections_mapped: int = 0
    sections_assembled: int = 0
    bindings_total: int = 0
    bindings_passed: int = 0
    escalations: int = 0
    # Cost tracking
    cost: PipelineCost = field(default_factory=PipelineCost)
    # Time tracking
    start_time: float = field(default_factory=time.time)
    deadline: float | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def deadline_expired(self) -> bool:
        return time.time() >= self.deadline if self.deadline else False
```

Cost is accumulated per-step via `_call_llm()` which already uses
`call_llm_structured()` returning `(parsed, cost_usd, tokens)`.

### 7.4 Integration with `run_discovery()` Harness

The mapping pipeline should use the common `run_discovery()` harness from
`imas_codex/cli/discover/common.py` for consistent display lifecycle:

```python
config = DiscoveryConfig(
    domain="mapping",
    facility=facility,
    display=MappingProgressDisplay(facility, ids_name, cost_limit, model),
    check_graph=True,
    check_embed=False,  # Mapping doesn't need embedding server
    check_ssh=False,    # Mapping is graph-only
)

result = run_discovery(config, async_main=run_mapping_pipeline)
```

This gives us:
- Service health monitoring (Neo4j)
- Graceful shutdown (Ctrl+C)
- Background display refresh at 4fps
- Consistent panel layout with header/servers/pipeline/resources

---

## Phase 8: Coverage and Quality Metrics

**Goal**: Extend mapping quality assessment.

### 8.1 Signal Source Coverage per IDS

The `map validate` command already computes leaf field coverage and signal
group coverage. Extend to report:

- **Mapped vs enriched**: Of all enriched SignalSource nodes with matching
  `physics_domain`, how many have MAPS_TO_IMAS for this IDS?
- **Underspecified sources**: Sources with `status = 'discovered'` (not enriched)
  that might improve with better enrichment.
- **Multi-target coverage**: Sources mapped to >1 IMAS path.

### 8.2 Assembly Coverage

Report assembly configuration completeness:
- Sections with assembly config vs sections without
- Sections using default `array_per_node` vs custom patterns
- Sections with `init_arrays` configured vs unconfigured

### 8.3 Mapping Confidence Distribution

Report distribution of mapping confidences:
- Low (<0.5): Flag for review
- Medium (0.5-0.8): Acceptable but could improve
- High (>0.8): Confident

---

## Phase 9: Validation Test Suite

**Goal**: Assert that the `imas map` CLI works as designed with automated tests.

### 9.1 Unit Tests for Pipeline Functions

Test each renamed function in isolation:

| Test | What it validates |
|------|-------------------|
| `test_gather_context_returns_all_fields` | All expected keys present in context dict, signal sources include `rep_unit`, `rep_cocos`, `physics_domain`, `sample_accessors` |
| `test_assign_sections_valid_output` | LLM response parses to `SectionAssignmentBatch`, all `imas_section_path` values are valid IDS paths |
| `test_map_signals_multi_target` | Same `source_id` can appear in multiple `FieldMappingEntry` rows with different `target_id` |
| `test_discover_assembly_patterns` | Each section gets an `AssemblyConfig` with a valid `AssemblyPattern` |
| `test_validate_mappings_catches_unit_mismatch` | Identity transform with different units raises escalation |
| `test_validate_mappings_catches_cocos_missing` | Sign-flip path with `transform="value"` raises escalation |
| `test_validate_multi_source_same_target` | Warning for multiple sources mapping to same target |
| `test_validate_same_source_diff_targets_ok` | No warning for one source mapping to multiple targets |

### 9.2 Integration Tests

| Test | What it validates |
|------|-------------------|
| `test_generate_mapping_end_to_end` | Full pipeline runs, produces `MappingResult` with bindings and assembly |
| `test_persist_and_load_roundtrip` | Persist result → `load_mapping()` → verify bindings match |
| `test_assembly_config_persisted_on_populates` | Assembly config properties appear on POPULATES relationships |
| `test_map_clear_removes_all` | Clear removes IMASMapping, MAPS_TO_IMAS, MappingEvidence |
| `test_map_validate_reports_coverage` | `map validate` outputs coverage stats |

### 9.3 Prompt Template Tests

| Test | What it validates |
|------|-------------------|
| `test_signal_mapping_prompt_renders` | Template renders without errors with all variables |
| `test_assembly_prompt_renders` | Template renders without errors with all variables |
| `test_section_assignment_prompt_renders` | Template renders without errors |
| `test_prompts_no_field_mapping_terminology` | No occurrences of "field mapping" in any prompt template |
| `test_static_first_ordering` | Task/rules sections appear before dynamic context |

### 9.4 CLI Tests

| Test | What it validates |
|------|-------------------|
| `test_map_run_with_cost_limit` | Pipeline respects `--cost-limit` flag |
| `test_map_run_with_time_limit` | Pipeline respects `--time` flag |
| `test_map_status_lists_all` | `map status FACILITY` lists all mappings |
| `test_map_show_outputs_json` | `map show` produces valid JSON |

---

## Implementation Order

```
Phase 1 ──────────────────────────── Rename Functions & Terminology
  1.1  Rename pipeline functions      [ids/mapping.py]
  1.2  Rename prompt templates        [agentic/prompts/mapping/]
  1.3  Update model descriptions      [ids/models.py]
  1.4  Update cost step names         [ids/mapping.py]
  1.5  Update module docstring        [ids/mapping.py]

Phase 2 ──────────────────────────── Enrich Context
  2.1  Extend query_signal_sources    [ids/tools.py]
  2.2  Add fetch_source_code_refs     [ids/tools.py]
  2.3  Inject COCOS context           [ids/mapping.py]
  2.4  Use dot-exp unit notation      [ids/mapping.py]

Phase 3 ──────────────────────────── Multi-Target Support
  3.1  Update signal mapping prompt   [signal_mapping.md]
  3.2  Verify model support           [ids/models.py]
  3.3  Update validation logic        [ids/validation.py]

Phase 4 ──────────────────────────── Assembly Discovery
  4.1  Create assembly prompt         [agentic/prompts/mapping/assembly.md]
  4.2  Add AssemblyConfig models      [ids/models.py]
  4.3  Implement discover_assembly()  [ids/mapping.py]
  4.4  Extend persist for assembly    [ids/models.py]
  4.5  Remove YAML recipe fallback    [ids/assembler.py]

Phase 5 ──────────────────────────── Prompt Improvements
  5.1  Structured source context      [signal_mapping.md]
  5.2  Physics domain emphasis        [section_assignment.md]
  5.3  Static-first ordering          [signal_mapping.md, section_assignment.md]

Phase 6 ──────────────────────────── Validation Improvements
  6.1  Unit compatibility hardening   [ids/validation.py]
  6.2  Physics domain cross-check     [ids/validation.py]
  6.3  COCOS sign-flip enforcement    [ids/validation.py]
  6.4  Multi-target validation        [ids/validation.py]

Phase 7 ──────────────────────────── Rich CLI
  7.1  Add CLI options                [cli/map.py]
  7.2  Create progress display        [ids/progress.py]
  7.3  Cost and time tracking         [ids/mapping.py, ids/progress.py]
  7.4  Integration with run_discovery [cli/map.py]

Phase 8 ──────────────────────────── Coverage Metrics
  8.1  Signal source coverage         [ids/validation.py, cli/map.py]
  8.2  Assembly coverage              [cli/map.py]
  8.3  Confidence distribution        [cli/map.py]

Phase 9 ──────────────────────────── Validation Test Suite
  9.1  Unit tests                     [tests/ids/test_mapping.py]
  9.2  Integration tests              [tests/ids/test_mapping_integration.py]
  9.3  Prompt template tests          [tests/ids/test_mapping_prompts.py]
  9.4  CLI tests                      [tests/test_cli.py]
```

---

## Dependencies on Signal Enrichment v3

| This plan (signal mapping) | Requires from signals plan |
|----------------------------|---------------------------|
| Phase 1 (rename) | Phase 4.1: SignalGroup → SignalSource schema migration |
| Phase 2.1 (rep metadata) | Phase 1.4: `enrichment_source` set |
| Phase 2.2 (code refs) | Phase 2.3: `fetch_signal_code_refs()` pattern |
| Phase 2.4 (dot-exp units) | Phase 3.3: pint-validated `unit` field |
| Phase 5.2 (physics domain) | Phase 3.2: physics_domain reliably set |
| Phase 6.2 (domain check) | Phase 3.2: physics_domain on all enriched signals |
| Phase 6.3 (COCOS) | Signal `cocos` field populated (future) |

These dependencies mean phases 2+ should be implemented AFTER
signal-enrichment-v3 Phases 1-5 are stable. Phase 1 (rename) and
Phase 4 (assembly) can proceed independently.

---

## Graph Schema Summary

No schema changes are required. All entities are already defined:

```
IMASMapping -[:AT_FACILITY]-> Facility
IMASMapping -[:POPULATES]-> IMASNode      (properties: structure, init_arrays,
                                           elements_config, nested_path, parent_size,
                                           source_system, source_data_source,
                                           source_epoch_field, source_select_via,
                                           enrichment)
IMASMapping -[:USES_SIGNAL_SOURCE]-> SignalSource
SignalSource -[:MAPS_TO_IMAS]-> IMASNode  (properties: source_property,
                                           transform_expression, source_units,
                                           target_units, cocos_source, cocos_target,
                                           driver, status, confidence)
SignalSource -[:HAS_EVIDENCE]-> MappingEvidence
SignalSource -[:AT_FACILITY]-> Facility
```

The `POPULATES` relationship properties already capture all assembly metadata.
The `AssemblyPattern` enum is a Python-side concept that maps to the `structure`
property string value on `POPULATES`.

---

## Design Decisions

### Why separate signal mapping and assembly into two LLM steps?

These are fundamentally different cognitive tasks:

1. **Signal mapping** asks: "What IMAS field does this signal correspond to,
   and what transform converts the value?" This is a physics matching task.

2. **Assembly** asks: "Given N individually-mapped signals, how do they compose
   into the IMAS array structure?" This is a structural/data-engineering task.

Combining them in one prompt creates confusion — the LLM conflates individual
signal transforms with array-level composition. Two focused prompts produce
better results for both tasks.

### Why not use YAML recipes?

YAML recipes were a bootstrapping mechanism for known mappings. The pipeline
should discover assembly patterns from graph context (signal source metadata,
IDS structure, existing mappings). This is more maintainable and generalizes
across facilities. If no IMASMapping exists, the user runs `imas map run`.

### Should the signals CLI suggest target IDS sections?

**No.** Signal enrichment assigns `physics_domain` from the `PhysicsDomain`
enum — a physics classification, not a mapping target. The signal mapping
pipeline's `assign_sections()` is responsible for connecting sources to
specific IMAS sections. These are separate concerns.

### Why allow multi-target mappings?

Real fusion data frequently maps to multiple IMAS locations:
- Plasma current (`Ip`) appears in `magnetics/ip/data`, `equilibrium/global_quantities/ip`,
  and potentially in `core_profiles`
- Coil positions appear in both geometry and measurement contexts
- The same time-base signal feeds multiple IDS sections

Restricting to one-to-one would require the user to manually replicate mappings.

### How does cost tracking integrate with the discover CLI pattern?

The mapping pipeline uses `PipelineCost` to accumulate per-step costs (already
implemented). The Rich progress display reads `cost.total_usd` to show running
cost in the display panel. The `--cost-limit` flag checks against accumulated
cost before each LLM call and stops the pipeline if exceeded. This mirrors the
`WorkerStats.cost` pattern in the discover CLI but is simpler because mapping
is sequential (not parallel workers).
