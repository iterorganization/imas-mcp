# IMAS Mapping Pipeline: LLM-Driven Signal-to-IDS Mapping

Status: **Planning**
Priority: High — core deliverable
Supersedes: `imas-mappings.md` (agent-teams approach), extends `mapping-redesign.md` (schema architecture)

## Summary

Build a multi-step, tool-augmented LLM pipeline that discovers mappings between
facility signals and IMAS Data Dictionary paths. The pipeline produces `MAPS_TO_IMAS`
relationships (per `mapping-redesign.md`) via three LLM calls with programmatic context
injection between each step. A top-level `map` CLI drives the pipeline. Signal grouping
is unified as a common infrastructure layer invoked during signal discovery, before
mapping begins.

Motivating use case: produce `pf_active` and `pf_passive` IDSs for every JET epoch.

## Design Principles

1. **Response model is IMASMapping** — the LLM iterates until mappings are correct.
   Not a "proposal" model with a separate acceptance step. Each pipeline step has
   its own Pydantic response model, but the final step produces data that persists
   directly as graph relationships.

2. **Ground truth from real facility data** — validation comes from existing mapped
   signals at real IMAS databases and validated datasets, not from hardcoded specs
   in our codebase. Hardcoded mapping specs are removed.

3. **Multi-step pipeline with programmatic tools** — not single-pass structured
   output and not full agentic. The LLM gets enriched context between steps via
   Python functions called deterministically (IMAS DD lookup, unit analysis, COCOS
   checking). This matches the existing code scorer pattern (triage → enrich → score).

4. **Signal grouping as common infrastructure** — grouping runs at signal discovery
   time. By the time the mapping pipeline starts, `SignalGroup` nodes already exist
   with enriched physics metadata. The mapping pipeline consumes groups, it doesn't
   create them.

5. **Escalation path** — when the pipeline can't confidently map a domain (COCOS
   ambiguity, complex coordinate transforms, multi-source composites), it flags
   the mapping with `status: escalated` and a diagnostic. These are collected for
   manual review or future agentic escalation. Designed in upfront, not bolted on.

## Current State

| Entity | Count (JET) | Notes |
|--------|------------|-------|
| SignalNode | 6,623 | MDSplus tree nodes (was DataNode) |
| FacilitySignal | 1,560 | Extracted from XML configs |
| SignalEpoch | 30 | 21 device_xml epochs + others (was StructuralEpoch) |
| SignalGroup | 0 | Schema exists, never instantiated |
| IMASMapping | 0 | Schema exists, never instantiated |
| MappingEvidence | 0 | Schema exists, never instantiated |
| MAPS_TO_IMAS | 0 | Not yet created |

Hardcoded mapping specs exist in `ids/graph_ops.py` (L280–L600):
`PF_ACTIVE_COIL_MAPPINGS`, `PF_ACTIVE_CIRCUIT_MAPPINGS`, `MAGNETICS_BPOL_MAPPINGS`,
`MAGNETICS_FLUX_LOOP_MAPPINGS`, `PF_PASSIVE_LOOP_MAPPINGS`, `WALL_LIMITER_MAPPINGS`,
plus assembly configs (L532–L596). These are removed in Phase 1.

## Architecture

### Schema Context (from mapping-redesign.md)

Field-level mappings are **relationships**, not nodes:

```
IMASMapping ──[:USES_SIGNAL_GROUP]──▶ SignalGroup ──[:MAPS_TO_IMAS {field props}]──▶ IMASNode
    │                                     ▲
    │                                [:MEMBER_OF]
    │                                     │
    │                              SignalNode / FacilitySignal
    │
    └──[:POPULATES {assembly props}]──▶ IMASNode (STRUCT_ARRAY root)
```

`MAPS_TO_IMAS` carries transform metadata: `source_property`, `transform_code`,
`units_in/out`, `cocos_source/target`, `confidence`, `status`.

`POPULATES` carries assembly config: `structure`, `init_arrays`, `elements_config`,
`enrichment_config`.

### Multi-Step LLM Pipeline (Option B)

Three LLM calls with programmatic context injection between each step.
Each step uses `call_llm_structured()` with a step-specific Pydantic response
model and a dedicated Jinja2 prompt template.

```
┌─────────────────────────────────────────────────────────────┐
│ Input: facility, ids_name, dd_version                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Programmatic] Gather signal groups for this IDS domain    │
│  [Programmatic] Fetch IMAS DD subtree structure             │
│  [Programmatic] Get COCOS sign-flip paths                   │
│                         ↓                                   │
│  ┌───────────────────────────────────┐                      │
│  │ Step 1: IDS Structure Exploration │                      │
│  │ LLM: match signal groups to IMAS  │                      │
│  │      structural sections          │                      │
│  │ Output: SectionAssignmentBatch    │                      │
│  └───────────────┬───────────────────┘                      │
│                  ↓                                           │
│  [Programmatic] For each assigned section:                   │
│    - Fetch detailed IMAS field info (docs, units, coords)   │
│    - Analyze unit compatibility (pint)                       │
│    - Analyze COCOS requirements                             │
│    - Get signal group member details                        │
│                  ↓                                           │
│  ┌───────────────────────────────────┐                      │
│  │ Step 2: Field-Level Mapping       │                      │
│  │ LLM: generate field mappings with │                      │
│  │      transforms, units, COCOS     │                      │
│  │ Output: FieldMappingBatch         │                      │
│  └───────────────┬───────────────────┘                      │
│                  ↓                                           │
│  [Programmatic] Validate:                                   │
│    - IMAS path existence (check_imas_paths)                 │
│    - Unit conversion validity                               │
│    - Completeness (unmapped required fields)                │
│    - Cross-reference existing mappings                       │
│                  ↓                                           │
│  ┌───────────────────────────────────┐                      │
│  │ Step 3: Validation & Assembly     │                      │
│  │ LLM: review validation results,  │                      │
│  │      fix issues, finalize mapping │
│  │ Output: ValidatedMappingResult    │                      │
│  └───────────────┬───────────────────┘                      │
│                  ↓                                           │
│  [Programmatic] Persist to graph:                           │
│    - Create/update MAPS_TO_IMAS relationships               │
│    - Create/update POPULATES relationship                   │
│    - Create/update IMASMapping node                         │
│    - Create MappingEvidence nodes                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Output: IMASMapping node + MAPS_TO_IMAS rels + evidence     │
└─────────────────────────────────────────────────────────────┘
```

### Why Not Single-Pass

A single `call_llm_structured()` must simultaneously:
- Understand signal group properties and units
- Navigate the IMAS DD tree to find correct target paths
- Determine unit conversions (deg→rad, Ohm→ohm, mm→m)
- Identify COCOS sign conventions
- Write transform code
- Assess confidence

This is too much reasoning in one pass. The existing codebase already demonstrates
this: the code scorer uses triage→score, the signal enricher injects wiki/code/tree
context before scoring. Mapping is harder because the LLM must *explore* the DD tree,
*verify* path existence, *check* unit compatibility, and *reason* about physics.

### Why Not Full Agentic (Option C)

The codebase has `smolagents` infrastructure (`CodeAgent`) but it uses the expensive
`agent` model tier (Claude Sonnet) and provides non-deterministic execution. The
multi-step pipeline (Option B) gives structured iteration with clear cost tracking
and independently testable steps.

**Escalation to Option C** is planned for specific IDS that prove too complex for
the fixed pipeline (equilibrium with COCOS, multi-source composites). This is why
the `status: escalated` flag exists — to identify exactly where the pipeline
reaches its limits.

### Step-Specific Response Models

These are Pydantic models for LLM structured output. They are NOT the graph models —
an adapter layer converts the final output to graph operations. This matches the
existing pattern where all discovery pipelines use step-specific response models
independent from LinkML.

```python
# Step 1 output
class SectionAssignment(BaseModel):
    signal_group_id: str           # e.g., "jet:PF:r"
    imas_struct_array: str         # e.g., "pf_active/coil"
    reasoning: str

class SectionAssignmentBatch(BaseModel):
    sections: list[SectionAssignment]
    unmapped_groups: list[str]     # groups the LLM couldn't place

# Step 2 output (per section)
class FieldMapping(BaseModel):
    signal_group_id: str
    source_property: str           # field on signal to extract
    imas_path: str                 # full IMAS leaf path
    transform_code: str | None     # Python expression with `value`
    units_in: str | None
    units_out: str | None
    cocos_source: int | None
    cocos_target: int | None
    confidence: float
    reasoning: str

class FieldMappingBatch(BaseModel):
    mappings: list[FieldMapping]
    escalations: list[EscalationFlag]   # fields that need manual review

class EscalationFlag(BaseModel):
    signal_group_id: str
    imas_path: str | None
    reason: str                    # why automated mapping failed
    category: str                  # "cocos_ambiguity" | "composite_source" | ...

# Step 3 output
class ValidatedMappingResult(BaseModel):
    mappings: list[FieldMapping]         # corrected after validation
    assembly_config: AssemblyConfig
    escalations: list[EscalationFlag]
    overall_confidence: float
    evidence_summary: str

class AssemblyConfig(BaseModel):
    imas_section: str              # STRUCT_ARRAY root (e.g., "pf_active/coil")
    structure: str                 # "array_per_node" | "nested_array"
    init_arrays: dict | None
    elements_config: dict | None
```

### Tool Functions (Programmatic Context Injection)

These are Python functions called between LLM steps. They wrap existing
infrastructure — MCP server tool underlying implementations, COCOS module,
pint-based units.

| Function | Source | Step | Purpose |
|----------|--------|------|---------|
| `fetch_imas_subtree(ids, path)` | Wraps `list_imas_paths` | 1 | IDS tree structure with leaf types |
| `fetch_imas_fields(ids, paths)` | Wraps `fetch_imas_paths` | 2 | Detailed field info: docs, units, coordinates, lifecycle |
| `search_imas_semantic(query, ids)` | Wraps `search_imas` | 1 | Semantic search for relevant IMAS paths |
| `get_sign_flip_paths(ids)` | `cocos/transforms.py` | 2 | Already exists — COCOS sign convention paths |
| `analyze_units(signal_unit, imas_unit)` | New, wraps `pint` | 2 | Unit compatibility check + conversion expression |
| `check_imas_paths(paths)` | Wraps `check_imas_paths` | 3 | Validate path existence in DD |
| `query_signal_groups(facility, ids)` | Graph query | 0 | Get signal groups with member properties |
| `search_existing_mappings(facility, ids)` | Graph query | 3 | What's already mapped |

**MCP Tool Quality Assessment** (from calling the tools during planning):

| Tool | Quality for Mapping | Notes |
|------|-------------------|-------|
| `list_imas_paths` | Excellent | 55 leaf paths for `pf_active/coil` in tree format. Perfect for Step 1 structure overview. |
| `fetch_imas_paths` | Excellent | Rich JSON per path: documentation (full hierarchical), units, data_type, lifecycle, validation_rules. Perfect for Step 2 field detail. |
| `search_imas` | Good | 10 paths with scores 0.90–0.97, documentation, units, physics domains. Good for Step 1 semantic context. |
| `search_signals` | Good | 10 signals with data access code templates. Useful for signal metadata but not directly needed in mapping pipeline (groups already enriched). |
| `search_docs` | Moderate | Wiki chunks have low relevance for mapping (score ~0.84). Document references (NodeList.pdf, IMASgo docs) more useful as human references. |
| `search_code` | Low priority | Returns large code examples (IDL, Fortran) that show how others access similar data. Useful for validation but too large for prompt injection. |

## Signal Grouping Unification

### The Problem

Two independent grouping systems produce `SignalGroup` nodes:

| Aspect | FacilitySignal (parallel.py) | SignalNode/MDSplus (graph_ops.py) |
|--------|------------------------------|-----------------------------------|
| Detection | Regex: `\d{2,}` → NNN on accessor | Tree structure: grandparent→parent→leaf |
| When | Continuously during enrich loop | Once after extraction, before enrichment |
| Source data | `FacilitySignal.accessor` | `SignalNode` tree (HAS_NODE traversal) |
| Group ID | `{facility}:{pattern}` | `{facility}:{tree}:{gp_path}:{leaf}` |
| Propagation | Per-signal after each enrichment | Batch after all groups enriched |

Both produce the same graph structure (`SignalGroup` + `MEMBER_OF` edges) but are
tangled into their respective enrichment workers.

### Solution: Common Signal Grouping Layer

Extract a common grouping interface in `imas_codex/discovery/base/grouping.py`:

1. **Detection stays separate** — regex pattern detection and tree structure detection
   are fundamentally different algorithms. A common interface
   (`detect_groups(facility) → list[GroupSpec]`) unifies the output format, but
   implementations remain domain-specific.

2. **Group creation is unified** — `create_signal_group(facility, group_key, member_ids, representative_id)` replaces both `detect_signal_groups()` graph writes and
   `detect_and_create_signal_groups()` graph writes.

3. **Claiming is unified** — both use `claim_token` + `ORDER BY rand()`. Extract to
   `claim_signal_groups(facility, claim_token, batch_size)`.

4. **Propagation is unified** — both copy enrichment metadata to `MEMBER_OF` members.
   `propagate_group_enrichment(group_id, enrichment_dict, member_label)` replaces
   both `propagate_signal_group_enrichment()` and the propagation in
   `mark_signal_groups_enriched()`.

5. **Terminal state** — groups must reach a terminal state (`enriched`) during signal
   discovery. The mapping pipeline queries only `enriched` groups. If a group can't
   be enriched (no description, no physics domain), it stays `discovered` and is
   excluded from mapping — this is an escalation, not a pipeline blocker.

### Grouping Runs at Discovery Time

Signal grouping is **not** a mapping concern. The pipeline flow:

```
Signal Discovery → Groups Created → Groups Enriched → [mapping pipeline starts here]
                                                        ↓
                                                 Query enriched groups
                                                        ↓
                                                 Generate MAPS_TO_IMAS
```

The mapping pipeline takes enriched `SignalGroup` nodes as input. If no groups exist
for a domain, the pipeline reports this as a prerequisite gap.

## CLI Design

> **CLI section superseded by `cli-unification.md`.** Map commands are now under
> `imas-codex imas map` and IDS commands under `imas-codex imas`.

### `imas map` Subgroup

Follows the `discover` pattern: flat namespace, `_CleanupGroup` for help formatting,
shared infrastructure.

```
imas-codex map <facility> <ids_name>           # generate mappings
imas-codex map status <facility> [ids_name]    # show mapping coverage
imas-codex map show <facility> <ids_name>      # display existing mappings
imas-codex map validate <facility> <ids_name>  # re-validate existing mappings
imas-codex map clear <facility> <ids_name>     # remove mappings
```

**Main command** (`map <facility> <ids_name>`):
- Takes `--model`, `--dd-version`, `--epoch`, `--dry-run`
- Queries enriched `SignalGroup` nodes for the given facility + IDS domain
- Runs the 3-step pipeline
- Persists results to graph
- Reports coverage, escalations, and cost

### CLI Restructure: `ids` → `imas`

The existing `ids` CLI group becomes `imas` with `dd` as a subgroup:

```
imas-codex imas list <facility>                # list IDS with mappings
imas-codex imas show <facility> <ids_name>     # show IDS mapping detail
imas-codex imas export <facility> <ids_name>   # export IDS to file
imas-codex imas epochs <facility>              # list epochs
imas-codex imas dd build                       # build DD graph
imas-codex imas dd status                      # DD build status
imas-codex imas dd search <query>              # search DD paths
imas-codex imas dd version                     # current DD version
imas-codex imas dd clear                       # clear DD graph
imas-codex imas dd path-history <path>         # path version history
```

The `seed` subcommand is removed (hardcoded specs deleted).

## Schema Changes

### Rename: StructuralEpoch → SignalEpoch

`StructuralEpoch` is misleading — these epochs represent signal validity ranges,
not structural changes only. The label `SignalEpoch` better reflects that the epoch
defines when a particular signal configuration was active.

Impact: schema class + ~20 references in MDSplus graph_ops.py + all queries that
filter on `StructuralEpoch` label + tests.

### Remove Hardcoded Mapping Specs

Delete from `imas_codex/ids/graph_ops.py`:
- `PF_ACTIVE_COIL_MAPPINGS` (L280)
- `PF_ACTIVE_CIRCUIT_MAPPINGS` (L296)
- `MAGNETICS_BPOL_MAPPINGS` (L300)
- `MAGNETICS_FLUX_LOOP_MAPPINGS` (L326)
- `PF_PASSIVE_LOOP_MAPPINGS` (L334)
- `WALL_LIMITER_MAPPINGS` (L343)
- Assembly configs (L532–L596)
- `seed_ids_mappings()` (L679)

Keep: `load_mapping()`, `load_sections()`, `load_field_mappings()`, `select_nodes()`,
`FieldMapping`, `Mapping` dataclasses — these are read-side utilities used by the
assembler.

Delete: `imas_codex/ids/recipes/jet/pf_active.yaml` (if it exists)

Remove: `ids seed` CLI subcommand from `imas_codex/cli/ids.py`

## Ground Truth & Benchmarking

### Ground Truth Sources

Ground truth for mapping validation cannot come from our own hardcoded specs (circular).
Sources ranked by reliability:

1. **Existing IMAS databases** — facilities that have already populated IDS via their
   own tooling (not imas-codex). Query what signals map to what IMAS paths in validated
   databases. This requires remote access to IMAS databases (stub for now).

2. **Published machine descriptions** — peer-reviewed papers and IMAS documentation
   that describe specific machine configurations (e.g., JET PF coil geometry in
   NodeList PDFs).

3. **Cross-facility consensus** — if multiple facilities map the same signal type to
   the same IMAS path, that's strong evidence (e.g., PF coil R → `pf_active/coil/element/geometry/rectangle/r`).

4. **Expert review** — domain expert validates pipeline output. The pipeline produces
   evidence summaries to support this.

Initial implementation stubs the ground truth interface. The mapping pipeline can run
without ground truth (it uses its own validation step). Ground truth enables quantitative
benchmarking later.

### Benchmarking Harness

Reserve namespace but don't prioritize as CLI. The harness:
- Takes a set of ground-truth mappings (known-good signal→IMAS pairs)
- Runs the pipeline with different models
- Measures: path accuracy, unit conversion correctness, COCOS handling, confidence calibration
- Reports per-model scores

Future work: `imas-codex map benchmark --ground-truth <file> --models gemini-3-flash,claude-sonnet`

## Escalation Design

Escalation is not an afterthought — it's a first-class output of every pipeline step.

### Escalation Categories

| Category | Example | Resolution Path |
|----------|---------|-----------------|
| `cocos_ambiguity` | Sign convention unclear for magnetic field | Expert review of COCOS tables |
| `composite_source` | IDS field requires data from multiple signal groups | Manual composition or agentic (Option C) |
| `unit_ambiguity` | Signal has no unit metadata; pint can't verify | Check facility documentation |
| `no_matching_group` | IDS section has no corresponding signal group | Signal discovery gap — run more discovery |
| `low_confidence` | LLM confidence below threshold | Re-run with more context or different model |
| `path_not_found` | LLM proposed an IMAS path that doesn't exist | DD version mismatch or LLM hallucination |
| `transform_complex` | Required transform involves coordinate system changes | Manual code or agentic |

### Escalation Flow

```python
# After pipeline completes:
result = await generate_mapping(facility, ids_name)

for escalation in result.escalations:
    # Persisted to graph as MappingEvidence with type="escalation"
    # Queryable: which IDS fields still need attention?
    persist_escalation(facility, ids_name, escalation)

# CLI reports:
# ✓ 42/55 fields mapped (confidence > 0.8)
# ⚠ 8 fields mapped (confidence 0.5–0.8)
# ✗ 5 fields escalated:
#   - cocos_ambiguity: pf_active/coil/current/data (sign convention)
#   - composite_source: magnetics/flux_loop/flux/data (multiple signal sources)
```

## Prompt Templates

Three Jinja2 templates, one per pipeline step. Each uses `schema_needs` frontmatter
for automatic schema injection.

### Step 1: `imas_codex/llm/prompts/mapping/exploration.md`

Context injected:
- Signal groups summary (group_key, member_count, physics_domain, description)
- IMAS DD subtree (from `list_imas_paths` — tree structure with types)
- Semantic search results (from `search_imas` — scored IMAS paths)

Task: Assign each signal group to an IMAS structural section (STRUCT_ARRAY).

Schema providers needed:
- `imas_dd_subtree` — formatted DD tree structure
- `signal_groups` — signal group summary for facility + domain

### Step 2: `imas_codex/llm/prompts/mapping/field_mapping.md`

Context injected:
- Section assignment from Step 1
- Detailed IMAS field info (from `fetch_imas_paths` — docs, units, coordinates)
- Signal group member details (properties, units, representative metadata)
- Unit analysis (pint compatibility check results)
- COCOS analysis (sign-flip paths from `get_sign_flip_paths()`)

Task: Generate field-level mappings with transforms, units, COCOS.

Schema providers needed:
- `imas_field_detail` — detailed field info for the assigned section
- `unit_analysis` — pint compatibility results
- `cocos_analysis` — sign-flip path information

### Step 3: `imas_codex/llm/prompts/mapping/validation.md`

Context injected:
- Proposed field mappings from Step 2
- Validation results (path existence, unit consistency, completeness)
- Existing mappings for this facility/IDS (if any)
- Escalation flags from Step 2

Task: Review, correct, and finalize mappings. Flag remaining issues as escalations.

Schema providers needed:
- `validation_results` — structured validation output
- `existing_mappings` — what's already mapped

## Phases

### Phase 1: Schema & CLI Restructuring

**1a. Rename StructuralEpoch → SignalEpoch**
- Update `imas_codex/schemas/facility.yaml` (class, references)
- Update `imas_codex/discovery/mdsplus/graph_ops.py` (~20 references)
- Update all queries filtering on StructuralEpoch label
- Update tests
- Run `uv run build-models --force` to regenerate graph models

**1b. Remove hardcoded mapping specs**
- Delete constants from `imas_codex/ids/graph_ops.py` (L280–L600)
- Delete `seed_ids_mappings()` function
- Delete `ids seed` CLI subcommand
- Keep read-side utilities (`load_mapping`, `load_sections`, etc.)

**1c. CLI restructure**
- Create `imas_codex/cli/map.py` — top-level `map` group (discover pattern)
- Merge `ids` → `imas` with `dd` subgroup
- Register in `imas_codex/cli/__init__.py`
- Delete `imas_codex/cli/ids.py` (after merging into `imas`)

### Phase 2: Signal Grouping Unification

**2a. Common grouping layer**
- Create `imas_codex/discovery/base/grouping.py`
- Extract unified `create_signal_group()`, `claim_signal_groups()`, `propagate_group_enrichment()`
- Refactor `parallel.py` to use common layer
- Refactor `mdsplus/graph_ops.py` to use common layer
- Ensure groups reach terminal state (`enriched`) during discovery

**2b. Verify signal groups exist for mapping targets**
- Run signal discovery for JET
- Verify SignalGroup nodes exist for PF coil signals, magnetics, wall limiter
- If groups are missing, debug detection (the 0 count in current graph suggests
  grouping hasn't been run for JET's FacilitySignals yet)

### Phase 3: Mapping Pipeline

**3a. Tool functions**
- Create `imas_codex/ids/tools.py`
- Implement: `fetch_imas_subtree()`, `fetch_imas_fields()`, `search_imas_semantic()`,
  `analyze_units()`, `query_signal_groups()`, `check_imas_paths()`,
  `search_existing_mappings()`

**3b. Response models + adapter**
- Create `imas_codex/ids/models.py`
- Define: `SectionAssignment(Batch)`, `FieldMapping(Batch)`, `EscalationFlag`,
  `ValidatedMappingResult`, `AssemblyConfig`
- Adapter: `ValidatedMappingResult` → graph operations (MAPS_TO_IMAS, POPULATES,
  IMASMapping node, MappingEvidence)

**3c. Prompt templates**
- Create `imas_codex/llm/prompts/mapping/exploration.md`
- Create `imas_codex/llm/prompts/mapping/field_mapping.md`
- Create `imas_codex/llm/prompts/mapping/validation.md`
- Register schema providers in `prompt_loader.py`

**3d. Pipeline orchestrator**
- Create `imas_codex/ids/mapping.py`
- Implement `generate_mapping(facility, ids_name, ...)` — the 3-step flow
- Wire to `map` CLI

### Phase 4: End-to-End Validation

**4a. Generate JET pf_active mappings**
- Run: `imas-codex map jet pf_active`
- Review output: coverage, confidence, escalations
- Iterate on prompts if needed

**4b. Export and verify**
- Run: `imas-codex imas export jet pf_active --epoch <version>`
- Verify IDS structure is valid
- Repeat for all JET epochs

**4c. Repeat for pf_passive**
- Run: `imas-codex map jet pf_passive`
- Export and verify

### Phase 5: Ground Truth & Benchmarking (Future)

**5a. Ground truth interface**
- Define format for ground-truth mapping files
- Stub interface for querying external IMAS databases

**5b. Benchmarking harness**
- Compare models on ground-truth set
- Report accuracy metrics
- Select optimal model for production runs

## Files

### Create

| File | Purpose |
|------|---------|
| `imas_codex/cli/map.py` | Top-level `map` CLI group |
| `imas_codex/ids/mapping.py` | Multi-step pipeline orchestrator |
| `imas_codex/ids/models.py` | Step-specific Pydantic response models + adapter |
| `imas_codex/ids/tools.py` | Tool functions (IMAS subtree, unit analysis, validation) |
| `imas_codex/discovery/base/grouping.py` | Common signal grouping infrastructure |
| `imas_codex/llm/prompts/mapping/exploration.md` | Step 1 prompt template |
| `imas_codex/llm/prompts/mapping/field_mapping.md` | Step 2 prompt template |
| `imas_codex/llm/prompts/mapping/validation.md` | Step 3 prompt template |

### Modify

| File | Changes |
|------|---------|
| `imas_codex/schemas/facility.yaml` | StructuralEpoch → SignalEpoch |
| `imas_codex/cli/__init__.py` | Register `map`, merge `ids` → `imas` |
| `imas_codex/ids/graph_ops.py` | Remove hardcoded specs, keep read-side utils |
| `imas_codex/discovery/signals/parallel.py` | Refactor to use common grouping layer |
| `imas_codex/discovery/mdsplus/graph_ops.py` | Refactor to use common grouping layer; StructuralEpoch → SignalEpoch |
| `imas_codex/llm/prompt_loader.py` | Register new schema providers |
| `imas_codex/units/__init__.py` | Add unit compatibility analysis function |
| ~10 other files | StructuralEpoch → SignalEpoch references |

### Remove

| File/Code | Reason |
|-----------|--------|
| `imas_codex/cli/ids.py` | Merged into `imas` CLI |
| Hardcoded constants in `ids/graph_ops.py` | Replaced by LLM pipeline |
| `seed_ids_mappings()` in `ids/graph_ops.py` | No longer needed |
| `imas_codex/ids/recipes/jet/pf_active.yaml` | If exists — replaced by pipeline |

## Verification

1. Schema builds successfully after rename (`uv run build-models --force`)
2. `imas-codex map --help` shows discover-style layout
3. `imas-codex imas --help` shows merged CLI with `dd` subgroup
4. Tool functions return correct data (IMAS subtree, unit analysis, COCOS paths)
5. Step 1 LLM correctly assigns signal groups to IMAS sections
6. Step 2 LLM produces field mappings with valid paths, units, transforms
7. Step 3 LLM validates and emits consistent final result with escalations
8. Graph persistence creates MAPS_TO_IMAS + POPULATES + IMASMapping + MappingEvidence
9. `imas-codex imas export jet pf_active --epoch <version>` produces valid IDS
10. Escalation flags are persisted and queryable

## Open Questions

1. **Shot-to-epoch resolution**: `ids export` takes `--epoch VERSION` but no shot
   number resolution exists. Should `map` generate epoch-specific mappings or
   epoch-agnostic ones? (Likely: epoch-agnostic mappings, epoch selected at export time.)

2. **Model section**: Should mapping use `language` (gemini-3-flash) or `agent`
   (claude-sonnet) model tier? Start with `language` for cost, escalate to `agent`
   if quality is insufficient.

3. **Batch vs per-group**: Should Step 2 run once per section (batching all field
   mappings) or once per signal group? Per-section is more efficient; per-group
   gives better focus. Start with per-section, split if quality drops.

4. **MAPS_TO_IMAS vs IMASMapping node**: `mapping-redesign.md` proposes relationships
   instead of nodes. This plan follows that decision. If richer metadata is needed
   (e.g., multi-step provenance), revisit.
