# IMAS Mapping Context Enrichment v1

> **Status**: Planning  
> **Priority**: High — mapping quality depends on context depth at each step  
> **Scope**: `imas map` CLI — signal-group-to-IMAS field mapping pipeline  
> **Dependency**: Requires signal enrichment v3 (companion plan) to be stable first  
> **Principle**: The mapping pipeline reads enriched signal metadata from the graph. 
> Better signal enrichment ⇒ better mapping input ⇒ better mapping output. The two
> pipelines are decoupled by the graph.

## LLM Routing & Caching

All LLM calls in the mapping pipeline route through the LiteLLM proxy
(`imas-codex llm start`) which proxies via OpenRouter. The `openrouter/`
prefix is required on model identifiers to preserve `cache_control` blocks
in message content — the `openai/` prefix strips them, silently disabling
prompt caching.

Mapping calls use `call_llm_structured()` from
`imas_codex.discovery.base.llm` (via the existing `_call_llm()` wrapper
in `ids/mapping.py`). Static system prompts (task description, transform
rules, COCOS rules) are cacheable across calls — dynamic context
(per-section signal source details, IMAS fields) varies per call.

The mapping pipeline already batches per-section: Step 1 is one LLM call
for all sources, Step 2 is one LLM call per section. No additional
batching changes needed — the pipeline is already efficient.

## Separation from Signal Enrichment

The `discover signals` CLI enriches facility signals — it generates
descriptions, extracts units, classifies physics domains, and creates
SignalSource nodes. All of that work lives in the companion plan
(`signal-enrichment-v3.md`).

This plan covers **only** the `imas map` CLI and the mapping pipeline
(`imas_codex/ids/mapping.py`), which reads enriched signal data from
the graph and generates IMAS field-level mappings.

```
signals CLI (enrichment)    │    imas map CLI (mapping)
────────────────────────────┼─────────────────────────────
scan → enrich → check       │    Step 0: gather context
FacilitySignal properties   │    Step 1: assign sections (LLM)
SignalSource creation       │    Step 2: field mappings (LLM)
physics_domain assignment   │    Step 3: programmatic validation
unit extraction             │    persist → activate
code ref injection          │    
────────────────────────────┼─────────────────────────────
        ▼ writes to graph   │    ▲ reads from graph
```

---

## Current Pipeline Architecture

The mapping pipeline in `imas_codex/ids/mapping.py` (`generate_mapping()`):

| Step | Type | Implementation | Purpose |
|------|------|----------------|---------|
| 0 | Deterministic | `_step0_gather_context()` | Fetch signal groups, IDS subtree, semantic search, existing mappings, COCOS paths |
| 1 | LLM | `_step1_assign_sections()` | Assign signal groups to IMAS struct-array sections |
| 2 | LLM | `_step2_field_mappings()` | Per section: generate field-level source→target mappings with transforms |
| 3 | Programmatic | `_step3_validate()` + `validate_mapping()` | Source/target existence, transform execution, unit compatibility, duplicate detection |
| Persist | Programmatic | `persist_mapping_result()` | Write IMASMapping node, POPULATES, USES_SIGNAL_GROUP, MAPS_TO_IMAS |

CLI commands (at `imas_codex/cli/map.py`):
```
imas-codex imas map run FACILITY IDS_NAME [-m model] [--dd-version] [--no-persist] [--no-activate]
imas-codex imas map status FACILITY [IDS_NAME]
imas-codex imas map show FACILITY IDS_NAME
imas-codex imas map validate FACILITY IDS_NAME
imas-codex imas map clear FACILITY IDS_NAME
imas-codex imas map activate FACILITY IDS_NAME
```

---

## Phase 1: Rename SignalGroup → SignalSource in Mapping Code

**Goal**: Align mapping pipeline code with the schema rename performed by
signal-enrichment-v3.md Phase 4.1.

### 1.1 Update Graph Queries

All Cypher in `ids/tools.py` and `ids/models.py` references `SignalGroup`:

| File | Location | Current | New |
|------|----------|---------|-----|
| `tools.py` | `query_signal_groups()` | `MATCH (sg:SignalGroup)` | `MATCH (sg:SignalSource)` |
| `tools.py` | `search_existing_mappings()` | `[:USES_SIGNAL_GROUP]->(sg:SignalGroup)` | `[:USES_SIGNAL_SOURCE]->(sg:SignalSource)` |
| `models.py` | `persist_mapping_result()` | `MATCH (sg:SignalGroup {id: $sg_id})` | `MATCH (sg:SignalSource {id: $sg_id})` |
| `models.py` | `persist_mapping_result()` | `MERGE (m)-[:USES_SIGNAL_GROUP]->(sg)` | `MERGE (m)-[:USES_SIGNAL_SOURCE]->(sg)` |
| `models.py` | `MappingEvidence` | `signal_group_id` property | `source_id` |

### 1.2 Update Model Field Names and Descriptions

In `ids/models.py`:

| Model | Field | Current Description | New Description |
|-------|-------|-------|-----|
| `SectionAssignment` | `signal_group_id` | "SignalGroup node id" | Rename field to `source_id`, description: "SignalSource node id" |
| `FieldMappingEntry` | `source_id` | "Source SignalGroup id" | "Source SignalSource id" |
| `EscalationFlag` | `source_id` | "SignalGroup id" | "SignalSource id" |

### 1.3 Update Prompts

In `exploration.md` and `field_mapping.md`:
- Replace "signal group" → "signal source" in instructions
- Update `signal_group_id` → `source_id` in output format descriptions
- Rename template variable `signal_groups` → `signal_sources` in exploration.md
- Rename template variable `signal_group_detail` → `signal_source_detail` in field_mapping.md

### 1.4 Rename Functions

| Current | New |
|---------|-----|
| `query_signal_groups()` | `query_signal_sources()` |
| `_format_groups()` | `_format_sources()` |

---

## Phase 2: Enrich Context at Step 0

**Goal**: The mapping LLM receives richer, more specific signal metadata
than just the SignalSource summary. Use deterministic graph queries and
function calls to inject structured metadata.

### 2.1 Extend `query_signal_sources()` Return Data

Currently returns: `id`, `group_key`, `description`, `keywords`,
`physics_domain`, `status`, `member_count`, `sample_members`, `imas_mappings`.

Add per-source:
- **Representative signal description**: The representative FacilitySignal's 
  enriched `description` field — the most detailed physics description.
- **Representative units**: `unit` field from the representative signal.
- **Representative sign_convention**: From the representative signal.
- **Member accessor examples**: Return first 10 member accessors (not just IDs).

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
       member_count, sample_members, sample_accessors,
       rep.description AS rep_description,
       rep.unit AS rep_unit,
       rep.sign_convention AS rep_sign_convention,
       rep.physics_domain AS rep_physics_domain,
       collect(DISTINCT {
           target_id: ip.id,
           transform: r.transform_expression,
           source_units: r.source_units,
           target_units: r.target_units
       }) AS imas_mappings
ORDER BY sg.group_key
```

### 2.2 Inject Code References per Source

Add `fetch_source_code_refs()` to `ids/tools.py` — the mapping-side
equivalent of the new `fetch_signal_code_refs()` from the signals plan:

```cypher
MATCH (sg:SignalSource {id: $source_id})
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
       -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
       <-[:RESOLVES_TO_NODE]-(dr:DataReference)
       <-[:CONTAINS_REF]-(cc:CodeChunk)
RETURN cc.text AS code, cc.language AS language,
       cc.source_file AS file LIMIT 5
```

This returns actual code showing how the signal is read, what units it
expects, and what transforms are applied. This is **critical** for
generating correct `transform_expression` values.

Include in `signal_source_detail` when rendering `field_mapping.md`.

### 2.3 Inject COCOS Context per Source

The current `get_sign_flip_paths()` returns IMAS paths requiring sign
flips. But it doesn't connect to the signal's COCOS convention.

If the representative signal has a `cocos` field set, inject it:
```markdown
### Source COCOS
Signal COCOS convention: {{ source_cocos }}
Target IMAS paths requiring sign flip: {{ cocos_paths }}
```

This gives the LLM concrete context to decide whether to include
`-value` in `transform_expression`.

### 2.4 Unit Analysis: Use Pint-Validated Units

After signal-enrichment-v3 Phase 3.3, signal units are pint-validated
and stored in the `unit` field (renamed from `units_extracted`).
The `_format_unit_analysis()` function currently extracts units from
`keywords` (looking for `unit:` prefix). Instead, use the
`rep_unit` field from Phase 2.1:

```python
def _format_unit_analysis(
    sources: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> str:
    lines: list[str] = []
    for src in sources:
        signal_unit = src.get("rep_unit")
        if not signal_unit:
            continue
        for f in fields:
            imas_unit = f.get("units")
            if imas_unit:
                result = analyze_units(signal_unit, imas_unit)
                # ... format as before
```

This replaces the fragile keyword-based unit extraction with the
direct structured field.

---

## Phase 3: Prompt Improvements

**Goal**: Improve the mapping prompts with better structure, richer
context injection, and static-first ordering for cache hits.

### 3.1 Field Mapping Prompt: Richer Source Context

The `field_mapping.md` template currently receives `signal_group_detail`
as a raw JSON dump of the `query_signal_groups()` row. Restructure to
highlight the most useful information:

```markdown
### Signal Source: {{ source_id }}

**Description**: {{ rep_description }}
**Physics Domain**: {{ physics_domain }}
**Units**: {{ rep_unit | default("unknown") }}
**Sign Convention**: {{ rep_sign_convention | default("unknown") }}
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

This replaces the raw JSON dump with a structured, readable format
that the LLM can reason about more effectively.

### 3.2 Exploration Prompt: Use physics_domain for Matching

The Step 1 (section assignment) prompt lists signal sources but
doesn't emphasize physics domain. Since signal-enrichment-v3 assigns
`physics_domain` from the `PhysicsDomain` enum, surface it prominently:

```markdown
- {{ source.id }} (domain={{ source.physics_domain }}, members={{ source.member_count }}): {{ source.description }}
```

This helps the LLM match sources to IDS sections by physics domain
overlap.

### 3.3 Static-First Prompt Ordering

Both mapping prompts already follow static-first ordering (task
description and rules first, dynamic context second). Verify and
document this matches the convention from AGENTS.md.

Current ordering in `field_mapping.md`:
1. Task description (static) ✅
2. Context variables (dynamic) — facility, ids_name, section, source detail
3. IMAS fields (dynamic)
4. Unit analysis (dynamic)
5. COCOS paths (quasi-static)
6. Existing mappings (dynamic)
7. Task + Transform Rules (static) ⚠️ should be earlier

**Fix**: Move the "Task" and "Transform Rules" sections before the
dynamic context, so the static instructions are part of the cacheable
prefix:

```
1. System prompt (static): "You are an IMAS mapping expert"
2. User prompt:
   a. Task description + Transform Rules (static/quasi-static)
   b. IMAS section fields (quasi-static per IDS)
   c. COCOS paths (quasi-static per IDS)
   d. Signal source detail (dynamic per source)
   e. Unit analysis (dynamic per source)
   f. Existing mappings (dynamic)
```

---

## Phase 4: Mapping-Side Validation Improvements

**Goal**: Strengthen Step 3 programmatic validation with additional checks
that leverage the richer signal metadata.

### 4.1 Unit Compatibility as Hard Validation

Currently `validate_mapping()` checks unit compatibility via `analyze_units()`.
After signal-enrichment-v3 ensures pint-validated units on signals, strengthen
this to a hard validation:

- If `source_units` and `target_units` are both set AND incompatible
  dimensionally → escalation severity = `error` (not `warning`)
- If `transform_expression = "value"` but `source_units ≠ target_units` →
  escalation: transform should include conversion

### 4.2 Physics Domain Cross-Check

After signals have canonical `physics_domain` from enrichment, validate
that the assigned IMAS section's physics domain is compatible. For example,
a source with `physics_domain = 'magnetics'` should not be mapped to
`core_profiles/profiles_1d/ion`.

This is a soft check (warning) — there are legitimate cross-domain mappings.

### 4.3 COCOS Sign-Flip Enforcement

If a target path appears in `get_sign_flip_paths()` AND the source signal
has a `cocos` field set, validate that the `transform_expression` includes
sign handling. If `transform_expression = "value"` for a sign-flip path →
escalation.

---

## Phase 5: Coverage and Quality Metrics

**Goal**: Extend mapping quality assessment using the richer signal metadata
from signal-enrichment-v3.

### 5.1 Signal Source Coverage per IDS

The `map validate` command already computes leaf field coverage and signal
group coverage via `compute_coverage()` and `compute_signal_coverage()`.

Extend to report:
- **Mapped vs enriched**: Of all enriched SignalSource nodes with matching
  `physics_domain`, how many have MAPS_TO_IMAS for this IDS?
- **Underspecified sources**: How many sources assigned to this IDS have
  `status = 'discovered'` (not enriched)? These may improve with better
  enrichment.

### 5.2 Mapping Confidence Distribution

Report distribution of mapping confidences:
- Low (<0.5): Flag for review
- Medium (0.5-0.8): Acceptable but could improve
- High (>0.8): Confident

---

## Implementation Order

```
Phase 1 ──────────────────────────── Rename SignalGroup → SignalSource
  1.1  Update graph queries         [ids/tools.py, ids/models.py]
  1.2  Rename model fields          [ids/models.py]
  1.3  Update prompts               [exploration.md, field_mapping.md]
  1.4  Rename functions             [ids/tools.py, ids/mapping.py]

Phase 2 ──────────────────────────── Enrich Context at Step 0
  2.1  Extend query_signal_sources  [ids/tools.py] 
  2.2  Add fetch_source_code_refs   [ids/tools.py]
  2.3  Add COCOS per-source context [ids/mapping.py, field_mapping.md]
  2.4  Use pint-validated units     [ids/mapping.py]

Phase 3 ──────────────────────────── Prompt Improvements
  3.1  Restructure source context   [field_mapping.md]
  3.2  Surface physics_domain       [exploration.md]
  3.3  Static-first prompt ordering [field_mapping.md]

Phase 4 ──────────────────────────── Validation  
  4.1  Unit compatibility hardening [ids/validation.py]
  4.2  Physics domain cross-check   [ids/validation.py]
  4.3  COCOS sign-flip enforcement  [ids/validation.py]

Phase 5 ──────────────────────────── Coverage Metrics
  5.1  Signal source coverage       [ids/validation.py, cli/map.py]
  5.2  Confidence distribution      [cli/map.py]
```

---

## Dependencies on Signal Enrichment v3

| This plan (mapping) | Requires from signals plan |
|---------------------|---------------------------|
| Phase 1 (rename) | Phase 4.1: SignalGroup → SignalSource schema + Cypher migration |
| Phase 2.1 (rep metadata) | Phase 1.4: `enrichment_source` set |
| Phase 2.2 (code refs) | Phase 2.3: `fetch_signal_code_refs()` pattern |
| Phase 2.4 (pint units) | Phase 3.3: pint-validated `unit` field (renamed from `units_extracted`) |
| Phase 3.2 (physics domain) | Phase 3.2: physics_domain reliably set |
| Phase 4.2 (domain check) | Phase 3.2: physics_domain on all enriched signals |
| Phase 4.3 (COCOS) | Signal `cocos` field populated (future) |

These dependencies mean the mapping plan should be implemented AFTER
signal-enrichment-v3 Phases 1-5 are stable.

---

## Design Decisions

### Should the signals CLI suggest target IDS sections?

**No.** The signal enrichment pipeline assigns `physics_domain` from the
`PhysicsDomain` enum, which is a physics classification — not a mapping
target. The mapping pipeline's Step 1 (section assignment) is responsible
for connecting signal sources to specific IMAS sections. Conflating these
would create coupling between the two pipelines.

The `physics_domain` field on enriched signals provides enough signal for
Step 1 to work effectively without the signals CLI needing to know about
IMAS structure.

### Should code references be injected at enrichment time or mapping time?

**Both.** Code references serve different purposes:
- At **enrichment time** (signals plan Phase 2.3): Help the LLM understand what
  the signal measures, extract units, classify physics domain. Context for
  description generation.
- At **mapping time** (this plan Phase 2.2): Show how the signal is read
  and transformed, informing `transform_expression` generation. Context for
  mapping accuracy.

The same code chunks serve both needs but are used differently.

### Should the mapping pipeline be re-run after signal re-enrichment?

**Yes, but manually.** After `--reenrich` improves signal descriptions,
existing mappings may be stale. Use:
```bash
uv run imas-codex imas map clear jet pf_active
uv run imas-codex imas map run jet pf_active
```

Do not auto-trigger mapping re-runs from signal enrichment — the two
pipelines are decoupled by the graph.
