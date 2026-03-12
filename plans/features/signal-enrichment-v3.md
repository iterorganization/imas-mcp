# Signal Enrichment Pipeline v3

> **Status**: Planning  
> **Priority**: Critical — enrichment quality directly determines mapping quality  
> **Scope**: `discover signals` CLI — signal scanning, LLM enrichment, structured output  
> **Principle**: LLM cost is not a constraint. Descriptions must be accurate and
> specific. Structured fields capture units, COCOS, and physics domain — do not
> conflate these into free-text descriptions. Deterministic context fetches use
> function calls, not LLM inference.

## LLM Routing & Caching

All LLM calls in the signal enrichment pipeline route through the LiteLLM
proxy (`imas-codex llm start`) which proxies via OpenRouter. This is critical
for prompt caching — `_build_kwargs()` in `discovery/base/llm.py` applies
`ensure_openrouter_prefix()` and `inject_cache_control()` to every call.

**Caching mechanism**: `inject_cache_control()` adds `cache_control: {"type":
"ephemeral"}` breakpoints to the last system message. The `openrouter/` prefix
is required — the `openai/` prefix strips `cache_control` blocks, silently
disabling prompt caching. All new LLM calls in this plan MUST use
`call_llm_structured()` or `acall_llm_structured()` from
`imas_codex.discovery.base.llm` — never direct `litellm.completion()`.

**Batching**: Signal enrichment already batches 50-200 signals per LLM call
via `SignalEnrichmentBatch`. All new LLM steps (Phase 4) MUST follow this
pattern — batch multiple items into a single structured output call. The system
prompt is static and cacheable; only the user prompt varies per batch.

## Separation of Concerns

This plan covers **signal discovery and enrichment only** — the `discover signals`
CLI tool. It does NOT cover IMAS mapping, which is a separate tool (`imas map`)
documented in a companion plan (`imas-map-context-v1.md`). The graph is the state
ledger between these two tools:

```
discover signals              graph              imas map
┌──────────────┐         ┌──────────┐      ┌──────────────┐
│ Scan         │────────▶│ Facility │      │              │
│ Enrich (LLM) │────────▶│ Signal   │◀────│ Read sources  │
│ Check        │────────▶│ Signal   │      │ Assign sect.  │
│ Embed        │         │ Source   │◀────│ Map fields    │
│              │         │          │      │ Validate      │
└──────────────┘         └──────────┘      └──────────────┘
```

The signals CLI enriches signals to understand them as physics measurements.
It assigns `physics_domain` from our LinkML enums, generates descriptions,
extracts units and sign conventions from metadata. It does NOT create IMAS
mappings — that is the responsibility of `imas map`.

---

## Phase 1: Normalize Signal Properties

**Goal**: Every FacilitySignal has consistent, queryable metadata regardless
of discovery source. The dual property + relationship model is maintained
per AGENTS.md schema design guidelines.

### 1.1 Fix `data_source_node` Dual Model Compliance

The `data_source_node` slot in the facility schema has `range: SignalNode`
and `relationship_type: HAS_DATA_SOURCE_NODE`. Per the dual model, BOTH
the property and the relationship must exist. Currently:

| Scanner | Property `s.data_source_node` | Edge `HAS_DATA_SOURCE_NODE` |
|---------|------------------------------|---------------------------|
| JET device_xml | ✅ set (SignalNode ID) | ✅ created |
| TCV tree_traversal | ❌ null | ✅ created (28,690 signals) |
| TCV tdi_scan | ❌ null | ❌ none |
| TCV wiki_extraction | ❌ null | ❌ none |

**Action 1 — Fix scanners**: Each scanner that creates a `HAS_DATA_SOURCE_NODE`
relationship must also set the `data_source_node` property. The device_xml
scanner already does this correctly. Fix the tree_traversal scanner to match.

**Action 2 — Graph migration** (run directly via Cypher, no script):

```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_node IS NULL
SET s.data_source_node = sn.id
```

### 1.2 Fix `data_source_path` Consistency

`data_source_path` is set by device_xml but null for tree_traversal. Set from
the SignalNode path for signals with the `HAS_DATA_SOURCE_NODE` edge.

**Action 1 — Fix scanners**: Set `data_source_path = signal_node.path` during
tree_traversal scanning when the SignalNode is available.

**Action 2 — Graph migration** (run directly via Cypher):

```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_path IS NULL
SET s.data_source_path = sn.path
```

### 1.3 Canonical vs Scanner-Specific Fields

**Canonical fields** (set by all scanners, available for shared enrichment):
- `accessor` — signal access path (required)
- `data_access` — how to read this signal (required, range `DataAccess`)
- `data_source_node` — backing SignalNode (property + edge, when applicable)
- `data_source_name` — tree or source name
- `data_source_path` — MDSplus path or equivalent
- `discovery_source` — which scanner found it
- `unit` — physical units (from metadata, not LLM)
- `name` — human-readable name

**Scanner-specific fields** (set only by specific scanners, legitimate schema
properties but NOT injected as universal enrichment context):
- `tdi_function` — TDI wrapper function (tdi_scan only)
- `tdi_quantity` — TDI quantity argument (tdi_scan only)

These fields appear in the per-signal section of the enrichment prompt **only
when present**, labeled by their scanner-specific origin.

### 1.4 Normalize `enrichment_source` Values

`mark_signals_enriched()` does not set `enrichment_source`, leaving it null
for directly-enriched signals. Only `propagate_signal_group_enrichment` sets
`signal_group_propagation`.

**Fix**: Set `enrichment_source = 'direct'` in `mark_signals_enriched()` and
`enrichment_source = 'direct_underspecified'` in `mark_signals_underspecified()`.

### 1.5 Add `--rescan` and `--reenrich` Flags

The `discover signals` CLI (at `imas_codex/cli/discover/signals.py`) currently
has `--scan-only` and `--enrich-only` but NO `--rescan` or `--reenrich` flags.
The wiki and code discovery CLIs have `--rescan`.

**`--rescan`**: Re-discover signals from data sources (re-scan trees, re-run
TDI functions) without re-enriching. Signals already in `enriched` or `checked`
status are not re-scanned — only `discovered` signals are re-scanned from their
data source. This is useful when the scanner code has been fixed (e.g., TDI
`source_code` serialization fix in commit `603604fd`).

Behavior: Reset status of matching signals to trigger re-scan, then run scan
phase. Does NOT re-enrich — enrichment is triggered separately by `--enrich`
on signals in `discovered` status.

**`--reenrich`**: Reset `enriched` signals back to `discovered` status so they
are re-enriched on the next run. Useful when the LLM model is upgraded, the
enrichment prompt is improved, or context injection has been enhanced (like this
plan). Can be scoped by scanner: `--reenrich -s tdi` to only re-enrich
TDI-sourced signals.

Behavior: Set `status = 'discovered'` and clear enrichment fields for matching
signals. Then run the enrichment phase normally.

Both flags should be documented in the CLI help text and follow the pattern
established by `discover wiki --rescan`.

---

## Phase 2: Fix Context Injection Gaps

**Goal**: Every enrichment context source reaches the LLM prompt with
maximum available information.

### 2.1 Fix Tree Context Filter (Critical Bug)

**File**: `imas_codex/discovery/signals/parallel.py` line 2950  
**Bug**: `tree_signal_ids = [s["id"] for s in signals if s.get("data_source_node")]`

This filters on the `data_source_node` *property*, which is null for all TCV
signals. After Phase 1.1 backfills the property, this filter will work
correctly. No code change needed to the filter itself — the property
normalization fixes it.

However, as a belt-and-suspenders approach, also consider Option B: send all
signal IDs to `fetch_tree_context()` and let the Cypher
`MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)` naturally filter. Signals
without the edge simply return no results. This is simpler and resilient to
future property gaps.

**Impact**: 28,690 TCV signals gain tree context (parent path, sibling paths,
TDI source code, epoch ranges).

### 2.2 Re-scan TDI Functions for `source_code`

**Problem**: All 189 TDI functions in the graph have empty `source_code`
fields. The graph data predates the scanner fix (commit `603604fd`).

**Action**: After adding `--rescan` (Phase 1.5):
```bash
uv run imas-codex discover signals tcv --rescan -s tdi
```
This is a deterministic step — no LLM calls. The scanner fix is already in the
codebase; only the graph data is stale.

### 2.3 Inject Deterministic Code References

**Problem**: 2,208 signals are reachable via:
```
CodeChunk →[CONTAINS_REF]→ DataReference →[RESOLVES_TO_NODE]→ SignalNode
   ←[HAS_DATA_SOURCE_NODE]← FacilitySignal
```
These code references show real usage (how the signal is read, units used,
transformations applied) but are never injected into the enrichment prompt.
Only `_fetch_code_context()` (vector search by group-level similarity) is used.

**Action**: Add `fetch_signal_code_refs()` — a deterministic graph traversal
per signal:
```cypher
MATCH (s:FacilitySignal {id: $signal_id})
      -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
      <-[:RESOLVES_TO_NODE]-(dr:DataReference)
      <-[:CONTAINS_REF]-(cc:CodeChunk)
RETURN cc.text AS code, cc.language AS language,
       cc.source_file AS file LIMIT 3
```
Inject as `## Direct Code References` in the per-signal prompt section, before
the existing group-level `## Source Code Context`.

### 2.4 Fix Wiki Path Matching for TCV

**Problem**: `_find_wiki_context()` looks up `data_source_path` from the signal
dict, but TCV signals have `data_source_path = null`. Wiki chunks with
`mdsplus_paths_mentioned` (1,829 chunks, 6,955 path references) are never
matched to TCV signals.

**Fix**: After Phase 1.2 backfills `data_source_path`, this resolves itself.
Additionally, add a fallback:
```python
path = signal.get("data_source_path") or signal.get("accessor")
```

---

## Phase 3: Structured Enrichment Output

**Goal**: Enrichment output captures physics metadata as structured fields,
not free-text descriptions. Descriptions describe the physics — units, COCOS,
and domain classification are orthogonal concerns stored in dedicated fields.

### 3.1 Rename `units_extracted` → `unit`

The `SignalEnrichmentResult` model field `units_extracted` should be renamed
to `unit` (singular) — matching the `unit` slot on `FacilitySignal` in the
schema. The plural/verbose name is inconsistent with the graph property name.

Update in:
- `imas_codex/discovery/signals/models.py`: rename field `units_extracted` → `unit`
- `imas_codex/llm/prompts/signals/enrichment.md`: all references
- `imas_codex/discovery/signals/parallel.py`: result handler field access

### 3.2 Separation of Concerns

The enrichment model separates concerns into dedicated fields:
- `physics_domain` — enum classification (from `PhysicsDomain`)
- `unit` — physical units (singular, from metadata)
- `sign_convention` — string field for sign convention
- `description` — free-text physics description
- `diagnostic` — diagnostic system name
- `keywords` — searchable terms

**The description should NOT contain units, COCOS indices, or other structured
data.** It should describe what the signal measures and its physics context.
The prompt should explicitly instruct the LLM NOT to include units or sign
conventions in descriptions — these belong in their dedicated fields.

### 3.3 Validate Extracted Units with Pint

The `unit` field is a plain string. Add a post-enrichment validation step
(deterministic, not LLM) that validates extracted units against the `pint`
unit registry — the same pattern used by `analyze_units()` in
`imas_codex/ids/tools.py`:

```python
from imas_codex.units import unit_registry

def validate_unit(unit_str: str) -> str | None:
    """Validate unit string against pint. Return canonical form or None."""
    try:
        q = unit_registry.Quantity(1.0, unit_str)
        return str(q.units)
    except Exception:
        return None
```

If the LLM-extracted unit is invalid, clear it rather than storing garbage.
If valid, store the pint-canonical form (e.g., `"ampere"` → `"A"`).

This runs as a post-processing step in `mark_signals_enriched()`, not as part
of the LLM call.

### 3.4 Context Quality Assessment

The current `context_quality` field (low/medium/high) is already implemented
in `SignalEnrichmentResult` and the enrichment prompt. Rather than defining
rigid thresholds, let the LLM assess quality based on the context it actually
receives. The existing prompt guidance is sufficient:

- `low` = only accessor/name, no source code, wiki, tree context
- `medium` = some context (tree path, group header, partial wiki)
- `high` = rich context (source code, wiki, code references, siblings)

Signals with `context_quality = low` are routed to `underspecified` status
and queued for re-enrichment when better context becomes available.

### 3.5 Enrichment Prompt: Description Guidance

Update the enrichment prompt to explicitly separate concerns:

```markdown
## Description Guidelines

Descriptions should capture the physics meaning of the signal:
- What physical quantity is measured
- The measurement technique or diagnostic principle
- Spatial/temporal characteristics (profile, time trace, scalar)
- Relevant physics context from tree structure and code references

Descriptions should NOT contain:
- Units (use the `unit` field)
- Sign conventions (use the `sign_convention` field) 
- COCOS indices (use the `cocos` field if applicable)
- Raw accessor paths or MDSplus tree addresses
```

### 3.6 Description Length

With improved context injection (tree context, code references, wiki), the LLM
receives significantly more information. Descriptions should be **longer and
more specific** than current output — 2-4 sentences capturing the physics
meaning and measurement context. The signal description is the primary input
for downstream IMAS mapping, so specificity is critical.

However, the description should remain a description — not a data sheet.
Structured fields handle the quantitative metadata.

---

## Phase 4: SignalSource Enrichment

**Goal**: Rename SignalGroup → SignalSource, and enrich SignalSource nodes
with group-level metadata.

### 4.1 Rename SignalGroup → SignalSource

The mapping source will not always be a group — a single signal can also be
a mapping source. Rename:

| Current | New |
|---------|-----|
| `SignalGroup` (schema class) | `SignalSource` |
| `SignalGroupStatus` (enum) | `SignalSourceStatus` |
| `USES_SIGNAL_GROUP` (relationship) | `USES_SIGNAL_SOURCE` |
| `signal_group` (slot on FacilitySignal) | `signal_source` |
| `MEMBER_OF` (relationship) | `MEMBER_OF` (keep — still accurate) |
| `detect_signal_groups()` | `detect_signal_sources()` |
| `propagate_signal_group_enrichment()` | `propagate_source_enrichment()` |
| `query_signal_groups()` | `query_signal_sources()` |
| `signal_group_id` (SectionAssignment) | `source_id` |

This is a schema change → `uv run build-models --force` → update all usages.
Per project philosophy: "When patterns change, update all usages — don't leave
old patterns alongside new."

**Graph migration** (run directly via Cypher, no migration script):
```cypher
// Rename node labels
MATCH (sg:SignalGroup) SET sg:SignalSource REMOVE sg:SignalGroup

// Relationship types cannot be renamed in Neo4j — must recreate
// USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE
MATCH (m:IMASMapping)-[r:USES_SIGNAL_GROUP]->(ss:SignalSource)
CREATE (m)-[:USES_SIGNAL_SOURCE]->(ss)
DELETE r
```

### 4.2 Structured LLM Enrichment for SignalSource Nodes

After enriching the representative signal, make a **batched** LLM call to
generate group-level metadata for multiple SignalSource nodes at once.
Uses `call_llm_structured()` from `imas_codex.discovery.base.llm` which
routes through the LiteLLM proxy → OpenRouter, preserving `cache_control`
breakpoints for prompt caching.

```python
class SignalSourceEnrichmentResult(BaseModel):
    """LLM enrichment for a single SignalSource node."""
    source_index: int = Field(description="1-based index matching input order")
    description: str = Field(
        description="What this collection of signals represents. "
        "Describe the shared measurement type and how members differ."
    )
    physics_domain: PhysicsDomain
    diagnostic: str = Field(default="")
    keywords: list[str] = Field(default_factory=list, max_length=5)
    member_variation: str = Field(
        description="How individual members differ within the source "
        "(e.g., 'spatial position', 'channel index', 'coil number')"
    )

class SignalSourceEnrichmentBatch(BaseModel):
    """Batch of SignalSource enrichments — multiple sources per LLM call."""
    results: list[SignalSourceEnrichmentResult]
```

**Batching**: Collect 20-50 SignalSource nodes per LLM call. The system
prompt (static: schema, PhysicsDomain enum, rules) is cacheable across
calls. Only the user prompt (per-batch source details) varies.

**Input context** (deterministic, via graph queries):
- Representative's enriched metadata
- List of all member accessors and names
- Tree context (parent paths, positions)

**Output**: Stored on the SignalSource node properties.

### 4.3 Integration with `discover clear`

The `discover clear -d signals` command currently clears FacilitySignal +
DataAccess + SignalEpoch nodes but NOT SignalSource nodes. This needs updating:

**Option A**: Add SignalSource clearing to `clear_facility_signals()` — since
SignalSource nodes are created by the signal discovery pipeline, they should be
cleared with it.

**Option B**: Keep SignalSource nodes when clearing signals, since they may
have IMAS mapping relationships. Flag them as `discovered` instead.

**Recommendation**: Option A for a clean slate. The `imas map clear` command
already handles clearing mapping relationships.

---

## Phase 5: Post-Propagation Individualization

**Goal**: After group enrichment is propagated, generate individualized
descriptions for member signals that capture their specific identity.

### 5.1 Design: Template-Based Individualization (Not N LLM Calls)

Currently `propagate_source_enrichment()` copies the representative's
description verbatim to all members. For a group of 181 `MAGB` probes, all
get the identical description.

Making N individual LLM calls is overkill — we already know the signal source
pattern and can enumerate members deterministically. Instead, use a
**single LLM call per source** to generate one description template with
placeholders, then enumerate programmatically:

**Step 1 — LLM generates template** (one call per source, batched across
sources using `SignalSourceIndividualizationBatch`):

```python
class SignalSourceIndividualization(BaseModel):
    """Template for individualizing member descriptions."""
    source_index: int = Field(description="1-based index matching input order")
    description_template: str = Field(
        description="Description template with {member_id} placeholder. "
        "E.g., 'Poloidal magnetic field measurement from probe {member_id} "
        "in the outboard midplane array.'"
    )
    variation_field: str = Field(
        description="Which part of the accessor varies across members "
        "(e.g., 'probe number', 'channel index', 'coil name')"
    )

class SignalSourceIndividualizationBatch(BaseModel):
    results: list[SignalSourceIndividualization]
```

**Step 2 — Deterministic enumeration** (no LLM):

```python
def individualize_members(source: dict, template: str, members: list[dict]) -> list[dict]:
    """Apply description template to each member signal."""
    results = []
    for member in members:
        # Extract the varying part from the accessor using the source pattern
        member_id = extract_member_identifier(source["group_key"], member["accessor"])
        description = template.format(member_id=member_id)
        results.append({"id": member["id"], "description": description})
    return results
```

**Why this works**: Signal sources are defined by accessor patterns
(e.g., `\\MAGNETICS::MAGB_{N}:BPOL`). The pattern itself tells us what
varies — the member identifier. The LLM's job is to write one good
description that includes the placeholder; enumeration is mechanical.

**Batching**: Collect 20-50 sources per LLM call in
`SignalSourceIndividualizationBatch`. The system prompt is static and
cacheable. Each source in the batch includes: source description,
member_variation field, representative accessor, 3 example member accessors.

Set `enrichment_source = 'individualized'` on updated members.

### 5.2 Pipeline Integration

The enrichment pipeline flow becomes:

```
1. detect_signal_sources()         — deterministic, create SignalSource nodes
2. enrich_worker()                 — LLM (batched), enrich representative signals
3. propagate_source_enrichment()   — deterministic, copy rep metadata to members
4. enrich_signal_sources()         — LLM (batched), generate source descriptions
5. individualize_members()         — LLM (1 batched call) + deterministic enumeration
```

Steps 4 and 5 run after the main enrichment loop completes for a batch. They
can be triggered by `--enrich-only` when running against already-enriched
signals.

**LLM call count**: For JET with ~100 sources, steps 4+5 require ~4-5 batched
LLM calls total (2-3 for source enrichment + 2-3 for individualization
templates). NOT 100+ individual calls.

---

## Phase 6: Prompt Architecture

**Goal**: Establish prompt ordering and caching principles as documented
engineering practice.

### 6.1 Add Prompt Ordering Guidance to AGENTS.md

The codebase follows static system prompt first, dynamic user prompt second —
optimal for LLM prompt caching. `inject_cache_control()` in `llm.py` adds
cache breakpoints to the last system message. This is undocumented.

Add to the **LLM Prompts** section of AGENTS.md:

```markdown
### Prompt Structure and Caching

All LLM calls route through the LiteLLM proxy → OpenRouter. Use
`call_llm_structured()` / `acall_llm_structured()` from
`imas_codex.discovery.base.llm` for all structured output calls — never
call `litellm.completion()` directly.

All prompts follow a **static-first ordering** to maximize prompt cache
hit rates via OpenRouter's prompt caching:

1. **System prompt** (static/quasi-static): Schema definitions, enum
   values, classification rules, output format. These change rarely and
   are shared across all LLM calls of the same type. `inject_cache_control()`
   sets a `cache_control: {"type": "ephemeral"}` breakpoint at the end of
   the system message.
2. **User prompt** (dynamic): Per-batch signal data, context chunks, and
   specific instructions. This varies per LLM call.

The `openrouter/` prefix is required on model identifiers — it preserves
`cache_control` blocks in message content. The `openai/` prefix strips
them, silently disabling prompt caching.

When building prompts, ensure that `{% include %}` blocks for schema
definitions and static rules appear **before** dynamic Jinja2 template
variables. This maximizes the cacheable prefix length.
```

### 6.2 Verify Existing Prompts

All current prompts follow static-first ordering:

| Prompt | Static Section | Dynamic Section | Order |
|--------|---------------|-----------------|-------|
| `enrichment.md` | Schema, physics domains, categories, rules | Per-batch signals | ✅ |
| `field_mapping.md` | Task, transform rules, escalation rules | Group detail, IMAS fields | ✅ |
| `exploration.md` | Task instructions | Groups, subtree | ✅ |

---

## Phase 7: Graph Data Migrations

**Goal**: Clean up legacy data. All migrations are run directly via Cypher
against the graph — no migration scripts written to file.

### 7.1 Migrations (Run Directly via Cypher)

Execute these Cypher statements directly against the graph (via the
`python()` MCP tool or `GraphClient().query()`). No script files needed.

**Migration 1 — Backfill `data_source_node` property from edge:**
```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_node IS NULL
SET s.data_source_node = sn.id
```

**Migration 2 — Backfill `data_source_path` from SignalNode:**
```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_path IS NULL
SET s.data_source_path = sn.path
```

**Migration 3 — Set `enrichment_source` for directly-enriched signals:**
```cypher
MATCH (s:FacilitySignal)
WHERE s.status IN ['enriched', 'checked']
  AND s.enrichment_source IS NULL
SET s.enrichment_source = 'direct'
```

**Migration 4 — Clean up legacy pattern properties:**
```cypher
MATCH (s:FacilitySignal)
WHERE s.pattern_representative_id IS NOT NULL
REMOVE s.pattern_representative_id, s.pattern_template
```

**Migration 5 — Normalize old `enrichment_source` values:**
```cypher
MATCH (s:FacilitySignal)
WHERE s.enrichment_source = 'pattern_propagation'
SET s.enrichment_source = 'signal_group_propagation'
```

**Migration 6 — Rename SignalGroup → SignalSource labels:**
```cypher
MATCH (sg:SignalGroup) SET sg:SignalSource REMOVE sg:SignalGroup
```

**Migration 7 — Recreate USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE relationships:**
```cypher
MATCH (m:IMASMapping)-[r:USES_SIGNAL_GROUP]->(ss:SignalSource)
CREATE (m)-[:USES_SIGNAL_SOURCE]->(ss)
DELETE r
```

### 7.2 Rename `_accessor_to_pattern()` in Code

Rename `_accessor_to_pattern()` → `_accessor_to_source_key()` in
`parallel.py` for consistency with the SignalSource schema.

---

## Phase 8: Data Reset and Re-Discovery

**Goal**: Once the enrichment pipeline is stable and tested, clear existing
signal data and re-run discovery for maximum quality.

### 8.1 Pre-Conditions (Must Complete Before Reset)

- [ ] Phase 1: Property normalization complete (scanner fixes + migrations run)
- [ ] Phase 2: Context injection gaps filled (tree filter, TDI source, code refs, wiki)
- [ ] Phase 3: Structured enrichment output validated (`unit` field, pint validation)
- [ ] Phase 4: SignalSource enrichment working
- [ ] Phase 5: Template-based individualization working
- [ ] Phase 6: Prompt ordering documented in AGENTS.md
- [ ] Phase 7: All migrations run directly via Cypher
- [ ] Integration tests pass: enrich → propagate → source enrich → individualize
- [ ] Dry run on ~100 signals verifying quality improvement

### 8.2 Reset Procedure

Signal discovery is scoped per facility and per scanner. Do not run everything
in a single pass:

```bash
# --- JET ---
# Clear existing signal data
uv run imas-codex discover clear jet -d signals

# Re-discover from device_xml scanner
uv run imas-codex discover signals jet -s device_xml

# Enrich discovered signals
uv run imas-codex discover signals jet --enrich-only

# Check signal access
uv run imas-codex discover signals jet  # full pipeline includes check

# --- TCV ---
# Clear existing signal data
uv run imas-codex discover clear tcv -d signals

# Re-discover from each scanner separately to control scope
uv run imas-codex discover signals tcv -s mdsplus --scan-only
uv run imas-codex discover signals tcv -s tdi --scan-only
uv run imas-codex discover signals tcv -s static --scan-only

# Then enrich all discovered signals
uv run imas-codex discover signals tcv --enrich-only

# --- Embed ---
uv run imas-codex embed index --labels FacilitySignal,SignalSource
```

### 8.3 Assessment: Code Bug vs Outdated Data

| Gap | Type | Fix |
|-----|------|-----|
| `data_source_node` property null for TCV | Code + Data | Fix scanners + run Cypher migration |
| `data_source_path` null for TCV | Code + Data | Fix scanners + run Cypher migration |
| TDI `source_code` empty (189 functions) | Outdated data | `--rescan -s tdi` |
| Tree context filter (line 2950) | Code bug | Fixed by Phase 1.1 property backfill |
| Wiki path matching TCV | Code + Data | Phase 1.2 backfill + accessor fallback |

**Recommendation**: Fix all code bugs (Phases 1-5), run migrations directly
via Cypher (Phase 7), verify on a small subset, then full data reset (Phase 8).

---

## Implementation Order

```
Phase 1 ──────────────────────────────────────────── Normalize Properties
  1.1  Fix data_source_node in scanners     [tree_traversal scanner]
  1.2  Fix data_source_path in scanners     [tree_traversal scanner]
  1.3  Document canonical vs specific       [plan only]
  1.4  Set enrichment_source in mark fns    [parallel.py]
  1.5  Add --rescan and --reenrich flags    [signals.py CLI]

Phase 2 ──────────────────────────────────────────── Context Gaps
  2.1  Fix tree context filter              [parallel.py or already fixed by 1.1]
  2.2  Re-scan TDI functions                [--rescan -s tdi]
  2.3  Add fetch_signal_code_refs()         [parallel.py + prompt injection]
  2.4  Fix wiki path matching               [parallel.py + fallback]

Phase 3 ──────────────────────────────────────────── Structured Output
  3.1  Rename units_extracted → unit        [models.py, prompt, parallel.py]
  3.2  Enforce description separation       [enrichment prompt]
  3.3  Add pint unit validation             [mark_signals_enriched]
  3.4  Keep context quality as LLM-assessed [no change needed]
  3.5  Update description guidance          [enrichment prompt]
  3.6  Allow longer descriptions            [prompt guidance]

Phase 4 ──────────────────────────────────────────── SignalSource
  4.1  Rename SignalGroup → SignalSource    [schema + code + run Cypher directly]
  4.2  SignalSource batched LLM enrichment  [new model + function via call_llm_structured]
  4.3  Update discover clear                [clear_facility_signals]

Phase 5 ──────────────────────────────────────────── Individualize
  5.1  Template-based individualization     [1 batched LLM call + deterministic enumeration]
  5.2  Wire into enrich_worker pipeline     [pipeline integration]

Phase 6 ──────────────────────────────────────────── Prompt Architecture
  6.1  Add prompt ordering to AGENTS.md     [documentation]
  6.2  Verify existing prompt ordering      [audit only]

Phase 7 ──────────────────────────────────────────── Migrations (Run Directly)
  7.1  Run all Cypher migrations directly   [GraphClient / python MCP tool]
  7.2  Rename _accessor_to_pattern()        [parallel.py]

Phase 8 ──────────────────────────────────────────── Reset + Re-Run
  8.1  Verify pre-conditions               [checklist]
  8.2  Scoped re-discovery per scanner      [CLI commands]
  8.3  Validate quality improvement         [comparison metrics]
```

---

## Deterministic vs LLM Steps

| Operation | Type | Implementation |
|-----------|------|----------------|
| Fetch tree context (parent, siblings, epochs) | Deterministic | `fetch_tree_context()` — Cypher |
| Fetch code references for signal | Deterministic | `fetch_signal_code_refs()` — Cypher (new) |
| Fetch wiki context by path | Deterministic | `_find_wiki_context()` — dict lookup |
| Fetch wiki context by semantics | Deterministic | `_fetch_group_wiki_context()` — vector search |
| Fetch code context by semantics | Deterministic | `_fetch_code_context()` — vector search |
| Fetch TDI source code | Deterministic | `get_tdi_source()` — graph query |
| Detect signal sources | Deterministic | `detect_signal_sources()` — accessor patterns |
| Validate extracted units | Deterministic | `validate_unit()` — pint registry |
| Enumerate individualized descriptions | Deterministic | `individualize_members()` — template + pattern |
| Generate signal description | **LLM** (batched) | `enrich_worker()` — `call_llm_structured(SignalEnrichmentBatch)` |
| Assess context quality | **LLM** (batched) | Part of `SignalEnrichmentBatch` structured output |
| Generate source descriptions | **LLM** (batched) | `call_llm_structured(SignalSourceEnrichmentBatch)` |
| Generate individualization templates | **LLM** (batched) | `call_llm_structured(SignalSourceIndividualizationBatch)` |

All LLM calls use `call_llm_structured()` → LiteLLM proxy → OpenRouter
with `cache_control` breakpoints on system prompts.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| TCV signals with tree context | 0 / 28,690 | 28,690 / 28,690 |
| TCV signals with `data_source_node` property | 0 / 28,690 | 28,690 / 28,690 |
| Signals with TDI source code | 0 / 338 | 338 / 338 |
| Signals with code references injected | 0 / 2,208 | 2,208 / 2,208 |
| Signals with wiki context (path match) | ~0 TCV | ~6,955 path matches |
| Unique descriptions (vs templated) | ~60% | >95% |
| `unit` validated by pint | 0% | 100% of non-empty |
| `enrichment_source` set | partial | 100% |
| SignalSource descriptions (group-specific) | 0 | All sources |
| `--rescan` flag available | ❌ | ✅ |
| `--reenrich` flag available | ❌ | ✅ |
