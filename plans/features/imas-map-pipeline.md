# Plan: IMAS Mapping Pipeline Overhaul

**Goal:** Transform the mapping pipeline from a single-shot, full-catalog LLM call into a physics-domain-scoped, batch-claim worker loop with source-level mapping, programmatic unit propagation, and rich progress display.

**Depends on:** Plan 1 (IMAS Path Enrichment) for enriched descriptions + embeddings on both sides.

---

## Phase 1: Fix Existing Issues

### 1.1 TCV SignalSource Status Bug

**Root cause:** 192 TCV SignalSources are in `discovered` status because their representatives were `skipped` (by the `channel_element` regex in `claim_signals_for_enrichment`). When the representative is skipped, `propagate_source_enrichment()` never runs, leaving the SignalSource in `discovered`.

**Why this happens:** The `channel_element` skip regex matches signals ending with numbered indices (e.g., `_001`, `:003`, `CHANNEL_007`). When a SignalSource group consists entirely of indexed channels (e.g., CHANNEL_001 through CHANNEL_192), the representative—chosen alphabetically, often CHANNEL_001—also matches the skip pattern.

**Why the original fix (propagating skip) is wrong:** Representatives should NEVER be skipped, regardless of their accessor pattern. The representative exists precisely to carry enrichment for its entire group. If we skip the representative, the whole group loses its enrichment. The fact that a signal "looks like" an array element is irrelevant if it's the representative—its description applies to all members.

**Correct fix:** Exclude representatives from the channel_element skip query. Representatives must always proceed to enrichment so their metadata propagates to all group members:

```python
# Skip channel_element signals EXCEPT those that are representatives
gc.query(
    """
    MATCH (s:FacilitySignal {facility_id: $facility})
    WHERE s.status = $discovered
      AND (
        s.accessor =~ '.*[_:]CHANNEL_?\\d{2,3}\\)?$'
        OR s.accessor =~ '.*[_:]\\d{2,3}\\)?$'
        OR s.accessor =~ '.*CHANNEL_?\\d{2,3}:.*'
        OR s.name =~ '^\\d{2,3}$'
      )
      // CRITICAL: Never skip representatives — they carry enrichment for their groups
      AND NOT EXISTS {
        MATCH (sg:SignalSource {representative_id: s.id})
      }
    SET s.status = $skipped,
        s.skip_reason = 'channel_element',
        s.claimed_at = null
    """,
    ...
)
```

**Graph cleanup (one-time):** Fix existing SignalSources by either (a) unskipping the representative so it can be enriched, or (b) if the representative was already processed long ago, copying its enrichment to the SignalSource:

```python
# Unskip representatives that were incorrectly skipped
MATCH (sg:SignalSource {facility_id: $facility, status: 'discovered'})
MATCH (rep:FacilitySignal {id: sg.representative_id, status: 'skipped', skip_reason: 'channel_element'})
SET rep.status = 'discovered', rep.skip_reason = null
```

**File:** `imas_codex/discovery/signals/parallel.py`

### 1.2 Remove Unit from LLM Responsibility

**Problem:** The LLM sometimes hallucinates units (e.g., `"A total amount of bootstrap current"`, `"keV"` for a position signal). Units should be **code-propagated only**, never LLM-generated.

**Changes:**

1. **Remove `unit` from `SignalEnrichmentResult`** — The LLM should not output units at all.

2. **Propagate units from SignalNode → FacilitySignal via code.** After the units_worker extracts units into SignalNode, add a propagation step:
   ```python
   MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
   WHERE sn.unit IS NOT NULL AND sn.unit <> ''
     AND (s.unit IS NULL OR s.unit = '')
   SET s.unit = sn.unit
   ```

3. **Remove unit-related instructions from enrichment prompt** — Strip the "Units Safety" section and `unit` field from the prompt.

4. **Keep `validate_unit()` as a safety net** on any remaining code paths that write units.

**Files:** `imas_codex/discovery/signals/models.py`, `imas_codex/discovery/signals/parallel.py`, `imas_codex/llm/prompts/signals/enrichment.md`

### 1.3 Remove Hardcoded DDA Fallback

**Problem:** `_get_dda_descriptions()` has a hardcoded fallback dict of ~30 JET DDAs. This should live in facility config.

**Fix (code):** Move the DDA descriptions to `imas_codex/config/facilities/jet.yaml` under `data_systems.ppf.dda_descriptions`. Add the facility config schema entry if not already present. Remove the hardcoded fallback from `parallel.py` entirely — if config is empty, the LLM gets no DDA context (which is honest).

**Retroactive extraction from existing wiki data (avoids re-ingestion):**

JET wiki pages that describe diagnostics are already ingested as WikiPage/WikiChunk nodes. We can extract DDA descriptions from this existing data instead of re-scraping:

1. **Query existing WikiChunks for DDA mentions:**
   ```python
   # Find WikiChunks that mention known DDAs
   MATCH (wc:WikiChunk)-[:HAS_CHUNK]-(wp:WikiPage {facility_id: 'jet'})
   WHERE wc.content =~ '(?i).*\\b(EFIT|HRTS|KK3|BOLO|MAGN|...)\\b.*'
   RETURN DISTINCT wp.title, wc.content, wp.url
   ```

2. **Semantic search for DDA descriptions:**
   ```python
   # For each known DDA, find chunks describing it
   for dda in ["EFIT", "HRTS", "KK3", "BOLO", ...]:
       results = semantic_search(
           query=f"JET {dda} diagnostic description purpose measurement",
           index="wiki_chunk_desc_embedding",
           filter={"facility_id": "jet"},
           top_k=5
       )
       # Chunks with high similarity to the DDA+description query likely contain its description
   ```

3. **One-time extraction script:**
   Create `scripts/extract_dda_descriptions.py` that:
   - Iterates over `_JET_KNOWN_DDAS` from `entity_extraction.py`
   - Queries WikiChunks using both keyword and semantic search
   - Extracts description snippets (either programmatically or with a focused LLM call)
   - Outputs YAML format for insertion into `jet.yaml`
   
   This is a one-time cost of ~30 LLM calls (one per DDA) vs. re-ingesting the entire JET wiki (hundreds of pages at non-negligible cost).

4. **Store in facility config:**
   ```yaml
   # imas_codex/config/facilities/jet.yaml
   data_systems:
     ppf:
       dda_descriptions:
         - code: EFIT
           description: "Equilibrium reconstruction using magnetic measurements"
         - code: HRTS
           description: "High Resolution Thomson Scattering for Te, ne profiles"
         # ... extracted from wiki chunks
   ```

5. **Hardcoded list updates:** Also sync `_JET_KNOWN_DDAS` in `entity_extraction.py` with the config so both stay in sync. Or better: load from config at runtime instead of hardcoding.

**Files:** `imas_codex/config/facilities/jet.yaml`, `imas_codex/discovery/signals/parallel.py`, `scripts/extract_dda_descriptions.py` (new)

### 1.4 Diagnostic Name — Enforce DiagnosticCategory Constraint

**Current state:** The `DiagnosticCategory` enum IS injected into the prompt via `diagnostic-categories.md`, but the `diagnostic` field on `FacilitySignal` is free-text. The LLM guidance says "use lowercase_snake_case matching the physical system" but doesn't enforce the enum.

**Fix:** Change the `diagnostic` field on `SignalEnrichmentResult` to use the `DiagnosticCategory` enum with an `Optional` wrapper (allow empty string for non-diagnostic signals). This constrains the LLM's structured output to only valid enum values. Signals that don't match a category get `diagnostic = ""`.

**Files:** `imas_codex/discovery/signals/models.py`

---

## Phase 2: Source-Level Mapping Architecture

### 2.1 Map at SignalSource Level, Not FacilitySignal Level

**Key principle:** The LLM generates mappings for **SignalSource** nodes only. All member FacilitySignals inherit identical source→target mappings. The assembly worker then unwinds the source group into struct-array entries.

**Current flow (problem):**
```
gather_context → ALL 2,300 signal sources → single LLM call for triage
```

**New flow:**
```
For each target IDS:
  1. Physics domain filter: keep only sources whose domain matches the IDS
  2. Semantic narrowing: rank candidates by embedding similarity
  3. LLM maps source → IMAS path (per group, not per member)
  4. Assembly: unwind source group into struct-array instances
```

### 2.2 Physics Domain Scoping

Both `SignalSource` nodes and `IMASNode` paths have physics domains. Use this to dramatically reduce candidates:

```python
# Fetch physics domains for target IDS
imas_domains = query("""
    MATCH (p:IMASNode)
    WHERE p.ids = $ids_name AND p.physics_domain IS NOT NULL
    RETURN DISTINCT p.physics_domain AS domain
""")

# Filter signal sources to matching domains
sources = query("""
    MATCH (sg:SignalSource {facility_id: $facility})
    WHERE sg.physics_domain IN $domains
      AND sg.status = 'enriched'
    RETURN sg.*
""")
```

For `pf_active`, the domains are `magnetic_field_systems`, `machine_operations`, etc. This filters 2,300 sources down to ~200.

### 2.3 Semantic Narrowing Within Domain

After physics domain filtering, use bidirectional semantic search:

**Forward:** For each IMAS section description → `facility_signal_desc_embedding` → top-K candidate sources
**Reverse:** For each source description → `imas_node_desc_embedding` (enriched by Plan 1) → top-K target paths

The intersection of forward + reverse matches produces high-confidence candidates.

Use the backing functions from MCP tools:
- `GraphSearchTool.search_imas_paths()` — hybrid vector + text search
- `GraphClustersTool.search_imas_clusters()` — cluster-level matching

### 2.4 Source Group Unwinding

After the LLM maps `SignalSource:jet:device_xml:pfcoils/NNN/r` → `pf_active/coil/element/geometry/rectangle/r`:

1. The mapping applies identically to all 13 members (pfcoils/1/r through pfcoils/13/r)
2. No per-member LLM calls needed
3. Assembly step maps member index → struct-array index
4. The `group_key` pattern (e.g., `device_xml:pfcoils/NNN/r`) tells the assembler that `NNN` varies across members
5. MAPS_TO_IMAS relationship is created on the SignalSource, not on individual signals

**The LLM never sees individual coil instances.** It sees:
```
Source: jet:device_xml:pfcoils/NNN/r (13 members)
  Description: Major radius position of PF coil
  Unit: m
  Sample accessors: device_xml:pfcoils/1/r, device_xml:pfcoils/2/r, ...
Target section: pf_active/coil
```

---

## Phase 3: DD Version Scoping

### 3.1 CLI `--dd-version` Flag Enhancement

The `--dd-version` flag already exists on `imas-codex imas map run`. Enhance it:

1. **Default to latest available version ≥ 4.x in the graph:**
   ```python
   def get_latest_dd_version(gc: GraphClient, min_major: int = 4) -> str:
       result = gc.query("""
           MATCH (v:DDVersion)
           WHERE v.id STARTS WITH $prefix
           RETURN v.id ORDER BY v.id DESC LIMIT 1
       """, prefix=f"{min_major}.")
       return result[0]["id"] if result else None
   ```

2. **Wire through to all tool functions.** Currently `dd_version` flows as an integer major version to filtering queries. Keep this — it scopes IMASNode queries to the correct version via `_dd_version_clause()`.

3. **Store full version string on IMASMapping node.** Already done — `ValidatedMappingResult.dd_version` holds the semver string.

### 3.2 Multi-Version Mapping Support

**Design principle:** The graph's epoched structure (INTRODUCED_IN, DEPRECATED_IN, HAS_PREDECESSOR) already supports multiple versions. Multiple IMASMapping nodes can coexist:

```
IMASMapping {id: "jet:pf_active:4.1.0", dd_version: "4.1.0"}
IMASMapping {id: "jet:pf_active:3.42.2", dd_version: "3.42.2"}
```

**Change the IMASMapping ID format** to include dd_version:
```
Current: "jet:pf_active" (only one mapping per facility+IDS)
New:     "jet:pf_active:4.1.0" (supports multiple versions)
```

Update `persist_mapping_result()` to use versioned IDs. Add `dd_version` to the match criteria so existing queries don't return wrong-version mappings.

### 3.3 COCOS Version Awareness

When mapping, query the target DD version's COCOS:
```python
dd_cocos = query("MATCH (v:DDVersion {id: $ver}) RETURN v.cocos", ver=dd_version)
```

Pass this to the LLM prompt alongside the source signal's COCOS (from `rep_cocos` on SignalSource). The LLM can then determine if a sign flip is needed.

COCOS labels (`ip_like`, `psi_like`, etc.) on target IMASNode paths are already in the graph. Include them in the mapping prompt context for each target field.

### 3.4 COCOS Label Workaround

The PR (IMAS-Data-Dictionary#214) to close the COCOS label gap is not yet merged. Current state: 435 paths have `cocos_label_transformation` in DD 4.1.0 — these are extracted from the DD XML during `build_dd_graph()`.

For missing labels: use `imas-python`'s `ids_convert._3to4_sign_flip_paths` (already used in `cocos/transforms.py`) to identify paths needing sign handling. Supplement with heuristic matching: if a path ends in `/ip`, `/current/data`, `/b_field_tor/data`, etc., flag for COCOS review.

---

## Phase 4: Batch Claim Worker Loop

### 4.1 Worker Architecture

Follow the established discovery worker pattern from `discovery/base/engine.py`:

```python
# imas_codex/ids/workers.py (new)

@dataclass
class MappingDiscoveryState(DiscoveryStateBase):
    target_ids: str
    dd_version: str
    dd_major: int
    dd_cocos: int
    target_domains: list[str]

async def map_worker(state: MappingDiscoveryState, on_progress) -> None:
    """Claim unmapped sources, generate mappings, persist."""
    while not state.should_stop():
        # Claim batch of unmapped SignalSources
        sources = claim_sources_for_mapping(state.facility, state.target_ids, 
                                              state.target_domains, batch_size=5)
        if not sources:
            state.map_phase.record_idle()
            await asyncio.sleep(1.0)
            continue
        
        for source in sources:
            # Semantic search for target candidates
            candidates = search_target_candidates(source, state)
            
            # LLM mapping call (per source, not per member)
            mapping = await generate_source_mapping(source, candidates, state)
            
            # Persist immediately  
            persist_source_mapping(source, mapping, state)
            
        state.map_stats.processed += len(sources)
```

### 4.2 Claim Pattern

Use the standard claim_token + ORDER BY rand() + @retry_on_deadlock pattern:

```python
@retry_on_deadlock()
def claim_sources_for_mapping(facility, ids_name, domains, batch_size=5):
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (sg:SignalSource {facility_id: $facility})
            WHERE sg.status = 'enriched'
              AND sg.physics_domain IN $domains
              AND NOT EXISTS { (sg)-[:MAPS_TO_IMAS]->(:IMASNode {ids: $ids_name}) }
              AND (sg.mapping_claimed_at IS NULL
                   OR sg.mapping_claimed_at < datetime() - duration('PT300S'))
            WITH sg ORDER BY rand() LIMIT $batch_size
            SET sg.mapping_claimed_at = datetime(), sg.mapping_claim_token = $token
        """, ...)
        
        return gc.query("""
            MATCH (sg:SignalSource {mapping_claim_token: $token})
            ...
        """, ...)
```

### 4.3 Rich Progress Display

Create `imas_codex/ids/progress.py` following the pattern from `imas_codex/discovery/signals/progress.py`. The existing `MappingProgressDisplay` placeholder should be expanded to show:

```
┌─────────────── JET → pf_active (DD 4.1.0, COCOS 17) ───────────────┐
│ SERVERS   graph:titan  models:iter (avg 2.1s)                        │
│                                                                       │
│ MAP        ████████████░░░░░░░░░░░░░░░░   12/45 sources  26.7%      │
│            mapping jet:device_xml:pfcoils/NNN/r → pf_active/coil     │
│ VALIDATE   ██████████░░░░░░░░░░░░░░░░░░    8/12 validated  66.7%    │
│                                                                       │
│ TIME       ████░░░░░░░░░░░░░░░░░░░░░░░░    3m 24s                   │
│ COST       ███░░░░░░░░░░░░░░░░░░░░░░░░░    $0.42       $2.00        │
│ STATS      mapped=12  validated=8  escalations=2  skipped=3          │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.4 CLI Integration

Wire into `imas_codex/cli/map.py` using `DiscoveryConfig` + `run_discovery()` from `imas_codex/cli/discover/common.py`:

```python
disc_config = DiscoveryConfig(
    domain="mapping",
    facility=facility,
    facility_config=config,
    display=display,
    check_graph=True,
    check_embed=True,
    check_model=True,
    check_ssh=False,
    model_section="language",
)
```

---

## Phase 5: Mapping Pipeline Steps (Revised)

### Step 0: Context Gathering (Programmatic)

For the target IDS + DD version:
1. `analyze_imas_structure(ids_name)` → section breakdown, field counts
2. `export_imas_ids(ids_name)` → full tree for prompt context
3. `get_sign_flip_paths(ids_name)` → COCOS paths
4. Query target IDS physics domains
5. Query `DDVersion.cocos` for target COCOS convention

### Step 1: Source Claiming (Per Domain)

For each physics domain matching the target IDS:
1. Claim batch of unmapped SignalSources in that domain
2. Each source carries: description, rep_unit, rep_sign_convention, rep_cocos, sample_accessors, member_count

### Step 2: Candidate Search (Per Source — Programmatic)

For each claimed source:
1. **Semantic search:** `search_imas_paths(source.description, ids_filter=ids_name, dd_version=dd_major)`
2. **Cluster matching:** `search_imas_clusters(query=source.description, ids_filter=ids_name)`
3. **Path context:** For top candidates, `fetch_imas_paths()` for full metadata
4. **Unit analysis:** `analyze_units(source.rep_unit, candidate.units)` for each candidate

### Step 3: LLM Mapping (Per Source)

Send to LLM:
- Source metadata (description, unit, COCOS, accessor pattern, member_count)
- Top-K candidate IMAS paths with descriptions, units, COCOS labels
- Code references if available
- Existing mappings for context

LLM returns: `source_id → target_path` with transform expression, confidence.

**The LLM maps at the source level — not per member.** The mapping is: "all PF coil radii map to `pf_active/coil/element/geometry/rectangle/r`".

Unit transforms are determined **by code** from the source unit and target unit, NOT by the LLM. The LLM only identifies the correct target path.

### Step 4: Assembly (Per Section — Programmatic)

After all sources in a section are mapped:
1. Collect all source→target bindings for the section
2. Determine struct-array population pattern from source group_key
3. Generate assembly config: how member signals map to struct-array indices
4. The signal source's `group_key` (e.g., `pfcoils/NNN/r`) provides the enumeration pattern
5. Member count → struct-array size

### Step 5: Validation (Programmatic)

Same as current: source/target existence, transform execution, unit compatibility, COCOS sign-flip enforcement, duplicate target detection.

### Step 6: Persist (Per Source)

Persist immediately after validation:
- Create/update `MAPS_TO_IMAS` on the SignalSource
- Create `IMASMapping` node (versioned ID)
- Create `POPULATES` relationships
- Clear claim

---

## Phase 6: Unit Handling — Code-Only Pipeline

### 6.1 Unit Propagation Chain

```
Scanner → SignalNode.unit → FacilitySignal.unit → SignalSource.rep_unit
       ↓
units_worker extracts from MDSplus metadata
       ↓
HAS_UNIT → Unit node
```

Add automatic propagation from SignalNode to FacilitySignal after units_worker:
```python
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE sn.unit IS NOT NULL AND sn.unit <> ''
  AND (s.unit IS NULL OR s.unit = '')
SET s.unit = sn.unit
```

### 6.2 Unit Transform in Mapping — Code-Determined

When the mapping pipeline identifies `source_unit` and `target_unit`:
```python
transform = analyze_units(source_unit, target_unit)
if transform["compatible"]:
    factor = transform["conversion_factor"]
    expression = f"value * {factor}" if factor != 1.0 else "value"
```

The LLM does NOT determine the transform expression for units. It only identifies the target path.

---

## Phase 7: Cross-Facility Learning

### 7.1 Exemplar Injection

When mapping `jet:pf_active` and `tcv:pf_active` is already mapped:
```python
existing = search_existing_mappings('tcv', 'pf_active', gc=gc)
```

Include TCV mappings as few-shot examples in the LLM prompt for JET. Same IDS structure, different signal names → the pattern transfers.

### 7.2 Mapping Transfer Heuristic

If a source from facility A maps to target T, and facility B has a source with:
- Same physics_domain
- Similar description (embedding similarity > 0.8)
- Same unit

Then auto-suggest the same mapping (with escalation flag for human review).

---

## Phase 8: Testing

### 8.1 Unit Tests

- Physics domain filtering reduces candidate count
- Source-level mapping creates correct MAPS_TO_IMAS relationships
- Assembly correctly unwinds source group to struct-array
- Unit transforms are code-determined, not LLM-determined
- DD version scoping filters correct paths
- Versioned IMASMapping IDs support multiple versions

### 8.2 Integration Tests

- End-to-end: `imas-codex imas map run jet pf_active --dd-version 4.1.0`
- Verify source-level mappings are created
- Verify assembly unwinds to correct struct-array size
- Verify cross-facility exemplar injection

---

## Implementation Order

```
Phase 1 (bug fixes) → independent, do first
Phase 2 (source-level architecture) → core design change
Phase 3 (DD version scoping) → affects ID format, do before persistence
Phase 4 (batch worker loop) → uses Phases 2+3
Phase 5 (revised pipeline steps) → core implementation
Phase 6 (unit code-only) → can be parallel with Phase 5
Phase 7 (cross-facility) → after first successful mapping
Phase 8 (testing) → continuous
```

**Phase 1** can ship immediately. **Phases 2-6** form the core overhaul. **Phase 7** is a follow-on optimization. The IMAS Path Enrichment plan (Plan 1) should ideally complete before Phase 5 to provide rich descriptions on the target side, but the pipeline can work with raw documentation + existing embeddings in the interim.

---

## Summary: All Recommendations Addressed

| Recommendation | Plan | Phase |
|---|---|---|
| TCV SignalSource status bug — exclude reps from skip | Plan 2 | 1.1 |
| Unit hallucination — remove from LLM | Plan 2 | 1.2, 6 |
| Hardcoded DDA fallback — extract from existing wiki | Plan 2 | 1.3 |
| Diagnostic enum enforcement | Plan 2 | 1.4 |
| Source-level mapping (not per-member) | Plan 2 | 2 |
| Physics domain pre-filtering | Plan 2 | 2.2 |
| Semantic narrowing / bidirectional search | Plan 2 | 2.3 |
| Source group unwinding | Plan 2 | 2.4 |
| DD version flag + scoping | Plan 2 | 3 |
| Multi-version support (epoched graph) | Plan 2 | 3.2 |
| COCOS version awareness | Plan 2 | 3.3 |
| COCOS label workaround | Plan 2 | 3.4 |
| Batch claim worker loop | Plan 2 | 4 |
| Rich progress display | Plan 2 | 4.3 |
| Boilerplate path pruning in prompts | Plan 1 | 2.5 |
| IMAS path LLM enrichment | Plan 1 | 2, 3 |
| Use MCP tool backing functions | Plan 1 | 2.2; Plan 2 | 5 |
| Cross-facility learning | Plan 2 | 7 |
| Adaptive batch sizing (signals) | Plan 2 | 1.2 (indirect) |
| Unit transform by code | Plan 2 | 6.2 |
| IMAS embedding coverage increase | Plan 1 | 3.4 |
