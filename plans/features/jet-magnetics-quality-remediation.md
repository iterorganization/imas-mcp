# JET Magnetics & PF Active Signal Quality Remediation

> **Status**: Planning  
> **Priority**: Critical — blocks EFIT-quality IMAS mapping for magnetics/PF active  
> **Goal**: Fix enrichment quality for JET device_xml, magnetics_config, and related  
> signal populations so that descriptions are accurate, individualized, and carry  
> sufficient metadata for downstream IMAS IDS field mapping (magnetics, pf_active,  
> pf_passive IDSs).

## Problem Summary

A data quality assessment of ~55,400 JET FacilitySignals revealed systemic
description degradation across the magnetics machine description population.
These signals are critical for EFIT reconstruction — if descriptions and metadata
are inaccurate, downstream IMAS mappings to `magnetics.bpol_probe`,
`pf_active.coil`, `pf_passive.loop`, and related IDS structures will fail.

### Root Causes Identified

#### RC-1: SignalNode Description Not Injected into Enrichment Prompt

**Location**: `parallel.py` lines 3200-3500 (user prompt construction)

The `fetch_tree_context()` function (line 590-660) queries the
`HAS_DATA_SOURCE_NODE→SignalNode` edge but returns only `tree_path`,
`parent_path`, `sibling_paths`, `tdi_source`, and `epoch_range`. It does
**NOT** return `n.description` — the SignalNode's own description property.

For device_xml signals, SignalNode descriptions are *rich*:
```
"Magnetic Probe 1, Radial position=4.292m, Vertical position=0.604m, Orientation angle=-74.1deg"
```

This data is created by the device_xml scanner (`scanners/device_xml.py`
lines 340-370) which builds descriptions from actual parsed XML values.
But this rich context is never passed to the LLM enrichment prompt.

The scanner also creates FacilitySignals with basic descriptions:
```python
description=f"{field_meta['desc']} of {meta['label']} {inst_id}"
# → "Radial position of magnetic probe 10"
```

This basic description IS passed (as `existing_description:` in the prompt),
but it lacks the actual numeric values and sibling field data that would
anchor the LLM's output.

**Impact**: The LLM has no access to the actual geometry values when
enriching device_xml signals. It sees only accessor patterns and basic
field descriptions, leading to hallucinated or generic descriptions.

#### RC-2: All device_xml Signals Grouped into One Context Group

**Location**: `parallel.py` lines 3040-3070 (`_signal_context_key`)

```python
dsn = signal.get("data_source_name")
if dsn == "device_xml":
    return "device_xml"
```

ALL device_xml signals — magprobes, pfcoils, pfpassive, flux loops,
pfcircuits, limiter contours — are grouped into a *single* context group
`"device_xml"`. The LLM receives a single group header followed by dozens
of heterogeneous signals (probe positions mixed with coil turns mixed with
passive structure resistance).

This overloads the LLM context and prevents section-specific enrichment
guidance (e.g., magprobes should mention "magnetic probe position"; pfcoils
should mention "PF coil geometry").

#### RC-3: Blind Signal Source Propagation

**Location**: `parallel.py` lines 1293-1395 (`propagate_source_enrichment`)

The `detect_signal_sources()` function groups device_xml signals by normalized
accessor pattern. For example, all `device_xml:magprobes/NNN/r` signals form one
SignalSource. The first signal alphabetically (deterministic but arbitrary) is the
"representative", and the LLM enrichment for that single representative is copied
verbatim to ALL members via `propagate_source_enrichment()`.

Critical flaw: **both `name` and `description` are copied identically**. If the
representative is probe 10, then `name = "Probe 10"` and `description` referencing
"probe 10" are copied to probes 1-191. This was confirmed in 544+ signals.

The `individualize_source_descriptions()` function (line 3681) exists as a post-hoc
fix that runs AFTER all enrichment/propagation completes. It generates a
`{member_id}` description template via a separate LLM call. However:

1. It only generates a simple template string — no physics-aware individualization
2. It runs after the damage is done (probe numbers already wrong in graph)
3. For device_xml signals, the varying part may not adequately capture the physics
   difference (e.g., probe 1 at R=4.292m vs probe 100 at R=2.891m)

#### RC-4: SECTION_METADATA Not Used in Enrichment

**Location**: `scanners/device_xml.py` lines 38-112

The device_xml scanner has a rich `SECTION_METADATA` dictionary mapping each
section (magprobes, pfcoils, pfpassive, flux, pfcircuits) to:
- `physics_domain` (correct domain assignment)
- `imas_ids` (target IMAS IDS path, e.g., `magnetics.bpol_probe`)
- `system` code (e.g., "MP", "PF", "FL")
- Per-field metadata with `unit` and `desc`
- Human-readable `label` (e.g., "magnetic probe", "PF coil")

This metadata is used during *scanning* to create basic signal properties, but is
NOT passed to the enrichment prompt. The LLM must independently rediscover what
"magprobes/10/r" means, despite this being deterministically known.

#### RC-5: No `is_static` Flag

No metadata distinguishes static machine description data from time-varying
measurement signals. This affects:
- Enrichment (static data needs different prompting)
- Check validation (static data should check SignalNode values, not MDSplus reads)
- IMAS mapping (static → `machine_description`, dynamic → time-varying IDS arrays)

#### RC-6: `context_quality` Not Persisted

`SignalEnrichmentResult` computes `context_quality` (low/medium/high) but
`mark_signals_enriched()` does not persist it on the FacilitySignal node.
This prevents analytics on enrichment quality and targeted re-enrichment.

---

## Plan A: Enrichment Context Injection (Critical)

**Estimated Benefit**: HIGH — directly fixes hallucinated descriptions for 4,659
device_xml + 2,557 magnetics_config signals by providing the LLM with actual values.

### A.1 Inject SignalNode Description into Enrichment Prompt

**File**: `imas_codex/discovery/signals/parallel.py`

Modify `fetch_tree_context()` (line 610) to also return `n.description`:

```cypher
MATCH (s:FacilitySignal {id: sid})-[:HAS_DATA_SOURCE_NODE]->(n:SignalNode)
...
WITH s.id AS signal_id,
     n.path AS tree_path,
     n.description AS node_description,    -- ADD THIS
     parent.path AS parent_path,
     ...
```

Then in the user prompt construction (line ~3450), inject it per-signal:

```python
sig_tree_ctx = tree_context.get(signal["id"])
if sig_tree_ctx:
    if sig_tree_ctx.get("node_description"):
        user_lines.append(
            f"source_node_description: {sig_tree_ctx['node_description']}"
        )
```

Update the enrichment prompt (`signals/enrichment.md`) to document this field:

```markdown
When `source_node_description` is provided:
- This is the description from the backing SignalNode (data source node)
- For device_xml signals, this contains actual geometry values (R, Z positions,
  angles, turns) — use these specific numbers in your description
- This is authoritative data — do NOT substitute different values
```

### A.2 Sub-group device_xml Signals by Section

**File**: `imas_codex/discovery/signals/parallel.py`

Change `_signal_context_key()` to sub-group device_xml signals by section:

```python
dsn = signal.get("data_source_name")
if dsn == "device_xml":
    # Sub-group by section (magprobes, pfcoils, pfpassive, flux, pfcircuits)
    dsp = signal.get("data_source_path", "")
    if "/" in dsp:
        section = dsp.split("/")[0]
        return f"device_xml:{section}"
    return "device_xml"
```

Add section-specific group headers in the prompt construction:

```python
elif group_key.startswith("device_xml:"):
    section = group_key.split(":")[1]
    # Import SECTION_METADATA for enrichment context
    from imas_codex.discovery.signals.scanners.device_xml import SECTION_METADATA
    meta = SECTION_METADATA.get(section, {})
    label = meta.get("label", section)
    imas_ids = meta.get("imas_ids", "")
    user_lines.append(f"\n## JET Device Description: {label}")
    user_lines.append(
        f"Hardware geometry data for {label}s from JET EFIT device XML. "
        f"These are static configuration values, NOT time-varying measurements."
    )
    if imas_ids:
        user_lines.append(f"IMAS IDS target: {imas_ids}")
    fields = meta.get("fields", {})
    if fields:
        field_desc = ", ".join(
            f"{f} ({fd['desc']}, {fd['unit']})" for f, fd in fields.items()
        )
        user_lines.append(f"Available fields per instance: {field_desc}")
```

### A.3 Persist `context_quality` on FacilitySignal

**File**: `imas_codex/discovery/signals/parallel.py`

In `mark_signals_enriched()` (line ~880), add `context_quality` to the SET clause:

```python
s.context_quality = sig.context_quality,
```

This requires adding `context_quality` to the enrichment result dict passed
to `mark_signals_enriched()`. In the enrichment worker where results are built
(line ~3620), add:

```python
entry = {
    ...
    "context_quality": result.context_quality.value,
}
```

Add `context_quality` to the FacilitySignal schema if not already present.

---

## Plan B: Group-to-Child Description Unwinding (Critical)

**Estimated Benefit**: HIGH — eliminates the "Probe 10 everywhere" anti-pattern
for 544+ signals, and provides physics-aware individualization that captures
actual differences between group members (different positions, coil numbers).

### B.1 Code-Generation Approach for Group Unwinding

Instead of simple `{member_id}` template substitution (current
`individualize_source_descriptions`), implement a code-generation approach
where the LLM generates Python code that deterministically produces
individualized descriptions for each group member.

**New Pydantic model** in `imas_codex/discovery/signals/models.py`:

```python
class SignalSourceCodeUnwind(BaseModel):
    """LLM-generated Python code to individualize group member descriptions.

    The LLM receives the group description, representative's enrichment,
    SECTION_METADATA, and sample SignalNode descriptions for several members.
    It produces Python code that takes (member_accessor, member_node_description)
    and returns an individualized (name, description) tuple.
    """
    source_index: int = Field(
        description="1-based index matching the input source order"
    )
    python_code: str = Field(
        description="Python function body. The function signature is: "
        "def individualize(accessor: str, node_description: str, member_index: str) "
        "-> tuple[str, str]. Returns (name, description)."
    )
    variation_field: str = Field(
        description="What varies across members (e.g., 'probe number', 'coil index')"
    )
```

**New prompt template** `imas_codex/llm/prompts/signals/source_unwind.md`:

The prompt provides:
1. The group's representative description and metadata
2. SECTION_METADATA for the relevant section (fields, units, IMAS IDS target)
3. 3-5 sample SignalNode descriptions showing actual values for different members
4. The accessor pattern showing what varies

The LLM generates a Python function that:
- Extracts the member index from the accessor
- Uses the node_description (which contains actual R, Z, angle values) to build
  a physics-specific description mentioning the actual component number and values
- Returns a proper (name, description) tuple

**Example for magprobes/NNN/r**:

Input to LLM:
```
Group: device_xml:magprobes/NNN/r
Representative description: "Radial position of magnetic probe in the poloidal field measurement array."
SECTION_METADATA: {"label": "magnetic probe", "imas_ids": "magnetics.bpol_probe", "fields": {"r": {"unit": "m", "desc": "Radial position"}, ...}}
Sample member node descriptions:
- magprobes/1: "Magnetic Probe 1, Radial position=4.292m, Vertical position=0.604m, Orientation angle=-74.1deg"
- magprobes/10: "Magnetic Probe 10, Radial position=5.036m, Vertical position=0.185m, Orientation angle=88.0deg"
- magprobes/100: "Magnetic Probe 100, Radial position=2.891m, Vertical position=-1.340m, Orientation angle=161.7deg"
```

LLM-generated Python code:
```python
def individualize(accessor: str, node_description: str, member_index: str) -> tuple[str, str]:
    name = f"Magnetic Probe {member_index} Radial Position"
    # Parse R value from node description if available
    r_val = ""
    if node_description:
        import re
        m = re.search(r"Radial position=([0-9.]+)m", node_description)
        if m:
            r_val = f" (R={m.group(1)}m)"
    description = (
        f"Major radius (R) coordinate of magnetic probe {member_index} "
        f"in the JET poloidal field measurement array{r_val}. "
        f"Used for magnetic equilibrium reconstruction (EFIT)."
    )
    return name, description
```

### B.2 Execution Pipeline

**File**: `imas_codex/discovery/signals/parallel.py`

Replace or enhance `individualize_source_descriptions()`:

1. Claim enriched SignalSources with `member_count > 1`
2. For each source, fetch 3-5 sample member SignalNode descriptions from graph
3. Build prompt with group context + SECTION_METADATA + sample node descriptions
4. LLM generates Python `individualize()` function
5. **Sandbox execution**: Run the generated code in a restricted namespace
   (no imports except `re`, no file access, no network)
6. Apply the function to each member: `individualize(accessor, node_desc, index)`
7. Update FacilitySignal `name` and `description` per member
8. Set `enrichment_source = 'code_unwound'`

**Security**: The generated code runs in a restricted `exec()` with only `re`
available. Input validation on the returned strings (max length, no executable
content). Log the generated code for audit.

### B.3 Fetch Sample Member Node Descriptions

New query to fetch sample SignalNode descriptions for group members:

```cypher
MATCH (sg:SignalSource {id: $source_id})
MATCH (m:FacilitySignal)-[:MEMBER_OF]->(sg)
MATCH (m)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE sn.description IS NOT NULL
WITH m, sn
ORDER BY m.accessor
WITH collect({
    accessor: m.accessor,
    node_description: sn.description
})[..5] AS samples
RETURN samples
```

### B.4 Integration with Enrichment Pipeline

Two options for when to run:

**Option 1 (Recommended)**: Run code-generation unwinding as part of the
post-enrichment individualization phase (replacing current template approach).
This is the existing integration point at `parallel.py` line 4595.

**Option 2**: Integrate into the enrichment worker itself, so propagation
immediately applies individualized descriptions instead of copying verbatim.
This is cleaner but requires more refactoring.

---

## Plan C: Check Pipeline Fixes (High Priority)

**Estimated Benefit**: HIGH — unblocks validation for 4,785 signals currently
failing due to infrastructure bugs, not actual data problems.

### C.1 Fix numpy Missing in Remote venv

**Root Cause**: 2,427 signals from magnetics_config, device_xml, pf_coil_turns,
greens_table, jec2020_geometry, and sensor_calibration ALL fail with:
```
Tree open failed: No module named 'numpy'
```

The remote Python environment used by `run_python_script()` is missing numpy.
These are device_xml-type signals that use the DeviceXMLScanner's `check()`
method, which queries the graph directly (doesn't need remote execution).

**Fix**: The device_xml scanner's `check()` method runs locally:

```python
async def check(self, facility, ssh_host, signals, config, reference_shot=None):
    results = []
    with GraphClient() as gc:
        for signal in signals:
            dn_path = signal.data_source_node
            rows = gc.query("MATCH (dn:SignalNode {id: $path}) ...", path=dn_path)
            ...
```

The error suggests these signals are being routed to the WRONG scanner for
checking. Verify `_get_signal_scanner_type()` returns `"device_xml"` for
magnetics_config, pf_coil_turns, greens_table, etc. If these scanner types
are not registered, their signals fall through to a default check that uses
remote SSH and requires numpy.

**Action**: Audit `_get_signal_scanner_type()` and ensure ALL static data
source names map to scanners with local-only check methods:
- device_xml → DeviceXMLScanner (✅ has local check)
- magnetics_config → needs scanner or route to DeviceXMLScanner
- pf_coil_turns → needs scanner or route to DeviceXMLScanner
- greens_table → needs scanner or route to DeviceXMLScanner
- jec2020_geometry → needs scanner or route to DeviceXMLScanner
- sensor_calibration → needs scanner or route to DeviceXMLScanner

### C.2 Fix JPF Empty Path Check Failure

**Root Cause**: 2,358 JPF signals fail with 100% failure rate. The JPF scanner's
`check()` method (line 234) builds batch entries as:
```python
batch.append({"id": s.id, "path": s.node_path or s.data_source_path or ""})
```

JPF signals have `node_path` stored (after enrichment) but `data_source_path`
may have been null at discovery time. The issue is that `FacilitySignalModel`
construction at check time uses:
```python
node_path=s.get("node_path") or s.get("data_source_path")
```

If `node_path` on the FacilitySignal graph node is set but the check query
doesn't return it, the model gets `node_path=None`. The `claim_signals_for_check`
query (line 815) returns `s.node_path AS node_path` — this should work.

**Investigation needed**: Query checked JPF signals to verify:
1. Does `s.node_path` have a value on the FacilitySignal node?
2. Is the value being passed correctly through the check pipeline?
3. Does the remote `check_jpf.py` script receive the path?

**Likely Fix**: The `path` field in the batch entry may be wrongly resolved.
JPF signals use `accessor` as the signal path (e.g., `"DA/C1D-IPLA"`), not
`node_path` or `data_source_path`. Fix:

```python
# In jpf.py check():
batch.append({
    "id": s.id,
    "path": s.accessor or s.node_path or s.data_source_path or ""
})
```

### C.3 Classify PPF ier=210002 Errors

**Root Cause**: 3,230 PPF signals fail with `ppfdata ier=210002`. This means
"data does not exist for this shot" — the specific shot used for checking
(99896) doesn't have data for these DDAs/dtypes.

**Fix**: In `_classify_check_error()`, add:
```python
if "ier=210002" in err_lower or "ier=210002" in error:
    return "not_available_for_shot"
```

Then add support for multi-shot checking: try the reference shot first,
and if it fails with `not_available_for_shot`, try 2-3 recent shots from
the facility config. Only mark as truly failed if ALL shots fail.

### C.4 Add error_type Classification for numpy Import

In `_classify_check_error()`:
```python
if "no module named" in err_lower or "import error" in err_lower:
    return "missing_dependency"
```

---

## Plan D: Schema & Metadata Enhancements (Medium Priority)

**Estimated Benefit**: MEDIUM — enables systematic routing of static vs dynamic
data through different enrichment/check/mapping pipelines.

### D.1 Add `is_static` Property to FacilitySignal

**File**: `imas_codex/schemas/facility.yaml`

Add to FacilitySignal slots:
```yaml
is_static:
  description: >-
    Whether this signal represents static machine description data
    (geometry, calibration) rather than time-varying measurements.
    Static signals map to machine_description/pf_active/magnetics IDSs
    at the hardware level, not per-shot.
  range: boolean
```

Set during scanning based on `data_source_name`:
```python
STATIC_DATA_SOURCES = {
    "device_xml", "magnetics_config", "pf_coil_turns",
    "greens_table", "jec2020_geometry", "sensor_calibration"
}
```

### D.2 Add `imas_ids_hint` Property to FacilitySignal

Store the IMAS IDS target hint from SECTION_METADATA on the FacilitySignal
at scan time:

```python
# In device_xml scanner:
FacilitySignal(
    ...
    imas_ids_hint=meta.get("imas_ids", ""),  # e.g., "magnetics.bpol_probe"
)
```

This provides a deterministic IMAS mapping hint that doesn't depend on LLM
enrichment quality. The mapping pipeline can use this as a strong prior.

### D.3 Persist Enrichment Analytics

Store on each FacilitySignal:
- `context_quality` (from Plan A.3)
- `enrichment_model` — which LLM model was used
- `enrichment_prompt_hash` — hash of system prompt version (enables
  targeted re-enrichment when prompt changes)

---

## Plan E: Enrichment Prompt Improvements (Medium Priority)

**Estimated Benefit**: MEDIUM — improves description quality by providing
clearer guidance and preventing common hallucination patterns.

### E.1 Anti-Hallucination Guidance for Device XML

Add to the enrichment prompt's "Device Description" section:

```markdown
**CRITICAL for device_xml signals:**
- Each signal has a unique instance number (e.g., magprobes/**1**/r,
  magprobes/**10**/r, pfcoils/**3**/turns). The instance number is in the
  accessor after the section name.
- **Include the correct instance number in your description and name.**
  Do NOT describe multiple signals with the same probe/coil number.
- When `source_node_description` provides actual values (R=4.292m),
  include these values in your description.
- Use the `data_source_path` to identify section (magprobes/pfcoils/etc.)
  and instance number.
```

### E.2 Separate Enrichment for Static vs Dynamic Signals

When building the user prompt, add a static-data-specific preamble:

```markdown
**These signals are STATIC machine description data.**
They do not change per shot — they represent fixed hardware geometry.
Each instance number identifies a unique physical component.
Descriptions should specify:
1. The physical component type and its unique identifier
2. What physical quantity this entry represents
3. The actual value if provided in source_node_description
```

### E.3 Reduce Batch Size for Device XML Enrichment

The current batch size of 50 signals is fine for PPF/JPF signals that have
consistent structure. For device_xml signals, reduce to 20-30 to give the
LLM more attention per signal and reduce templating effects.

Implement by checking group key in the enrichment worker:
```python
# After grouping, if any group is device_xml, reduce effective batch
if any(k.startswith("device_xml") for k in context_groups):
    effective_batch = min(batch_size, 30)
```

---

## Plan F: Re-enrichment of Degraded Signals (High Priority)

**Estimated Benefit**: HIGH — this is the execution step that applies all
the fixes from Plans A-E to the existing degraded signal population.

### F.1 Identify Degraded Signals

Query to find signals needing re-enrichment:

```cypher
// Probe index hallucination (name doesn't match accessor instance number)
MATCH (s:FacilitySignal {facility_id: 'jet', data_source_name: 'device_xml'})
WHERE s.status IN ['enriched', 'checked']
  AND s.data_source_path STARTS WITH 'magprobes/'
WITH s,
     split(s.data_source_path, '/')[1] AS actual_instance,
     s.name AS current_name
WHERE NOT current_name CONTAINS actual_instance
RETURN s.id, actual_instance, current_name
ORDER BY toInteger(actual_instance)
```

### F.2 Reset and Re-enrich

After implementing Plans A-E:

```bash
# 1. Reset degraded device_xml signals to discovered
uv run imas-codex discover signals jet --reenrich --data-source device_xml

# 2. Re-run enrichment with improved context injection
uv run imas-codex discover signals jet --enrich-only --data-source device_xml
```

The `--reenrich` flag (from signal-enrichment-v3 Plan 1.5) resets status
to `discovered` and clears enrichment fields. The subsequent `--enrich-only`
run uses the improved enrichment pipeline with SignalNode descriptions,
section-specific grouping, and code-generation unwinding.

### F.3 Validation

After re-enrichment, run quality queries:

```cypher
// Verify no duplicate names within a section
MATCH (s:FacilitySignal {facility_id: 'jet', data_source_name: 'device_xml'})
WHERE s.status = 'enriched'
WITH s.data_source_path AS path,
     split(s.data_source_path, '/')[0] AS section,
     split(s.data_source_path, '/')[1] AS instance,
     s.name AS name, s.description AS desc
WITH section, name, collect(DISTINCT instance) AS instances
WHERE size(instances) > 1
RETURN section, name, size(instances) AS duplicate_count
ORDER BY duplicate_count DESC
```

---

## Dependencies and Risk Assessment

| Plan | Depends On | Risk | Mitigation |
|------|-----------|------|------------|
| A.1 (Node desc injection) | None | Low | Pure addition, no existing behavior changes |
| A.2 (Section sub-grouping) | None | Low | Additive change to grouping |
| A.3 (context_quality persist) | Schema change | Low | Additive schema property |
| B.1-B.4 (Code-gen unwinding) | A.1, A.2 | Medium | Sandbox execution, fallback to template |
| C.1 (numpy fix) | None | Low | Scanner routing fix |
| C.2 (JPF path fix) | None | Low | Path resolution fix |
| C.3 (PPF error classify) | None | Low | String matching addition |
| D.1 (is_static flag) | Schema change | Low | Additive, no breaking changes |
| D.2 (imas_ids_hint) | Schema change | Low | Deterministic from metadata |
| E.1-E.3 (Prompt fixes) | A.1, A.2 | Low | Prompt-only changes |
| F.1-F.3 (Re-enrichment) | A, B, E | Medium | Requires successful LLM calls |
