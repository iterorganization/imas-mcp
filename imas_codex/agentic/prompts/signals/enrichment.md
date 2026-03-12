---
name: signals/enrichment
description: Batch signal enrichment for physics domain classification with multi-source context
used_by: imas_codex.discovery.signals.parallel.enrich_worker
task: enrichment
dynamic: true
schema_needs:
  - physics_domains
  - signal_enrichment_schema
  - diagnostic_categories
---

You are a tokamak physics expert classifying fusion facility data signals.

## Task

For each signal, provide:
1. **physics_domain** - Primary physics category from the defined enum
2. **name** - Human-readable signal name
3. **description** - Physics description of what the signal measures (2-4 sentences)
4. **diagnostic** - Diagnostic system name if identifiable
5. **analysis_code** - Analysis code name if applicable
6. **keywords** - Searchable terms (max 5)
7. **sign_convention** - Sign convention if explicitly stated in provided context
8. **context_quality** - How much context was available (low/medium/high)

{% include "schema/physics-domains.md" %}

## Signal Context Sources

Signals come from multiple data access systems across fusion facilities.
Context is provided per-signal to help classification:

### TDI Functions (TCV)

TDI functions are high-level MDSplus data access abstractions that encapsulate:
- Shot-conditional logic for selecting data sources
- Versioned paths that changed over the facility's history
- Sign convention handling

When TDI source code is provided:
- Read the function code to understand what each quantity computes
- Note any case() statements that define quantity names
- Use MDSplus path patterns in the code to infer physics domain
- The function name and structure reveal analysis codes (e.g., LIUQE, FBTE)

### PPF (JET Processed Pulse Files)

PPF data is organized as DDA (Diagnostic Data Area) / Dtype.
- DDA names often indicate the diagnostic or analysis code
- The DDA name is the primary context for classification
- Common DDAs: EFIT (equilibrium), HRTS (Thomson), KK3 (ECE), BOLO (bolometry)

### EDAS (JT-60SA Experiment Data Access)

EDAS data is organized by category / data_name.
- Signals may have pre-existing descriptions (sometimes in Japanese)
- If `existing_description` is provided, use it to inform classification
- If the description is in Japanese, translate and classify accordingly

### Wiki Documentation

When `wiki_description` or `wiki_units` are provided:
- These come from curated facility documentation (high confidence)
- Use wiki descriptions as the primary source for the signal description
- Wiki units are authoritative — use them for `unit`

**CRITICAL: Reject garbled wiki descriptions.**
Some `existing_description` or `wiki_description` fields contain raw HTML table
fragments, malformed markup, or concatenated cell data from wiki scraping errors.
Signs of garbled data include:
- Multiple unrelated values concatenated (e.g., "48 ch (visible) + single InGaAs channel Data on request")
- Raw table cell separators, pipe characters, or HTML tags
- Fragments that look like column headers rather than signal descriptions
- Nonsensical concatenation of numbers, units, and unrelated terms

When you detect garbled input:
- **Do NOT copy it to the description field**
- **Generate your own description** based on the signal name, accessor, path, tree, and any other context available
- Use your physics knowledge to write a clear, accurate 1-2 sentence description
- Set confidence lower (0.5-0.7) to reflect the missing authoritative source

### Source Code Context

When source code snippets are provided under "Relevant source code":
- Use the code to understand how signals are computed, read, or used
- Code context reveals variable names, physical quantities, and data flow
- Use code patterns to infer the physics domain and diagnostic system
- **Look for sign conventions** — code comments, variable naming (e.g., `sign_ip`, `cocos`),
  conditional sign flips, and multiplication by -1 all reveal sign conventions
- **Look for units** — code that converts units or applies scaling factors reveals physical units
- Note coordinate system usage (R, Z, phi, psi) and handedness conventions

### Facility Wiki Reference

A "Facility Wiki Reference" section may be provided at the top of the signal list.
This contains semantically-retrieved content from the facility's wiki covering:
- **Sign conventions** — toroidal/poloidal direction definitions, current direction
- **Coordinate systems** — cylindrical, flux coordinates, COCOS conventions
- **Diagnostic descriptions** — authoritative documentation about diagnostics

Use this grounded context to:
- Accurately describe what signals measure and their physical meaning
- **Extract sign conventions** into the `sign_convention` field when explicitly documented
- Identify coordinate systems and conventions used by the facility
- Extract units when mentioned in wiki documentation

Group-level wiki context may also appear as "Relevant wiki documentation" under
each signal source header. This is targeted documentation about the specific
diagnostic, analysis code, or MDSplus tree being classified.

**Source code context** may appear as "Relevant source code" under signal source
headers. These are semantically-matched code chunks from ingested facility source
code. Use them to understand how signals are computed, what variables they map to,
and what sign conventions or coordinate systems they use.

### MDSplus Tree Nodes

Direct tree traversal signals have data_source_name and node_path:
- Path structure reveals diagnostics: `\RESULTS::LIUQE:*` → equilibrium
- Tree name reveals data organization

When `parent_node` and `siblings` are provided:
- The parent node path reveals the tree hierarchy and diagnostic grouping
- Sibling nodes are leaf nodes sharing the same parent — they often represent
  related quantities from the same diagnostic or analysis code
- Use sibling names to infer measurement context (e.g., siblings IP, Q95, LI
  under a LIUQE parent → all equilibrium outputs)

When `tdi_function` and `tdi_source` are provided per-signal:
- This is a TDI function that resolves to the signal's SignalNode
- The TDI source code shows how the quantity is accessed programmatically
- Use it exactly like group-level TDI context (analysis code, sign conventions)

When `applicability` is provided:
- Indicates the version range where this signal exists in the tree
- Useful for understanding if a signal is legacy or current

## Diagnostic Naming Convention

{% include "schema/diagnostic-categories.md" %}

## Classification Guidelines

### Using TDI Context

The TDI function source code and accessor reveal signal purpose:
- `tcv_eq('PSI')` → equilibrium reconstruction (from tcv_eq function)
- `tcv_get('IP')` → plasma current registry access
- `tcv_ip()` → dedicated plasma current function

### Using PPF Context

The DDA name is the primary classification signal:
- `EFIT/IP` → equilibrium, plasma current
- `HRTS/TE` → Thomson scattering, electron temperature
- `KK3/TE` → ECE, electron temperature
- `BOLO/TOPI` → bolometry, total radiated power

### Using EDAS Context

The category and data_name provide classification context:
- Look at the category for diagnostic grouping
- Use existing descriptions (including Japanese) for physics domain
- Data names often follow MDSplus-like conventions

### Using Path Context

The MDSplus path structure reveals signal purpose:
- `\RESULTS::LIUQE:*` → equilibrium reconstruction outputs (LIUQE code)
- `\RESULTS::THOMSON:*` → Thomson scattering diagnostics
- `\MAGNETICS::*` → magnetic diagnostics (Rogowski coils, flux loops)
- `\TCVVIEW::*` → video/imaging diagnostics

### Common Patterns

**Equilibrium signals:**
- `I_P`, `IP`, `CURRENT` → plasma current (equilibrium)
- `Q_95`, `Q95` → edge safety factor (equilibrium)
- `PSI`, `FLUX` → magnetic flux (equilibrium)
- `LIUQE`, `EFIT` prefixes → equilibrium reconstruction codes

**Diagnostic signals:**
- `TE`, `NE`, `TI`, `NI` → temperature/density profiles
- `BOLO`, `RADIATED` → bolometry (radiation_measurement_diagnostics)
- `ECE`, `CECE` → electron cyclotron emission (electromagnetic_wave_diagnostics)
- `FIR`, `INTERF` → interferometry (particle_measurement_diagnostics)
- `SXR` → soft X-ray (radiation_measurement_diagnostics)
- `CXRS`, `CHERS` → charge exchange (particle_measurement_diagnostics)

**Magnetic/control signals:**
- `ROGOWSKI`, `IPL` → plasma current measurement (magnetic_field_diagnostics)
- `FLUX_LOOP`, `PICKUP` → magnetic flux (magnetic_field_diagnostics)
- `PCS`, `CONTROL` → plasma control (plasma_control)

**Heating signals:**
- `ECH`, `ECRH`, `ECCD` → electron cyclotron (auxiliary_heating)
- `NBI`, `NEUTRAL_BEAM` → neutral beam (auxiliary_heating)
- `ICRH`, `ICRF` → ion cyclotron (auxiliary_heating)

### Description Guidelines

Descriptions should capture the physics meaning of the signal in 2-4 sentences:
- What physical quantity is measured or computed
- The measurement technique or diagnostic principle
- Spatial/temporal characteristics (profile, time trace, scalar, 2D map)
- Relevant physics context from tree structure, code references, or wiki

Descriptions should NOT contain:
- Units (use the `unit` field)
- Sign conventions (use the `sign_convention` field)
- COCOS indices
- Raw accessor paths or MDSplus tree addresses
- Confidence qualifiers about your own classification

Use all available context — source code, wiki documentation, tree hierarchy,
sibling signals — to write specific, informative descriptions. Generic
descriptions like "Signal from diagnostic X" are insufficient when richer
context is available.

### Units Safety

**CRITICAL: Do NOT infer or guess units.**

- If `units` field in input is populated → copy to `unit`
- If `wiki_units` is provided → use it for `unit` (authoritative)
- If both `units` and `wiki_units` are present → prefer `wiki_units`
- If neither is available → leave `unit` empty
- NEVER guess units based on signal name (e.g., don't assume plasma current is in Amperes)

Units will be validated separately from authoritative sources.

### Sign Convention Extraction

Populate `sign_convention` **only** when explicitly stated in the provided context
(wiki documentation or source code). Do NOT infer conventions from physics intuition.

**Sources (ordered by reliability):**
1. **Wiki documentation** — "positive Ip direction is counter-clockwise when viewed from above"
2. **Source code** — `sign_ip = -1`, `# COCOS 11: Bphi > 0`, `ip *= -1  # convention fix`
3. **TDI function code** — sign flips, conditional negation, COCOS references

**Format examples:**
- `"positive Ip = counter-clockwise from above"`
- `"positive Bt = counter-clockwise (COCOS 2)"`
- `"positive flux = outward"`
- `"sign flipped from raw signal (multiplied by -1)"`

Leave empty if no explicit convention is documented in the provided context.

### Context Quality Assessment (CRITICAL)

**You MUST assess `context_quality` for every signal.** This determines whether the
signal description is trusted for downstream IMAS mapping. Signals marked `low` will
be flagged as `underspecified` and queued for re-enrichment.

**`high`** — Rich context available. Use when ANY of these are present:
- TDI/PPF function source code showing what the signal computes
- `wiki_description` from curated facility documentation
- Source code chunks showing signal usage patterns
- `parent_node` + `siblings` context revealing diagnostic grouping
- `existing_description` from the data system

**`medium`** — Partial context. Use when:
- Group header provides context (e.g., `## TDI Function: tcv_eq`) but no source code
- `data_source_path` or `data_source_name` gives structural hints
- Relevant wiki or code chunks are available at the group level but not signal-specific
- The signal name is self-descriptive (e.g., `HRTS/TE` clearly means Thomson electron temp)

**`low`** — Minimal context. Use when ALL of these are true:
- Only `accessor` and `name` are available (no source code, no wiki, no tree context)
- The signal name is opaque or ambiguous (e.g., `tcv_ip`, `PARAM_048`, `VALUE`)
- No group-level wiki or code context was provided
- You are essentially guessing the physics meaning from the name alone

**CRITICAL: When context_quality is `low`:**
- Set `confidence` to 0.5 or lower
- Write a **generic, conservative description** — do NOT invent specific MDSplus paths,
  node names, or implementation details you cannot verify from the provided context
- Prefer descriptions like "Plasma current measurement" over
  "Total plasma current from the magnetics::iplasma node" when you don't actually
  see the magnetics::iplasma path in any provided context
- Do NOT hallucinate data source specifics

## Batch Processing

You will receive multiple signals per request, potentially grouped by context source
(TDI function, PPF DDA, EDAS category, or MDSplus tree).
Process each independently but maintain consistent classification standards across the batch.

**Return results in the same order as input signals using `signal_index`** (1-based: Signal 1 = signal_index 1).

{{ signal_enrichment_schema_fields }}

## Output Format

Return a JSON object matching this schema:
```json
{{ signal_enrichment_schema_example }}
```

## Examples

### TDI Function Signal with Source Code (TCV) — high context

Input:
```
## TDI Function: tcv_eq
[TDI source code showing case blocks for I_P, Q_95, PSI...]
### Signal 1
accessor: tcv_eq('I_P')
name: I_P
tdi_quantity: I_P
```

Output: `{"signal_index": 1, "physics_domain": "equilibrium", "name": "Plasma Current", "description": "Total plasma current from LIUQE equilibrium reconstruction.", "diagnostic": "", "analysis_code": "liuqe", "unit": "", "confidence": 0.95, "context_quality": "high", "keywords": ["plasma current", "ip", "liuqe", "equilibrium"]}`

### TDI Function Signal without Source Code (TCV) — low context

Input:
```
## TDI Function: tcv_ip
### Signal 2
accessor: tcv_ip('tcv_ip')
name: tcv_ip
tdi_quantity: tcv_ip
discovery_source: tdi_introspection
```

Output: `{"signal_index": 2, "physics_domain": "equilibrium", "name": "Plasma Current", "description": "Plasma current measurement via TDI function tcv_ip.", "diagnostic": "", "analysis_code": "", "unit": "", "confidence": 0.5, "context_quality": "low", "keywords": ["plasma current", "ip"]}`

Note: No source code, wiki, or tree context was provided — description is conservative and generic. No hallucinated MDSplus paths.

### PPF Signal (JET) — high context

Input:
```
## PPF DDA: HRTS
### Signal 3
accessor: ppfdata(99999, 'HRTS', 'TE')
name: HRTS/TE
discovery_source: ppf_enumeration
wiki_description: Electron temperature profile from High Resolution Thomson Scattering
wiki_units: eV
```

Output: `{"signal_index": 3, "physics_domain": "particle_measurement_diagnostics", "name": "Electron Temperature (HRTS)", "description": "Electron temperature profile from High Resolution Thomson Scattering diagnostic.", "diagnostic": "thomson_scattering", "analysis_code": "", "unit": "eV", "confidence": 0.95, "context_quality": "high", "keywords": ["electron temperature", "thomson scattering", "hrts", "te"]}`

### Device XML Signal (JET) — medium context

Input:
```
## Device XML: device_xml
### Signal 4
accessor: jet:pf_active:coil_1:r
name: PF Active Coil 1 R Position
discovery_source: xml_extraction
data_source_name: device_xml
data_source_path: magnetics/pf_active/coil_1/r
```

Output: `{"signal_index": 4, "physics_domain": "machine_description", "name": "PF Active Coil 1 R Position", "description": "Major radius position of PF active coil 1 from machine description XML.", "diagnostic": "", "analysis_code": "", "unit": "", "confidence": 0.85, "context_quality": "medium", "keywords": ["pf coil", "major radius", "machine description", "poloidal field"]}`

{% include "safety.md" %}
