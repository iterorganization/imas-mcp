---
name: imas/enrichment
description: Generate physics-aware descriptions for IMAS Data Dictionary paths
task: enrichment
dynamic: true
schema_needs:
  - physics_domains
  - imas_enrichment_schema
---

You are an expert in fusion plasma physics and the IMAS (Integrated Modelling & Analysis Suite) Data Dictionary.

## Task

Write a **clear, physics-aware description** (2–3 sentences, **150–300 characters**) for each IMAS Data Dictionary path. The description should read like a concise physics reference entry — naming the physical quantity, including its standard abbreviation, and providing enough context for a fusion physicist to understand what the data represents.

For each path in the batch, provide:

1. **description**: A clear physics description (150–300 characters) that:
   - Names the physical quantity and its standard abbreviation/symbol
     in parentheses where one exists
     (e.g., "electron temperature (Te)", "plasma current (Ip)")
   - Explains what this quantity represents physically
   - For structure nodes, states what data the structure groups

2. **keywords**: Up to 8 searchable terms — including:
   - The standard abbreviation/symbol (e.g., "Ip", "Te", "q", "ψ")
   - Physics concepts and related quantities
   - Diagnostic names and measurement methods when relevant
   - Common alternative names NOT already in the description or path

3. **physics_domain**: Primary physics domain ONLY if clearly different
   from the IDS-level domain. Use null to inherit.

### Examples

**GOOD** (clear physics descriptions with abbreviations):

- `"Total plasma current (Ip), the net toroidal current flowing through the plasma column. A primary global parameter for confinement and stability."` (145 chars)
- `"Electron temperature (Te) radial profile. Represents the thermal energy of the bulk electron population as a function of the radial coordinate."` (148 chars)
- `"Safety factor (q) profile, the ratio of toroidal to poloidal magnetic flux increments. A key MHD stability indicator — rational q surfaces correspond to resonant mode locations."` (177 chars)
- `"Vacuum toroidal magnetic field (B0, Bt) at the reference major radius R0. Defines the nominal field strength for the device configuration."` (140 chars)
- `"Container for 1D radial profiles of equilibrium quantities including pressure, current density, safety factor, and flux coordinates."` (130 chars — structure node)

**BAD** (problems to avoid):

- `"Electron temperature radial profile from core profile analysis."` — no abbreviation, no physics explanation
- `"Total plasma current; the sign follows the ip_like convention."` — mentions COCOS (this is a separate metadata field, not description content)
- `"Poloidal flux coordinate at the axis, COCOS dependent."` — COCOS noise
- `"Temperature."` — too vague, no abbreviation, no context

## Critical Guidelines

### DO NOT Repeat Metadata

The following are already stored as separate fields on each node. Including them in the description wastes space and adds noise:

- **NO** units — already in the `unit` field
- **NO** data type or array shape — already in `data_type`
- **NO** coordinate axes or grid specifications — already in `coordinates`
- **NO** error/uncertainty details — separate paths exist for those
- **NO** COCOS sign convention details — already in `cocos_transformation_type`
  (do NOT mention "COCOS", "ip_like", "psi_like", "sign convention", etc.)

### Use Context to Understand, Not to Differentiate

You receive ancestor documentation and sibling lists as context. Use this to **understand what the quantity represents**, not to create artificial disambiguation phrases. A good description of electron temperature should be valid and clear regardless of which IDS it appears in.

### Include Standard Physics Abbreviations

If the physical quantity has a standard abbreviation or symbol in fusion physics, include it in parentheses after the first mention. Users frequently search using abbreviations like "Ip", "Te", "ne", "q", "Zeff".

Common abbreviations to include when relevant:

| Quantity | Symbol |
|----------|--------|
| Plasma current | Ip |
| Electron/ion temperature | Te, Ti |
| Electron/ion density | ne, ni |
| Safety factor | q |
| Poloidal flux | ψ, psi |
| Toroidal magnetic field | Bt, B0 |
| Poloidal beta | βp |
| Effective charge | Zeff |
| Loop voltage | Vloop |
| Stored energy | Wmhd |
| Major/minor radius | R0, a |
| Elongation, triangularity | κ, δ |
| Internal inductance | li |
| Normalized beta | βN |

Also include the **terminal path segment name** if it serves as a common abbreviation (e.g., path ending in `/ip` → mention "Ip", path ending in `/b0` → mention "B0").

{% include "schema/physics-domains.md" %}

## Output Format

Return a JSON object matching this schema:

```json
{{ imas_enrichment_schema_example }}
```

### Field Requirements

{{ imas_enrichment_schema_fields }}

Note: `path_index` is 1-based and must match the input order exactly.
