## Scoring Dimensions

Rate each dimension from 0 to 20. The total is normalized to a 0-1 score (sum / 120).

### 1. Grammar Correctness (0-20)
- Does the name parse correctly under the standard name grammar?
- Are all segments valid enum values from the vocabulary?
- Is the field decomposition consistent with the composed name?
- Does the name round-trip: `parse(name) → compose() == name`?

**20**: Perfect parse, valid segments, consistent decomposition.
**10**: Parses correctly but uses unusual segment combinations.
**0**: Would fail grammar validation or uses invalid tokens.

### 2. Semantic Accuracy (0-20)
- Does the name correctly describe the physics quantity from the source?
- Is the physical_base appropriate for what is being measured?
- Are qualifier segments (subject, position, component) correctly applied?
{% if batch_context %}
- Does the name match the DD path description and cluster context provided?
{% endif %}

**20**: Name unambiguously identifies the quantity; domain expert would agree.
**10**: Name is defensible but there may be a more precise choice.
**0**: Name is misleading or describes a different quantity.

### 3. Documentation Quality (0-20)
- Does the documentation include LaTeX mathematical notation?
- Are typical value ranges provided?
- Is measurement/diagnostic context mentioned?
- Are cross-references to related quantities included (using inline links)?
- Is the documentation substantive (not just rephrasing the name)?

**20**: Rich docs with LaTeX, value ranges, measurement context, cross-refs.
**10**: Adequate docs — correct but thin, missing some elements.
**0**: Empty or circular documentation (just restates the name).

### 4. Naming Conventions (0-20)
- Does the name follow established patterns for similar quantities?
- Is the name concise but unambiguous?
- Does it avoid overly generic terms (data, signal, value)?
- Is it specific enough to be useful as a standard identifier?

**20**: Follows best practices, concise, unambiguous, specific.
**10**: Acceptable but could be improved — slightly verbose or generic.
**0**: Vague, generic, or violates naming conventions.

### 5. Entry Completeness (0-20)
- Is the unit correct for this quantity (or null if dimensionless)?
- Is the kind (scalar/vector/metadata) appropriate?
- Are relevant tags assigned from the controlled vocabulary?
- Are grammar fields properly populated?

**20**: All metadata fields correct and complete.
**10**: Most fields present but some missing or questionable.
**0**: Missing critical fields (wrong unit, no tags, wrong kind).

### 6. Prompt Compliance (0-20)
- Did the composer follow the unit policy? (Unit must come from DD, not be invented)
- Are anti-patterns avoided? (No "_profile" suffix, no generic "signal_value", no IDS name in the name)
- Is concept identity preserved? (Same concept across IDSs → same standard name)
- If the source is a coordinate or index, was it correctly skipped or handled?
- Are vocab_gaps flagged when a needed grammar token doesn't exist?

**20**: Perfect compliance with all composition instructions.
**10**: Minor deviations — one anti-pattern or missing vocab_gap flag.
**0**: Systematic disregard for instructions.

## Quality Tiers

Map the normalized score (0-1) to a tier:
- **outstanding** (≥0.85): Exemplary entry ready for publication
- **good** (≥0.60): Solid entry with minor improvements possible
- **inadequate** (≥0.40): Acceptable but needs enrichment
- **poor** (<0.40): Needs fundamental rework

## Verdict Rules

Derive your verdict from the scores:
- **accept**: Normalized score ≥ 0.60 AND no dimension scores 0
- **reject**: Normalized score < 0.40 OR any dimension scores 0
- **revise**: Otherwise → fixable issues; provide `revised_name` and `revised_fields`

When revising, fix ONLY grammar and naming issues. Do not rewrite documentation.
