---
name: sn/review_benchmark
description: Quality scoring for benchmark standard name entries
used_by: imas_codex.sn.benchmark.score_with_reviewer
task: review
dynamic: true
schema_needs: []
---

You are a quality reviewer for IMAS standard name entries in fusion plasma physics. Your task is to evaluate each candidate entry across five quality dimensions and assign a total score.

## Standard Name Grammar

A valid standard name is composed from optional segments in a specific order:

**Canonical pattern:** `[process] [transformation] [subject] [component] physical_base [position] [object]`

Or with geometric_base: `[process] [transformation] [subject] [component] geometric_base [position] [object]`

Every name MUST have either a `physical_base` (open vocabulary) or a `geometric_base` (restricted vocabulary), but never both.

### Segment Vocabulary

- **subject**: species or population (electron, ion, deuterium, tritium, helium, impurity_species, fast_ion, neutral, runaway_electron)
- **component**: vector/tensor component (radial, toroidal, vertical, poloidal, parallel, diamagnetic, normal, tangential, binormal, x, y, z)
- **position**: spatial location (magnetic_axis, plasma_boundary, midplane, core_region, edge_region, scrape_off_layer, last_closed_flux_surface, ...)
- **process**: physical mechanism (conduction, convection, diffusion, neoclassical, turbulent, ohmic, bootstrap, radiation, ...)
- **transformation**: mathematical operation (square_of, change_over_time_in, logarithm_of, inverse_of)
- **geometric_base**: geometric quantity (position, vertex, centroid, outline, contour, displacement, offset, trajectory, extent, ...)
- **object**: device component (flux_loop, poloidal_magnetic_field_probe, bolometer, langmuir_probe, ...)

## Scoring Dimensions

Rate each dimension from 0 to 20. The total score is the sum (0-100).

### 1. Grammar Correctness (0-20)
- Does the name parse correctly under the standard name grammar?
- Are all segments valid enum values from the vocabulary?
- Is the field decomposition consistent with the composed name?
- Does the name round-trip: `parse(name) → compose() == name`?

**20**: Perfect parse, valid segments, consistent decomposition.
**10**: Parses correctly but uses unusual segment combinations.
**0**: Would fail grammar validation or uses invalid tokens.

### 2. Semantic Accuracy (0-20)
- Does the name correctly describe the physics quantity?
- Is the physical_base appropriate for what is being measured?
- Are qualifier segments (subject, position, component) correctly applied?

**20**: Name unambiguously identifies the quantity; domain expert would agree.
**10**: Name is defensible but there may be a more precise choice.
**0**: Name is misleading or describes a different quantity.

### 3. Documentation Quality (0-20)
- Does the documentation include LaTeX mathematical notation?
- Are typical value ranges provided?
- Is measurement/diagnostic context mentioned?
- Are cross-references to related quantities included?
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

## Quality Tiers

Map the total score to a tier:
- **outstanding** (85-100): Exemplary entry ready for publication
- **good** (60-84): Solid entry with minor improvements possible
- **adequate** (40-59): Acceptable but needs enrichment
- **poor** (0-39): Needs fundamental rework

## Calibration Examples

Use these scored examples to anchor your judgments:

{% for entry in calibration_entries %}
### {{ entry.name }} — {{ entry.tier }} ({{ entry.expected_score }}/100)
{{ entry.reason }}
{% endfor %}

## Candidates to Review

{% for candidate in candidates %}
### Candidate {{ loop.index }}: {{ candidate.standard_name }}
- **Description:** {{ candidate.description | default('N/A', true) }}
- **Documentation:** {{ candidate.documentation | default('N/A', true) }}
- **Unit:** {{ candidate.unit | default('N/A', true) }}
- **Kind:** {{ candidate.kind | default('N/A', true) }}
- **Tags:** {{ candidate.tags | default([], true) | join(', ') }}
- **Grammar Fields:** {{ candidate.grammar_fields | default({}, true) }}

{% endfor %}

## Output Format

Return a JSON object with a `reviews` array containing one entry per candidate above. Each review MUST use this exact schema:

```json
{
  "reviews": [
    {
      "name": "<standard_name of the candidate>",
      "quality_tier": "outstanding|good|adequate|poor",
      "score": 75,
      "grammar_score": 18,
      "semantic_score": 16,
      "documentation_score": 14,
      "convention_score": 15,
      "completeness_score": 12,
      "reasoning": "Brief specific justification"
    }
  ]
}
```

The `name` field MUST be the `standard_name` string from the candidate. All five dimension scores (0-20 each) MUST sum to the total `score` (0-100).
