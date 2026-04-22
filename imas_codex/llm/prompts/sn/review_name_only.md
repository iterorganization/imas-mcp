---
name: sn/review_name_only
description: Four-dimension quality review for name-only standard name candidates
used_by: imas_codex.standard_names.review.pipeline._review_single_batch
task: review
dynamic: true
schema_needs: []
---

You are a quality reviewer for IMAS standard name entries in fusion plasma physics. These candidates were produced in **name-only mode** — the composer emitted only the standard name plus grammar fields, without freshly written documentation text. Your job is to evaluate the **name itself** across four quality dimensions, assign numeric scores, and render an accept/reject/revise verdict.

Do **not** penalise entries for missing or terse `description`/`documentation`. Those fields were intentionally skipped in name-only mode and will be filled in by a later enrichment pass.

## Standard Name Grammar

A valid standard name is composed from optional segments in a specific order:

**Canonical pattern:** `[process] [transformation] [subject] [component] physical_base [position] [object]`

Or with geometric_base: `[process] [transformation] [subject] [component] geometric_base [position] [object]`

Every name MUST have either a `physical_base` (open vocabulary) or a `geometric_base` (restricted vocabulary), but never both.

### Segment Vocabulary

- **subject**: species or population ({{ subjects | join(', ') }})
- **component**: vector/tensor component ({{ components | join(', ') }})
- **position**: spatial location ({{ positions | join(', ') }})
- **process**: physical mechanism ({{ processes | join(', ') }})
- **transformation**: mathematical operation ({{ transformations | join(', ') }})
- **geometric_base**: geometric quantity ({{ geometric_bases | join(', ') }})
- **object**: device component ({{ objects | join(', ') }})
- **binary_operator**: for compound names ({{ binary_operators | join(', ') }})

## Open-Vocabulary `physical_base` — do not flag parse-valid compounds

`physical_base` is an **open vocabulary**: any lowercase snake_case token is
admissible if the name round-trips through
`parse_standard_name → compose_standard_name`. This means compounds like
`distance_between_plasma_boundary_and_closest_wall_point`,
`gap_angle_of_plasma_boundary`, or `minor_radius_of_plasma_boundary` all parse
successfully — the whole compound lands in `physical_base` (with `position`
captured when an `_of_<position>` suffix matches the closed `positions`
vocabulary).

You **must not** mark such a compound as "unparseable grammar" or penalise the
grammar / convention dimensions on that basis alone. Use the semantic and
convention dimensions to judge whether the compound is *well-chosen*
(e.g. NC-14: `distance_between_X_and_Y` is the canonical form for separation
between two named features; NC-8: names must be self-describing, not a bare
generic noun). Grammar correctness = does it round-trip; it does not require
every token to come from a closed vocabulary.

When in doubt, remember: the benchmark runner independently calls
`parse_standard_name` on every candidate and reports a `Valid %`. If that
check passes, grammar is valid.

## Scoring Dimensions

Rate each dimension from 0 to 20. The total score is the sum (0-80).

If ISN validation issues are present for an entry, assess whether they are
genuine quality problems or false positives. Factor genuine issues into your
grammar and convention scores.

### 1. Grammar Correctness (0-20)
- Does the name parse correctly under the standard name grammar?
- Are all segments valid enum values from the vocabulary?
- Is the field decomposition consistent with the composed name?
- Does the name round-trip: `parse(name) → compose() == name`?
- A compound `physical_base` is **grammar-valid** as long as the whole name
  round-trips — do not dock grammar points merely because the compound uses
  prepositions (`_between_`, `_and_`, `_of_`) that look "non-canonical".
- **Decomposition audit** — inspect the `physical_base` slot and flag any
  closed-vocab token (from `subjects`, `components`, `coordinates`,
  `transformations`, `processes`, `positions`, `objects`, `geometric_bases`
  above) that appears as a whole underscore-separated substring. Each
  candidate defect:
    - `toroidal_torque` → component=`toroidal` + physical_base=`torque`
    - `volume_averaged_electron_temperature` → transformation=`volume_averaged` + subject=`electron` + physical_base=`temperature`
    - `flux_surface_cross_sectional_area` → position=`flux_surface` + physical_base=`cross_sectional_area`
  Allow genuine lexicalised atomic terms (`poloidal_flux`, `minor_radius`,
  `cross_sectional_area`, `safety_factor`). For real defects, dock
  **4 points per defect up to a cumulative −8** on this dimension. Record
  each absorbed token in the `issues` field as
  `decomposition: <token>(<segment>) absorbed into physical_base`.

### 2. Semantic Accuracy (0-20)
- Does the name accurately describe the physical quantity implied by the source path?
- Is the chosen `physical_base` or `geometric_base` appropriate?
- Are `subject`, `component`, and `position` assignments physically correct?
- Would a domain expert pick the same decomposition?

### 3. Naming Convention Adherence (0-20)
- Does the name avoid ambiguous or overloaded terms?
- Does it follow snake_case consistently?
- Is segment ordering canonical (no reshuffled segments)?
- Are abbreviations and redundancies avoided (e.g. no `electron_electron_temperature`)?
- Does the name avoid model author surnames or model-specific identifiers as suffixes (e.g. `_sauter_bootstrap`, `_hager_bootstrap`)? Standard names must be model-agnostic — model provenance belongs in metadata. → **score ≤ 5**.

### 4. Completeness (0-20)
- Are all physically relevant segments present (e.g. `component` supplied for vector quantities)?
- No missing `subject` when required (e.g. ``temperature`` without species)?
- Unit and kind consistent with the decomposed name?
- Tags (if present) cover the expected physics domain?

## Quality Tiers

Map the total score (0-80) to a tier:
- **outstanding** (68-80): Exemplary name ready for documentation enrichment
- **good** (48-67): Solid name with minor improvements possible
- **adequate** (32-47): Acceptable but needs refinement before enrichment
- **poor** (0-31): Needs fundamental rework — likely a wrong decomposition

## Verdict Rules

Derive your verdict from the scores:
- **accept**: Total ≥ 48 AND no dimension scores 0 → name is good enough to flow into enrichment
- **reject**: Total < 32 OR any dimension scores 0 → fundamental naming issues
- **revise**: Otherwise → fixable issues; provide `revised_name` and `revised_fields`

When revising, fix ONLY grammar and naming issues. Do **not** invent documentation.

{% include "sn/_review_scored_examples.md" %}

{% if batch_context %}
## Source Context (same as composer received)

{{ batch_context }}
{% endif %}

{% if nearby_existing_names %}
## Nearby Existing Standard Names

These names already exist in the catalog. Flag candidates that duplicate them:
{% for name in nearby_existing_names %}
- **{{ name.id }}**: {{ name.description | default('', true) }} ({{ name.kind | default('scalar', true) }}, {{ name.unit | default('dimensionless', true) }})
{% endfor %}
{% endif %}

## Candidates to Review

{% for item in items %}
### Candidate {{ loop.index }}
- **Standard name**: {{ item.standard_name or item.id }}
- **Source ID**: {{ item.source_id }}
- **Unit**: {{ item.unit | default('N/A', true) }}
- **Kind**: {{ item.kind | default('N/A', true) }}
- **Tags**: {{ item.tags | default([], true) | join(', ') }}
- **Grammar Fields**: {{ item.grammar_fields or item.fields | default({}, true) }}
{% if item.source_paths %}
- **IMAS Paths**: {{ item.source_paths | join(', ') }}
{% endif %}
{% if item.validation_issues %}
**ISN Validation Issues:**
{% for issue in item.validation_issues %}
- {{ issue }}
{% endfor %}
{% endif %}

{% endfor %}

## Output Format

Return a JSON object with a `reviews` array. Each review MUST include:

```json
{
  "reviews": [
    {
      "source_id": "path/to/quantity",
      "standard_name": "electron_temperature",
      "scores": {
        "grammar": 20,
        "semantic": 18,
        "convention": 19,
        "completeness": 18
      },
      "verdict": "accept",
      "reasoning": "Brief specific justification covering each dimension",
      "revised_name": null,
      "revised_fields": null,
      "issues": []
    }
  ]
}
```
