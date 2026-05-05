---
name: sn/review_names_user
description: Dynamic user-message portion of name-axis review (companion to review_names_system)
used_by: imas_codex.standard_names.workers.process_review_name_batch
task: review
dynamic: true
schema_needs: []
---

Apply the rubric (provided in the system prompt) to the candidate(s) below.

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

{% if batch_context %}
## Source Context (same as composer received)

{{ batch_context }}
{% endif %}

## Sibling-Comparison Context

Use these accepted, in-catalog names as your **third-party reference set**. They are NOT to be reviewed. Score the candidate(s) below against the **patterns** these siblings establish (decomposition style, segment usage, naming consistency). Cite specific sibling `id`s when you dock points.

{% if vector_neighbours %}
### Nearest by description (vector similarity)
{% for n in vector_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}{% if n.score is defined %} [sim={{ '%.2f' | format(n.score) }}]{% endif %}
{% endfor %}
{% endif %}

{% if same_base_neighbours %}
### Same `physical_base` (sibling decomposition pattern)
{% for n in same_base_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if same_path_neighbours %}
### Same DD IDS source family
{% for n in same_path_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if not vector_neighbours and not same_base_neighbours and not same_path_neighbours %}
*No accepted siblings found — score on grammar + DD provenance alone.*
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
      "reasoning": "Brief specific justification covering each dimension",
      "revised_name": null,
      "revised_fields": null,
      "issues": []
    }
  ]
}
```
