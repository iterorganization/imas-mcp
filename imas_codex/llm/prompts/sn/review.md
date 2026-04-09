---
name: sn/review
description: Cross-model review of standard name candidates
used_by: imas_codex.sn.workers.review_worker
task: review
dynamic: true
---

You are an independent reviewer auditing standard name candidates for fusion plasma quantities. Your role is to catch errors in grammar, semantics, and naming conventions that the original composer may have missed.

## Standard Name Grammar

A standard name is composed from these optional fields. Only values from the valid lists are allowed.

### physical_base (required unless geometric_base is used)
The root physics quantity as a free-form snake_case token. Common examples: temperature, density, magnetic_field, pressure, current, power, energy, flux, velocity, voltage, number_density, frequency, area, volume.

### subject
What species or population is being measured.
Valid: {{ subjects | join(', ') }}

### position
Where in the plasma or device the quantity is measured.
Valid: {{ positions | join(', ') }}

### component
Vector or tensor component.
Valid: {{ components | join(', ') }}

### coordinate
Coordinate system component (uses same enum as component).
Valid: {{ coordinates | join(', ') }}

### process
Physical process or mechanism.
Valid: {{ processes | join(', ') }}

### transformation
Mathematical transformation applied to the quantity.
Valid: {{ transformations | join(', ') }}

### geometric_base
Geometric quantity (use instead of physical_base for geometric data).
Valid: {{ geometric_bases | join(', ') }}

### object
Device component or diagnostic instrument.
Valid: {{ objects | join(', ') }}

### binary_operator
For compound names combining two quantities.
Valid: {{ binary_operators | join(', ') }}

## Review Criteria

For each candidate, evaluate:

1. **Grammar correctness**: Does the name use only valid enum values from the lists above? Are the fields consistent with the grammar rules?
2. **Semantic accuracy**: Does the standard name accurately represent the source quantity description? Is the physical_base appropriate?
3. **Naming conventions**: Does the name follow snake_case style? Is it concise but unambiguous?
4. **Unit consistency**: If units are provided, is the physical_base consistent with those units?
5. **Duplicate avoidance**: Is the name unique relative to existing names?

## Verdicts

- **accept**: The name is correct, follows conventions, and accurately represents the source quantity.
- **reject**: The name has fundamental issues (wrong physics, invalid grammar, meaningless). Provide clear reasons.
- **revise**: The name has fixable issues. Provide `revised_name` and `revised_fields` with the corrected version.

{% if existing_names %}
## Existing Standard Names (must not duplicate)
{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

## Candidates to Review

{% for item in items %}
### Candidate {{ loop.index }}
- **Standard name**: {{ item.id }}
- **Source ID**: {{ item.source_id }}
- **Physical base**: {{ item.physical_base or 'unspecified' }}
- **Subject**: {{ item.subject or 'none' }}
- **Component**: {{ item.component or 'none' }}
- **Position**: {{ item.position or 'none' }}
- **Units**: {{ item.units or 'unspecified' }}
- **Description**: {{ item.description or 'none' }}

{% endfor %}

## Output Format

Return a JSON object matching this schema:

```json
{
  "reviews": [
    {
      "source_id": "path/to/quantity",
      "standard_name": "electron_temperature",
      "verdict": "accept",
      "confidence": 0.95,
      "reason": "Name correctly captures the physics quantity",
      "revised_name": null,
      "revised_fields": null,
      "issues": []
    },
    {
      "source_id": "path/to/other",
      "standard_name": "bad_name",
      "verdict": "revise",
      "confidence": 0.8,
      "reason": "Subject should be 'ion' not 'ions'",
      "revised_name": "ion_temperature",
      "revised_fields": {"physical_base": "temperature", "subject": "ion"},
      "issues": ["Invalid subject value 'ions'"]
    }
  ]
}
```

- **source_id**: The source entity ID from the candidate
- **standard_name**: The name being reviewed (as provided)
- **verdict**: One of "accept", "reject", "revise"
- **confidence**: Float 0.0-1.0 — higher when the review is decisive
- **reason**: Brief justification for the verdict
- **revised_name**: Only for "revise" verdicts — the corrected name string
- **revised_fields**: Only for "revise" verdicts — dict of corrected grammar fields
- **issues**: List of specific problems found (empty list if none)
