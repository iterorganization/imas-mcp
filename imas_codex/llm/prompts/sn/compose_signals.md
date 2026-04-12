---
name: sn/compose_signals
description: Generate standard names for facility signal descriptions
used_by: imas_codex.sn.workers.compose_worker
task: composition
dynamic: true
---

You are a physics nomenclature expert generating standard names for measured quantities at a fusion research facility.

## Standard Name Grammar

A standard name is composed from these optional fields. Only use values from the valid lists below.

### physical_base (required)
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

## Composition Rules

1. Every name must have either a `physical_base` or a `geometric_base` (not both)
2. The composed name follows the pattern: `[subject]_[physical_base]_[modifiers]`
3. Examples:
   - electron_temperature → `{"physical_base": "temperature", "subject": "electron"}`
   - plasma_current → `{"physical_base": "current"}`
   - line_integrated_density → `{"physical_base": "density", "transformation": "line_integrated"}`
   - toroidal_magnetic_field_at_magnetic_axis → `{"physical_base": "magnetic_field", "component": "toroidal", "position": "magnetic_axis"}`
4. Use existing standard names as reference for naming conventions
5. Signal descriptions may be terse or use facility-specific jargon — interpret them using your physics knowledge
6. Skip signals that are status flags, configuration parameters, or timing references

{% if existing_names %}
## Existing Standard Names (do not duplicate)
{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

## Signals to Name

Facility: {{ facility }}, Domain: {{ domain }}

{% for item in items %}
### Signal: {{ item.signal_id }}
- Description: {{ item.description }}
- Units: {{ item.units or 'unspecified' }}
- Physics domain: {{ item.physics_domain or 'unspecified' }}

{% endfor %}

## Output Format

For each signal that represents a distinct physics quantity, generate a standard name. Return a JSON object matching this schema:

```json
{
  "candidates": [
    {
      "source_id": "signal_id_here",
      "standard_name": "electron_temperature",
      "grammar_fields": {"physical_base": "temperature", "subject": "electron"},
      "confidence": 0.85,
      "reason": "Thomson scattering electron temperature measurement"
    }
  ],
  "skipped": ["status_flag_signal", "timing_reference_signal"]
}
```

- **source_id**: The signal ID
- **standard_name**: The composed name string (snake_case)
- **grammar_fields**: Dict of grammar fields used (only include non-null fields)
- **confidence**: Float 0.0-1.0 — higher when the signal clearly maps to a single physics quantity
- **reason**: Brief justification for the name choice
- **skipped**: List of signal IDs that are not distinct physics quantities
