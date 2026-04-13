## Standard Name Grammar

A valid standard name is composed from optional segments in a specific order:

**Canonical pattern:** {{ canonical_pattern }}

### Segment Order

{{ segment_order }}

### Exclusive Pairs

These segment pairs are mutually exclusive — never use both in the same name:
{% for pair in exclusive_pairs %}
- **{{ pair[0] }}** and **{{ pair[1] }}**
{% endfor %}

### Segment Vocabulary

{% for section in vocabulary_sections %}
#### {{ section.segment }}{% if section.is_open %} (open vocabulary){% endif %}

{{ section.description }}
{% if section.template %}
Template: `{{ section.template }}`
{% endif %}
{% if section.tokens %}
Valid tokens: {{ section.tokens | join(', ') }}
{% endif %}
{% if section.is_open %}
Use any physics quantity in snake_case (e.g., temperature, density, magnetic_field, pressure).
{% endif %}

{% endfor %}
