{% if compose_scored_examples %}
## SCORED EXAMPLES

Previously reviewed standard names demonstrating the target quality range.
Use these to calibrate your output quality.

{% for ex in compose_scored_examples %}
### {{ ex.tier | capitalize }} (score {{ "%.2f"|format(ex.score) }}, {{ ex.domain }})
- **`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
  - {{ ex.description }}
  - Reviewer note: *{{ ex.comments | truncate(200, True, "…") }}*
{% endfor %}
{% endif %}
