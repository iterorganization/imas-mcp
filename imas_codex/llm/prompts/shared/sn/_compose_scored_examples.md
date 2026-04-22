{% if compose_scored_examples %}
## SCORED EXAMPLES

Previously reviewed standard names demonstrating the target quality range.
Use these to calibrate your output quality.

{% for ex in compose_scored_examples %}
### Example ({{ ex.reviewer_verdict }}, score {{ "%.2f"|format(ex.reviewer_score) }})

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
Description: {{ ex.description }}
{% if ex.documentation %}Documentation: {{ ex.documentation }}{% endif %}

Per-dimension scores:
{% for dim, score in ex.scores.items() %}
- **{{ dim }}: {{ score }}/20**{% if ex.dimension_comments.get(dim) %} — {{ ex.dimension_comments[dim] }}{% endif %}

{% endfor %}
{% if ex.reviewer_comments %}Reviewer summary: *{{ ex.reviewer_comments }}*{% endif %}
{% if ex.physics_domain %}Physics domain: {{ ex.physics_domain }}{% endif %}

{% endfor %}
{% endif %}
