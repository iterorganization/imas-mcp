{% if compose_scored_examples %}
## SCORED EXAMPLES — calibrate your output quality

Previously reviewed standard names spanning the quality spectrum.
**Emulate** Outstanding/Good examples; **avoid** the patterns in Threshold/Inadequate ones.

{% for ex in compose_scored_examples %}
### {{ ex.reviewer_verdict }} example (score {{ "%.2f"|format(ex.reviewer_score) }}, band {{ "%.2f"|format(ex.target_score) }}){% if ex.target_score >= 0.80 %} ✅ EMULATE{% else %} ⚠️ AVOID{% endif %}

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
Description: {{ ex.description }}

Per-dimension scores:
{% for dim, score in ex.scores.items() %}
- **{{ dim }}: {{ score }}/20**{% if ex.dimension_comments.get(dim) %} — {{ ex.dimension_comments[dim] }}{% endif %}

{% endfor %}
{% if ex.reviewer_comments %}Reviewer summary: *{{ ex.reviewer_comments }}*{% endif %}
{% if ex.physics_domain %}Physics domain: {{ ex.physics_domain }}{% endif %}

{% endfor %}
{% endif %}
