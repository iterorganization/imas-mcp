{% if review_scored_examples %}
## REVIEWER CALIBRATION EXAMPLES

Previously reviewed standard names spanning the full score range. Each
example shows the per-dimension score you must produce and the reasoning
tied to each dimension. Use these to anchor your own scores to a consistent
absolute scale across batches.

{% for ex in review_scored_examples %}
### Aggregate score {{ "%.2f"|format(ex.reviewer_score) }}

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
Description: {{ ex.description }}
{% if ex.documentation %}Documentation: {{ ex.documentation }}{% endif %}

Per-dimension scores and reasoning:
{% for dim, score in ex.scores.items() %}
- **{{ dim }}: {{ score }}/20** — {{ ex.dimension_comments.get(dim, '(no per-dimension comment recorded)') }}
{% endfor %}

{% if ex.reviewer_comments %}Reviewer summary: *{{ ex.reviewer_comments }}*{% endif %}
{% if ex.physics_domain %}Physics domain: {{ ex.physics_domain }}{% endif %}

{% endfor %}
{% endif %}
