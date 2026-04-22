{% if review_scored_examples %}
## REVIEWER CALIBRATION EXAMPLES

Previously reviewed standard names spanning the full score range. Each
example shows the per-dimension score you must produce and the reasoning
tied to each dimension. Use these to anchor your own scores to a consistent
absolute scale across batches.

{% for ex in review_scored_examples %}
### {{ ex.tier | capitalize }} — aggregate score {{ "%.2f"|format(ex.score) }} ({{ ex.domain }})

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
Description: {{ ex.description }}

Per-dimension scores and reasoning:
{% for dim, score in ex.scores.items() if dim in ex.dimension_comments %}
- **{{ dim }}: {{ score }}/20** — {{ ex.dimension_comments[dim] | default('(no per-dimension comment recorded)') }}
{% endfor %}

{% if ex.issues %}Reviewer-flagged issues: {{ ex.issues | join('; ') }}{% endif %}
{% if ex.verdict %}Verdict: **{{ ex.verdict }}**{% endif %}
{% endfor %}
{% endif %}
