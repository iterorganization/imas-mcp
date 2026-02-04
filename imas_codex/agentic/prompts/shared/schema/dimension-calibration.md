## Calibration Examples by Dimension and Score Level

These examples show what scores have historically been assigned across facilities.
Use them to calibrate your scoring decisions - aim for consistency with prior decisions.

{% for dim, levels in dimension_calibration.items() %}
### {{ dim | replace('score_', '') | replace('_', ' ') | title }}

{% for level_name, examples in levels.items() %}
{% if examples %}
**{{ level_name | title }} (~{{ {'lowest': '0.0-0.1', 'low': '0.1-0.3', 'medium': '0.4-0.6', 'high': '0.7-0.9', 'highest': '0.85-0.95'}[level_name] }}):**
{% for ex in examples %}
- `{{ ex.path }}` [{{ ex.facility }}] → {{ ex.score }} - {{ ex.description }}
{% endfor %}
{% else %}
**{{ level_name | title }}:** *No examples available at this level yet.*
{% endif %}
{% endfor %}

{% endfor %}
