### Naming Consistency

{% for rule in composition_rules %}
**{{ rule.id }} {{ rule.title }}{% if rule.severity == 'hard' %} — HARD PROHIBITION{% endif %}.** {{ rule.rule | trim }}
{% if rule.examples_good %}
✓ Good: {% for ex in rule.examples_good %}`{{ ex }}`{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}
{% if rule.examples_bad %}
✗ Bad: {% for ex in rule.examples_bad %}`{{ ex }}`{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}

{% endfor %}
