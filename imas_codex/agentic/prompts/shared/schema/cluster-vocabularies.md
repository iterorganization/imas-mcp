## Controlled Vocabularies (CRITICAL — use ONLY these values)

### Physics Concepts (select 1-3)

{% for c in cluster_physics_concepts %}
- `{{ c.value }}`: {{ c.description }}
{% endfor %}

### Data Types (select exactly 1)

{% for d in cluster_data_types %}
- `{{ d.value }}`: {{ d.description }}
{% endfor %}

### Tags (select 1-5)

{% for t in cluster_tags %}
- `{{ t.value }}`: {{ t.description }}
{% endfor %}

### Mapping Relevance (select exactly 1)

{% for r in cluster_mapping_relevance %}
- `{{ r.value }}`: {{ r.description }}
{% endfor %}
