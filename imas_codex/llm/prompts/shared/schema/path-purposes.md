## path_purpose Values (CRITICAL - use exactly these values)

### Code Categories
{% for p in path_purposes_code %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Data Categories
{% for p in path_purposes_data %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Infrastructure Categories
{% for p in path_purposes_infra %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Support Categories
{% for p in path_purposes_support %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Structural Categories
{% for p in path_purposes_structural %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}
