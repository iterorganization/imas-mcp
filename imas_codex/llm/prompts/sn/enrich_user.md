---
name: sn/enrich_user
description: Per-batch user prompt for standard name documentation enrichment
used_by: imas_codex.standard_names.workers.enrich_worker
task: enrichment
dynamic: true
schema_needs: []
---

# Names to enrich

{% for item in items %}
## {{ item.name }}

**Unit:** {{ item.unit or "—" }}  **COCOS:** {{ item.cocos or "—" }}
**Kind:** {{ item.kind or "scalar" }}
**Physics domain:** {{ item.physics_domain or "—" }}

{% if item.grammar %}**Grammar decomposition:**
{% for seg, val in item.grammar.items() %}- {{ seg }}: `{{ val }}`
{% endfor %}{% endif %}

### DD path documentation
{% if item.dd_paths %}{% for path in item.dd_paths %}
- `{{ path.path }}`: {{ path.documentation or path.description or "(no documentation)" }}{% endfor %}
{% else %}
_(no linked DD paths)_
{% endif %}

### Nearby standard names (vector similarity)
{% if item.nearby %}{% for n in item.nearby %}
- `{{ n.name }}` — {{ n.description or "(no description)" }}{% endfor %}
{% else %}
_(none available)_
{% endif %}

### Sibling names in {{ item.physics_domain or "same domain" }}
{% if item.siblings %}{% for s in item.siblings %}
- `{{ s.name }}` — {{ s.description or "(no description)" }}{% endfor %}
{% else %}
_(none available)_
{% endif %}

{% if item.current and (item.current.description or item.current.documentation) %}### Current enrichment (improve upon this)
{% if item.current.description %}- **Description:** {{ item.current.description }}{% endif %}
{% if item.current.documentation %}- **Documentation:** {{ item.current.documentation }}{% endif %}
{% if item.current.tags %}- **Tags:** {{ item.current.tags | join(", ") }}{% endif %}
{% if item.current.links %}- **Links:** {{ item.current.links | join(", ") }}{% endif %}
{% endif %}

{% endfor %}

Return a JSON object matching the output schema with one entry per name above.
