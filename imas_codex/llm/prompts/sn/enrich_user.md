---
name: sn/enrich_user
description: Dynamic user prompt for standard name documentation enrichment
used_by: imas_codex.cli.sn.sn_enrich
task: enrichment
dynamic: true
schema_needs: []
---

Enrich the documentation for the following {{ names | length }} standard name(s).

For each name, improve the description, documentation, tags, links, validity_domain, and constraints fields. Use ALL the linked DD path context to write comprehensive, physics-accurate documentation.

Do NOT change the standard name itself — return it exactly as given.

{% for item in names %}
---

### {{ loop.index }}. `{{ item.name }}`

- **Kind**: {{ item.kind or "scalar" }}
- **Unit**: {{ item.unit or "dimensionless" }}
{% if item.description %}- **Current description**: {{ item.description }}{% endif %}
{% if item.documentation %}- **Current documentation**: {{ item.documentation }}{% endif %}
{% if item.tags %}- **Current tags**: {{ item.tags | join(", ") }}{% endif %}
{% if item.links %}- **Current links**: {{ item.links | join(", ") }}{% endif %}
{% if item.validity_domain %}- **Current validity_domain**: {{ item.validity_domain }}{% endif %}
{% if item.constraints %}- **Current constraints**: {{ item.constraints | join(", ") }}{% endif %}

{% if item.grammar %}**Grammar decomposition**:
{% for seg, val in item.grammar.items() %}- {{ seg }}: `{{ val }}`
{% endfor %}{% endif %}

{% if item.dd_paths %}**Linked DD paths** ({{ item.dd_paths | length }}):
{% for p in item.dd_paths %}- `{{ p.path }}`{% if p.description %} — {{ p.description }}{% endif %}{% if p.documentation %} | Docs: {{ p.documentation[:200] }}{% endif %}
{% endfor %}{% endif %}

{% endfor %}
