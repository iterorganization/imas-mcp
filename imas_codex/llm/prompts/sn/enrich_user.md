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

{% set _all_paths = item.dd_paths | map(attribute='path') | join(' ') if item.dd_paths else '' %}
{% if 'uncertainty_index' in _all_paths or '_index_' in _all_paths %}- **precision**: {{ item.name }} is a dimensionless integer index
{% elif 'grid_object' in _all_paths or 'grid_element' in _all_paths %}- **precision**: {{ item.name }} describes a GGD container, not a leaf quantity
{% elif '/calibration/' in _all_paths or 'jones_matrix' in _all_paths %}- **precision**: {{ item.name }} is a calibration parameter — functional definition only
{% endif %}
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
{% if item.current.links %}- **Links:** {{ item.current.links | join(", ") }}{% endif %}
{% endif %}

{% if item.docs_review_feedback %}### Prior docs-axis reviewer feedback

The previous enrichment was reviewed and scored on the docs axis. Address
the specific weaknesses below — do not regress on dimensions that already
scored well.

- **Docs score:** {{ "%.2f"|format(item.docs_review_feedback.reviewer_score) if item.docs_review_feedback.reviewer_score is not none else "—" }}
{% if item.docs_review_feedback.reviewer_verdict %}- **Verdict:** {{ item.docs_review_feedback.reviewer_verdict }}
{% endif %}{% if item.docs_review_feedback.reviewer_scores %}- **Per-dimension scores (0–20):** {% for dim, val in item.docs_review_feedback.reviewer_scores.items() %}{{ dim }}={{ val }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}{% if item.docs_review_feedback.reviewer_comments %}- **Reviewer critique:** {{ item.docs_review_feedback.reviewer_comments }}
{% endif %}{% if item.docs_review_feedback.validation_issues %}- **Validation issues:** {% for iss in item.docs_review_feedback.validation_issues %}{{ iss }}{% if not loop.last %}; {% endif %}{% endfor %}
{% endif %}{% endif %}

{% if item.link_candidates %}### Candidate cross-references (for `links` field)

Prefer `name:` when the target is already minted. Use `dd:` for paths not
yet named — the pipeline resolves them after this round.

{% for c in item.link_candidates %}- `{{ c.tag }}` [{{ c.kind_hint }}] — {{ c.doc_short }}
{% endfor %}{% endif %}

{% if item.related_neighbours %}### Graph-relationship neighbours (explicit cross-IDS peers)

{% for r in item.related_neighbours %}- `{{ r.path }}` ({{ r.ids }}) — {{ r.relationship_type }}{% if r.via %} via {{ r.via }}{% endif %}
{% endfor %}{% endif %}

{% endfor %}

Return a JSON object matching the output schema with one entry per name above.
