---
name: sn/fanout_evidence_block
description: Renderer template for the markdown evidence block (informational — actual rendering done in fanout/render.py)
used_by: imas_codex.standard_names.fanout.render
task: composition
dynamic: false
schema_needs: []
---
## Fan-out evidence (queries={{ n_queries }}, errors={{ n_errors }})

{% for result in results %}
### {{ result.fn_id }}({{ result.args_repr }})
{% for hit in result.hits %}
- {{ hit.label }}{% if hit.score is not none %}  (score={{ hit.score }}){% endif %}
{% endfor %}
{% endfor %}
