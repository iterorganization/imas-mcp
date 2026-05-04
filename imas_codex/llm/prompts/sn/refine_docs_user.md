---
name: sn/refine_docs_user
description: User prompt for refine_docs — renders DOCS_REVISION_OF chain history so the LLM learns from prior reviewer feedback on documentation
used_by: imas_codex.standard_names.workers.refine_docs_worker
task: enrichment
dynamic: true
schema_needs: []
---
You are refining the documentation for an existing standard name. A reviewer
scored a previous documentation attempt below the acceptance threshold. Study
the revision history below and produce improved documentation that materially
addresses the reviewer's concerns.

---

## Standard name being documented

- **Name:** `{{ sn_name }}`
- **Unit:** {{ unit or "—" }}
- **Kind:** {{ kind or "scalar" }}
- **Physics domain:** {{ physics_domain or "—" }}
{% if description %}
- **One-line description:** {{ description }}
{% endif %}

### Linked DD paths
{% if dd_paths %}
{% for path in dd_paths %}
- `{{ path.path }}`{% if path.ids %} ({{ path.ids }}{% if path.unit %}, unit: {{ path.unit }}{% endif %}){% elif path.unit %} (unit: {{ path.unit }}){% endif %}: {{ path.documentation or path.description or "(no documentation)" }}
{% endfor %}
{% else %}
_(no linked DD paths)_
{% endif %}

---

## Docs revision history (oldest first; docs chain length so far: {{ docs_chain_length }})

{% if docs_chain_history %}
{% for h in docs_chain_history %}
### Revision {{ loop.index }} (model: {{ h.model }})

- **Reviewer score:** {{ "%.2f"|format(h.reviewer_score) }}
- **Per-dimension comments:**
{% if h.reviewer_comments_per_dim %}
{% for dim, comment in h.reviewer_comments_per_dim.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

**Prior documentation:**

{{ h.documentation or "(empty)" }}

---
{% endfor %}
{% else %}
_(no prior revision history — this is the first docs refine attempt)_
{% endif %}
{% if item.reviewer_score_docs is not none or item.reviewer_comments_per_dim_docs %}
### Current node docs review

- **Reviewer score:** {{ "%.2f"|format(item.reviewer_score_docs) if item.reviewer_score_docs is not none else "—" }}
- **Per-dimension comments:**
{% set _per_dim_docs = (item.reviewer_comments_per_dim_docs | fromjson) if item.reviewer_comments_per_dim_docs else {} %}
{% if _per_dim_docs %}
{% for dim, comment in _per_dim_docs.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

{% endif %}

---

## Your task

Produce updated documentation for `{{ sn_name }}` that materially addresses
the **lowest-scoring dimensions** identified in the revision history above.

Rules:
- `documentation`: rich text with LaTeX math notation where appropriate,
  typical value ranges, physical intuition, measurement context.
  Do **not** regress on dimensions that already scored well.
- `links`: list of related standard names in `name:xxx` or `dd:path/here`
  format.  Only include genuine conceptual relationships.
- `tags`: relevant physics domain / IDS tags (lowercase, snake_case).
- The `name` field must equal `{{ sn_name }}` exactly — do not alter it.

Return a JSON object matching the output schema with fields:
``description``, ``documentation``, ``links``, ``tags``.
