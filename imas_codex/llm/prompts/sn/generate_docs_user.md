---
name: sn/generate_docs_user
description: Per-item user prompt for generate_docs — writes description and documentation for a single accepted standard name
used_by: imas_codex.standard_names.workers.process_generate_docs_batch
task: generate_docs
dynamic: true
schema_needs: []
---

# Generate documentation for: {{ item.name }}

This standard name has passed name review and is now accepted. Your task is to write
clear, complete `description` and `documentation` fields. You must NOT change the name,
kind, unit, tags, or any other identity field.

**Standard name:** `{{ item.name }}`
**Unit:** {{ item.unit or "—" }}
**Kind:** {{ item.kind or "scalar" }}
**Physics domain:** {{ item.physics_domain or "—" }}
{% if item.tags %}**Tags:** {{ item.tags | join(", ") }}{% endif %}

## Why this name was accepted (reviewer feedback)

{% if item.reviewer_score_name is defined and item.reviewer_score_name is not none %}
- **Reviewer score:** {{ "%.2f"|format(item.reviewer_score_name) }}
{% endif %}
{% if item.reviewer_comments_name %}
- **Reviewer comments:** {{ item.reviewer_comments_name }}
{% endif %}
{% if (item.reviewer_score_name is not defined or item.reviewer_score_name is none) and not item.reviewer_comments_name %}
_(no reviewer feedback available)_
{% endif %}

{% if item.chain_history and item.chain_history | length > 0 %}
## Name evolution history (chain)

This name was refined through {{ item.chain_history | length }} predecessor(s). Write
documentation that reflects the FINAL accepted name — the chain is provided as context
to understand what the name represents and how reviewers refined it.

{% for h in item.chain_history %}
### Predecessor {{ loop.index }}: `{{ h.name }}`
{% if h.description %}- Description: {{ h.description }}{% endif %}
{% if h.reviewer_score_name is defined and h.reviewer_score_name is not none %}- Reviewer score: {{ "%.2f"|format(h.reviewer_score_name) }}{% endif %}
{% if h.reviewer_comments_per_dim_name %}- Reviewer comments: {{ h.reviewer_comments_per_dim_name }}{% endif %}
{% endfor %}
{% endif %}

{% if item.description %}
## Existing description (improve or replace)

{{ item.description }}
{% endif %}

## Output schema

Return a JSON object with exactly these two fields:

```json
{
  "description": "1-3 sentences, ≤500 chars, no LaTeX, American spelling",
  "documentation": "Rich markdown with $LaTeX$, typical values, measurement methods, cross-references"
}
```

### description requirements
- 1–3 sentences maximum, ≤ 500 characters
- Physics-meaningful: add information beyond what the name tokens alone encode
- American spelling (ionization, behavior, etc.)
- No LaTeX; no inline units (unit is shown separately)
- No trailing "See also:" blocks

### documentation requirements
- ≥ 3 sentences
- Cover: physical meaning, governing equations (LaTeX), typical values, measurement methods
- Cross-references to related standard names use `[label](name:bare_id)` inline links only
- Sign convention sentence starting "Positive when …" if COCOS-dependent; omit if sign-invariant
- American spelling throughout
- Minimum 20 characters
