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
kind, unit, or any other identity field.

**Standard name:** `{{ item.name }}`
**Unit:** {{ item.unit or "—" }}
**Kind:** {{ item.kind or "scalar" }}
**Physics domain:** {{ item.physics_domain or "—" }}

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

{% if item.cocos_label %}
## COCOS Sign Convention

This quantity has COCOS transformation type **{{ item.cocos_label }}**.
{% if item.cocos_guidance %}

{{ item.cocos_guidance }}
{% endif %}

You **MUST** include a sign convention paragraph in the documentation using exactly
this format: `Sign convention: Positive when …` as a standalone paragraph (blank line
before and after, plain text — no markdown headings, no bold).
{% endif %}

{% if item.parent_sn %}
## Parent Standard Name

This is a component of `{{ item.parent_sn.name }}`.
{% if item.parent_sn.description %}- Parent description: {{ item.parent_sn.description }}{% endif %}
{% if item.parent_sn.documentation %}- Parent documentation excerpt: {{ item.parent_sn.documentation[:300] }}{% endif %}

Focus on what distinguishes this {{ item.component_axis or "component" }} specifically.
Cross-reference the parent: `[{{ item.parent_sn.name }}](name:{{ item.parent_sn.name }})`.
{% endif %}

{% if item.child_components %}
## Component Standard Names

This quantity has the following directional/projection components:
{% for c in item.child_components %}- `{{ c.name }}`{% if c.axis %} ({{ c.axis }}){% endif %}{% if c.description %}: {{ c.description }}{% endif %}
{% endfor %}

Write primary documentation here. Components reference this name for shared physics context.
{% endif %}

{% if item.source_paths %}
## IMAS DD Paths

This standard name is sourced from the following IMAS Data Dictionary paths.
Cite at least one verbatim in the documentation prose.

{% for p in item.source_paths %}- `{{ p }}`
{% endfor %}{% endif %}

{% if item.dd_source_docs %}
## DD Source Documentation

Reference material from the Data Dictionary nodes linked to this standard name.
Use these definitions to anchor descriptions; do NOT copy them verbatim.

{% for p in item.dd_source_docs %}- `{{ p.id }}` [{{ p.unit }}]: {{ p.documentation }}
{% endfor %}{% endif %}

{% if item.dd_aliases %}
## DD Aliases (context only — do NOT cite in documentation)

{{ item.dd_aliases | join(', ') }}
{% endif %}

{% if item.nearest_peers %}
## Nearest Peer Standard Names

Concept-similar names already in the catalog.
Use these for inline cross-references `[label](name:bare_id)` where naturally relevant.

{% for n in item.nearest_peers %}- `{{ n.tag }}` [{{ n.unit }}, {{ n.physics_domain }}]: {{ n.doc_short }}{% if n.cocos_label %} (COCOS {{ n.cocos_label }}){% endif %}
{% endfor %}{% endif %}

{% if item.related_neighbours %}
## DD-Related Paths

Cross-IDS related paths sharing cluster membership, coordinates, or units.

{% for r in item.related_neighbours %}- `{{ r.path }}` ({{ r.ids }}) — {{ r.relationship_type }}{% if r.via %} via {{ r.via }}{% endif %}
{% endfor %}{% endif %}

{% if nearby_existing_names %}
## Nearby Existing Names (same physics domain)

For consistency, compare your documentation style and cross-references against these
accepted names in the same physics domain.

{% for n in nearby_existing_names %}- **{{ n.id }}**: {{ n.description | default('', true) }} ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }})
{% endfor %}{% endif %}

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
- American spelling (ionization, behavior, center, etc.)
- No LaTeX; no inline units (unit is shown separately)
- No trailing "See also:" blocks
- **No storage-shape tags**: NEVER write "1D", "2D", "3D", "scalar", "array",
  "profile", or "time-dependent" — describe the *physics*, not the data layout

### documentation requirements
- ≥ 3 sentences
- Cover: physical meaning, governing equations (LaTeX), typical values, measurement methods
- Cross-references to related standard names use `[label](name:bare_id)` inline links only
- {% if item.cocos_label %}Sign convention is REQUIRED for this quantity (see COCOS section above): use exactly `Sign convention: Positive when …` as a standalone paragraph (blank line before and after, plain text — no markdown headings, no bold){% else %}Sign convention (if COCOS-dependent): use exactly `Sign convention: Positive when …` as a standalone paragraph (blank line before and after, plain text — no markdown headings, no bold); omit if sign-invariant{% endif %}
- American spelling throughout
- Minimum 20 characters
