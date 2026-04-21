{# Preferred physical_base anchors — injected into compose and review prompts. #}
{# Rendered by: imas_codex/standard_names/context.py::build_compose_context #}
{% if preferred_bases %}

## Preferred ``physical_base`` Anchors (ordering tiebreaker)

The ``physical_base`` segment is **open vocabulary** — new compound tokens
are always allowed. However, when two grammatically-valid forms compete
for the same concept (e.g. ``plasma_boundary_gap_angle`` vs
``angle_of_plasma_boundary_gap``), **prefer the form whose
``physical_base`` is on the anchor list below**. These anchors were mined
from the graph's own emergent vocabulary (physical bases already used by
≥ 2 high-scoring standard names).

**Rules of thumb:**

1. If you are about to emit a `physical_base` that ends with `_<anchor>`
   where `<anchor>` is on the list, rewrite as `<anchor>_of_<prefix>`
   before returning. Example: ❌ `plasma_boundary_gap_angle` →
   ✅ `angle_of_plasma_boundary_gap` (anchor = `angle`).
2. If two candidates differ only in which sub-token is treated as the
   base noun, pick the one whose base noun is on the anchor list.
3. **Never invent a synonym of an anchor.** If `temperature` is on the
   list, do not emit `thermal_level` or `thermal_energy_proxy` as a
   base — use `temperature`.
4. This list is advisory, not closed. If the concept genuinely needs a
   new compound physical_base (e.g. a novel derived quantity with no
   anchor suffix), emit it — but prefer anchor-led forms whenever one
   applies.

**Preferred anchors** (domain in parentheses indicates where the anchor
is most heavily used; anchors are reusable across domains):

{% for anchor in preferred_bases %}
- `{{ anchor.token }}` ({{ anchor.domain }}){% if anchor.examples %} — e.g. {% for ex in anchor.examples %}`{{ ex }}`{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}{% if anchor.note %} — *{{ anchor.note }}*{% endif %}
{% endfor %}

When a DD path suggests a concept whose natural base would be an anchor
suffix (gap, angle, radius, flux, velocity, temperature, pressure,
density, …), use the anchor-led `<anchor>_of_<X>` form.
{% endif %}
