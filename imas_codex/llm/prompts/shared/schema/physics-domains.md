## physics_domain Values (CRITICAL - use exactly these values)

Select the primary physics domain for code directories. Use `general` if no clear domain applies.

{% for d in physics_domains %}
- `{{ d.value }}`: {{ d.description }}
{% endfor %}

**Examples:**
- Equilibrium reconstruction code → `equilibrium`
- Thomson scattering analysis → `electromagnetic_wave_diagnostics`
- Transport solver → `transport`
- MHD stability analysis → `magnetohydrodynamics` (NOT "mhd" or "MHD")
- General utilities, data containers → `general`
