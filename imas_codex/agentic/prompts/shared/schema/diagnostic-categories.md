## Diagnostic Categories

Use these canonical categories when the signal comes from a physical diagnostic system.
The `diagnostic` field should be a **lowercase_snake_case** name matching the physical
system (e.g., `thomson_scattering`, `bolometer`, `langmuir_probes`).

{% for c in diagnostic_categories %}
- `{{ c.value }}`: {{ c.description }}
{% endfor %}

**Naming rules:**
- Always use lowercase_snake_case: `thomson_scattering` not `Thomson_Scattering`
- Use the physical system name, not the MDSplus tree/node name
- Leave `diagnostic` empty for analysis outputs (set `analysis_code` instead)
- Leave `diagnostic` empty for control/machine parameters
