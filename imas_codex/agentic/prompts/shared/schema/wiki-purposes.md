## page_purpose Values (CRITICAL - use exactly these values)

Select the most appropriate classification for each wiki page based on its **content**, not just title or graph position.

### High-Value Technical Content (priority 1.0)

{% for p in wiki_purposes_high %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Medium-Value Content (priority 0.8)

{% for p in wiki_purposes_medium %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

### Low-Value Content (priority 0.3)

{% for p in wiki_purposes_low %}
- `{{ p.value }}`: {{ p.description }}
{% endfor %}

**Examples:**
- Signal list with MDSplus paths → `data_source`
- Thomson scattering documentation → `diagnostic`
- LIUQE equilibrium code docs → `code`
- How to connect to MDSplus → `data_access`
- Sensor calibration factors → `calibration`
- Weekly meeting notes → `administrative`
- User:John sandbox → `personal`
