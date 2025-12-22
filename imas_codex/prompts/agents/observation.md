---
name: observation
description: Template for formatting script execution results as observations
tags: [agent, common, observation]
---

**Exit code:** {{ exit_code }}

{% if stdout %}
**stdout:**
```
{{ stdout }}
```
{% endif %}

{% if stderr %}
**stderr:**
```
{{ stderr }}
```
{% endif %}

Analyze these results. You may:
- Generate another script to explore further
- Refine your approach if there were errors
- Signal completion with `{"done": true, "findings": {...}, "learnings": [...]}`

