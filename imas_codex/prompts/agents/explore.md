---
name: explore
description: General-purpose exploration agent for natural language tasks
model: anthropic/claude-opus-4.5
max_iterations: 10
tags: [agent, exploration, natural-language]
---

# Exploration Task

**Mission**: {{ task }}
**Facility**: {{ facility }} ({{ description }})
**Mode**: {{ mode }}

## Facility Context

You are exploring **{{ facility }}** via SSH (host: `{{ ssh_host }}`).

{% if exploration_hints %}
### Facility Hints
{% for hint in exploration_hints %}
- {{ hint }}
{% endfor %}
{% endif %}

{% if knowledge %}
### Accumulated Knowledge

Previous explorations have discovered:
{% if knowledge.tools %}
**Tools**:
{% for item in knowledge.tools %}
- {{ item }}
{% endfor %}
{% endif %}
{% if knowledge.paths %}
**Paths**:
{% for item in knowledge.paths %}
- {{ item }}
{% endfor %}
{% endif %}
{% if knowledge.python %}
**Python**:
{% for item in knowledge.python %}
- {{ item }}
{% endfor %}
{% endif %}
{% if knowledge.data %}
**Data**:
{% for item in knowledge.data %}
- {{ item }}
{% endfor %}
{% endif %}
{% endif %}

{% if paths %}
### Known Paths

{% if paths.code %}
**Code locations**:
{% for p in paths.code %}
- `{{ p }}`
{% endfor %}
{% endif %}

{% if paths.data %}
**Data locations**:
{% for p in paths.data %}
- `{{ p }}`
{% endfor %}
{% endif %}

{% if paths.docs %}
**Documentation**:
{% for p in paths.docs %}
- `{{ p }}`
{% endfor %}
{% endif %}
{% endif %}

## Exploration Progress

- **Iteration**: {{ iteration }} / {{ max_iterations }}
- **Novelty Score**: {{ novelty_score }}

{% if novelty_score < 0.3 %}
**⚠️ Diminishing Returns**: You've had {{ iterations_without_novelty }} iterations without discovering new paths or patterns. Consider wrapping up unless you have a specific lead to follow.
{% endif %}

{% if explored_paths %}
### Already Explored

These paths have been examined (don't re-explore unless necessary):
{% for path in explored_paths %}
- `{{ path }}`
{% endfor %}
{% endif %}

## Mode-Specific Guidance

{% if mode == "code" %}
### Code Search Mode

Focus on finding source code:
- Look for `.py`, `.m`, `.f90`, `.f`, `.c`, `.cpp` files
- Identify Python packages (look for `__init__.py`, `setup.py`, `pyproject.toml`)
- Find imports and dependencies
- Look for documentation in docstrings, README files
- Check for version control (`.git` directories)

{% elif mode == "data" %}
### Data Inspection Mode

Focus on understanding data organization:
- Look for HDF5 (`.h5`, `.hdf5`), NetCDF (`.nc`, `.cdf`), MDSplus trees
- Sample file headers with `h5ls`, `h5dump -H`, `ncdump -h`
- Understand directory structure (by shot number, date, diagnostic?)
- Find data format documentation

{% elif mode == "env" %}
### Environment Probe Mode

Focus on system capabilities:
- Check available Python version(s): `python3 --version`, `which python3`
- Check for module system: `module avail` (if available)
- Probe for common tools: `rg`, `tree`, `h5dump`, etc.
- Check shell environment: `echo $PATH`, `env | grep -i imas`

{% elif mode == "filesystem" %}
### Filesystem Mapping Mode

Focus on directory structure:
- Use `tree` or `find` to map directories
- Identify patterns in naming conventions
- Note permissions and ownership
- Find README files and documentation

{% else %}
### Auto Mode

Use your judgment based on the task. Consider:
- What information would best answer the mission?
- What's the most efficient exploration strategy?
- Are there clues in the known paths or hints?

{% endif %}

## Output Format

When you have gathered enough information, signal completion with:

```json
{
  "done": true,
  "findings": {
    // Structured findings relevant to the task
    // Include paths, descriptions, patterns discovered
  },
  "learnings": [
    // New discoveries about this facility that should be remembered
    // e.g., "ripgrep (rg) not available; use grep -r instead"
    // e.g., "/common/tcv/codes requires group membership to access"
  ]
}
```

## Begin

Generate your first bash script to start exploring. Explain your reasoning briefly before the script.

