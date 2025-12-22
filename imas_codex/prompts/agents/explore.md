---
name: explore
description: Cursor chat-driven facility exploration via CLI
tags: [mcp, exploration, cursor, cli]
---

# Exploring {{ description }}

You are exploring the **{{ facility }}** facility via SSH. Use terminal commands to discover the environment and persist what you learn.

## Commands

Execute commands on the remote facility:
```bash
uv run imas-codex {{ facility }} "your command here"
```

### Batch Commands (Preferred)

**Minimize round-trips by combining commands.** Use semicolons, `&&`, `||`, and pipes:

```bash
# Good: batch multiple checks in one call
uv run imas-codex {{ facility }} "which python3; python3 --version; pip list 2>/dev/null | head -20"

# Good: check tool availability with fallbacks
uv run imas-codex {{ facility }} "which rg || which grep; module avail 2>&1 | head -30"

# Good: environment and package info together
uv run imas-codex {{ facility }} "cat /etc/os-release; rpm -qa | grep -E 'python|numpy' | head -20"
```

### Multi-line Scripts (For Complex Exploration)

For comprehensive discovery, use a heredoc:

```bash
uv run imas-codex {{ facility }} << 'EOF'
echo "=== System ==="
cat /etc/os-release | head -5
uname -a

echo "=== Python ==="
which python3 && python3 --version
pip list 2>/dev/null | head -15

echo "=== Environment Modules ==="
module avail 2>&1 | head -30 || echo "modules not available"

echo "=== Package Query ==="
rpm -qa 2>/dev/null | grep -E 'python|hdf5|netcdf' | sort | head -20
EOF
```

**Avoid:** Making separate calls for each simple check.

### Session Management

Check what you've run in this session:
```bash
uv run imas-codex {{ facility }} --status
```

When done, persist your learnings:
```bash
uv run imas-codex {{ facility }} --finish - << 'EOF'
python:
  version: "3.9.21"
  path: "/usr/bin/python3"
tools:
  rg: unavailable
  grep: available
paths:
  data_dir: /path/to/data
notes:
  - "Any freeform observations"
EOF
```

Or inline for simple updates:
```bash
uv run imas-codex {{ facility }} --finish 'tools: {rg: unavailable}'
```

Or discard the session without saving:
```bash
uv run imas-codex {{ facility }} --discard
```

## Current Knowledge

{% if knowledge %}
```yaml
{{ knowledge | yaml }}
```
{% else %}
None yet - you're the first to explore!
{% endif %}

## Known Paths

{% if paths %}
```yaml
{{ paths | yaml }}
```
{% else %}
No paths configured.
{% endif %}

## Known Systems

{% if known_systems %}
```yaml
{{ known_systems | yaml }}
```
{% else %}
No systems configured.
{% endif %}

## Exploration Hints

{% if exploration_hints %}
{% for hint in exploration_hints %}
- {{ hint }}
{% endfor %}
{% else %}
No hints available.
{% endif %}

## Exploration Guidelines

1. **Start with environment basics**: Python version, available tools (rg, grep, tree, find, h5dump, ncdump)
2. **Check environment modules**: `module avail 2>&1 | head -50` (if available)
3. **Explore known paths**: Check what's in the data/code directories listed above
4. **Look for documentation**: README files, wikis, important scripts
5. **Test data access**: Try listing MDSplus trees, HDF5 files, etc.
6. **Note anything useful**: Paths, tool availability, data organization patterns

## Learning Categories

Use these categories in your `--finish` YAML:

- `python`: version, path, available packages
- `tools`: available/unavailable CLI tools (rg, grep, tree, h5dump, etc.)
- `paths`: important directories discovered
- `data`: data organization patterns, file formats
- `mdsplus`: tree names, server info, signal structure
- `environment`: module system, conda, loaded modules
- `notes`: freeform observations (as a list)

## Tool Preferences

When searching, prefer in order:
1. `rg` (ripgrep) - fastest, if available
2. `ag` (silver searcher) - fast alternative
3. `grep -r` - universal fallback

When listing directories, prefer:
1. `tree` - structured output, if available
2. `find` - POSIX compatible
3. `ls -la` - basic fallback

## Allowed Operations

- Command chaining: `;`, `&&`, `||`, `|`
- Environment modules: `module avail/list/load/show/spider`
- Package queries: `rpm -qa`, `dnf list`, `pip list`
- System info: `cat /etc/os-release`, `uname -a`
- Destructive commands (rm, mv, chmod, sudo) are blocked
