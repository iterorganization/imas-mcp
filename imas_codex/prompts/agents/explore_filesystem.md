---
name: explore_filesystem
description: Discover directory structure and file organization
tags: [mcp, exploration, cursor, cli, filesystem]
artifact_type: filesystem
---

{% include "explore_core.md" %}

---

## Filesystem Exploration

Focus on discovering:
- **Directory structure**: Layout of code, data, and documentation
- **Important paths**: Key directories for this facility
- **File patterns**: Common file types and naming conventions
- **Symlinks**: How paths are organized

## Recommended Commands

```bash
# Explore a specific path with tree (if available)
uv run imas-codex {{ facility }} "tree -L 2 /path/to/explore 2>/dev/null || find /path/to/explore -maxdepth 2 -type d"

# Check known paths from facility config
uv run imas-codex {{ facility }} "ls -la /common/tcv/codes 2>/dev/null; ls -la /common/tcv/data 2>/dev/null"

# Find file patterns
uv run imas-codex {{ facility }} "find /path -maxdepth 3 -name '*.py' | head -20"
```

## Comprehensive Script

```bash
uv run imas-codex {{ facility }} << 'EOF'
echo "=== Known Code Paths ==="
{% for path in paths.code %}
echo "{{ path }}:"
ls -la {{ path }} 2>/dev/null | head -10 || echo "  (not accessible)"
{% endfor %}

echo "=== Known Data Paths ==="
{% for path in paths.data %}
echo "{{ path }}:"
ls -la {{ path }} 2>/dev/null | head -10 || echo "  (not accessible)"
{% endfor %}

echo "=== File Type Summary ==="
for ext in py m sh h5 nc mat; do
    count=$(find /common -name "*.$ext" 2>/dev/null | wc -l)
    echo "  .$ext files: $count"
done
EOF
```

## Exploration Tips

1. Start with paths listed in the facility config above
2. Use `tree -L 2` for quick structure overview
3. Note symlinks - they often point to shared resources
4. Look for README files: `find /path -name 'README*' -o -name '*.md'`
5. Check for hidden directories: `.git`, `.svn`, `.config`

{% include "schemas/filesystem_schema.md" %}

