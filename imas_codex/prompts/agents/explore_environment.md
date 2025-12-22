---
name: explore_environment
description: Discover Python, OS, compilers, and module systems
tags: [mcp, exploration, cursor, cli, environment]
artifact_type: environment
---

{% include "explore_core.md" %}

---

## Environment Exploration

Focus on discovering:
- **Python**: version, path, installed packages
- **Operating System**: distribution, version, kernel
- **Compilers**: gcc, gfortran, Intel compilers (icx, ifx, icc)
- **Module System**: Lmod, Environment Modules availability

## Recommended Commands

```bash
# Python info
uv run imas-codex {{ facility }} "which python python3; python3 --version; pip list 2>/dev/null | head -20"

# OS info
uv run imas-codex {{ facility }} "cat /etc/os-release | head -10; uname -a"

# Compilers
uv run imas-codex {{ facility }} "gcc --version | head -1; gfortran --version | head -1; which icx ifx icc 2>/dev/null"

# Module system
uv run imas-codex {{ facility }} "which module; module avail 2>&1 | head -30 || echo 'No module system'"
```

## Comprehensive Script

```bash
uv run imas-codex {{ facility }} << 'EOF'
echo "=== Python ==="
which python python3 2>/dev/null
python3 --version 2>/dev/null
pip list 2>/dev/null | head -20

echo "=== OS ==="
cat /etc/os-release | head -10
uname -a

echo "=== Compilers ==="
gcc --version 2>/dev/null | head -1
gfortran --version 2>/dev/null | head -1
which icx ifx icc icpc 2>/dev/null
icx --version 2>/dev/null | head -1

echo "=== Module System ==="
type module 2>/dev/null && module avail 2>&1 | head -30 || echo "No module system"
EOF
```

{% include "schemas/environment_schema.md" %}

