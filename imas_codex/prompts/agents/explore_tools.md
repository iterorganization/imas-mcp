---
name: explore_tools
description: Discover available CLI tools on the remote system
tags: [mcp, exploration, cursor, cli, tools]
artifact_type: tools
---

{% include "explore_core.md" %}

---

## Tools Exploration

Focus on discovering availability of CLI tools in these categories:

| Category | Tools to Check |
|----------|----------------|
| **Search** | rg, ag, grep, find |
| **Text** | awk, sed, jq |
| **File** | tree, file, stat, ls |
| **Editor** | vim, nano, emacs |
| **VCS** | git, svn, hg |
| **Network** | wget, curl, rsync, ssh |
| **Build** | make, cmake, ninja |
| **Compiler** | gcc, gfortran, icc, icx, ifx |
| **MPI** | mpirun, mpicc, srun |
| **Scheduler** | sbatch, qsub, bsub |
| **Data** | h5dump, ncdump, mdstcl |
| **Python** | pip, conda, jupyter, ipython |

## Recommended Commands

```bash
# Quick check for common tools
uv run imas-codex {{ facility }} "for t in rg ag grep find tree h5dump ncdump git make cmake; do which \$t 2>/dev/null && echo '  -> available' || echo \"\$t: not found\"; done"

# Check with versions
uv run imas-codex {{ facility }} "git --version; cmake --version | head -1; make --version | head -1"
```

## Comprehensive Script

```bash
uv run imas-codex {{ facility }} << 'EOF'
echo "=== Search Tools ==="
for tool in rg ag grep find; do
    loc=$(which $tool 2>/dev/null)
    if [ -n "$loc" ]; then
        echo "$tool: $loc"
    else
        echo "$tool: not found"
    fi
done

echo "=== Text Processing ==="
for tool in awk sed jq; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== File Tools ==="
for tool in tree file stat; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Editors ==="
for tool in vim nano emacs; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Build Tools ==="
for tool in make cmake ninja; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Data Tools ==="
for tool in h5dump ncdump mdstcl; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Job Schedulers ==="
for tool in sbatch srun qsub bsub; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done
EOF
```

{% include "schemas/tools_schema.md" %}

