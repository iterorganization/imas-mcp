---
name: check-tools
description: Check available exploration tools and their versions at a facility
arguments:
  facility:
    type: string
    description: Facility SSH alias (e.g., "epfl")
    required: true
---

# Tool Inventory

Check which exploration tools are available at **{facility}** and record their versions.

## Tools to Check

| Tool | Purpose | Check Command |
|------|---------|---------------|
| `fd` | Fast file finder | `fd --version` |
| `rg` | Fast grep | `rg --version` |
| `tokei` | Lines of code | `tokei --version` |
| `scc` | Code complexity | `scc --version` |
| `dust` | Disk usage | `dust --version` |
| `python3` | Python runtime | `python3 --version` |
| `git` | Version control | `git --version` |

## Steps

1. **Check each tool**:
   ```bash
   ssh {facility} "which <tool> && <tool> --version 2>&1 | head -1"
   ```

2. **Check user bin paths**:
   ```bash
   ssh {facility} "ls -la ~/bin/ 2>/dev/null | head -20"
   ```

3. **Record results** in private file:
   ```python
   update_private("{facility}", {{
       "tools": {{
           "fd": "10.2.0",      # or "unavailable"
           "rg": "14.1.1",
           "tokei": "unavailable",
           "scc": "3.1.0",
           "python3": "3.11.4",
           "git": "2.34.1"
       }}
   }})
   ```

4. **Report summary**:
   - Available fast tools (fd, rg, scc, tokei)
   - Fallback commands needed (find, grep)
   - Python version and key packages

## Notes

- Tools at `~/bin/` may need full path
- Record unavailable tools too (for fallback logic)
- Check if tools are in PATH vs need full path
