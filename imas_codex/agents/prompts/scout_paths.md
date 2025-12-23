---
name: scout_paths
description: Explore directories at a given depth, ingesting FacilityPath nodes
arguments:
  facility:
    type: string
    description: Facility SSH alias (e.g., "epfl")
    required: true
  root_path:
    type: string
    description: Starting path for exploration
    default: "/"
  depth:
    type: integer
    description: Maximum directory depth to explore
    default: 3
---

# Scout Depth Exploration

Explore the filesystem at **{facility}** starting from `{root_path}` to depth **{depth}**.

## Steps

1. **Check existing state** - Call `get_facility_info("{facility}")` to see:
   - Already discovered paths (avoid duplicates)
   - Excluded paths (skip these)
   - Available tools (rg, fd, etc.)

2. **List directories** via SSH:
   ```bash
   ssh {facility} "fd -t d --max-depth {depth} {root_path} 2>/dev/null | head -200"
   ```
   If `fd` unavailable, fallback to:
   ```bash
   ssh {facility} "find {root_path} -maxdepth {depth} -type d 2>/dev/null | head -200"
   ```

3. **Ingest each directory** as a FacilityPath node:
   ```python
   ingest_node("FacilityPath", {{
       "id": "{facility}:<path>",
       "facility_id": "{facility}",
       "path": "<path>",
       "path_type": "directory",
       "status": "discovered",
       "depth": <calculated_depth>
   }})
   ```

4. **Report summary**:
   - Total directories found
   - New paths ingested
   - Paths already in graph (skipped)

## Notes

- Skip paths matching excludes from `get_facility_info()`
- Calculate depth relative to root_path
- Use `discovered` status for new paths
