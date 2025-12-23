---
name: scout-paths
description: Discover directories and score them for IMAS relevance
arguments:
  facility:
    type: string
    description: "Facility SSH alias (e.g., 'epfl')"
    required: true
    default: "epfl"
  root_path:
    type: string
    description: "Starting path for exploration (e.g., '/home/codes')"
    default: "/"
  depth:
    type: integer
    description: "Maximum directory depth to explore (1-5)"
    default: 3
  quick_score:
    type: boolean
    description: "Run quick pattern matching to score paths"
    default: true
---

# Scout Paths

Discover directories at **{facility}** from `{root_path}` to depth **{depth}** and score them.

## Steps

1. **Check existing state**:
   ```python
   info = get_facility_info("{facility}")
   # Check: excluded_paths, actionable_paths, tools available
   ```

2. **Discover directories** via SSH:
   ```bash
   ssh {facility} "fd -t d --max-depth {depth} {root_path} 2>/dev/null | head -200"
   ```
   Fallback if `fd` unavailable:
   ```bash
   ssh {facility} "find {root_path} -maxdepth {depth} -type d 2>/dev/null | head -200"
   ```

3. **Quick score each directory** (if `quick_score` enabled):
   ```bash
   ssh {facility} "rg -l --max-count 1 --max-depth 1 \
       'imas|IMAS|equilibrium|mdsplus' <path> -g '*.py' 2>/dev/null | wc -l"
   ```

   | Matches | Score | Status |
   |---------|-------|--------|
   | 3+ files | 0.8 | flagged |
   | 1-2 files | 0.5 | discovered |
   | 0 files | 0.2 | discovered |
   | Excluded pattern | 0.0 | excluded |

4. **Ingest each path**:
   ```python
   ingest_node("FacilityPath", {
       "id": "{facility}:<path>",
       "facility_id": "{facility}",
       "path": "<path>",
       "path_type": "directory",
       "status": "<status>",
       "interest_score": <score>,
       "depth": <calculated_depth>
   })
   ```

5. **Report summary**:
   - Total directories discovered
   - Paths flagged (high score)
   - Paths skipped (excluded patterns)

## Exclude Patterns

Skip directories matching:
- `/tmp`, `/var`, `/proc`, `/sys`
- `__pycache__`, `.git`, `node_modules`
- Paths already in graph with status != stale

## Notes

- Use `score-paths` prompt for refined re-scoring later
- High-score paths are ready for `scout-code`
