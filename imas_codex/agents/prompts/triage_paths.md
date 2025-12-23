---
name: triage_paths
description: Score discovered paths by IMAS/MDSplus pattern matches and flag interesting ones
arguments:
  facility:
    type: string
    description: Facility SSH alias (e.g., "epfl")
    required: true
  limit:
    type: integer
    description: Maximum paths to triage in one session
    default: 20
---

# Triage Paths

Score and prioritize discovered paths at **{facility}** for further analysis.

## Steps

1. **Get actionable paths** - Call `get_facility_info("{facility}")` and examine `actionable_paths` 
   sorted by interest_score. Focus on paths with status `discovered` or `scanned`.

2. **For each path (up to {limit})**, run pattern search:
   ```bash
   ssh {facility} "rg -l --max-count 1 --max-depth 2 'imas|IMAS|IDS|equilibrium|core_profiles|mdsplus|MDSplus' <path> -g '*.py' -g '*.f90' 2>/dev/null | head -20"
   ```

3. **Score based on matches**:
   | Pattern Found | Interest Score |
   |---------------|----------------|
   | IMAS import/write_ids | 0.9 |
   | MDSplus connection | 0.7 |
   | equilibrium/core_profiles | 0.8 |
   | Generic physics code | 0.5 |
   | No matches | 0.2 |

4. **Update path status**:
   ```python
   ingest_node("FacilityPath", {{
       "id": "{facility}:<path>",
       "status": "flagged",  # or "skipped" if low score
       "interest_score": <score>,
       "patterns_found": ["imas", "equilibrium"],  # what matched
       "description": "Brief note about what was found"
   }})
   ```

5. **Report summary**:
   - Paths flagged for analysis (score >= 0.6)
   - Paths skipped (score < 0.4)
   - Paths needing manual review (0.4-0.6)

## Priority Order

Process paths in this order:
1. Shallow paths (depth 1-2) first
2. Paths with names suggesting code: `code`, `python`, `analysis`, `tools`
3. User home directories with development activity
