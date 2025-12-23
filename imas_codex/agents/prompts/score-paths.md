---
name: score-paths
description: Re-score discovered paths with refined criteria (optional, scout-paths does basic scoring)
arguments:
  facility:
    type: string
    description: "Facility SSH alias (e.g., 'epfl')"
    required: true
    default: "epfl"
  status_filter:
    type: string
    description: "Filter paths by status: 'discovered', 'scanned', or 'flagged'"
    default: "discovered"
  limit:
    type: integer
    description: "Maximum paths to score in one session (1-100)"
    default: 30
---

# Score Paths

Re-score paths at **{facility}** with refined criteria.

Use this when:
- Initial scout-paths scoring was too coarse
- You want to apply updated patterns
- Resuming after interruption

## Steps

1. **Get paths to score**:
   ```python
   cypher('''
       MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: "{facility}"}})
       WHERE p.status = "{status_filter}"
       RETURN p.path AS path, p.interest_score AS score
       ORDER BY p.depth, p.path
       LIMIT {limit}
   ''')
   ```

2. **For each path, run pattern search**:
   ```bash
   ssh {facility} "rg -l --max-count 1 --max-depth 2 \
       'imas|IMAS|IDS|equilibrium|core_profiles|mdsplus|MDSplus|write_ids|get_ids' \
       <path> -g '*.py' -g '*.f90' -g '*.F90' 2>/dev/null | head -20"
   ```

3. **Assign scores based on matches**:

   | Pattern Found | Score | Status |
   |---------------|-------|--------|
   | `write_ids`, `put_slice` | 0.95 | flagged |
   | `imas.DBEntry`, `import imas` | 0.9 | flagged |
   | `equilibrium`, `core_profiles` | 0.8 | flagged |
   | `mdsplus`, `MDSplus` | 0.7 | flagged |
   | Generic code, no IMAS | 0.4 | scanned |
   | No code files | 0.1 | skipped |

4. **Batch update all scored paths**:
   ```python
   ingest_nodes("FacilityPath", [
       {
           "id": "{facility}:<path1>",
           "status": "flagged",
           "interest_score": 0.9,
           "patterns_found": ["imas", "write_ids"],
           "last_examined": "<timestamp>"
       },
       {
           "id": "{facility}:<path2>",
           "status": "scanned",
           "interest_score": 0.4,
           "patterns_found": [],
           "last_examined": "<timestamp>"
       },
       # ... all paths in one call
   ])
   ```

5. **Report summary**:
   - Paths upgraded to flagged
   - Paths downgraded to skipped
   - Average score change

## Adaptive Scoring (Future)

Scores could adapt based on:
- Success rate of similar paths (sibling directories)
- Code ingestion yield (files/chunks per path)
- User feedback on false positives
