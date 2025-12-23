---
name: check-progress
description: Get exploration progress metrics and recommend next actions
arguments:
  facility:
    type: string
    description: "Facility SSH alias (e.g., 'epfl')"
    required: true
    default: "epfl"
---

# Exploration Status

Get comprehensive exploration progress for **{facility}** and recommend next actions.

## Steps

1. **Get facility info**:
   ```python
   info = get_facility_info("{facility}")
   ```

2. **Query path statistics**:
   ```python
   cypher('''
       MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: "{facility}"}})
       RETURN p.status AS status, count(*) AS count
       ORDER BY status
   ''')
   ```

3. **Calculate metrics**:
   | Metric | Formula |
   |--------|---------|
   | Total paths | Sum of all statuses |
   | Actionable | discovered + listed + scanned |
   | Processed | flagged + analyzed + ingested |
   | Completion % | (processed + skipped) / total × 100 |

4. **Query interest-weighted progress** (optional):
   ```python
   cypher('''
       MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: "{facility}"}})
       WHERE p.interest_score IS NOT NULL
       RETURN 
           sum(CASE WHEN p.status IN ["analyzed", "ingested"] 
               THEN p.interest_score ELSE 0 END) AS completed_interest,
           sum(p.interest_score) AS total_interest
   ''')
   ```

5. **Recommend next action** based on state:

   | State | Recommendation |
   |-------|----------------|
   | No paths discovered | Run `/scout_depth` |
   | Many discovered, few scanned | Run `/triage_paths` |
   | Many flagged, few ingested | Run `/code_hunt` |
   | High completion | Increase depth or explore new roots |

6. **Report**:
   - Progress bar visualization
   - Top 5 actionable paths by interest score
   - Recommended next prompt to run
