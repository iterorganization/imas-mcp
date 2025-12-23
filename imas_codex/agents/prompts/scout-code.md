---
name: scout-code
description: Find and ingest Python files with IMAS imports at flagged paths
arguments:
  facility:
    type: string
    description: "Facility SSH alias (e.g., 'epfl')"
    required: true
    default: "epfl"
  path:
    type: string
    description: "Specific path to search (leave empty to use all flagged paths)"
    default: ""
  max_files:
    type: integer
    description: "Maximum files to ingest per session (1-50)"
    default: 10
---

# Code Hunt

Find Python files with IMAS/MDSplus usage at **{facility}** and ingest them as code examples.

## Steps

1. **Get target paths**:
   - If `{path}` specified, use that
   - Otherwise, get flagged paths from `get_facility_info("{facility}")`
   - Sort by `interest_score` descending

2. **Find IMAS-related Python files**:
   ```bash
   ssh {facility} "rg -l 'import imas|from imas|imas.DBEntry|write_ids|get_ids|put_ids' {path} -g '*.py' --max-depth 4 2>/dev/null | head -50"
   ```

3. **For top {max_files} files, get code statistics**:
   ```bash
   ssh {facility} "wc -l <file> && head -50 <file>"
   ```

4. **Ingest promising files**:
   ```python
   ingest_code_files("{facility}", [
       "/path/to/file1.py",
       "/path/to/file2.py"
   ], description="IMAS integration examples from <path>")
   ```

5. **Update path status** (batch all updates):
   ```python
   ingest_nodes("FacilityPath", [
       {
           "id": "{facility}:{path}",
           "status": "ingested",
           "files_ingested": <count>,
           "last_examined": "<timestamp>"
       }
   ])
   ```

6. **Report**:
   - Files found with IMAS imports
   - Files ingested (with chunk counts)
   - IDS references discovered

## File Selection Criteria

Prioritize files that:
- Have `imas` imports
- Contain `write_ids`, `put_slice`, `get_ids` calls
- Reference specific IDS: `equilibrium`, `core_profiles`, `summary`
- Are not test files or notebooks (skip `test_*.py`, `*.ipynb`)
