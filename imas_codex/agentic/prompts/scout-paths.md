---
name: scout-paths
description: Discover and score directories at a facility
---

# Directory Discovery Workflow

## 1. Check Facility Constraints First

Before running any disk-intensive commands, check the facility's excludes:

```python
info = get_facility('FACILITY_ID')
excludes = info.get('excludes', {})
print(f"Large dirs to avoid: {excludes.get('large_dirs', [])}")
print(f"Depth limits: {excludes.get('depth_limits', {})}")
```

## 2. Safe Discovery Commands

Use depth-limited scans for large directories:

```bash
# Safe: depth-limited
fd -t d --max-depth 3 /work
dust -d 2 /home/codes

# Dangerous: full scan on large dirs
dust /work  # May timeout!
```

## 3. Discovery Process

1. Start from known entry points in `info['paths']`
2. Use `fd -t d --max-depth N` for directory discovery
3. Run `rg -l --max-count 5` for quick pattern matching
4. Score paths by code density and pattern matches

## 4. Handle Timeouts Gracefully

If a command times out, persist the constraint immediately:

```python
# Persist the problem so future runs avoid it
update_facility_infrastructure('FACILITY_ID', {
    'excludes': {
        'large_dirs': ['/work'],  # Add the problematic path
        'depth_limits': {'/work': 2}  # Or set a depth limit
    }
})

# Add context for humans
add_exploration_note('FACILITY_ID', '/work too large for full scan - use depth=2 or target subdirs')
```

## 5. Batch Ingest Results

```python
add_to_graph("FacilityPath", [
    {"id": "facility:/path", "path": "/path", "facility_id": "facility",
     "path_type": "code_directory", "status": "discovered", "interest_score": 0.7}
])
```

Skip system directories: `/tmp`, `/var`, `/proc`, `__pycache__`, `.git`, `node_modules`.

{% include "safety.md" %}
