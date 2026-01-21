---
name: scout-code
description: Find code files and queue them for ingestion
---

# Code File Discovery Workflow

## 1. Check Facility Constraints First

Before scanning directories, check the facility's excludes:

```python
info = get_facility('FACILITY_ID')
excludes = info.get('excludes', {})
large_dirs = excludes.get('large_dirs', [])
depth_limits = excludes.get('depth_limits', {})

# Avoid large_dirs or use depth limits
for path in large_dirs:
    print(f"AVOID: {path} - too large for full scan")
```

## 2. Safe Search Commands

Use limits to prevent timeouts:

```bash
# Safe: limited results and depth
rg -l --max-count 10 --max-depth 5 "IMAS|MDSplus" /home/codes
fd -e py --max-depth 4 /home/codes | head -100

# Dangerous: unlimited on large dirs
rg -l "pattern" /work  # May timeout!
```

## 3. Find Code Files

Find code files using `rg -l` or `fd` with patterns like 'equilibrium', 'IMAS', 'MDSplus', 'write_ids'.

## Queue Files via CLI

The CLI accepts paths directly as arguments (LLM-friendly):

```bash
# Queue specific files directly
imas-codex ingest queue <facility> /path/a.py /path/b.py /path/c.py

# With priority score (0.0-1.0)
imas-codex ingest queue <facility> /path/a.py /path/b.py -s 0.9
```

For large batches, pipe from SSH:

```bash
# Pipe search results directly
ssh facility 'rg -l "MDSplus|IMAS" /home/codes -g "*.py"' | imas-codex ingest queue facility --stdin

# Or save to file first
ssh facility 'rg -l "pattern" /path' > files.txt
imas-codex ingest queue facility -f files.txt
```

Then run ingestion:

```bash
imas-codex ingest run <facility> -n 500
```

## What the Pipeline Extracts

The ingestion pipeline automatically extracts from file content:
- MDSplus paths: `\\RESULTS::I_P`, `\\MAGNETICS::IPLASMA`
- TDI function calls
- IDS references for linking
- Embeddings for semantic search

Do NOT fabricate `patterns_matched` metadata - the pipeline does real extraction.

## Quality Tips

- Preview 3-5 files before bulk queueing
- Skip test files and backups
- Prioritize IMAS integration and equilibrium code

## 4. Handle Timeouts Gracefully

If a command times out or hangs, **persist the constraint immediately**:

```python
# Get current excludes and update
info = get_facility('FACILITY_ID')
excludes = info.get('excludes', {})
large_dirs = excludes.get('large_dirs', [])
large_dirs.append('/problematic/path')

update_facility_infrastructure('FACILITY_ID', {
    'excludes': {
        'large_dirs': large_dirs,
        'depth_limits': {'/problematic/path': 3}
    }
})

# Add human-readable context
add_exploration_note('FACILITY_ID', '/problematic/path causes timeouts - use --max-depth 3')
```

This ensures future exploration sessions don't repeat the same mistake.

{% include "safety.md" %}
