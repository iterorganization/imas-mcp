---
name: scout-code
description: Find code files and queue them for ingestion
---

# Code File Discovery Workflow

Find code files using `rg -l` or `fd` with patterns like 'equilibrium', 'IMAS', 'MDSplus', 'write_ids'.

## Queue Files via CLI

The CLI accepts paths directly as arguments (LLM-friendly):

```bash
# Queue specific files directly
imas-codex ingest queue epfl /path/a.py /path/b.py /path/c.py

# With priority score (0.0-1.0)
imas-codex ingest queue epfl /path/a.py /path/b.py -s 0.9
```

For large batches, pipe from SSH:

```bash
# Pipe search results directly
ssh epfl 'rg -l "MDSplus|IMAS" /home/codes -g "*.py"' | imas-codex ingest queue epfl --stdin

# Or save to file first
ssh epfl 'rg -l "pattern" /path' > files.txt
imas-codex ingest queue epfl -f files.txt
```

Then run ingestion:

```bash
imas-codex ingest run epfl -n 500
```

## What the Pipeline Extracts

The ingestion pipeline automatically extracts from file content:
- MDSplus paths: `\\RESULTS::I_P`, `\\MAGNETICS::IPLASMA`
- TDI function calls: `tcv_eq("PSI")`, `tcv_get("IP")`
- IDS references for linking
- Embeddings for semantic search

Do NOT fabricate `patterns_matched` metadata - the pipeline does real extraction.

## Quality Tips

- Preview 3-5 files before bulk queueing
- Skip test files and backups
- Prioritize IMAS integration and equilibrium code

## ⚠️ Read-Only Policy

**CRITICAL**: Remote facilities are READ-ONLY. You must NOT:

- **Modify files**: No `mv`, `rm`, `cp`, `touch`, `chmod`, `chown`
- **Edit content**: No `sed -i`, `vim`, `nano`, `echo >`, `cat >`
- **Create files**: No `mkdir`, `touch` except in `~/` home directory
- **Change state**: No `git commit`, `git push`, database writes

**Exceptions** (home directory only):
- Install utilities to `~/.local/bin/` or `~/bin/` using cargo, pip --user
- Create temporary working files in `~/tmp/` or `~/.cache/`

If you need to modify facility data, report findings and request human intervention.
