---
name: scout-code
description: Find code files and queue them for ingestion
---

Find code files at flagged paths using `rg -l` or `fd` with patterns like 'equilibrium', 'IMAS', 'write_ids'. Preview promising files to assess relevance.

Queue discovered files using ingest_nodes:
```
ingest_nodes("SourceFile", [
    {"id": "epfl:/path/file.py", "path": "/path/file.py", "facility_id": "epfl",
     "status": "queued", "interest_score": 0.8, "patterns_matched": ["IMAS"]}
])
```

After queueing, the user runs `imas-codex ingest run <facility>` to process them. Already-queued or ingested files are automatically skipped. Skip test files and generated code unless relevant.
