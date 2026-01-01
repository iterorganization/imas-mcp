---
name: scout-code
description: Find code files and queue them for ingestion
---

Find code files at flagged paths. Search with `rg -l` or `fd` to find candidates matching the patterns you're looking for. Preview promising files to assess relevance. Queue discovered files using `queue_source_files(facility, [paths], interest_score=0.8)` which creates SourceFile nodes. After queueing, the user runs `imas-codex ingest run <facility>` to process them. Skip test files and generated code unless specifically relevant.
