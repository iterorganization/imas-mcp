---
name: scout-paths
description: Discover and score directories at a facility
---

Discover directories at the facility starting from a root path. Use `fd -t d` if available, otherwise `find`. For each directory, run a quick pattern search with `rg -l` looking for code files. Score paths higher if they contain substantial code, lower if sparse or empty. Batch ingest all discovered paths using `ingest_nodes("FacilityPath", [...])` with appropriate status and interest_score. Skip system directories like `/tmp`, `/var`, `/proc` and cache directories like `__pycache__`, `.git`. High-score paths are ready for code scouting.
