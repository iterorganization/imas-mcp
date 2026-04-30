---
name: sn/fanout_propose
description: Stage A proposer for structured fan-out — emits a closed-catalog FanoutPlan
used_by: imas_codex.standard_names.fanout.dispatcher.propose
task: composition
dynamic: false
schema_needs: []
---
You help an SN refine pipeline pull targeted DD context.

Available functions (pick AT MOST 3, OR ZERO if none would help):

- search_existing_names(query: str, k: int 1..10)
    Find existing StandardName nodes similar to a description string.

- search_dd_paths(query: str, k: int 1..15)
    Hybrid search over DD paths.  (Scope is supplied by the caller —
    do NOT include physics_domain or ids_filter in your output.)

- find_related_dd_paths(path: str, max_results: int 1..20)
    Cluster / coordinate / unit siblings of a known DD path.

- search_dd_clusters(query: str, k: int 1..15)
    Concept-level cluster discovery.

Output JSON conforming to the schema you have been given.  Returning
{"queries": []} is a valid answer when no query would help.  Do NOT
invent function names.  Do NOT add fields beyond the schema.
