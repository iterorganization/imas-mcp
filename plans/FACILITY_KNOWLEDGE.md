# Facility-Side Knowledge Graph Design

**Status**: Draft
**Related**: CODEX_PLAN.md (Zone B: Source Inventory, Zone C: Forensic Evidence)

---

## Problem Statement

Current exploration workflow has gaps:

1. **Knowledge stored as flat YAML** - The `knowledge:` section in `facilities/*.yaml` is unstructured
2. **No exploration history** - We don't track what was explored, when, or what commands were run
3. **Deduplication impossible** - Future LLMs can't tell if a path was already surveyed
4. **No graph-ready artifacts** - Discoveries are text blobs, not typed entities

The `--finish` command persists learnings, but there's no:
- Versioned history of explorations
- Structured entities (files, directories, tools, MDSplus trees)
- Relationship tracking (e.g., "this Python script imports MDSplus")

---

## Proposed Architecture

### 1. Exploration Sessions → Structured Artifacts

Instead of just persisting YAML to config files, create **typed discovery artifacts**:

```
~/.cache/imas-codex/
├── epfl/
│   ├── sessions/
│   │   ├── 2025-01-15T10-30-00.jsonl   # Command log (current)
│   │   └── ...
│   ├── artifacts/
│   │   ├── environment.json            # Python, tools, OS
│   │   ├── directories.json            # Discovered paths
│   │   ├── mdsplus_trees.json          # MDSplus structure
│   │   └── ...
│   └── manifest.json                   # Index of all artifacts
```

### 2. LinkML Schema for Discoveries

Define typed entities in `ontology/discovery.yaml`:

```yaml
classes:
  Facility:
    attributes:
      id: string
      description: string
      ssh_host: string

  Environment:
    attributes:
      facility: Facility
      python_version: string
      python_path: string
      os_name: string
      os_version: string
      discovered_at: datetime

  ToolAvailability:
    attributes:
      facility: Facility
      tool_name: string
      available: boolean
      path: string
      version: string
      discovered_at: datetime

  DirectoryEntry:
    attributes:
      facility: Facility
      path: string
      entry_type: enum [file, directory, symlink]
      size_bytes: integer
      purpose: string  # e.g., "data", "code", "documentation"
      explored: boolean
      explored_at: datetime

  MDSplusTree:
    attributes:
      facility: Facility
      server: string
      tree_name: string
      explored: boolean
      signal_count: integer
```

### 3. Exploration State Machine

Track exploration progress to avoid duplication:

```
┌─────────────┐    explore    ┌─────────────┐    finish    ┌─────────────┐
│   Unknown   │ ──────────────▶│  Exploring  │ ────────────▶│  Explored   │
└─────────────┘                └─────────────┘              └─────────────┘
                                     │
                                     │ discard
                                     ▼
                               ┌─────────────┐
                               │   Skipped   │
                               └─────────────┘
```

For each path/entity:
- **Unknown**: Not yet discovered
- **Exploring**: Currently being investigated
- **Explored**: Fully surveyed, artifacts persisted
- **Skipped**: Intentionally not explored (e.g., too large, irrelevant)

### 4. Manifest for LLM Context

On each `/explore` prompt, generate a summary manifest:

```yaml
# Auto-generated exploration summary for LLM context
last_exploration: 2025-01-15T10:30:00Z
explorer: "claude-opus-4.5"

environment:
  python: "3.9.21 (/usr/bin/python3)"
  os: "RHEL 9.6"
  status: explored

tools:
  explored: [grep, tree, python3, pip, h5dump]
  unavailable: [rg, conda, module]
  unknown: [ncdump, mdstcl]

directories:
  explored:
    - path: /common/tcv/codes
      files: 1234
      last_explored: 2025-01-15T10:30:00Z
  partial:
    - path: /common/tcv/data
      note: "Too large, only surveyed top-level"
  unknown:
    - /home/*/diagnostics

mdsplus:
  explored_trees: [tcv, tcv_shot]
  unknown_trees: [results, magnetics]
```

This manifest:
1. **Loads into prompt context** - LLM sees what's been done
2. **Prevents duplication** - LLM knows not to re-explore `/common/tcv/codes`
3. **Guides next steps** - LLM sees `unknown` items to prioritize

### 5. CLI Commands

```bash
# View exploration status
uv run imas-codex epfl --status

# List explored artifacts
uv run imas-codex epfl --artifacts

# View specific artifact
uv run imas-codex epfl --artifact environment

# Export for graph building
uv run imas-codex epfl --export-graph > epfl_zone_b.json
```

---

## Implementation Phases

### Phase 1: Structured Finish (Minimal)

**Goal**: Make `--finish` persist to structured JSON, not just YAML blob.

1. Define `EnvironmentArtifact`, `ToolArtifact` Pydantic models
2. Parse the YAML from `--finish` into typed artifacts
3. Store in `~/.cache/imas-codex/{facility}/artifacts/`
4. Keep YAML in config file for backward compatibility

### Phase 2: Exploration Manifest

**Goal**: Track what's been explored vs unknown.

1. Create `Manifest` model with exploration state
2. Load manifest into prompt context
3. Update manifest on `--finish`

### Phase 3: Graph Export

**Goal**: Export artifacts as graph-ready JSON-LD or LinkML instances.

1. Define LinkML schema in `ontology/discovery.yaml`
2. Generate Pydantic models from LinkML
3. Export command produces JSON-LD for Neo4j import

### Phase 4: Session History

**Goal**: Track exploration history for audit trail.

1. Archive session logs after `--finish`
2. Link session to artifacts it produced
3. Track which LLM model did the exploration

---

## Open Questions

1. **Where to store artifacts?**
   - Option A: `~/.cache/imas-codex/` (current session logs location)
   - Option B: `data/facilities/` in repo (version controlled)
   - Option C: Separate `imas-codex-data` repo

2. **How to handle large discoveries?**
   - Directory listings can be huge
   - Store summary + pointer to detailed cache?

3. **Merge strategy for concurrent explorations?**
   - Multiple LLMs exploring same facility
   - CRDT-style merge? Or session locks?

4. **Integration with Neo4j graph?**
   - Should artifacts go directly to graph?
   - Or stay as files until explicit "build graph" command?

---

## Next Steps

1. [ ] Review this design
2. [ ] Define Phase 1 Pydantic models
3. [ ] Implement structured `--finish` parser
4. [ ] Update explore prompt with exploration manifest

