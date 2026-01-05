# Agent Guidelines

> **TL;DR**: Use `uv run` for Python commands, `ruff` for linting, conventional commits with single quotes, and `pytest` for testing. No backward compatibility constraints.

## Critical Rules

### NEVER Delete Graph Data Without Backup

Before ANY operation that modifies or deletes graph nodes:

1. **ALWAYS dump the graph first**: `uv run imas-codex neo4j dump`
2. **NEVER use `DETACH DELETE` on production data** without explicit user confirmation
3. **For re-embedding**: Update nodes in place, don't delete and recreate
4. **Ask before destructive operations**: "This will delete X nodes. Should I back up first?"

**Re-embedding workflow** (preserves nodes):
```cypher
-- Update embeddings on existing nodes, don't delete them
MATCH (c:CodeChunk {source_file: $file})
SET c.embedding = $new_embedding
```

## Quick Reference

| Task | Command |
|------|---------|
| Run Python | `uv run <script>` |
| Run tests | `uv run pytest` |
| Run tests with coverage | `uv run pytest --cov=imas_codex` |
| Lint/format | `uv run ruff check --fix . && uv run ruff format .` |
| Sync dependencies | `uv sync --extra test` |
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Start Neo4j | `uv run imas-codex neo4j start` |
| Stop Neo4j | `uv run imas-codex neo4j stop` |
| Neo4j status | `uv run imas-codex neo4j status` |
| Cypher shell | `uv run imas-codex neo4j shell` |
| Dump graph | `uv run imas-codex neo4j dump` |
| Push graph to GHCR | `uv run imas-codex neo4j push v1.0.0` |
| Pull graph from GHCR | `uv run imas-codex neo4j pull` |
| Load graph dump | `uv run imas-codex neo4j load graph.dump` |
| Queue files for ingestion | `uv run imas-codex ingest queue epfl /path/a.py /path/b.py` |
| Queue files from stdin | `ssh epfl 'rg -l pattern /path' \| uv run imas-codex ingest queue epfl --stdin` |
| Ingest queue status | `uv run imas-codex ingest status epfl` |
| Run code ingestion | `uv run imas-codex ingest run epfl` |
| List queued files | `uv run imas-codex ingest list epfl` |
| Create release | `uv run imas-codex release v1.0.0 -m 'message'` |
| Run LLM agent task | `uv run imas-codex agent run "describe path"` |
| Batch enrich paths | `uv run imas-codex agent enrich "\\RESULTS::IBS"` |

## Project Overview

`imas-codex` provides two MCP servers:

**IMAS DD Server** (`imas-codex serve imas`):
- Semantic search across fusion physics data structures
- IDS (Interface Data Structures) exploration
- Path validation and relationship discovery
- Physics domain exploration

**Agents Server** (`imas-codex serve agents`):
- Remote facility exploration via subagents
- File system mapping, code search, data inspection
- Command/Deploy architecture with specialist agents

**LlamaIndex Agents** (`imas-codex agent`):
- Autonomous ReActAgents for metadata enrichment
- IMAS ↔ MDSplus mapping discovery
- Tools for graph queries, SSH, and semantic search
- See [agents/README.md](imas_codex/agents/README.md) and [plans/LLAMAINDEX_AGENTS.md](plans/LLAMAINDEX_AGENTS.md)

## Agent Workflows

### Committing Changes

**Step-by-step procedure:**

```bash
# 1. Check current state
git status

# 2. Run linting on Python files only
uv run ruff check --fix .
uv run ruff format .

# 3. Stage specific files (NEVER use git add -A)
git add <file1> <file2> ...

# 4. Commit with conventional format (use single quotes)
git commit -m 'type: brief description

Detailed body explaining what changed and why.'

# 5. Fix pre-commit errors and repeat steps 3-4 until clean

# 6. Push
git push origin main
```

**Commit message format:**

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `docs` | Documentation |
| `test` | Test changes |
| `chore` | Maintenance |

**Breaking changes**: Add `BREAKING CHANGE:` footer in the body (not `type!:` suffix).

### Testing

```bash
# Sync dependencies first (required in worktrees)
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=imas_codex

# Run specific test
uv run pytest tests/path/to/test.py::test_function

# Fast iteration: use restricted IDS filters for build scripts
uv run build-clusters --ids-filter "core_profiles equilibrium" -v -f
```

### Working in Worktrees

Cursor remote agents often work in auto-created worktrees. Follow this workflow for clean commits:

**Step 1: Commit in the worktree**
```bash
cd /path/to/worktree

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Stage and commit
git add <file1> <file2> ...
git commit -m 'type: description'
```

**Step 2: Cherry-pick to main workspace**

The `main` branch is typically checked out in the primary workspace, so cherry-pick:
```bash
cd /home/ITER/mcintos/Code/imas-codex
git cherry-pick <commit-hash-from-worktree>
git push origin main
```

**Step 3: Clean up worktree**
```bash
cd /path/to/worktree
git checkout -- .  # Discard any remaining changes
```

**Why this workflow?**
- Worktrees share the same `.git` directory, so commits are visible across all worktrees
- `main` cannot be checked out in multiple places simultaneously
- Cherry-picking preserves commit metadata and allows focused commits
- Clean worktrees prevent stale files from appearing in Cursor's review panel

## Rules

### DO

- Use `uv run` for all Python commands
- Use single quotes for commit messages
- Stage files individually (`git add <file>`)
- Use modern Python 3.12 syntax: `list[str]`, `X | Y`
- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations
- Use exception chaining with `from`

### DON'T

- Don't prefix commands with `cd /path &&`
- Don't manually activate venv (`.venv/bin/activate`)
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use double quotes with special characters in commits
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

## Project Structure

```
imas_codex/
├── agents/         # MCP server for exploration prompts
├── config/         # Facility and core configuration (YAML)
├── core/           # Data models, XML parsing, physics domains
├── discovery/      # Facility config loading
├── embeddings/     # Vector embeddings and semantic search
├── graph/          # Neo4j knowledge graph (see graph/README.md)
├── models/         # Pydantic models
├── remote/         # Remote facility connection
├── schemas/        # LinkML schemas (source of truth)
├── search/         # Search functionality
├── services/       # External services
├── tools/          # MCP tool implementations
└── resources/      # Generated data files

tests/              # Mirror source structure
```

## Knowledge Graph Strategy

**Single source of truth**: Neo4j graph database (versioned as OCI artifacts on GHCR)

### Neo4j Service (Apptainer)

Local development uses Neo4j via Apptainer. The CLI handles all setup:

```bash
# Start Neo4j (first run pulls image automatically)
uv run imas-codex neo4j start

# Check status
uv run imas-codex neo4j status

# Stop Neo4j
uv run imas-codex neo4j stop

# Open interactive Cypher shell
uv run imas-codex neo4j shell

# Access Neo4j Browser
open http://localhost:7474
```

**Environment variables:**
- `NEO4J_IMAGE`: Path to SIF image (default: `~/apptainer/neo4j_2025.11-community.sif`)
- `NEO4J_DATA`: Data directory (default: `~/.local/share/imas-codex/neo4j`)
- `NEO4J_PASSWORD`: Password (default: `imas-codex`)

**First-time setup:**
```bash
# Pull the Neo4j image (one-time)
apptainer pull --dir ~/apptainer docker://neo4j:2025.11-community

# Start - password will be set automatically
uv run imas-codex neo4j start
```

**Connection details for GraphClient:**
- URI: `bolt://localhost:7687`
- User: `neo4j`
- Password: `imas-codex` (or `$NEO4J_PASSWORD`)

### Graph Artifact Versioning

The graph database is versioned as an OCI artifact on GHCR:

```bash
# Pull latest graph from GHCR
uv run imas-codex neo4j pull

# Load into Neo4j
uv run imas-codex neo4j load imas-codex-graph.dump

# After making changes, dump the graph
uv run imas-codex neo4j stop
uv run imas-codex neo4j dump

# Push to GHCR (requires GHCR_TOKEN in .env)
uv run imas-codex neo4j push v1.0.0
```

**GHCR Authentication:**
```bash
# Add to .env (gitignored) - required for pushing
# Graph is public, so pulling doesn't require auth
GHCR_TOKEN=ghp_your_token_here
```

### Release Workflow

The release command has two modes based on the target remote.
The project version is derived from git tags via hatch-vcs.

**Mode 1: Prepare PR (`--remote origin`)**
- Pushes branch to origin
- Creates and pushes tag to origin

**Mode 2: Finalize Release (`--remote upstream`, default)**
- Verifies clean tree, synced with upstream
- Updates `_GraphMeta` node with version
- Dumps and pushes graph to GHCR
- Creates and pushes tag to upstream (triggers CI)

**Step-by-step:**
```bash
# 1. Prepare PR (pushes tag to fork)
uv run imas-codex release v4.0.0 -m 'Release message' --remote origin

# 2. Create PR on GitHub, merge to upstream

# 3. Sync with upstream
git pull upstream main

# 4. Finalize release (graph to GHCR, tag to upstream)
uv run imas-codex release v4.0.0 -m 'Release message'
```

**Options:**
```bash
# Preview changes
uv run imas-codex release v4.0.0 -m 'Test' --dry-run

# Skip graph operations
uv run imas-codex release v4.0.0 -m 'Code only' --skip-graph

# Skip git tag
uv run imas-codex release v4.0.0 -m 'Graph only' --skip-git
```

### LLM-First Cypher Queries

Generate Cypher directly instead of using Python wrappers. Use `UNWIND` for batch operations:

```python
from imas_codex.graph import GraphClient

with GraphClient() as client:
    # Batch insert tools
    tools = [{"id": "epfl:gcc", "name": "gcc", "available": True, "version": "11.5.0"}]
    client.query("""
        UNWIND $tools AS tool
        MERGE (t:Tool {id: tool.id})
        SET t += tool
        WITH t
        MATCH (f:Facility {id: 'epfl'})
        MERGE (t)-[:FACILITY_ID]->(f)
    """, tools=tools)

    # Query unavailable tools (useful for agent guidance)
    result = client.query("""
        MATCH (t:Tool)-[:FACILITY_ID]->(f:Facility {id: 'epfl'})
        WHERE t.available = false
        RETURN t.name ORDER BY t.name
    """)
```

### Project Versioning

This project uses **automatic versioning via git tags** (`hatch-vcs`). The version is derived directly from tags:

```bash
# Check current version
uv run python -c "import imas_codex; print(imas_codex.__version__)"
# e.g., "3.2.0" (from tag v3.2.0)

# Development versions include commit info
# e.g., "3.2.0.dev5+g1a2b3c4" (5 commits after v3.2.0)
```

**Key points:**
- **No manual version edits** - version comes from git tags
- **LinkML schema versions** in `schemas/*.yaml` are updated by `release` command to match
- **SemVer**: Use minor bumps (`v3.3.0`) for new features, patch (`v3.2.1`) for fixes

**Do not** manually edit version fields in `pyproject.toml` or schema files.

### Schema Evolution

Neo4j is schema-optional. Additive changes are safe:

| Change | Safe? | Notes |
|--------|-------|-------|
| New class (node label) | ✅ | Old dumps load fine |
| New property | ✅ | Missing = `null` |
| Rename property | ❌ | Requires migration |
| Remove class | ⚠️ | Old data orphaned |

**Policy:** Additive changes only. Never rename/remove, just add.

### _GraphMeta Node

Every graph has a metadata node tracking version and contents:

```cypher
MATCH (m:_GraphMeta) RETURN m
-- Returns: {version: "1.0.0", message: "Add EPFL", facilities: ["epfl"]}
```

### Principles

1. **Schema-first**: All data models defined in LinkML YAML
2. **Auto-generate**: Pydantic models generated via `gen-pydantic`
3. **Runtime introspection**: Node labels, relationships derived from schema via `GraphSchema`
4. **No hard-coded duplication**: All graph structure comes from LinkML

### Files

| File | Status | Purpose |
|------|--------|---------|
| `schemas/facility.yaml` | ✅ Source | LinkML schema definition |
| `graph/models.py` | ✅ Generated | Pydantic models from LinkML |
| `graph/schema.py` | ✅ Runtime | Schema-driven graph ontology via SchemaView |
| `graph/client.py` | ✅ Stable | Neo4j operations via `query()` |

### Usage

```python
from imas_codex.graph import GraphSchema, get_schema

schema = get_schema()
print(schema.node_labels)         # ['Facility', 'MDSplusServer', ...]
print(schema.relationship_types)  # ['FACILITY_ID', 'TREE_NAME', ...]
print(schema.get_identifier("Facility"))  # 'id'
```

When modifying graph structure:
1. Add new classes/relationships to `schemas/facility.yaml`
2. Regenerate models: `uv run build-models --force`
3. Schema changes automatically propagate to `GraphSchema` at runtime

See `graph/README.md` for detailed usage examples.

## Remote Facility Exploration

The Agents MCP server enables exploration of remote fusion facilities.

> **MDSplus Trees**: For ingesting MDSplus tree structures, see [plans/MDSPLUS_INGESTION.md](plans/MDSPLUS_INGESTION.md).

### Architecture

The system uses a **Command/Deploy** pattern:
1. **Commander** (you, the LLM in chat) reads instructions and orchestrates
2. **Direct SSH** for exploration (faster than CLI wrapper)
3. **Artifact capture** via CLI for validated persistence

### Facility Configuration Files

Each facility has two config files:

| File | Visibility | Content |
|------|------------|---------|
| `<facility>.yaml` | ✅ Git tracked | Public data (machine, data systems) |
| `<facility>_infrastructure.yaml` | ❌ Gitignored | Sensitive data (paths, tools, OS) |

**Example:**
```
config/facilities/
├── epfl.yaml                    # Public - goes to graph
└── epfl_infrastructure.yaml     # Private - agent guidance only
```

### Data Classification Policy

**Rule**: If the LinkML schema has a property for it, store it in the graph. Otherwise, use infrastructure files.

**Graph (Public)** - Data access semantics:

| Data Type | Schema Property | Purpose |
|-----------|-----------------|---------|
| MDSplus tree names | `MDSplusTree.name` | Data discovery |
| Diagnostic names | `Diagnostic.name` | Data discovery |
| Analysis code names | `AnalysisCode.name` | Code discovery |
| Code versions | `AnalysisCode.version` | Reproducibility |
| Code paths | `AnalysisCode.path` | Data access |
| TDI function names | `TDIFunction.name` | Data access |

**Infrastructure (Private)** - Operational/security data:

| Data Type | Why Private |
|-----------|-------------|
| Hostnames, IPs, NFS mounts | Network reconnaissance risk |
| OS/kernel versions | CVE matching risk |
| System tool availability | Reconnaissance risk |
| User home directories | Privacy |
| Credentials, tokens | Security |

### Agents MCP Server Tools

The agents server (`imas-codex serve agents`) provides tools for exploration:

| Tool | Purpose |
|------|---------|
| `cypher(query)` | Execute Cypher (read-only) |
| `ingest_nodes(type, data)` | Schema-validated batch node creation (always use list) |
| `private(facility, data?)` | Read or update sensitive infrastructure data |
| `get_graph_schema()` | Get complete schema for Cypher generation |
| `get_facility_info(facility)` | Comprehensive facility + graph state |
| `search_code_examples(query)` | Semantic code search |
| `get_exploration_progress(facility)` | Progress metrics and recommendations |

> **Before writing Cypher queries**, call `get_graph_schema()` to get node labels, properties, enums, and relationship types.

> **IMPORTANT**: Always use MCP tools (`private`, `ingest_nodes`, `cypher`) to persist discoveries. Never directly edit YAML files.

**Usage Examples:**

```python
# Read private infrastructure data
private("epfl")

# Update private data (returns merged result)
private("epfl", {"tools": {"rg": "14.1.1", "fd": "10.2.0"}})

# Queue source files for ingestion (auto-deduplicates)
ingest_nodes("SourceFile", [
    {"id": "epfl:/home/codes/liuqe.py", "path": "/home/codes/liuqe.py",
     "facility_id": "epfl", "status": "queued", "interest_score": 0.8,
     "patterns_matched": ["equilibrium", "IMAS"]},
])
# Returns: {"processed": 1, "skipped": 0, "errors": []}
# Already-queued or ingested files are automatically skipped

# Batch persist FacilityPaths
ingest_nodes("FacilityPath", [
    {"id": "epfl:/home/codes", "path": "/home/codes", "facility_id": "epfl", "path_type": "code_directory"},
])
```
```

### Exploring Facilities

Read the exploration guide:
```bash
cat imas_codex/config/README.md
```

SSH directly using host aliases from `~/.ssh/config`:
```bash
ssh epfl "which python3; python3 --version; pip list | head"
```

### Exploration Persistence Checklist

**After every exploration session, persist ALL discoveries:**

| Discovery Type | Destination | Tool |
|----------------|-------------|------|
| Analysis codes (name, version, path) | Graph | `ingest_nodes("AnalysisCode", [...])` |
| Diagnostics, MDSplus trees, TDI functions | Graph | `ingest_nodes(...)` |
| Rich directory paths (e.g., `/home/codes`) | Graph | `ingest_nodes("FacilityPath", [...])` |
| Source files for ingestion | Graph | `ingest_nodes("SourceFile", [...])` |
| OS/kernel versions, tool availability | Infrastructure | `private(facility, {...})` |
| SVN/Git repo URLs discovered | Infrastructure | `private(facility, {...})` |
| Unstructured findings to review later | Graph staging | `cypher("CREATE (:_Discovery {...})")` |

**Key principle**: If you `find` or `ls` a useful path during exploration, persist it immediately.

### FacilityPath Multi-Pass Workflow

Track exploration progress using `FacilityPath` nodes with status progression:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  SCOUT PASS │   │ TRIAGE PASS │   │ INGEST PASS │
│             │   │             │   │             │
│ discovered  │   │ scanned     │   │ flagged     │
│     ↓       │   │     ↓       │   │     ↓       │
│  listed     │   │  flagged    │   │ analyzed    │
│     ↓       │   │    or       │   │     ↓       │
│  scanned    │   │  skipped    │   │ ingested    │
└─────────────┘   └─────────────┘   └─────────────┘
```

| Status | Pass | Meaning |
|--------|------|---------|
| `discovered` | Scout | Found but not examined |
| `listed` | Scout | Contents counted (optional) |
| `scanned` | Scout | Pattern search complete |
| `flagged` | Triage | Interesting, should analyze |
| `skipped` | Triage | Not interesting enough |
| `analyzed` | Ingest | Structure understood |
| `ingested` | Ingest | Code extracted to graph |
| `excluded` | Any | Permanently skip (e.g., /tmp) |
| `stale` | Any | May no longer exist |

**Scout Pass Example:**
```python
# 1. Get actionable paths (sorted by interest_score)
result = get_facility("epfl")
for path in result["actionable_paths"]:
    print(f"{path['status']}: {path['path']} (score: {path['interest_score']})")

# 2. After pattern search, batch update status
ingest_nodes("FacilityPath", [
    {
        "id": "epfl:/home/codes/transport",
        "status": "scanned",
        "patterns_found": ["equilibrium", "IMAS", "write_ids"],
        "last_examined": "2025-01-15T10:30:00Z"
    }
])
```

**Triage Pass Example:**
```python
# Batch update paths after triage
ingest_nodes("FacilityPath", [
    {
        "id": "epfl:/home/codes/transport",
        "status": "flagged",
        "interest_score": 0.9,
        "notes": "Found IMAS integration, multiple IDS writes"
    },
    {
        "id": "epfl:/home/codes/old_backup",
        "status": "skipped",
        "notes": "Stale backup, no active development"
    }
])
```

**Interest Score Guidelines:**

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| 0.3+ | Utilities, helpers |
| <0.3 | Config files, documentation |

### Fast Analysis Tools

Pre-installed Rust tools at `~/bin/` for fast directory analysis:

| Tool | Purpose | Speed | Output |
|------|---------|-------|--------|
| `dust` | Disk usage tree | ~10x faster than `du` | Visual tree |
| `tokei` | Lines of code by language | Parallel, language-aware | JSON/table |
| `scc` | SLOC + complexity metrics | Very fast | JSON with complexity |
| `fd` | Find files | ~5x faster than `find` | File paths |
| `rg` | Search content | ~10x faster than `grep` | Matching lines |

**Quick directory assessment:**
```bash
# Disk usage tree (depth 2)
ssh epfl "~/bin/dust -d 2 /home/codes"

# Code statistics with complexity (JSON for parsing)
ssh epfl "~/bin/scc /home/codes/liuqe --format json"

# Lines of code by language
ssh epfl "~/bin/tokei /home/codes/liuqe"

# Count Python files quickly
ssh epfl "~/bin/fd -e py /home/codes | wc -l"
```

**Prioritizing paths with scc:**
```bash
# Get complexity score to prioritize paths
ssh epfl '~/bin/scc /home/codes/liuqe --format json' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); \
  print(sum(l['Code'] for l in d), 'lines,', sum(l['Complexity'] for l in d), 'complexity')"
# Output: 4858 lines, 487 complexity
```

### Coverage Tracking (Optional)

The schema includes optional coverage fields. Use them judiciously:

| Field | When to Use | Performance Risk |
|-------|-------------|------------------|
| `file_count` | Small dirs (<1000 files) | Low with fd |
| `dir_count` | Small dirs | Low with fd |
| `files_scanned` | After pattern search | Safe (rg limits) |
| `files_ingested` | After ingestion | Safe (we control) |

**Safe counting with fast tools:**
```bash
# fd is fast even on large directories
ssh epfl "~/bin/fd -t f /home/codes/transport | wc -l"

# With timeout as failsafe for unknown dirs
ssh epfl "timeout 10s ~/bin/fd -t f /home | wc -l"

# Or limit depth
ssh epfl "~/bin/fd -t f --max-depth 3 /home/codes | wc -l"
```

**Skip counting for known massive paths:**
```python
# Known large dirs: skip counting, use pattern search instead
SKIP_COUNTING = ["/home", "/usr/local"]

paths_to_update = [
    {
        "id": f"epfl:{path}",
        "status": "scanned",
        "notes": "Large directory - skipped file counting"
    }
    for path in SKIP_COUNTING
]
ingest_nodes("FacilityPath", paths_to_update)
```

**Performance-safe pattern search:**
```bash
# Use rg with limits to prevent hangs
ssh epfl "~/bin/rg -l --max-count 1 --max-depth 4 'equilibrium|IMAS' /path -g '*.py' 2>/dev/null | head -50"

# With timeout as failsafe
ssh epfl "timeout 30s ~/bin/rg -l 'pattern' /path --max-depth 3"
```

### Code Ingestion Workflow

The ingestion pipeline uses a **graph-driven approach** with `SourceFile` nodes as the unit of work. Scouts queue files; the CLI processes them.

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SCOUT (LLM)    │     │   GRAPH (Neo4j) │     │    CLI (User)   │
│                 │     │                 │     │                 │
│  ssh + rg/fd    │────▶│  SourceFile     │────▶│  imas-codex     │
│  queue_source_  │     │  status=queued  │     │  ingest run     │
│  files()        │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**SourceFile Lifecycle:**
```
queued ──▶ fetching ──▶ embedding ──▶ ready
  │                                    │
  │         ┌── failed ◀───────────────┘
  ▼         ▼
stale ◀── (re-scan)
```

**Key Features:**
- **Graph-driven**: Scouts queue files, CLI processes the queue
- **Deduplication**: Already-ingested files are automatically skipped
- **Interrupt-safe**: Partial ingestion can be resumed safely
- **Status tracking**: `SourceFile` nodes track progress through lifecycle
- **MDSplus linking**: Extracted paths are linked to `TreeNode` entities

**Scout Workflow (CLI - Preferred):**

```bash
# 1. Queue files directly (LLM-friendly)
imas-codex ingest queue epfl /path/a.py /path/b.py /path/c.py

# 2. Or pipe from SSH search for large batches
ssh epfl 'rg -l "equilibrium|IMAS" /home/codes -g "*.py"' | imas-codex ingest queue epfl --stdin

# 3. Or use a file list
ssh epfl 'rg -l "pattern" /path' > files.txt
imas-codex ingest queue epfl -f files.txt
```

The pipeline automatically extracts MDSplus paths, TDI calls, and IDS references.
Do NOT fabricate `patterns_matched` metadata - let the pipeline do real extraction.

**CLI Ingestion:**

```bash
# Check queue status
imas-codex ingest status epfl

# Process queued files with progress bar
imas-codex ingest run epfl

# Process only high-priority files
imas-codex ingest run epfl --min-score 0.7

# Process more files
imas-codex ingest run epfl -n 500

# Preview what would be processed
imas-codex ingest run epfl --dry-run

# List queued files
imas-codex ingest list epfl

# List failed files
imas-codex ingest list epfl -s failed
```

**Recovery from Interrupts:**

If ingestion is interrupted:
- `SourceFile` nodes retain their status
- Rerun `imas-codex ingest run epfl` to continue
- Already-ready files are skipped automatically

**Force Re-ingestion:**

To re-ingest files (e.g., after code changes):
```bash
imas-codex ingest run epfl --force
```

### Available Facilities

```bash
uv run imas-codex facilities list
uv run imas-codex facilities show epfl
```

## Version Control

- **Default branch**: `main`
- **GitHub CLI**: `gh` available in PATH
- **Authentication**: SSH

### Container Image Tags

| Registry | Purpose | Tags |
|----------|---------|------|
| ACR | Test server | `latest-{transport}`, `X.Y.Z-rcN-{transport}` |
| ACR | Production | `prod-{transport}`, `X.Y.Z-{transport}` |
| GHCR | Public releases | `X.Y.Z-{transport}`, `X.Y-{transport}` |

**RC Workflow:**

```bash
# Create RC
git tag v1.2.0-rc1 && git push origin v1.2.0-rc1

# Iterate: v1.2.0-rc1 → v1.2.0-rc2 → v1.2.0-rc3

# Release
git tag v1.2.0 && git push origin v1.2.0
```

## Code Style

### Type Annotations

```python
# Correct
def process(items: list[str]) -> dict[str, int]: ...
if isinstance(e, ValueError | TypeError): ...

# Wrong
def process(items: List[str]) -> Dict[str, int]: ...
if isinstance(e, (ValueError, TypeError)): ...
```

### Error Handling

```python
# Correct - chain exceptions
try:
    operation()
except IOError as e:
    raise ProcessingError("failed to process") from e

# Wrong - loses context
except IOError:
    raise ProcessingError("failed to process")
```

## Philosophy

This is a **green field project**:
- No backward compatibility constraints
- Write code as if it's always been this way
- Avoid legacy naming patterns
