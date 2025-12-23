# Agent Guidelines

> **TL;DR**: Use `uv run` for Python commands, `ruff` for linting, conventional commits with single quotes, and `pytest` for testing. No backward compatibility constraints.

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
| Create release | `uv run imas-codex release v1.0.0 -m 'message'` |

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

# Push to GHCR (requires GHCR_TOKEN and GHCR_USERNAME in .env)
uv run imas-codex neo4j push v1.0.0
```

**GHCR Authentication:**
```bash
# Add to .env (gitignored)
GHCR_TOKEN=ghp_your_token_here
GHCR_USERNAME=your_github_username
```

### Release Workflow

Use the release command to sync schema versions, graph artifacts, and git tags:

```bash
# Full release: updates schemas, dumps graph, pushes to GHCR, creates git tag
uv run imas-codex release v1.0.0 -m 'Add EPFL facility knowledge'

# Preview changes without executing
uv run imas-codex release v1.0.0 -m 'Test' --dry-run

# Schema-only changes (no graph)
uv run imas-codex release v1.0.1 -m 'Fix schema typo' --skip-graph
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

### Architecture

The system uses a **Command/Deploy** pattern:
1. **Commander** (you, the LLM in chat) reads instructions and orchestrates
2. **Direct SSH** for exploration (faster than CLI wrapper)
3. **Artifact capture** via CLI for validated persistence

### Exploring Facilities

Read the exploration guide:
```bash
cat imas_codex/config/README.md
```

SSH directly using host aliases from `~/.ssh/config`:
```bash
ssh epfl "which python3; python3 --version; pip list | head"
```

Capture findings:
```bash
uv run imas-codex epfl --capture environment << 'EOF'
python:
  version: "3.9.21"
EOF
```

### Knowledge Hierarchy

Agent instructions are organized in levels:

| Level | Location | Content |
|-------|----------|---------|
| Guide | `config/README.md` | Batch patterns, safety, capture |
| Facility | `config/facilities/*.yaml` | Paths, tools, knowledge |
| Schema | `discovery/models/*.py` | Artifact structure |

### Persisting Knowledge

When a subagent discovers facility-specific information (e.g., "rg not available"), 
the Commander should update the facility config:

```yaml
# config/facilities/epfl.yaml
knowledge:
  tools:
    - "ripgrep (rg) not available; use grep -r instead"
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
