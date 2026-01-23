# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), MCP `python()` for chained processing and graph queries, `uv run` for git/tests/CLI. Conventional commits. See [agents/](agents/) for domain-specific deep-dives.

## Project Philosophy

This is a greenfield project under active development. We move fast and do not maintain backwards compatibility.

**Key principles:**
- No backwards compatibility requirements - breaking changes are expected
- Strategic pivots are normal - remove deprecated code decisively
- Avoid "enhanced", "new", "refactored" in names - just use the good name
- When patterns change, update all usages - don't leave old patterns alongside new ones
- Prefer explicit over clever - future agents will read this code

## Schema System

All graph node types, relationships, and properties are defined in LinkML schemas. These schemas are the single source of truth.

**Schema files:**
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, TreeNode, CodeChunk, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASPath, DDVersion, Unit, CoordinateSpec
- `imas_codex/schemas/common.yaml` - Shared: status enums, PhysicsDomain

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- Output: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`

**Enforcement rules:**

Always import enums and classes from generated models. Never hardcode status values.

```python
# Correct - import from generated models
from imas_codex.graph.models import SourceFile, SourceFileStatus, TreeNode

sf = SourceFile(
    id="epfl:/home/codes/liuqe.py",
    facility_id="epfl",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])

# Wrong - hardcoded string bypasses validation
data = {"id": "...", "status": "discovered"}  # No type checking!
```

**Extending schemas:**
1. Edit LinkML YAML in `imas_codex/schemas/`
2. Run `uv run build-models --force`
3. Import new classes from `imas_codex.graph.models`

Schema changes are additive only in the graph. Add properties, never rename or remove.

## Critical Rules

### Locality Check

Always determine if you're on the target facility before choosing execution method:

```bash
hostname
pwd
```

**Command Execution Decision Tree:**
1. Single command on local facility? → Terminal directly (`rg`, `fd`, `dust`)
2. Single command on remote facility? → Direct SSH (`ssh facility "command"`)
3. Chained processing with logic? → `python()` with `run()` (auto-detects local/remote)
4. Graph queries or MCP functions? → `python()` with `query()`, `add_to_graph()`, etc.

### MCP Tool Selection

**Use dedicated MCP tools** for single operations:
- `add_to_graph()` - Create graph nodes with schema validation
- `get_graph_schema()` - Get schema for Cypher generation
- `update_facility_infrastructure()` - Update private facility data
- `get_facility_infrastructure()` - Read private facility data
- `add_exploration_note()` - Add timestamped exploration note

**Use `python()` REPL** when you need:
- Chained operations with intermediate processing
- Graph queries with Cypher (`query()` function)
- IMAS/COCOS domain operations (`search_imas`, `validate_cocos`)
- Complex data transformations requiring Python execution

**Use terminal directly** for single operations:
- Local: `rg`, `fd`, `git`, `dust`, `uv run`
- Remote: `ssh facility "command"`

**Never use `python()` for:**
- Text formatting (LLM can do this natively)
- Single infrastructure updates (use dedicated MCP tools)
- Simple string operations

### Graph Backup

Before ANY operation that modifies or deletes graph nodes:

1. Always dump first: `uv run imas-codex neo4j dump`
2. Never use `DETACH DELETE` on production data without user confirmation
3. For re-embedding: update nodes in place, don't delete and recreate

### Token Cost Optimization

Graph queries can be expensive. Follow these rules:

**Never return full nodes** - always project specific properties:

```python
# Bad - embeddings cost ~2k tokens per node
query("MATCH (n:IMASPath) RETURN n LIMIT 10")

# Good - project only needed properties (~50 tokens per node)
query("MATCH (n:IMASPath) RETURN n.id, n.name, n.documentation LIMIT 10")
```

**Use Cypher aggregations** instead of Python post-processing:

```python
# Bad - multiple calls, Python aggregation
files = query("MATCH (f:SourceFile) RETURN f.status")
# Then Counter() in another call

# Good - single call with Cypher aggregation
query("""
    MATCH (f:SourceFile)
    RETURN f.status AS status, count(*) AS count
    ORDER BY count DESC
""")
```

## Fast Tools

Rust-based CLI tools defined in `imas_codex/config/fast_tools.yaml`. Always prefer these:

| Fast Tool | Instead Of | Example |
|-----------|------------|---------|
| `rg` | `grep -r` | `rg 'IMAS' /work -g '*.py'` |
| `fd` | `find` | `fd -e py /work/projects` |
| `tokei` | `wc -l` | `tokei /path` |
| `dust` | `du -h` | `dust -d 2 /work` |

Critical: `fd` requires path as trailing argument. `fd -e py` without path will hang.

```bash
# Local facility
rg -l "IMAS" /path/to/search
fd -e py /path/to/search

# Remote facility
ssh epfl "rg -l 'IMAS' /home/codes"
ssh epfl "fd -e py /home/codes | head -20"
```

## Commit Workflow

```bash
# 1. Lint and format first
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (never git add -A)
git add <file1> <file2> ...

# 3. Commit with conventional format
uv run git commit -m "type: concise summary

Detailed explanation of what changed and why.
- First significant change
- Second change with rationale

BREAKING CHANGE: description (if applicable)"

# 4. If pre-commit fails, fix and repeat steps 2-3

# 5. Push
git push origin main
```

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring |
| docs | Documentation |
| test | Test changes |
| chore | Maintenance |

Breaking changes use `BREAKING CHANGE:` footer, not `type!:` suffix.

## Working in Worktrees

Cursor remote agents often work in auto-created worktrees. Commits in worktrees are NOT on `main` until merged.

**Always complete both steps together:**

```bash
# Step 1: Commit in worktree
uv run ruff check --fix . && uv run ruff format .
git add <file1> <file2> ...
uv run git commit -m "type: description"

# Step 2: Immediately merge to main
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

Never end a session after step 1. Check for un-merged commits:
```bash
git log --oneline --all --not main | head -20
```

## Code Style

### Python Version

This project requires **Python 3.12** (`requires-python = ">=3.12,<3.13"`). Do not write fallback code for older Python versions. Remove any legacy compatibility patterns you encounter.

```python
# Wrong - unnecessary fallback for older Python
try:
    script_path = importlib.resources.files("package").joinpath("file")
except (AttributeError, TypeError):
    import pkg_resources  # Python 3.8 fallback - DELETE THIS
    ...

# Correct - use Python 3.12 features directly
script_path = importlib.resources.files("package").joinpath("file")
```

### Type Annotations

```python
# Correct - Python 3.12 syntax
def process(items: list[str]) -> dict[str, int]: ...
if isinstance(e, ValueError | TypeError): ...

# Wrong - legacy typing module
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

### General Rules

- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations
- Stage files individually, never `git add -A`
- Use `uv run` for all Python commands (handles venv automatically)

### DO

- Check locality first (`hostname`) before choosing execution method
- Use terminal directly for single operations on local facility
- Use direct SSH for single operations on remote facility
- Use MCP `python()` only for chained processing and graph queries
- Batch multiple operations in single `python()` calls
- Use `python("print(reload())")` after editing source files

### DON'T

- Don't use `python()` wrapper for single commands (local or remote)
- Don't manually activate venv - use `uv run`
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

## Testing

```bash
# Sync dependencies (required in worktrees)
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=imas_codex

# Run specific test
uv run pytest tests/path/to/test.py::test_function

# Fast iteration with restricted IDS filter
uv run imas-codex clusters build --ids-filter "core_profiles equilibrium" -v -f
```

## Python REPL Discovery

The `python()` tool provides a persistent REPL with pre-loaded utilities. Discover the API first:

```python
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")
python("import inspect; print(inspect.signature(get_facility))")
```

After editing `imas_codex/` source files, reload:
```python
python("print(reload())")
```

## Quick Reference

| Task | Command |
|------|---------|
| Check locality | `hostname` |
| Local command | `rg pattern /path` |
| Remote command | `ssh facility "rg pattern /path"` |
| Graph query | `python("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(search_code('equilibrium'))")` |
| Facility info | `python("print(get_facility('epfl'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Update infrastructure | `update_facility_infrastructure('epfl', {...})` |

## Domain Workflows

For detailed domain-specific workflows, see:

| Workflow | Documentation |
|----------|---------------|
| Facility exploration | [agents/explore.md](agents/explore.md) |
| Development | [agents/develop.md](agents/develop.md) |
| Code ingestion | [agents/ingest.md](agents/ingest.md) |
| Graph operations | [agents/graph.md](agents/graph.md) |

## Custom Agents

Select from the VS Code agent dropdown for restricted toolsets:

| Agent | Purpose |
|-------|---------|
| Explore | Remote facility discovery (read-only + MCP) |
| Develop | Code development (standard + MCP) |
| Ingest | Code ingestion pipeline (core + MCP) |
| Graph | Knowledge graph operations (core + MCP) |

## Fallback: MCP Server Not Running

If `python()` is unavailable, use CLI:

```bash
# Graph operations
uv run imas-codex neo4j status
uv run imas-codex neo4j shell
uv run imas-codex neo4j dump

# Ingestion
uv run imas-codex ingest queue epfl /path/*.py
uv run imas-codex ingest run epfl

# Testing
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```
