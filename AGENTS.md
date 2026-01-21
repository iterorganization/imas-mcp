# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), MCP `python()` for chained processing and graph queries, `uv run` for git/tests/CLI. Conventional commits. See [agents/](agents/) for domain-specific workflows.

## Custom Agents

Select from the VS Code agent dropdown:

| Agent | Purpose | Tools |
|-------|---------|-------|
| Explore | Remote facility discovery | Read-only + MCP |
| Develop | Code development | Standard + MCP |
| Ingest | Code ingestion pipeline | Core + MCP |
| Graph | Knowledge graph operations | Core + MCP |

### Subagent Delegation

When the main VS Code agent receives requests matching these patterns, delegate using `runSubagent`:

| User Request Pattern | Delegate To | Example |
|---------------------|-------------|---------|
| Graph queries, Neo4j, "search graph for..." | Graph | "Find all CHEASE files in graph" |
| Remote facility, SSH, explore paths | Explore | "Explore EPFL for equilibrium codes" |
| Ingest files, queue code, run ingestion | Ingest | "Queue these Python files" |
| Code changes, tests, commits, refactoring | Develop | "Add tests for this module" |

## Token Optimization

VS Code injects ~30k tokens of tool definitions per request. To reduce costs:

1. **Use terminal for direct operations** - `rg`, `fd`, `git` instead of VS Code search tools or MCP wrappers
2. **Use MCP `python()` only for chained processing** - avoid for single commands
3. **Close unused editor tabs** - open files add to context
4. **Use custom agents** - restricted toolsets = fewer tool definitions
5. **Project properties in Cypher** - never `RETURN n`, always `RETURN n.id, n.name`

## Critical Rules

### Determine Local vs Remote Execution

**Always check locality first** before choosing execution method:

```bash
# Quick check - are you on the target facility?
hostname
pwd
```

```python
# Programmatic check
python("import socket; print(f'Current host: {socket.gethostname()}')")
python("info = get_facility('facility_name'); print(f'Is local: {info.get(\"is_local\", False)}')")
```

**Command Execution Decision Tree:**

1. **Single command on local facility?** → Use terminal directly
2. **Single command on remote facility?** → Use `ssh facility "command"`
3. **Chained processing with logic?** → Use `python()` with `run()` (auto-detects local/remote)
4. **Graph queries or MCP functions?** → Use `python()` with `query()`, `add_to_graph()`, etc.

### MCP Tool Selection: When to Use What

**Use dedicated MCP tools** (preferred for single operations):
- `update_facility_infrastructure()` - Update private facility data
- `get_facility_infrastructure()` - Read private facility data
- `add_exploration_note()` - Add timestamped exploration note
- `update_facility_paths()` - Update path mappings
- `update_facility_tools()` - Update tool availability
- `add_to_graph()` - Create graph nodes
- `get_graph_schema()` - Get schema for queries

**Use MCP `python()` only when you need**:
- **Chained operations** with intermediate processing and logic
- **Graph queries** with Cypher (use `query()` function)
- **REPL state** to avoid import overhead across multiple operations
- **IMAS/COCOS domain operations** (search_imas, validate_cocos, etc.)
- **Complex data transformations** that require Python execution

**Use terminal directly** for single operations:
- Local facility: `rg`, `fd`, `git`, `dust`, `uv run`
- Remote facility: `ssh facility "command"`

**Never use `python()` for**:
- Text formatting or markdown generation (LLM can do this natively)
- Single infrastructure updates (use dedicated MCP tools)
- Simple string operations that don't require execution

**Batch operations in single calls** to reduce tool invocations:
```python
# Good - multiple operations in one call
python("""
facility = get_facility('epfl')
paths = facility['actionable_paths'][:5]
for p in paths:
    print(f"Path: {p['path']}, Score: {p['interest_score']}")
""")

# Avoid - multiple separate calls
python("facility = get_facility('epfl')")
python("paths = facility['actionable_paths'][:5]")
python("for p in paths: print(p)")
```

**Local facility** (you're running on the target system):
```bash
# Correct - direct terminal
rg -l "IMAS" /work/imas
fd -e py /home/codes
dust -d 2 /work

# Wrong - unnecessary python wrapper
python("print(run('rg pattern /work/imas', facility='current'))")
```

**Remote facility** (accessing a different system):
```bash
# Correct - direct SSH
ssh epfl "rg -l 'IMAS' /home/codes"
ssh epfl "fd -e py /home/codes | head -20"

# Wrong - python wrapper for single command
python("print(run('rg -l IMAS /home/codes', facility='epfl'))")
```

### Discovering the Python REPL API

The `python()` tool provides a persistent REPL with pre-loaded utilities. **Always discover the API first** before using it:

```python
# Discover available tools via introspection
python("print([name for name in dir() if not name.startswith('_')])")

# Get function help and signatures
python("help(search_imas)")  # Full docstring
python("import inspect; print(inspect.signature(get_facility))")  # Signature only
python("print(search_imas.__doc__)")  # Just the docstring
```

**Why discover instead of relying on pre-loaded knowledge?**
1. Functions may be added/removed between versions
2. Signatures may change
3. New utilities may be available
4. Ensures you're using the current API, not outdated assumptions

**Common REPL Functions** (verify with `dir()` first):

```python
# Discover available tools via introspection
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")  # Get function docstring
python("import inspect; print(inspect.signature(get_facility))")  # Get signature

# Check locality
python("import socket; print(f'Current host: {socket.gethostname()}')")
python("info = get_facility('epfl'); print(f'Is local: {info.get(\"is_local\", False)}')")

# Graph queries
python("result = query('MATCH (f:Facility) RETURN f.id'); print(result)")

# Chained processing (run() auto-detects local vs SSH)
python("""
files = run('fd -e py /home/codes', facility='epfl').strip().split('\\n')
for f in files[:10]:
    content = run(f'head -20 {f}', facility='epfl')
    if 'write_ids' in content:
        print(f'IDS writer: {f}')
""")

# IMAS search
python("print(search_imas('electron temperature'))")

# Code search
python("print(search_code('equilibrium reconstruction'))")
```

**After editing `imas_codex/` source files**, reload the REPL to pick up changes:

```python
python("print(reload())")  # Clears module cache and reinitializes
```

**Use `uv run` for:** git operations, ruff linting/formatting, pytest, and package management.

**Never use `python()` for:**
- Text formatting or markdown generation (LLM can do this natively)
- Single infrastructure updates (use dedicated MCP tools)
- Simple string operations that don't require execution

### Fast Tools (Prefer Over Standard Unix)

Fast Rust-based CLI tools are defined in [`imas_codex/config/fast_tools.yaml`](imas_codex/config/fast_tools.yaml).

**ALWAYS use fast tools instead of standard Unix equivalents:**

| Fast Tool | Instead Of | Speed | Example |
|-----------|------------|-------|---------|
| `rg` | `grep -r` | 10x faster | `rg 'IMAS' /work/projects -g '*.py'` |
| `fd` | `find` | 5x faster | `fd -e py /work/projects` |
| `tokei` | `wc -l` | Better | `tokei /path` |
| `dust` | `du -h` | Visual | `dust -d 2 /work` |

**Critical: fd requires path as trailing argument:**
```bash
# CORRECT - path is required, especially on remote/large filesystems
fd -e py /work/projects        # Find .py files in /work/projects
fd 'pattern' /path             # Find pattern in /path

# WRONG - will hang or search cwd unexpectedly
fd -e py                       # Missing path!
```

**Terminal usage** (preferred for all single operations):

```bash
# Check locality first
hostname
pwd

# Local facility - direct commands
rg -l "IMAS" /path/to/search
fd -e py /path/to/search
git log --oneline -10
dust -d 2 /work

# Remote facility - direct SSH
ssh epfl "rg -l 'IMAS' /home/codes"
ssh epfl "fd -e py /home/codes | head -20"
ssh epfl "dust -d 2 /home/codes"

# Tool management
uv run imas-codex tools check              # Check local tools
uv run imas-codex tools check epfl         # Check on EPFL (via SSH)
uv run imas-codex tools install epfl       # Install on EPFL
uv run imas-codex tools list               # List all tools
```

**Python API** (for chained processing and graph operations only):

```python
# Python code that requires execution
python("from pathlib import Path; print([p.stem for p in Path('/path').glob('*.py')])")

# Chained processing - search, filter, analyze (run() auto-detects local/remote)
python("""
result = run('rg -l "IMAS" /home/codes', facility='epfl')
files = result.strip().split('\\n')
for f in files[:5]:
    content = run(f'cat {f}', facility='epfl')
    if 'write_ids' in content:
        print(f'Found IDS writer: {f}')
""")

# Tool setup with result processing
python("result = setup_tools('epfl'); print(result.summary)")

# Note: run() automatically uses SSH for remote facilities, direct execution for local
```

### Pre-commit Hooks

Pre-commit hooks use `.venv/bin/python3` and will fail if the venv is not accessible. Always use:

```bash
uv run git commit -m "type: description"
```

### Environment Variables

The `.env` file contains secrets. Never expose or commit it.

### Graph Backup

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

## Commit Workflow

Follow this exact sequence:

```bash
# 1. Lint and format FIRST (before staging)
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (NEVER git add -A)
git add <file1> <file2> ...

# 3. Commit with conventional format (uv run ensures pre-commit hooks work)
# Use multi-line format for substantial changes:
uv run git commit -m "type: concise summary (50 chars)

Detailed explanation of what changed and why. Use bullet points
for multiple changes:
- First significant change with context
- Second change explaining the rationale
- Third change noting any important details

Include relevant technical details, design decisions, or
breaking changes. This body helps reviewers and future
maintainers understand the commit.

BREAKING CHANGE: description (if applicable)"

# For simple changes, single line is fine:
uv run git commit -m "type: brief description"

# 4. If pre-commit fails, fix issues and repeat steps 2-3

# 5. Push
git push origin main
```

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `docs` | Documentation |
| `test` | Test changes |
| `chore` | Maintenance |

**Commit message guidelines:**
- **Summary line**: Concise, imperative mood, ~50 chars
- **Body**: Detailed explanation for non-trivial changes (wrap at 72 chars)
- **Bullet points**: Use `-` for listing multiple changes
- **Breaking changes**: Use `BREAKING CHANGE:` footer, not `type!:` suffix
- **Context**: Explain *why* not just *what* - include design decisions

## Testing

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

## Working in Worktrees

Cursor remote agents often work in auto-created worktrees. **CRITICAL: Commits in worktrees are NOT on `main` until merged.**

### Commit and Merge Workflow

**ALWAYS complete both steps together** - never end a session after step 1:

```bash
# Step 1: Commit in the worktree
uv run ruff check --fix . && uv run ruff format .
git add <file1> <file2> ...
uv run git commit -m "type: description"

# Step 2: IMMEDIATELY merge to main (do not skip!)
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

**Why merge instead of cherry-pick?**
- Merge tracks that changes have been applied (cherry-pick doesn't)
- Merge handles conflicts better with clear ancestry
- Cherry-pick duplicates commits without tracking, leading to future conflicts

### Session-End Checklist

Before ending any worktree session, verify:
- [ ] All commits have been merged to main
- [ ] Changes are pushed to origin (`git log origin/main..main` should be empty)

To check for un-merged worktree commits:
```bash
cd /home/mcintos/Code/imas-codex
git log --oneline --all --not main | head -20  # Shows commits not on main
```

### Worktree Lifecycle

Worktrees share the same `.git` directory, so commits exist but are on detached/feature branches. Git manages worktrees automatically - no manual cleanup required.

```bash
# List all worktrees
git worktree list

# Prune stale worktree metadata (safe, run periodically)
git worktree prune
```

**NEVER run `git checkout -- .` in worktrees** - this discards all unstaged changes permanently with no recovery possible.

## Quick Reference

| Task | Command |
|------|---------|
| Check locality | `hostname` or `python("import socket; print(socket.gethostname())")` |
| Local single cmd | `rg pattern /path` (direct terminal) |
| Remote single cmd | `ssh facility "rg pattern /path"` (direct SSH) |
| Graph query | `python("print(query('MATCH (n) RETURN n.id, n.name LIMIT 5'))")` |
| Run command | `python("print(run('rg pattern', facility='epfl'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(search_code('equilibrium'))")` |
| Facility info | `python("print(get_facility('epfl'))")` |
| Check tools | `python("print(check_tools('epfl'))")` |
| Setup tools | `python("result = setup_tools('epfl'); print(result.summary)")` |
| Add to graph | `add_to_graph('SourceFile', [...])` (MCP tool) |
| Update infrastructure | `update_facility_infrastructure('epfl', {...})` (MCP tool) |
| Get infrastructure | `get_facility_infrastructure('epfl')` (MCP tool) |
| Add exploration note | `add_exploration_note('epfl', 'Found IMAS at /work')` (MCP tool) |
| Update paths | `update_facility_paths('epfl', {'imas': {...}})` (MCP tool) |
| Update tools | `update_facility_tools('epfl', {'rg': {...}})` (MCP tool) |

**MCP Tools vs python():**
- Use MCP tools for single-purpose operations (better discoverability, type safety)
- Use `python()` for chained processing, graph queries, IMAS/COCOS operations

Never `RETURN n` - always project properties (`n.id, n.name`). Embeddings add ~2k tokens/node. See [agents/graph.md](agents/graph.md#token-cost-optimization).


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

### General Rules

- Stage files individually, never `git add -A`
- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations

### DO

- **Check locality first** (`hostname` or check facility info) before choosing execution method
- Use terminal directly for **single operations on local facility** (`rg`, `fd`, `git`)
- Use direct SSH for **single operations on remote facility** (`ssh facility "command"`)
- Use MCP `python()` only for **chained processing** and **graph queries**
- Let `run()` auto-detect local/remote when doing chained processing
- Batch multiple operations in single `python()` calls to reduce tool invocations
- Use `uv run` for git operations, ruff, pytest, and package management
- Use `python("print(reload())")` after editing `imas_codex/` source files to pick up changes
- Use modern Python 3.12 syntax: `list[str]`, `X | Y`
- Use exception chaining with `from`

### DON'T

- Don't assume you're always on a specific facility - **check locality first**
- Don't use `python()` wrapper for **single commands** (local or remote)
- Don't manually activate venv - use `uv run` which handles venv automatically
- Don't use `python()` for text formatting - LLM can do this natively
- Don't make multiple `python()` calls when one batched call would work
- Don't prefix terminal commands with `cd /path &&` - terminal session is persistent
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

## Markdown Style

- Use headings for hierarchy, bullets for lists, code fences for code
- Keep paragraphs short; one idea per section
- Avoid excessive bold, emojis, and deep nesting

## Project Structure

```
agents/                  # Portable agent instructions
.github/agents/          # VS Code agent shims
imas_codex/
├── agentic/             # LlamaIndex agents, MCP server
├── graph/               # Neo4j knowledge graph
├── code_examples/       # Code ingestion pipeline
├── schemas/             # LinkML schemas (authoritative graph structure)
│   ├── imas_dd.yaml     # DD graph: IMASPath, DDVersion, Unit, CoordinateSpec
│   ├── facility.yaml    # Facility graph: SourceFile, TreeNode, CodeChunk
│   └── common.yaml      # Shared: Unit, PhysicsDomain enums
└── ...
```

**LinkML Schemas**: All graph node types, relationships, and properties are defined in
`imas_codex/schemas/`. Before adding properties to graph nodes, check the schema first.
Key classes: `IMASPath` (DD paths), `Unit` (physical units), `CoordinateSpec` (index specs),
`SourceFile`/`TreeNode`/`CodeChunk` (code ingestion).

## Domain Workflows

| Workflow | Agent | Documentation |
|----------|-------|---------------|
| Facility exploration | Explore | [agents/explore.md](agents/explore.md) |
| Development | Develop | [agents/develop.md](agents/develop.md) |
| Code ingestion | Ingest | [agents/ingest.md](agents/ingest.md) |
| Graph operations | Graph | [agents/graph.md](agents/graph.md) |

## Fallback: When MCP Server is Not Running

If `python()` tool is unavailable, use `uv run` commands:

```bash
# Graph operations
uv run imas-codex neo4j status
uv run imas-codex neo4j shell
uv run imas-codex neo4j dump

# Ingestion
uv run imas-codex ingest queue epfl /path/*.py
uv run imas-codex ingest run epfl
uv run imas-codex ingest status epfl

# Testing and linting
uv run pytest
uv run ruff check --fix . && uv run ruff format .

# Dependencies
uv sync --extra test
```
