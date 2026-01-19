# Agent Guidelines

Use MCP `python()` for exploration/queries, `uv run` for git/tests/CLI. Conventional commits with single quotes. See [agents/](agents/) for domain-specific workflows.

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

1. **Use terminal for search** - `rg`, `fd`, `git` instead of VS Code search tools
2. **Close unused editor tabs** - open files add to context
3. **Use custom agents** - restricted toolsets = fewer tool definitions
4. **Project properties in Cypher** - never `RETURN n`, always `RETURN n.id, n.name`

## Critical Rules

### MCP `python()` is Primary

When the Codex MCP server is running, prefer `python()` over terminal for exploration and queries:

```python
# Discover available tools via introspection
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")  # Get function docstring
python("import inspect; print(inspect.signature(get_facility))")  # Get signature

# Graph queries
python("result = query('MATCH (f:Facility) RETURN f.id'); print(result)")

# Remote exploration (auto-detects local vs SSH)
python("print(run('ls /home/codes', facility='epfl'))")  # SSH to EPFL
python("print(run('ls /work/imas', facility='iter'))")   # Local on SDCC

# IMAS search
python("print(search_imas('electron temperature'))")

# Persist discoveries
python("ingest_nodes('SourceFile', [{'id': 'epfl:/path', 'path': '/path', ...}])")

# Facility info and exploration targets
python("info = get_facility('epfl'); print(info['actionable_paths'][:5])")

# Code search
python("print(search_code('equilibrium reconstruction'))")
```

After editing `imas_codex/` source files, reload the REPL to pick up changes:

```python
python("print(reload())")  # Clears module cache and reinitializes
```

**Use `uv run` for:** git operations, ruff linting/formatting, pytest, and package management.

**Never use `python()` for:** formatting text output, generating markdown, or text manipulation.

### Fast Tools (Prefer Over Standard Unix)

Fast Rust-based CLI tools are defined in [`imas_codex/config/fast_tools.yaml`](imas_codex/config/fast_tools.yaml).

| Tool | Purpose | Fallback |
|------|---------|----------|
| `rg` | Fast pattern search (10x grep) | `grep -r` |
| `fd` | Fast file finder (5x find) | `find . -name` |
| `tokei` | LOC by language | `wc -l` |
| `scc` | Code complexity metrics | - |
| `dust` | Visual disk usage | `du -h` |
| `eza` | Modern ls with git status | `ls -la` |
| `bat` | Syntax-highlighted cat | `cat` |
| `delta` | Better git diff | `diff` |
| `fzf` | Fuzzy finder | - |
| `yq` | YAML processor | - |
| `jq` | JSON processor | - |

**Python API** (via `run()` which auto-detects local vs SSH):

```python
# run() auto-detects: local on SDCC, SSH to EPFL
python("print(run('rg -l \"IMAS\" /home/codes', facility='epfl'))")
python("print(run('rg pattern', facility='iter'))")  # Local on SDCC

# Check and install tools
python("print(check_tools('epfl'))")
python("result = setup_tools('epfl'); print(result.summary)")
python("print(quick_setup('iter', required_only=True))")
```

**CLI commands** (fallback when MCP server not running):

```bash
uv run imas-codex tools check              # Check local tools
uv run imas-codex tools check epfl         # Check on EPFL (via SSH)
uv run imas-codex tools install epfl       # Install on EPFL
uv run imas-codex tools install --dry-run  # Show install commands
uv run imas-codex tools list               # List all tools
```

### Pre-commit Hooks

Pre-commit hooks use `.venv/bin/python3` and will fail if the venv is not accessible. Always use:

```bash
uv run git commit -m 'type: description'
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
uv run git commit -m 'type: brief description'

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

Cursor remote agents often work in auto-created worktrees. Follow this workflow for clean commits:

**Step 1: Commit in the worktree**
```bash
cd /path/to/worktree

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Stage and commit
git add <file1> <file2> ...
uv run git commit -m 'type: description'
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

## Quick Reference

| Task | Command |
|------|---------|
| Graph query | `python("print(query('MATCH (n) RETURN n.id, n.name LIMIT 5'))")` |
| Run command | `python("print(run('rg pattern', facility='epfl'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(search_code('equilibrium'))")` |
| Facility info | `python("print(get_facility('epfl'))")` |
| Check tools | `python("print(check_tools('epfl'))")` |
| Setup tools | `python("result = setup_tools('epfl'); print(result.summary)")` |
| Ingest nodes | `python("ingest_nodes('SourceFile', [...])")` |
| Private data | `python("print(private('epfl'))")` |

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

- Use single quotes for commit messages
- Stage files individually, never `git add -A`
- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations

### DO

- Use MCP `python()` for exploration, queries, and graph operations
- Use `uv run` for git operations, ruff, pytest, and package management
- Use `python("print(reload())")` after editing `imas_codex/` source files to pick up changes
- Use modern Python 3.12 syntax: `list[str]`, `X | Y`
- Use exception chaining with `from`

### DON'T

- Don't manually activate venv - use `uv run` which handles venv automatically
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use double quotes with special characters in commits
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
└── ...
```

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
