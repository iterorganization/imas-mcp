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

When the Codex MCP server is running, prefer `python()` over terminal:

```python
# Discover available tools via introspection
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")  # Get function docstring
python("import inspect; print(inspect.signature(get_facility))")  # Get signature

# Graph queries
python("result = query('MATCH (f:Facility) RETURN f.id'); print(result)")

# Remote exploration
python("print(ssh('ls /home/codes'))")

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

Use `uv run` for git, ruff, pytest, and package management.

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

**CLI commands:**

```bash
uv run imas-codex tools check              # Check local tools
uv run imas-codex tools check epfl         # Check on EPFL (via SSH)
uv run imas-codex tools install epfl       # Install on EPFL
uv run imas-codex tools install --dry-run  # Show install commands
uv run imas-codex tools list               # List all tools
```

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

### Pre-commit Hooks

```bash
uv run git commit -m 'type: description'
```

### Environment Variables

The `.env` file contains secrets. Never expose or commit it.

### Graph Backup

Always dump before destructive operations:
```bash
uv run imas-codex neo4j dump
```

## Commit Workflow

```bash
# 1. Lint
uv run ruff check --fix . && uv run ruff format .

# 2. Stage specific files (NEVER git add -A)
git add <file1> <file2> ...

# 3. Commit with conventional format
uv run git commit -m 'type: brief description'

# 4. Push
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

- Python 3.12 syntax: `list[str]`, `X | Y`, not `List[str]`, `Union[X, Y]`
- Exception chaining: `raise Error("msg") from e`
- Single quotes for commit messages
- Stage files individually, never `git add -A`

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
