# Agent Guidelines

## Introduction

This project, `imas-codex`, is a Model Context Protocol (MCP) server for the IMAS Data Dictionary. It enables AI agents to explore fusion physics data structures, search for relevant IDS (Interface Data Structures) entries, understand relationships between data paths, and access comprehensive physics documentation.

The server provides semantic search across the IMAS data dictionary, physics domain exploration, path validation, and relationship discovery—all accessible through the MCP protocol.

## Project Setup

### Terminal Usage

**Working Directory**: All terminal commands assume you're in the project root (`/home/ITER/mcintos/Code/imas-codex`). Do not prefix commands with `cd /path &&`.

```bash
# Correct - use uv run for Python commands
uv run pytest

# Wrong - unnecessary cd and manual venv activation
cd /home/ITER/mcintos/Code/imas-codex && source .venv/bin/activate && pytest
```

### Package Management

- **Package manager**: `uv`
- **Add dependencies**: `uv add <package>`
- **Add dev dependencies**: `uv add --dev <package>`
- **Sync environment**: `uv sync --extra test` (includes pytest-cov)

**IMPORTANT for CLI agents in worktrees**: Always run `uv sync --extra test` before running tests or any Python commands. The `[test]` extra includes `pytest-cov` which is required for the test suite.

### Code Quality

- **Pre-commit hooks**: Enabled for all commits
- **Linting & formatting**: `ruff` (configuration in `pyproject.toml`)

### Version Control

- **Branch naming**: Use `main` as default branch
- **GitHub CLI**: `gh` is available in PATH
- **Authentication**: SSH

#### Commit Messages

Use conventional commit format with a detailed body:

```bash
git status                      # Check current state
git add <files>                 # Stage specific files (avoid git add -A)
git commit -m "type: description

Detailed body explaining what changed and why.

BREAKING CHANGE: description (if applicable)"
git push origin main            # Push to remote
```

**Commit types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

**Breaking changes**: Use `BREAKING CHANGE:` footer in the body, not `type!:` suffix (the `!` causes shell escaping issues and is redundant when using the footer)

**Shell escaping**: Use single quotes for commit messages containing special characters:

```bash
# Correct - single quotes avoid bash history expansion issues
git commit -m 'refactor: description'

# Wrong - double quotes with special chars can fail
git commit -m "refactor!: description"  # ! causes bash errors
```

### Testing

- **Framework**: `pytest`
- **Run tests**: Use VS Code Test Explorer (Ctrl+Shift+T) or `uv run pytest`
- **Async support**: `pytest-asyncio` with auto mode
- **Coverage**: `uv run pytest --cov=imas_codex`

**Before running tests**: Ensure dependencies are synced with `uv sync --extra test`

## Project Structure

```
imas_codex/
├── core/           # Data models, XML parsing, physics domains
├── embeddings/     # Vector embeddings and semantic search
├── models/         # Pydantic models for all data structures
├── search/         # Search functionality
├── services/       # External services (docs server)
├── tools/          # MCP tool implementations
└── resources/      # Generated data files

tests/
├── conftest.py     # Shared fixtures
└── */              # Mirror source structure
```

## Python Style Guide

### Version & Syntax

- **Python version**: 3.12
- Use modern type syntax: `list[str]` not `List[str]`
- Use union syntax: `X | Y` not `Union[X, Y]` or `(X, Y)` in isinstance

```python
# Correct
if isinstance(e, ValueError | TypeError):
    raise

# Wrong (ruff UP038)
if isinstance(e, (ValueError, TypeError)):
    raise
```

### Data Structures

- **Schemas**: Use `pydantic` models
- **Data classes**: Use `dataclasses` for non-schema classes

### Asynchronous Programming

- **Library**: Use `anyio` for async operations
- **When to use**: All I/O-bound operations (network, file, database)

### Error Handling

- Use specific exception types
- Include context in error messages
- Use exception chaining with `from`

## Code Philosophy

### Green Field Project

- No backward compatibility constraints
- Avoid "new", "refactored", "enhanced" in names
- Write code as if it's always been this way
