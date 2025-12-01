# Agent Guidelines

## Introduction

This project, `imas-mcp`, is a Model Context Protocol (MCP) server for the IMAS Data Dictionary. It enables AI agents to explore fusion physics data structures, search for relevant IDS (Interface Data Structures) entries, understand relationships between data paths, and access comprehensive physics documentation.

The server provides semantic search across the IMAS data dictionary, physics domain exploration, path validation, and relationship discovery—all accessible through the MCP protocol.

## Project Setup

### Terminal Usage

**Working Directory**: All terminal commands assume you're in the project root (`/home/ITER/mcintos/Code/imas-mcp`). Do not prefix commands with `cd /path &&`.

```bash
# Correct - use uv run for Python commands
uv run pytest

# Wrong - unnecessary cd and manual venv activation
cd /home/ITER/mcintos/Code/imas-mcp && source .venv/bin/activate && pytest
```

### Package Management

- **Package manager**: `uv`
- **Add dependencies**: `uv add <package>`
- **Add dev dependencies**: `uv add --dev <package>`
- **Sync environment**: `uv sync`

### Code Quality

- **Pre-commit hooks**: Enabled for all commits
- **Linting & formatting**: `ruff` (configuration in `pyproject.toml`)

### Version Control

- **Branch naming**: Use `main` as default branch
- **GitHub CLI**: `gh` is available in PATH
- **Authentication**: SSH
- **Commit messages**: Use conventional commit format

```bash
git status                      # Check current state
git add -A                      # Stage all changes
git commit -m "message"         # Commit (triggers pre-commit)
git push origin main            # Push to remote
```

### Testing

- **Framework**: `pytest`
- **Run tests**: Use VS Code Test Explorer (Ctrl+Shift+T) or `pytest`
- **Async support**: `pytest-asyncio` with auto mode

## Project Structure

```
imas_mcp/
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
if isinstance(e, DocsServerError | PortAllocationError):
    raise

# Wrong (ruff UP038)
if isinstance(e, (DocsServerError, PortAllocationError)):
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
