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

## Project Overview

`imas-codex` is an MCP server for the IMAS Data Dictionary enabling:
- Semantic search across fusion physics data structures
- IDS (Interface Data Structures) exploration
- Path validation and relationship discovery
- Physics domain exploration

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
```

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
├── core/           # Data models, XML parsing, physics domains
├── embeddings/     # Vector embeddings and semantic search
├── models/         # Pydantic models
├── search/         # Search functionality
├── services/       # External services
├── tools/          # MCP tool implementations
└── resources/      # Generated data files

tests/              # Mirror source structure
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
