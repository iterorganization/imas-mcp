# Development Workflows

Code development with testing, linting, and git workflow guardrails.

## Commit Workflow

**Step-by-step procedure:**

```bash
# 1. Check current state
git status

# 2. Run linting on Python files
uv run ruff check --fix .
uv run ruff format .

# 3. Stage specific files (NEVER use git add -A)
git add <file1> <file2> ...

# 4. Commit with conventional format
uv run git commit -m 'type: brief description

Detailed body explaining what changed and why.'

# 5. Fix pre-commit errors and repeat steps 3-4 until clean

# 6. Push
git push origin main
```

## Commit Message Format

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `docs` | Documentation |
| `test` | Test changes |
| `chore` | Maintenance |

**Breaking changes**: Add `BREAKING CHANGE:` footer in the body (not `type!:` suffix).

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

# Fast iteration: use restricted IDS filters
uv run build-clusters --ids-filter "core_profiles equilibrium" -v -f
```

## Working in Worktrees

Cursor remote agents often work in auto-created worktrees:

**Step 1: Commit in the worktree**
```bash
cd /path/to/worktree
uv run ruff check --fix . && uv run ruff format .
git add <file1> <file2> ...
uv run git commit -m 'type: description'
```

**Step 2: Cherry-pick to main workspace**
```bash
cd /home/ITER/mcintos/Code/imas-codex
git cherry-pick <commit-hash-from-worktree>
git push origin main
```

**Step 3: Clean up worktree**
```bash
cd /path/to/worktree
git checkout -- .  # Discard remaining changes
```

## Pre-commit Hooks

The pre-commit hook uses `.venv/bin/python3`. In worktrees:

```bash
# Option 1: Use uv run (recommended)
uv run git commit -m 'type: description'

# Option 2: Activate venv first
source .venv/bin/activate
git commit -m 'type: description'
```

## DO

- Use `uv run` for all Python commands
- Use single quotes for commit messages
- Stage files individually (`git add <file>`)
- Use modern Python 3.12 syntax: `list[str]`, `X | Y`
- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations
- Use exception chaining with `from`

## DON'T

- Don't prefix commands with `cd /path &&`
- Don't manually activate venv
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use double quotes with special characters in commits
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

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
