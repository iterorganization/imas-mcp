# Development Workflows

Code development with testing, linting, and git workflow guardrails.

This agent builds on the core rules in [AGENTS.md](../AGENTS.md). The main file covers commit workflow, code style, and testing. This file covers development-specific details.

## Pre-commit Hooks

The pre-commit hook uses `.venv/bin/python3`. In worktrees:

```bash
# Option 1: Use uv run (recommended)
uv run git commit -m "type: description"

# Option 2: Activate venv first
source .venv/bin/activate
git commit -m "type: description"
```

## Worktree Session Checklist

Before ending any worktree session, verify:
- All commits have been merged to main
- Changes are pushed to origin (`git log origin/main..main` should be empty)

To check for un-merged worktree commits:
```bash
cd /home/mcintos/Code/imas-codex
git log --oneline --all --not main | head -20
```

## Environment Variables

The `.env` file contains secrets. Never expose or commit it.

## DO

- Use `uv run` for all Python commands
- Stage files individually (`git add <file>`)
- Use modern Python 3.12 syntax: `list[str]`, `X | Y`
- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations
- Use exception chaining with `from`

## DON'T

- Don't prefix commands with `cd /path &&` - terminal session is persistent
- Don't manually activate venv - use `uv run`
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names
