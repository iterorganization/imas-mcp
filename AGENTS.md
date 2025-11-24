# Agent Guidelines

This document contains rules and guidelines for AI agents working on this codebase.

## Code Style Rules

### Ruff Linting Rules

#### UP038: Use Union Types in `isinstance` Calls

When using `isinstance()` with multiple types, use Python 3.10+ union syntax (`X | Y`) instead of tuples `(X, Y)`.

**Correct:**
```python
if isinstance(e, DocsServerUnavailableError | PortAllocationError):
    raise
```

**Incorrect:**
```python
if isinstance(e, (DocsServerUnavailableError, PortAllocationError)):
    raise
```

This follows ruff's UP038 rule which enforces modern Python type syntax.

