---
name: project-dev
description: How to develop, test, lint, and build the imas-codex project. Use when making code changes, running tests, or building the project.
---

# imas-codex Development

## Essential Commands

All commands use `uv run` — never bare `python` or `pytest`.

### Testing

```bash
uv run pytest                                        # All tests
uv run pytest tests/path/to/test.py::test_func -v    # Specific test
uv run pytest --cov=imas_codex --cov-report=term     # With coverage
uv run pytest -m "not slow"                           # Skip slow tests
uv run pytest tests/graph/ -m graph                   # Graph tests (needs Neo4j)
```

### Linting & Formatting

```bash
uv run ruff check --fix .     # Lint with auto-fix
uv run ruff format .          # Format
```

### Schema Workflow

When modifying LinkML schemas (`imas_codex/schemas/*.yaml`):

```bash
uv run build-models --force   # Rebuild generated models
uv run pytest tests/graph/test_schema_compliance.py -v  # Verify compliance
```

Never commit auto-generated files: `models.py`, `dd_models.py`, `physics_domain.py`, `schema-reference.md`.

### Dependency Management

```bash
uv sync                  # Install all deps
uv sync --extra test     # Include test deps (needed in worktrees)
uv sync --extra gpu      # Include GPU deps (for embedding)
```

## Git Workflow

```bash
git add <file1> <file2>                          # Stage specific files only
uv run git commit -m 'type: concise summary'     # Conventional commit
git pull --no-rebase origin main                  # Always merge, never rebase
git push origin main
```

### Commit Types

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring |
| docs | Documentation |
| test | Test changes |
| chore | Maintenance |

### Never Do

- `git add -A` or `git add .` — stage specific files only
- Commit auto-generated files (models.py, dd_models.py, physics_domain.py)
- Use `git rebase` — always merge
- Run bare `python` or `pytest` (always `uv run`)
- Include AI co-authorship trailers in commits
- Add phase labels or step numbers in commit titles
- Stage files you didn't modify (other agents may be working in parallel)

## Test Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.graph` | Requires live Neo4j (auto-skipped if unavailable) |
| `@pytest.mark.slow` | Excluded by default |
| `@pytest.mark.integration` | Full integration tests |
| `@pytest.mark.unit` | Fast unit tests |

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `imas_codex/cli/` | CLI commands (entry point: `imas-codex`) |
| `imas_codex/schemas/` | LinkML schema definitions (source of truth) |
| `imas_codex/graph/` | Graph client, generated models |
| `imas_codex/discovery/` | Discovery pipelines (wiki, code, signals, paths) |
| `imas_codex/tools/` | MCP tool implementations |
| `imas_codex/remote/` | Remote execution (SSH, scripts) |
| `imas_codex/llm/` | LLM integration and prompt templates |
| `tests/` | Test suite (mirrors source structure) |
| `plans/features/` | Active feature plans |
| `agents/` | Agent documentation and schema reference |

## Key Patterns

- **Import from generated models**: `from imas_codex.graph.models import SourceFile, SourceFileStatus`
- **LLM calls**: Use `call_llm_structured()` from `imas_codex.discovery.base.llm`
- **Prompt rendering**: Use `render_prompt()` from `imas_codex.llm.prompt_loader`
- **Model selection**: Use `get_model(section)` from `imas_codex.settings`
- **Facility config**: Use `get_facility(facility)` — never hardcode facility values
- **Remote execution**: Use `run_python_script()` from `imas_codex.remote.executor`
