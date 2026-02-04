# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), MCP `python()` for chained processing and graph queries, `uv run` for git/tests/CLI. Conventional commits. **CRITICAL: Commit and push all changes before ending every response that modifies files.**

## Project Philosophy

This is a greenfield project under active development. We move fast and do not maintain backwards compatibility.

**Key principles:**
- No backwards compatibility requirements - breaking changes are expected
- Strategic pivots are normal - remove deprecated code decisively
- Avoid "enhanced", "new", "refactored" in names - just use the good name
- When patterns change, update all usages - don't leave old patterns alongside new ones
- Prefer explicit over clever - future agents will read this code
- Never create documentation for transient analysis - `docs/` is for mature infrastructure only
- Exploration notes go in facility YAML, not markdown files

## Schema System

All graph node types, relationships, and properties are defined in LinkML schemas. These schemas are the single source of truth.

**Schema files:**
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, TreeNode, CodeChunk, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASPath, DDVersion, Unit, IMASCoordinateSpec
- `imas_codex/schemas/common.yaml` - Shared: status enums, PhysicsDomain

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- Output: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`

**Enforcement rules:**

Always import enums and classes from generated models. Never hardcode status values.

```python
# Correct - import from generated models
from imas_codex.graph.models import SourceFile, SourceFileStatus, TreeNode

sf = SourceFile(
    id="tcv:/home/codes/liuqe.py",
    facility_id="tcv",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])

# Wrong - hardcoded string bypasses validation
data = {"id": "...", "status": "discovered"}  # No type checking!
```

**Extending schemas:**
1. Edit LinkML YAML in `imas_codex/schemas/`
2. Run `uv run build-models --force`
3. Import new classes from `imas_codex.graph.models`

Schema changes are additive only in the graph. Add properties, never rename or remove.

## LLM Prompts

Prompts live in `imas_codex/agentic/prompts/` using Jinja2 templates with schema injection.

**Key principles:**
- Never hardcode JSON examples - use Pydantic schema injection
- Each prompt declares its `schema_needs` to load only required context
- LLM structured output uses Pydantic models via LiteLLM `response_format`

**Schema injection pattern** (`prompt_loader.py`):

```python
# Prompts get schema context via providers:
_DEFAULT_SCHEMA_NEEDS = {
    "discovery/scorer": ["path_purposes", "score_dimensions", "scoring_schema"],
    "discovery/rescorer": ["rescore_schema"],  # Only 2 context keys
}

# Template uses injected variables:
{{ scoring_schema_example }}    # JSON example from Pydantic
{{ scoring_schema_fields }}     # Field descriptions
{{ path_purposes }}             # Enum values from LinkML
```

**Pydantic models for LLM output** (`discovery/paths/models.py`):

```python
class DirectoryScoringBatch(BaseModel):
    """LLM returns this structure via response_format."""
    results: list[DirectoryScoringResult] = Field(...)

# Scorer uses:
response = litellm.completion(
    model=model_id,
    response_format=DirectoryScoringBatch,  # Enforced by LLM
    messages=[...],
)
batch = DirectoryScoringBatch.model_validate_json(response.content)
```

**DO:**
- Define Pydantic models for all LLM structured output
- Use `get_pydantic_schema_json()` to generate JSON examples
- Add new prompts to `_DEFAULT_SCHEMA_NEEDS` with minimal context
- Include `{% include "schema/output-format.md" %}` for schema documentation

**DON'T:**
- Hardcode JSON examples in prompts (breaks on schema changes)
- Load full schema context when prompt only needs specific fields
- Use plain text parsing when structured output is available

## Critical Rules

### Locality Check

Always determine if you're on the target facility before choosing execution method:

```bash
hostname
pwd
```

**Command Execution Decision Tree:**
1. Single command on local facility? → Terminal directly (`rg`, `fd`, `dust`)
2. Single command on remote facility? → Direct SSH (`ssh facility "command"`)
3. Chained processing with logic? → `python()` with `run()` (auto-detects local/remote)
4. Graph queries or MCP functions? → `python()` with `query()`, `add_to_graph()`, etc.

### MCP Tool Selection

**Use dedicated MCP tools** for single operations:
- `add_to_graph()` - Create graph nodes with schema validation
- `get_graph_schema()` - Get schema for Cypher generation
- `update_facility_infrastructure()` - Update private facility data
- `get_facility_infrastructure()` - Read private facility data
- `add_exploration_note()` - Add timestamped exploration note

**Use `python()` REPL** when you need:
- Chained operations with intermediate processing
- Graph queries with Cypher (`query()` function)
- IMAS/COCOS domain operations (`search_imas`, `validate_cocos`)
- Complex data transformations requiring Python execution

**Use terminal directly** for single operations:
- Local: `rg`, `fd`, `git`, `dust`, `uv run`
- Remote: `ssh facility "command"`

**Never use `python()` for:**
- Text formatting (LLM can do this natively)
- Single infrastructure updates (use dedicated MCP tools)
- Simple string operations

### CLI in Chat Sessions

When running CLI commands in chat/agent sessions, **always use `--no-rich`** if available:

```bash
# Correct - no-rich prevents animation loops in non-TTY contexts
uv run imas-codex discover paths tcv --no-rich --seed

# Wrong - rich progress bars cause output loops in chat
uv run imas-codex discover paths tcv --seed
```

### Graph Backup

Before ANY operation that modifies or deletes graph nodes:

- Never use `DETACH DELETE` on production data without user confirmation
- For re-embedding: update nodes in place, don't delete and recreate

### Token Cost Optimization

Graph queries can be expensive. Follow these rules:

**Never return full nodes** - always project specific properties:

```python
# Bad - embeddings cost ~2k tokens per node
query("MATCH (n:IMASPath) RETURN n LIMIT 10")

# Good - project only needed properties (~50 tokens per node)
query("MATCH (n:IMASPath) RETURN n.id, n.name, n.documentation LIMIT 10")
```

**Use Cypher aggregations** instead of Python post-processing:

```python
# Bad - multiple calls, Python aggregation
files = query("MATCH (f:SourceFile) RETURN f.status")
# Then Counter() in another call

# Good - single call with Cypher aggregation
query("""
    MATCH (f:SourceFile)
    RETURN f.status AS status, count(*) AS count
    ORDER BY count DESC
""")
```

## Fast Tools

High-performance Rust-based CLI tools that **must be used instead of slower builtins**. Defined in `imas_codex/config/fast_tools.yaml`.

### Installation

```bash
# Check environment status (tools + Python + venv)
uv run imas-codex tools status local

# Install everything (tools + Python + venv)
uv run imas-codex tools install local

# Install specific tool only
uv run imas-codex tools install local --tool rg

# Install on remote facility
uv run imas-codex tools install tcv
```

After installation, ensure `~/bin` is in your PATH (the installer adds this to `.bashrc` automatically).

### Required Tools

These are essential for effective exploration:

| Tool | Purpose | Use Instead Of |
|------|---------|----------------|
| `rg` | Pattern search across files (10x faster than grep) | `grep -r` |
| `fd` | Find files by name/extension (5x faster than find) | `find` |
| `eza` | Modern ls with tree view for directory hierarchy | `ls -la`, `tree` |
| `git` | Version control, metadata extraction | - |
| `gh` | GitHub API access, repo visibility checking | curl to API |
| `uv` | Fast Python package manager for venv/dependency management | pip, virtualenv |

### Optional Tools

Enhance exploration but not required:

| Tool | Purpose | Use Instead Of |
|------|---------|----------------|
| `tokei` | Lines of code by language | `wc -l`, `cloc` |
| `scc` | Code complexity and SLOC metrics | `cloc` |
| `dust` | Visual disk usage analyzer | `du -h` |
| `bat` | Syntax-highlighted file viewing | `cat`, `less` |
| `delta` | Better git diff viewer | `diff` |
| `fzf` | Fuzzy finder for interactive selection | - |
| `yq` | YAML processor | Python yaml parsing |
| `jq` | JSON processor | Python json parsing |

### Usage Rules

**Always prefer fast tools:**

```bash
# CORRECT - use rg
rg 'IMAS' /path -g '*.py'

# WRONG - never use grep for code search
grep -r 'IMAS' /path --include='*.py'

# CORRECT - use fd
fd -e py /path

# WRONG - never use find for file search
find /path -name '*.py'
```

**Critical: `fd` requires path as trailing argument:**

```bash
# CORRECT - path specified
fd -e py /work/projects

# WRONG - will hang without path on large filesystems
fd -e py
```

**Remote facility usage:**

```bash
# Single command via SSH
ssh tcv "rg -l 'IMAS' /home/codes"
ssh tcv "fd -e py /home/codes | head -20"
```

## Python Environments

Use uv to ensure modern Python (3.10+) on all facilities, avoiding version compatibility issues.

### Why This Matters

- System Python varies: JET has 3.13, TCV has 3.9, ITER has 3.9
- Development on modern Python, then deployment fails on old Python
- uv installs Python from python-build-standalone (GitHub), not PyPI
- Works on airgapped facilities if GitHub is accessible

### Quick Setup

```bash
# Check environment status on a facility (tools + Python + venv)
uv run imas-codex tools status tcv

# Complete setup (tools + uv + Python + venv) in one command
uv run imas-codex tools install tcv

# Install specific tool only
uv run imas-codex tools install tcv --tool rg

# Install tools only, skip Python/venv
uv run imas-codex tools install tcv --tools-only
```

### Facility Status Reference

| Facility | System Python | uv | PyPI | Strategy |
|----------|--------------|-----|------|----------|
| JET | 3.13.3 | ✗ | ✓ | Already modern, install uv for venv |
| TCV | 3.9.25 | ✗ | ✓ | Install uv, then Python 3.12 |
| ITER | 3.9.16 | ✗ | ✓ | Install uv, then Python 3.12 |
| JT60SA | 3.9.10 | ✓ | ✗ | Already has 3.12/3.13 via uv |

### Using the Remote venv

```bash
# Activate on remote
ssh tcv "source ~/.local/share/imas-codex/venv/bin/activate && python --version"

# Run scripts with the venv Python
ssh tcv "~/.local/share/imas-codex/venv/bin/python script.py"
```

## Commit Workflow

```bash
# 1. Lint and format first
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (never git add -A)
git add <file1> <file2> ...

# 3. Commit with conventional format
uv run git commit -m "type: concise summary

Detailed explanation of what changed and why.
- First significant change
- Second change with rationale

BREAKING CHANGE: description (if applicable)"

# 4. If pre-commit fails, fix and repeat steps 2-3

# 5. Push
git push origin main
```

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring |
| docs | Documentation |
| test | Test changes |
| chore | Maintenance |

Breaking changes use `BREAKING CHANGE:` footer, not `type!:` suffix.

## Working in Worktrees

Cursor remote agents often work in auto-created worktrees. Commits in worktrees are NOT on `main` until merged.

**Always complete both steps together:**

```bash
# Step 1: Commit in worktree
uv run ruff check --fix . && uv run ruff format .
git add <file1> <file2> ...
uv run git commit -m "type: description"

# Step 2: Immediately merge to main
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

Never end a session after step 1. Check for un-merged commits:
```bash
git log --oneline --all --not main | head -20
```

## Session Completion

**CRITICAL - MANDATORY FOR EVERY RESPONSE THAT MODIFIES FILES:**

After completing any file modifications, you MUST commit before responding to the user:

1. `git status --short` - check for uncommitted changes
2. If changes exist: `uv run ruff check --fix . && uv run ruff format .`
3. `git add <files>` - stage specific files (never `git add -A`)
4. `uv run git commit -m "type: description"` - commit with conventional format
5. `git push origin main` - push immediately
6. End your response with the **full commit message** followed by a **rich summary**:

```
✓ Committed: `<commit-hash>`

type: concise summary

Detailed explanation of what changed and why.
- First significant change
- Second change with rationale
```

**Rich Summary** (after commit message):
Provide a brief, user-friendly summary highlighting:
- What capability was added or changed
- Key files affected
- Any breaking changes or migration notes
- Next steps if applicable

Example:
> FacilitySignal now enforces PhysicsDomain enum for categorization. The schema imports `physics_domains.yaml` directly, eliminating the redundant string description. Graph nodes updated with proper domain values.

**Why show full message:** The commit message serves as the session's work record. Showing only a one-line summary loses context about what was done and why. The full message allows the user to track changes across sessions without checking git log.

**Why commit is mandatory:** Uncommitted changes are lost when sessions end. The user cannot see your work until it's committed. Never describe changes without committing them first.

## Code Style

### Python Version

This project requires Python greater or equal to 3.12 (`requires-python = ">=3.12"`). Do not write fallback code for older Python versions. Remove any legacy compatibility patterns you encounter.

```python
# Wrong - unnecessary fallback for older Python
try:
    script_path = importlib.resources.files("package").joinpath("file")
except (AttributeError, TypeError):
    import pkg_resources  # Python 3.8 fallback - DELETE THIS
    ...

# Correct - use Python 3.12 features directly
script_path = importlib.resources.files("package").joinpath("file")
```

### Type Annotations

```python
# Correct - Python 3.12 syntax
def process(items: list[str]) -> dict[str, int]: ...
if isinstance(e, ValueError | TypeError): ...

# Wrong - legacy typing module
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

- Use `pydantic` for schemas, `dataclasses` for other data classes
- Use `anyio` for async operations
- Stage files individually, never `git add -A`
- Use `uv run` for all Python commands (handles venv automatically)

### DO

- Check locality first (`hostname`) before choosing execution method
- Use terminal directly for single operations on local facility
- Use direct SSH for single operations on remote facility
- Use MCP `python()` only for chained processing and graph queries
- Batch multiple operations in single `python()` calls
- Use `python("print(reload())")` after editing source files

### DON'T

- Don't use `python()` wrapper for single commands (local or remote)
- Don't manually activate venv - use `uv run`
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

## Testing

```bash
# Sync dependencies (required in worktrees)
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=imas_codex

# Run specific test
uv run pytest tests/path/to/test.py::test_function

# Fast iteration with restricted IDS filter
uv run imas-codex clusters build --ids-filter "core_profiles equilibrium" -v -f
```

## Python REPL Discovery

The `python()` tool provides a persistent REPL with pre-loaded utilities. Discover the API first:

```python
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")
python("import inspect; print(inspect.signature(get_facility))")
```

After editing `imas_codex/` source files, reload:
```python
python("print(reload())")
```

## Quick Reference

| Task | Command |
|------|---------|
| Check locality | `hostname` |
| Local command | `rg pattern /path` |
| Remote command | `ssh facility "rg pattern /path"` |
| Graph query | `python("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(search_code('equilibrium'))")` |
| Facility info | `python("print(get_facility('tcv'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Update infrastructure | `update_facility_infrastructure('tcv', {...})` |

## Domain Workflows

For detailed domain-specific workflows, see:

| Workflow | Documentation |
|----------|---------------|
| Facility exploration | [agents/explore.md](agents/explore.md) |
| Development | [agents/develop.md](agents/develop.md) |
| Code ingestion | [agents/ingest.md](agents/ingest.md) |
| Graph operations | [agents/graph.md](agents/graph.md) |

## Custom Agents

Select from the VS Code agent dropdown for restricted toolsets:

| Agent | Purpose |
|-------|---------|
| Explore | Remote facility discovery (read-only + MCP) |
| Develop | Code development (standard + MCP) |
| Ingest | Code ingestion pipeline (core + MCP) |
| Graph | Knowledge graph operations (core + MCP) |

## Remote Embedding Server

The MCP server uses remote embedding by default (`embedding-backend = "remote"` in pyproject.toml). The embedding server runs on the ITER cluster GPU and is accessed via SSH tunnel.

**Architecture:**
- Server: FastAPI app running on ITER with Tesla T4 GPU
- Client: HTTP requests via SSH tunnel (port 18765)
- Model: Qwen/Qwen3-Embedding-0.6B (1024-dim embeddings)

**Local workstation setup:**

```bash
# Establish SSH tunnel (required for MCP embedding functions)
ssh -f -N -L 18765:127.0.0.1:18765 iter

# Check server status
imas-codex serve embed status
```

**ITER cluster management:**

```bash
# Check service status
ssh iter "systemctl --user status imas-codex-embed"

# Start/stop/restart
ssh iter "systemctl --user start imas-codex-embed"
ssh iter "systemctl --user stop imas-codex-embed"
ssh iter "systemctl --user restart imas-codex-embed"

# View logs
ssh iter "journalctl --user -u imas-codex-embed -f"
```

**First-time installation on ITER cluster:**

```bash
ssh iter
cd ~/Code/imas-codex
uv sync --extra gpu
imas-codex serve embed service install --gpu 1
imas-codex serve embed service start
```

**Troubleshooting:**

If embedding calls fail from MCP, check in order:
1. SSH tunnel active: `lsof -i :18765`
2. Server running: `ssh iter "systemctl --user status imas-codex-embed"`
3. Server health: `curl http://localhost:18765/health`

## Fallback: MCP Server Not Running

If `python()` is unavailable, use CLI:

```bash
# Graph operations
uv run imas-codex neo4j status
uv run imas-codex neo4j shell
uv run imas-codex neo4j dump

# Ingestion
uv run imas-codex ingest queue tcv /path/*.py
uv run imas-codex ingest run tcv

# Testing
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```
