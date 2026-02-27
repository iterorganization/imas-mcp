# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), MCP `python()` for chained processing and graph queries, `uv run` for git/tests/CLI. Conventional commits. **CRITICAL: Commit and push all changes before ending every response that modifies files.**

## Project Philosophy

Greenfield project under active development. No backwards compatibility.

- Breaking changes are expected - remove deprecated code decisively
- Avoid "enhanced", "new", "refactored" in names - just use the good name
- When patterns change, update all usages - don't leave old patterns alongside new
- Prefer explicit over clever - future agents will read this code
- Exploration notes go in facility YAML, not markdown files
- `docs/` is for mature infrastructure only
- **Build on common infrastructure** — before implementing functionality, search for existing utilities that solve the same problem. Remote SSH execution, graph queries, file parsing, and LLM calls all have canonical patterns in the codebase. New features must compose from these shared primitives rather than reimplementing them. When a pattern is needed by multiple modules, extract it to a shared location (`imas_codex/remote/`, `imas_codex/graph/`, etc.) and have all consumers import from there. Never inline SSH subprocess calls — use `run_python_script()` / `async_run_python_script()` from `imas_codex.remote.executor` with scripts in `imas_codex/remote/scripts/`.
- **Don't repeat yourself across domains** — the `files` and `wiki` discovery pipelines share the same worker architecture (`discovery/base/`). When adding a feature (filtering, scoring heuristics, worker naming), apply it consistently across all domains that use the same pattern. If data is already available in the graph (e.g., public repo detection via `SoftwareRepo` nodes), don't reimplement the check locally. One source of truth, one implementation.

## Model & Tool Configuration

All model and tool settings live in `pyproject.toml` under `[tool.imas-codex]`. No backward-compatible aliases — use the canonical accessors from `imas_codex.settings`.

**Sections** (each with a `model` parameter):

| Section | Purpose | Accessor |
|---------|---------|----------|
| `[graph]` | Neo4j connection, graph name/location | `get_graph_uri()`, `get_graph_username()`, `get_graph_password()`, `resolve_graph()` |
| `[embedding]` | Embedding model, dimension, location, scheduler | `get_model("embedding")`, `get_embedding_location()` |
| `[language]` | Structured output (scoring, discovery, labeling), batch-size | `get_model("language")` |
| `[vision]` | Image/document tasks | `get_model("vision")` |
| `[agent]` | Planning, exploration, autonomous tasks | `get_model("agent")` |
| `[compaction]` | Summarization/compaction | `get_model("compaction")` |
| `[data-dictionary]` | DD version, include-ggd, include-error-fields | `get_dd_version()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"agent"`, `"compaction"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

**Graph access:** Graph profiles separate **name** (what data) from **location** (where Neo4j runs). The default graph `"codex"` contains all facilities + IMAS DD and runs at location `"iter"`. `IMAS_CODEX_GRAPH` env var selects the graph name. `IMAS_CODEX_GRAPH_LOCATION` overrides where it runs. Each location maps to a unique bolt+HTTP port pair by convention:

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt-60sa | 7689 | 7476 |

Env var overrides (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) still apply as escape hatches over any profile. Use `resolve_graph(name)` from `imas_codex.graph.profiles` for direct profile resolution. All CLI `graph` commands accept `--graph/-g` to target a specific graph.

**Location-aware connections:** `is_local_host(host)` determines direct vs tunnel access at connection time. For edge cases, configure `login_nodes` and `local_hosts` in the facility's private YAML. Check with: `imas-codex config local-hosts`.

## Schema System

All graph node types, relationships, and properties are defined in LinkML schemas — the single source of truth.

**Schema files:**
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, TreeNode, CodeChunk, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASPath, DDVersion, Unit, IMASCoordinateSpec
- `imas_codex/schemas/common.yaml` - Shared: status enums, PhysicsDomain

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- Output: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`

**CRITICAL: Never commit auto-generated files.** These are gitignored and rebuilt on `uv sync`. If `git status` shows a generated model file as untracked or modified, do NOT stage it. Generated files:
- `imas_codex/graph/models.py`
- `imas_codex/graph/dd_models.py`
- `imas_codex/config/models.py`
- `imas_codex/core/physics_domain.py`
- `agents/schema-reference.md`

Always import enums and classes from generated models. Never hardcode status values:

```python
from imas_codex.graph.models import SourceFile, SourceFileStatus, TreeNode

sf = SourceFile(
    id="tcv:/home/codes/liuqe.py",
    facility_id="tcv",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])
```

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Schema changes are additive only — add properties, never rename or remove.

**Full schema reference:** [agents/schema-reference.md](agents/schema-reference.md) — auto-generated list of all node labels, properties, vector indexes, relationships, and enums. Rebuilt on `uv sync`.

## Facility Configuration

Per-facility YAML configs define discovery roots, wiki sites, data sources, and infrastructure details. Schema enforced via LinkML (`imas_codex/schemas/facility_config.yaml`).

**Files:**
- `imas_codex/config/facilities/<facility>.yaml` - Public config (git-tracked)
- `imas_codex/config/facilities/<facility>_private.yaml` - Private config (gitignored)

**Editing configs:** Always use MCP tools rather than direct file editing:

```python
# Add seeding paths, wiki URLs, or exploration notes
update_facility_infrastructure('tcv', {'discovery_roots': ['/new/path']})
add_exploration_note('tcv', 'Found equilibrium codes at /home/codes/liuqe')
```

**Validation:** Configs are validated against schema at load time. Check compliance:

```python
from imas_codex.discovery.base.facility import validate_facility_config
errors = validate_facility_config('tcv')  # Returns list of error strings
```

**Schema access:** The config schema is exposed via `get_graph_schema()` MCP tool — agents can query it to understand required/optional fields before editing.

## Graph State Machine

Status enums represent **durable states only**. No transient states like `scanning`, `scoring`, or `ingesting`.

**Worker coordination:** Claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`. Orphan recovery is automatic via timeout check in claim queries.

### FacilityPath Lifecycle

```
discovered → explored | skipped | stale
```

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| <0.3 | Config files, documentation |

### SourceFile Lifecycle

```
discovered → ingested | failed | stale
```

Ingestion is interrupt-safe — rerun to continue. Already-ingested files are skipped.

## Command Execution

**CRITICAL: Never pipe, tee, or redirect CLI output.** All `imas-codex` CLI commands auto-log full DEBUG output to `~/.local/share/imas-codex/logs/<command>_<facility>.log`. Piping (`|`), teeing (`tee`), or redirecting (`>`, `2>&1`) to files prevents auto-approval of terminal commands, stalling agentic workflows. Run commands directly and read the log file afterwards.

**Decision tree:**
1. Single command, local → Terminal directly (`rg`, `fd`, `tokei`, `uv run`)
2. Single command, remote → SSH (`ssh facility "command"`)
3. Chained processing → `python()` with `run()` (auto-detects local/remote)
4. Graph queries / MCP → `python()` with `query()`, `add_to_graph()`, etc.

**MCP tool routing:**
- Dedicated MCP tools for single operations: `add_to_graph()`, `get_graph_schema()`, `update_facility_infrastructure()`, `add_exploration_note()`
- `python()` REPL for chained processing, Cypher queries, IMAS/COCOS operations
- Terminal for `rg`, `fd`, `git`, `uv run`; SSH for remote single commands

**CLI in agent sessions:** Rich output is auto-detected. Non-TTY contexts (CI, stdio, pipes) automatically disable rich. Override with `IMAS_CODEX_RICH=0` if needed.

## LLM Prompts

Prompts live in `imas_codex/agentic/prompts/` using Jinja2 templates with schema injection.

- Never hardcode JSON examples - use Pydantic schema injection via `get_pydantic_schema_json()`
- Each prompt declares `schema_needs` in `prompt_loader.py` to load only required context
- LLM structured output uses Pydantic models via LiteLLM `response_format`

## Exploration

Before disk-intensive operations, **check facility excludes** to avoid repeating known timeouts:

```python
info = get_facility('tcv')
excludes = info.get('excludes', {})
print(excludes.get('large_dirs', []))
print(excludes.get('depth_limits', {}))
print(info.get('exploration_notes', [])[-3:])
```

When a command times out, **persist the constraint immediately** via `update_infrastructure()` and `add_exploration_note()`. Never repeat a timeout.

### Persistence

| Discovery Type | Destination |
|----------------|-------------|
| Source files, paths, codes, trees | `add_to_graph()` (public graph) |
| Hostnames, IPs, OS, tool versions | `update_infrastructure()` (private YAML) |
| Context for future sessions | `add_exploration_note()` |

### Data Classification

- **Graph (public):** MDSplus tree names, analysis code names/versions/paths, TDI functions, diagnostic names
- **Infrastructure (private):** Hostnames, IPs, NFS mounts, OS versions, tool availability, user directories

## Ingestion

```bash
# Queue files for ingestion
uv run imas-codex ingest queue tcv /path/a.py /path/b.py

# Pipe from remote search
ssh tcv 'rg -l "equilibrium|IMAS" /home/codes -g "*.py"' | \
  uv run imas-codex ingest queue tcv --stdin

# Process queue
uv run imas-codex ingest run tcv
uv run imas-codex ingest run tcv --min-score 0.7  # High-priority only
uv run imas-codex ingest run tcv --dry-run         # Preview

# Monitor
uv run imas-codex ingest status tcv
uv run imas-codex ingest list tcv -s failed
uv run imas-codex ingest run tcv --retry-failed
```

The pipeline extracts MDSplus tree paths, TDI function calls, IDS references, and import dependencies. Do not fabricate `patterns_matched` metadata.

## Graph Operations

### Neo4j Management

```bash
uv run imas-codex graph start                 # Start Neo4j (auto-detects mode)
uv run imas-codex graph stop                  # Stop Neo4j
uv run imas-codex graph status                # Check graph status
uv run imas-codex graph profiles              # List all profiles and ports
uv run imas-codex graph shell                 # Interactive Cypher (active profile)
uv run imas-codex graph export               # Export graph to archive
uv run imas-codex graph export -f tcv        # Per-facility export (filtered)
uv run imas-codex graph load graph.tar.gz    # Load graph archive
uv run imas-codex graph pull                 # Pull latest from GHCR
uv run imas-codex graph pull --facility tcv  # Pull per-facility graph
uv run imas-codex graph push --dev           # Push to GHCR
uv run imas-codex graph push --facility tcv  # Push per-facility graph
uv run imas-codex graph backup               # Create neo4j-admin dump backup
uv run imas-codex graph restore              # Restore from backup
uv run imas-codex graph clear                # Clear graph (with auto-backup)
uv run imas-codex graph clean --dev          # Remove all dev GHCR tags
uv run imas-codex graph clean --backups --older-than 30d  # Clean old backups
uv run imas-codex tunnel start iter          # Start SSH tunnel to remote host
uv run imas-codex tunnel status              # Show active tunnels
uv run imas-codex config private push        # Push private YAML to Gist
uv run imas-codex config secrets push iter   # Push .env to remote host
```

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Neo4j Lock Files — CRITICAL

Neo4j uses several lock file types. Mishandling them **causes data loss**.

| Lock File | Location | Purpose | Safe to Delete? |
|-----------|----------|---------|----------------|
| `store_lock` | `data/databases/` | Coordinates single-writer access | Yes — after confirming Neo4j is stopped |
| `database_lock` | `data/databases/*/` | Per-database writer lock | Yes — after confirming Neo4j is stopped |
| `write.lock` | `data/databases/*/schema/index/*/` | Lucene index segment lock | **NEVER** — deletion corrupts vector indexes |

**Rules:**
1. **Never use `find -name "*.lock"` to clean locks** — this matches Lucene `write.lock` files inside vector index directories.
2. Only remove `store_lock` and `database_lock` explicitly by path, and only after confirming Neo4j has fully stopped.
3. On GPFS/NFS, stale POSIX locks can survive process death. The safe workaround is inode replacement (`cp file file.unlock && mv -f file.unlock file`), not deletion.
4. If Lucene `write.lock` is deleted while Neo4j is running, it triggers `AlreadyClosedException`, checkpoint failure, and potential database reinitialization on next start.

**Never use the Docker entrypoint** (`/startup/docker-entrypoint.sh`) to start Neo4j in Apptainer. It calls `neo4j-admin dbms set-initial-password` and runs `rm -rf conf/*` on every start, which can reinitialize an existing database after a crash. Always use `neo4j console` directly with a host-side `conf/` bind mount.

### Vector Indexes

Embeddings require a vector index for `db.index.vector.queryNodes()` to work:

```python
from imas_codex.settings import get_embedding_dimension

dim = get_embedding_dimension()  # 1024 for Qwen3, 384 for MiniLM
gc.query(f"""
    CREATE VECTOR INDEX my_index IF NOT EXISTS
    FOR (n:MyNodeType) ON n.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}
    }}
""")
```

**Existing indexes:** See [agents/schema-reference.md](agents/schema-reference.md) for the full list of vector indexes derived from the LinkML schema.

### Semantic Search & Graph RAG

Use `semantic_search(text, index, k)` in the python REPL:

```python
# Document content (wiki, code)
semantic_search("COCOS sign conventions", index="wiki_chunk_embedding", k=5)

# Descriptive metadata (signals, paths - search by physics meaning)
semantic_search("plasma current measurement", index="facility_signal_desc_embedding", k=10)
```

Combine vector similarity with link traversal for richer context:

```python
results = query("""
    CALL db.index.vector.queryNodes('facility_signal_desc_embedding', 5, $embedding)
    YIELD node AS signal, score
    MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)
    OPTIONAL MATCH (signal)-[:MAPS_TO_IMAS]->(imas:IMASPath)
    RETURN signal.id, signal.description, da.template_python,
           collect(imas.id) AS imas_paths, score
    ORDER BY score DESC
""", embedding=embed("electron density profile"))
```

**Key relationships for traversal:**

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | MAPS_TO_IMAS | IMASPath |
| WikiChunk | HAS_CHUNK← | WikiPage |
| FacilityPath | AT_FACILITY | Facility |

**Token cost:** Always project specific properties in Cypher (`RETURN n.id, n.name`), never return full nodes. Use Cypher aggregations instead of Python post-processing.

### Batch Operations

Use `UNWIND` for batch graph writes:

```python
query('''
    UNWIND $items AS item
    MERGE (n:Tool {id: item.id})
    SET n += item
    WITH n
    MATCH (f:Facility {id: 'tcv'})
    MERGE (n)-[:AT_FACILITY]->(f)
''', items=tools)
```

### Release Workflow

```bash
# Prepare PR (pushes tag to fork)
uv run imas-codex release v4.0.0 -m 'Release message' --remote origin
# After PR merge, finalize (graph to GHCR, tag to upstream)
uv run imas-codex release v4.0.0 -m 'Release message'
```

## Remote Tools

Prefer these Rust-based CLI tools over standard Unix commands. Defined in `imas_codex/config/remote_tools.yaml`.

| Tool | Purpose | Use Instead Of |
|------|---------|----------------|
| `rg` | Pattern search | `grep -r` |
| `fd` | File finder | `find` |
| `eza` | Directory listing with tree view | `ls -la`, `tree` |
| `tokei` | LOC by language | `wc -l`, `cloc` |
| `uv` | Python package manager | `pip`, `virtualenv` |

Install on any facility: `uv run imas-codex tools install <facility>`

**Critical:** `fd` requires a path argument on large filesystems to avoid hanging: `fd -e py /path`

**Remote Python:** Remote scripts run via `run_python_script()` use the imas-codex venv Python (3.12+), not the system Python. The `_REMOTE_PATH_PREFIX` in `executor.py` prepends `~/.local/share/imas-codex/venv/bin` to PATH so `python3` resolves to the venv. Remote scripts can use modern Python syntax (`X | Y` unions, `match`, `isinstance(x, int | float)`). If a remote script fails with a syntax or type error, verify the venv is installed: `uv run imas-codex tools status <facility>`.

**Remote zombie prevention:** All remote SSH commands are wrapped with `timeout <seconds>` on the server side. When a local `subprocess.run()` times out, it kills the SSH client process but the remote process keeps running indefinitely as a zombie. The server-side `timeout` (set to local timeout + 5s) ensures the remote process self-terminates independently. This is enforced in `executor.py` for `run_command()`, `run_script_via_stdin()`, `run_python_script()`, and `async_run_python_script()`. Never bypass this by constructing raw SSH commands — always use the executor functions.

## Commit Workflow

```bash
# 1. Lint and format
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (never git add -A)
# NEVER stage auto-generated files (models.py, dd_models.py, physics_domain.py)
# NEVER stage gitignored files — run `git status --ignored` to check
# NEVER commit *_private.yaml files — they contain sensitive infrastructure data
git add <file1> <file2> ...

# 3. Commit with conventional format
uv run git commit -m "type: concise summary

Detailed explanation.
- Key changes

BREAKING CHANGE: description (if applicable)"

# 4. If pre-commit fails, fix and repeat 2-3
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

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

### Parallel Agents

Multiple agents may be working on this repository simultaneously. Assume another agent could be editing files or committing right now.

**CRITICAL — Do not touch files you didn't modify:**

- **Only stage files you modified** — never `git add -A` or `git add .`
- **NEVER run `git checkout`, `git restore`, or `git reset` on files you didn't change** — this silently destroys another agent's in-progress work with no way to recover it. Even if a file appears "dirty" or has unexpected changes, leave it alone — another agent put those changes there deliberately.
- **NEVER run `git checkout -- .` or `git restore .`** — these wipe ALL unstaged changes across the entire repo, including other agents' work
- **Never rebase** — rebase rewrites history and clobbers parallel agents' work
- **Pull before push** if push is rejected: `git pull --no-rebase origin main && git push origin main`
- **Avoid broad formatting runs** (`ruff format .`) unless you are the only agent active — prefer formatting only your changed files
- **If `git stash` is needed**, only stash your own files: `git stash push -- file1 file2`, never `git stash` (which stashes everything)

### Session Completion

**MANDATORY** after any file modifications: commit and push before responding to the user.

End every response that modifies files with the **full commit message** and a brief summary.

## Code Style

- Python ≥3.12: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`
- Exception chaining: `raise Error("msg") from e`
- `pydantic` for schemas, `dataclasses` for other data classes
- `anyio` for async
- `uv run` for all Python commands (never activate venv manually)
- Never use `git add -A`
- The `.env` file contains secrets — never expose or commit it

## CLI Logs

All discovery and DD CLI commands write DEBUG-level rotating logs to disk. The rich progress display suppresses most log output to keep the TUI clean, but full details are always available in the log files.

**Log directory:** `~/.local/share/imas-codex/logs/`

**Log naming:** `{command}_{facility}.log` (e.g. `paths_tcv.log`, `wiki_jet.log`, `imas_dd.log`). Logs rotate at 10 MB with 3 backups.

```bash
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log  # Follow live
rg "ERROR|WARNING" ~/.local/share/imas-codex/logs/     # Find errors
```

**NEVER pipe, tee, or redirect CLI output.** Piping blocks auto-approval in agentic contexts. The logging infrastructure already captures everything to disk — run commands directly and read the log file afterwards.

## Testing

```bash
uv sync --extra test          # Required in worktrees
uv run pytest                 # All tests
uv run pytest --cov=imas_codex  # With coverage
uv run pytest tests/path/to/test.py::test_function  # Specific test
```

## Python REPL

The `python()` MCP tool provides a persistent REPL with pre-loaded utilities:

```python
python("print([name for name in dir() if not name.startswith('_')])")
python("help(search_imas)")
python("import inspect; print(inspect.signature(get_facility))")
python("print(reload())")  # After editing imas_codex/ source files
```

## Quick Reference

| Task | Command |
|------|---------|
| Graph query | `python("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(search_code('equilibrium'))")` |
| Facility info | `python("print(get_facility('tcv'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Update infra | `update_facility_infrastructure('tcv', {...})` |
| Remote command | `ssh facility "rg pattern /path"` |

## Embedding Server

Config lives in `pyproject.toml` under `[tool.imas-codex.embedding]`. Key accessor: `get_embedding_location()` returns the facility name or `"local"`. Port derived from position in shared `locations` list: `18765 + offset`.

```bash
imas-codex serve embed deploy    # Deploy per config (slurm or systemd)
imas-codex serve embed status    # Check server health + SLURM jobs
imas-codex serve embed restart   # Stop + redeploy
imas-codex serve embed stop      # Stop all embed processes
imas-codex serve embed logs      # View SLURM logs
imas-codex serve embed start     # Start server locally (foreground)
imas-codex serve embed service install  # Install systemd service
```

If embedding fails, check in order:
1. Tunnel active: `lsof -i :18765`
2. Service running: `ssh iter "systemctl --user status imas-codex-embed"`
3. Server health: `curl http://localhost:18765/health`

## Domain Workflows

Extended examples and edge cases for each domain: [agents/](agents/)

| Agent | Purpose |
|-------|---------|
| Explore | Remote facility discovery (read-only + MCP) |
| Develop | Code development (standard + MCP) |
| Ingest | Code ingestion pipeline (core + MCP) |
| Graph | Knowledge graph operations (core + MCP) |

## Fallback: MCP Server Not Running

```bash
uv run imas-codex graph status          # Graph operations
uv run imas-codex graph shell           # Interactive Cypher
uv run imas-codex ingest run tcv        # Ingestion
uv run pytest                           # Testing
```
