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

## Model & Tool Configuration

All model and tool settings live in `pyproject.toml` under `[tool.imas-codex]`. No backward-compatible aliases — use the canonical accessors from `imas_codex.settings`.

**Sections** (each with a `model` parameter):

| Section | Purpose | Accessor |
|---------|---------|----------|
| `[graph]` | Neo4j connection, location/ports | `get_graph_uri()`, `get_graph_username()`, `get_graph_password()`, `resolve_neo4j()` |
| `[embedding]` | Embedding model, dimension, backend | `get_model("embedding")` |
| `[language]` | Structured output (scoring, discovery, labeling), batch-size | `get_model("language")` |
| `[vision]` | Image/document tasks | `get_model("vision")` |
| `[agent]` | Planning, exploration, autonomous tasks | `get_model("agent")` |
| `[compaction]` | Summarization/compaction | `get_model("compaction")` |
| `[data-dictionary]` | DD version, include-ggd, include-error-fields | `get_dd_version()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"agent"`, `"compaction"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

**Graph access:** Two orthogonal concerns: **location** (pyproject.toml) controls where Neo4j runs, **name** (CLI/env) selects which data directory to use. Graph identity (name + facilities) lives in a `(:GraphMeta)` node inside the graph itself, set via `graph init`. `IMAS_CODEX_GRAPH` env var selects the graph name (default: `"codex"`). `IMAS_CODEX_GRAPH_LOCATION` overrides where Neo4j runs (default: `"iter"`). Each location maps to a unique bolt+HTTP port pair by convention:

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt60sa | 7689 | 7476 |

Env var overrides (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) still apply as escape hatches over any profile. Use `resolve_neo4j(name)` from `imas_codex.graph.profiles` for direct profile resolution. All CLI `graph` commands accept `--graph/-g` to target a specific graph.

**Location-aware connections:** The `host` field on `Neo4jProfile` records where Neo4j physically runs (SSH alias or hostname). `is_local_host(host)` determines direct vs tunnel access at connection time. For remote hosts, set `IMAS_CODEX_TUNNEL_BOLT_{HOST}` env var to override the tunnel port. Locality detection uses hostname matching, SSH config resolution, and IP bind probes. For edge cases (VIP/load-balancer sites), configure `login_nodes` and `local_hosts` in the facility's private YAML (syncs with `config private push/pull`). Session-level override: `IMAS_CODEX_LOCAL_HOSTS=iter` env var (do NOT put in `.env` — it travels with `config secrets push`).

**Facility locality config:** Add to `<facility>_private.yaml`:
```yaml
login_nodes:
  - "hostname-pattern-*.example.org"  # Glob patterns for login node FQDNs
local_hosts:
  - facility_alias                    # SSH aliases treated as local
```
When the current machine's FQDN matches a `login_nodes` pattern, that facility's `local_hosts` are treated as local by `is_local_host()`. Check with: `imas-codex config local-hosts`.

**Graph config in pyproject.toml:**
```toml
[tool.imas-codex.graph]
location = "iter"       # Where it runs (override: IMAS_CODEX_GRAPH_LOCATION=local)
username = "neo4j"
password = "imas-codex"
# Graph name (data identity) lives in (:GraphMeta) node, set via: graph init
# Override active graph name: IMAS_CODEX_GRAPH=dev
```

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

**CLI in agent sessions:** Always use `--no-rich` to prevent animation loops in non-TTY contexts.

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
uv run imas-codex serve neo4j status         # Check active graph status
uv run imas-codex serve neo4j status -g tcv  # Check specific profile
uv run imas-codex serve neo4j start -g tcv   # Start specific profile
uv run imas-codex serve neo4j profiles       # List all profiles and ports
uv run imas-codex serve neo4j shell          # Interactive Cypher (active profile)
uv run imas-codex graph export               # Export graph to archive
uv run imas-codex graph export -f tcv        # Per-facility export (filtered)
uv run imas-codex graph load graph.tar.gz    # Load graph archive
uv run imas-codex graph pull                 # Pull latest from GHCR
uv run imas-codex graph pull --facility tcv  # Pull per-facility graph
uv run imas-codex graph push --dev           # Push to GHCR
uv run imas-codex graph push --facility tcv  # Push per-facility graph
uv run imas-codex graph init --name codex --facility iter  # Initialize GraphMeta
uv run imas-codex graph facility list        # List facilities in GraphMeta
uv run imas-codex graph facility add tcv     # Add facility to GraphMeta
uv run imas-codex graph facility remove tcv  # Remove facility from GraphMeta
uv run imas-codex graph backup               # Create neo4j-admin dump backup
uv run imas-codex graph restore              # Restore from backup
uv run imas-codex graph clear                # Clear graph (--force required)
uv run imas-codex graph clean --dev          # Remove all dev GHCR tags
uv run imas-codex graph clean --backups --older-than 30d  # Clean old backups
uv run imas-codex tunnel start iter          # Start SSH tunnel to remote host
uv run imas-codex tunnel status              # Show active tunnels
uv run imas-codex config private push        # Push private YAML to Gist
uv run imas-codex config secrets push iter   # Push .env to remote host
```

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

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

## Fast Tools

Prefer these Rust-based CLI tools over standard Unix commands. Defined in `imas_codex/config/fast_tools.yaml`.

| Tool | Purpose | Use Instead Of |
|------|---------|----------------|
| `rg` | Pattern search | `grep -r` |
| `fd` | File finder | `find` |
| `eza` | Directory listing with tree view | `ls -la`, `tree` |
| `tokei` | LOC by language | `wc -l`, `cloc` |
| `uv` | Python package manager | `pip`, `virtualenv` |

Install on any facility: `uv run imas-codex tools install <facility>`

**Critical:** `fd` requires a path argument on large filesystems to avoid hanging: `fd -e py /path`

## Commit Workflow

```bash
# 1. Lint and format
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (never git add -A)
# NEVER stage auto-generated files (models.py, dd_models.py, physics_domain.py)
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

- **Only stage files you modified** — never `git add -A` or `git add .`
- **Never `git checkout` or `git restore` files you didn't change** — this silently discards another agent's work
- **Never rebase** — rebase rewrites history and clobbers parallel agents' work
- **Pull before push** if push is rejected: `git pull --no-rebase origin main && git push origin main`
- **Avoid broad formatting runs** (`ruff format .`) unless you are the only agent active — prefer formatting only your changed files

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

**Log naming:** `{command}_{facility}.log` — each facility gets its own log file, enabling parallel runs without interleaved output. Commands without a facility parameter use `{command}.log`.

| CLI Command | Log File |
|-------------|----------|
| `discover paths tcv` | `paths_tcv.log` |
| `discover paths iter` | `paths_iter.log` |
| `discover wiki jet` | `wiki_jet.log` |
| `discover signals jt60sa` | `signals_jt60sa.log` |
| `discover files tcv` | `files_tcv.log` |
| `imas build` | `imas_dd.log` |
| `imas clear` | `imas_dd.log` |

```bash
# View logs during or after a run
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log  # Follow live
cat ~/.local/share/imas-codex/logs/wiki_iter.log       # Full log
ls -la ~/.local/share/imas-codex/logs/                 # List all logs

# Search for errors across all CLI logs
rg "ERROR|WARNING" ~/.local/share/imas-codex/logs/

# Compare facilities side-by-side
tail -f ~/.local/share/imas-codex/logs/paths_*.log

# Diagnose a specific worker (e.g., artifact worker hang)
rg "artifact_worker" ~/.local/share/imas-codex/logs/wiki_tcv.log
```

Logs rotate at 10 MB with 3 backups (e.g., `paths_tcv.log`, `paths_tcv.log.1`). The file handler captures all `imas_codex.*` loggers at DEBUG level regardless of the `--no-rich` or `--verbose` flags.

**NEVER pipe, tee, or redirect CLI output.** This is the #1 cause of stalled agentic workflows. The logging infrastructure already captures everything — piping adds zero value and blocks auto-approval.

```bash
# WRONG — blocks auto-approval, requires user interaction
uv run imas-codex discover wiki tcv --no-rich 2>&1 | tee /tmp/wiki_tcv.txt
uv run imas-codex discover signals tcv --no-rich > /tmp/signals.log 2>&1
uv run imas-codex discover paths jet --no-rich 2>&1 | cat

# RIGHT — run directly, logs are written automatically
uv run imas-codex discover wiki tcv --no-rich
uv run imas-codex discover signals tcv --no-rich
uv run imas-codex discover paths jet --no-rich

# MONITOR — read the auto-generated logs
tail -f ~/.local/share/imas-codex/logs/signals_tcv.log  # Follow live
rg "ERROR|WARNING" ~/.local/share/imas-codex/logs/      # Find errors
cat ~/.local/share/imas-codex/logs/wiki_tcv.log          # Full output
```

**Why this matters:** Terminal commands with pipes (`|`), redirects (`>`), or subshells require explicit user approval in agentic contexts. A single piped command can block an entire parallel workflow for minutes. The built-in `RotatingFileHandler` writes DEBUG-level logs to disk automatically — the same data you would capture via piping is already there.

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

Login-node GPU server on ITER (T4 GPU 1, Qwen3-Embedding-0.6B, 256-dim).

Architecture: `workstation → SSH tunnel → login:18765` or on ITER: direct `localhost:18765`

Establish tunnel (from workstation): `ssh -f -N -L 18765:127.0.0.1:18765 iter`

The server runs as a systemd user service on the login node. Management:

```bash
imas-codex serve embed start     # Start server locally (foreground)
imas-codex serve embed status    # Check server health
imas-codex serve embed service install  # Install systemd service
imas-codex serve embed service start    # Start via systemd
imas-codex serve embed service status   # Check systemd service
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
uv run imas-codex serve neo4j status    # Graph operations
uv run imas-codex serve neo4j shell     # Interactive Cypher
uv run imas-codex ingest run tcv        # Ingestion
uv run pytest                           # Testing
```
