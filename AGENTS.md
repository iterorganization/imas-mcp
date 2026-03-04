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
| `[discovery]` | Discovery threshold for high-value processing | `get_discovery_threshold()` |
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

**CRITICAL: Never commit auto-generated files.** These are gitignored and rebuilt on `uv sync`. Generated files: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`, `imas_codex/core/physics_domain.py`, `agents/schema-reference.md`.

Always import enums and classes from generated models. Never hardcode status values:

```python
from imas_codex.graph.models import SourceFile, SourceFileStatus, TreeNode
sf = SourceFile(id="tcv:/home/codes/liuqe.py", facility_id="tcv",
    path="/home/codes/liuqe.py", status=SourceFileStatus.discovered)
add_to_graph("SourceFile", [sf.model_dump()])
```

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Schema changes are additive only — add properties, never rename or remove.

**Full schema reference:** [agents/schema-reference.md](agents/schema-reference.md) — auto-generated list of all node labels, properties, vector indexes, relationships, and enums.

### Schema Design Guidelines

#### Dual Property + Relationship Model

Every slot that references another class produces **both** a node property AND a Neo4j relationship. This is intentional — it supports multiple search and traversal patterns:

- **Property** (`n.facility_id = 'tcv'`): Fast `WHERE` filtering without relationship traversal.
- **Relationship** (`(n)-[:AT_FACILITY]->(f:Facility)`): Graph traversal, multi-hop queries, path-finding.

`create_nodes()` in `client.py` implements this: `SET n += item` stores all properties, then for each slot with a class range it creates relationships via `MERGE`. **Never remove one side of the dual model.**

#### Relationship Type Annotations

When a slot has `range: SomeClass`, the relationship type is:
1. **Explicit annotation** (preferred): `annotations: { relationship_type: AT_FACILITY }`
2. **Auto-derived fallback**: slot name uppercased: `signal` → `SIGNAL`, `has_chunk` → `HAS_CHUNK`

All `facility_id` slots MUST have `range: Facility` and `annotations: { relationship_type: AT_FACILITY }`.

#### Key Conventions

- Use `identifier: true` on exactly one slot per class (always `id`)
- Composite IDs use colon separator: `facility_id:unique_part` (e.g., `"tcv:/home/codes/liuqe.py"`)
- Nodes with `embedding` + `description` slots automatically get a vector index named `{snake_case_label}_desc_embedding`
- Status enums represent **durable states only** — no transient states. Worker coordination uses `claimed_at` timestamps.
- Slots annotated with `is_private: true` are excluded from the graph — they exist only in facility YAML configs.

#### What NOT to Do in Schemas

- Don't hardcode enum values in Python — import from generated models
- Don't create a `facility_id` slot as plain `string` — always use `range: Facility` + `relationship_type: AT_FACILITY`
- Don't add transient states to status enums — use `claimed_at` for worker coordination
- Don't skip the `description` field — it enables semantic search via embeddings
- Don't use `multivalued: true` on relationship slots unless genuinely many-to-many

## Facility Configuration

Per-facility YAML configs define discovery roots, wiki sites, data sources, and infrastructure details. Schema enforced via LinkML (`imas_codex/schemas/facility_config.yaml`).

- `imas_codex/config/facilities/<facility>.yaml` — Public config (git-tracked): discovery_roots, data_sources, wiki_sites, data_access_patterns
- `imas_codex/config/facilities/<facility>_private.yaml` — Private config (gitignored): hostnames, IPs, NFS mounts, OS versions

**CRITICAL: All facility-specific configuration MUST live in YAML files.** Never hardcode facility names, tree names, version numbers, or any other facility-specific values in Python code. Load config via `get_facility(facility)` from `imas_codex.discovery.base.facility`.

**Editing configs:** Use MCP tools: `update_facility_infrastructure()`, `add_exploration_note()`. Config schema is exposed via `get_graph_schema()`.

## Graph State Machine

Status enums represent **durable states only**. Worker coordination: claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`.

- **FacilityPath**: `discovered → explored | skipped | stale` (score 0.9+ = IMAS integration, 0.7+ = MDSplus access, <0.3 = config/docs)
- **SourceFile**: `discovered → ingested | failed | stale` (interrupt-safe — rerun to continue)

## Command Execution

**CRITICAL: Never pipe, tee, or redirect CLI output.** All `imas-codex` CLI commands auto-log full DEBUG output to `~/.local/share/imas-codex/logs/<command>_<facility>.log`. Piping prevents auto-approval of terminal commands, stalling agentic workflows. Read logs directly: `tail -f ~/.local/share/imas-codex/logs/paths_tcv.log`

**Decision tree:**
1. Single command, local → Terminal directly (`rg`, `fd`, `tokei`, `uv run`)
2. Single command, remote → SSH (`ssh facility "command"`)
3. Chained processing → `python()` with `run()` (auto-detects local/remote)
4. Graph queries / MCP → `python()` with `query()`, `add_to_graph()`, etc.

## LLM Prompts

Prompts live in `imas_codex/agentic/prompts/` using Jinja2 templates with schema injection. Never hardcode JSON examples — use Pydantic schema injection via `get_pydantic_schema_json()`. Each prompt declares `schema_needs` in `prompt_loader.py`.

## Exploration

Before disk-intensive operations, **check facility excludes** to avoid repeating known timeouts. When a command times out, **persist the constraint immediately** via `update_infrastructure()` and `add_exploration_note()`.

| Discovery Type | Destination |
|----------------|-------------|
| Source files, paths, codes, trees | `add_to_graph()` (public graph) |
| Hostnames, IPs, OS, tool versions | `update_infrastructure()` (private YAML) |
| Context for future sessions | `add_exploration_note()` |

**Graph (public):** MDSplus tree names, analysis code names/versions/paths, TDI functions, diagnostic names. **Infrastructure (private):** Hostnames, IPs, NFS mounts, OS versions, tool availability, user directories.

## Ingestion

```bash
uv run imas-codex ingest queue tcv /path/a.py /path/b.py
uv run imas-codex ingest run tcv
uv run imas-codex ingest run tcv --min-score 0.7  # High-priority only
uv run imas-codex ingest status tcv
uv run imas-codex ingest run tcv --retry-failed
```

The pipeline extracts MDSplus tree paths, TDI function calls, IDS references, and import dependencies. Do not fabricate `patterns_matched` metadata.

## Graph Operations

**Schema verification:** Before writing Cypher queries, verify property names against [agents/schema-reference.md](agents/schema-reference.md) or call `get_graph_schema()`. Common pitfall: WikiChunk/CodeChunk text content is stored in the `text` property.

### Neo4j Management

```bash
uv run imas-codex graph start       # Start Neo4j (auto-detects mode)
uv run imas-codex graph stop        # Stop Neo4j
uv run imas-codex graph status      # Check graph status
uv run imas-codex graph shell       # Interactive Cypher
uv run imas-codex graph pull        # Pull latest from GHCR
uv run imas-codex graph push --dev  # Push to GHCR
uv run imas-codex graph backup      # Create neo4j-admin dump backup
uv run imas-codex graph clear       # Clear graph (with auto-backup)
uv run imas-codex tunnel start iter # Start SSH tunnel to remote host
```

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Neo4j Lock Files — CRITICAL

| Lock File | Safe to Delete? |
|-----------|----------------|
| `store_lock`, `database_lock` | Yes — after confirming Neo4j is stopped |
| `write.lock` (Lucene index) | **NEVER** — deletion corrupts vector indexes |

**Never use `find -name "*.lock"` to clean locks** — this matches Lucene `write.lock` files. Only remove `store_lock` and `database_lock` explicitly by path. On GPFS/NFS, use inode replacement (`cp file file.unlock && mv -f file.unlock file`), not deletion.

**Never use the Docker entrypoint** in Apptainer. Always use `neo4j console` directly with a host-side `conf/` bind mount.

### Key Relationships

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | MAPS_TO_IMAS | IMASPath |
| WikiChunk | HAS_CHUNK← | WikiPage |
| FacilityPath | AT_FACILITY | Facility |

**Token cost:** Always project specific properties in Cypher (`RETURN n.id, n.name`), never return full nodes.

### Release Workflow

```bash
uv run imas-codex release v4.0.0 -m 'Release message' --remote origin  # Prepare PR
uv run imas-codex release v4.0.0 -m 'Release message'                  # Finalize after merge
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

Install on any facility: `uv run imas-codex tools install <facility>`. **Critical:** `fd` requires a path argument on large filesystems to avoid hanging: `fd -e py /path`

### Two-Interpreter Architecture

| Executor | Interpreter | Min Python | When Used |
|----------|-------------|------------|----------|
| `run_python_script()` / `async_run_python_script()` | Venv `python3` via `_REMOTE_PATH_PREFIX` | 3.12+ | Individual script calls, MDSplus enumeration, TDI extraction |
| `SSHWorkerPool` / `pooled_run_python_script()` | `/usr/bin/python3` (hardcoded) | 3.9+ | Batch discovery operations (scan, enrich, signal check) |

Pool scripts **must be Python 3.9+ compatible** with stdlib-only imports — do **not** use 3.10+ syntax (`match`, `X | Y` type unions). Ruff skips type-hint modernization for `imas_codex/remote/scripts/*`.

**Remote zombie prevention:** All remote SSH commands are wrapped with `timeout <seconds>` server-side. Never bypass this by constructing raw SSH commands — always use the executor functions.

## Commit Workflow

```bash
uv run ruff check --fix .  # Lint (Python only)
uv run ruff format .       # Format
git add <file1> <file2>    # Stage specific files (never git add -A)
uv run git commit -m "type: concise summary"
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

**CRITICAL: Never commit auto-generated files** (models.py, dd_models.py, physics_domain.py) or `*_private.yaml` files.

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

### Parallel Agents

Multiple agents may be working simultaneously. **Only stage files you modified** — never `git add -A` or `git add .`. **NEVER** run `git checkout`, `git restore`, or `git reset` on files you didn't change — this destroys another agent's work. Never rebase. Pull before push if rejected: `git pull --no-rebase origin main`.

### Session Completion

**MANDATORY** after any file modifications: commit and push before responding to the user.

## Code Style

- Python ≥3.12: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`
- Exception chaining: `raise Error("msg") from e`
- `pydantic` for schemas, `dataclasses` for other data classes
- `anyio` for async
- `uv run` for all Python commands (never activate venv manually)
- Never use `git add -A`
- The `.env` file contains secrets — never expose or commit it

## Testing

```bash
uv run pytest                 # All tests
uv run pytest --cov=imas_codex  # With coverage
uv run pytest tests/path/to/test.py::test_function  # Specific test
```

## Quick Reference

**Primary MCP tools** — use these first, they return formatted reports:

| Task | MCP Tool |
|------|----------|
| Signal lookup | `search_signals("plasma current", facility="tcv")` |
| Documentation | `search_docs("fishbone instabilities", facility="jet")` |
| Code examples | `search_code("equilibrium reconstruction", facility="tcv")` |
| IMAS DD paths | `search_imas("electron temperature", facility="tcv")` |

**python() REPL** — for custom queries not covered by the search tools:

| Task | Command |
|------|---------|
| Wiki keyword | `python("print(find_wiki(text_contains='fishbone'))")` |
| Page chunks | `python("print(wiki_page_chunks('equilibrium', facility='tcv'))")` |
| Signal→IMAS map | `python("print(map_signals_to_imas(facility='tcv', physics_domain='magnetics'))")` |
| Graph search | `python("print(graph_search('WikiChunk', where={'text__contains': 'IMAS'}))")` |
| Facility info | `python("print(get_facility('tcv'))")` |
| Raw Cypher | `python("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Update infra | `update_facility_infrastructure('tcv', {...})` |
| Remote command | `ssh facility "rg pattern /path"` |
| Full REPL API | `python("repl_help()")` |

## Embedding Server

```bash
imas-codex embed start     # Start per config (SLURM or systemd)
imas-codex embed status    # Check server health + SLURM jobs
imas-codex embed restart   # Restart
imas-codex embed stop      # Stop all embed processes
```

If embedding fails, check: tunnel active (`lsof -i :18765`), service running (`ssh iter "systemctl --user status imas-codex-embed"`), health (`curl http://localhost:18765/health`).

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
