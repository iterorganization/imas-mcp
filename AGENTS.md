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

### Schema Design Guidelines

Follow these conventions when adding new classes, properties, or relationships to LinkML schemas. Consistency here is critical — the build pipeline, `create_nodes()`, and query builder all depend on predictable schema structure.

#### Dual Property + Relationship Model

Every slot that references another class produces **both** a node property AND a Neo4j relationship. This is intentional — it supports multiple search and traversal patterns:

- **Property** (`n.facility_id = 'tcv'`): Fast `WHERE` filtering without relationship traversal. Enables simple queries and aggregation grouping.
- **Relationship** (`(n)-[:AT_FACILITY]->(f:Facility)`): Graph traversal, multi-hop queries, path-finding. Enables joining across node types.

`create_nodes()` in `client.py` implements this: `SET n += item` stores all properties on the node first, then for each slot with a class range it creates relationships via `MERGE (n)-[:REL_TYPE]->(t:TargetClass {id: item.slot_name})`.

**Never remove one side of the dual model.** Both the property and the relationship must exist for every class-ranged slot.

#### Relationship Type Annotations

When a slot has `range: SomeClass`, the Cypher relationship type is derived as follows:

1. **Explicit annotation (preferred for clarity)**:
   ```yaml
   facility_id:
     range: Facility
     annotations:
       relationship_type: AT_FACILITY
   ```
   Use explicit annotations when the auto-derived name would be unclear (e.g., `FACILITY_ID` is less readable than `AT_FACILITY`).

2. **Auto-derived fallback**: If no `relationship_type` annotation, the slot name is uppercased: `signal` → `SIGNAL`, `data_access` → `DATA_ACCESS`, `has_chunk` → `HAS_CHUNK`.

**Rules for new relationships:**
- Use `relationship_type: AT_FACILITY` for all `facility_id` slots — this is the standard pattern across the entire schema.
- Prefer verb-based names: `MAPS_TO_IMAS`, `BELONGS_TO_DIAGNOSTIC`, `DOCUMENTED_BY`.
- If the auto-derived name is clear enough (e.g., `has_chunk` → `HAS_CHUNK`), omit the annotation.
- All `facility_id` slots MUST have `range: Facility` and `annotations: { relationship_type: AT_FACILITY }`. No exceptions.

#### Class Structure Template

Every concrete class should follow this structure:

```yaml
MyNewNode:
  description: >-
    What this node represents. Include example Cypher queries
    that agents would use to query this node type.
  class_uri: facility:MyNewNode
  attributes:
    id:
      identifier: true
      description: Composite key format (e.g., "facility:unique_part")
      required: true
    facility_id:
      description: Parent facility ID
      required: true
      range: Facility
      annotations:
        relationship_type: AT_FACILITY
    # ... domain-specific properties ...
    status:
      description: Lifecycle status
      range: MyNewNodeStatus  # Define enum in same schema
      required: true
    description:
      description: Human-readable description
    embedding:
      description: Vector embedding of description for semantic search
      multivalued: true
      range: float
    embedded_at:
      description: When the embedding was last computed
      range: datetime
```

#### ID Conventions

- Use `identifier: true` on exactly one slot per class (always `id` unless there's a domain reason).
- Composite IDs use colon separator: `facility_id:unique_part` (e.g., `"tcv:/home/codes/liuqe.py"`, `"tcv:ip/measured"`).
- IDs must be globally unique across all facilities.

#### Vector Indexes

Nodes with `embedding` + `description` slots automatically get a vector index named `{snake_case_label}_desc_embedding` (e.g., `FacilitySignal` → `facility_signal_desc_embedding`).

For non-standard embedding slots, use the `vector_index_name` annotation:

```yaml
embedding:
  multivalued: true
  range: float
  annotations:
    vector_index_name: cluster_embedding
```

The build pipeline validates all vector indexes and generates them into `schema_context_data.py`.

#### Status Enums and Lifecycles

Define status enums in the same schema file as the class. Statuses must represent **durable states only** — no transient states like `scanning` or `processing`. Worker coordination uses `claimed_at` timestamps, not status values.

```yaml
enums:
  MyNewNodeStatus:
    permissible_values:
      discovered:
        description: Initial state after creation
      processed:
        description: Successfully processed
      failed:
        description: Processing failed
      stale:
        description: May have changed, needs re-processing
```

#### Private Fields

Slots annotated with `is_private: true` are excluded from the graph — they exist only in facility YAML configs:

```yaml
ssh_host:
  description: SSH host alias
  annotations:
    is_private: true
```

#### What NOT to Do in Schemas

- **Don't hardcode enum values in Python** — import from generated models.
- **Don't create a `facility_id` slot as plain `string`** — always use `range: Facility` + `relationship_type: AT_FACILITY` so both the property and relationship are created.
- **Don't add transient states to status enums** — use `claimed_at` for worker coordination.
- **Don't define the same relationship type with different semantics** — `AT_FACILITY` always means "belongs to this facility".
- **Don't skip the `description` field** — it enables semantic search via embeddings.
- **Don't use `multivalued: true` on relationship slots** unless the relationship is genuinely many-to-many. Cardinality affects query patterns.

## Facility Configuration

Per-facility YAML configs define discovery roots, wiki sites, data sources, and infrastructure details. Schema enforced via LinkML (`imas_codex/schemas/facility_config.yaml`).

**Files:**
- `imas_codex/config/facilities/<facility>.yaml` - Public config (git-tracked)
- `imas_codex/config/facilities/<facility>_private.yaml` - Private config (gitignored)

**CRITICAL: All facility-specific configuration MUST live in YAML files.** Never hardcode facility names, tree names, version numbers, setup commands, system descriptions, or any other facility-specific values in Python code. Scripts and CLI commands must be fully generic — they load all configuration from the facility YAML at runtime via `get_facility(facility)`.

**What goes in public facility YAML** (`<facility>.yaml`):
- `discovery_roots` — paths to scan for code/data
- `data_sources.tdi.*` — TDI function directories, reference shots, exclude lists
- `data_sources.mdsplus.*` — tree names, subtrees, node usages, setup commands
- `data_sources.mdsplus.static_trees` — static tree versions, first_shot, descriptions, systems
- `data_access_patterns` — primary method, naming conventions, key tools
- `wiki_sites` — wiki URLs for scraping

**What goes in private facility YAML** (`<facility>_private.yaml`, gitignored):
- Hostnames, IPs, NFS mount points
- OS versions, kernel info
- Login node names, local host overrides
- User-specific paths, tool locations

**How to load config in Python:**

```python
from imas_codex.discovery.base.facility import get_facility

config = get_facility(facility)  # Loads <facility>.yaml + <facility>_private.yaml
mdsplus = config.get("data_sources", {}).get("mdsplus", {})
setup_commands = mdsplus.get("setup_commands", [])
static_trees = mdsplus.get("static_trees", [])
```

**When adding a new discovery pipeline or data source**, add the required config fields to the facility YAML schema (`imas_codex/schemas/facility_config.yaml`) and load them via `get_facility()`. The Python code should work unchanged across all facilities — only the YAML differs.

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

**Schema verification:** Before writing Cypher queries, verify property names against `agents/schema-reference.md` (auto-generated) or call `get_graph_schema()`. Common pitfall: WikiChunk/CodeChunk text content is stored in the `text` property.\n\n### Neo4j Management

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

**Remote Python — Two-interpreter architecture:**

| Executor | Interpreter | Min Python | When Used |
|----------|-------------|------------|----------|
| `run_python_script()` / `async_run_python_script()` | Venv `python3` via `_REMOTE_PATH_PREFIX` | 3.12+ | Individual script calls, MDSplus enumeration, TDI extraction |
| `SSHWorkerPool` / `pooled_run_python_script()` | `/usr/bin/python3` (hardcoded) | 3.9+ | Batch discovery operations (scan, enrich, signal check) |

- **Venv path**: Scripts dispatched via `run_python_script()` get the venv Python (3.12+) because `_REMOTE_PATH_PREFIX` puts `~/.local/share/imas-codex/venv/bin` first in PATH. These scripts may use modern syntax (`X | Y` unions, `match`, `isinstance(x, int | float)`).
- **System path**: The `SSHWorkerPool` hardcodes `/usr/bin/python3` to avoid 60-100s NFS venv startup penalty. Scripts dispatched through the pool `exec()` inside system Python and **must be Python 3.9+ compatible** with stdlib-only imports. Do **not** use 3.10+ syntax (`match`, `X | Y` type unions) in these scripts.
- If a venv-path script fails with a syntax error, verify the venv: `uv run imas-codex tools status <facility>`.
- Remote scripts declare their Python version in a docstring header (`Python 3.8+` or `Python 3.12+`). Always check before adding modern syntax.
- Ruff skips type-hint modernization for `imas_codex/remote/scripts/*` — see `pyproject.toml` per-file ignores.

**Remote zombie prevention:** All remote SSH commands are wrapped with `timeout <seconds>` on the server side. When a local `subprocess.run()` times out, it kills the SSH client process but the remote process keeps running indefinitely as a zombie. The server-side `timeout` (set to local timeout + 5s) ensures the remote process self-terminates independently. This is enforced in `executor.py` for `run_command()`, `run_script_via_stdin()`, `run_python_script()`, and `async_run_python_script()`. Never bypass this by constructing raw SSH commands — always use the executor functions.

## Commit Workflow

```bash
# 1. Lint and format (Python files only — ruff does not support other formats)
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

The `python()` MCP tool provides a persistent REPL with pre-loaded utilities. The tool description includes exact function signatures — use them directly without calling `repl_help()` first.

### REPL Workflow

1. **Use domain query functions** (`find_wiki`, `find_signals`, `graph_search`, etc.) instead of raw Cypher. They handle embeddings, schema validation, and relationship traversal internally. Never write raw Cypher when a domain function covers your task.
2. **Chain operations** in a single `python()` call to minimize round-trips. Each call has overhead.
3. **For raw Cypher** (only when no domain function fits), call `schema_for(task='wiki')` first to get node labels, properties, relationships, and enums derived from the LinkML schemas. Never guess property names — they are code-generated.
4. **Format output** with `as_table(pick(results, 'col1', 'col2'))` for structured results.

### Schema-First Queries

All graph node types, properties, enums, and relationships are derived from LinkML schemas. The REPL exposes this via:

- `schema_for(task='signals')` — schema context for a domain (signals, wiki, imas, code, facility, trees)
- `schema_for('WikiChunk', 'WikiPage')` — schema for specific node labels
- `get_schema()` — full `GraphSchema` object with `node_labels`, `get_model()`, `get_properties()`
- `repl_help()` — auto-generated API reference with all function signatures

**Never hardcode property names.** Before writing raw Cypher, verify against the schema:

```python
python('''
# Use domain functions — not raw Cypher
results = find_wiki('fishbone instabilities', facility='jet')
chunks = wiki_page_chunks('fishbone', facility='jet', text_contains='ICRH')
signals = find_signals('fishbone', facility='jet')
print(as_table(pick(results, 'page_title', 'section', 'score')))
print(as_table(pick(chunks, 'page_title', 'section', 'text')))
print(as_table(pick(signals, 'id', 'description', 'score')))
''')
```

For queries not covered by domain functions, use `schema_for()` first:

```python
python('''
print(schema_for(task="wiki"))
# Then write Cypher using verified property names
''')
```

### Discovery

```python
python("repl_help()")                # Full API with signatures
python("help(find_wiki)")            # Detailed docstring for one function
python("print(schema_for())")        # Graph schema overview
```

## Quick Reference

| Task | Command |
|------|---------|
| Wiki search | `python("print(find_wiki('plasma control', facility='jet'))")` |
| Wiki keyword | `python("print(find_wiki(text_contains='fishbone'))")` |
| Page chunks | `python("print(wiki_page_chunks('equilibrium', facility='tcv'))")` |
| Signal search | `python("print(find_signals('electron density', facility='tcv'))")` |
| IMAS search | `python("print(search_imas('electron temperature'))")` |
| Code search | `python("print(find_code('equilibrium', facility='tcv'))")` |
| Graph search | `python("print(graph_search('WikiChunk', where={'text__contains': 'IMAS'}))")` |
| Format table | `python("print(as_table(find_signals('ip', facility='tcv')))")` |
| Facility info | `python("print(get_facility('tcv'))")` |
| Raw Cypher | `python("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Update infra | `update_facility_infrastructure('tcv', {...})` |
| Remote command | `ssh facility "rg pattern /path"` |

Chain multiple operations in a single `python()` call to minimize round-trips:

```python
python('''
signals = find_signals("electron density", facility="tcv")
mapped = map_signals_to_imas(facility="tcv", physics_domain="magnetics")
wiki = find_wiki("equilibrium reconstruction", facility="jet", k=10)
print(as_table(pick(signals, "id", "description", "score")))
print(f"\n{len(mapped)} mapped signals")
print(as_table(pick(wiki, "page_title", "section", "score")))
''')
```

## Embedding Server

Config lives in `pyproject.toml` under `[tool.imas-codex.embedding]`. Key accessor: `get_embedding_location()` returns the facility name or `"local"`. Port derived from position in shared `locations` list: `18765 + offset`.

```bash
imas-codex embed start           # Start per config (SLURM or systemd)
imas-codex embed start -g 2      # Start with 2 GPUs
imas-codex embed start -f        # Run in foreground (debugging)
imas-codex embed status          # Check server health + SLURM jobs
imas-codex embed restart -g 8    # Restart with 8 GPUs (~18s cycle)
imas-codex embed stop            # Stop all embed processes
imas-codex embed logs            # View SLURM logs
imas-codex embed service install # Install systemd service
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
