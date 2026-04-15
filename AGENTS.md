# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), MCP `repl()` for chained processing and graph queries, `uv run` for git/tests/CLI. Conventional commits. **CRITICAL: Always commit and push when files have been modified — no confirmation, no asking, just do it. This is non-negotiable. Every response that modifies files MUST end with `git add`, `git commit`, and `git push`.** **Never use `vscode_askQuestions` or any interactive VS Code popup/dialog tools — present all questions inline in the chat response so the user can answer them in one message.**

**Git sync discipline (fork-based workflow):** All development happens on the fork's `main` branch. Always **merge** on pull — never rebase. Never use feature branches (`develop`, `feature/*`) — they add merge overhead and break the release CLI which requires `main`.
1. **Session start:** `git pull origin main` before any work.
2. **Before push:** `git pull origin main && git push origin main` — never push without pulling first. Push to `origin` (fork), **never directly to `upstream`**.
3. **Always work on `main`** — the release CLI requires `main` branch. Never create or switch to feature branches without explicit user approval.
4. **Dirty worktree:** Commit or stash your own files before pulling. Never stash everything (`git stash`) — only your files: `git stash push -- file1 file2`.
5. **Conflict resolution:** If merge conflicts, resolve and commit. Never force-push without user approval.
6. **Repo-local config:** Each clone must run the setup commands below to override any global/system rebase defaults.

### New Clone Setup

Run these commands once after cloning on any machine. They are stored in `.git/config` (local scope) and override global/system settings that vary across installations:

```bash
git config --local pull.rebase false      # merge on pull, never rebase
git config --local rebase.autoStash false  # don't silently stash — make dirty worktree visible
git config --local merge.ff true           # allow fast-forward merges
```

**Why this matters:** Different machines (WSL, ITER, TCV) may have different global git configs. An ITER system policy might set `pull.rebase=true` globally, which silently converts `git pull` into a rebase. Rebase fails with dirty worktrees (auto-generated files from `uv sync`) and rewrites history that other agents depend on. Local config takes precedence over global/system, ensuring consistent behavior everywhere.

## Project Philosophy

Greenfield project under active development. No backwards compatibility.

- Breaking changes are expected - remove deprecated code decisively
- Avoid "enhanced", "new", "refactored" in names - just use the good name
- When patterns change, update all usages - don't leave old patterns alongside new
- **Stale context kills** — if your session is more than a few hours old, your memory of file contents may be wrong. Before modifying any file, re-read it from disk (`view` / `cat`) to verify your assumptions match reality. Never write code from memory alone.
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
| `[reasoning]` | Complex structured output (IMAS mapping, multi-step reasoning) | `get_model("reasoning")` |
| `[discovery]` | Discovery threshold for high-value processing | `get_discovery_threshold()` |
| `[data-dictionary]` | DD version, include-ggd, include-error-fields | `get_dd_version()` |
| `[sn.benchmark]` | SN benchmark compose-models list and reviewer-model | `get_sn_benchmark_compose_models()`, `get_sn_benchmark_reviewer_model()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"reasoning"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

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
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, SignalNode, CodeChunk, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASNode, DDVersion, Unit, IMASCoordinateSpec
- `imas_codex/schemas/common.yaml` - Shared: status enums

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- Output: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`, `agents/schema-reference.md`, `imas_codex/graph/schema_context_data.py`

**CRITICAL: Never commit auto-generated files.** These are gitignored and rebuilt on `uv sync`. If `git status` shows a generated model file as untracked or modified, do NOT stage it. Generated files:
- `imas_codex/graph/models.py`
- `imas_codex/graph/dd_models.py`
- `imas_codex/config/models.py`
- `agents/schema-reference.md`
- `imas_codex/graph/schema_context_data.py`

**PhysicsDomain enum**: Imported from the `imas-standard-names` PyPI package and re-exported from `imas_codex.core.physics_domain`. The canonical vocabulary is maintained in the imas-standard-names project. Contains 32 physics domain values. `imas_codex/core/physics_domain.py` is a hand-written one-line re-export — it IS committed and should NOT be treated as auto-generated.

Always import enums and classes from generated models. Never hardcode status values:

```python
from imas_codex.graph.models import SourceFile, SourceFileStatus, SignalNode

sf = SourceFile(
    id="tcv:/home/codes/liuqe.py",
    facility_id="tcv",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])
```

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Prefer additive changes, but renames and removals are fine when they improve consistency — the schema must stay clean. When renaming or removing: update all code references, migrate graph data, and rebuild models in a single commit.

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
- Prefer verb-based names: `SOURCE_PATH`, `TARGET_PATH`, `BELONGS_TO_DIAGNOSTIC`, `DOCUMENTED_BY`.
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

#### Private Fields

Slots annotated with `is_private: true` are excluded from the graph — they exist only in facility YAML configs.

#### What NOT to Do in Schemas

- **Don't hardcode enum values in Python** — import from generated models.
- **Don't create a `facility_id` slot as plain `string`** — always use `range: Facility` + `relationship_type: AT_FACILITY` so both the property and relationship are created.
- **Don't add transient states to status enums** — use `claimed_at` for worker coordination.
- **Don't define the same relationship type with different semantics** — `AT_FACILITY` always means "belongs to this facility".
- **Don't skip the `description` field** — it enables semantic search via embeddings.
- **Don't use `multivalued: true` on relationship slots** unless the relationship is genuinely many-to-many. Cardinality affects query patterns.

### Schema-Driven Testing

Tests in `tests/graph/` are **parametrized from the schema** — they do not hardcode node labels, relationship types, or enum values. This creates a closed loop:

1. Declare types, relationships, and enums in LinkML YAML
2. `uv run build-models --force` generates models + schema context
3. Code writes data to the graph using generated models
4. Schema-driven tests validate **all** graph data against schema declarations

**Key test modules:**
- `test_schema_compliance.py` — every node label, property, and enum value must be declared in the schema
- `test_referential_integrity.py` — every relationship type must be declared as a slot with the correct `relationship_type` annotation
- `test_data_quality.py` — embedding coverage and data consistency checks

**When a schema compliance test fails, investigate root cause before touching the schema.** A failing test means data in the graph doesn't match schema declarations. The correct response depends on *why*:

1. **You are building a new capability** that genuinely requires a new relationship type, enum value, or node label → declare it in the LinkML YAML, rebuild models, then write the code that uses it. Schema first, code second.
2. **Existing code is writing non-compliant data** (bug) → fix the code that produces the bad data, or fix the data directly in the graph. Do not expand the schema to accommodate a bug.
3. **Stale data from a previous schema version** → migrate or remove the data. Do not re-add removed schema elements to pass tests.

Do not add schema declarations solely to make tests green. The schema defines what *should* exist — tests verify the graph matches that intent.

## Facility Configuration

Per-facility YAML configs define discovery roots, wiki sites, data sources, and infrastructure details. Schema enforced via LinkML (`imas_codex/schemas/facility_config.yaml`).

**Files:**
- `imas_codex/config/facilities/<facility>.yaml` - Public config (git-tracked)
- `imas_codex/config/facilities/<facility>_private.yaml` - Private config (gitignored)

**CRITICAL: All facility-specific configuration MUST live in YAML files.** Never hardcode facility names, tree names, version numbers, setup commands, system descriptions, or any other facility-specific values in Python code. Scripts and CLI commands must be fully generic — they load all configuration from the facility YAML at runtime via `get_facility(facility)`.

**What goes in public facility YAML** (`<facility>.yaml`):
- `discovery_roots` — paths to scan for code/data
- `data_systems.tdi.*` — TDI function directories, reference shots, exclude lists
- `data_systems.mdsplus.*` — tree names, subtrees, node usages, setup commands
- `data_systems.mdsplus.static_trees` — static tree versions, first_shot, descriptions, systems
- `data_access_patterns` — primary method, naming conventions, key tools
- `wiki_sites` — wiki URLs for scraping

**What goes in private facility YAML** (`<facility>_private.yaml`, gitignored):
- Hostnames, IPs, NFS mount points
- OS versions, kernel info
- Login node names, local host overrides
- User-specific paths, tool locations

**How to load config:** `get_facility(facility)` from `imas_codex.discovery.base.facility` loads both public + private YAML and returns a dict.

**When adding a new discovery pipeline or data source**, add the required config fields to the facility YAML schema (`imas_codex/schemas/facility_config.yaml`) and load them via `get_facility()`. The Python code should work unchanged across all facilities — only the YAML differs.

**Editing configs:** Always use MCP tools rather than direct file editing:

```python
# Update public facility config (wiki sites, discovery roots, data systems)
update_facility_config('tcv', {'discovery_roots': ['/new/path']})

# For infrastructure notes, use the repl tool directly
repl("update_infrastructure('tcv', {'exploration_notes': ['Found equilibrium codes at /home/codes/liuqe']})")
```

**Validation:** `validate_facility_config('tcv')` returns a list of error strings. The config schema is also exposed via the `get_graph_schema()` MCP tool.

## Graph State Machine

Status enums represent **durable states only**. No transient states like `scanning`, `scoring`, or `ingesting`.

**Worker coordination:** Claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`. Orphan recovery is automatic via timeout check in claim queries.

### Claim Patterns — Deadlock Avoidance

All claim functions **must** use three anti-deadlock patterns. Reference implementations: `discovery/wiki/graph_ops.py`, `discovery/code/graph_ops.py`. Shared infrastructure: `discovery/base/claims.py`.

1. **`@retry_on_deadlock()`** — decorator from `claims.py`. Retries on `TransientError` with exponential backoff + jitter. Apply to every function that writes `claimed_at`.
2. **`ORDER BY rand()`** — randomize lock acquisition order. Deterministic ordering (`ORDER BY v.version ASC`, `ORDER BY score DESC`) causes lock convoys where concurrent workers deadlock on the same rows.
3. **`claim_token` two-step verify** — SET a UUID token in step 1, then read back by token in step 2. Prevents double-claiming race conditions.

```python
from imas_codex.discovery.base.claims import retry_on_deadlock

@retry_on_deadlock()
def claim_items(facility: str, limit: int = 10) -> list[dict]:
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (n:MyNode {facility_id: $facility})
            WHERE n.status = 'discovered' AND n.claimed_at IS NULL
            WITH n ORDER BY rand() LIMIT $limit
            SET n.claimed_at = datetime(), n.claim_token = $token
        """, facility=facility, limit=limit, token=token)
        return list(gc.query("""
            MATCH (n:MyNode {claim_token: $token})
            RETURN n.id AS id, n.path AS path
        """, token=token))
```

**Never** use deterministic `ORDER BY` in claim queries. **Never** write a manual retry loop for deadlocks — use `@retry_on_deadlock()`. See `imas_codex/discovery/README.md` for detailed rationale.

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

## Compute Infrastructure

**Never compile or run compute-intensive tasks on login nodes.** Login nodes are shared — CPU-intensive builds, test suites, and long-running processes cause severe performance degradation for all users.

When on an HPC system with SLURM (check `which srun`), use debug/interactive partitions for builds and batch partitions for longer workloads. If `$SLURM_JOB_ID` is set, you're already on a compute node — run directly. Use `-march=x86-64-v3` instead of `-march=native` for portable binaries. Check `~/.agents/skills/` for site-specific partition names, modules, and resource templates.

## Command Execution

**CRITICAL: Always use `uv run` for project Python code.** This project manages dependencies (including `imas`) via `uv`. Running `python` or `python -m pytest` directly will miss project dependencies and fail with `ModuleNotFoundError`. Always use `uv run python`, `uv run pytest`, `uv run imas-codex`, etc.

**CRITICAL: Never pipe, tee, or redirect CLI output.** All `imas-codex` CLI commands auto-log full DEBUG output to `~/.local/share/imas-codex/logs/<command>_<facility>.log`. Piping (`|`), teeing (`tee`), or redirecting (`>`, `2>&1`) to files prevents auto-approval of terminal commands, stalling agentic workflows. Run commands directly and read the log file afterwards.

**Decision tree:**
1. Single command, local → Terminal directly (`rg`, `fd`, `tokei`, `uv run`)
2. Single command, remote → SSH (`ssh facility "command"`)
3. Chained processing → `repl()` with `run()` (auto-detects local/remote)
4. Graph queries / MCP → `repl()` with `query()`, `add_to_graph()`, etc.

**MCP tool routing:**
- Dedicated MCP tools for single operations: `add_to_graph()`, `get_graph_schema()`, `update_facility_config()`
- `repl()` REPL for chained processing, Cypher queries, IMAS/COCOS operations
- Terminal for `rg`, `fd`, `git`, `uv run`; SSH for remote single commands

**Read-only mode:** `imas-codex serve --read-only` suppresses all write tools (`repl()` REPL, `add_to_graph()`, `update_facility_config()`) and exposes only the search/read tools (`search_signals`, `search_docs`, `search_code`, `search_dd_paths`, `fetch_content`, `get_graph_schema`, etc.). Use for any context where graph mutation is not desired.

**DD-only mode:** `imas-codex serve --dd-only` hides facility-specific tools and **implies `--read-only`**. Use for container deployments with a DD-only graph. Auto-detected from graph content if omitted.

```bash
# Full mode (default) — all tools including REPL and write operations
imas-codex serve

# Read-only mode — search and read tools only, no REPL or graph writes
imas-codex serve --read-only

# DD-only mode — DD tools only, implies read-only (typical container deployment)
imas-codex serve --dd-only --transport streamable-http
```

## LLM Access

All LLM interaction flows through two canonical modules. Never call `litellm.completion()` directly — the shared functions handle prompt caching flags, cost tracking, retries with exponential backoff, and structured output parsing.

### Calling LLMs

Use `call_llm_structured()` / `acall_llm_structured()` from `imas_codex.discovery.base.llm`:

```python
from imas_codex.discovery.base.llm import call_llm_structured

result, cost, tokens = call_llm_structured(
    model=get_model("language"),
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_model=MyPydanticModel,
)
```

These functions automatically: apply `inject_cache_control()` to system messages, retry on API/parse errors with backoff, accumulate cost across retries, and parse structured output via Pydantic `response_format`.

### Rendering Prompts

Use `render_prompt()` from `imas_codex.llm.prompt_loader` — never construct paths to prompt files manually:

```python
from imas_codex.llm.prompt_loader import render_prompt

system_prompt = render_prompt("paths/scorer", {"facility": "tcv", "batch": batch_data})
```

For path access (e.g., in tests), import `PROMPTS_DIR` from the same module — never hardcode path segments like `"llm" / "prompts"`.

### Rules

- Model identifiers require the `openrouter/` prefix to preserve `cache_control` blocks
- Use `get_model(section)` from `imas_codex.settings` for model selection — never hardcode model names
- Pydantic schema injection via `get_pydantic_schema_json()` — never hardcode JSON examples in prompts
- Each prompt declares `schema_needs` in frontmatter to load only required schema context

### Prompt Structure and Caching

All LLM calls route through the LiteLLM proxy → OpenRouter. Use `call_llm_structured()` / `acall_llm_structured()` from `imas_codex.discovery.base.llm` for all structured output calls — never call `litellm.completion()` directly.

All prompts follow a **static-first ordering** to maximize prompt cache hit rates via OpenRouter's prompt caching:

1. **System prompt** (static/quasi-static): Schema definitions, enum values, classification rules, output format. These change rarely and are shared across all LLM calls of the same type. `inject_cache_control()` sets a `cache_control: {"type": "ephemeral"}` breakpoint at the end of the system message.
2. **User prompt** (dynamic): Per-batch signal data, context chunks, and specific instructions. This varies per LLM call.

The `openrouter/` prefix is required on model identifiers — it preserves `cache_control` blocks in message content. The `openai/` prefix strips them, silently disabling prompt caching.

When building prompts, ensure that `{% include %}` blocks for schema definitions and static rules appear **before** dynamic Jinja2 template variables. This maximizes the cacheable prefix length.

## Exploration

Before disk-intensive operations, **check facility excludes** to avoid repeating known timeouts:

```python
info = get_facility('tcv')
excludes = info.get('excludes', {})
print(excludes.get('large_dirs', []))
print(excludes.get('depth_limits', {}))
print(info.get('exploration_notes', [])[-3:])
```

When a command times out, **persist the constraint immediately** via `update_infrastructure()` in the repl. Never repeat a timeout.

### Persistence

| Discovery Type | Destination |
|----------------|-------------|
| Source files, paths, codes, trees | `add_to_graph()` (public graph) |
| Public facility config, wiki sites, discovery roots | `update_facility_config()` |
| Infrastructure notes (hostnames, tool versions) | `repl("update_infrastructure('tcv', {...})")` |

### Data Classification

- **Graph (public):** MDSplus tree names, analysis code names/versions/paths, TDI functions, diagnostic names
- **Infrastructure (private):** Hostnames, IPs, NFS mounts, OS versions, tool availability, user directories

## Graph Operations

**Schema verification:** Before writing Cypher queries, verify property names against `agents/schema-reference.md` (auto-generated) or call `get_graph_schema()`. Common pitfall: WikiChunk/CodeChunk text content is stored in the `text` property.

### Cypher Compatibility — Neo4j 2026

We run **Neo4j 2026.01.x** with `db.query.default_language: CYPHER_5`. Most Cypher syntax works normally, with one critical exception:

```cypher
-- WRONG (syntax error in Cypher 5):
WHERE n.type NOT IN ['A', 'B']

-- RIGHT:
WHERE NOT (n.type IN ['A', 'B'])
```

**`x NOT IN [list]` is removed in Cypher 5.** Always use `NOT (x IN [list])` instead. This is the **only** breaking syntax change that affects this codebase.

**`CASE WHEN` works fine** — use it freely for conditional logic, counting, SET, ORDER BY, FOREACH, etc. Do not replace `CASE WHEN` with `nullIf()` hacks unless there is a measured performance benefit.

**Preferred patterns for conditional SET (update-if-non-empty):**
```cypher
-- Use coalesce/nullIf for "keep old value if new is empty" — cleaner than CASE WHEN:
SET s.diagnostic = coalesce(nullIf(sig.diagnostic, ''), s.diagnostic)
-- Instead of:
SET s.diagnostic = CASE WHEN sig.diagnostic <> '' THEN sig.diagnostic ELSE s.diagnostic END
```

**Rules:**
- Never use `x NOT IN [...]` — syntax error. Use `NOT (x IN [...])`
- `CASE WHEN` is supported — do not gratuitously replace it
- `coalesce(nullIf(new_val, ''), old_val)` is preferred for conditional SET updates
- Test new Cypher against the live graph before committing

### Neo4j Management

```bash
uv run imas-codex graph start                 # Start Neo4j (auto-detects mode)
uv run imas-codex graph stop                  # Stop Neo4j
uv run imas-codex graph status                # Check graph status
uv run imas-codex graph profiles              # List all profiles and ports
uv run imas-codex graph shell                 # Interactive Cypher (active profile)
uv run imas-codex graph export               # Export graph to archive (add -f <facility> for filtered)
uv run imas-codex graph load graph.tar.gz    # Load graph archive
uv run imas-codex graph pull                 # Pull latest from GHCR (add --facility for per-facility)
uv run imas-codex graph push --dev           # Push to GHCR (add --facility for per-facility)
uv run imas-codex graph backup               # Create neo4j-admin dump backup
uv run imas-codex graph restore              # Restore from backup
uv run imas-codex graph clear                # Clear graph (with auto-backup)
uv run imas-codex graph prune --dev          # Remove all dev GHCR tags (or --backups --older-than 30d)
uv run imas-codex tunnel start iter          # Start SSH tunnel to remote host
uv run imas-codex tunnel status              # Show active tunnels
uv run imas-codex config private push        # Push private YAML to Gist
uv run imas-codex config secrets push iter   # Push .env to remote host
```

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Graph Migrations

**Run migrations as inline Cypher, never as scripts.** Migrations are one-off operations — do not create `scripts/migrate_*.py` or `scripts/repair_*.py` files. Instead, run the migration Cypher directly via `uv run imas-codex graph shell` or the MCP `repl()` REPL with `query()`. This keeps the `scripts/` directory clean for reusable tooling only.

```python
# Example: backfill a new property on existing nodes
query("""
    MATCH (cc:CodeChunk)
    WHERE cc.embedding IS NOT NULL AND cc.embedded_at IS NULL
    WITH cc LIMIT 1000
    SET cc.embedded_at = datetime()
    RETURN count(cc) AS updated
""")
```

For large migrations (>10K nodes), batch in a loop with `LIMIT` to avoid transaction timeouts. Always verify counts before and after.

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

Vector indexes use Neo4j 2026.01's native `SEARCH` clause for in-index pre-filtering.
All indexes include quantization for ~4× memory savings:

```python
from imas_codex.settings import get_embedding_dimension

dim = get_embedding_dimension()
gc.query(f"""
    CREATE VECTOR INDEX my_index IF NOT EXISTS
    FOR (n:MyNodeType) ON n.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine',
            `vector.quantization.enabled`: true
        }}
    }}
""")
```

`ensure_vector_indexes()` auto-detects dimension mismatches and drops/recreates stale indexes.

**Existing indexes:** See [agents/schema-reference.md](agents/schema-reference.md) for the full list of vector indexes derived from the LinkML schema.

### Semantic Search & Graph RAG

Use `semantic_search(text, index, k)` in the python REPL:

```python
# Document content (wiki, code)
semantic_search("COCOS sign conventions", index="wiki_chunk_embedding", k=5)

# Descriptive metadata (signals, paths - search by physics meaning)
semantic_search("plasma current measurement", index="facility_signal_desc_embedding", k=10)
```

Combine vector similarity with link traversal using the Cypher 25 SEARCH clause:

```python
results = query("""
    CYPHER 25
    MATCH (signal:FacilitySignal)
    SEARCH signal IN (
      VECTOR INDEX facility_signal_desc_embedding
      FOR $embedding
      LIMIT 5
    ) SCORE AS score
    WHERE signal.facility_id = $facility
    WITH signal, score
    MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)
    OPTIONAL MATCH (signal)-[:HAS_DATA_SOURCE_NODE]->(dn:SignalNode)
        <-[:SOURCE_PATH]-(m:IMASMapping)-[:TARGET_PATH]->(imas:IMASNode)
    RETURN signal.id, signal.description, da.data_template,
           collect(imas.id) AS imas_paths, score
    ORDER BY score DESC
""", embedding=embed("electron density profile"), facility="tcv")
```

Use `build_vector_search()` from `imas_codex.graph.vector_search` to generate
SEARCH clauses programmatically. All WHERE conditions are post-filters (in-index
pre-filtering requires properties registered as additional vector index properties).

**Key relationships for traversal:**

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | HAS_DATA_SOURCE_NODE | SignalNode |
| IMASMapping | SOURCE_PATH | SignalNode |
| IMASMapping | TARGET_PATH | IMASNode |
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

The release CLI is state-machine driven. State is derived from the latest git tag:

- **Stable** (`vX.Y.Z`) — on a release
- **RC mode** (`vX.Y.Z-rcN`) — testing a release candidate

**Remote defaults:** RC releases target `origin` (fork), final releases target `upstream` (iterorganization). Override with `--remote`.

**Dirty worktree policy:** RC releases allow dirty worktrees (warning only) since parallel agents often modify files concurrently. Final releases (`--final`) require a clean worktree — commit or stash first.

```bash
# Check current state and permitted commands
uv run imas-codex release status

# From stable (e.g., v5.0.0):
uv run imas-codex release --bump major -m 'IMAS DD 4.1.0 support'    # → v6.0.0-rc1 (origin)
uv run imas-codex release --bump minor -m 'New discovery features'    # → v5.1.0-rc1 (origin)
uv run imas-codex release --bump patch -m 'Bug fixes'                 # → v5.0.1-rc1 (origin)
uv run imas-codex release --bump major --final -m 'Direct release'    # → v6.0.0 (upstream)

# From RC mode (e.g., v5.0.0-rc1):
uv run imas-codex release -m 'Fix CI issues'                          # → v5.0.0-rc2 (origin)
uv run imas-codex release --final -m 'Production release'             # → v5.0.0 (upstream)
uv run imas-codex release --bump patch -m 'Abandon RC, new patch'     # → v5.0.1-rc1 (origin)

# Options: --remote origin|upstream, --skip-git, --dry-run, --version
```

**What the release CLI does (no manual steps needed):**

1. Computes the next version from the latest git tag (state machine)
2. Creates a local git tag
3. Validates no private fields in graph
4. Tags DDVersion node with release metadata
5. Pushes graph variants to GHCR (dd-only + full for RC; + per-facility for final)
6. Pushes git tag to the target remote → triggers CI

**What CI does (automatically triggered by the tag push):**

1. `graph-quality` — pulls dd-only graph from GHCR, runs `tests/graph/` against it
2. `smoke-test` — builds container, starts it, verifies health endpoint
3. `build-and-push` — pushes container image to Azure Container Registry (ACR)

**How Azure deploys (automatic, no webhook needed):**

Azure Web App has continuous deployment enabled on ACR. When a new image appears, Azure pulls and restarts the container automatically. There is no webhook — deployment is triggered by ACR image push. Allow 5–15 minutes for the new version to appear on the test URL.

**Fork-based development workflow:**

1. **All work on fork's `main`** — no feature branches. Multiple agents use the same `main` branch with merge discipline.
2. **RC releases → fork CI → Azure test** — `imas-codex release -m "..."` pushes graph + tag to origin. Fork CI builds and pushes to ACR (hardcoded `iterorganization/` path). Azure auto-deploys the `latest-rc` tag.
3. **Verify RC** — exercise tools on test deployment at `https://app-imas-mcp-server-test-frc.azurewebsites.net/health`. Run A/B tests against all MCP tools.
4. **PR to upstream** — when RC is confirmed working, PR fork/main → upstream/main.
5. **Final release → upstream** — after PR merges: `imas-codex release --final -m "..."` tags upstream, production CI deploys with `latest-stable` tag.

**Rules:**
- **Never push directly to `upstream/main`** — always PR. Use `git push origin main` for day-to-day work.
- **Never push the same tag to both origin and upstream** — RC tags go to origin only, final tags to upstream only. Duplicate tags cause ACR race conditions.
- RC tags on fork are disposable — iterate freely
- Graph push runs from the ITER machine where Neo4j runs — CI cannot build graph data
- The release CLI handles everything — do not manually push graphs or tags separately

## Standard Names

> Architecture docs: `docs/architecture/standard-names.md` (pipeline detail),
> `docs/architecture/standard-names-decisions.md` (design rationale).

### Pipeline

Five-phase DAG: **EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST**

| Phase | Worker | Key Operation |
|-------|--------|---------------|
| EXTRACT | `extract_worker` | Query DD paths, classify (quantity/metadata/skip), enrich with clusters, group into batches. Writes `StandardNameSource` nodes to graph for crash resilience |
| COMPOSE | `compose_worker` | LLM generates names per batch; unit injected from DD (never from LLM output). Auto-attaches DD paths to existing matching standard names without regeneration. Updates `StandardNameSource` status (composed/attached/vocab_gap) |
| VALIDATE | `validate_worker` | ISN 3-layer validation (Pydantic → semantic → description) + grammar round-trip |
| CONSOLIDATE | `consolidate_worker` | Cross-batch dedup, conflict detection (unit/kind/source), coverage accounting |
| PERSIST | `persist_worker` | Conflict-detecting Neo4j writes with coalesce semantics |

**Key modules:**

| Module | Purpose |
|--------|---------|
| `imas_codex/standard_names/classifier.py` | 13-rule path classifier: quantity (→ name), metadata (→ skip), skip (→ discard). Rules 11a/11b added: structural keywords in description → skip, fit diagnostic paths → skip |
| `imas_codex/standard_names/enrichment.py` | Primary cluster selection (IDS > domain > global scope), grouping cluster selection (global > domain > IDS), global grouping by (cluster × unit) |
| `imas_codex/standard_names/consolidation.py` | Cross-batch dedup, 5 conflict checks, coverage gap accounting |
| `imas_codex/standard_names/graph_ops.py` | Neo4j read/write with unit conflict detection + StandardNameSource CRUD (merge, claim, mark, reconcile) |
| `imas_codex/standard_names/pipeline.py` | DAG orchestrator wiring workers into `run_discovery_engine()` |
| `imas_codex/standard_names/workers.py` | Five async worker functions (review is standalone, not in generate pipeline) |
| `imas_codex/standard_names/models.py` | Pydantic response models (`StandardNameComposeBatch`, `StandardNameAttachment`) |
| `imas_codex/standard_names/source_paths.py` | Central encode/parse/split/merge utilities for StandardName source paths |
| `imas_codex/standard_names/context.py` | Grammar context builder (vocabulary, examples, tokamak ranges) |
| `imas_codex/standard_names/calibration.py` | Centralized loader (cached) for `benchmark_calibration.yaml` |
| `imas_codex/standard_names/search.py` | Vector search for similar existing StandardName nodes (collision avoidance) |

**Unit safety:** Units flow exclusively from the DD `HAS_UNIT` relationship → EXTRACT → prompt
(marked read-only) → injected into candidate dict by worker → `HAS_UNIT` relationship in graph.
The LLM never provides the unit field.

### CLI Commands

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `sn generate` | Generate standard names from DD paths or facility signals via LLM pipeline | `--source {dd,signals}`, `--ids`, `--domain`, `--facility`, `--paths`, `--limit`, `-c/--cost-limit`, `--dry-run`, `--force`, `--reset-to`, `--from-model`, `--name-only` |
| `sn review` | Score and tier existing valid standard names via batched reviewer LLM (1:1 scoring invariant, retry-unmatched) | `--ids`, `--domain`, `--source`, `--limit`, `-c/--cost-limit`, `--dry-run`, `--force` |
| `sn enrich` | Enrich existing standard names with documentation | `--ids`, `--domain`, `--status`, `-c/--cost-limit`, `--dry-run`, `--limit`, `--batch-size` |
| `sn publish` | Export validated StandardName nodes to YAML catalog files | `--output-dir`, `--ids`, `--domain`, `--group-by {ids,domain,confidence}`, `--confidence-min`, `--catalog-dir`, `--create-pr` |
| `sn import` | Import reviewed YAML catalog entries back into graph | `--catalog-dir` (required), `--tags`, `--dry-run`, `--check` |
| `sn status` | Show standard name and StandardNameSource pipeline statistics | — |
| `sn gaps` | List grammar vocabulary gaps from composition | `--segment`, `--export {table,yaml}` |
| `sn reset` | Reset standard names for re-processing | `--status` (required), `--to`, `--source`, `--ids`, `--dry-run` |
| `sn clear` | Delete standard names from the graph (relationship-first safety model) | `--status`, `--all`, `--source`, `--ids`, `--include-accepted`, `--include-sources`, `--dry-run` |
| `sn reconcile` | Reconcile StandardNameSource nodes after DD/signal rebuild | `--source-type {dd,signals}` |
| `sn benchmark` | Benchmark LLM models on standard name generation quality | `--models`, `--ids`, `--reviewer-model`, `--max-candidates` |

### Benchmark

`sn benchmark` uses the same prompt pipeline as `sn generate` (system/user message split via
`build_compose_context()`). Model lists default from `[tool.imas-codex.sn.benchmark]` in
pyproject.toml. Output table includes a **Cache %** column showing the prompt-cache
hit rate per model (provider-side via OpenRouter — not something we implement). Scoring is
**6-dimensional**: grammar, semantic, documentation, convention, completeness, and compliance
(each 0-20 integer, aggregate normalized to 0-1 via `sum / 120.0`), evaluated by a reviewer LLM.
Scoring criteria are defined in `imas_codex/llm/config/sn_review_criteria.yaml`. The calibration
dataset (`benchmark_calibration.yaml`) is loaded via `imas_codex/standard_names/calibration.py` (cached
singleton) and provides known-quality examples for reviewer consistency checks.

**Qualified models** (benchmark evidence from equilibrium + core_profiles + magnetics):

| Role | Model | Avg Quality | Notes |
|------|-------|-------------|-------|
| **Compose (recommended)** | `anthropic/claude-sonnet-4.6` | 76.5 | 32% Outstanding, best grammar + documentation |
| **Compose (budget)** | `google/gemini-3.1-pro-preview` | 74.6 | Near-top quality, good consistency |
| **Compose (light)** | `anthropic/claude-haiku-4.5` | 61.2 | Adequate for bulk generation |
| **Review/scoring** | `anthropic/claude-opus-4.6` | — | Reviewer (6-dimensional rubric judge) |

**GPT-5.4 compatibility:** GPT-5.x models require `strict: false` JSON schema wrapping
(handled automatically in `llm.py`) and cannot use `temperature=0.0` (handled in benchmark).
Quality is adequate (68.9 avg) but not top-tier for standard name generation.

### StandardName Lifecycle

```
drafted → published → accepted
```

- **drafted**: Generated by `sn generate` (LLM pipeline). All names are persisted with quality scores — low-scoring names can be regenerated via `--paths`
- **published**: Exported by `sn publish` to YAML catalog for human review
- **accepted**: Imported by `sn import` from reviewed catalog (catalog-authoritative)

### StandardName Validation Status

A separate `validation_status` field gates names before they participate in review, consolidation, and publish:

```
pending → valid | quarantined
```

- **pending**: Default state when a name is first drafted
- **valid**: Passed all critical validation checks — eligible for `sn review`, consolidation, and `sn publish`
- **quarantined**: Failed one or more critical checks — excluded from downstream pipeline stages

**Critical failures (→ quarantine):** grammar round-trip failure, Pydantic construction error, detected ambiguity.

**Non-critical issues (→ valid):** semantic warnings, description quality hints — persisted as `validation_issues` but do not gate the name.

Only `valid` names participate in `sn review`, consolidation, and `sn publish`.

### StandardNameSource Lifecycle

`StandardNameSource` nodes track individual DD path / facility signal extraction through the pipeline. Written by the extract worker, updated by the compose worker.

```
extracted → composed | attached | vocab_gap | failed | stale
```

- **extracted**: Path queued for composition (written by extract worker)
- **composed**: LLM generated a new standard name for this source
- **attached**: Source auto-attached to an existing standard name (no LLM call needed)
- **vocab_gap**: Grammar vocabulary gap prevented naming this source
- **failed**: Composition failed (LLM error, validation rejection)
- **stale**: Source no longer exists in DD/signals graph (set by `sn reconcile`)

**ID format**: `dd:{full_dd_path}` or `signals:{facility}:{signal_id}` — the `dd:` prefix is the canonical URI scheme for DD sources (e.g. `dd:equilibrium/time_slice/profiles_1d/psi`).

**Reconciliation**: `sn reconcile --source-type dd` detects StandardNameSource nodes whose backing DD path or facility signal no longer exists in the graph and marks them `stale`.

### Reset and Clear Semantics

**`sn reset`** — Re-processes existing nodes without deleting them. Clears transient fields
(embedding, model, confidence, generated_at) and removes HAS_STANDARD_NAME and HAS_UNIT
relationships. Optionally changes `review_status` via `--to <status>`. Default (no `--to`) leaves
status unchanged, only clears fields.

**`sn clear`** — Deletes StandardName nodes. Uses a relationship-first safety model: HAS_STANDARD_NAME
edges are removed before deleting nodes, and scoped deletes only remove orphaned nodes. Requires
either `--status <value>` or `--all`. Pass `--include-sources` to also delete associated
`StandardNameSource` nodes.

**Safety guard:** Both commands require `--include-accepted` to touch names with `review_status=accepted`.
Accepted names are catalog-authoritative and should rarely be deleted from the graph.

**`sn generate --reset-to`** — Runs a `sn reset` before minting, scoped to the same `--ids`/`--source`
filter. Accepts `extracted` or `drafted` as the target status. Useful for a clean re-run on a
specific IDS without touching the rest of the graph.

**`sn generate --from-model`** — Provenance-based regeneration filter. Selects only nodes whose
`model` field contains the given substring (e.g. `--from-model gemini`). Useful for re-generating
names produced by a specific model after benchmarking reveals a better alternative.

**`sn generate --name-only`** — Name-only composition mode. The compose prompt focuses exclusively
on naming and grammar — no documentation, description, or enrichment fields are generated. Faster
and cheaper for bulk naming passes where enrichment will be added later via `sn enrich`.

### Write Semantics

Two distinct write paths with different semantics:

- **`write_standard_names()` (build path)**: Uses `coalesce(b.field, sn.field)` for ALL fields — passing None preserves existing graph data. Safe to re-run without erasing imported data. Also persists `validation_issues` (list of tagged strings from ISN 3-layer validation) and `validation_layer_summary` (JSON with per-layer pass/fail counts).
- **`_write_catalog_entries()` (import path)**: Catalog fields SET directly (overwrite) — catalog is authoritative. Graph-only fields (embedding, model, generated_at, confidence) preserved via coalesce.

### MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `search_standard_names` | Semantic + keyword search over StandardName descriptions | `query`, `kind`, `tags`, `review_status`, `cocos_type`, `k` |
| `fetch_standard_names` | Fetch full entries by name ID | `names` (space/comma separated) |
| `list_standard_names` | List with optional filters | `tag`, `kind`, `review_status`, `cocos_type` |

### Schema

StandardName and StandardNameSource nodes defined in `imas_codex/schemas/standard_name.yaml`. Key relationships:

- `(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)`
- `(FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)`
- `(StandardName)-[:HAS_UNIT]->(Unit)`
- `(StandardName)-[:HAS_COCOS]->(COCOS)`
- `(StandardNameSource)-[:SOURCE_DD_PATH]->(IMASNode)` — DD-sourced extraction tracking
- `(StandardNameSource)-[:SOURCE_SIGNAL]->(FacilitySignal)` — signal-sourced extraction tracking
- `(StandardNameSource)-[:PRODUCED_STANDARD_NAME]->(StandardName)` — links source to result

**COCOS provenance:** `cocos_transformation_type` (string, e.g. `psi_like`, `ip_like`) records
how a quantity transforms under COCOS convention changes. `cocos` (integer) links directly to
the COCOS singleton node whose convention applies — works for any source (DD, signals, manual).
`dd_version` (string) is optional provenance recording which DD snapshot was used for DD-sourced names.
Both `cocos` and `cocos_transformation_type` are injected post-LLM (like `unit`) — never generated
by the model.

**`physics_domain`:** Taken directly from the DD `IMASNode.physics_domain` field — DD-authoritative only. The LLM never fills this field. For ISN validation purposes, falls back to `"general"` when a name's `physics_domain` is absent or unrecognised.

**Provenance fields** (v0.5.0): `reviewer_model`, `reviewer_score` (float 0-1, normalized from
6×0-20), `reviewer_scores` (JSON: grammar/semantic/documentation/convention/completeness/compliance,
each 0-20), `reviewer_comments`, `reviewed_at`, `review_tier` (outstanding/good/adequate/poor),
`vocab_gap_detail` (JSON: segment/needed_token/reason), `catalog_commit_sha`,
`validation_issues` (list of tagged strings), `validation_layer_summary` (JSON).

**VocabGap nodes** record missing grammar tokens identified during composition when a needed
vocabulary token does not exist in the ISN grammar. Linked via `HAS_SN_VOCAB_GAP` from IMASNode
and FacilitySignal to the VocabGap node. Each VocabGap captures the segment type, needed token,
and reason. Use `sn gaps` to view gaps as a table and `sn gaps --export yaml` to produce
YAML suitable for ISN vocabulary issue filing.

### Architecture Boundary

> Full details: `docs/architecture/boundary.md`

ISN owns grammar, vocabulary, and validation. Codex owns the pipeline, evaluation, and graph persistence.

**Import boundary** (ISN ≥0.7.0rc3):
- `get_grammar_context()` — single entry point for all grammar data (19 keys)
- `create_standard_name_entry()` — Pydantic model construction (18 validators)
- `run_semantic_checks()` — 9 semantic grammar checks
- `validate_description()` — description quality checks
- `parse_standard_name()` / `compose_standard_name()` — grammar round-trip

**Rules:** Never import from ISN private modules. Never hardcode grammar rules — get them from `get_grammar_context()`. Review criteria and scoring live in codex (`sn_review_criteria.yaml`).

### Prompt Infrastructure

Compose and review prompts use shared fragments via `{% include %}`:
- `{% include "sn/_grammar_reference.md" %}` — grammar vocabulary and segment order (used in `compose_system.md`)
- `llm/prompts/shared/sn/_scoring_rubric.md` — 6-dimension scoring rubric (shared reference)
- `llm/config/sn_review_criteria.yaml` — scoring dimensions, tiers, verdict rules (loaded via `load_prompt_config()`)
- ISN context keys (`quick_start`, `common_patterns`, `critical_distinctions`) rendered in compose prompt

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
uv run ruff check --fix .           # Lint (Python only)
uv run ruff format .                # Format
git add <file1> <file2> ...         # Stage specific files (never git add -A)
uv run git commit -m "type: concise summary"  # Conventional format
git pull --no-rebase origin main     # Merge fork changes first
git push origin main                 # Push to fork (NEVER upstream)
```

**Never stage:** auto-generated files (models.py, dd_models.py, schema_context_data.py), gitignored files, `*_private.yaml` files.

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring |
| docs | Documentation |
| test | Test changes |
| chore | Maintenance |

Breaking changes use `BREAKING CHANGE:` footer, not `type!:` suffix.

**No AI co-authorship trailers.** Never include `Co-authored-by: Copilot`, `Co-authored-by: Claude`, or any AI assistant co-authorship trailers in commit messages. The `includeCoAuthoredBy` setting is disabled in `~/.copilot/config.json`. All commits are authored solely by the human developer.

**No phase labels or step numbers in commit titles.** Never prefix or embed phase identifiers (e.g., "Phase 1:", "Step 2:", "P3:", "(Phase 4)") in commit messages. Each commit should describe *what* changed, not *which step of a plan* it belongs to. Planning context belongs in session artifacts, not in the permanent git history.

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main```

### Parallel Agents

Multiple agents may be working on this repository simultaneously. Assume another agent could be editing files or committing right now.

**CRITICAL — Verify before modifying:**

- **Re-read files before editing** — your in-memory view of a file may be hours or days old. Another agent may have renamed functions, added features, or restructured code since you last read it. Always `view` or `cat` the current file from disk before making changes. If the file looks different from what you expect, **stop and re-read** — do not "fix" it back to what you remember.
- **Check recent git history** before modifying shared files: `git log --oneline -5 -- <file>`. If there are commits you don't recognize, read the file fresh and understand the current state before editing.
- **If you see unfamiliar method names, imports, or patterns** in a file, assume they are correct and intentional. Another agent renamed them. Do not revert unfamiliar changes.

**CRITICAL — Do not touch files you didn't modify:**

- **Only stage files you modified** — never `git add -A` or `git add .`
- **NEVER run `git checkout`, `git restore`, or `git reset` on files you didn't change** — this silently destroys another agent's in-progress work with no way to recover it. Even if a file appears "dirty" or has unexpected changes, leave it alone — another agent put those changes there deliberately.
- **NEVER run `git checkout -- .` or `git restore .`** — these wipe ALL unstaged changes across the entire repo, including other agents' work
- **Never rebase** — rebase rewrites history and clobbers parallel agents' work. Always merge: `git pull --no-rebase upstream main`
- **Pull before push** if push is rejected: `git pull --no-rebase upstream main && git push upstream main`
- **Avoid broad formatting runs** (`ruff format .`) unless you are the only agent active — prefer formatting only your changed files
- **If `git stash` is needed**, only stash your own files: `git stash push -- file1 file2`, never `git stash` (which stashes everything)
- **NEVER run `git stash pop` or `git stash apply`** without first checking the stash age and contents. Stale stashes from prior work sessions can silently overwrite the working tree with old code, reverting committed renames and improvements. Always verify: `git stash show stash@{N} --stat` and `git log -1 --format='%ci' stash@{N}` before applying. If the stash is more than a day old, drop it — the code has moved on.
- **Auto-generated files cause dirty worktrees** — `uv sync` regenerates model files that are gitignored. These should never be staged, but their presence will block `git pull --rebase`. This is another reason merge is the correct policy.

**Session hygiene:**

- **Close sessions when done** — `ctrl+d`, `/exit`, or `/quit`. Idle `copilot` processes with stale context are the #1 cause of regressions.
- **Audit periodically:** `ps aux | grep copilot` — kill any process older than your current session.
- **Avoid long-lived `--yolo` sessions** — auto-approve + stale context is the most dangerous combination. Start fresh sessions for new tasks.

### Session Completion

**MANDATORY** after any file modifications: commit and push before responding to the user.

End every response that modifies files with the **full commit message** and a brief summary.

## Feature Plan Documentation

Plans live in `plans/features/`. Delete when fully implemented — the code is the documentation.

**Every feature plan must include a documentation phase** as its final step. Before a plan is considered complete, the implementing agent must update all affected documentation to maintain self-consistency across the project. This is not optional — undocumented features create drift between what the code does and what agents/users expect.

### Required documentation checklist

Each plan must include a section titled "Documentation Updates" listing which of these apply:

| Target | When to update |
|--------|----------------|
| `AGENTS.md` | New CLI commands, MCP tools, config sections, workflows, or conventions |
| `README.md` | User-facing features, installation changes, quick-start examples |
| `plans/README.md` | Plan added, completed, or moved to pending |
| `.claude/skills/*.md` | New reusable workflows agents should know |
| `.claude/agents/*.md` | New agent capabilities or tool access changes |
| `docs/` | Mature architecture documentation for implemented systems |
| Prompt templates | New or changed LLM prompts referenced by the feature |
| Schema reference | Handled automatically by `uv run build-models` — but verify after schema changes |

### Plan lifecycle

```
plans/features/<name>.md          → Active plan (unstarted or in-progress)
plans/features/pending/<name>.md  → Partially implemented, gaps documented
DELETE                            → Fully implemented (code is the documentation)
```

- **Gap documents** (`gaps-*.md`) consolidate remaining work from multiple related pending plans. These are the canonical handoff documents for agents.
- **Pending plans** are reference material for gap documents — not direct work items.
- **Unstarted plans** remain in `features/` until work begins or they are superseded.

### Self-consistency rule

When implementing a feature, check whether your changes contradict or extend existing documentation. A feature is not done until:

1. All code changes are committed and tested
2. Every documentation target in the checklist above is reviewed and updated if affected
3. `plans/README.md` is updated to reflect the plan's new status
4. The plan file itself is deleted (fully implemented) or moved to `pending/` (gaps remain)

## Code Style

- Python ≥3.12: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`
- Exception chaining: `raise Error("msg") from e`
- `pydantic` for schemas, `dataclasses` for other data classes
- `anyio` for async
- `uv run` for all Python commands (never activate venv manually)
- Never use `git add -A`
- The `.env` file contains secrets — never expose or commit it

### Naming

**Never name files after implementation plans.** File names (tests, modules, scripts) must be understandable without knowledge of any plan document. Once a plan is deleted (per project rules), names like `test_capability_gaps` become meaningless. Instead, name files after what they test or implement: `test_dd_tool_features`, `test_lifecycle_filtering`, `test_migration_guide`.

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

The `repl()` MCP tool provides a persistent REPL for custom queries not covered by the search tools. Prefer `search_signals`, `search_docs`, `search_code`, and `search_dd_paths` for common lookups — they perform multi-index vector search with graph enrichment and return formatted reports in one call.

### REPL Workflow

1. **Use search_* MCP tools first** for signal, documentation, code, and IMAS lookups. They handle embeddings, multi-index fan-out, enrichment, and formatting automatically.
2. **Use repl() for custom queries** — signal→IMAS mapping, facility overviews, flexible graph_search(), raw Cypher, or chaining multiple domain functions.
3. **Chain operations** in a single `repl()` call to minimize round-trips. Each call has overhead.
4. **For raw Cypher** (only when no domain function fits), call `schema_for(task='wiki')` first to get node labels, properties, relationships, and enums derived from the LinkML schemas. Never guess property names — they are code-generated.
5. **Format output** with `as_table(pick(results, 'col1', 'col2'))` for structured results.

### Schema-First Queries

All graph node types, properties, enums, and relationships are derived from LinkML schemas. The REPL exposes this via:

- `schema_for(task='signals')` — schema context for a domain (signals, wiki, imas, code, facility, trees)
- `schema_for('WikiChunk', 'WikiPage')` — schema for specific node labels
- `get_schema()` — full `GraphSchema` object with `node_labels`, `get_model()`, `get_properties()`
- `repl_help()` — auto-generated API reference with all function signatures

**Never hardcode property names.** Before writing raw Cypher, call `schema_for(task='wiki')` to verify property names. Use `repl_help()` for the full API reference.

## Quick Reference

**Primary MCP tools** — use these first, they return formatted reports:

| Task | MCP Tool |
|------|----------|
| Signal lookup | `search_signals("plasma current", facility="tcv")` |
| Documentation | `search_docs("fishbone instabilities", facility="jet")` |
| Code examples | `search_code("equilibrium reconstruction", facility="tcv")` |
| IMAS DD paths | `search_dd_paths("electron temperature", facility="tcv")` — results include semantic cluster labels and "See Also" cross-IDS siblings for top hits |
| Full content | `fetch_content("jet:Fishbone_proposal_2018.ppt")` — use IDs/URLs from search results |

**repl() REPL** — for custom queries not covered by the search tools:

| Task | Command |
|------|---------|
| Wiki keyword | `repl("print(find_wiki(text_contains='fishbone'))")` |
| Page chunks | `repl("print(wiki_page_chunks('equilibrium', facility='tcv'))")` |
| Signal→IMAS map | `repl("print(map_signals_to_imas(facility='tcv', physics_domain='magnetics'))")` |
| Graph search | `repl("print(graph_search('WikiChunk', where={'text__contains': 'IMAS'}))")` |
| Format table | `repl("print(as_table(find_signals('ip', facility='tcv')))")` |
| Facility info | `repl("print(get_facility('tcv'))")` |
| Raw Cypher | `repl("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Remote command | `ssh facility "rg pattern /path"` |

Chain multiple operations in a single `repl()` call to minimize round-trips.

## Embedding Server

Config lives in `pyproject.toml` under `[tool.imas-codex.embedding]`. Key accessor: `get_embedding_location()` returns the facility name or `"local"`. Port derived from position in shared `locations` list: `18765 + offset`.

### CRITICAL — SLURM Only

**All services (embed, Neo4j) MUST run as SLURM jobs.** Never bypass SLURM with `nohup`, `ssh … &`, `screen`, `tmux`, or any other manual process management on compute nodes. SLURM provides:
- cgroup resource isolation (GPU, memory, CPU)
- clean lifecycle management (`scancel` = graceful stop)
- accurate resource accounting via `squeue`/`sacct`
- automatic cleanup on node drain/failure

**Never start services directly on compute nodes via SSH.** If SLURM won't schedule (node draining/down), the fix is to get the node resumed — not to work around SLURM. Rogue processes outside SLURM cause "Duplicate jobid" errors that drain nodes for all users.

### Commands

```bash
imas-codex embed start           # Start per config (SLURM or systemd)
imas-codex embed start -g 2      # Start with 2 GPUs
imas-codex embed start -f        # Foreground only (debugging, or inside SLURM batch)
imas-codex embed status          # Health + SLURM job + node state
imas-codex embed restart -g 8    # Restart with 8 GPUs (~18s cycle)
imas-codex embed stop            # Stop SLURM job + cleanup rogue processes
imas-codex embed logs            # View SLURM logs
imas-codex embed service install # Install systemd service (login node only)
```

### Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `embed status` shows "⚠ Node draining" | SLURM won't schedule new jobs | Ask admin: `scontrol update NodeName=<node> State=RESUME` |
| PENDING job never starts | Node may be draining or at resource limit | `imas-codex embed status` shows node state |
| Server healthy but no SLURM job | Rogue process running outside SLURM | `imas-codex embed stop` kills rogues automatically |
| Rapid FAILED jobs in `sacct` | Package/env issue on compute node | Check `imas-codex embed logs`, run `uv sync` on node |
| Embedding calls timeout | Tunnel not active or server down | `lsof -i :18765` then `imas-codex embed status` |

## Domain Workflows

Extended examples and edge cases for each domain: [agents/](agents/)

| Agent | Purpose |
|-------|---------|
| Explore | Remote facility discovery (read-only + MCP) |
| Develop | Code development (standard + MCP) |

| Graph | Knowledge graph operations (core + MCP) |

## AI Tooling Configuration

This project supports multiple AI coding tools (Claude Code, VS Code Copilot, Cursor, etc.) from **canonical sources** — no duplication.

### Canonical Sources

| What | Canonical File | Read By |
|------|---------------|---------|
| **Project instructions** | `AGENTS.md` | Claude Code (via `CLAUDE.md` → `@AGENTS.md`), VS Code Copilot (native `AGENTS.md` support), Cursor (via `.cursorrules` import) |
| **MCP servers** | `.mcp.json` + `.vscode/mcp.json` | Claude Code (`.mcp.json`, `mcpServers` key), VS Code (`.vscode/mcp.json`, `servers` key) |
| **Custom agents** | `.claude/agents/*.md` | Claude Code (native) |
| **Skills/commands** | `.claude/skills/*.md` | Claude Code (native) |
| **Tool-specific settings** | `.claude/settings.json`, `.vscode/settings.json` | Their respective tools (not shared) |

### Architecture

```
AGENTS.md                    ← canonical project instructions (all tools)
CLAUDE.md                    ← Claude Code entry point (@AGENTS.md import)
.mcp.json                    ← MCP config for Claude Code (mcpServers key)
.vscode/mcp.json             ← MCP config for VS Code (servers key)
.claude/
├── agents/                  ← Claude Code custom agents
│   ├── facility-explorer.md
│   └── graph-querier.md
├── skills/                  ← Claude Code skills (reusable prompts)
│   ├── facility-access.md
│   ├── graph-queries.md
│   ├── schema-summary.md
│   └── mapping-workflow.md
└── settings.json            ← Claude Code permissions & env
.vscode/
├── settings.json            ← VS Code/Copilot settings
├── toolsets.jsonc            ← VS Code agent toolset definitions
└── instructions.json         ← VS Code instruction file patterns
```

### Rules

1. **Never duplicate instructions** — `AGENTS.md` is the single source of truth for all project guidelines. `CLAUDE.md` imports it via `@AGENTS.md`. For other tools, configure their instruction path to read `AGENTS.md`.
2. **MCP servers defined in two files** — `.mcp.json` (Claude Code, `mcpServers` key) and `.vscode/mcp.json` (VS Code, `servers` key). Both are tracked in git. VS Code only reads `.vscode/mcp.json`; Claude Code only reads `.mcp.json`.
3. **Tool-specific config stays tool-specific** — permissions, env vars, and model preferences differ per tool and belong in their respective settings files.
4. **When adding an MCP server**, add it to both `.mcp.json` (`mcpServers`) and `.vscode/mcp.json` (`servers`).

## MCP Server Deployment

```bash
# Default: full mode with all tools (REPL, search, write)
uv run imas-codex serve

# DD-only mode: DD tools only, implies read-only (container deployments)
uv run imas-codex serve --dd-only --transport streamable-http

# Read-only: search and read tools only
uv run imas-codex serve --read-only --transport streamable-http

# STDIO transport for MCP clients (VS Code, Claude Desktop)
uv run imas-codex serve --transport stdio
```

**Deployment topology:**

| Deployment | Command | Tools Available |
|------------|---------|-----------------|
| Development | `imas-codex serve` | All (REPL, search, write, infrastructure) |
| DD-only container | `imas-codex serve --dd-only` | DD search and read only |
| Public / read-only | `imas-codex serve --read-only` | Search and read only |
| HPC / SLURM | `scripts/imas_codex_slurm_stdio.sh` | All (inside allocation) |

## Service Availability

**The Neo4j graph and embedding server are always running.** Both services run as SLURM jobs and should be assumed available at all times on all development machines (ITER, WSL). If a service is down, restart it — do not work around it.

### Connecting from ITER

On ITER login/compute nodes, connections resolve automatically via the graph profile system. `GraphClient()` (no arguments) discovers the SLURM compute node running Neo4j and connects directly. Never hardcode `bolt://localhost:7687` — use the profile-aware accessors:

```python
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri, get_graph_username, get_graph_password

# Preferred: uses profile resolution (handles SLURM, tunnels, env overrides)
gc = GraphClient()

# Explicit (still profile-aware):
gc = GraphClient(uri=get_graph_uri(), username=get_graph_username(), password=get_graph_password())
```

### Connecting from WSL / remote machines

From machines outside the ITER network, start an SSH tunnel first:

```bash
uv run imas-codex tunnel start iter    # Tunnel Neo4j bolt port to localhost
uv run imas-codex tunnel status        # Verify tunnel is active
```

The profile system auto-detects remote hosts and creates tunnels on demand. If auto-tunneling fails, set the tunnel port explicitly:

```bash
export IMAS_CODEX_TUNNEL_BOLT_ITER=17687
```

### Authentication

**Always use the Python client methods for graph and embedding connections** — never call Neo4j or the embedding server directly via raw HTTP/bolt. The Python methods (`GraphClient`, `get_graph_uri()`, `Encoder`) handle authentication, SLURM node discovery, tunnel setup, and retry logic automatically. The `.env` file provides credentials that raw connections would miss.

## Fallback: MCP Server Not Running

```bash
uv run imas-codex graph status          # Graph operations
uv run imas-codex graph shell           # Interactive Cypher
uv run pytest                           # Testing
```
