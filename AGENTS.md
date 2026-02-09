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

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Schema changes are additive only in the graph—add properties, never rename or remove.

## Graph State Machine

Status enums represent **durable states only**. No transient states like `scanning`, `scoring`, or `ingesting`.

**Worker coordination:** Claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`. Orphan recovery is automatic via timeout check in claim queries. This applies to all discovery pipelines: paths, wiki, signals.

## LLM Prompts

Prompts live in `imas_codex/agentic/prompts/` using Jinja2 templates with schema injection.

- Never hardcode JSON examples - use Pydantic schema injection via `get_pydantic_schema_json()`
- Each prompt declares `schema_needs` in `prompt_loader.py` to load only required context
- LLM structured output uses Pydantic models via LiteLLM `response_format`

## Command Execution

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

## Graph Operations

### Backup

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

**Existing indexes:**

| Index | Content |
|-------|---------|
| `imas_path_embedding` | IMASPath nodes |
| `cluster_centroid` | IMASSemanticCluster centroids |
| `code_chunk_embedding` | CodeChunk nodes |
| `wiki_chunk_embedding` | WikiChunk nodes |
| `facility_signal_desc_embedding` | FacilitySignal descriptions |
| `facility_path_desc_embedding` | FacilityPath descriptions |
| `tree_node_desc_embedding` | TreeNode descriptions |
| `wiki_artifact_desc_embedding` | WikiArtifact descriptions |

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
    MATCH (signal)-[:ACCESS_METHOD]->(am:AccessMethod)
    OPTIONAL MATCH (signal)-[:MAPS_TO_IMAS]->(imas:IMASPath)
    RETURN signal.id, signal.description, am.template_python,
           collect(imas.id) AS imas_paths, score
    ORDER BY score DESC
""", embedding=embed("electron density profile"))
```

**Key relationships for traversal:**

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | ACCESS_METHOD | AccessMethod |
| FacilitySignal | MAPS_TO_IMAS | IMASPath |
| WikiChunk | HAS_CHUNK← | WikiPage |
| FacilityPath | FACILITY_ID | Facility |

**Token cost:** Always project specific properties in Cypher (`RETURN n.id, n.name`), never return full nodes. Use Cypher aggregations instead of Python post-processing.

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
# 1. Lint and format first
uv run ruff check --fix .
uv run ruff format .

# 2. Stage specific files (never git add -A, never add gitignored files)
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

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

### Session Completion

**MANDATORY** after any file modifications: commit and push before responding to the user.

End every response that modifies files with the **full commit message** and a brief summary:

```
✓ Committed: `<commit-hash>`

type: concise summary

Detailed explanation.
- Key changes
```

## Code Style

- Python ≥3.12: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`
- Exception chaining: `raise Error("msg") from e`
- `pydantic` for schemas, `dataclasses` for other data classes
- `anyio` for async
- `uv run` for all Python commands (never activate venv manually)
- Never use `git add -A`
- Never use "new", "refactored", "enhanced" in names

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

Uses SSH tunnel to ITER GPU cluster (port 18765, Qwen3-Embedding-0.6B, 1024-dim).

Establish tunnel: `ssh -f -N -L 18765:127.0.0.1:18765 iter`

If embedding fails, check in order:
1. Tunnel active: `lsof -i :18765`
2. Server running: `ssh iter "systemctl --user status imas-codex-embed"`
3. Server health: `curl http://localhost:18765/health`

## Domain Workflows

| Workflow | Documentation |
|----------|---------------|
| Facility exploration | [agents/explore.md](agents/explore.md) |
| Development | [agents/develop.md](agents/develop.md) |
| Code ingestion | [agents/ingest.md](agents/ingest.md) |
| Graph operations | [agents/graph.md](agents/graph.md) |

## Custom Agents

| Agent | Purpose |
|-------|---------|
| Explore | Remote facility discovery (read-only + MCP) |
| Develop | Code development (standard + MCP) |
| Ingest | Code ingestion pipeline (core + MCP) |
| Graph | Knowledge graph operations (core + MCP) |

## Fallback: MCP Server Not Running

```bash
uv run imas-codex neo4j status    # Graph operations
uv run imas-codex neo4j shell     # Interactive Cypher
uv run imas-codex ingest run tcv  # Ingestion
uv run pytest                     # Testing
```
