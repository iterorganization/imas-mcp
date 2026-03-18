# Dynamic Tool Registration & Server Unification

**Status:** Planned
**Created:** 2026-03-18
**Priority:** High — blocks TCV facility deployment
**Scope:** `imas_codex/server.py`, `imas_codex/llm/server.py`, `imas_codex/tools/__init__.py`,
`imas_codex/cli/serve.py`, `.vscode/mcp.json`, `.cursor/mcp.json`

## Problem Statement

The project currently has two separate MCP servers:

1. **`serve imas`** (`imas_codex/server.py`) — 11 graph-backed IMAS DD tools
2. **`serve agents`** (`imas_codex/llm/server.py`) — 15+ tools including search, REPL,
   facility infrastructure, graph write operations, and log inspection

When deploying to a facility like TCV in read-only mode (graph data baked into a
container), the full `agents` toolset is inappropriate:

- **Write tools** (`add_to_graph`, `update_facility_infrastructure`,
  `add_exploration_note`) cannot work against a read-only deployment
- **`python` REPL** exposes arbitrary code execution and is not appropriate
  for a shared read-only service
- **`map_signals_to_imas`** requires development-time graph write access
- **`get_facility_infrastructure`** reads private YAML not shipped in containers
- **`find_data_nodes`** duplicates `search_signals` and should be consolidated

Additionally, not every graph deployment has the same content. A DD-only graph
has no `SignalNode`, `WikiPage`, or `CodeFile` nodes, so signal/docs/code
search tools are meaningless. A TCV+DD graph has signals but may not have wiki
pages.

## Design

### Core Principle: Label-Driven Discovery

At startup, the server queries `db.labels()` to discover what data is in the
graph. Each tool class declares the graph labels it requires via a
`requires_labels` class attribute. Only tools whose required labels are all
present get registered.

```python
class GraphSearchTool:
    requires_labels: ClassVar[set[str]] = {"IMASNode"}  # Always present
    ...

class SignalSearchTool:
    requires_labels: ClassVar[set[str]] = {"SignalNode"}
    ...

class DocSearchTool:
    requires_labels: ClassVar[set[str]] = {"WikiPage"}
    ...

class CodeSearchTool:
    requires_labels: ClassVar[set[str]] = {"CodeFile"}
    ...

class FacilityTool:
    requires_labels: ClassVar[set[str]] = {"Facility"}
    ...
```

### Single Server

Replace the `imas` / `agents` split with a single `serve` command. The
server name and available tools are derived entirely from what the graph
contains.

```
imas-codex serve [--transport stdio|sse|streamable-http] [--read-only]
```

The `--read-only` flag suppresses all write tools regardless of graph content.
This is the default for container deployments.

### Tool Categorization

| Category | Tools | Condition |
|----------|-------|-----------|
| **Always** | `search_imas_paths`, `fetch_imas_paths`, `check_imas_paths`, `list_imas_paths`, `get_imas_overview`, `get_imas_identifiers`, `get_imas_path_context`, `get_imas_structure`, `search_imas_clusters`, `get_dd_versions`, `get_dd_graph_schema`, `query_imas_graph` | `IMASNode` in labels (every graph) |
| **Signals** | `search_signals` | `SignalNode` in labels |
| **Docs** | `search_docs`, `fetch` | `WikiPage` in labels |
| **Code** | `search_code` | `CodeFile` in labels |
| **Facility** | `get_facility_infrastructure`, `update_facility_infrastructure`, `get_discovery_context` | `Facility` in labels AND NOT `--read-only` |
| **Development** | `add_to_graph`, `add_exploration_note`, `map_signals_to_imas`, `python` (REPL) | NOT `--read-only` |
| **Schema** | `get_json_schema` | Always (file-based, no graph dependency) |
| **Logs** | `list_logs`, `get_logs`, `tail_logs` | NOT `--read-only` |

### Consolidation: `find_data_nodes` → `search_signals`

`find_data_nodes` (in `imas_codex/llm/server.py`) duplicates `search_signals`.
Both query the same `SignalNode` graph data. Remove `find_data_nodes` entirely
and direct users to `search_signals`.

## Implementation Phases

### Phase 1: `requires_labels` on Tool Classes

Add `requires_labels: ClassVar[set[str]]` to each tool class. Refactor
`Tools.__init__()` to accept `available_labels: set[str]` and conditionally
instantiate tools.

**Files:**
- `imas_codex/tools/__init__.py` — conditional instantiation
- `imas_codex/tools/graph_search.py` — add `requires_labels` to each class
- `imas_codex/tools/cypher_tool.py` — add `requires_labels`
- `imas_codex/tools/schema_tool.py` — add `requires_labels`
- `imas_codex/tools/version_tool.py` — add `requires_labels`

### Phase 2: Read-Only Mode

Add `read_only: bool = False` parameter to `Tools` and `Server`. When True,
skip registration of write tools (`add_to_graph`, `add_exploration_note`,
`update_facility_infrastructure`, `update_facility_config`, `python` REPL).

**Files:**
- `imas_codex/server.py` — accept `--read-only` flag
- `imas_codex/tools/__init__.py` — filter by read_only
- `imas_codex/llm/server.py` — filter tool registration by read_only

### Phase 3: Server Unification

Merge `imas_codex/llm/server.py` tools into `imas_codex/server.py`. Replace
the `serve imas` / `serve agents` commands with a single `serve` command.
The server discovers its capabilities from the graph.

**Files:**
- `imas_codex/server.py` — absorb agent tools
- `imas_codex/cli/serve.py` — single `serve` command
- `.vscode/mcp.json` — single server config
- `.cursor/mcp.json` — single server config
- `Dockerfile` — single serve command in entrypoint
- `docker-entrypoint.sh` — simplified

### Phase 4: Remove `find_data_nodes`

Remove the duplicate tool and update any references.

**Files:**
- `imas_codex/llm/server.py` — remove `find_data_nodes` function
- Update REPL docstring API reference

## Deployment Topology

### DD-Only Container (imas-dd.iter.org)

```
Graph: DDVersion, IMASNode, IMASCluster, IdentifierSchema, Unit
Tools: IMAS DD suite (12 tools) + schema
Mode: --read-only
```

### TCV + DD Container (TCV facility)

```
Graph: DDVersion, IMASNode, IMASCluster, SignalNode, WikiPage, CodeFile, Facility
Tools: IMAS DD suite + search_signals + search_docs + search_code + fetch
Mode: --read-only
```

### Full Development Server (developer workstation)

```
Graph: All labels
Tools: All tools including REPL, write tools, facility management
Mode: read-write (default)
```

## Testing

- Unit test: `Tools(graph_client=gc, read_only=True)` excludes write tools
- Unit test: `Tools(graph_client=gc, available_labels={"IMASNode"})` excludes signal/doc/code tools
- Integration test: server with DD-only graph reports correct tool list
- Integration test: server with TCV graph reports correct tool list
- Integration test: `--read-only` flag suppresses write tools

## Security Considerations

- Read-only mode is enforced at registration time — write tool functions are
  never exposed to MCP clients, not just disabled
- The `python` REPL is the highest-risk tool — it must never be registered
  in read-only mode
- `query_imas_graph` (Cypher) already has mutation detection that rejects
  CREATE/MERGE/DELETE/SET queries (validated by `TestCypherReadOnlyCheck`)
- Container deployments should always use `--read-only`
- Bearer token auth via nginx prevents unauthorized access (see deployment plan)
