# Unified MCP Server Architecture

**Status:** Planned
**Created:** 2026-03-19
**Priority:** High — blocks facility deployment, reduces context bloat
**Supersedes:** `dynamic-tool-registration.md` (absorbed and extended)
**Related:** `shared-tool-consolidation.md`

## Problem Statement

The project has two separate MCP servers with overlapping functionality:

1. **`serve imas`** (`imas_codex/server.py`) — 13 IMAS DD tools, immediate graph
   connection at startup, health endpoint, resource provider
2. **`serve agents`** (`imas_codex/llm/server.py`) — 30 tools (strict superset of
   `serve imas`), lazy graph connection, Python REPL, prompts

This creates several problems:

- **Startup blocking:** `serve imas` connects to Neo4j immediately, failing when
  the graph is remote (titan). The agents server defers this correctly.
- **Duplicate tools:** All 13 IMAS tools exist in both servers via delegation.
  The agents server wraps them with formatters for better output.
- **Context bloat:** Running both servers registers 43 tools (13 + 30), inflating
  the API schema payload. Each tool description adds ~100-500 chars to the prompt.
- **Unused code:** `CypherTool`, `SchemaTool`, `HealthEndpoint`, and
  `Resources` are only used by the old server and have equivalents in agents.
- **No deployment flexibility:** Neither server supports read-only mode or
  label-driven tool registration for facility-specific containers.

### Root Cause: Copilot CLI 400 Errors

The investigation that led to this plan revealed that running multiple MCP servers
with complex schemas causes `$ref`/`$defs` in JSON Schema to reach the Copilot API
backend, which rejects them with HTTP 400. Consolidating to a single server with
clean schemas eliminates this risk.

## Design

### Naming

**Single command:** `imas-codex serve`

No suffix needed. The old `serve imas` / `serve agents` split was an artifact of
development history, not a design requirement. The server name reported to MCP
clients becomes `imas-codex` (or `imas-codex-{facility}` when facility-specific).

### Server Architecture

```
imas-codex serve [--transport stdio|sse|streamable-http]
                 [--read-only]
                 [--host HOST] [--port PORT]
                 [--log-level LEVEL]
```

- **Default mode (no flags):** Full development server with all tools, including
  write tools and Python REPL. Graph connection deferred to first tool call.
- **`--read-only`:** Suppresses write tools (`add_to_graph`, `update_*`,
  `add_exploration_note`) and the Python REPL. Suitable for containers and
  shared deployments.

### Label-Driven Tool Registration

At first tool call (when graph connection is established), the server queries
`db.labels()` to discover what data is in the graph. Each tool declares its
required labels. Only tools whose required labels are all present get registered.

This is deferred registration — tools are registered eagerly with the MCP server
at startup (so clients see them immediately), but their execution checks for
required labels at call time and returns a clear error if the data isn't available.

**Rationale:** MCP stdio transport requires all tools to be declared before the
first client message. We can't defer registration, but we can defer validation.

```python
# Tool categories and their label requirements
ALWAYS = set()                          # No graph labels needed
IMAS_DD = {"IMASNode"}                 # DD tools
SIGNALS = {"FacilitySignal"}           # Signal search
DOCS = {"WikiPage"}                    # Doc search, fetch
CODE = {"CodeFile"}                    # Code search
FACILITY = {"Facility"}               # Facility management
```

### Tool Inventory: 30 Tools → Assessment

All tools from the agents server, assessed for retention:

| # | Tool | Labels | Read-Only | Assessment |
|---|------|--------|-----------|------------|
| 1 | `python` | ALWAYS | ❌ suppress | **Keep** — primary power-user interface |
| 2 | `get_graph_schema` | ALWAYS | ✅ keep | **Keep** — essential for Cypher query generation |
| 3 | `add_to_graph` | ALWAYS | ❌ suppress | **Keep** — primary graph write interface |
| 4 | `update_facility_config` | ALWAYS | ❌ suppress | **Keep** — facility config management |
| 5 | `update_facility_infrastructure` | ALWAYS | ❌ suppress | **Keep** — convenience over `update_facility_config` |
| 6 | `get_facility_infrastructure` | ALWAYS | ✅ keep | **Keep** — read-only facility data |
| 7 | `add_exploration_note` | ALWAYS | ❌ suppress | **Keep** — convenience wrapper |
| 8 | `get_discovery_context` | FACILITY | ✅ keep | **Keep** — essential for exploration |
| 9 | `search_signals` | SIGNALS | ✅ keep | **Keep** — primary signal search |
| 10 | `signal_analytics` | SIGNALS | ✅ keep | **Keep** — batch analytics |
| 11 | `search_docs` | DOCS | ✅ keep | **Keep** — primary doc search |
| 12 | `search_code` | CODE | ✅ keep | **Keep** — primary code search |
| 13 | `search_imas` | IMAS_DD | ✅ keep | **Keep** — primary IMAS search |
| 14 | `check_imas_paths` | IMAS_DD | ✅ keep | **Keep** — path validation |
| 15 | `fetch_imas_paths` | IMAS_DD | ✅ keep | **Keep** — detailed path docs |
| 16 | `fetch_error_fields` | IMAS_DD | ✅ keep | **Keep** — error field discovery |
| 17 | `list_imas_paths` | IMAS_DD | ✅ keep | **Keep** — IDS browsing |
| 18 | `get_imas_overview` | IMAS_DD | ✅ keep | **Keep** — IDS summary |
| 19 | `get_imas_identifiers` | IMAS_DD | ✅ keep | **Keep** — enumeration schemas |
| 20 | `search_imas_clusters` | IMAS_DD | ✅ keep | **Keep** — semantic clusters |
| 21 | `get_imas_path_context` | IMAS_DD | ✅ keep | **Keep** — cross-IDS context |
| 22 | `analyze_imas_structure` | IMAS_DD | ✅ keep | **Keep** — IDS structure analysis |
| 23 | `export_imas_ids` | IMAS_DD | ✅ keep | **Keep** — full IDS export |
| 24 | `export_imas_domain` | IMAS_DD | ✅ keep | **Keep** — domain-wide export |
| 25 | `get_dd_version_context` | IMAS_DD | ✅ keep | **Keep** — version history |
| 26 | `get_dd_versions` | IMAS_DD | ✅ keep | **Keep** — DD metadata |
| 27 | `fetch` | DOCS or CODE | ✅ keep | **Keep** — content retrieval |
| 28 | `list_logs` | ALWAYS | ❌ suppress | **Keep** — log inspection |
| 29 | `get_logs` | ALWAYS | ❌ suppress | **Keep** — filtered log reading |
| 30 | `tail_logs` | ALWAYS | ❌ suppress | **Keep** — recent log entries |

### Tool Consolidation Candidates

After thorough assessment, **no tools should be dropped**. Each serves a distinct
purpose. However, some warrant notes:

1. **`update_facility_config` vs `update_facility_infrastructure`:** These overlap
   (the latter is a convenience wrapper for the former with `private=True`). However,
   `update_facility_infrastructure` has clearer semantics for agents and its removal
   would break AGENTS.md documentation. **Keep both.**

2. **`get_facility_infrastructure` vs `update_facility_config(data=None)`:** Same
   read operation. But having a dedicated read tool is clearer. **Keep both.**

3. **`list_logs` + `get_logs` + `tail_logs`:** Three log tools could theoretically
   be one with a `mode` parameter, but the current split matches Unix conventions
   (`ls`, `grep`, `tail`) that agents understand. **Keep all three.**

4. **`find_data_nodes` (REPL only):** Overlaps with `search_signals`. Already
   not an MCP tool — only available in the REPL. **Keep in REPL for backward
   compatibility, but mark for deprecation.**

### Schema Context Assessment

The `get_graph_schema` tool returns schema context text. Current sizes by scope:

| Scope | Purpose | Assessment |
|-------|---------|------------|
| `overview` | Compact label summary | ~1-2K chars — **appropriate** |
| `signals` | Signal node schema | ~2-3K chars — **appropriate** |
| `wiki` | Wiki/doc schema | ~2-3K chars — **appropriate** |
| `imas` | IMAS DD schema | ~3-4K chars — **appropriate** |
| `code` | Code chunk schema | ~2K chars — **appropriate** |
| `facility` | Facility schema | ~2-3K chars — **appropriate** |
| `trees` | Tree node schema | ~2K chars — **appropriate** |

**Assessment:** Schema context is demand-loaded (only returned when agents call
`get_graph_schema`), NOT injected into tool descriptions. This is the correct
pattern. No changes needed.

**The schema is NOT part of the tool registration payload.** Agents request it
when they need it. This is already optimized.

### Tool Description Bloat Analysis

Total tool description text across all 30 tools: **~34K characters** (~8.5K tokens).
This is the fixed cost per MCP session.

**Top contributors by description length:**
1. `python()` — ~1.8K chars (includes API reference) — justified, primary interface
2. `add_to_graph()` — ~1.2K chars (includes examples) — justified, complex tool
3. `update_facility_infrastructure()` — ~1.0K chars — could trim examples
4. `update_facility_config()` — ~1.0K chars — could trim examples
5. `search_imas()` — ~0.8K chars — appropriate for complex tool

**Optimization opportunities (conservative):**
- Trim examples from `update_facility_infrastructure` and `update_facility_config`
  descriptions (save ~400 chars each)
- Standardize "dd_version" parameter docs to a shorter form (save ~200 chars total)
- **Total potential savings: ~1K chars (~250 tokens) — NOT WORTH the risk of
  degraded agent understanding**

**Decision: Do not optimize tool descriptions.** The current descriptions are
well-written and agents use them effectively. The 8.5K token cost is a fixed
cost that enables correct tool usage. Reducing it risks incorrect tool calls.

## What Gets Removed

### Files to Delete

| File | Reason |
|------|--------|
| `imas_codex/server.py` | Old server — superseded by `llm/server.py` |
| `imas_codex/health.py` | Health endpoint — only used by old server |
| `imas_codex/resource_provider.py` | MCP resources — only used by old server, content is stale |
| `imas_codex/tools/cypher_tool.py` | Raw Cypher tool — replaced by `python()` REPL's `query()` |
| `imas_codex/tools/schema_tool.py` | Schema introspection — replaced by `get_graph_schema` tool |

### CLI Changes

- Remove `serve imas` command from `imas_codex/cli/serve.py`
- Rename `serve agents` to `serve` (single command)
- Move `serve_agents()` logic into top-level `serve()` or keep as default subcommand
- Add `--read-only` flag

### Code Changes

- `imas_codex/tools/__init__.py`: Remove `CypherTool` and `SchemaTool` imports,
  remove from `_tool_instances` list
- `imas_codex/providers.py`: Keep (still used by Tools class)
- `imas_codex/__init__.py`: Remove any `serve imas` references

### Test Updates

| Test File | Change |
|-----------|--------|
| `tests/integration/test_mcp_server.py` | Update to use `AgentsServer` |
| `tests/integration/test_server_extended.py` | Update to use `AgentsServer` |
| `tests/integration/test_health_endpoint.py` | Remove (health endpoint removed) |
| `tests/conftest.py` | Remove old Server fixtures |
| `tests/core/test_cli.py` | Update `serve` CLI tests |
| `tests/graph_mcp/test_graph_search.py` | Update imports |
| `tests/embeddings/test_embeddings_deferred.py` | Update imports |
| `benchmarks/benchmarks.py` | Update imports |

## Implementation Phases

### Phase 1: Remove `serve imas` and Dead Code

Remove the old server, health endpoint, resource provider, CypherTool, and
SchemaTool. Update the `serve` CLI to have a single command. Update all tests.

**Files modified:**
- `imas_codex/cli/serve.py` — merge into single `serve` command
- `imas_codex/tools/__init__.py` — remove CypherTool/SchemaTool
- All test files listed above

**Files deleted:**
- `imas_codex/server.py`
- `imas_codex/health.py`
- `imas_codex/resource_provider.py`
- `imas_codex/tools/cypher_tool.py`
- `imas_codex/tools/schema_tool.py`

### Phase 2: Add `--read-only` Flag

Add `read_only` parameter to `AgentsServer`. When True, suppress registration
of write tools and the Python REPL. This is the minimum needed for container
deployments.

**Files modified:**
- `imas_codex/llm/server.py` — conditional tool registration
- `imas_codex/cli/serve.py` — `--read-only` CLI flag

### Phase 3: Label-Driven Validation (Future)

Add label requirements to tools and validate at call time. This enables
DD-only containers to report clear errors when signal/doc/code tools are
called against a graph without those labels.

**Deferred:** This is not blocking and adds complexity. The current server
works correctly for all current deployment scenarios (full graph).

## Deployment Topology

### Developer Workstation (Current)

```
Command: imas-codex serve (or: imas-codex serve agents — preserved as alias)
Graph: Full graph on titan (tunneled)
Tools: All 30 tools
Mode: read-write (default)
Transport: stdio (for Copilot CLI / VS Code)
```

### DD-Only Container (Future: imas-dd.iter.org)

```
Command: imas-codex serve --read-only --transport streamable-http
Graph: DDVersion, IMASNode, IMASCluster, Unit, IdentifierSchema
Tools: 16 IMAS DD tools + get_graph_schema (no write/REPL/facility/search)
Labels present: {IMASNode, DDVersion, IMASCluster, ...}
```

### Facility Container (Future: TCV, JET)

```
Command: imas-codex serve --read-only --transport streamable-http
Graph: Full DD + FacilitySignal + WikiPage + CodeFile + Facility
Tools: All read-only tools (26 of 30)
Labels present: All
```

## Alignment with `dynamic-tool-registration.md`

This plan **absorbs and extends** the dynamic-tool-registration plan:

| Dynamic Registration Plan | This Plan | Status |
|---------------------------|-----------|--------|
| Single `serve` command | ✅ Adopted as Phase 1 | Identical |
| `requires_labels` on tool classes | Deferred to Phase 3 | Not blocking |
| `--read-only` flag | ✅ Adopted as Phase 2 | Identical |
| Remove `find_data_nodes` | Not removed (REPL only) | Low priority |
| Label-driven discovery via `db.labels()` | Deferred to Phase 3 | Not blocking |
| Consolidation table | ✅ Extended with full assessment | Enhanced |

The `dynamic-tool-registration.md` plan should be updated to reference this plan
as the superseding document. Its Phases 1-3 map directly to our Phases 1-2,
and its Phase 3 maps to our Phase 3 (deferred).

## Security Considerations

- **Read-only mode** is enforced at registration time — write tool functions are
  never exposed to MCP clients, not just disabled at call time
- **Python REPL** is suppressed in read-only mode (highest-risk tool)
- **CypherTool removal** eliminates a direct Cypher execution path that was
  less controlled than the REPL's `query()` function
- Container deployments MUST use `--read-only`
- Bearer token auth via nginx prevents unauthorized access (existing pattern)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tests break on import changes | High | Low | Systematic test updates in Phase 1 |
| External scripts reference `serve imas` | Low | Medium | Keep as deprecated alias initially |
| Tool descriptions too long | Low | Low | Assessed — no optimization needed |
| Read-only bypass via REPL | N/A | High | REPL suppressed in read-only mode |
