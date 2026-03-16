# Codex IMAS Gap Closure Plan

**Status:** Implemented  
**Date:** 2026-03-16

This plan has been executed on the graph-backed Codex IMAS toolchain. The
clean-break legacy file-backed IMAS search/path/cluster modules have been
retired from the supported Python path, shared DD tooling contracts have been
normalized, and the targeted regression suites listed in this plan pass in the
current repository state.

## Objective

Bring the graph-backed Codex IMAS toolchain to parity with, and then beyond, the legacy IMAS MCP server for all supported IMAS workflows, with a clean-break implementation strategy: fix the shared/base graph-backed functions first, remove conflicting legacy paths where appropriate, and do not preserve backward compatibility for broken or superseded contracts.

This plan has been re-validated against the current codebase. It distinguishes between:

- gaps that were observed during the comparative MCP review and are still open,
- gaps that have already been partially or fully fixed in shared code,
- product-contract gaps where shared functionality exists but is not exposed or not formatted correctly,
- structural cleanup work needed to eliminate split behavior between legacy file-backed tool implementations and the graph-backed Codex path.

## Why This Matters

The same shared graph-backed functions are reused by more than the MCP wrappers:

- `imas_codex/ids/tools.py` uses `GraphSearchTool.search_imas_paths()` and `GraphPathTool.fetch_imas_paths()` to gather DD context for the mapping pipeline.
- `imas_codex/ids/mapping.py` calls `VersionTool.get_dd_version_context()` while building prompt context.
- `imas_codex/llm/server.py` exposes the same shared results to MCP clients and formats them for LLM-facing output.

That means wrapper-only fixes would leave the CLI/discover/mapping context path unstable. The base-layer contracts need to be corrected first.

## Current State After Code Validation

The following items from earlier diagnostics are already implemented in the current base code and should not remain as primary remediation items in this plan:

- `_normalize_paths()` already handles JSON-array transport input.
- path annotation stripping is already centralized through `imas_codex/core/paths.py`.
- cluster semantic search already uses the `cluster_embedding` vector index rather than `cluster_description_embedding`.
- unlabeled cluster search results are already enriched with member-derived context.
- `_resolve_physics_domain(...)` already exists and is used by `GraphStructureTool.export_imas_domain()`.
- `fetch_imas_paths()` already returns `introduced_after_version`, `structure_reference`, `lifecycle_status`, `lifecycle_version`, `node_type`, and coordinate references from the graph.

Those items should now be treated as completed precursor work. The remaining plan below focuses only on open gaps and on bringing the MCP-facing and prompt-context-facing surfaces into a coherent, graph-backed, clean-break implementation.

## Confirmed Open Gap Inventory

### 1. Search formatter and error contract mismatch

**Observed failure**

- Codex `search_imas` failed with `'ToolError' object has no attribute 'hits'`.
- Codex `search_imas_clusters` failed with `'ToolError' object has no attribute 'get'`.

**Confirmed root cause**

- Shared tools in `imas_codex/tools/graph_search.py` are decorated with `@handle_errors(...)` and therefore return `ToolError` objects on failure rather than raising.
- `format_search_imas_report()` in `imas_codex/llm/search_formatters.py` assumes a typed success result and accesses `result.hits` unconditionally.
- `format_cluster_report()` assumes a dict-like success result and calls `result.get(...)` unconditionally.
- `imas_codex/llm/server.py` passes tool results directly into those formatters without normalizing `ToolError` or mixed result types.

**Impact**

- Search tools fail noisily for MCP users.
- Shared search failures are masked behind secondary formatter crashes, which makes root-cause diagnosis harder.
- Prompt-context consumers that correctly guard against `ToolError` behave differently from MCP consumers, so the same base function has inconsistent failure semantics across product surfaces.

### 2. `fetch_imas_paths()` data model mismatch

**Observed failure**

- Codex `fetch_imas_paths` failed with Pydantic validation errors on `cluster_labels`.

**Confirmed root cause**

- `GraphPathTool.fetch_imas_paths()` builds `IdsNode(cluster_labels=[...strings...])`.
- `IdsNode` in `imas_codex/core/data_model.py` currently defines `cluster_labels: list[dict[str, str]] | None`.
- `format_fetch_paths_report()` in `imas_codex/llm/search_formatters.py` also assumes string-like labels and joins them directly.

This is a hard contract mismatch across the base data model, shared graph tool, and formatter layer.

**Impact**

- Fetch fails in MCP.
- The same mismatch is a latent risk for any other consumer that instantiates `IdsNode` from graph-backed fetch results.

### 3. MCP and shared exposures for DD version tooling are incomplete and inconsistent

**Observed gap**

- `get_dd_version_context` exists and is exposed, but its formatted output collapses multiple distinct states into `No version context found...`.
- `get_dd_versions` exists on `VersionTool` and is registered through generic `@mcp_tool` registration, but it is not part of the explicit Codex MCP formatting surface in `imas_codex/llm/server.py` and is not surfaced through the `Tools` convenience delegator.

**Confirmed root cause**

- There is a real product gap between implemented shared capability and the explicitly maintained Codex MCP presentation layer.
- The current version-context formatter in `imas_codex/llm/server.py` collapses the result to `No version context found...` without distinguishing not-found paths, no-change paths, or graph sparsity.

**Impact**

- Users cannot inspect DD version coverage separately from per-path change history.
- Dynamic prompt context gets less explainable version metadata than the graph can theoretically provide.

### 4. Domain export is partially fixed in the base tool but still under-specified end-to-end

**Observed gap**

- During comparison, `export_imas_domain` behaved inconsistently and appeared effectively unusable from the Codex MCP path.

**Current code status**

- `GraphStructureTool.export_imas_domain()` in `imas_codex/tools/graph_search.py` already contains `_resolve_physics_domain(...)`-based resolution and returns `resolved_domains` plus `resolution`.
- `format_export_domain_report()` in `imas_codex/llm/search_formatters.py` ignores those fields and only prints the original `domain`, total counts, and grouped paths.
- There is no focused regression coverage proving the full MCP flow is now correct for friendly aliases, substring matches, and no-match diagnostics.

**Likely remaining gap**

- The base function appears partially improved, but the formatted MCP output does not surface resolution details or empty-state diagnostics.
- There is no regression evidence yet that friendly names such as `magnetics` and ambiguous names such as `magnetic` consistently produce explainable output through the full MCP stack.

**Impact**

- Users cannot tell whether a domain alias resolved correctly, resolved partially, or failed.
- This reduces trust in the graph-backed export path even if the underlying resolver is now functional.

### 5. Shared result contracts are inconsistent across tool families

**Observed pattern**

- Some shared graph tools return typed result models.
- Some return plain dicts.
- Some return `ToolError` objects via decorators.
- Formatters in `imas_codex/llm/search_formatters.py` are written against individual success shapes and mostly do not normalize errors first.

**Confirmed examples**

- `search_imas_paths()` returns a typed `SearchPathsResult`.
- `search_imas_clusters()` returns a dict.
- both may return `ToolError` because both are wrapped by `@handle_errors(...)`.

**Impact**

- Every formatter call site must know the full cross-product of possible payload types.
- Small internal changes in tool behavior can surface as runtime crashes in MCP formatting instead of controlled user-visible errors.

### 6. Structure and leaf detection are wrong in multiple graph-backed tools

**Observed gap**

- `GraphListTool.list_imas_paths()` uses the correct uppercase leaf filter: `['STRUCTURE', 'STRUCT_ARRAY']`.
- other graph-backed methods still use lowercase `'structure'`, which does not match the stored graph values.

**Confirmed root cause**

- `build_dd.py` writes `data_type` values in uppercase such as `STRUCTURE` and `STRUCT_ARRAY`.
- the following shared graph-backed paths still compare with lowercase `'structure'`:
  - `GraphStructureTool.analyze_imas_structure()`
  - `GraphStructureTool.export_imas_ids()`
  - `_text_search_imas_paths()`
- those comparisons are case-sensitive and therefore incorrect.

**Impact**

- structure metrics are wrong,
- leaf counts are inflated,
- `STRUCT_ARRAY` nodes are not excluded where they should be,
- lexical scoring in `_text_search_imas_paths()` incorrectly treats structures as leaf-like data paths.

### 7. Prompt-context call paths need explicit regression coverage

**Observed risk**

- The user explicitly called out dynamic context injection in discover and IMAS CLI prompts as critical.
- `imas_codex/ids/tools.py` already guards some `ToolError` paths by returning empty lists, but those guards are narrow and not systematically tested.

**Impact**

- A fix that only repairs MCP formatting can still leave the mapping/discover path degraded.
- A base-layer contract change can silently break prompt assembly if the prompt helpers are not exercised end-to-end.

### 8. `fetch_error_fields` is implemented only in the MCP layer

**Observed gap**

- `fetch_error_fields` is currently implemented inline in `imas_codex/llm/server.py` using raw Cypher and a fresh `GraphClient`.
- there is no shared graph-backed tool method for this functionality.

**Impact**

- MCP is not using the same reusable base functionality as other IMAS DD operations.
- prompt-context and other internal consumers cannot reuse the same error-field lookup without duplicating query logic.
- this violates the requirement to fix and centralize the backing functions rather than leaving logic in wrappers.

### 9. Legacy file-backed tool implementations remain in the codebase and compete with the graph-backed architecture

**Observed gap**

- `imas_codex/tools/search_tool.py`, `imas_codex/tools/path_tool.py`, and `imas_codex/tools/clusters_tool.py` still exist as legacy file/document-store implementations.
- the active `Tools` provider uses `GraphSearchTool`, `GraphPathTool`, and `GraphClustersTool` from `graph_search.py` instead.

**Impact**

- the repository still contains two competing implementations for the same conceptual tool families.
- maintenance and future bug-fixing can drift if developers patch the wrong layer.
- this directly conflicts with a clean-break model where Codex is graph-backed and no backward compatibility is required.

## Non-Goals

- Rewriting the MCP layer around a new transport model.
- Large schema redesign beyond what is necessary to stabilize shared result contracts.
- Fixing unrelated facility/document/code-search issues outside IMAS/DD scope.

## Design Principles For The Clean Break

- The graph-backed implementations under `imas_codex/tools/graph_search.py` and `imas_codex/tools/version_tool.py` are the source of truth.
- MCP wrappers in `imas_codex/llm/server.py` should format and expose shared results, not contain unique IMAS-DD business logic.
- Legacy file-backed tool implementations for IMAS search/path/cluster functionality should be retired or explicitly isolated from the supported path.
- We do not preserve backward compatibility for incorrect result types, incomplete wrappers, or split implementations.
- Every IMAS-facing formatter must handle `ToolError` explicitly.
- Every prompt-context helper that uses shared tools must have regression coverage.

## Remediation Strategy

## Workstream 1: Normalize Error and Success Contracts

### Goal

Make all IMAS-facing shared tools return results that can be safely inspected by both MCP formatters and non-MCP consumers.

### Changes

1. Add a shared normalization helper for formatter inputs.

- Create a small adapter in the formatter layer, for example `normalize_tool_result(result)`, that can detect:
  - typed success result
  - dict success result
  - `ToolError`
- Use it before any field access in:
  - `format_search_imas_report()`
  - `format_cluster_report()`
  - `format_fetch_paths_report()`
  - `format_export_domain_report()`
  - any other formatter that currently assumes success shape only.

2. Standardize `ToolError` rendering.

- Define one formatter path for `ToolError` that prints:
  - primary error message
  - fallback suggestions when present
  - fallback data summary when present
- This removes the current pattern where a genuine tool failure turns into a secondary formatter exception.

3. Decide and document one preferred shared contract.

- Keep typed result models where they already exist and are useful.
- Keep dict outputs where the payload is inherently free-form.
- But document, in code, that every formatter must handle `ToolError` first.

4. Remove MCP-wrapper-only assumptions.

- The Codex MCP `search_imas` wrapper in `imas_codex/llm/server.py` must not pass unchecked tool results into `format_search_imas_report()`.
- The cluster wrapper must also normalize `ToolError` before calling `format_cluster_report()`.
- Align the REPL and Codex MCP error behavior so they do not diverge for the same shared tool failure.

### Rationale

The minimal safe change is not to remove `@handle_errors(...)`, because the prompt-context helpers already rely on non-raising behavior. The lower-risk change is to make all downstream consumers handle `ToolError` explicitly and consistently.

## Workstream 2: Fix `IdsNode.cluster_labels` at the Base Contract

### Goal

Remove the Pydantic mismatch in `fetch_imas_paths()` by aligning the shared data model, graph query assembly, and formatter expectations.

### Changes

1. Choose the correct semantic contract for `cluster_labels`.

- Based on current graph query behavior and formatter usage, the actual payload is a list of label strings.
- If richer cluster metadata is needed later, add a new field such as `clusters` or `cluster_metadata` instead of overloading `cluster_labels`.

2. Update `IdsNode` in `imas_codex/core/data_model.py`.

- Change `cluster_labels` from `list[dict[str, str]] | None` to `list[str] | None` if that matches intended behavior.
- Audit any other model consumers expecting dicts.

3. Keep `GraphPathTool.fetch_imas_paths()` simple.

- Preserve the current graph query that collects `c.label` values.
- Filter nulls and duplicates, but do not invent structured dicts unless the graph query is expanded to fetch structured cluster metadata.

4. Make `format_fetch_paths_report()` resilient.

- Render string labels directly.
- If future callers pass dict-like cluster metadata, degrade gracefully rather than failing.

### Rationale

This is a base-model defect. Fixing it in the MCP wrapper would leave `IdsNode` invalid for other shared consumers.

## Workstream 3: Fix Structure/Leaf Semantics Everywhere

### Goal

Make all graph-backed IMAS DD tools agree on what constitutes a structure node versus a data leaf.

### Changes

1. Replace lowercase `'structure'` comparisons with the graph’s canonical uppercase values.

- Use `NOT IN ['STRUCTURE', 'STRUCT_ARRAY']` where the code is trying to identify leaf/data nodes.
- Audit all remaining uses in:
  - `GraphStructureTool.analyze_imas_structure()`
  - `GraphStructureTool.export_imas_ids()`
  - `_text_search_imas_paths()`

2. Normalize this logic into a shared helper.

- Add a small helper or constant for structure-like data types so this logic is not re-implemented with inconsistent casing.

3. Update dependent analytics and formatting expectations.

- Any summary or scoring logic that depends on leaf-only behavior should be updated to use the canonical helper.

### Rationale

This is a shared graph semantic bug, not a wrapper bug. It affects correctness of metrics, ranking, and exported structures.

## Workstream 4: Close the Version-Context Product Gap

### Goal

Make DD version metadata and change history usable and explainable from both MCP and shared non-MCP flows.

### Changes

1. Expose `get_dd_versions` through the shared `Tools` delegator and explicit Codex MCP layer.

- Add delegation in `imas_codex/tools/__init__.py`.
- Add MCP exposure and formatting in `imas_codex/llm/server.py`.
- Add a small formatter that clearly shows current version, range, count, and chain.

2. Improve `get_dd_version_context` empty-state reporting.

- Distinguish between:
  - queried path not found in graph
  - path found but no notable changes
  - graph has no `IMASNodeChange` coverage for that DD region
- Surface that distinction in `_format_version_context_report()`.

3. Add graph coverage diagnostics.

- Add a lightweight summary field from `VersionTool.get_dd_version_context()` such as:
  - `paths_found`
  - `paths_without_changes`
  - `graph_change_nodes_seen`
- This makes dynamic prompt assembly easier to debug when version context is legitimately sparse.

### Rationale

The version tool is already part of prompt assembly. The remaining work is mostly exposure, observability, and better result semantics.

## Workstream 5: Finish and Validate Domain Export Behavior

### Goal

Ensure `export_imas_domain` is transparent, testable, and explainable end-to-end.

### Changes

1. Preserve the base-domain resolver already present in `GraphStructureTool.export_imas_domain()`.

- Do not move alias resolution into wrappers.
- Treat the graph-backed resolver as the source of truth.

2. Improve the formatter to show resolution.

- Update `format_export_domain_report()` to print:
  - requested domain
  - resolved domain(s)
  - resolution mode
  - empty-state reason if no domains resolved or no paths matched after resolution.

3. Add explicit alias-resolution tests.

- Cover canonical name, friendly IDS-style name, partial match, and miss case.

4. Add MCP regression coverage.

- Validate that the formatted output is non-empty and explanatory for both success and no-match cases.

### Rationale

This area may already be partially fixed in the base function, but it still needs end-to-end validation and visible output improvements.

## Workstream 6: Move `fetch_error_fields` Into Shared Tooling

### Goal

Move error-field retrieval out of the MCP wrapper and into the shared graph-backed IMAS tool layer.

### Changes

1. Add a shared method in graph-backed tooling.

- Implement `fetch_error_fields(...)` in `GraphPathTool` or a dedicated shared graph-backed IMAS DD helper.
- The method should return structured data, not formatted text.

2. Update the MCP wrapper to delegate.

- `imas_codex/llm/server.py` should call the shared method and format the returned structure.

3. Add tests for both the shared result and the formatted MCP output.

### Rationale

This removes IMAS-DD business logic from wrappers and makes the feature reusable by prompt-context flows.

## Workstream 7: Remove or Isolate Legacy File-Backed IMAS Tool Implementations

### Goal

Eliminate ambiguity about which IMAS tool implementation is supported.

### Changes

1. Identify all dead or unsupported legacy tool files.

- `imas_codex/tools/search_tool.py`
- `imas_codex/tools/path_tool.py`
- `imas_codex/tools/clusters_tool.py`

2. Remove them if unused.

- Because backward compatibility is not required, the preferred outcome is deletion if no supported code path imports them.

3. If temporary retention is required during implementation, isolate them clearly.

- Mark them unsupported and prevent accidental registration or reuse.
- Then delete them in the final cleanup phase.

### Rationale

The clean-break requirement means the codebase should not keep two competing implementations for the same supported IMAS tool family.

## Workstream 8: Add Prompt-Context Regression Coverage

### Goal

Prove that fixes in shared DD tools improve the prompt-context injection paths instead of only repairing MCP output.

### Changes

1. Add focused tests around `imas_codex/ids/tools.py`.

- `fetch_imas_fields()` should return stable dict output after the `IdsNode` contract fix.
- `search_imas_semantic()` should degrade cleanly when the underlying tool returns `ToolError`.
- `fetch_imas_subtree()` should keep returning structured path rows.

2. Add a prompt-assembly regression around `imas_codex/ids/mapping.py`.

- Exercise the version-context block so the prompt builder still works when:
  - version history exists
  - version history is empty
  - some paths are not found.

3. Add one end-to-end MCP formatting regression per broken tool.

- `search_imas`
- `search_imas_clusters`
- `fetch_imas_paths`
- `export_imas_domain`
- `get_dd_version_context`

### Rationale

The user’s priority is not just tool correctness in isolation. It is dependable dynamic context injection for prompting workflows.

## Implementation Sequence

### Phase 0: Confirm and Freeze the Supported Architecture

1. Declare graph-backed IMAS tooling as the only supported path.
2. Identify legacy file-backed IMAS tool modules slated for removal.
3. Document the clean-break contract in code comments or module docstrings where needed.

### Phase 1: Repair Base Contracts and Formatter Safety

1. Fix `IdsNode.cluster_labels` typing.
2. Add formatter-side `ToolError` normalization.
3. Update `format_search_imas_report()` and `format_cluster_report()` to handle both error and empty success states.
4. Update `format_fetch_paths_report()` and `format_export_domain_report()` to handle `ToolError` and empty success states.
5. Add direct unit tests for those formatter paths.

### Phase 2: Repair Shared Graph Semantics

1. Replace incorrect lowercase structure checks with canonical uppercase structure constants.
2. Centralize structure-like type detection in a shared helper or constant.
3. Re-test `analyze_imas_structure()`, `export_imas_ids()`, and `_text_search_imas_paths()`.

### Phase 3: Complete Version and Error-Field Shared Features

1. Expose `get_dd_versions` through `Tools` and MCP.
2. Improve `get_dd_version_context` reporting semantics.
3. Move `fetch_error_fields` into shared graph-backed tooling.
4. Add formatter tests and MCP regression tests.

### Phase 4: Domain Export Hardening

1. Verify `_resolve_physics_domain(...)` behavior against representative aliases.
2. Update `format_export_domain_report()` to surface resolution details.
3. Add end-to-end regression coverage.

### Phase 5: Prompt-Context Validation

1. Add/extend tests in the mapping-context helpers.
2. Run the relevant IMAS and mapping test subsets.
3. Confirm that shared tools still degrade safely under `ToolError` without breaking prompt construction.

### Phase 6: Remove Legacy IMAS Tool Paths

1. Delete or isolate unused legacy file-backed IMAS tool modules.
2. Remove any stale tests or references tied to unsupported implementations.
3. Re-run the full IMAS tool regression set on the graph-backed path only.

## Test Plan

### Unit tests

- `IdsNode` accepts graph-backed fetch payloads with `cluster_labels` as strings.
- `format_search_imas_report()` renders `ToolError` without crashing.
- `format_cluster_report()` renders `ToolError` without crashing.
- `format_fetch_paths_report()` renders empty, success, and error cases.
- `format_export_domain_report()` renders resolution metadata and empty-state reasons.
- `_format_version_context_report()` distinguishes no changes from not found.
- structure-type helper correctly classifies `STRUCTURE` and `STRUCT_ARRAY` as non-leaf.

### Shared-tool integration tests

- `GraphPathTool.fetch_imas_paths()` returns a valid `FetchPathsResult` for a known path.
- `GraphSearchTool.search_imas_paths()` returns typed hits and tolerates backend failures via `ToolError`.
- `GraphClustersTool.search_imas_clusters()` returns explainable output for text search, path search, and error cases.
- `GraphStructureTool.export_imas_domain()` resolves friendly domain names and returns grouped results.
- `VersionTool.get_dd_versions()` and `VersionTool.get_dd_version_context()` return consistent metadata.
- `GraphStructureTool.analyze_imas_structure()` returns correct leaf and structure counts.
- `GraphStructureTool.export_imas_ids()` honors leaf filtering for `STRUCTURE` and `STRUCT_ARRAY`.
- shared `fetch_error_fields` returns reusable structured data.

### Prompt-context regressions

- `imas_codex/ids/tools.py::fetch_imas_fields`
- `imas_codex/ids/tools.py::search_imas_semantic`
- prompt assembly around the version-context block in `imas_codex/ids/mapping.py`

## Risks and Tradeoffs

### Risk: changing `IdsNode` may affect existing callers

Mitigation:

- grep all `cluster_labels` consumers before changing the type.
- use a compatibility formatter path that accepts both strings and dicts during the transition.

### Risk: hiding too much detail behind generic `ToolError` rendering

Mitigation:

- keep original error text.
- include `suggestions` and `fallback_data` when present.
- log structured error context for debugging.

### Risk: version-context sparsity is a data issue rather than a code issue

Mitigation:

- do not treat empty result as a hard failure.
- expose coverage diagnostics so users and developers can tell whether the graph lacks change metadata.

### Risk: deleting legacy tool implementations may expose hidden imports

Mitigation:

- grep all imports before removal.
- remove in a dedicated cleanup phase after graph-backed regressions are green.
- prefer deletion over indefinite coexistence.

## Definition of Done

- `search_imas` no longer crashes when the shared search tool returns `ToolError`.
- `search_imas_clusters` no longer crashes when the shared cluster tool returns `ToolError`.
- `fetch_imas_paths` returns valid results for known paths without Pydantic type failures.
- all shared structure/leaf calculations use canonical uppercase graph values.
- `get_dd_versions` is exposed through the supported Codex MCP surface.
- `get_dd_version_context` provides explainable empty states.
- `fetch_error_fields` is backed by shared graph tooling rather than inline MCP-only Cypher.
- `export_imas_domain` shows resolved-domain diagnostics and passes end-to-end regression tests.
- Prompt-context helpers in `imas_codex/ids/tools.py` and `imas_codex/ids/mapping.py` are covered by regression tests for the repaired contracts.
- unsupported legacy file-backed IMAS tool modules are removed or explicitly isolated from the supported path.

## Recommended File Targets

- `imas_codex/core/data_model.py`
- `imas_codex/tools/graph_search.py`
- `imas_codex/tools/__init__.py`
- `imas_codex/tools/version_tool.py`
- `imas_codex/llm/search_formatters.py`
- `imas_codex/llm/server.py`
- `imas_codex/ids/tools.py`
- `imas_codex/ids/mapping.py`
- `imas_codex/tools/search_tool.py`
- `imas_codex/tools/path_tool.py`
- `imas_codex/tools/clusters_tool.py`
- relevant tests under `tests/tools/`, `tests/ids/`, and `tests/llm/` or their nearest existing equivalents