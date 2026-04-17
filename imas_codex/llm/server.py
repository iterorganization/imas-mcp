"""
Agents MCP Server - Streamlined tools for LLM-driven facility exploration.

This server provides 15 MCP tools organized by purpose:

Unified Search (primary interface):
- search_signals: Signal search with data access and IMAS enrichment
- search_docs: Wiki/document/image search with cross-links
- search_code: Code search with data reference enrichment
- search_dd_paths: IMAS DD search with cluster and facility cross-refs

Retrieval:
- fetch: Full content retrieval by ID or URL (WikiPage, Document, CodeFile, Image)

Graph Operations:
- get_graph_schema: Schema introspection for query generation
- add_to_graph: Schema-validated node creation with privacy filtering

Facility Infrastructure (Private Data):
- update_facility_infrastructure: Deep-merge update to private YAML
- get_facility_infrastructure: Read private infrastructure data
- add_exploration_note: Append timestamped exploration note

Configuration:
- update_facility_config: Read/update facility config (public or private)
- get_discovery_context: Get facility discovery context

Log Inspection:
- list_logs: List available log files with sizes and timestamps
- get_logs: Read logs with level/grep/time filtering
- tail_logs: Get most recent log entries

Advanced:
- python: Persistent Python REPL for custom queries

The python() REPL provides advanced operations not covered by search tools:
- Graph: query(), semantic_search(), embed()
- Domain: find_signals(), find_wiki(), find_dd_paths(), find_code(), graph_search()
- Remote: run(), check_tools() (auto-detects local vs SSH)
- Facility: get_facility(), get_exploration_targets(), get_tree_structure()
- IMAS DD: search_dd_paths(), fetch_dd_paths(), list_dd_paths(), check_dd_paths()
- COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()

Use search_* MCP tools for:
- Common signal, documentation, code, and IMAS lookups
- Formatted reports with enriched results in one call

Use python() for:
- Complex multi-step operations requiring state
- Graph queries with Cypher
- Chained processing with intermediate logic
- IMAS/COCOS domain-specific operations
- Better discoverability and documentation

REPL state is initialized lazily on first use to avoid import deadlocks.
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable
from ruamel.yaml import YAML

from imas_codex.llm._warmup import warmup
from imas_codex.llm.prompt_loader import (
    PromptDefinition,
    load_prompts,
)

# Start background warmup threads immediately so that by the time the first
# tool call arrives (after the MCP stdio handshake), slow imports are ready.
warmup.start()

logger = logging.getLogger(__name__)

# Configure ruamel.yaml for comment-preserving round-trips
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120

# Neo4j connection error message
NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Check service with: systemctl --user status imas-codex-neo4j"
)


def _serialize_neo4j_value(value: Any) -> Any:
    """Serialize Neo4j values to JSON-compatible types."""
    if value is None:
        return None
    if hasattr(value, "isoformat") and hasattr(value, "tzinfo"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_neo4j_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_neo4j_value(v) for v in value]
    return value


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    msg = str(e)
    if "Connection refused" in msg or "ServiceUnavailable" in msg:
        return NEO4J_NOT_RUNNING_MSG
    if "critical error" in msg.lower() or "needs to be restarted" in msg.lower():
        return (
            "Neo4j database has entered a critical error state and needs to be "
            "restarted. Run: imas-codex graph stop && imas-codex graph start"
        )
    return msg


# =============================================================================
# Persistent Python REPL - Initialized lazily on first use
# =============================================================================

_repl_globals: dict[str, Any] | None = None
_repl_lock = threading.Lock()
_imas_tools_instance = None
_no_embed: bool = False  # Set by AgentsServer when --no-embed is passed

# ---------------------------------------------------------------------------
# Module-level placeholders — populated by _require_warmup() on first tool call
# ---------------------------------------------------------------------------
# Type: Any avoids NameError for code that references these names before warmup.

_get_facility_config: Any = None
get_facility_infrastructure: Any = None
get_facility_validated: Any = None
update_infrastructure: Any = None
update_metadata: Any = None
EncoderConfig: Any = None
Encoder: Any = None
EmbeddingBackendError: Any = None
GraphClient: Any = None
get_schema: Any = None
to_cypher_props: Any = None
get_embedding_location: Any = None
_run: Any = None
_check_all_tools: Any = None
_install_all_tools: Any = None

_warmup_lock = threading.Lock()
_warmup_applied = False

_graph_warmup_lock = threading.Lock()
_graph_warmup_applied = False


def _require_graph_only() -> None:
    """Populate graph-related module globals from background warmup.

    Blocks only on the graph warmup group — NOT on embeddings, discovery,
    or remote.  Use for graph-only tools (Tier 2 DD) that don't need
    semantic search.
    No-op after first call (or after ``_require_warmup()`` has run).
    """
    global _graph_warmup_applied
    global GraphClient, get_schema, to_cypher_props

    if _graph_warmup_applied:
        return
    with _graph_warmup_lock:
        if _graph_warmup_applied:
            return
        graph_ns = warmup.graph()
        GraphClient = graph_ns["GraphClient"]
        get_schema = graph_ns["get_schema"]
        to_cypher_props = graph_ns["to_cypher_props"]
        _graph_warmup_applied = True


def _require_warmup() -> None:
    """Populate module globals from background warmup groups.

    Blocks until all warmup groups are ready (no-op after first call).
    Call at the top of any tool handler that references the module-level
    placeholders above.
    """
    global _warmup_applied, _graph_warmup_applied
    global _get_facility_config, get_facility_infrastructure, get_facility_validated
    global update_infrastructure, update_metadata
    global EncoderConfig, Encoder, EmbeddingBackendError, get_embedding_location
    global GraphClient, get_schema, to_cypher_props
    global _run, _check_all_tools, _install_all_tools

    if _warmup_applied:  # fast path — avoids lock on every call
        return
    with _warmup_lock:
        if _warmup_applied:
            return
        disc = warmup.discovery()
        _get_facility_config = disc["get_facility"]
        get_facility_infrastructure = disc["get_facility_infrastructure"]
        get_facility_validated = disc["get_facility_validated"]
        update_infrastructure = disc["update_infrastructure"]
        update_metadata = disc["update_metadata"]

        graph_ns = warmup.graph()
        GraphClient = graph_ns["GraphClient"]
        get_schema = graph_ns["get_schema"]
        to_cypher_props = graph_ns["to_cypher_props"]

        try:
            emb_ns = warmup.embeddings()
            EncoderConfig = emb_ns["EncoderConfig"]
            Encoder = emb_ns["Encoder"]
            EmbeddingBackendError = emb_ns["EmbeddingBackendError"]
            get_embedding_location = emb_ns["get_embedding_location"]
        except Exception as exc:
            logger.warning(
                "Embedding warmup failed: %s — semantic search will "
                "error at call time. Use 'imas-codex embed start' to fix.",
                exc,
            )

        rem_ns = warmup.remote()
        _run = rem_ns["run"]
        _check_all_tools = rem_ns["check_all_tools"]
        _install_all_tools = rem_ns["install_all_tools"]

        _warmup_applied = True
        _graph_warmup_applied = True  # graph is a subset of full warmup


# =============================================================================
# API Reference — compact task-to-function mapping for python() docstring
# =============================================================================


def _generate_api_reference() -> str:
    """Generate compact API reference with inline parameter names.

    This runs at tool registration time — no REPL init needed.
    Keeps the reference short so agents actually read it.
    """
    return "\n".join(
        [
            "Use search_signals/search_docs/search_code/search_dd_paths for common lookups.",
            "Use python() for custom queries not covered by the search tools.",
            "",
            "REPL functions (for custom queries in python()):",
            "  find_wiki(query, facility=, text_contains=, page_title_contains=, k=10)",
            "  wiki_page_chunks(title_contains, facility=, text_contains=, limit=50)",
            "  find_signals(query, facility=, diagnostic=, physics_domain=, limit=20)",
            "  find_dd_paths(query) | find_code(query, facility=, limit=10)",
            "  find_data_nodes(query, facility=, data_source_name=)",
            "  map_signals_to_imas(facility, diagnostic=, physics_domain=)",
            "  facility_overview(facility)",
            "  graph_search(label, where={}, semantic=, traverse=[], return_props=[], limit=25)",
            "  query(cypher, **params)  — raw Cypher, only if no domain function fits",
            "  semantic_search(text, index=, k=5)",
            "",
            "  Format: as_table(pick(results, 'col1', 'col2'))",
            "  Schema: schema_for(task='wiki') before raw Cypher",
            "  Full API: repl_help()",
            "",
        ]
    )


_graph_client: Any = None
_graph_client_lock = threading.Lock()


def _get_graph_client():
    """Get or create standalone GraphClient singleton (thread-safe).

    Creates a ``GraphClient.from_profile()`` without going through the
    REPL or requiring embedding warmup.  Only blocks on graph warmup.
    """
    global _graph_client
    if _graph_client is not None:
        return _graph_client
    with _graph_client_lock:
        if _graph_client is not None:
            return _graph_client
        _require_graph_only()
        _graph_client = GraphClient.from_profile()
        return _graph_client


_imas_tools_lock = threading.Lock()


def _get_imas_tools(gc: GraphClient | None = None, semantic_search: bool = False):
    """Get or create singleton Tools instance with shared GraphClient.

    All DD tools require the graph. Tools that perform semantic search
    also need the embedding server — pass ``semantic_search=True``
    for those.

    Args:
        gc: Optional pre-existing GraphClient to use.
        semantic_search: If True, also warm up the embedding server
            for vector similarity queries.
    """
    global _imas_tools_instance
    if _imas_tools_instance is not None:
        return _imas_tools_instance
    with _imas_tools_lock:
        if _imas_tools_instance is not None:
            return _imas_tools_instance

        if semantic_search:
            _require_warmup()
        else:
            _require_graph_only()

        from imas_codex.tools import Tools

        if gc is None:
            gc = _get_graph_client()
        _imas_tools_instance = Tools(graph_client=gc)
        return _imas_tools_instance


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


def _format_rename_chains_section(rename_chains: list[dict]) -> list[str]:
    """Render rename chain records as a list of formatted lines.

    Each entry in rename_chains must have keys 'chain' (list of path IDs)
    and 'hops' (int).

    Returns:
        List of string lines (no trailing newline) representing the section.
    """
    if not rename_chains:
        return ["### Rename Chains", "No rename chains found."]

    lines = [f"### Rename Chains ({len(rename_chains)} chain(s))"]
    for entry in rename_chains:
        chain = entry.get("chain") or []
        hops = entry.get("hops", len(chain) - 1)
        if not chain:
            continue
        chain_str = " → ".join(f"`{p}`" for p in chain)
        hop_label = f"{hops} hop{'s' if hops != 1 else ''}"
        lines.append(f"- {chain_str} ({hop_label})")
    return lines


def _format_version_context_report(result: dict) -> str:
    """Format version context result into a readable report."""
    if result.get("error"):
        return f"Error: {result['error']}"

    # ── Bulk query mode ───────────────────────────────────────────────────────
    if result.get("mode") == "bulk_query":
        change_type = result.get("change_type_filter", "?")
        change_count = result.get("change_count", 0)
        ids_affected = result.get("ids_affected", [])
        version_range = result.get("version_range")
        changes = result.get("changes", [])

        range_str = ""
        if version_range:
            frm = version_range.get("from", "")
            to = version_range.get("to", "")
            if frm and to:
                range_str = f" (versions {frm} → {to})"
            elif to:
                range_str = f" (up to v{to})"
            elif frm:
                range_str = f" (after v{frm})"

        lines = [
            f"Bulk Change Query — type: **{change_type}**{range_str}",
            f"{change_count} change(s) found across {len(ids_affected)} IDS: "
            + (", ".join(ids_affected) if ids_affected else "none"),
            "",
        ]
        for c in changes:
            path = c.get("path", "?")
            version = c.get("version", "?")
            severity = c.get("severity", "informational")
            old_val = c.get("old_value") or ""
            new_val = c.get("new_value") or ""
            summary = c.get("summary") or ""
            if old_val and new_val:
                val_str = f": `{old_val}` → `{new_val}`"
            elif new_val:
                val_str = f": added `{new_val}`"
            elif old_val:
                val_str = f": removed `{old_val}`"
            else:
                val_str = ""
            suffix = f" — {summary}" if summary else ""
            lines.append(f"  - v{version} [{severity}] **{path}**{val_str}{suffix}")

        rename_chains = result.get("rename_chains")
        if rename_chains:
            lines.append("")
            lines.extend(_format_rename_chains_section(rename_chains))

        return "\n".join(lines)

    # ── Rename-chain-only mode ────────────────────────────────────────────────
    if "rename_chains" in result and "mode" not in result and "paths" not in result:
        lines: list[str] = []
        rename_chains = result.get("rename_chains", [])
        lines.extend(_format_rename_chains_section(rename_chains))
        return "\n".join(lines) if lines else "No rename chains found."

    # ── Per-path mode ─────────────────────────────────────────────────────────
    paths_data = result.get("paths", {})
    lines = [
        f"Version Context Report: {result.get('total_paths', 0)} paths queried, "
        f"{len(result.get('paths_found', []))} found, "
        f"{result.get('paths_with_changes', 0)} with changes",
        "",
    ]

    if not paths_data:
        not_found = result.get("not_found", [])
        if not_found:
            lines.append(
                f"No matching graph paths found. Not found: {', '.join(not_found)}"
            )
            return "\n".join(lines)
        return "No version context found for the requested paths."

    for path_id, ctx in paths_data.items():
        changes = ctx.get("changes", [])
        introduced = ctx.get("introduced_in")
        deprecated = ctx.get("deprecated_in")
        lifecycle = ctx.get("lifecycle_status")
        renamed_to = ctx.get("renamed_to", [])
        renamed_from = ctx.get("renamed_from", [])
        lifecycle_parts = []
        if lifecycle and lifecycle != "active":
            lifecycle_parts.append(lifecycle)
        if introduced:
            lifecycle_parts.append(f"introduced v{introduced}")
        if deprecated:
            lifecycle_parts.append(f"deprecated v{deprecated}")
        lifecycle_str = f" ({', '.join(lifecycle_parts)})" if lifecycle_parts else ""

        if changes:
            lines.append(f"**{path_id}**{lifecycle_str} — {len(changes)} change(s):")
            for c in changes:
                version = c.get("version", "?")
                change_type = c.get("change_type", "?")
                semantic = c.get("semantic_type")
                old_val = c.get("old_value", "")
                new_val = c.get("new_value", "")
                label = f"{change_type}" + (
                    f"/{semantic}" if semantic and semantic != "none" else ""
                )
                if old_val and new_val:
                    lines.append(f"  - v{version} [{label}]: `{old_val}` → `{new_val}`")
                elif new_val:
                    lines.append(f"  - v{version} [{label}]: added `{new_val}`")
                elif old_val:
                    lines.append(f"  - v{version} [{label}]: removed `{old_val}`")
                else:
                    lines.append(f"  - v{version} [{label}]")
        else:
            lines.append(f"**{path_id}**{lifecycle_str}: no metadata changes recorded")

        # Show immediate rename links
        if renamed_to:
            lines.append(f"  → Renamed to: {', '.join(renamed_to)}")
        if renamed_from:
            lines.append(f"  ← Renamed from: {', '.join(renamed_from)}")

    not_found = result.get("not_found", [])
    if not_found:
        lines.append("")
        lines.append(f"Not found: {', '.join(not_found)}")

    paths_without_changes = result.get("paths_without_changes", [])
    if paths_without_changes:
        lines.append("")
        lines.append(
            f"Paths without notable changes: {', '.join(paths_without_changes)}"
        )

    # Append rename chains if present
    rename_chains = result.get("rename_chains")
    if rename_chains:
        lines.append("")
        lines.extend(_format_rename_chains_section(rename_chains))

    return "\n".join(lines)


def _format_dd_versions_report(result: dict) -> str:
    """Format DD version metadata into readable text."""
    if result.get("error"):
        return f"Error: {result['error']}"

    current_version = result.get("current_version") or "unknown"
    version_range = result.get("version_range") or "unknown"
    version_count = result.get("version_count", 0)
    versions = result.get("versions", []) or []

    lines = [
        "DD Version Metadata",
        f"Current version: {current_version}",
        f"Version range: {version_range}",
        f"Version count: {version_count}",
    ]
    if versions:
        lines.append(f"Version chain: {' -> '.join(versions)}")
    return "\n".join(lines)


def _format_error_fields_report(result: dict) -> str:
    """Format HAS_ERROR traversal results into readable text."""
    if result.get("error"):
        return f"Error: {result['error']}"

    path = result.get("path", "")
    if result.get("not_found"):
        return f"Path not found: {path}"

    error_fields = result.get("error_fields", []) or []
    if not error_fields:
        return f"No error fields found for '{path}'"

    lines = [f"Error fields for {path}:"]
    for ef in error_fields:
        line = f"  {ef.get('path', '')} ({ef.get('error_type', 'unknown')})"
        documentation = ef.get("documentation")
        if documentation:
            line += f" - {documentation[:100]}"
        lines.append(line)
    return "\n".join(lines)


def _init_repl() -> dict[str, Any]:
    """Initialize the persistent REPL environment with all utilities.

    Called lazily on first python() tool invocation. Blocks until background
    warmup completes, then performs I/O-bound setup (Neo4j connection).
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals

    _require_warmup()
    logger.info("Initializing Python REPL...")

    from imas_codex.graph import domain_queries as _dq
    from imas_codex.graph.formatters import as_summary, as_table, pick
    from imas_codex.graph.query_builder import graph_search as _graph_search
    from imas_codex.graph.schema_context import schema_for as _schema_for
    from imas_codex.graph.schema_context_data import VECTOR_INDEXES as _VECTOR_INDEXES

    gc = GraphClient.from_profile()

    # Create encoder with lazy initialization - respects embedding-backend config
    # This will NOT load the model until actually used
    backend = get_embedding_location() if get_embedding_location else "unavailable"
    logger.info(f"Embedding location: {backend}")

    _encoder: Encoder | None = None

    def _get_encoder() -> Encoder:
        """Get or create the encoder, retrying on failure."""
        nonlocal _encoder

        if _encoder is None:
            if EncoderConfig is None or Encoder is None:
                raise (
                    EmbeddingBackendError(
                        "Embedding classes not loaded — warmup failed. "
                        "Check 'imas-codex embed status'."
                    )
                    if EmbeddingBackendError
                    else RuntimeError("Embedding subsystem not available")
                )
            try:
                config = EncoderConfig()
                _encoder = Encoder(config)
                logger.info(f"Encoder initialized (backend={config.backend})")
            except Exception as e:
                logger.error(f"Embedding initialization failed: {e}")
                _err_cls = EmbeddingBackendError or RuntimeError
                raise _err_cls(
                    f"Embedding backend '{backend}' unavailable: {e}. "
                    f"Check configuration or use a different backend."
                ) from e

        return _encoder

    # =========================================================================
    # Core utilities
    # =========================================================================

    def query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute Cypher query and return results.

        Args:
            cypher: Cypher query string
            **params: Query parameters

        Returns:
            List of result records as dicts

        Examples:
            query('MATCH (f:Facility) RETURN f.id, f.name')
            query('MATCH (t:SignalNode {data_source_name: $tree}) RETURN t.path LIMIT 10', tree='results')
        """
        return gc.query(cypher, **params)

    def embed(text: str) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (dimension depends on configured model)

        Raises:
            EmbeddingBackendError: If embedding backend is unavailable
        """
        encoder = _get_encoder()
        embeddings = encoder.embed_texts([text])
        return embeddings[0].tolist()

    def semantic_search(
        text: str,
        index: str = "imas_node_embedding",
        k: int = 5,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on graph embeddings.

        Args:
            text: Query text to embed and search
            index: Vector index name (use get_graph_schema() to list all)
            k: Number of results to return
            include_deprecated: If True, include deprecated IMAS paths in results.
                Only applies to imas_node_embedding index. Default False (active only).

        Returns:
            List of flat dicts with all node properties + score + labels.
            For wiki_chunk_embedding: also includes page_title, page_url.
            For code_chunk_embedding: also includes source_file.

        Raises:
            EmbeddingBackendError: If embedding backend is unavailable
        """
        encoder = _get_encoder()
        embeddings = encoder.embed_texts([text])
        embedding = embeddings[0].tolist()

        # Use index-specific queries for richer context
        if index == "wiki_chunk_embedding":
            return _search_wiki_chunks(embedding, k)
        if index == "code_chunk_embedding":
            return _search_code_chunks(embedding, k)

        # Filter deprecated paths for imas_node_embedding unless explicitly included
        post_where = ""
        if index == "imas_node_embedding" and not include_deprecated:
            post_where = "WHERE NOT (node)-[:DEPRECATED_IN]->(:DDVersion) "

        # Look up label for the given index to use SEARCH clause
        index_meta = _VECTOR_INDEXES.get(index)
        if index_meta is None:
            # Unknown index — fall back to generic MATCH
            results = gc.query(
                "MATCH (node) WHERE any(lbl IN labels(node) WHERE true) "
                "RETURN [k IN keys(node) "
                "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
                "AS properties, labels(node) AS labels, 0.0 AS score "
                "LIMIT $k",
                k=k,
            )
        else:
            node_label, _emb_prop = index_meta
            where_clauses = []
            if post_where:
                where_clauses.append("NOT (node)-[:DEPRECATED_IN]->(:DDVersion)")
            where_str = (
                ("\nWHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            )
            results = gc.query(
                f"CYPHER 25\n"
                f"MATCH (node:{node_label})\n"
                f"SEARCH node IN (\n"
                f"  VECTOR INDEX {index}\n"
                f"  FOR $embedding\n"
                f"  LIMIT $k\n"
                f") SCORE AS score"
                f"{where_str}\n"
                "RETURN [k IN keys(node) "
                "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
                "AS properties, labels(node) AS labels, score "
                "ORDER BY score DESC",
                k=k,
                embedding=embedding,
            )
        # Flatten: properties at top level alongside score and labels
        return [
            {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            for r in results
        ]

    def _search_wiki_chunks(embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Wiki-specific search that enriches results with parent page context."""
        results = gc.query(
            "CYPHER 25\n"
            "MATCH (node:WikiChunk)\n"
            "SEARCH node IN (\n"
            "  VECTOR INDEX wiki_chunk_embedding\n"
            "  FOR $embedding\n"
            "  LIMIT $k\n"
            ") SCORE AS score\n"
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(node) "
            "OPTIONAL MATCH (wa:Document)-[:HAS_CHUNK]->(node) "
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score, "
            "p.title AS page_title, p.url AS page_url, "
            "wa.id AS document_id, wa.title AS document_title, wa.url AS document_url "
            "ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )
        out = []
        for r in results:
            d: dict[str, Any] = {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            # Add parent page context (WikiPage or Document)
            if r.get("page_title"):
                d["page_title"] = r["page_title"]
                d["page_url"] = r["page_url"]
            elif r.get("document_id"):
                d["page_title"] = r["document_title"] or r["document_id"]
                if r.get("document_url"):
                    d["page_url"] = r["document_url"]
            out.append(d)
        return out

    def _search_code_chunks(embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Code-specific search that enriches results with source file context."""
        results = gc.query(
            "CYPHER 25\n"
            "MATCH (node:CodeChunk)\n"
            "SEARCH node IN (\n"
            "  VECTOR INDEX code_chunk_embedding\n"
            "  FOR $embedding\n"
            "  LIMIT $k\n"
            ") SCORE AS score\n"
            "OPTIONAL MATCH (sf:CodeFile)-[:HAS_CHUNK]->(node) "
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score, "
            "sf.path AS source_file, sf.facility_id AS source_facility "
            "ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )
        out = []
        for r in results:
            d: dict[str, Any] = {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            if r.get("source_file"):
                d["source_file"] = r["source_file"]
            if r.get("source_facility"):
                d["source_facility"] = r["source_facility"]
            out.append(d)
        return out

    # =========================================================================
    # Facility utilities
    # =========================================================================

    def get_facility(facility: str) -> dict[str, Any]:
        """Get comprehensive facility info including graph state.

        Loads the full facility config validated against the LinkML schema
        (FacilityConfig model) so all typed fields (data_systems, data_systems,
        data_access_patterns, wiki_sites, etc.) are included.

        Args:
            facility: Facility ID (e.g., 'tcv')

        Returns:
            Dict with config (full LinkML-validated), graph_summary,
            actionable_paths
        """
        result: dict[str, Any] = {"facility": facility}

        # Load facility config via LinkML-validated Pydantic model
        try:
            validated = get_facility_validated(facility)
            result["config"] = validated.model_dump(exclude_none=True)
        except Exception as e:
            # Fall back to raw dict if validation fails
            try:
                data = _get_facility_config(facility)
                result["config"] = data
                result["validation_error"] = str(e)
            except Exception as e2:
                result["error"] = str(e2)
                return result

        # Query graph for facility summary
        try:
            summary = gc.query(
                """
                MATCH (f:Facility {id: $fid})
                OPTIONAL MATCH (a:AnalysisCode)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (d:Diagnostic)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (t:TDIFunction)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (m:DataSource)-[:AT_FACILITY]->(f)
                RETURN
                    count(DISTINCT a) AS analysis_codes,
                    count(DISTINCT d) AS diagnostics,
                    count(DISTINCT t) AS tdi_functions,
                    count(DISTINCT m) AS mdsplus_trees
                """,
                fid=facility,
            )
            if summary:
                result["graph_summary"] = summary[0]

            # Get actionable paths
            actionable = gc.query(
                """
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $fid})
                WHERE p.status = 'discovered'
                RETURN p.path AS path, p.score_composite AS score, p.description AS description
                ORDER BY COALESCE(p.score_composite, 0) DESC
                LIMIT 20
                """,
                fid=facility,
            )
            result["actionable_paths"] = actionable
        except Exception as e:
            result["graph_error"] = str(e)

        return result

    def get_exploration_targets(facility: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get prioritized exploration targets for a facility.

        Args:
            facility: Facility ID
            limit: Maximum targets to return

        Returns:
            List of targets with priority, type, action, and effort
        """
        targets: list[dict[str, Any]] = []

        try:
            # Get MDSplus tree coverage
            trees = gc.query(
                """
                MATCH (t:DataSource)-[:AT_FACILITY]->(f:Facility {id: $fid})
                OPTIONAL MATCH (n:SignalNode {data_source_name: t.name})-[:AT_FACILITY]->(f)
                RETURN t.name AS tree,
                       t.node_count_total AS total,
                       count(DISTINCT n) AS ingested
                ORDER BY t.name
                """,
                fid=facility,
            )

            for row in trees:
                total = row["total"] or 0
                ingested = row["ingested"] or 0
                if total > 0:
                    pct = round(100 * ingested / total, 1)
                    if pct == 0:
                        targets.append(
                            {
                                "priority": 1,
                                "type": "mdsplus_tree",
                                "target": row["tree"],
                                "action": f"Ingest {row['tree']} tree ({total} nodes)",
                                "effort": "high" if total > 1000 else "medium",
                            }
                        )
                    elif pct < 10:
                        targets.append(
                            {
                                "priority": 2,
                                "type": "mdsplus_tree",
                                "target": row["tree"],
                                "action": f"Continue {row['tree']} ({pct}% complete)",
                                "effort": "medium",
                            }
                        )

            # Get discovered paths
            paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $fid})
                WHERE p.status = 'discovered'
                RETURN p.path AS path, p.score_composite AS score
                ORDER BY COALESCE(p.score_composite, 0) DESC
                LIMIT 5
                """,
                fid=facility,
            )

            for row in paths:
                targets.append(
                    {
                        "priority": 3,
                        "type": "facility_path",
                        "target": row["path"],
                        "action": f"Explore {row['path']}",
                        "score": row["score"],
                        "effort": "medium",
                    }
                )

        except Exception as e:
            targets.append({"error": str(e)})

        targets.sort(key=lambda x: x.get("priority", 99))
        return targets[:limit]

    def get_tree_structure(
        data_source_name: str, path_prefix: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get SignalNode structure from the graph.

        Args:
            data_source_name: MDSplus tree name (e.g., 'results', 'magnetics')
            path_prefix: Optional path prefix to filter
            limit: Maximum nodes to return

        Returns:
            List of {path, description, unit, physics_domain}
        """
        if path_prefix:
            return gc.query(
                """
                MATCH (n:SignalNode {data_source_name: $tree})
                WHERE n.path STARTS WITH $prefix
                RETURN n.path AS path, n.description AS description,
                       n.unit AS unit, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=data_source_name,
                prefix=path_prefix,
                limit=limit,
            )
        else:
            return gc.query(
                """
                MATCH (n:SignalNode {data_source_name: $tree})
                RETURN n.path AS path, n.description AS description,
                       n.unit AS unit, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=data_source_name,
                limit=limit,
            )

    # =========================================================================
    # Code search utilities
    # =========================================================================

    def search_code(
        query_text: str,
        top_k: int = 5,
        facility: str | None = None,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search code examples using semantic similarity.

        Args:
            query_text: Natural language query
            top_k: Maximum results to return
            facility: Optional facility filter
            min_score: Minimum similarity score

        Returns:
            List of code results with content, source_file, score
        """
        from imas_codex.ingestion.search import search_code_chunks

        results = search_code_chunks(
            query=query_text,
            top_k=top_k,
            facility=facility,
            min_score=min_score,
        )
        return [
            {
                "content": r.content[:500] + "..."
                if len(r.content) > 500
                else r.content,
                "function_name": r.function_name,
                "source_file": r.source_file,
                "facility_id": r.facility_id,
                "score": round(r.score, 3),
            }
            for r in results
        ]

    # =========================================================================
    # IMAS DD utilities
    # =========================================================================

    def search_dd_paths(
        query_text: str,
        ids_filter: str | None = None,
        max_results: int = 10,
        dd_version: int | None = None,
        node_category: str | None = None,
        cocos_transformation_type: str | None = None,
    ) -> str:
        """Search IMAS Data Dictionary using semantic search.

        Excludes error fields and metadata subtrees from results.
        Use fetch_dd_paths to access error fields via HAS_ERROR relationships.

        Args:
            query_text: Natural language query
            ids_filter: Optional IDS name filter (space-delimited)
            max_results: Maximum results
            dd_version: Filter by DD major version (e.g., 3 or 4)
            node_category: Filter by node category (e.g., "quantity", "geometry", "coordinate"). Default: no filter.
            cocos_transformation_type: Filter by COCOS transformation type (e.g., "psi_like", "ip_like", "b0_like"). Default: no filter.

        Returns:
            Formatted string with matching paths and documentation
        """
        try:
            tools = _get_imas_tools(semantic_search=True)
            result = _run_async(
                tools.search_tool.search_dd_paths(
                    query=query_text,
                    ids_filter=ids_filter,
                    max_results=max_results,
                    dd_version=dd_version,
                    node_category=node_category,
                    cocos_transformation_type=cocos_transformation_type,
                )
            )
            if not result.hits:
                return f"No IMAS paths found for '{query_text}'"
            output = []
            for hit in result.hits:
                line = f"{hit.path} (score: {hit.score:.2f})"
                output.append(line)
                if hit.documentation:
                    output.append(f"  {hit.documentation[:150]}...")
                if hit.units:
                    output.append(f"  Units: {hit.units}")
            return "\n".join(output)
        except Exception as e:
            return f"IMAS search error: {e}"

    def fetch_dd_paths(paths: str, dd_version: int | None = None) -> str:
        """Get full documentation for IMAS paths.

        Args:
            paths: Space-delimited IMAS paths
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Detailed path documentation
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.path_tool.fetch_dd_paths(paths=paths, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Fetch error: {e}"

    def list_dd_paths(
        paths: str,
        leaf_only: bool = True,
        max_paths: int = 100,
        dd_version: int | None = None,
        node_category: str | None = None,
        cocos_transformation_type: str | None = None,
    ) -> str:
        """List data paths in IDS.

        Args:
            paths: Space-separated IDS names or path prefixes
            leaf_only: Only return data fields
            max_paths: Limit output size
            dd_version: Filter by DD major version (e.g., 3 or 4)
            node_category: Filter by node category (e.g., "quantity", "geometry", "coordinate"). Default: no filter.
            cocos_transformation_type: Filter by COCOS transformation type (e.g., "psi_like", "ip_like"). Default: no filter.

        Returns:
            Tree structure in YAML format
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.list_tool.list_dd_paths(
                    paths=paths,
                    leaf_only=leaf_only,
                    max_paths=max_paths,
                    dd_version=dd_version,
                    node_category=node_category,
                    cocos_transformation_type=cocos_transformation_type,
                )
            )
            return str(result)
        except Exception as e:
            return f"List error: {e}"

    def check_dd_paths(paths: str, dd_version: int | None = None) -> str:
        """Validate IMAS paths for existence.

        Args:
            paths: Space-delimited IMAS paths
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Validation results with existence status
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.path_tool.check_dd_paths(paths=paths, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Check error: {e}"

    def get_dd_catalog(
        dd_version: int | None = None,
    ) -> str:
        """Get full catalog of all IMAS IDSs.

        Args:
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Catalog with all IDS names, descriptions, path counts, physics domains
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.overview_tool.get_dd_catalog(
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"Overview error: {e}"

    def find_related_dd_paths(
        path: str,
        relationship_types: str = "all",
        dd_version: int | None = None,
    ) -> str:
        """Get cross-IDS structural context for an IMAS path.

        Args:
            path: Exact IMAS path
            relationship_types: 'cluster', 'coordinate', 'unit', 'identifier', or 'all'
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Cross-IDS connections grouped by type
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.path_context_tool.find_related_dd_paths(
                    path=path,
                    relationship_types=relationship_types,
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"Path context error: {e}"

    def export_dd_ids(
        ids_name: str,
        leaf_only: bool = False,
        dd_version: int | None = None,
    ) -> str:
        """Export full IDS structure with documentation.

        Args:
            ids_name: IDS name (e.g. 'equilibrium')
            leaf_only: If true, return only leaf nodes
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Full IDS path listing
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.structure_tool.export_dd_ids(
                    ids_name=ids_name,
                    leaf_only=leaf_only,
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"Export error: {e}"

    def export_dd_domain(
        domain: str,
        ids_filter: str | None = None,
        dd_version: int | None = None,
    ) -> str:
        """Export all IMAS paths in a physics domain.

        Args:
            domain: Physics domain name (e.g. 'magnetics')
            ids_filter: Optional IDS name filter
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Domain export grouped by IDS
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.structure_tool.export_dd_domain(
                    domain=domain,
                    ids_filter=ids_filter,
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"Domain export error: {e}"

    # =========================================================================
    # COCOS utilities
    # =========================================================================

    def validate_cocos(
        declared_cocos: int,
        psi_axis: float,
        psi_edge: float,
        ip: float,
        b0: float,
        q: float | None = None,
        dp_dpsi: float | None = None,
    ) -> dict[str, Any]:
        """Validate declared COCOS against physics data.

        Uses Eq. 23 from Sauter & Medvedev paper to check consistency
        between declared COCOS and equilibrium physics quantities.

        Args:
            declared_cocos: COCOS value to validate (1-8 or 11-18)
            psi_axis: Poloidal flux at magnetic axis [Wb]
            psi_edge: Poloidal flux at plasma edge [Wb]
            ip: Plasma current [A] (sign matters)
            b0: Toroidal field at axis [T] (sign matters)
            q: Safety factor at mid-radius (optional)
            dp_dpsi: Pressure gradient dp/dψ (optional)

        Returns:
            Dict with is_consistent, calculated_cocos, confidence, inconsistencies

        Example:
            validate_cocos(17, psi_axis=0.5, psi_edge=-0.2, ip=-1e6, b0=-5.0, q=3.0)
        """
        from imas_codex.cocos import validate_cocos_from_data

        result = validate_cocos_from_data(
            declared_cocos=declared_cocos,
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
        )
        return {
            "is_consistent": result.is_consistent,
            "declared_cocos": result.declared_cocos,
            "calculated_cocos": result.calculated_cocos,
            "confidence": round(result.confidence, 2),
            "inconsistencies": result.inconsistencies,
        }

    def determine_cocos(
        psi_axis: float,
        psi_edge: float,
        ip: float,
        b0: float,
        q: float | None = None,
        dp_dpsi: float | None = None,
    ) -> dict[str, Any]:
        """Determine COCOS from equilibrium physics quantities.

        Uses Eq. 23 from Sauter & Medvedev paper to infer COCOS.

        Args:
            psi_axis: Poloidal flux at magnetic axis [Wb]
            psi_edge: Poloidal flux at plasma edge [Wb]
            ip: Plasma current [A] (sign matters)
            b0: Toroidal field at axis [T] (sign matters)
            q: Safety factor at mid-radius (optional)
            dp_dpsi: Pressure gradient dp/dψ (optional)

        Returns:
            Dict with cocos value and confidence

        Example:
            determine_cocos(psi_axis=0.5, psi_edge=-0.2, ip=-1e6, b0=-5.0)
        """
        from imas_codex.cocos import determine_cocos as _determine_cocos

        cocos, confidence = _determine_cocos(
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
        )
        return {"cocos": cocos, "confidence": round(confidence, 2)}

    def cocos_sign_flip_paths(ids_name: str | None = None) -> dict[str, Any]:
        """Get paths requiring COCOS sign flip between DD3/DD4.

        Args:
            ids_name: IDS name (e.g., 'equilibrium'). If None, lists all IDS.

        Returns:
            Dict with IDS name(s) and their sign-flip paths

        Example:
            cocos_sign_flip_paths('equilibrium')
            cocos_sign_flip_paths()  # List all IDS with sign flips
        """
        from imas_codex.cocos import get_sign_flip_paths, list_ids_with_sign_flips

        if ids_name:
            paths = get_sign_flip_paths(ids_name)
            return {"ids": ids_name, "paths": paths, "count": len(paths)}
        else:
            ids_list = list_ids_with_sign_flips()
            return {
                "ids_with_sign_flips": [
                    {"ids": ids, "count": len(get_sign_flip_paths(ids))}
                    for ids in ids_list
                ],
                "total_ids": len(ids_list),
            }

    def cocos_info(cocos_value: int) -> dict[str, Any]:
        """Get COCOS parameters for a given value.

        Args:
            cocos_value: COCOS index (1-8 or 11-18)

        Returns:
            Dict with the four COCOS parameters from Sauter Table I

        Example:
            cocos_info(17)  # IMAS DD4 / TCV convention
        """
        from imas_codex.cocos import KNOWN_CODE_COCOS, VALID_COCOS, cocos_to_parameters

        if cocos_value not in VALID_COCOS:
            return {"error": f"Invalid COCOS {cocos_value}. Valid: 1-8, 11-18"}

        params = cocos_to_parameters(cocos_value)
        codes = [code for code, val in KNOWN_CODE_COCOS.items() if val == cocos_value]
        return {
            "cocos": cocos_value,
            "sigma_bp": params.sigma_bp,
            "e_bp": params.e_bp,
            "sigma_r_phi_z": params.sigma_r_phi_z,
            "sigma_rho_theta_phi": params.sigma_rho_theta_phi,
            "used_by": codes,
        }

    # =========================================================================
    # Tool management utilities (from remote.tools)
    # =========================================================================

    def run(cmd: str, facility: str | None = None, timeout: int = 60) -> str:
        """Execute command locally or via SSH depending on facility.

        This is the unified execution interface. If facility is None or
        the facility has local=True (e.g., 'iter'), runs locally.
        Otherwise uses SSH.

        Args:
            cmd: Shell command to execute
            facility: Facility ID (None = local, 'iter' = local, 'tcv' = SSH)
            timeout: Command timeout in seconds

        Returns:
            Command output (stdout + stderr)

        Examples:
            run('rg pattern', facility='iter')  # Local (ITER is local)
            run('rg pattern', facility='tcv')  # SSH to EPFL
            run('rg pattern')                   # Local (no facility)
        """
        return _run(cmd, facility=facility, timeout=timeout)

    def check_tools(facility: str | None = None) -> dict[str, Any]:
        """Check availability of all fast CLI tools.

        Args:
            facility: Facility ID (None = local)

        Returns:
            Dict with tool statuses and summary

        Example:
            check_tools('tcv')
            check_tools('iter')  # Local check
            check_tools()        # Local check
        """
        return _check_all_tools(facility=facility)

    def install_tools(
        facility: str | None = None,
        required_only: bool = False,
    ) -> dict[str, Any]:
        """Install all fast CLI tools on target system.

        Args:
            facility: Facility ID (None = local)
            required_only: Only install required tools (rg, fd)

        Returns:
            Dict with installation results

        Example:
            install_tools('tcv')           # Install all on EPFL
            install_tools('iter')           # Install all locally
            install_tools(required_only=True)  # Just rg and fd
        """
        return _install_all_tools(facility=facility, required_only=required_only)

    # =========================================================================
    # Domain query functions (bound to this REPL's gc/embed)
    # =========================================================================

    import functools as _ft

    def _bind_dq(fn):
        """Bind gc and embed_fn into a domain query function."""

        @_ft.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("gc", gc)
            kwargs.setdefault("embed_fn", embed)
            return fn(*args, **kwargs)

        return wrapper

    find_signals = _bind_dq(_dq.find_signals)
    find_wiki = _bind_dq(_dq.find_wiki)
    wiki_page_chunks = _bind_dq(_dq.wiki_page_chunks)
    find_dd_paths = _bind_dq(_dq.find_dd_paths)
    find_code = _bind_dq(_dq.find_code)
    find_data_nodes = _bind_dq(_dq.find_data_nodes)
    map_signals_to_imas = _bind_dq(_dq.map_signals_to_imas)
    facility_overview = _bind_dq(_dq.facility_overview)
    graph_search = _bind_dq(_graph_search)

    # =========================================================================
    # REPL Registry — single source of truth for exposed functions
    # =========================================================================
    # To add a function: define it above, then add one entry here.
    # This registry drives: _repl_globals, repl_help(), and the python()
    # tool docstring. No other place needs updating.
    #
    # Format: list of (category, [(name, function), ...])
    # The name is what agents type in the REPL.

    _REPL_REGISTRY: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "DOMAIN QUERIES (prefer over raw Cypher)",
            [
                ("find_signals", find_signals),
                ("find_wiki", find_wiki),
                ("wiki_page_chunks", wiki_page_chunks),
                ("find_code", find_code),
                ("find_dd_paths", find_dd_paths),
                ("find_data_nodes", find_data_nodes),
                ("map_signals_to_imas", map_signals_to_imas),
                ("facility_overview", facility_overview),
            ],
        ),
        (
            "QUERY BUILDER",
            [("graph_search", graph_search)],
        ),
        (
            "FORMATTERS",
            [
                ("as_table", as_table),
                ("as_summary", as_summary),
                ("pick", pick),
            ],
        ),
        (
            "SCHEMA (call before writing Cypher)",
            [
                ("schema_for", _schema_for),
                ("get_schema", get_schema),
            ],
        ),
        (
            "GRAPH",
            [
                ("query", query),
                ("semantic_search", semantic_search),
                ("embed", embed),
            ],
        ),
        (
            "FACILITY",
            [
                ("get_facility", get_facility),
                ("get_facility_infrastructure", get_facility_infrastructure),
                ("update_infrastructure", update_infrastructure),
                ("get_exploration_targets", get_exploration_targets),
                ("get_tree_structure", get_tree_structure),
            ],
        ),
        (
            "REMOTE",
            [
                ("run", run),
                ("check_tools", check_tools),
            ],
        ),
        (
            "IMAS DD",
            [
                ("search_dd_paths", search_dd_paths),
                ("fetch_dd_paths", fetch_dd_paths),
                ("list_dd_paths", list_dd_paths),
                ("check_dd_paths", check_dd_paths),
                ("get_dd_catalog", get_dd_catalog),
                ("find_related_dd_paths", find_related_dd_paths),
                ("export_dd_ids", export_dd_ids),
                ("export_dd_domain", export_dd_domain),
            ],
        ),
        (
            "COCOS",
            [
                ("validate_cocos", validate_cocos),
                ("determine_cocos", determine_cocos),
                ("cocos_info", cocos_info),
            ],
        ),
    ]

    # Internal params injected by REPL binding — hidden from API reference
    _INTERNAL_PARAMS = {"gc", "embed_fn"}

    def _generate_repl_help() -> str:
        """Generate compact API reference from the registry."""
        import inspect

        lines = ["=== CODEX REPL API ===", ""]

        for cat, funcs in _REPL_REGISTRY:
            lines.append(f"  {cat}:")
            for name, fn in funcs:
                try:
                    sig = inspect.signature(fn)
                    params = {
                        k: v
                        for k, v in sig.parameters.items()
                        if k not in _INTERNAL_PARAMS
                    }
                    clean_sig = sig.replace(parameters=list(params.values()))
                    lines.append(f"    {name}{clean_sig}")
                except (ValueError, TypeError):
                    lines.append(f"    {name}(...)")
            lines.append("")

        lines.extend(
            [
                "  TIPS:",
                "  - Chain queries in a single python() call",
                "  - Call schema_for(task='wiki') before raw Cypher",
                "  - Use as_table(pick(results, 'col1', 'col2')) for output",
                "  - Call help(fn) for full docstring",
                "",
            ]
        )

        return "\n".join(lines)

    def repl_help() -> str:
        """Print auto-generated API reference for all REPL functions."""
        ref = _generate_repl_help()
        print(ref)
        return ref

    # =========================================================================
    # Build REPL globals from registry
    # =========================================================================

    _repl_globals = {name: fn for _, funcs in _REPL_REGISTRY for name, fn in funcs}
    _repl_globals.update(
        {
            # Core objects
            "gc": gc,
            "EmbeddingBackendError": EmbeddingBackendError,
            # Additional utilities not in the API reference
            "update_metadata": update_metadata,
            "install_tools": install_tools,
            "search_code": search_code,
            "get_dd_catalog": get_dd_catalog,
            "cocos_sign_flip_paths": cocos_sign_flip_paths,
            # REPL management
            "reload": _reload_repl,
            "repl_help": repl_help,
            # Standard library
            "subprocess": subprocess,
            # Result storage
            "_": None,
        }
    )

    logger.info(
        "Python REPL initialized with graph, IMAS, COCOS, and facility utilities"
    )
    return _repl_globals


def _get_repl() -> dict[str, Any]:
    """Get the persistent REPL environment, initializing lazily on first call.

    Thread-safe: uses a lock to prevent concurrent initialization.
    All imports are at module level, so no import deadlock risk.
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals
    with _repl_lock:
        # Double-check after acquiring lock
        if _repl_globals is None:
            _init_repl()
    return _repl_globals


def _reload_repl() -> str:
    """Reload the REPL environment after code changes.

    Clears cached modules and reinitializes all utilities.
    Use after editing imas_codex source files.

    Returns:
        Status message
    """
    global _repl_globals, _imas_tools_instance, _graph_client

    # Clear REPL state
    _repl_globals = None
    _imas_tools_instance = None
    _graph_client = None

    # Invalidate imas_codex module cache
    modules_to_reload = [name for name in sys.modules if name.startswith("imas_codex")]
    for name in modules_to_reload:
        try:
            del sys.modules[name]
        except KeyError:
            pass

    logger.info(f"Cleared {len(modules_to_reload)} cached imas_codex modules")

    # Reinitialize
    _init_repl()

    return f"REPL reloaded. Cleared {len(modules_to_reload)} modules and reinitialized utilities."


# =============================================================================
# MCP Server with 9 Core Tools
# =============================================================================


@dataclass
class AgentsServer:
    """
    MCP server with 7 core tools for facility exploration.

    Uses lazy initialization for the REPL — the first python() call
    triggers GraphClient connection and encoder setup. This avoids
    import deadlocks that occur when background threads perform imports.

    Tools:
    - search_signals: Signal search with data access and IMAS enrichment
    - search_docs: Wiki/document/image search with cross-links
    - search_code: Code search with data reference enrichment
    - search_dd_paths: IMAS DD search with cluster and facility cross-refs
    - python: Persistent REPL for custom queries not covered above
    - get_graph_schema: Schema introspection for query generation
    - add_to_graph: Schema-validated node creation with privacy filtering
    - update_facility_config: Read/update facility config (public or private)
    - update_facility_infrastructure: Deep-merge update to private YAML
    - get_facility_infrastructure: Read private infrastructure data
    - add_exploration_note: Append timestamped exploration note

    The python() REPL provides access to:
    - Graph: query(), semantic_search(), embed(), graph_search()
    - Domain: find_signals(), find_wiki(), wiki_page_chunks(), find_dd_paths(), find_code()
    - Formatters: as_table(), as_summary(), pick()
    - Remote: run(), check_tools() (auto-detects local vs SSH)
    - Facility: get_facility(), get_exploration_targets(), get_tree_structure()
    - IMAS DD: search_dd_paths(), fetch_dd_paths(), list_dd_paths(), check_dd_paths()
    - COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
    - Code: search_code()
    """

    read_only: bool = False
    dd_only: bool | None = None
    no_embed: bool = False
    mcp: FastMCP = field(init=False, repr=False)
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)
    _started_at: float = field(init=False, repr=False)

    # Tools that require facility data — not registered in DD-only mode
    FACILITY_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {
            "search_signals",
            "signal_analytics",
            "search_docs",
            "search_code",
            "fetch_content",
            "get_facility_coverage",
        }
    )

    def __post_init__(self):
        """Initialize the MCP server with lazy REPL loading.

        REPL initialization is deferred to the first python() tool call.
        All imports are at module level to avoid deadlocks. Only I/O-bound
        work (Neo4j connection) happens lazily.
        """
        import time

        global _no_embed
        _no_embed = self.no_embed

        self._started_at = time.monotonic()

        # Auto-detect DD-only mode from graph content.
        # Done in a background thread to avoid blocking the MCP handshake.
        # Default to False (full mode) so tools are registered immediately;
        # log if detection finds a mismatch (restart with --dd-only to fix).
        if self.dd_only is None:
            self.dd_only = False
            threading.Thread(
                target=self._background_detect_dd_only,
                daemon=True,
                name="dd-only-detect",
            ).start()

        # DD-only implies read-only: no write tools needed for a DD-only deployment
        if self.dd_only:
            self.read_only = True

        name = "imas-codex-readonly" if self.read_only else "imas-codex"
        self.mcp = FastMCP(name=name)
        self._prompts = load_prompts()

        self._register_tools()
        self._register_prompts()
        self._register_health_check()

        # Strip output_schema from all tools for MCP protocol backward
        # compatibility.  FastMCP 3.0 auto-generates outputSchema from
        # return type annotations, but the field was introduced in MCP
        # spec 2025-03-26.  Clients negotiating 2024-11-05 (e.g. GitHub
        # Copilot) reject the extra field with 400 Bad Request.
        for key, component in self.mcp._local_provider._components.items():
            if key.startswith("tool:") and hasattr(component, "output_schema"):
                component.output_schema = None

        # In DD-only mode, facility tools are never registered (see guards
        # in _register_tools). Log the active tool count for diagnostics.
        tool_count = sum(
            1 for k in self.mcp._local_provider._components if k.startswith("tool:")
        )
        mode_parts = []
        if self.read_only:
            mode_parts.append("read-only")
        if self.dd_only:
            mode_parts.append("dd-only")
        mode = ", ".join(mode_parts) if mode_parts else "read-write"
        logger.info(
            f"MCP server ready ({mode}) with {tool_count} tools and {len(self._prompts)} prompts"
        )

        # Pre-warm heavy imports in a background thread so the first
        # search_dd_paths call doesn't pay the full cold-start penalty.
        # Only imports modules — does NOT load the model, to avoid OOM
        # from two concurrent model instances on memory-constrained containers.
        def _warmup_encoder():
            try:
                import torch  # noqa: F401
                from sentence_transformers import SentenceTransformer  # noqa: F401

                from imas_codex.embeddings.encoder import Encoder  # noqa: F401

                logger.info("Encoder warmup complete (imports only)")
            except Exception as e:
                logger.warning(
                    f"Encoder warmup failed (will retry on first query): {e}"
                )

        threading.Thread(
            target=_warmup_encoder, daemon=True, name="encoder-warmup"
        ).start()

    @staticmethod
    def _detect_dd_only() -> bool:
        """Detect DD-only mode by checking for Facility nodes in the graph."""
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import get_graph_meta

            gc = GraphClient.from_profile()
            try:
                meta = get_graph_meta(gc)
                if meta:
                    facilities = meta.get("facilities") or []
                    dd_only = len(facilities) == 0
                    if dd_only:
                        logger.info("Auto-detected DD-only graph (no facilities)")
                    else:
                        logger.info(
                            f"Auto-detected full graph with facilities: "
                            f"{', '.join(facilities)}"
                        )
                    return dd_only
            finally:
                gc.close()
        except Exception:
            logger.warning("Could not detect graph mode, defaulting to full mode")
        return False

    def _background_detect_dd_only(self) -> None:
        """Run DD-only detection in background; log if result differs from startup default."""
        try:
            result = self._detect_dd_only()
            if result != self.dd_only:
                logger.info(
                    "Auto-detected DD-only=%s (server started in full mode). "
                    "Restart with explicit --dd-only flag to change tool set.",
                    result,
                )
            else:
                logger.debug("DD-only auto-detection confirmed: %s", result)
            self.dd_only = result
        except Exception as e:
            logger.warning("DD-only background detection failed: %s", e)

    def _register_tools(self):
        """Register all MCP tools."""

        # Generate API reference at registration time from the source
        # functions (no REPL init needed — just inspect.signature on
        # the unbound originals).
        api_reference = _generate_api_reference()

        if not self.read_only:
            # =====================================================================
            # Tool 1: python - Persistent REPL (primary interface)
            # =====================================================================

            # Build description before decoration — FastMCP 3.0 captures
            # the description eagerly at decoration time, so setting
            # __doc__ after @mcp.tool() has no effect on the MCP schema.
            repl_description = (
                "Execute Python in a persistent REPL with pre-imported graph "
                "query helpers, embedding functions, and facility config "
                "accessors. State persists across calls. Use for custom Cypher "
                "queries, signal-to-IMAS mapping, batch graph writes, and "
                "chained operations not covered by the dedicated search tools.\n\n"
                "Prefer search_signals/search_docs/search_code/search_dd_paths "
                "for standard lookups — they handle embeddings, enrichment, "
                "and formatting automatically.\n\n"
                f"{api_reference}\n"
                "Args:\n"
                "    code (str, required): Python code to execute. Multi-line "
                "supported. Use print() for output; bare expressions return "
                "their repr.\n\n"
                "Returns:\n"
                "    str: Captured stdout, or repr of the last expression if "
                "nothing was printed. Returns traceback on error."
            )

            @self.mcp.tool(description=repl_description)
            def repl(code: str) -> str:
                repl = _get_repl()

                stdout_capture = io.StringIO()

                try:
                    with redirect_stdout(stdout_capture):
                        try:
                            result = eval(code, repl)
                            if result is not None:
                                repl["_"] = result
                                print(repr(result))
                        except SyntaxError:
                            exec(code, repl)

                    output = stdout_capture.getvalue()
                    if not output:
                        output = "(no output)"
                    return output

                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    return f"Error: {e}\n\n{tb}"

        # =====================================================================
        # Tool 2: get_graph_schema - Schema introspection (REPL companion)
        # =====================================================================
        # Only available in full mode — provides schema context for Cypher
        # queries via the REPL, which dd-only mode does not expose.

        if not self.dd_only:

            @self.mcp.tool()
            def get_graph_schema(
                scope: str = "overview",
            ) -> str:
                """Get graph schema context for Cypher query generation.

                Returns compact, task-relevant schema in text format. Use scope to
                get only the schema slice you need, reducing token usage. Call this
                before writing any raw Cypher to verify node labels, property names,
                relationship types, and enum values.

                Args:
                    scope: Schema slice to return. One of:
                        - "overview": compact summary of all node labels, relationship
                          types, vector indexes, and task groupings (default).
                        - "signals": FacilitySignal, DataAccess, Diagnostic, AccessCheck.
                        - "wiki": WikiPage, WikiChunk, Document, Image.
                        - "imas": IMASNode, IDS, IMASSemanticCluster, DDVersion, Unit,
                          IMASNodeChange, IMASCoordinateSpec.
                        - "code": CodeFile, CodeChunk, CodeExample.
                        - "facility": Facility, FacilityPath, FacilitySignal, SignalNode,
                          Diagnostic.
                        - "data_sources": data source nodes and tree-related relationships.

                Returns:
                    Formatted text containing property tables (name, type, description),
                    relationship definitions as (From)-[:REL]->(To), available vector
                    indexes, and enum values for the requested scope.
                """
                from imas_codex.graph.schema_context import schema_for

                return schema_for(task=scope)

        if not self.read_only:
            # =====================================================================
            # Tool 3: add_to_graph - Schema-validated writes
            # =====================================================================

            @self.mcp.tool()
            def add_to_graph(
                node_type: str,
                data: dict[str, Any] | list[dict[str, Any]],
                create_facility_relationship: bool = True,
                batch_size: int = 50,
            ) -> dict[str, Any]:
                """Create or update nodes in the Neo4j knowledge graph with schema validation.

                Validates each item against the Pydantic model for node_type, strips
                private fields, then MERGEs nodes by id. Use for semantic data (files,
                signals, paths). For infrastructure metadata, use update_facility_infrastructure.

                CodeFile nodes are auto-deduplicated against existing discovered/ingested files.

                Args:
                    node_type (str, required): Graph node label. Must match a schema-defined
                        type — call get_graph_schema() to list valid labels.
                    data (dict | list[dict], required): One dict or a list of dicts. Each
                        dict must include at least "id" and "facility_id". Properties are
                        validated against the node_type's Pydantic model.
                    create_facility_relationship (bool): When true (default), auto-creates
                        an AT_FACILITY relationship to the Facility node.
                    batch_size (int): Nodes per UNWIND batch. Default 50.

                Returns:
                    dict with keys: "processed" (int), "skipped" (int),
                    "relationships" (dict of rel_type→count), "errors" (list[str]).
                """
                _require_graph_only()
                schema = get_schema()

                if node_type not in schema.node_labels:
                    msg = f"Unknown node type: {node_type}. Valid: {schema.node_labels}"
                    raise ValueError(msg)

                items = [data] if isinstance(data, dict) else data
                if not items:
                    return {"processed": 0, "skipped": 0, "errors": []}

                private_slots = set(schema.get_private_slots(node_type))
                model_class = schema.get_model(node_type)

                valid_items: list[dict[str, Any]] = []
                errors: list[str] = []

                for i, item in enumerate(items):
                    try:
                        filtered = {
                            k: v for k, v in item.items() if k not in private_slots
                        }
                        validated = model_class.model_validate(filtered)
                        props = to_cypher_props(validated)
                        valid_items.append(props)
                    except Exception as e:
                        item_id = item.get("id", f"item[{i}]")
                        errors.append(f"{item_id}: {e}")
                        logger.warning(
                            f"Validation failed for {node_type} {item_id}: {e}"
                        )

                if not valid_items:
                    return {"processed": 0, "skipped": len(errors), "errors": errors}

                try:
                    with GraphClient() as client:
                        skipped_dedup = 0

                        # Ingestion gating: verify facility is allowed
                        facility_ids = {
                            item.get("facility_id")
                            for item in valid_items
                            if item.get("facility_id")
                        }
                        for fid in facility_ids:
                            try:
                                from imas_codex.graph.meta import gate_ingestion

                                gate_ingestion(client, fid)
                            except ValueError as gate_err:
                                return {
                                    "processed": 0,
                                    "skipped": len(valid_items),
                                    "errors": [str(gate_err)],
                                }

                        # CodeFile deduplication
                        if node_type == "CodeFile":
                            existing = client.query(
                                """
                                UNWIND $items AS item
                                OPTIONAL MATCH (sf:CodeFile {id: item.id})
                                OPTIONAL MATCH (ce:CodeExample {source_file: item.path, facility_id: item.facility_id})
                                RETURN item.id AS id, sf.status AS sf_status, ce.id AS ce_id
                                """,
                                items=valid_items,
                            )
                            skip_ids = {
                                row["id"]
                                for row in existing
                                if row["sf_status"] in ("discovered", "ingested")
                                or row["ce_id"]
                            }
                            if skip_ids:
                                valid_items = [
                                    i for i in valid_items if i["id"] not in skip_ids
                                ]
                                skipped_dedup = len(skip_ids)
                            if not valid_items:
                                return {
                                    "processed": 0,
                                    "skipped": len(errors) + skipped_dedup,
                                    "errors": errors,
                                }

                        result = client.create_nodes(
                            label=node_type,
                            items=valid_items,
                            batch_size=batch_size,
                            create_relationships=create_facility_relationship,
                        )

                        return {
                            "processed": result["processed"],
                            "relationships": result.get("relationships", {}),
                            "skipped": len(errors) + skipped_dedup,
                            "errors": errors,
                        }

                except Exception as e:
                    logger.exception("Failed to ingest nodes")
                    raise RuntimeError(
                        f"Failed to ingest nodes: {_neo4j_error_message(e)}"
                    ) from e

        if not self.read_only:
            # =====================================================================
            # Tool 4: update_facility_config - Facility configuration management
            # =====================================================================

            @self.mcp.tool()
            def update_facility_config(
                facility: str,
                data: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                """Read or update a facility's public YAML configuration.

                Reads or deep-merges data into the facility's public metadata
                (git-tracked, deployed in CLI tools). Use for facility description,
                data_systems config, wiki sites, and discovery roots. For
                graph-stored data, use add_to_graph.

                Args:
                    facility (str, required): Facility identifier (e.g. "tcv",
                        "iter", "jet", "jt-60sa", "west", "mast-u").
                    data (dict | None): Dict to deep-merge into the public config.
                        If None (default), returns current config without modification.

                Returns:
                    dict: Current public metadata after any update.
                """
                try:
                    if data is not None:
                        update_metadata(facility, data)

                    from imas_codex.discovery import get_facility_metadata

                    return get_facility_metadata(facility) or {}
                except Exception as e:
                    logger.exception(f"Failed to access config for {facility}")
                    raise RuntimeError(f"Failed to access config: {e}") from e

        # NOTE: update_facility_infrastructure, get_facility_infrastructure, and
        # add_exploration_note were removed (private-infra deprecation).
        # Use the repl tool with update_infrastructure() / get_facility_infrastructure()
        # from imas_codex.discovery for those operations if still needed.

        # =====================================================================
        # Tool 7b: get_facility_coverage - Scored path coverage and gaps
        # =====================================================================

        if not self.dd_only:

            @self.mcp.tool()
            def get_facility_coverage(facility: str) -> dict[str, Any]:
                """Get scored path coverage and discovery gaps for a facility.

                Queries the graph for scored FacilityPath nodes and compares
                coverage against expected path_purpose categories. Call before
                exploring to identify underrepresented areas and avoid duplication.

                Args:
                    facility (str, required): Facility identifier (e.g. "tcv",
                        "iter", "jet", "jt-60sa", "west", "mast-u").

                Returns:
                    dict with keys:
                    - "facility" (str): Echo of input.
                    - "discovery_roots" (list[dict]): Configured root paths.
                    - "coverage_by_category" (dict[str, int]): Scored path counts
                      per path_purpose value.
                    - "total_scored_paths" (int): Sum of all coverage counts.
                    - "high_value_paths" (list[dict]): Top 15 paths with
                      score > 0.7 (keys: path, purpose, score, description).
                    - "missing_categories" (list[str]): Expected categories
                      with zero coverage.
                    - "unexplored_containers" (list[dict]): Containers scoring
                      > 0.4 that have not been expanded yet.
                    - "schema" (dict): Valid path_purpose category values.
                """
                try:
                    from imas_codex.discovery import (
                        get_facility_infrastructure as _get_infra,
                    )
                    from imas_codex.graph import GraphClient
                    from imas_codex.llm.prompt_loader import get_schema_for_prompt

                    # Get configured roots from infrastructure
                    infra = _get_infra(facility) or {}
                    discovery_roots = infra.get("discovery_roots", [])

                    # Get schema context for valid category values
                    schema_ctx = get_schema_for_prompt(
                        "discovery/roots", ["discovery_categories"]
                    )

                    # Use GraphClient for graph queries
                    with GraphClient() as client:
                        # Query coverage by category
                        coverage_query = """
                            MATCH (p:FacilityPath {facility_id: $facility})
                            WHERE p.status = 'scored' AND p.path_purpose IS NOT NULL
                            RETURN p.path_purpose AS purpose, count(*) AS count
                            ORDER BY count DESC
                        """
                        coverage_results = client.query(
                            coverage_query, facility=facility
                        )
                        coverage_by_category = {
                            record["purpose"]: record["count"]
                            for record in coverage_results
                        }

                        # Query high-value paths
                        high_value_query = """
                            MATCH (p:FacilityPath {facility_id: $facility})
                            WHERE coalesce(p.score_composite, p.triage_composite) > 0.7
                            RETURN p.path AS path, p.path_purpose AS purpose,
                                   coalesce(p.score_composite, p.triage_composite) AS score, p.description AS description
                            ORDER BY score DESC LIMIT 15
                        """
                        high_value_paths = client.query(
                            high_value_query, facility=facility
                        )

                        # Determine missing categories (expected but not found)
                        expected_categories = [
                            c["value"] for c in schema_ctx["discovery_categories"]
                        ]
                        found_categories = set(coverage_by_category.keys())
                        missing_categories = [
                            c for c in expected_categories if c not in found_categories
                        ]

                        # Query containers not yet expanded (potential new roots)
                        unexplored_query = """
                            MATCH (p:FacilityPath {facility_id: $facility})
                            WHERE p.path_purpose = 'container'
                                  AND coalesce(p.score_composite, p.triage_composite) > 0.4
                                  AND p.should_expand = false
                                  AND p.terminal_reason IS NULL
                            RETURN p.path AS path, coalesce(p.score_composite, p.triage_composite) AS score, p.description AS description
                            ORDER BY score DESC LIMIT 10
                        """
                        unexplored_containers = client.query(
                            unexplored_query, facility=facility
                        )

                    return {
                        "facility": facility,
                        "discovery_roots": discovery_roots,
                        "coverage_by_category": coverage_by_category,
                        "total_scored_paths": sum(coverage_by_category.values()),
                        "high_value_paths": high_value_paths,
                        "missing_categories": missing_categories,
                        "unexplored_containers": unexplored_containers,
                        "schema": {
                            "valid_categories": schema_ctx["discovery_categories"],
                        },
                    }
                except Exception as e:
                    logger.exception(f"Failed to get facility coverage for {facility}")
                    raise RuntimeError(f"Failed to get facility coverage: {e}") from e

        # NOTE: update_facility_paths and update_facility_tools were removed as
        # MCP tools (Phase 5 consolidation). Use update_infrastructure() in the
        # REPL instead: update_infrastructure('facility', {'paths': {...}})

        # =====================================================================
        # Tier 1 — semantic search (embeddings + graph)
        # =====================================================================

        if not self.dd_only:

            @self.mcp.tool()
            def search_signals(
                query: str,
                facility: str,
                diagnostic: str | None = None,
                physics_domain: str | None = None,
                check_status: str | None = None,
                error_type: str | None = None,
                include_check_details: bool = False,
                k: int = 20,
            ) -> str:
                """Find facility signals by physics concept and return data-access details.

                Hybrid search (vector similarity + keyword) over FacilitySignal
                descriptions, enriched with DataAccess templates, IMAS mappings,
                diagnostic context, and related MDSplus/TDI tree nodes.

                Args:
                    query: Natural-language description of the quantity to find
                        (e.g. "plasma current", "electron density profile").
                    facility: Facility identifier (required). One of: tcv, jet,
                        iter, jt-60sa, west, mast-u, etc.
                    diagnostic: Keep only signals from this diagnostic system
                        (e.g. "magnetics", "thomson_scattering").
                    physics_domain: Keep only signals in this physics domain
                        (e.g. "magnetics", "kinetics", "equilibrium").
                    check_status: Filter by data-access check outcome.
                        Values: "passed", "failed", "unchecked".
                    error_type: Filter by check error classification
                        (e.g. "not_available_for_shot"). Implies check_status="failed".
                    include_check_details: If true, append per-signal check metadata
                        (shot, shape, error, timestamp) to each result.
                    k: Maximum results to return (default 20).

                Returns:
                    Formatted text report. Each signal entry includes: signal id,
                    description, diagnostic, data-access template, IMAS mapping
                    path (if mapped), and related tree node paths. A secondary
                    section lists raw tree-node matches from the SignalNode index.
                """
                from imas_codex.llm.search_tools import _search_signals as _ss

                return _ss(
                    query,
                    facility,
                    diagnostic=diagnostic,
                    physics_domain=physics_domain,
                    check_status=check_status,
                    error_type=error_type,
                    include_check_details=include_check_details,
                    k=k,
                )

            @self.mcp.tool()
            def signal_analytics(
                facility: str,
                group_by: list[str] | None = None,
                filters: dict[str, str] | None = None,
            ) -> str:
                """Aggregate signal counts grouped by one or more dimensions.

                Returns a markdown table of counts and percentages. Use this
                instead of raw Cypher for questions like "how many signals per
                physics domain?" or "pass/fail breakdown for magnetics?".

                Args:
                    facility: Facility identifier (required). One of: tcv, jet,
                        iter, jt-60sa, west, mast-u, etc.
                    group_by: Dimensions to group by (default: ["status"]).
                        Allowed values: status, physics_domain, data_source_name,
                        discovery_source, diagnostic, check_status, error_type.
                        Multiple values produce a cross-tabulation.
                    filters: Key-value pairs to restrict the counted population
                        before grouping. Keys must be from the allowed group_by
                        set (e.g. {"physics_domain": "magnetics",
                        "check_status": "failed"}).

                Returns:
                    Markdown table with one row per group combination, showing
                    count and percentage of total. Includes a total signal count.
                """
                from imas_codex.llm.search_tools import _signal_analytics as _sa

                return _sa(facility, group_by=group_by, filters=filters)

            @self.mcp.tool()
            def search_docs(
                query: str,
                facility: str,
                k: int = 15,
                site: str | None = None,
                physics_domain: str | None = None,
                min_score: float | None = None,
                score_dimension: str | None = None,
            ) -> str:
                """Search wiki pages, linked documents, and images for a topic.

                Hybrid search (vector similarity + keyword) across WikiChunk
                text, plus a secondary vector search on Document and Image
                nodes. Results are enriched with cross-references to signals,
                tree nodes, and IMAS paths found on the same wiki page.

                Args:
                    query: Natural-language description of the topic
                        (e.g. "fishbone instabilities", "COCOS conventions").
                    facility: Facility identifier (required). One of: tcv, jet,
                        iter, jt-60sa, west, mast-u, etc.
                    k: Maximum results per search index (default 15).
                    site: Substring filter on the wiki site URL
                        (e.g. "crpp" to restrict to CRPP wiki pages).
                    physics_domain: Keep only pages classified under this domain
                        (e.g. "mhd", "heating", "diagnostics").
                    min_score: Minimum page-level score (0.0–1.0) on
                        score_dimension. Pages below this threshold are excluded.
                    score_dimension: Which page score to apply min_score to.
                        Values: score_data_documentation, score_physics_content,
                        score_code_documentation, score_data_access,
                        score_calibration, score_imas_relevance, score_composite
                        (default: score_composite).

                Returns:
                    Formatted text report. Wiki results are grouped by page with
                    chunk excerpts, cross-linked signal/IMAS refs, and relevance
                    scores. A separate section lists matching documents/images.
                """
                from imas_codex.llm.search_tools import _search_docs as _sd

                return _sd(
                    query,
                    facility,
                    k=k,
                    site=site,
                    physics_domain=physics_domain,
                    min_score=min_score,
                    score_dimension=score_dimension,
                )

            @self.mcp.tool()
            def search_code(
                query: str,
                facility: str | None = None,
                k: int = 10,
                physics_domain: str | None = None,
                min_score: float | None = None,
                score_dimension: str | None = None,
            ) -> str:
                """Search ingested source code for relevant functions and snippets.

                Hybrid search (vector similarity + keyword) over CodeChunk text
                and CodeExample descriptions, enriched with MDSplus node paths,
                TDI function calls, IMAS path references, and parent directory
                context from FacilityPath.

                Args:
                    query: Natural-language description of the code to find
                        (e.g. "equilibrium reconstruction", "read interferometer data").
                    facility: Facility identifier to scope results. Omit to
                        search across all facilities. One of: tcv, jet, iter,
                        jt-60sa, west, mast-u, etc.
                    k: Maximum results to return (default 10).
                    physics_domain: Keep only code under paths classified in this
                        domain (e.g. "equilibrium", "transport").
                    min_score: Minimum path-level score (0.0–1.0) on
                        score_dimension. Paths below this threshold are excluded.
                    score_dimension: Which path score to apply min_score to.
                        Values: score_modeling_code, score_analysis_code,
                        score_operations_code, score_data_access, score_workflow,
                        score_visualization, score_documentation, score_imas,
                        score_convention, score_composite (default: score_composite).

                Returns:
                    Formatted text report. Each result includes the code snippet,
                    source file path, data references (MDSplus nodes, TDI calls,
                    IMAS paths), and relevance score.
                """
                from imas_codex.llm.search_tools import _search_code as _sc

                return _sc(
                    query,
                    facility=facility,
                    k=k,
                    physics_domain=physics_domain,
                    min_score=min_score,
                    score_dimension=score_dimension,
                )

        @self.mcp.tool()
        def search_dd_paths(
            query: str,
            ids_filter: str | None = None,
            facility: str | None = None,
            include_version_context: bool = False,
            dd_version: int | None = None,
            k: int = 20,
            physics_domain: str | None = None,
            lifecycle_filter: str | None = None,
            node_category: str | None = None,
            cocos_transformation_type: str | None = None,
        ) -> str:
            """Find IMAS Data Dictionary paths matching a concept. Use when you need to discover which paths store a given physical quantity.

            Performs hybrid search (vector + keyword) across path and cluster embeddings. Results include data type, units, coordinates, cluster membership, and optional facility signal cross-references. Error fields and metadata subtrees (ids_properties/*, code/*) are excluded — use fetch_dd_error_fields for those.

            Args:
                query: Natural-language description of the quantity to find (e.g. "electron temperature", "plasma boundary shape").
                ids_filter: Restrict results to a single IDS name (e.g. "core_profiles"). Default: search all IDSs.
                facility: Include cross-references to facility signals (e.g. "tcv", "jet"). Default: no facility enrichment.
                include_version_context: If true, append DD version change history for each matched path. Default: false.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.
                k: Maximum number of results. Default: 20.
                physics_domain: Filter by physics domain (e.g., "magnetics", "equilibrium", "transport"). Default: no filter.
                lifecycle_filter: Filter by lifecycle status ('active', 'alpha', 'obsolescent'). Default: no filter.
                node_category: Filter by node category (e.g., "quantity", "geometry", "coordinate"). Default: no filter.
                cocos_transformation_type: Filter by COCOS transformation type (e.g., "psi_like", "ip_like", "b0_like"). Default: no filter.

            Returns:
                Formatted text report listing matched paths with types, units, cluster labels, and optional facility cross-references.
            """
            from concurrent.futures import ThreadPoolExecutor

            from imas_codex.llm.search_formatters import format_search_dd_report
            from imas_codex.models.error_models import ToolError

            tools = _get_imas_tools(semantic_search=True)

            # Run path search and cluster search in parallel — they are
            # independent operations sharing the same encoder singleton.
            def _path_search():
                return _run_async(
                    tools.search_tool.search_dd_paths(
                        query=query,
                        ids_filter=ids_filter,
                        max_results=k,
                        facility=facility,
                        include_version_context=include_version_context,
                        dd_version=dd_version,
                        physics_domain=physics_domain,
                        lifecycle_filter=lifecycle_filter,
                        node_category=node_category,
                        cocos_transformation_type=cocos_transformation_type,
                    )
                )

            def _cluster_search():
                try:
                    cr = _run_async(
                        tools.clusters_tool.search_dd_clusters(
                            query=query,
                            ids_filter=ids_filter,
                            dd_version=dd_version,
                        )
                    )
                    if isinstance(cr, ToolError):
                        logger.debug(
                            "Cluster search returned ToolError, "
                            "omitting optional cluster context"
                        )
                        return None
                    return cr
                except Exception:
                    logger.debug(
                        "Cluster search failed, continuing without",
                        exc_info=True,
                    )
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                path_future = executor.submit(_path_search)
                cluster_future = executor.submit(_cluster_search)
                result = path_future.result()
                cluster_result = cluster_future.result()

            if isinstance(result, ToolError):
                return format_search_dd_report(result)

            return format_search_dd_report(result, cluster_result)

        # =====================================================================
        # Tier 2 — graph-only (no embeddings)
        # =====================================================================

        @self.mcp.tool()
        def check_dd_paths(
            paths: str,
            ids: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Check whether specific IMAS paths exist in the Data Dictionary. Use to validate paths before accessing data or to diagnose typos.

            For each path, reports whether it exists, its data type and units, and suggests corrections for misspelled or renamed paths.

            Args:
                paths: Space- or comma-separated IMAS paths to validate (e.g. "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature").
                ids: Optional IDS name to prepend to all paths (e.g. "equilibrium" turns "time_slice/profiles_1d/psi" into "equilibrium/time_slice/profiles_1d/psi"). Default: none.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report with existence status, data type, and units per path. Invalid paths include suggested corrections.
            """
            from imas_codex.llm.search_formatters import format_check_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.path_tool.check_dd_paths(
                    paths=paths, ids=ids, dd_version=dd_version
                )
            )
            return format_check_report(result)

        @self.mcp.tool()
        def fetch_dd_paths(
            paths: str,
            ids: str | None = None,
            dd_version: int | None = None,
            include_version_history: bool = False,
            include_children: bool = False,
        ) -> str:
            """Get full documentation for known IMAS paths. Use after search_dd_paths or check_dd_paths to retrieve detailed metadata for specific paths.

            Returns per-path: documentation text, data type, units, coordinate specifications, semantic cluster labels, physics domain, identifier schemas, and optionally version change history.

            Args:
                paths: Space- or comma-separated IMAS paths (e.g. "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature").
                ids: Optional IDS name to prepend to all paths (e.g. "core_profiles"). Default: none.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.
                include_version_history: If true, include notable changes across DD versions for each path. Default: false.
                include_children: If true, include a preview of child paths for structure nodes. Default: false.

            Returns:
                Formatted text report with complete documentation per path.
            """
            from imas_codex.llm.search_formatters import format_fetch_paths_report

            if include_children:
                logger.debug(
                    "include_children not yet implemented in backend, ignoring"
                )
            tools = _get_imas_tools()
            result = _run_async(
                tools.path_tool.fetch_dd_paths(
                    paths=paths,
                    ids=ids,
                    dd_version=dd_version,
                    include_version_history=include_version_history,
                    # include_children not yet implemented in backend
                )
            )
            return format_fetch_paths_report(result)

        @self.mcp.tool()
        def fetch_dd_error_fields(
            path: str,
            dd_version: int | None = None,
        ) -> str:
            """Get the uncertainty/error fields associated with a data path. IMAS data paths can have companion error fields (_error_upper, _error_lower, _error_index) that quantify measurement uncertainty. These are excluded from search and list results — use this tool to discover them for a known path.

            Args:
                path: Exact IMAS data path (e.g. "equilibrium/time_slice/profiles_1d/psi").
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text listing error fields and their data types, or empty if the path has no error fields.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.path_tool.fetch_error_fields(path=path, dd_version=dd_version)
            )
            return _format_error_fields_report(result)

        @self.mcp.tool()
        def list_dd_paths(
            paths: str,
            leaf_only: bool = False,
            max_paths: int | None = None,
            dd_version: int | None = None,
            physics_domain: str | None = None,
            node_type: str | None = None,
            lifecycle_filter: str | None = None,
            node_category: str | None = None,
            cocos_transformation_type: str | None = None,
        ) -> str:
            """Enumerate all paths under an IDS or subtree. Use to browse the structure of an IDS or discover available fields under a path prefix.

            Lists child paths hierarchically. Error fields and metadata subtrees (ids_properties/*, code/*) are excluded — use fetch_dd_error_fields for uncertainty data.

            Args:
                paths: Space-separated IDS names or path prefixes (e.g. "equilibrium" or "equilibrium/time_slice"). Multiple values list paths from each.
                leaf_only: If true, return only leaf data fields (skip intermediate structures). Default: false.
                max_paths: Cap the number of paths returned. Default: no limit.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.
                physics_domain: Filter by physics domain (e.g., "magnetics", "equilibrium", "transport"). Default: no filter.
                node_type: Filter by node type ('dynamic', 'static', 'constant'). Default: no filter.
                lifecycle_filter: Filter by lifecycle status ('active', 'alpha', 'obsolescent'). Default: no filter.
                node_category: Filter by node category (e.g., "quantity", "geometry", "coordinate"). Default: no filter.
                cocos_transformation_type: Filter by COCOS transformation type (e.g., "psi_like", "ip_like"). Default: no filter.

            Returns:
                Formatted text listing of paths with their data types.
            """
            from imas_codex.llm.search_formatters import format_list_report

            if (
                physics_domain is not None
                or node_type is not None
                or lifecycle_filter is not None
                or node_category is not None
                or cocos_transformation_type is not None
            ):
                logger.debug(
                    "Applying filters: physics_domain=%s node_type=%s lifecycle=%s node_category=%s cocos=%s",
                    physics_domain,
                    node_type,
                    lifecycle_filter,
                    node_category,
                    cocos_transformation_type,
                )
            tools = _get_imas_tools()
            result = _run_async(
                tools.list_tool.list_dd_paths(
                    paths=paths,
                    leaf_only=leaf_only,
                    max_paths=max_paths,
                    dd_version=dd_version,
                    physics_domain=physics_domain,
                    node_type=node_type,
                    lifecycle_filter=lifecycle_filter,
                    node_category=node_category,
                    cocos_transformation_type=cocos_transformation_type,
                )
            )
            return format_list_report(result)

        @self.mcp.tool()
        def get_dd_catalog(
            dd_version: int | None = None,
        ) -> str:
            """List all available IDSs (Interface Data Structures) with descriptions and statistics. Use as a starting point to discover which IDS contains the data you need.

            Each IDS entry includes its description, total path count, and physics domain classification.

            Args:
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report listing each IDS with its description, path count, and physics domain.
            """
            from imas_codex.llm.search_formatters import format_overview_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.overview_tool.get_dd_catalog(
                    dd_version=dd_version,
                )
            )
            return format_overview_report(result)

        @self.mcp.tool()
        def get_dd_identifiers(
            query: str | None = None,
        ) -> str:
            """Browse IMAS enumeration/identifier schemas and their allowed values. Use to find valid options for typed fields like coordinate systems, grid types, or probe types.

            IMAS paths that reference an identifier schema accept only the enumerated values defined in that schema.

            Args:
                query: Optional keyword to filter schema names (e.g. "coordinate", "grid_type", "magnetics"). Default: list all schemas.

            Returns:
                Formatted text report listing each identifier schema with its allowed name/index/description options.
            """
            from imas_codex.llm.search_formatters import format_identifiers_report

            tools = _get_imas_tools()
            result = _run_async(tools.identifiers_tool.get_dd_identifiers(query=query))
            return format_identifiers_report(result)

        @self.mcp.tool()
        def search_dd_clusters(
            query: str | None = None,
            scope: str | None = None,
            ids_filter: str | None = None,
            section_only: bool = False,
            dd_version: int | None = None,
        ) -> str:
            """Find groups of semantically related IMAS paths (clusters). Clusters group paths that represent the same physical concept across different IDSs — e.g. all "electron temperature" paths regardless of which IDS they live in.

            Supports three modes: (1) natural-language search for clusters by topic, (2) find which clusters contain a specific path, (3) list all clusters for an IDS (pass ids_filter without query).

            Args:
                query: Natural-language topic (e.g. "boundary geometry") or exact IMAS path (e.g. "equilibrium/time_slice/boundary/outline/r"). Optional when ids_filter is provided.
                scope: Filter by cluster scope — "global" (cross-IDS), "domain" (within physics domain), or "ids" (within single IDS). Default: all scopes.
                ids_filter: Restrict to clusters containing paths from this IDS (e.g. "equilibrium"). Default: all IDSs.
                section_only: If true, return only clusters that contain structural section nodes. Default: false.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report listing matched clusters with their member paths and descriptions.
            """
            from imas_codex.llm.search_formatters import format_cluster_report
            from imas_codex.models.error_models import ToolError

            tools = _get_imas_tools(semantic_search=True)
            result = _run_async(
                tools.clusters_tool.search_dd_clusters(
                    query=query,
                    scope=scope,
                    ids_filter=ids_filter,
                    section_only=section_only,
                    dd_version=dd_version,
                )
            )
            if isinstance(result, ToolError):
                return format_cluster_report(result)
            return format_cluster_report(result)

        @self.mcp.tool()
        def find_related_dd_paths(
            path: str,
            relationship_types: str = "all",
            max_results: int = 20,
            dd_version: int | None = None,
        ) -> str:
            """Find paths in other IDSs that are related to a given path. Use to discover cross-IDS connections — e.g. where the same physical quantity appears in different data structures.

            Combines multiple relationship signals: vector similarity, shared cluster membership, common physics coordinates, matching units, shared identifier schemas, and shared COCOS transformations. Error and metadata fields are filtered out. Unit matches are no longer restricted to the same physics domain, surfacing cross-domain peers.

            Args:
                path: Exact IMAS path to find relatives for (e.g. "equilibrium/time_slice/profiles_1d/psi").
                relationship_types: Which relationship types to include — "semantic", "cluster", "coordinate", "unit", "identifier", "cocos", or "all". Default: "all".
                max_results: Maximum results per relationship type. Default: 20.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report with related paths grouped by relationship type, each showing the target path, IDS, and relevance score.
            """
            from imas_codex.core.paths import normalize_imas_path
            from imas_codex.llm.search_formatters import format_path_context_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.path_context_tool.find_related_dd_paths(
                    path=normalize_imas_path(path),
                    relationship_types=relationship_types,
                    max_results=max_results,
                    dd_version=dd_version,
                )
            )
            return format_path_context_report(result)

        @self.mcp.tool()
        def get_ids_summary(
            ids_name: str,
            dd_version: int | None = None,
        ) -> str:
            """Analyze the internal structure and organization of a specific IMAS IDS.

            Returns a compact overview including: metrics (path counts, depth), top-level sections,
            semantic clusters, identifier schemas, COCOS fields, coordinate arrays, and data type distribution.

            Args:
                ids_name: IDS name to analyze (e.g. "equilibrium", "core_profiles").
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report with structural overview of the IDS.
            """
            from imas_codex.llm.search_formatters import format_structure_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.structure_tool.get_ids_summary(
                    ids_name=ids_name,
                    dd_version=dd_version,
                )
            )
            return format_structure_report(result)

        @self.mcp.tool()
        def get_dd_cocos_fields(
            transformation_type: str | None = None,
            ids_filter: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Get all COCOS-dependent fields across the Data Dictionary, grouped by
            transformation type. Use when migrating code between COCOS conventions
            or verifying sign handling.

            Args:
                transformation_type: Filter to specific type (e.g., 'psi_like', 'ip_like', 'b0_like'). Default: all types.
                ids_filter: Limit to specific IDS (e.g., 'equilibrium'). Default: all IDSs.
                dd_version: Filter by DD major version (3 or 4). Default: latest version.

            Returns:
                Formatted text report listing COCOS-dependent fields grouped by transformation type.
            """
            from imas_codex.llm.search_formatters import format_cocos_fields_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.structure_tool.get_dd_cocos_fields(
                    transformation_type=transformation_type,
                    ids_filter=ids_filter,
                    dd_version=dd_version,
                )
            )
            return format_cocos_fields_report(result)

        @self.mcp.tool()
        def get_dd_version_context(
            paths: str | None = None,
            change_type_filter: str | None = None,
            ids_filter: str | None = None,
            from_version: str | None = None,
            to_version: str | None = None,
            follow_rename_chains: bool = False,
        ) -> str:
            """Get version change history for specific IMAS paths, or list all changes of a specific type across the Data Dictionary.

            Mode 1 (per-path): Provide paths to get version history for specific paths.
            Mode 2 (bulk query): Omit paths and set change_type_filter to list all changes of that type.

            Tracks change types: sign_convention, coordinate_convention, units, definition_clarification,
            path_renamed, data_type, cocos_label_transformation, added.

            Args:
                paths: Space- or comma-separated IMAS paths (e.g. "equilibrium/time_slice/profiles_1d/psi").
                    Optional when change_type_filter is set.
                change_type_filter: Filter to a specific change type for bulk queries
                    (e.g. 'path_renamed', 'units', 'cocos_label_transformation', 'added').
                ids_filter: Limit results to a specific IDS (e.g. 'equilibrium').
                from_version: Start of version range filter (exclusive).
                to_version: End of version range filter (inclusive).
                follow_rename_chains: When True, traverse RENAMED_TO graph edges to return
                    full multi-hop rename lineages alongside the normal version context output.

            Returns:
                Formatted text report listing notable changes per path across DD versions,
                or a bulk change listing when change_type_filter is used without paths.
                When follow_rename_chains is True, a '### Rename Chains' section is appended.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.version_tool.get_dd_version_context(
                    paths=paths,
                    change_type_filter=change_type_filter,
                    ids_filter=ids_filter,
                    from_version=from_version,
                    to_version=to_version,
                    follow_rename_chains=follow_rename_chains,
                )
            )
            return _format_version_context_report(result)

        @self.mcp.tool()
        def get_dd_versions() -> str:
            """Get metadata about available Data Dictionary versions. Use to determine the current DD version, the range of versions available, and the version chain ordering.

            Returns:
                Formatted text report with current version, version count, available version range, and ordered version chain.
            """
            tools = _get_imas_tools()
            result = _run_async(tools.version_tool.get_dd_versions())
            return _format_dd_versions_report(result)

        @self.mcp.tool()
        def get_dd_changelog(
            ids_filter: str | None = None,
            from_version: str | None = None,
            to_version: str | None = None,
            limit: int = 50,
        ) -> str:
            """Rank IMAS Data Dictionary paths by how much they have changed across DD versions.

            Returns a volatility-scored table showing which paths changed most often,
            what types of changes occurred, and whether they were renamed. Useful for
            answering "which paths change the most?" or identifying unstable paths.

            Args:
                ids_filter: Restrict to one IDS (e.g. 'equilibrium'). Default: all IDSs.
                from_version: Start of version range filter (exclusive, e.g. '3.30.0').
                to_version: End of version range filter (inclusive, e.g. '3.39.0').
                limit: Maximum number of results to return (default 50).

            Returns:
                Formatted ranked table with volatility scores, change type breakdown,
                and rename history per path.
            """
            from imas_codex.llm.search_formatters import format_dd_changelog_report

            tools = _get_imas_tools()
            result = _run_async(
                tools.version_tool.get_dd_changelog(
                    ids_filter=ids_filter,
                    from_version=from_version,
                    to_version=to_version,
                    limit=limit,
                )
            )
            return format_dd_changelog_report(result)

        @self.mcp.tool()
        def get_dd_migration_guide(
            from_version: str,
            to_version: str,
            ids_filter: str | None = None,
            summary_only: bool = False,
            include_recipes: bool = True,
        ) -> str:
            """Generate a migration guide between two DD versions. Returns breaking changes, COCOS sign-flip tables, path renames, unit changes, and code update recipes.

            Args:
                from_version: Source DD version (e.g. "3.39.0").
                to_version: Target DD version (e.g. "4.0.0").
                ids_filter: Optional IDS name to restrict output to a single IDS.
                summary_only: If true, return only aggregate statistics without per-path details. Much faster and smaller response.
                include_recipes: Whether to include code update snippets. Default: true.

            Returns:
                Structured markdown migration guide with breaking changes, COCOS tables, renames, and recipes.
            """
            from imas_codex.tools.migration_guide import generate_migration_guide

            gc = _get_graph_client()
            return generate_migration_guide(
                gc=gc,
                from_version=from_version,
                to_version=to_version,
                ids_filter=ids_filter,
                include_recipes=include_recipes,
                summary_only=summary_only,
            )

        if not self.dd_only:

            @self.mcp.tool()
            def fetch_content(resource: str) -> str:
                """Fetch full text content of a graph resource by ID, URL, or title.

                Use after search_docs/search_code/search_signals returns a
                resource ID you want to read in full. Resolves the resource,
                fetches all associated chunks in reading order, and returns
                concatenated text. For images, returns OCR text or description.

                Args:
                    resource (str, required): One of:
                        - Graph node ID from search results (e.g. "jet:Fishbone_proposal_2018.ppt")
                        - Full URL (e.g. "https://wiki.jetdata.eu/tf/...")
                        - Partial title for fuzzy matching against indexed pages

                Returns:
                    str: Full content report with metadata header and all chunks,
                    or an error message if the resource is not found.
                """
                from imas_codex.llm.search_tools import _fetch as _f

                return _f(resource)

        # =====================================================================
        # Standard Name tools
        # =====================================================================

        @self.mcp.tool()
        def search_standard_names(
            query: str,
            kind: str | None = None,
            tags: list[str] | None = None,
            review_status: str | None = None,
            k: int = 20,
            cocos_type: str | None = None,
        ) -> str:
            """Search standard names by physics concept.

            Hybrid search (vector + keyword) over StandardName descriptions
            and documentation. Enriched with DD path links, unit info, and
            grammar decomposition.

            Args:
                query: Natural-language description of the quantity to find
                    (e.g. "electron temperature", "plasma boundary shape").
                kind: Filter by kind (e.g. "scalar", "vector", "metadata").
                tags: Filter by tags (e.g. ["equilibrium", "core_profiles"]).
                review_status: Filter by review status (e.g. "drafted", "published").
                k: Maximum results to return (default 20).
                cocos_type: Filter by COCOS transformation type (e.g. "psi_like",
                    "ip_like", "b0_like"). Only returns names with that transformation.

            Returns:
                Formatted text report with matched standard names, descriptions,
                units, tags, grammar fields, and relevance scores.
            """
            from imas_codex.llm.sn_tools import _search_standard_names as _ssn

            return _ssn(
                query,
                kind=kind,
                tags=tags,
                review_status=review_status,
                k=k,
                cocos_type=cocos_type,
            )

        @self.mcp.tool()
        def fetch_standard_names(names: str) -> str:
            """Fetch full entries for known standard names.

            Returns complete metadata: description, documentation, unit, kind,
            tags, links, ids_paths, grammar fields, provenance, review status.

            Args:
                names: Space- or comma-separated standard name IDs
                    (e.g. "electron_temperature plasma_current").

            Returns:
                Formatted text report with complete documentation per name.
            """
            from imas_codex.llm.sn_tools import _fetch_standard_names as _fsn

            return _fsn(names)

        @self.mcp.tool()
        def list_standard_names(
            tag: str | None = None,
            kind: str | None = None,
            review_status: str | None = None,
            cocos_type: str | None = None,
        ) -> str:
            """List standard names with optional filters.

            Returns name, description, kind, unit, status for each entry.

            Args:
                tag: Filter by tag (e.g. "equilibrium", "magnetics").
                kind: Filter by kind (e.g. "scalar", "vector").
                review_status: Filter by review status (e.g. "drafted").
                cocos_type: Filter by COCOS transformation type (e.g. "psi_like",
                    "ip_like", "b0_like"). Only returns names with that transformation.

            Returns:
                Formatted markdown table of standard names.
            """
            from imas_codex.llm.sn_tools import _list_standard_names as _lsn

            return _lsn(
                tag=tag, kind=kind, review_status=review_status, cocos_type=cocos_type
            )

        if not self.read_only:
            # =====================================================================
            # Log Tools (Phase 3: MCP Logs)
            # =====================================================================

            @self.mcp.tool()
            def list_logs() -> str:
                """List available log files with sizes and modification times.

                Scans the imas-codex log directory for log files written by
                discovery CLI commands (e.g., paths_tcv.log, wiki_jet.log,
                imas_dd.log). Reads from local disk or remote host via SSH
                depending on configuration.

                Returns:
                    Formatted table of log files with columns: file name,
                    size (B/KB/MB), last-modified ISO timestamp, and age in
                    hours. Returns a message if no log files exist.
                """
                from imas_codex.cli.logging import list_log_files

                files = list_log_files()
                if not files:
                    return "No log files found in ~/.local/share/imas-codex/logs/"

                lines = ["Available log files:", ""]
                for f in files:
                    size = f["size_bytes"]
                    if size >= 1_000_000:
                        size_str = f"{size / 1_000_000:.1f}MB"
                    elif size >= 1_000:
                        size_str = f"{size / 1_000:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    lines.append(
                        f"  {f['name']:<40} {size_str:>8}  "
                        f"modified {f['modified_iso']}  ({f['age_hours']:.1f}h ago)"
                    )
                return "\n".join(lines)

            @self.mcp.tool()
            def get_logs(
                command: str = "signals",
                facility: str | None = None,
                lines: int = 100,
                level: str = "WARNING",
                grep: str | None = None,
                since: str | None = None,
            ) -> str:
                """Read log files with level, text, and time filtering.

                Reads the log file for a given CLI command and facility. Log
                files follow the naming convention {command}_{facility}.log
                (e.g., signals_tcv.log, wiki_jet.log). Use list_logs() first
                to see available files.

                Args:
                    command: CLI command name that produced the log. Common
                        values: "paths", "wiki", "code", "signals",
                        "documents", "imas_dd", "embed". Default: "signals".
                    facility: Facility ID (e.g., "tcv", "jet", "iter",
                        "west", "mast-u"). If omitted, reads {command}.log.
                    lines: Maximum number of matching lines to return.
                        Default: 100.
                    level: Minimum log level filter. One of: "DEBUG", "INFO",
                        "WARNING", "ERROR", "CRITICAL". Default: "WARNING".
                    grep: Case-insensitive substring filter. Only lines
                        containing this text are returned.
                    since: Time filter. Relative: "30m", "1h", "2d".
                        Absolute ISO: "2024-03-13T10:00". Lines older than
                        this threshold are excluded.

                Returns:
                    Filtered log content as text. Empty string if no lines
                    match. Error message if the log file does not exist.
                """
                from imas_codex.cli.logging import read_log

                return read_log(
                    command=command,
                    facility=facility,
                    lines=lines,
                    level=level,
                    grep=grep,
                    since=since,
                )

            @self.mcp.tool()
            def tail_logs(
                command: str = "signals",
                facility: str | None = None,
                lines: int = 50,
            ) -> str:
                """Get the most recent log entries without filtering.

                Returns the last N lines from a log file, regardless of log
                level or content. Use this for a quick look at recent activity;
                use get_logs() when you need level/text/time filtering.

                Args:
                    command: CLI command name that produced the log. Common
                        values: "paths", "wiki", "code", "signals",
                        "documents", "imas_dd", "embed". Default: "signals".
                    facility: Facility ID (e.g., "tcv", "jet", "iter").
                        If omitted, reads {command}.log.
                    lines: Number of lines to return from the end of the
                        file. Default: 50.

                Returns:
                    The last N lines of the log file as text. Error message
                    if the log file does not exist.
                """
                from imas_codex.cli.logging import tail_log

                return tail_log(
                    command=command,
                    facility=facility,
                    lines=lines,
                )

    def _register_prompts(self):
        """Register MCP prompts from markdown files.

        Static prompts: Return content as-is with includes resolved.
        Dynamic prompts (dynamic: true in frontmatter): Accept parameters
        and render with Jinja2 + schema context.
        """
        from imas_codex.llm.prompt_loader import render_prompt

        for name, prompt_def in self._prompts.items():
            is_dynamic = prompt_def.metadata.get("dynamic", False)

            if is_dynamic:
                # Dynamic prompt: accept facility parameter, render with context
                def make_dynamic_prompt_fn(prompt_name: str, pd: PromptDefinition):
                    def prompt_fn(facility: str = "FACILITY") -> str:
                        """Render prompt with facility context and schema values.

                        Args:
                            facility: Facility identifier (e.g., "tcv", "iter")
                        """
                        try:
                            # Get facility infrastructure for ssh_host
                            from imas_codex.discovery import (
                                get_facility_infrastructure as _get_infra,
                            )

                            infra = _get_infra(facility) or {}
                            ssh_host = infra.get("ssh_host", facility)

                            context = {
                                "facility": facility,
                                "ssh_host": ssh_host,
                            }
                            return render_prompt(prompt_name, context)
                        except Exception as e:
                            logger.warning(f"Dynamic render failed: {e}, using static")
                            return pd.content

                    prompt_fn.__name__ = pd.name.replace("-", "_")
                    return prompt_fn

                self.mcp.prompt(name=name, description=prompt_def.description)(
                    make_dynamic_prompt_fn(name, prompt_def)
                )
                logger.debug(f"Registered dynamic prompt: {name}")
            else:
                # Static prompt: return content as-is
                def make_prompt_fn(pd: PromptDefinition):
                    def prompt_fn() -> str:
                        return pd.content

                    prompt_fn.__name__ = pd.name.replace("-", "_")
                    return prompt_fn

                self.mcp.prompt(name=name, description=prompt_def.description)(
                    make_prompt_fn(prompt_def)
                )
                logger.debug(f"Registered static prompt: {name}")

    def _register_health_check(self):
        """Register /health endpoint with graph metadata and uptime."""
        import importlib.metadata
        import time

        from starlette.requests import Request
        from starlette.responses import JSONResponse

        server = self

        def _get_version() -> str:
            try:
                return importlib.metadata.version("imas-codex")
            except Exception:
                return "unknown"

        def _format_uptime(seconds: float) -> str:
            if seconds < 0:
                seconds = 0
            remainder = int(seconds)
            days, remainder = divmod(remainder, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, secs = divmod(remainder, 60)
            parts: list[str] = []
            if days:
                parts.append(f"{days}d")
            if hours or days:
                parts.append(f"{hours}h")
            if minutes or hours or days:
                parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
            return " ".join(parts)

        def _query_graph() -> dict:
            """Query graph for IMAS DD and facility metadata."""
            try:
                from imas_codex.graph.client import GraphClient

                gc = GraphClient.from_profile()
                result: dict = {"db_status": "unavailable"}

                # Graph meta (name, facilities, imas flag)
                try:
                    from imas_codex.graph.meta import get_graph_meta

                    meta = get_graph_meta(gc)
                    if meta:
                        result["graph_name"] = meta.get("name")
                        result["facilities"] = meta.get("facilities") or []
                        result["imas"] = meta.get("imas", False)
                except Exception as exc:
                    logger.warning("Health: graph meta query failed: %s", exc)

                # Node and relationship counts
                try:
                    stats = gc.get_stats()
                    result["node_count"] = stats.get("nodes", 0)
                    result["relationship_count"] = stats.get("relationships", 0)
                    result["db_status"] = (
                        "online" if result["node_count"] > 0 else "empty"
                    )
                except Exception as exc:
                    logger.warning("Health: graph stats query failed: %s", exc)

                # IMAS DD version info
                try:
                    dd_rows = gc.query(
                        "MATCH (d:DDVersion) "
                        "RETURN d.id AS version, d.is_current AS is_current "
                        "ORDER BY d.id"
                    )
                    if dd_rows:
                        versions = [r["version"] for r in dd_rows]
                        current = [r["version"] for r in dd_rows if r.get("is_current")]
                        result["imas_dd"] = {
                            "version": current[0] if current else versions[-1],
                            "min_version": versions[0],
                            "version_count": len(versions),
                        }

                    # IDS and path counts
                    imas_rows = gc.query(
                        "MATCH (p:IMASNode) RETURN count(p) AS paths, "
                        "count(DISTINCT p.ids) AS ids_count"
                    )
                    if imas_rows:
                        result["ids_count"] = imas_rows[0].get("ids_count", 0)
                        result["path_count"] = imas_rows[0].get("paths", 0)
                except Exception:
                    pass

                gc.close()
                return result
            except Exception as exc:
                return {"error": str(exc)}

        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            uptime_seconds = time.monotonic() - server._started_at

            response: dict = {
                "status": "ok",
                "version": _get_version(),
                "uptime": _format_uptime(uptime_seconds),
                "uptime_seconds": round(uptime_seconds, 1),
            }

            graph = _query_graph()
            if "error" in graph:
                response["graph"] = {"status": "unavailable", "error": graph["error"]}
            else:
                db_status = graph.get("db_status", "unavailable")
                graph_section: dict = {
                    "status": db_status,
                    "name": graph.get("graph_name"),
                    "node_count": graph.get("node_count"),
                    "relationship_count": graph.get("relationship_count"),
                }
                # Add GHCR package name for deployment identification
                try:
                    from imas_codex.graph.ghcr import get_package_name

                    graph_section["package"] = get_package_name(dd_only=server.dd_only)
                except Exception:
                    pass
                if server.dd_only:
                    graph_section["variant"] = "dd-only"
                elif graph.get("facilities"):
                    graph_section["variant"] = "full"
                response["graph"] = graph_section

                if graph.get("imas_dd"):
                    response["imas_dd"] = graph["imas_dd"]
                    response["imas_dd"]["ids_count"] = graph.get("ids_count", 0)
                    response["imas_dd"]["path_count"] = graph.get("path_count", 0)

                # Only show facilities when not in dd-only mode
                if not server.dd_only:
                    response["facilities"] = graph.get("facilities", [])

            # Embedding model metadata for semantic search diagnostics
            try:
                from imas_codex.settings import (
                    get_embedding_dimension,
                    get_embedding_model,
                )

                response["embedding"] = {
                    "model": get_embedding_model(),
                    "dimension": get_embedding_dimension(),
                }
            except Exception:
                pass

            # Tool inventory — strip FastMCP internal "@" suffix from keys
            tool_names = sorted(
                k.removeprefix("tool:").rstrip("@")
                for k in server.mcp._local_provider._components
                if k.startswith("tool:")
            )
            mode_parts = []
            if server.read_only:
                mode_parts.append("read-only")
            if server.dd_only:
                mode_parts.append("dd-only")
            response["tools"] = {
                "count": len(tool_names),
                "mode": ", ".join(mode_parts) if mode_parts else "read-write",
                "available": tool_names,
            }

            return JSONResponse(response)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the agents server."""
        if transport == "stdio":
            logger.debug("Starting Agents server with stdio transport")
            self.mcp.run(transport=transport)
        else:
            logger.info(f"Starting Agents server on {host}:{port}")
            # Pass host/port as transport_kwargs — FastMCP.run() forwards
            # them to run_http_async() which passes them to uvicorn.
            self.mcp.run(transport=transport, host=host, port=port)
