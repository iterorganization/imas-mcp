"""Domain query functions for the REPL.

Pre-composed graph queries for common agent operations. These functions
build optimized Cypher, handle embedding calls internally, and return
flat dicts with only relevant properties.

All functions accept ``gc`` (GraphClient) and ``embed_fn`` (callable)
so they can be tested without Neo4j.  When used from the REPL, these
are injected automatically at registration time.
"""

from __future__ import annotations

from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.graph.vector_search import build_vector_search

# ---------------------------------------------------------------------------
# Signal discovery
# ---------------------------------------------------------------------------


def find_signals(
    query: str | None = None,
    *,
    facility: str | None = None,
    diagnostic: str | None = None,
    physics_domain: str | None = None,
    has_data: bool | None = None,
    limit: int = 20,
    include_access: bool = True,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Find facility signals by semantic search and/or filters.

    Combines vector search (when *query* given) with property filters.
    Optionally traverses to ``DataAccess`` for access templates.

    Args:
        query: Natural-language search text (triggers vector search).
        facility: Required. Facility id to scope results.
        diagnostic: Filter by diagnostic name.
        physics_domain: Filter by physics domain.
        has_data: Filter by ``checked`` status.
        limit: Max results.
        include_access: Traverse to ``DataAccess`` for access templates.
        gc: GraphClient instance (injected by REPL).
        embed_fn: Embedding function (injected by REPL).

    Returns:
        List of flat dicts with signal properties and optional access info.

    Raises:
        ValueError: If *facility* is not provided.
    """
    if facility is None:
        raise ValueError("facility is required for find_signals")
    gc, embed_fn = _resolve(gc, embed_fn)

    if query is not None:
        return _find_signals_semantic(
            query,
            facility=facility,
            diagnostic=diagnostic,
            physics_domain=physics_domain,
            has_data=has_data,
            limit=limit,
            include_access=include_access,
            gc=gc,
            embed_fn=embed_fn,
        )

    # Property-only filter
    where_parts = ["(s)-[:AT_FACILITY]->(:Facility {id: $facility})"]
    params: dict[str, Any] = {"facility": facility, "limit": limit}

    if diagnostic is not None:
        where_parts.append("s.diagnostic = $diagnostic")
        params["diagnostic"] = diagnostic
    if physics_domain is not None:
        where_parts.append("s.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain
    if has_data is not None:
        where_parts.append("s.checked = $has_data")
        params["has_data"] = has_data

    where_clause = " AND ".join(where_parts)

    if include_access:
        cypher = (
            f"MATCH (s:FacilitySignal) WHERE {where_clause} "
            "OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess) "
            "RETURN s.id AS id, s.name AS name, s.diagnostic AS diagnostic, "
            "s.description AS description, s.physics_domain AS physics_domain, "
            "s.unit AS unit, s.checked AS checked, "
            "da.data_template AS template_python, "
            "da.access_type AS access_type "
            "ORDER BY s.name LIMIT $limit"
        )
    else:
        cypher = (
            f"MATCH (s:FacilitySignal) WHERE {where_clause} "
            "RETURN s.id AS id, s.name AS name, s.diagnostic AS diagnostic, "
            "s.description AS description, s.physics_domain AS physics_domain, "
            "s.unit AS unit, s.checked AS checked "
            "ORDER BY s.name LIMIT $limit"
        )

    return gc.query(cypher, **params)


def _find_signals_semantic(
    query: str,
    *,
    facility: str,
    diagnostic: str | None,
    physics_domain: str | None,
    has_data: bool | None,
    limit: int,
    include_access: bool,
    gc: GraphClient,
    embed_fn: Any,
) -> list[dict[str, Any]]:
    """Semantic search branch for find_signals."""
    embedding = embed_fn(query)

    # Pre-filters: property-based, pushed into SEARCH block
    pre_filter_parts: list[str] = []
    params: dict[str, Any] = {
        "facility": facility,
        "k": limit,
        "embedding": embedding,
    }
    if diagnostic is not None:
        pre_filter_parts.append("s.diagnostic = $diagnostic")
        params["diagnostic"] = diagnostic
    if physics_domain is not None:
        pre_filter_parts.append("s.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain
    if has_data is not None:
        pre_filter_parts.append("s.checked = $has_data")
        params["has_data"] = has_data

    search_block = build_vector_search(
        "facility_signal_desc_embedding",
        "FacilitySignal",
        where_clauses=pre_filter_parts or None,
        node_alias="s",
        score_alias="score",
    )

    # Post-filter: relationship-based, must remain outside SEARCH
    post_where = "WHERE (s)-[:AT_FACILITY]->(:Facility {id: $facility})\n"

    access_clause = ""
    access_return = ""
    if include_access:
        access_clause = "OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)\n"
        access_return = (
            ", da.data_template AS template_python, da.access_type AS access_type"
        )

    cypher = (
        f"{search_block}\n"
        f"{post_where}"
        f"{access_clause}"
        "RETURN s.id AS id, s.name AS name, s.diagnostic AS diagnostic, "
        "s.description AS description, s.physics_domain AS physics_domain, "
        f"s.unit AS unit, s.checked AS checked, score{access_return} "
        "ORDER BY score DESC"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Wiki search
# ---------------------------------------------------------------------------


def find_wiki(
    query: str | None = None,
    *,
    facility: str | None = None,
    text_contains: str | None = None,
    page_title_contains: str | None = None,
    k: int = 10,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Search wiki content by semantic similarity, keyword, or both.

    Supports three modes:

    1. **Semantic only** (``query`` given): Vector search on chunk embeddings.
    2. **Keyword only** (``text_contains`` / ``page_title_contains``):
       Text filtering without embeddings.
    3. **Combined** (``query`` + keyword filters): Semantic search
       post-filtered by keyword match.

    Args:
        query: Search text for vector similarity. Omit for keyword-only.
        facility: Optional facility filter.
        text_contains: Filter chunks whose text contains this substring
            (case-insensitive via ``toLower``).
        page_title_contains: Filter by parent page title substring
            (case-insensitive).
        k: Number of results.
        gc: GraphClient instance.
        embed_fn: Embedding function.
    """
    gc, embed_fn = _resolve(gc, embed_fn)

    if query is not None:
        return _find_wiki_semantic(
            query,
            facility=facility,
            text_contains=text_contains,
            page_title_contains=page_title_contains,
            k=k,
            gc=gc,
            embed_fn=embed_fn,
        )

    # Keyword-only mode (no semantic search)
    if text_contains is None and page_title_contains is None:
        raise ValueError(
            "Provide 'query' for semantic search, or 'text_contains'/"
            "'page_title_contains' for keyword search"
        )

    where_parts: list[str] = []
    params: dict[str, Any] = {"limit": k}

    if facility is not None:
        where_parts.append("(c)-[:AT_FACILITY]->(:Facility {id: $facility})")
        params["facility"] = facility
    if text_contains is not None:
        where_parts.append("toLower(c.text) CONTAINS toLower($text_kw)")
        params["text_kw"] = text_contains
    if page_title_contains is not None:
        where_parts.append("toLower(p.title) CONTAINS toLower($title_kw)")
        params["title_kw"] = page_title_contains

    where_clause = " AND ".join(where_parts) if where_parts else "true"

    cypher = (
        "MATCH (p:WikiPage)-[:HAS_CHUNK]->(c:WikiChunk) "
        f"WHERE {where_clause} "
        "RETURN c.text AS text, c.section AS section, "
        "p.title AS page_title, p.url AS page_url "
        "ORDER BY p.title LIMIT $limit"
    )

    return gc.query(cypher, **params)


def _find_wiki_semantic(
    query: str,
    *,
    facility: str | None,
    text_contains: str | None,
    page_title_contains: str | None,
    k: int,
    gc: GraphClient,
    embed_fn: Any,
) -> list[dict[str, Any]]:
    """Semantic search branch for find_wiki, with optional keyword filters."""
    embedding = embed_fn(query)
    params: dict[str, Any] = {"k": k, "embedding": embedding}

    # No property pre-filters: facility uses AT_FACILITY relationship,
    # text CONTAINS is a substring match — both stay as post-filters.
    search_block = build_vector_search(
        "wiki_chunk_embedding",
        "WikiChunk",
        node_alias="c",
        score_alias="score",
    )

    # Build post-filter conditions on the vector search results
    post_filters: list[str] = []
    if facility is not None:
        post_filters.append("(c)-[:AT_FACILITY]->(:Facility {id: $facility})")
        params["facility"] = facility
    if text_contains is not None:
        post_filters.append("toLower(c.text) CONTAINS toLower($text_kw)")
        params["text_kw"] = text_contains

    post_where = ""
    if post_filters:
        post_where = "WHERE " + " AND ".join(post_filters) + "\n"

    # Page title filter is applied after the OPTIONAL MATCH join
    title_filter = ""
    if page_title_contains is not None:
        title_filter = "WHERE toLower(p.title) CONTAINS toLower($title_kw)\n"
        params["title_kw"] = page_title_contains

    cypher = (
        f"{search_block}\n"
        f"{post_where}"
        "OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(c)\n"
        f"{title_filter}"
        "RETURN c.text AS text, c.section AS section, "
        "p.title AS page_title, p.url AS page_url, score "
        "ORDER BY score DESC"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Wiki page chunk retrieval
# ---------------------------------------------------------------------------


def wiki_page_chunks(
    title_contains: str,
    *,
    facility: str | None = None,
    text_contains: str | None = None,
    limit: int = 50,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Get all chunks from wiki pages matching a title pattern.

    Common pattern: find pages by title, then retrieve all their content.
    Replaces the two-step "find pages → query chunks" manual Cypher pattern.

    Args:
        title_contains: Substring to match in page title (case-insensitive).
        facility: Optional facility filter.
        text_contains: Optional keyword filter on chunk text.
        limit: Max chunks to return.
        gc: GraphClient instance.
        embed_fn: Embedding function (unused, accepted for bind consistency).
    """
    gc, _ = _resolve(gc, None)

    where_parts = ["toLower(p.title) CONTAINS toLower($title_kw)"]
    params: dict[str, Any] = {"title_kw": title_contains, "limit": limit}

    if facility is not None:
        where_parts.append("p.facility_id = $facility")
        params["facility"] = facility
    if text_contains is not None:
        where_parts.append("toLower(c.text) CONTAINS toLower($text_kw)")
        params["text_kw"] = text_contains

    where_clause = " AND ".join(where_parts)

    cypher = (
        "MATCH (p:WikiPage)-[:HAS_CHUNK]->(c:WikiChunk) "
        f"WHERE {where_clause} "
        "RETURN p.title AS page_title, p.url AS page_url, "
        "p.facility_id AS facility, "
        "c.section AS section, c.text AS text "
        "ORDER BY p.title, c.section LIMIT $limit"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# IMAS path search
# ---------------------------------------------------------------------------


def find_imas(
    query: str,
    *,
    ids_filter: str | None = None,
    include_deprecated: bool = False,
    limit: int = 20,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Find IMAS paths by semantic similarity.

    Returns path id, name, IDS, documentation, data type, units,
    physics domain, and score. Filters deprecated paths by default.

    Args:
        query: Search text.
        ids_filter: Filter by IDS name (e.g. ``"equilibrium"``).
        include_deprecated: Include deprecated paths (default False).
        limit: Max results.
        gc: GraphClient instance.
        embed_fn: Embedding function.
    """
    gc, embed_fn = _resolve(gc, embed_fn)
    embedding = embed_fn(query)

    params: dict[str, Any] = {"k": limit, "embedding": embedding}

    # Pre-filters: property-based, pushed into SEARCH block
    pre_filter_parts: list[str] = ["p.node_category = 'data'"]
    if ids_filter is not None:
        pre_filter_parts.append("p.ids = $ids_filter")
        params["ids_filter"] = ids_filter

    search_block = build_vector_search(
        "imas_node_embedding",
        "IMASNode",
        where_clauses=pre_filter_parts,
        node_alias="p",
        score_alias="score",
    )

    # Post-filter: relationship-based deprecated check stays outside SEARCH
    deprecated_filter = ""
    if not include_deprecated:
        deprecated_filter = "WHERE NOT (p)-[:DEPRECATED_IN]->(:DDVersion)\n"

    cypher = (
        f"{search_block}\n"
        f"{deprecated_filter}"
        "OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster) "
        "OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit) "
        "RETURN p.id AS id, p.name AS name, p.ids AS ids, "
        "p.documentation AS documentation, p.data_type AS data_type, "
        "u.symbol AS unit, p.physics_domain AS physics_domain, "
        "collect(DISTINCT cl.label) AS clusters, score "
        "ORDER BY score DESC"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Code search
# ---------------------------------------------------------------------------


def find_code(
    query: str,
    *,
    facility: str | None = None,
    limit: int = 10,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Semantic search over ingested code chunks.

    Returns code text (truncated), function name, source file, and score.

    Args:
        query: Search text.
        facility: Optional facility filter.
        limit: Max results.
        gc: GraphClient instance.
        embed_fn: Embedding function.
    """
    gc, embed_fn = _resolve(gc, embed_fn)
    embedding = embed_fn(query)

    params: dict[str, Any] = {"k": limit, "embedding": embedding}

    # No pre-filters: facility is accessed via CodeFile join, not a property on CodeChunk
    search_block = build_vector_search(
        "code_chunk_embedding",
        "CodeChunk",
        node_alias="cc",
        score_alias="score",
    )

    facility_filter = ""
    if facility is not None:
        facility_filter = (
            "MATCH (cf:CodeFile) WHERE cf.id = ce.source_file "
            "AND cf.facility_id = $facility\n"
        )
        params["facility"] = facility

    cypher = (
        f"{search_block}\n"
        "OPTIONAL MATCH (cc)-[:CODE_EXAMPLE_ID]->(ce:CodeExample)\n"
        f"{facility_filter}"
        "RETURN substring(cc.text, 0, 500) AS text, "
        "cc.function_name AS function_name, "
        "ce.source_file AS source_file, score "
        "ORDER BY score DESC"
    )

    return gc.query(cypher, **params)

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Tree node exploration
# ---------------------------------------------------------------------------


def find_data_nodes(
    query: str | None = None,
    *,
    facility: str | None = None,
    data_source_name: str | None = None,
    path_prefix: str | None = None,
    physics_domain: str | None = None,
    limit: int = 30,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Explore MDSplus tree nodes by semantic search or property filters.

    Args:
        query: Search text (triggers vector search on description embedding).
        facility: Required facility id.
        data_source_name: Filter by tree name.
        path_prefix: Filter by path prefix (e.g. ``"\\\\RESULTS::"``)..
        physics_domain: Filter by physics domain.
        limit: Max results.
        gc: GraphClient instance.
        embed_fn: Embedding function.
    """
    gc, embed_fn = _resolve(gc, embed_fn)

    if query is not None:
        return _find_data_nodes_semantic(
            query,
            facility=facility,
            data_source_name=data_source_name,
            path_prefix=path_prefix,
            physics_domain=physics_domain,
            limit=limit,
            gc=gc,
            embed_fn=embed_fn,
        )

    # Property-only search
    where_parts: list[str] = []
    params: dict[str, Any] = {"limit": limit}

    if facility is not None:
        where_parts.append("(n)-[:AT_FACILITY]->(:Facility {id: $facility})")
        params["facility"] = facility
    if data_source_name is not None:
        where_parts.append("n.data_source_name = $data_source_name")
        params["data_source_name"] = data_source_name
    if path_prefix is not None:
        where_parts.append("n.path STARTS WITH $path_prefix")
        params["path_prefix"] = path_prefix
    if physics_domain is not None:
        where_parts.append("n.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts) + " "

    cypher = (
        f"MATCH (n:SignalNode) {where_clause}"
        "RETURN n.id AS id, n.path AS path, n.data_source_name AS data_source_name, "
        "n.description AS description, n.unit AS unit, "
        "n.physics_domain AS physics_domain, n.node_type AS node_type "
        "ORDER BY n.path LIMIT $limit"
    )

    return gc.query(cypher, **params)


def _find_data_nodes_semantic(
    query: str,
    *,
    facility: str | None,
    data_source_name: str | None,
    path_prefix: str | None,
    physics_domain: str | None,
    limit: int,
    gc: GraphClient,
    embed_fn: Any,
) -> list[dict[str, Any]]:
    """Semantic search branch for find_data_nodes."""
    embedding = embed_fn(query)

    # Pre-filters: property-based, pushed into SEARCH block
    pre_filter_parts: list[str] = []
    # Post-filters: relationship-based, must remain outside SEARCH
    post_filter_parts: list[str] = []
    params: dict[str, Any] = {"k": limit, "embedding": embedding}

    if facility is not None:
        post_filter_parts.append("(n)-[:AT_FACILITY]->(:Facility {id: $facility})")
        params["facility"] = facility
    if data_source_name is not None:
        pre_filter_parts.append("n.data_source_name = $data_source_name")
        params["data_source_name"] = data_source_name
    if path_prefix is not None:
        pre_filter_parts.append("n.path STARTS WITH $path_prefix")
        params["path_prefix"] = path_prefix
    if physics_domain is not None:
        pre_filter_parts.append("n.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain

    search_block = build_vector_search(
        "signal_node_desc_embedding",
        "SignalNode",
        where_clauses=pre_filter_parts or None,
        node_alias="n",
        score_alias="score",
    )

    post_where = ""
    if post_filter_parts:
        post_where = "WHERE " + " AND ".join(post_filter_parts) + "\n"

    cypher = (
        f"{search_block}\n"
        f"{post_where}"
        "RETURN n.id AS id, n.path AS path, n.data_source_name AS data_source_name, "
        "n.description AS description, n.unit AS unit, "
        "n.physics_domain AS physics_domain, n.node_type AS node_type, score "
        "ORDER BY score DESC"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Signal-to-IMAS mapping
# ---------------------------------------------------------------------------


def map_signals_to_imas(
    facility: str,
    *,
    diagnostic: str | None = None,
    physics_domain: str | None = None,
    limit: int = 50,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Find facility signals and their IMAS path mappings.

    Traverses FacilitySignal -> HAS_DATA_SOURCE_NODE -> SignalNode
    -> MEMBER_OF -> SignalSource -> MAPS_TO_IMAS -> IMASNode.

    Args:
        facility: Facility id.
        diagnostic: Optional diagnostic filter.
        physics_domain: Optional physics domain filter.
        limit: Max results.
        gc: GraphClient instance.
        embed_fn: Embedding function.
    """
    gc, _ = _resolve(gc, None)

    where_parts = ["(s)-[:AT_FACILITY]->(:Facility {id: $facility})"]
    params: dict[str, Any] = {"facility": facility, "limit": limit}

    if diagnostic is not None:
        where_parts.append("s.diagnostic = $diagnostic")
        params["diagnostic"] = diagnostic
    if physics_domain is not None:
        where_parts.append("s.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain

    where_clause = " AND ".join(where_parts)

    cypher = (
        f"MATCH (s:FacilitySignal) WHERE {where_clause} "
        "OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess) "
        "OPTIONAL MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(dn:SignalNode)"
        "-[:MEMBER_OF]->(sg:SignalSource)-[:MAPS_TO_IMAS]->(ip:IMASNode) "
        "RETURN s.id AS signal_id, s.name AS signal_name, "
        "s.diagnostic AS diagnostic, s.description AS signal_description, "
        "ip.id AS imas_path, ip.documentation AS imas_documentation, "
        "da.data_template AS template_python "
        "ORDER BY s.diagnostic, s.name LIMIT $limit"
    )

    return gc.query(cypher, **params)


# ---------------------------------------------------------------------------
# Facility overview
# ---------------------------------------------------------------------------


def facility_overview(
    facility: str,
    *,
    gc: GraphClient | None = None,
    embed_fn: Any = None,
) -> dict[str, Any]:
    """Single-call facility summary with counts and key entities.

    Args:
        facility: Facility id.
        gc: GraphClient instance.
        embed_fn: Embedding function.

    Returns:
        Dict with facility name, counts, diagnostics, trees, etc.
    """
    gc, _ = _resolve(gc, None)

    # Aggregate counts in one query
    results = gc.query(
        """
        MATCH (f:Facility {id: $facility})
        OPTIONAL MATCH (d:Diagnostic)-[:AT_FACILITY]->(f)
        OPTIONAL MATCH (t:DataSource)-[:AT_FACILITY]->(f)
        OPTIONAL MATCH (s:FacilitySignal)-[:AT_FACILITY]->(f)
        OPTIONAL MATCH (wp:WikiPage)-[:AT_FACILITY]->(f)
        OPTIONAL MATCH (cf:CodeFile)-[:AT_FACILITY]->(f)
        RETURN count(DISTINCT d) AS diagnostics,
               count(DISTINCT t) AS trees,
               count(DISTINCT s) AS signals,
               count(DISTINCT wp) AS wiki_pages,
               count(DISTINCT cf) AS code_files
        """,
        facility=facility,
    )

    counts = (
        results[0]
        if results
        else {
            "diagnostics": 0,
            "trees": 0,
            "signals": 0,
            "wiki_pages": 0,
            "code_files": 0,
        }
    )

    return {"facility": facility, **counts}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(gc: GraphClient | None, embed_fn: Any) -> tuple[GraphClient, Any]:
    """Resolve gc/embed_fn from defaults when called outside REPL."""
    if gc is None:
        gc = GraphClient()
    if embed_fn is None:

        def _noop_embed(text: str) -> list[float]:
            raise RuntimeError(
                "embed_fn not provided — pass embed_fn= or call from REPL"
            )

        embed_fn = _noop_embed
    return gc, embed_fn
