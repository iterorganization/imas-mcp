"""Unified MCP search tool implementations.

Each function performs:
1. Embed the query text
2. Fan out vector searches across relevant indexes
3. Enrich results via graph traversal
4. Format into a text report

Functions are prefixed with ``_`` — they are registered as MCP tools
in ``server.py`` via ``@self.mcp.tool()``.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j.exceptions import ServiceUnavailable

from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
from imas_codex.graph.client import GraphClient
from imas_codex.llm.search_formatters import (
    format_code_report,
    format_docs_report,
    format_fetch_report,
    format_signals_report,
)

logger = logging.getLogger(__name__)

NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Check service with: systemctl --user status imas-codex-neo4j"
)

_EMBEDDING_UNAVAILABLE_MSG = (
    "Embedding service unavailable. Use python() with graph_search() "
    "for property-based queries."
)


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    msg = str(e)
    if "Connection refused" in msg or "ServiceUnavailable" in msg:
        return NEO4J_NOT_RUNNING_MSG
    return msg


def _embed(encoder: Encoder, text: str) -> list[float]:
    """Embed a single text string, returning the vector."""
    result = encoder.embed_texts([text])[0]
    return result.tolist() if hasattr(result, "tolist") else list(result)


# ---------------------------------------------------------------------------
# search_signals
# ---------------------------------------------------------------------------


def _search_signals(
    query: str,
    facility: str,
    *,
    diagnostic: str | None = None,
    physics_domain: str | None = None,
    check_status: str | None = None,
    error_type: str | None = None,
    include_check_details: bool = False,
    k: int = 10,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search facility signals with graph enrichment.

    Performs vector search on signal descriptions, enriches with
    DataAccess, Diagnostic, SignalNode, and IMASNode traversals,
    then formats the result.

    Optional check outcome filtering:
        check_status: "passed", "failed", or "unchecked"
        error_type: filter by error classification (e.g. "not_available_for_shot")
        include_check_details: include CHECKED_WITH relationship data in results
    """
    try:
        if gc is None:
            gc = GraphClient()
        if encoder is None:
            from imas_codex.embeddings.config import EncoderConfig

            encoder = Encoder(EncoderConfig())

        embedding = _embed(encoder, query)
    except EmbeddingBackendError as e:
        return f"{_EMBEDDING_UNAVAILABLE_MSG}\n\nDetail: {e}"
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error initializing search: {e}"

    try:
        # Step 1: Vector search on signal descriptions
        signal_ids, scores = _vector_search_signals(
            gc, embedding, facility, k, diagnostic, physics_domain
        )

        # Step 1b: Text search for keyword matches (hybrid boost)
        text_signals = _text_search_signals(gc, query, facility, k)
        for r in text_signals:
            sid = r["id"]
            text_score = round(r["score"], 3)
            if sid in scores:
                scores[sid] = round(scores[sid] * 0.7 + text_score * 0.3 + 0.1, 3)
            else:
                scores[sid] = text_score
                signal_ids.append(sid)

        # Re-sort by score and limit to k
        signal_ids = sorted(
            set(signal_ids), key=lambda sid: scores.get(sid, 0), reverse=True
        )[:k]

        # Step 1c: Filter by check outcome if requested
        if check_status or error_type:
            signal_ids = _filter_by_check_outcome(
                gc, signal_ids, check_status, error_type
            )

        if not signal_ids:
            # Fall back to data node search only
            data_node_results = _vector_search_data_nodes(gc, embedding, facility, k)
            return format_signals_report([], data_node_results, {})

        # Step 2: Enrich with graph traversals
        enriched = _enrich_signals(gc, signal_ids)

        # Step 2b: Add check details if requested
        if include_check_details:
            check_details = _get_check_details(gc, signal_ids)
            for sig in enriched:
                sid = sig.get("id", "")
                if sid in check_details:
                    sig["check_details"] = check_details[sid]

        # Step 3: Data node search (secondary index)
        data_node_results = _vector_search_data_nodes(gc, embedding, facility, k)

        # Step 4: Format
        return format_signals_report(enriched, data_node_results, scores)

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("search_signals failed")
        return f"Search error: {_neo4j_error_message(e)}"


def _filter_by_check_outcome(
    gc: GraphClient,
    signal_ids: list[str],
    check_status: str | None,
    error_type: str | None,
) -> list[str]:
    """Filter signal IDs by check outcome via CHECKED_WITH relationship."""
    if not signal_ids:
        return []

    if check_status == "unchecked":
        # Return signals WITHOUT a CHECKED_WITH relationship
        cypher = """
            UNWIND $signal_ids AS sid
            MATCH (s:FacilitySignal {id: sid})
            WHERE NOT EXISTS { (s)-[:CHECKED_WITH]->() }
            RETURN s.id AS id
        """
        results = gc.query(cypher, signal_ids=signal_ids)
        return [r["id"] for r in results]

    where_parts = []
    params: dict[str, Any] = {"signal_ids": signal_ids}

    if check_status == "passed":
        where_parts.append("c.success = true")
    elif check_status == "failed":
        where_parts.append("c.success = false")

    if error_type:
        where_parts.append("c.error_type = $error_type")
        params["error_type"] = error_type

    where_clause = " AND ".join(where_parts) if where_parts else "true"

    cypher = f"""
        UNWIND $signal_ids AS sid
        MATCH (s:FacilitySignal {{id: sid}})-[c:CHECKED_WITH]->()
        WHERE {where_clause}
        RETURN DISTINCT s.id AS id
    """
    results = gc.query(cypher, **params)
    return [r["id"] for r in results]


def _get_check_details(
    gc: GraphClient,
    signal_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Get CHECKED_WITH relationship details for signals."""
    if not signal_ids:
        return {}

    cypher = """
        UNWIND $signal_ids AS sid
        MATCH (s:FacilitySignal {id: sid})-[c:CHECKED_WITH]->(da:DataAccess)
        RETURN s.id AS id,
               c.success AS success,
               c.error AS error,
               c.error_type AS error_type,
               c.shot AS shot,
               c.shape AS shape,
               c.checked_at AS checked_at,
               da.id AS data_access_id
    """
    results = gc.query(cypher, signal_ids=signal_ids)
    details: dict[str, dict[str, Any]] = {}
    for r in results:
        details[r["id"]] = {
            "success": r["success"],
            "error": r.get("error"),
            "error_type": r.get("error_type"),
            "shot": r.get("shot"),
            "shape": r.get("shape"),
            "checked_at": str(r["checked_at"]) if r.get("checked_at") else None,
            "data_access_id": r.get("data_access_id"),
        }
    return details


# ---------------------------------------------------------------------------
# signal_analytics
# ---------------------------------------------------------------------------

# Allowed grouping dimensions for signal analytics
_ALLOWED_GROUP_BY = {
    "status",
    "physics_domain",
    "data_source_name",
    "discovery_source",
    "diagnostic",
    "check_status",
    "error_type",
}


def _signal_analytics(
    facility: str,
    group_by: list[str] | None = None,
    filters: dict[str, str] | None = None,
    gc: GraphClient | None = None,
) -> str:
    """Aggregate signal counts by specified dimensions.

    Replaces the need for direct Cypher for common analytics queries.

    Args:
        facility: Facility identifier (e.g. "jet", "tcv")
        group_by: Dimensions to group by. Allowed values:
            status, physics_domain, data_source_name, discovery_source,
            diagnostic, check_status, error_type
        filters: Optional key-value filters (e.g. {"status": "checked"})

    Returns:
        Formatted analytics report with counts per group.
    """
    if gc is None:
        gc = GraphClient()

    if not group_by:
        group_by = ["status"]

    # Validate group_by dimensions
    invalid = set(group_by) - _ALLOWED_GROUP_BY
    if invalid:
        return f"Invalid group_by dimensions: {invalid}. Allowed: {sorted(_ALLOWED_GROUP_BY)}"

    needs_check_join = "check_status" in group_by or "error_type" in group_by
    filter_needs_check = filters and (
        "check_status" in filters or "error_type" in filters
    )
    needs_check = needs_check_join or filter_needs_check

    try:
        if needs_check:
            return _analytics_with_checks(gc, facility, group_by, filters)
        return _analytics_simple(gc, facility, group_by, filters)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("signal_analytics failed")
        return f"Analytics error: {_neo4j_error_message(e)}"


def _analytics_simple(
    gc: GraphClient,
    facility: str,
    group_by: list[str],
    filters: dict[str, str] | None,
) -> str:
    """Signal analytics without check relationship joins."""
    where_parts = ["s.facility_id = $facility"]
    params: dict[str, Any] = {"facility": facility}

    if filters:
        for fkey, fval in filters.items():
            if fkey in _ALLOWED_GROUP_BY and fkey not in ("check_status", "error_type"):
                param_name = f"f_{fkey}"
                where_parts.append(f"s.{fkey} = ${param_name}")
                params[param_name] = fval

    where_clause = " AND ".join(where_parts)
    return_cols = ", ".join(f"s.{dim} AS {dim}" for dim in group_by)

    cypher = f"""
        MATCH (s:FacilitySignal)
        WHERE {where_clause}
        RETURN {return_cols}, count(s) AS count
        ORDER BY count DESC
    """
    results = gc.query(cypher, **params)
    return _format_analytics(group_by, results, facility)


def _analytics_with_checks(
    gc: GraphClient,
    facility: str,
    group_by: list[str],
    filters: dict[str, str] | None,
) -> str:
    """Signal analytics with CHECKED_WITH relationship joins."""
    where_parts = ["s.facility_id = $facility"]
    params: dict[str, Any] = {"facility": facility}

    if filters:
        for fkey, fval in filters.items():
            if fkey in ("check_status", "error_type"):
                continue  # Handled via relationship
            if fkey in _ALLOWED_GROUP_BY:
                param_name = f"f_{fkey}"
                where_parts.append(f"s.{fkey} = ${param_name}")
                params[param_name] = fval

    where_clause = " AND ".join(where_parts)

    # Build return columns, mapping check_status and error_type to relationship
    return_parts = []
    for dim in group_by:
        if dim == "check_status":
            return_parts.append(
                "CASE WHEN c IS NULL THEN 'unchecked' "
                "WHEN c.success = true THEN 'passed' "
                "ELSE 'failed' END AS check_status"
            )
        elif dim == "error_type":
            return_parts.append("c.error_type AS error_type")
        else:
            return_parts.append(f"s.{dim} AS {dim}")

    return_cols = ", ".join(return_parts)

    # Apply check-specific filters
    check_where = []
    if filters and "check_status" in filters:
        cs = filters["check_status"]
        if cs == "passed":
            check_where.append("c.success = true")
        elif cs == "failed":
            check_where.append("c.success = false")
        elif cs == "unchecked":
            check_where.append("c IS NULL")
    if filters and "error_type" in filters:
        params["f_error_type"] = filters["error_type"]
        check_where.append("c.error_type = $f_error_type")

    check_filter = (" AND " + " AND ".join(check_where)) if check_where else ""

    cypher = f"""
        MATCH (s:FacilitySignal)
        WHERE {where_clause}
        OPTIONAL MATCH (s)-[c:CHECKED_WITH]->()
        WITH s, c
        WHERE true{check_filter}
        RETURN {return_cols}, count(DISTINCT s) AS count
        ORDER BY count DESC
    """
    results = gc.query(cypher, **params)
    return _format_analytics(group_by, results, facility)


def _format_analytics(
    group_by: list[str],
    results: list[dict[str, Any]],
    facility: str,
) -> str:
    """Format analytics results into a readable table."""
    if not results:
        return f"No signals found for facility '{facility}'."

    total = sum(r["count"] for r in results)
    parts = [f"## Signal Analytics for {facility}\n"]
    parts.append(f"Total: {total:,} signals\n")

    # Header
    headers = group_by + ["count", "%"]
    parts.append("| " + " | ".join(headers) + " |")
    parts.append("| " + " | ".join("---" for _ in headers) + " |")

    for r in results:
        row = []
        for dim in group_by:
            row.append(str(r.get(dim, "—")))
        count = r["count"]
        pct = f"{count / total * 100:.1f}" if total > 0 else "0.0"
        row.extend([str(count), pct])
        parts.append("| " + " | ".join(row) + " |")

    return "\n".join(parts)


def _vector_search_signals(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
    diagnostic: str | None,
    physics_domain: str | None,
) -> tuple[list[str], dict[str, float]]:
    """Vector search on facility_signal_desc_embedding index.

    Uses property-based facility filter with over-fetching to avoid
    facility starvation when one facility dominates the index.
    """
    # Over-fetch to avoid facility starvation
    internal_k = max(k * 5, 200)
    where_parts = ["s.facility_id = $facility"]
    params: dict[str, Any] = {
        "k": internal_k,
        "embedding": embedding,
        "facility": facility,
        "limit": k,
    }

    if diagnostic is not None:
        where_parts.append("s.diagnostic = $diagnostic")
        params["diagnostic"] = diagnostic
    if physics_domain is not None:
        where_parts.append("s.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain

    where_clause = " AND ".join(where_parts)

    cypher = (
        'CALL db.index.vector.queryNodes("facility_signal_desc_embedding", $k, $embedding) '
        f"YIELD node AS s, score WHERE {where_clause} "
        "RETURN s.id AS id, score "
        "ORDER BY score DESC LIMIT $limit"
    )
    results = gc.query(cypher, **params)
    # Deduplicate: vector search can return the same signal from multiple
    # embedding entries. Keep highest score per signal ID.
    scores: dict[str, float] = {}
    for r in results:
        sid = r["id"]
        score = round(r["score"], 3)
        if sid not in scores or score > scores[sid]:
            scores[sid] = score
    ids = sorted(scores, key=lambda sid: scores[sid], reverse=True)
    return ids, scores


def _text_search_signals(
    gc: GraphClient,
    query: str,
    facility: str,
    k: int,
) -> list[dict[str, Any]]:
    """Text-based search on signals by name, node_path, and description.

    Uses fulltext index for BM25 scoring when available, falls back to
    CONTAINS with hardcoded scores otherwise.
    """
    # Try fulltext index first (BM25 scoring)
    try:
        cypher = """
            CALL db.index.fulltext.queryNodes('facility_signal_text', $query)
            YIELD node AS s, score
            WHERE s.facility_id = $facility
            RETURN s.id AS id, score
            LIMIT $limit
        """
        results = gc.query(cypher, query=query, facility=facility, limit=k * 2)
        if results:
            return results
    except Exception:
        pass

    # Fallback: CONTAINS with fixed score
    query_lower = query.lower()
    cypher = """
        MATCH (s:FacilitySignal)
        WHERE s.facility_id = $facility
          AND (
            toLower(s.name) CONTAINS $query_lower
            OR toLower(s.description) CONTAINS $query_lower
            OR toLower(s.node_path) CONTAINS $query_lower
          )
        RETURN s.id AS id, 0.6 AS score
        LIMIT $limit
    """
    return gc.query(cypher, facility=facility, query_lower=query_lower, limit=k * 2)


def _enrich_signals(
    gc: GraphClient,
    signal_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich signal IDs with full context via graph traversal.

    Collects multiple DataAccess methods into arrays to avoid cartesian
    product duplication when a signal has 2+ access methods.

    Also returns signal properties ``node_path``, ``accessor``, and
    ``data_source_name`` for template placeholder interpolation.
    """
    cypher = """
        UNWIND $signal_ids AS sid
        MATCH (s:FacilitySignal {id: sid})
        OPTIONAL MATCH (s)-[:BELONGS_TO_DIAGNOSTIC]->(diag:Diagnostic)
        OPTIONAL MATCH (s)-[:HAS_DATA_SOURCE_NODE]->(tn:SignalNode)
        OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
        OPTIONAL MATCH (tn)-[:MEMBER_OF]->(sg:SignalSource)-[:MAPS_TO_IMAS]->(ip:IMASNode)
        OPTIONAL MATCH (ip)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (s)-[:HAS_UNIT]->(su:Unit)
        OPTIONAL MATCH (s)-[:MEMBER_OF]->(fsgrp:SignalSource)
        OPTIONAL MATCH (wc:WikiChunk)-[:DOCUMENTS]->(s)
        WITH s, tn, su, fsgrp,
             collect(DISTINCT diag.name) AS diagnostic_names,
             head(collect(DISTINCT diag.category)) AS diagnostic_category,
             collect(DISTINCT {
               access_template: da.data_template,
               access_type: da.access_type,
               method_type: da.method_type,
               imports_template: da.imports_template,
               connection_template: da.connection_template,
               imas_path: ip.id,
               imas_docs: ip.documentation,
               imas_unit: u.symbol
             }) AS access_methods,
             collect(DISTINCT wc.section) AS wiki_mentions
        RETURN s.id AS id, s.name AS name, s.description AS description,
               s.physics_domain AS physics_domain, s.unit AS unit,
               s.keywords AS keywords, s.aliases AS aliases,
               s.sign_convention AS sign_convention, s.cocos AS cocos,
               s.analysis_code AS analysis_code,
               s.tdi_function AS tdi_function, s.tdi_quantity AS tdi_quantity,
               s.enrichment_source AS enrichment_source,
               s.checked AS checked, s.example_shot AS example_shot,
               s.node_path AS node_path, s.accessor AS accessor,
               s.data_source_name AS data_source_name,
               su.symbol AS unit_symbol,
               head(diagnostic_names) AS diagnostic_name,
               diagnostic_category,
               tn.path AS tree_path, tn.data_source_name AS tree_data_source_name,
               fsgrp.id AS signal_source_id,
               fsgrp.group_key AS signal_source_key,
               fsgrp.description AS signal_source_description,
               fsgrp.member_count AS signal_source_member_count,
               wiki_mentions,
               access_methods
    """
    return gc.query(cypher, signal_ids=signal_ids)


def _vector_search_data_nodes(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
) -> list[dict[str, Any]]:
    """Vector search on signal_node_desc_embedding index.

    Uses property-based facility filter with over-fetching.
    """
    internal_k = max(k * 5, 200)
    cypher = (
        'CALL db.index.vector.queryNodes("signal_node_desc_embedding", $k, $embedding) '
        "YIELD node AS n, score "
        "WHERE n.facility_id = $facility "
        "RETURN n.id AS id, n.path AS path, n.data_source_name AS data_source_name, "
        "n.description AS description, n.unit AS unit, score "
        "ORDER BY score DESC LIMIT $limit"
    )
    results = gc.query(
        cypher, k=internal_k, embedding=embedding, facility=facility, limit=k
    )
    # Deduplicate by id, keeping the highest-scoring entry
    seen: set[str] = set()
    deduplicated = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduplicated.append(r)
    return deduplicated


# ---------------------------------------------------------------------------
# search_docs
# ---------------------------------------------------------------------------


def _search_docs(
    query: str,
    facility: str,
    *,
    k: int = 10,
    site: str | None = None,
    physics_domain: str | None = None,
    min_score: float | None = None,
    score_dimension: str | None = None,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search documentation (wiki, documents, images) with enrichment.

    Performs vector search across wiki chunks, documents, and images,
    enriches with cross-links to signals, tree nodes, and IMAS paths.

    Args:
        query: Natural language search text.
        facility: Facility identifier.
        k: Number of results.
        site: Optional wiki site filter.
        physics_domain: Filter wiki pages by physics domain.
        min_score: Minimum score threshold for score_dimension.
        score_dimension: Score dimension to filter on (default: score_composite).
            Valid content dimensions: score_data_documentation,
            score_physics_content, score_code_documentation, score_data_access,
            score_calibration, score_imas_relevance, score_composite.
    """
    try:
        if gc is None:
            gc = GraphClient()
        if encoder is None:
            from imas_codex.embeddings.config import EncoderConfig

            encoder = Encoder(EncoderConfig())

        embedding = _embed(encoder, query)
    except EmbeddingBackendError as e:
        return f"{_EMBEDDING_UNAVAILABLE_MSG}\n\nDetail: {e}"
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error initializing search: {e}"

    try:
        # Step 1: Vector search on wiki chunks
        chunk_ids, scores = _vector_search_wiki_chunks(
            gc,
            embedding,
            facility,
            k,
            site=site,
            physics_domain=physics_domain,
            min_score=min_score,
            score_dimension=score_dimension,
        )

        # Step 1b: Text search for keyword matches (hybrid boost)
        text_chunks = _text_search_wiki_chunks(gc, query, facility, k)
        for r in text_chunks:
            cid = r["id"]
            text_score = round(r["score"], 3)
            if cid in scores:
                scores[cid] = round(scores[cid] * 0.7 + text_score * 0.3 + 0.1, 3)
            else:
                scores[cid] = text_score
                chunk_ids.append(cid)

        # Re-sort and limit to k
        chunk_ids = sorted(
            set(chunk_ids), key=lambda cid: scores.get(cid, 0), reverse=True
        )[:k]

        # Step 2: Vector search on documents/images
        document_results, document_scores = _vector_search_documents(
            gc, embedding, facility, k
        )
        scores.update(document_scores)

        if not chunk_ids and not document_results:
            return (
                f"No documentation found for '{query}' at {facility}. "
                "Try search_signals() or search_code() instead."
            )

        # Step 3: Enrich chunks with cross-links
        enriched_chunks = []
        if chunk_ids:
            enriched_chunks = _enrich_wiki_chunks(gc, chunk_ids)

            # Title-match boost: if query terms appear in page title, boost score
            query_terms = set(query.lower().split())
            for chunk in enriched_chunks:
                cid = chunk.get("id", "")
                page_title = (
                    chunk.get("page_title") or chunk.get("page_id") or ""
                ).lower()
                title_terms = set(page_title.replace("_", " ").split())
                overlap = len(query_terms & title_terms)
                if overlap > 0 and cid in scores:
                    scores[cid] = min(1.0, scores[cid] + 0.1 * overlap)

        # Step 4: Format
        return format_docs_report(enriched_chunks, document_results, scores)

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("search_docs failed")
        return f"Search error: {_neo4j_error_message(e)}"


def _vector_search_wiki_chunks(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
    site: str | None = None,
    physics_domain: str | None = None,
    min_score: float | None = None,
    score_dimension: str | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Vector search on wiki_chunk_embedding index.

    Uses property-based facility filter with over-fetching.
    Optionally filters by wiki site URL substring, physics domain,
    and score dimension thresholds.
    """
    internal_k = max(k * 5, 200)
    # Build dynamic WHERE clauses for WikiPage join
    page_filters: list[str] = []
    params: dict[str, Any] = {
        "k": internal_k,
        "embedding": embedding,
        "facility": facility,
        "limit": k,
    }
    if site:
        page_filters.append("p.wiki_site CONTAINS $site")
        params["site"] = site
    if physics_domain:
        page_filters.append("p.physics_domain = $physics_domain")
        params["physics_domain"] = physics_domain
    if min_score is not None:
        dim = score_dimension or "score_composite"
        # Sanitize dimension name to prevent injection
        dim = "".join(c for c in dim if c.isalnum() or c == "_")
        page_filters.append(f"p.{dim} >= $min_score")
        params["min_score"] = min_score

    if page_filters:
        page_where = " AND ".join(page_filters)
        cypher = (
            'CALL db.index.vector.queryNodes("wiki_chunk_embedding", $k, $embedding) '
            "YIELD node AS c, score "
            "WHERE c.facility_id = $facility "
            "WITH c, score "
            "MATCH (p:WikiPage)-[:HAS_CHUNK]->(c) "
            f"WHERE {page_where} "
            "RETURN c.id AS id, score "
            "ORDER BY score DESC LIMIT $limit"
        )
    else:
        cypher = (
            'CALL db.index.vector.queryNodes("wiki_chunk_embedding", $k, $embedding) '
            "YIELD node AS c, score "
            "WHERE c.facility_id = $facility "
            "RETURN c.id AS id, score "
            "ORDER BY score DESC LIMIT $limit"
        )

    results = gc.query(cypher, **params)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _enrich_wiki_chunks(
    gc: GraphClient,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich wiki chunk IDs with page context and cross-links.

    Uses relationship-based traversals for cross-references, with
    property-based fallback for metadata that hasn't been linked yet.
    """
    cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (c:WikiChunk {id: cid})
        OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(c)
        OPTIONAL MATCH (c)-[:DOCUMENTS]->(sig:FacilitySignal)
        OPTIONAL MATCH (c)-[:DOCUMENTS]->(tn:SignalNode)
        OPTIONAL MATCH (c)-[:MENTIONS_IMAS]->(ip:IMASNode)
        WITH c, p,
             collect(DISTINCT sig.id) AS rel_signals,
             collect(DISTINCT tn.path) AS rel_data_nodes,
             collect(DISTINCT ip.id) AS rel_imas
        RETURN c.id AS id, c.text AS text, c.section AS section,
               p.id AS page_id, p.title AS page_title, p.url AS page_url,
               CASE WHEN size(rel_signals) > 0 THEN rel_signals
                    ELSE coalesce(c.ppf_paths_mentioned, []) END AS linked_signals,
               CASE WHEN size(rel_data_nodes) > 0 THEN rel_data_nodes
                    ELSE coalesce(c.mdsplus_paths_mentioned, []) END AS linked_data_nodes,
               CASE WHEN size(rel_imas) > 0 THEN rel_imas
                    ELSE coalesce(c.imas_paths_mentioned, []) END AS imas_refs,
               c.tool_mentions AS tool_mentions
    """
    return gc.query(cypher, chunk_ids=chunk_ids)


def _text_search_wiki_chunks(
    gc: GraphClient,
    query: str,
    facility: str,
    k: int,
) -> list[dict[str, Any]]:
    """Text-based search on wiki chunks by content.

    Uses fulltext index for BM25 scoring when available, falls back to
    CONTAINS with hardcoded scores otherwise.
    """
    # Try fulltext index first (BM25 scoring)
    try:
        cypher = """
            CALL db.index.fulltext.queryNodes('wiki_chunk_text', $query)
            YIELD node AS c, score
            WHERE c.facility_id = $facility
            RETURN c.id AS id, score
            LIMIT $limit
        """
        results = gc.query(cypher, query=query, facility=facility, limit=k * 2)
        if results:
            return results
    except Exception:
        pass

    # Fallback: CONTAINS with fixed score
    query_lower = query.lower()
    cypher = """
        MATCH (c:WikiChunk)
        WHERE c.facility_id = $facility
          AND toLower(c.text) CONTAINS $query_lower
        RETURN c.id AS id, 0.5 AS score
        LIMIT $limit
    """
    return gc.query(cypher, facility=facility, query_lower=query_lower, limit=k * 2)


def _vector_search_documents(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Vector search on document_desc_embedding and image_desc_embedding.

    Uses property-based facility filter with over-fetching.
    """
    results: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    internal_k = max(k * 5, 200)

    # Search documents
    try:
        document_cypher = (
            'CALL db.index.vector.queryNodes("document_desc_embedding", $k, $embedding) '
            "YIELD node AS a, score "
            "WHERE a.facility_id = $facility "
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_DOCUMENT]->(a) "
            "RETURN a.id AS id, a.title AS title, a.description AS description, "
            "a.url AS url, p.title AS page_title, score "
            "ORDER BY score DESC LIMIT $limit"
        )
        arts = gc.query(
            document_cypher,
            k=internal_k,
            embedding=embedding,
            facility=facility,
            limit=k,
        )
        for a in arts:
            scores[a["id"]] = round(a["score"], 3)
            results.append(a)
    except Exception:
        logger.debug("wiki_document_desc_embedding index not available", exc_info=True)

    # Search images
    try:
        image_cypher = (
            'CALL db.index.vector.queryNodes("image_desc_embedding", $k, $embedding) '
            "YIELD node AS img, score "
            "WHERE img.facility_id = $facility "
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_IMAGE]->(img) "
            "RETURN img.id AS id, img.title AS title, img.description AS description, "
            "p.title AS page_title, score "
            "ORDER BY score DESC LIMIT $limit"
        )
        imgs = gc.query(
            image_cypher,
            k=internal_k,
            embedding=embedding,
            facility=facility,
            limit=k,
        )
        for img in imgs:
            scores[img["id"]] = round(img["score"], 3)
            results.append(img)
    except Exception:
        logger.debug("image_desc_embedding index not available", exc_info=True)

    return results, scores


# ---------------------------------------------------------------------------
# fetch — retrieve full content for any graph resource by ID or URL
# ---------------------------------------------------------------------------


def _fetch(
    resource: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """Fetch full content for a graph resource identified by ID or URL.

    Use after search_* tools identify resources of interest. Resolves the
    resource to its node type and returns all available content.

    Supported node types (resolved in order):
    - WikiPage: all chunks in reading order
    - Document: all parsed document chunks
    - CodeFile: all code chunks with function names
    - Image: description, OCR text, and source URL

    The resource parameter can be:
    - A graph node ID (e.g., "jet:Fishbone_proposal_2018.ppt")
    - A URL (e.g., "https://wiki.jetdata.eu/tf/...")
    - A partial title/filename for fuzzy matching
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    try:
        # Try each node type in order until we find a match
        for resolver in [
            _fetch_wiki_page,
            _fetch_wiki_document,
            _fetch_code_file,
            _fetch_image,
            _fetch_imas_path,
        ]:
            result = resolver(gc, resource)
            if result is not None:
                return result

        return (
            f"No resource found matching '{resource}'. "
            "Use search_docs(), search_code(), or search_signals() to discover resources first."
        )

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("fetch failed")
        return f"Error: {_neo4j_error_message(e)}"


def _fetch_wiki_page(gc: GraphClient, resource: str) -> str | None:
    """Resolve and fetch a WikiPage by ID, URL, or title substring."""
    chunks = gc.query(
        "MATCH (p:WikiPage)-[:HAS_CHUNK]->(c:WikiChunk) "
        "WHERE p.id = $resource OR p.url = $resource "
        "   OR toLower(p.title) CONTAINS toLower($resource) "
        "RETURN 'wiki_page' AS source_type, "
        "p.title AS title, p.url AS url, p.id AS source_id, "
        "c.section AS section, c.text AS text, "
        "c.chunk_index AS chunk_index, "
        "c.mdsplus_paths_mentioned AS mdsplus_paths, "
        "c.imas_paths_mentioned AS imas_paths "
        "ORDER BY p.title, c.chunk_index",
        resource=resource,
    )
    if not chunks:
        return None
    return format_fetch_report(chunks)


def _fetch_wiki_document(gc: GraphClient, resource: str) -> str | None:
    """Resolve and fetch a Document by ID, URL, or filename."""
    chunks = gc.query(
        "MATCH (a:Document)-[:HAS_CHUNK]->(c:WikiChunk) "
        "WHERE a.id = $resource OR a.url = $resource "
        "   OR toLower(a.filename) CONTAINS toLower($resource) "
        "   OR toLower(a.title) CONTAINS toLower($resource) "
        "RETURN 'document' AS source_type, "
        "a.title AS title, a.url AS url, a.id AS source_id, "
        "c.section AS section, c.text AS text, "
        "c.chunk_index AS chunk_index, "
        "c.mdsplus_paths_mentioned AS mdsplus_paths, "
        "c.imas_paths_mentioned AS imas_paths "
        "ORDER BY a.title, c.chunk_index",
        resource=resource,
    )
    if chunks:
        return format_fetch_report(chunks)

    # No chunks — try metadata-only for unparsed documents (PowerPoint, etc.)
    meta = gc.query(
        "MATCH (a:Document) "
        "WHERE a.id = $resource OR a.url = $resource "
        "   OR toLower(a.filename) CONTAINS toLower($resource) "
        "   OR toLower(a.title) CONTAINS toLower($resource) "
        "RETURN a.id AS id, a.title AS title, a.url AS url, "
        "a.filename AS filename, a.file_type AS file_type, "
        "a.description AS description "
        "LIMIT 1",
        resource=resource,
    )
    if not meta:
        return None

    doc = meta[0]
    parts = [f"## Document: {doc.get('title') or doc.get('filename') or 'Untitled'}"]
    if doc.get("url"):
        parts.append(f"URL: {doc['url']}")
    if doc.get("file_type"):
        parts.append(f"Type: {doc['file_type']}")
    if doc.get("description"):
        parts.append(f"\n{doc['description']}")
    parts.append("\n*Content not parsed — binary or unsupported format.*")
    return "\n".join(parts)


def _fetch_code_file(gc: GraphClient, resource: str) -> str | None:
    """Resolve and fetch a code file by ID or path.

    Tries multiple traversal strategies:
    1. CodeExample matched by source_file → HAS_CHUNK → CodeChunk
    2. CodeFile → HAS_EXAMPLE → CodeExample → HAS_CHUNK → CodeChunk
    3. FacilityPath → HAS_EXAMPLE → CodeExample → HAS_CHUNK → CodeChunk
    """
    # Strategy 1: Match CodeExample by source_file path
    chunks = gc.query(
        "MATCH (ce:CodeExample) "
        "WHERE ce.source_file = $resource OR ce.id = $resource "
        "MATCH (ce)-[:HAS_CHUNK]->(cc:CodeChunk) "
        "RETURN 'code' AS source_type, "
        "ce.source_file AS title, ce.id AS source_id, "
        "ce.source_file AS url, "
        "cc.function_name AS section, cc.text AS text, "
        "cc.start_line AS chunk_index, "
        "null AS mdsplus_paths, null AS imas_paths "
        "ORDER BY cc.start_line",
        resource=resource,
    )
    if chunks:
        return format_fetch_report(chunks)

    # Strategy 2: Match via CodeFile → HAS_EXAMPLE → CodeExample
    chunks = gc.query(
        "MATCH (cf:CodeFile)-[:HAS_EXAMPLE]->(ce:CodeExample) "
        "WHERE cf.id = $resource OR cf.path = $resource "
        "MATCH (ce)-[:HAS_CHUNK]->(cc:CodeChunk) "
        "RETURN 'code' AS source_type, "
        "cf.path AS title, cf.id AS source_id, "
        "cf.path AS url, "
        "cc.function_name AS section, cc.text AS text, "
        "cc.start_line AS chunk_index, "
        "null AS mdsplus_paths, null AS imas_paths "
        "ORDER BY cc.start_line",
        resource=resource,
    )
    if chunks:
        return format_fetch_report(chunks)

    # Strategy 3: Match via FacilityPath → HAS_EXAMPLE → CodeExample
    chunks = gc.query(
        "MATCH (fp:FacilityPath)-[:HAS_EXAMPLE]->(ce:CodeExample) "
        "WHERE fp.id = $resource OR fp.path = $resource "
        "MATCH (ce)-[:HAS_CHUNK]->(cc:CodeChunk) "
        "RETURN 'code' AS source_type, "
        "fp.path AS title, fp.id AS source_id, "
        "fp.path AS url, "
        "cc.function_name AS section, cc.text AS text, "
        "cc.start_line AS chunk_index, "
        "null AS mdsplus_paths, null AS imas_paths "
        "ORDER BY cc.start_line",
        resource=resource,
    )
    if not chunks:
        return None
    return format_fetch_report(chunks)


def _fetch_image(gc: GraphClient, resource: str) -> str | None:
    """Resolve and fetch an Image by ID, URL, or description substring."""
    results = gc.query(
        "MATCH (img:Image) "
        "WHERE img.id = $resource OR img.url = $resource "
        "OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img) "
        "WHERE p:WikiPage OR p:Document "
        "RETURN 'image' AS source_type, "
        "coalesce(img.description, img.alt_text, img.filename, 'Untitled') AS title, "
        "img.url AS url, img.id AS source_id, "
        "img.description AS description, "
        "img.ocr_text AS ocr_text, "
        "img.mermaid_diagram AS mermaid, "
        "img.keywords AS keywords, "
        "img.width AS width, img.height AS height, "
        "collect(DISTINCT p.title) AS parent_pages",
        resource=resource,
    )
    if not results:
        return None

    img = results[0]
    parts: list[str] = []
    title = img.get("title") or "Untitled"
    url = img.get("url") or ""
    parts.append(f"## Image: {title}")
    if url:
        parts.append(f"Source: {url}")
    w, h = img.get("width"), img.get("height")
    if w and h:
        parts.append(f"Dimensions: {w}×{h}")
    parents = img.get("parent_pages") or []
    if parents:
        parts.append(f"Found in: {', '.join(parents)}")
    keywords = img.get("keywords") or []
    if keywords:
        kw = keywords if isinstance(keywords, list) else [keywords]
        parts.append(f"Keywords: {', '.join(kw)}")
    desc = img.get("description") or ""
    if desc:
        parts.append(f"\n### Description\n{desc}")
    ocr = img.get("ocr_text") or ""
    if ocr:
        parts.append(f"\n### OCR Text\n{ocr}")
    mermaid = img.get("mermaid") or ""
    if mermaid:
        parts.append(f"\n### Diagram\n```mermaid\n{mermaid}\n```")
    if not desc and not ocr:
        parts.append(f"\nNo extracted content available. Fetch from source: {url}")
    return "\n".join(parts)


def _fetch_imas_path(gc: GraphClient, resource: str) -> str | None:
    """Resolve and fetch an IMASNode by ID or partial path."""
    results = gc.query(
        """
        MATCH (p:IMASNode)
        WHERE p.id = $resource
           OR toLower(p.id) CONTAINS toLower($resource)
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
        OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
        OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
        OPTIONAL MATCH (p)-[:DEPRECATED_IN]->(dep:DDVersion)
        RETURN p.id AS id, p.name AS name, p.ids AS ids,
               p.documentation AS documentation, p.data_type AS data_type,
               p.ndim AS ndim, p.node_type AS node_type,
               p.physics_domain AS physics_domain,
               p.cocos_label_transformation AS cocos_label,
               p.lifecycle_status AS lifecycle_status,
               p.lifecycle_version AS lifecycle_version,
               p.path_doc AS structure_reference,
               u.symbol AS unit, u.name AS unit_name,
               collect(DISTINCT cl.label) AS clusters,
               collect(DISTINCT {id: coord.id, type: coord.coordinate_type}) AS coordinates,
               intro.id AS introduced_in,
               dep.id AS deprecated_in
        ORDER BY size(p.id) ASC
        LIMIT 1
        """,
        resource=resource,
    )
    if not results:
        return None

    p = results[0]
    parts: list[str] = []
    parts.append(f"## IMAS Path: `{p['id']}`")
    parts.append(f"**IDS**: {p.get('ids') or 'N/A'}")
    parts.append(f"**Name**: {p.get('name') or 'N/A'}")
    if p.get("data_type"):
        dtype = p["data_type"]
        ndim = p.get("ndim")
        parts.append(f"**Data type**: {dtype}" + (f" (ndim={ndim})" if ndim else ""))
    if p.get("unit"):
        unit_str = p["unit"]
        if p.get("unit_name"):
            unit_str += f" ({p['unit_name']})"
        parts.append(f"**Unit**: {unit_str}")
    if p.get("physics_domain"):
        parts.append(f"**Physics domain**: {p['physics_domain']}")
    if p.get("node_type"):
        parts.append(f"**Node type**: {p['node_type']}")

    doc = p.get("documentation") or ""
    if doc:
        parts.append(f"\n### Documentation\n{doc}")
    if p.get("structure_reference"):
        parts.append(f"\n**Structure reference**: {p['structure_reference']}")

    if p.get("cocos_label"):
        parts.append(f"**COCOS transformation**: {p['cocos_label']}")
    if p.get("lifecycle_status"):
        lc = p["lifecycle_status"]
        if p.get("lifecycle_version"):
            lc += f" (version {p['lifecycle_version']})"
        parts.append(f"**Lifecycle**: {lc}")
    if p.get("introduced_in"):
        parts.append(f"**Introduced in**: DD {p['introduced_in']}")
    if p.get("deprecated_in"):
        parts.append(f"**Deprecated in**: DD {p['deprecated_in']}")

    clusters = [c for c in (p.get("clusters") or []) if c]
    if clusters:
        parts.append("\n### Semantic Clusters\n" + ", ".join(clusters))

    coords = [c for c in (p.get("coordinates") or []) if c and c.get("id")]
    if coords:
        coord_strs = [f"- `{c['id']}` ({c.get('type', 'N/A')})" for c in coords]
        parts.append("\n### Coordinates\n" + "\n".join(coord_strs))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


def _search_code(
    query: str,
    *,
    facility: str | None = None,
    k: int = 10,
    physics_domain: str | None = None,
    min_score: float | None = None,
    score_dimension: str | None = None,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search ingested code with data reference enrichment.

    Performs hybrid search (vector + text) on code chunks, enriches
    with data references (MDSplus, TDI, IMAS) and directory context.

    Args:
        query: Natural language search text.
        facility: Optional facility filter.
        k: Number of results.
        physics_domain: Filter code by FacilityPath physics domain.
        min_score: Minimum score threshold for score_dimension.
        score_dimension: Score dimension to filter on (default: score_composite).
            Valid code dimensions: score_modeling_code, score_analysis_code,
            score_operations_code, score_data_access, score_workflow,
            score_visualization, score_documentation, score_imas,
            score_convention, score_composite.
    """
    try:
        if gc is None:
            gc = GraphClient()
        if encoder is None:
            from imas_codex.embeddings.config import EncoderConfig

            encoder = Encoder(EncoderConfig())

        embedding = _embed(encoder, query)
    except EmbeddingBackendError as e:
        return f"{_EMBEDDING_UNAVAILABLE_MSG}\n\nDetail: {e}"
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error initializing search: {e}"

    try:
        # Step 1: Vector search on code chunks
        chunk_ids, scores = _vector_search_code_chunks(
            gc,
            embedding,
            facility,
            k,
            physics_domain=physics_domain,
            min_score=min_score,
            score_dimension=score_dimension,
        )

        # Step 1b: Text search for keyword matches (hybrid boost)
        text_chunks = _text_search_code_chunks(gc, query, facility, k)
        for r in text_chunks:
            cid = r["id"]
            text_score = round(r["score"], 3)
            if cid in scores:
                scores[cid] = round(scores[cid] * 0.7 + text_score * 0.3 + 0.1, 3)
            else:
                scores[cid] = text_score
                chunk_ids.append(cid)

        # Step 1c: CodeExample-level vector search (find relevant examples by description)
        example_chunks = _vector_search_code_examples(gc, embedding, facility, k)
        for r in example_chunks:
            cid = r["id"]
            ex_score = round(r["score"], 3)
            if cid in scores:
                scores[cid] = round(max(scores[cid], ex_score) + 0.05, 3)
            else:
                scores[cid] = ex_score
                chunk_ids.append(cid)

        # Re-sort and limit to k
        chunk_ids = sorted(
            set(chunk_ids), key=lambda cid: scores.get(cid, 0), reverse=True
        )[:k]

        if not chunk_ids:
            facility_msg = f" at {facility}" if facility else ""
            return (
                f"No code examples found for '{query}'{facility_msg}. "
                "Try search_docs() or search_signals() instead."
            )

        # Step 2: Enrich with data references and directory context
        enriched = _enrich_code_chunks(gc, chunk_ids)

        # Step 3: Format
        return format_code_report(enriched, scores)

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("search_code failed")
        return f"Search error: {_neo4j_error_message(e)}"


def _vector_search_code_chunks(
    gc: GraphClient,
    embedding: list[float],
    facility: str | None,
    k: int,
    physics_domain: str | None = None,
    min_score: float | None = None,
    score_dimension: str | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Vector search on code_chunk_embedding index.

    Facility filtering uses CodeChunk's ``facility_id`` property directly.
    Optionally filters by FacilityPath physics domain and score dimensions.
    """
    # Over-fetch to avoid facility starvation when one facility dominates
    internal_k = max(k * 5, 200)
    params: dict[str, Any] = {"k": internal_k, "embedding": embedding, "limit": k}

    where_parts: list[str] = []
    if facility is not None:
        where_parts.append("cc.facility_id = $facility")
        params["facility"] = facility

    where_clause = " AND ".join(where_parts) if where_parts else "true"

    # Build optional path-level filter join for physics_domain/score
    path_join = ""
    if physics_domain or min_score is not None:
        path_parts: list[str] = []
        if physics_domain:
            path_parts.append("fp.physics_domain = $physics_domain")
            params["physics_domain"] = physics_domain
        if min_score is not None:
            dim = score_dimension or "score_composite"
            dim = "".join(c for c in dim if c.isalnum() or c == "_")
            path_parts.append(f"fp.{dim} >= $min_score")
            params["min_score"] = min_score
        path_where = " AND ".join(path_parts)
        path_join = (
            "WITH cc, score "
            "MATCH (cf:CodeFile {path: cc.source_file})-[:IN_DIRECTORY]->(fp:FacilityPath) "
            f"WHERE {path_where} "
        )

    cypher = (
        'CALL db.index.vector.queryNodes("code_chunk_embedding", $k, $embedding) '
        f"YIELD node AS cc, score WHERE {where_clause} "
        f"{path_join}"
        "RETURN cc.id AS id, score "
        "ORDER BY score DESC LIMIT $limit"
    )

    results = gc.query(cypher, **params)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _vector_search_code_examples(
    gc: GraphClient,
    embedding: list[float],
    facility: str | None,
    k: int,
) -> list[dict[str, Any]]:
    """Vector search on CodeExample descriptions, returning their chunk IDs.

    Searches the code_example_desc_embedding index and traverses to child
    CodeChunks, returning chunk IDs with scores inherited from the parent.
    Gracefully returns empty on missing index.
    """
    try:
        params: dict[str, Any] = {"k": max(k, 20), "embedding": embedding}
        facility_filter = ""
        if facility is not None:
            facility_filter = "AND ce.facility_id = $facility"
            params["facility"] = facility

        cypher = f"""
            CALL db.index.vector.queryNodes('code_example_desc_embedding', $k, $embedding)
            YIELD node AS ce, score
            WHERE true {facility_filter}
            MATCH (ce)-[:HAS_CHUNK]->(cc:CodeChunk)
            RETURN cc.id AS id, score
            ORDER BY score DESC
            LIMIT $limit
        """
        params["limit"] = k * 2
        return gc.query(cypher, **params)
    except Exception:
        return []


def _enrich_code_chunks(
    gc: GraphClient,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich code chunks with data references and directory context.

    Uses traversals that work with both current and migrated graph states:
    ``CodeExample -[:HAS_CHUNK]-> CodeChunk`` (inverse of schema CODE_EXAMPLE_ID)
    ``CodeChunk -[:CONTAINS_REF]-> DataReference -[:RESOLVES_TO_NODE]-> SignalNode``
    ``DataReference -[:RESOLVES_TO_IMAS_PATH]-> IMASNode``
    ``CodeFile -[:IN_DIRECTORY]-> FacilityPath``
    """
    cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (cc:CodeChunk {id: cid})
        OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)
        OPTIONAL MATCH (cf:CodeFile {path: cc.source_file})
            WHERE cf.facility_id = cc.facility_id
        OPTIONAL MATCH (cc)-[:CONTAINS_REF]->(dr:DataReference)
        OPTIONAL MATCH (dr)-[:RESOLVES_TO_NODE]->(tn)
        OPTIONAL MATCH (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip:IMASNode)
        OPTIONAL MATCH (cf)-[:IN_DIRECTORY]->(fp:FacilityPath)
        RETURN cc.id AS id, cc.text AS text,
               cc.function_name AS function_name,
               cc.language AS language,
               coalesce(ce.source_file, cc.source_file) AS source_file,
               cf.id AS source_file_id,
               coalesce(ce.facility_id, cc.facility_id) AS facility_id,
               collect(DISTINCT {type: dr.ref_type, raw: dr.raw_string,
                       tree: tn.path, imas_path: ip.id}) AS data_refs,
               fp.path AS directory, fp.description AS dir_description
    """
    return gc.query(cypher, chunk_ids=chunk_ids)


def _text_search_code_chunks(
    gc: GraphClient,
    query: str,
    facility: str | None,
    k: int,
) -> list[dict[str, Any]]:
    """Text-based search on code chunks by content and function name.

    Uses fulltext index for BM25 scoring when available, falls back to
    CONTAINS with hardcoded scores otherwise.
    """
    # Try fulltext index first (BM25 scoring)
    try:
        if facility is not None:
            cypher = """
                CALL db.index.fulltext.queryNodes('code_chunk_text', $query)
                YIELD node AS cc, score
                WHERE cc.facility_id = $facility
                RETURN cc.id AS id, score
                LIMIT $limit
            """
            results = gc.query(cypher, query=query, facility=facility, limit=k * 2)
        else:
            cypher = """
                CALL db.index.fulltext.queryNodes('code_chunk_text', $query)
                YIELD node AS cc, score
                RETURN cc.id AS id, score
                LIMIT $limit
            """
            results = gc.query(cypher, query=query, limit=k * 2)
        if results:
            return results
    except Exception:
        pass

    # Fallback: CONTAINS with fixed score
    query_lower = query.lower()
    params: dict[str, Any] = {"query_lower": query_lower, "limit": k * 2}

    if facility is not None:
        cypher = """
            MATCH (cc:CodeChunk)
            WHERE cc.facility_id = $facility
              AND (
                toLower(cc.text) CONTAINS $query_lower
                OR toLower(cc.function_name) CONTAINS $query_lower
              )
            RETURN cc.id AS id, 0.5 AS score
            LIMIT $limit
        """
        params["facility"] = facility
    else:
        cypher = """
            MATCH (cc:CodeChunk)
            WHERE toLower(cc.text) CONTAINS $query_lower
               OR toLower(cc.function_name) CONTAINS $query_lower
            RETURN cc.id AS id, 0.5 AS score
            LIMIT $limit
        """

    return gc.query(cypher, **params)
