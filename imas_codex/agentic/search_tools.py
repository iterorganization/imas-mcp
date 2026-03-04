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

from imas_codex.agentic.search_formatters import (
    format_code_report,
    format_docs_report,
    format_imas_report,
    format_signals_report,
)
from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
from imas_codex.graph.client import GraphClient

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
    k: int = 10,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search facility signals with graph enrichment.

    Performs vector search on signal descriptions, enriches with
    DataAccess, Diagnostic, TreeNode, and IMASPath traversals,
    then formats the result.
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

        if not signal_ids:
            # Fall back to tree node search only
            tree_results = _vector_search_tree_nodes(gc, embedding, facility, k)
            return format_signals_report([], tree_results, {})

        # Step 2: Enrich with graph traversals
        enriched = _enrich_signals(gc, signal_ids)

        # Step 3: Tree node search (secondary index)
        tree_results = _vector_search_tree_nodes(gc, embedding, facility, k)

        # Step 4: Format
        return format_signals_report(enriched, tree_results, scores)

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("search_signals failed")
        return f"Search error: {_neo4j_error_message(e)}"


def _vector_search_signals(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
    diagnostic: str | None,
    physics_domain: str | None,
) -> tuple[list[str], dict[str, float]]:
    """Vector search on facility_signal_desc_embedding index."""
    where_parts = ["(s)-[:AT_FACILITY]->(:Facility {id: $facility})"]
    params: dict[str, Any] = {"k": k, "embedding": embedding, "facility": facility}

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
        "ORDER BY score DESC"
    )
    results = gc.query(cypher, **params)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _enrich_signals(
    gc: GraphClient,
    signal_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich signal IDs with full context via graph traversal."""
    cypher = """
        UNWIND $signal_ids AS sid
        MATCH (s:FacilitySignal {id: sid})
        OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
        OPTIONAL MATCH (s)-[:BELONGS_TO_DIAGNOSTIC]->(diag:Diagnostic)
        OPTIONAL MATCH (s)-[:SOURCE_NODE]->(tn:TreeNode)
        OPTIONAL MATCH (da)-[:MAPS_TO_IMAS]->(ip:IMASPath)
        OPTIONAL MATCH (ip)-[:HAS_UNIT]->(u:Unit)
        RETURN s.id AS id, s.name AS name, s.description AS description,
               s.physics_domain AS physics_domain, s.unit AS unit,
               s.checked AS checked, s.example_shot AS example_shot,
               diag.name AS diagnostic_name, diag.category AS diagnostic_category,
               da.data_template AS access_template, da.access_type AS access_type,
               da.imports_template AS imports_template,
               da.connection_template AS connection_template,
               tn.path AS tree_path, tn.tree_name AS tree_name,
               ip.id AS imas_path, ip.documentation AS imas_docs,
               u.symbol AS imas_unit
    """
    return gc.query(cypher, signal_ids=signal_ids)


def _vector_search_tree_nodes(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
) -> list[dict[str, Any]]:
    """Vector search on tree_node_desc_embedding index."""
    cypher = (
        'CALL db.index.vector.queryNodes("tree_node_desc_embedding", $k, $embedding) '
        "YIELD node AS n, score "
        "WHERE (n)-[:AT_FACILITY]->(:Facility {id: $facility}) "
        "RETURN n.id AS id, n.path AS path, n.tree_name AS tree_name, "
        "n.description AS description, n.unit AS unit "
        "ORDER BY score DESC"
    )
    return gc.query(cypher, k=k, embedding=embedding, facility=facility)


# ---------------------------------------------------------------------------
# search_docs
# ---------------------------------------------------------------------------


def _search_docs(
    query: str,
    facility: str,
    *,
    k: int = 10,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search documentation (wiki, artifacts, images) with enrichment.

    Performs vector search across wiki chunks, artifacts, and images,
    enriches with cross-links to signals, tree nodes, and IMAS paths.
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
        chunk_ids, scores = _vector_search_wiki_chunks(gc, embedding, facility, k)

        # Step 2: Vector search on artifacts/images
        artifact_results, artifact_scores = _vector_search_artifacts(
            gc, embedding, facility, k
        )
        scores.update(artifact_scores)

        if not chunk_ids and not artifact_results:
            return (
                f"No documentation found for '{query}' at {facility}. "
                "Try search_signals() or search_code() instead."
            )

        # Step 3: Enrich chunks with cross-links
        enriched_chunks = []
        if chunk_ids:
            enriched_chunks = _enrich_wiki_chunks(gc, chunk_ids)

        # Step 4: Format
        return format_docs_report(enriched_chunks, artifact_results, scores)

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
) -> tuple[list[str], dict[str, float]]:
    """Vector search on wiki_chunk_embedding index."""
    cypher = (
        'CALL db.index.vector.queryNodes("wiki_chunk_embedding", $k, $embedding) '
        "YIELD node AS c, score "
        "WHERE (c)-[:AT_FACILITY]->(:Facility {id: $facility}) "
        "RETURN c.id AS id, score "
        "ORDER BY score DESC"
    )
    results = gc.query(cypher, k=k, embedding=embedding, facility=facility)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _enrich_wiki_chunks(
    gc: GraphClient,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich wiki chunk IDs with page context and cross-links."""
    cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (c:WikiChunk {id: cid})
        OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(c)
        OPTIONAL MATCH (c)-[:DOCUMENTS]->(sig:FacilitySignal)
        OPTIONAL MATCH (c)-[:DOCUMENTS]->(tn:TreeNode)
        OPTIONAL MATCH (c)-[:MENTIONS_IMAS]->(ip:IMASPath)
        RETURN c.id AS id, c.text AS text, c.section AS section,
               p.title AS page_title, p.url AS page_url,
               collect(DISTINCT sig.id) AS linked_signals,
               collect(DISTINCT tn.path) AS linked_tree_nodes,
               collect(DISTINCT ip.id) AS imas_refs
    """
    return gc.query(cypher, chunk_ids=chunk_ids)


def _vector_search_artifacts(
    gc: GraphClient,
    embedding: list[float],
    facility: str,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Vector search on wiki_artifact_desc_embedding and image_desc_embedding."""
    results: list[dict[str, Any]] = []
    scores: dict[str, float] = {}

    # Search artifacts
    try:
        artifact_cypher = (
            'CALL db.index.vector.queryNodes("wiki_artifact_desc_embedding", $k, $embedding) '
            "YIELD node AS a, score "
            "WHERE (a)-[:AT_FACILITY]->(:Facility {id: $facility}) "
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_ARTIFACT]->(a) "
            "RETURN a.id AS id, a.title AS title, a.description AS description, "
            "a.url AS url, p.title AS page_title, score "
            "ORDER BY score DESC"
        )
        arts = gc.query(artifact_cypher, k=k, embedding=embedding, facility=facility)
        for a in arts:
            scores[a["id"]] = round(a["score"], 3)
            results.append(a)
    except Exception:
        logger.debug("wiki_artifact_desc_embedding index not available", exc_info=True)

    # Search images
    try:
        image_cypher = (
            'CALL db.index.vector.queryNodes("image_desc_embedding", $k, $embedding) '
            "YIELD node AS img, score "
            "WHERE (img)-[:AT_FACILITY]->(:Facility {id: $facility}) "
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_IMAGE]->(img) "
            "RETURN img.id AS id, img.title AS title, img.description AS description, "
            "p.title AS page_title, score "
            "ORDER BY score DESC"
        )
        imgs = gc.query(image_cypher, k=k, embedding=embedding, facility=facility)
        for img in imgs:
            scores[img["id"]] = round(img["score"], 3)
            results.append(img)
    except Exception:
        logger.debug("image_desc_embedding index not available", exc_info=True)

    return results, scores


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


def _search_code(
    query: str,
    *,
    facility: str | None = None,
    k: int = 5,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search ingested code with data reference enrichment.

    Performs vector search on code chunks, enriches with data references
    (MDSplus, TDI, IMAS) and directory context.
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
        chunk_ids, scores = _vector_search_code_chunks(gc, embedding, facility, k)

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
) -> tuple[list[str], dict[str, float]]:
    """Vector search on code_chunk_embedding index."""
    facility_filter = ""
    params: dict[str, Any] = {"k": k, "embedding": embedding}

    if facility is not None:
        facility_filter = (
            "MATCH (cf:CodeFile)-[:HAS_CHUNK]->(cc) "
            "WHERE cf.facility_id = $facility "
            "WITH cc, score "
        )
        params["facility"] = facility

    cypher = (
        'CALL db.index.vector.queryNodes("code_chunk_embedding", $k, $embedding) '
        f"YIELD node AS cc, score "
        f"{facility_filter}"
        "RETURN cc.id AS id, score "
        "ORDER BY score DESC"
    )
    results = gc.query(cypher, **params)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _enrich_code_chunks(
    gc: GraphClient,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich code chunks with data references and directory context."""
    cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (cc:CodeChunk {id: cid})
        OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)
        OPTIONAL MATCH (cf:CodeFile {id: ce.source_file})
        OPTIONAL MATCH (cf)-[:CONTAINS_REF]->(dr:DataReference)
        OPTIONAL MATCH (dr)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
        OPTIONAL MATCH (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip:IMASPath)
        OPTIONAL MATCH (dr)-[:CALLS_TDI_FUNCTION]->(tdi:TDIFunction)
        OPTIONAL MATCH (cf)-[:IN_DIRECTORY]->(fp:FacilityPath)
        RETURN cc.id AS id, cc.text AS text,
               cc.function_name AS function_name, ce.source_file AS source_file,
               cf.facility_id AS facility_id,
               collect(DISTINCT {type: dr.ref_type, raw: dr.raw_string,
                       tree: tn.path, imas: ip.id, tdi: tdi.id}) AS data_refs,
               fp.path AS directory, fp.description AS dir_description
    """
    return gc.query(cypher, chunk_ids=chunk_ids)


# ---------------------------------------------------------------------------
# search_imas
# ---------------------------------------------------------------------------


def _search_imas(
    query: str,
    *,
    ids_filter: str | None = None,
    facility: str | None = None,
    include_version_context: bool = False,
    k: int = 10,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Search IMAS Data Dictionary with cross-domain enrichment.

    Performs vector search across path and cluster embeddings,
    enriches with cluster membership, coordinate context, version
    history, and facility cross-references.
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
        # Step 1: Vector search on IMAS path embeddings
        path_ids, scores = _vector_search_imas_paths(gc, embedding, k, ids_filter)

        # Step 2: Vector search on cluster embeddings
        cluster_results, cluster_scores = _vector_search_clusters(gc, embedding, k)
        scores.update(cluster_scores)

        if not path_ids and not cluster_results:
            return (
                f"No IMAS paths found for '{query}'. "
                "Try search_docs() or search_signals() instead."
            )

        # Step 3: Enrich paths
        enriched_paths = []
        if path_ids:
            enriched_paths = _enrich_imas_paths(gc, path_ids)

        # Step 4: Facility cross-references (optional)
        facility_xrefs: dict[str, dict[str, Any]] = {}
        if facility and path_ids:
            facility_xrefs = _get_facility_crossrefs(gc, path_ids, facility)

        # Step 5: Version context (optional)
        version_context: dict[str, dict[str, Any]] = {}
        if include_version_context and path_ids:
            version_context = _get_version_context(gc, path_ids)

        # Step 6: Format
        return format_imas_report(
            enriched_paths, cluster_results, facility_xrefs, version_context, scores
        )

    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        logger.exception("search_imas failed")
        return f"Search error: {_neo4j_error_message(e)}"


def _vector_search_imas_paths(
    gc: GraphClient,
    embedding: list[float],
    k: int,
    ids_filter: str | None,
) -> tuple[list[str], dict[str, float]]:
    """Vector search on imas_path_embedding index."""
    where_parts = ["NOT (p)-[:DEPRECATED_IN]->(:DDVersion)"]
    params: dict[str, Any] = {"k": k, "embedding": embedding}

    if ids_filter is not None:
        where_parts.append("p.ids = $ids_filter")
        params["ids_filter"] = ids_filter

    where_clause = " AND ".join(where_parts)

    cypher = (
        'CALL db.index.vector.queryNodes("imas_path_embedding", $k, $embedding) '
        f"YIELD node AS p, score WHERE {where_clause} "
        "RETURN p.id AS id, score "
        "ORDER BY score DESC"
    )
    results = gc.query(cypher, **params)
    ids = [r["id"] for r in results]
    scores = {r["id"]: round(r["score"], 3) for r in results}
    return ids, scores


def _enrich_imas_paths(
    gc: GraphClient,
    path_ids: list[str],
) -> list[dict[str, Any]]:
    """Enrich IMAS paths with cluster, unit, coordinate context."""
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (p:IMASPath {id: pid})
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
        OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord)
        OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(intro:DDVersion)
        RETURN p.id AS id, p.name AS name, p.ids AS ids,
               p.documentation AS documentation, p.data_type AS data_type,
               p.physics_domain AS physics_domain,
               p.cocos_label_transformation AS cocos_label_transformation,
               u.symbol AS unit,
               collect(DISTINCT cl.label) AS clusters,
               collect(DISTINCT coord.id) AS coordinates,
               intro.id AS introduced_in
    """
    return gc.query(cypher, path_ids=path_ids)


def _vector_search_clusters(
    gc: GraphClient,
    embedding: list[float],
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Vector search on cluster_description_embedding index."""
    try:
        cypher = (
            'CALL db.index.vector.queryNodes("cluster_description_embedding", $k, $embedding) '
            "YIELD node AS cl, score "
            "RETURN cl.id AS id, cl.label AS label, cl.scope AS scope, "
            "cl.path_count AS path_count, cl.sample_paths AS sample_paths, score "
            "ORDER BY score DESC"
        )
        results = gc.query(cypher, k=k, embedding=embedding)
        scores = {r["id"]: round(r["score"], 3) for r in results}
        return results, scores
    except Exception:
        logger.debug("cluster_description_embedding index not available", exc_info=True)
        return [], {}


def _get_facility_crossrefs(
    gc: GraphClient,
    path_ids: list[str],
    facility: str,
) -> dict[str, dict[str, Any]]:
    """Get facility cross-references for IMAS paths."""
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (ip:IMASPath {id: pid})
        OPTIONAL MATCH (da:DataAccess)-[:MAPS_TO_IMAS]->(ip)
        OPTIONAL MATCH (sig:FacilitySignal)-[:DATA_ACCESS]->(da)
        WHERE sig.facility_id = $facility
        OPTIONAL MATCH (wc:WikiChunk)-[:MENTIONS_IMAS]->(ip)
        WHERE wc.facility_id = $facility
        OPTIONAL MATCH (dr:DataReference)-[:RESOLVES_TO_IMAS_PATH]->(ip)
        OPTIONAL MATCH (cf:CodeFile)-[:CONTAINS_REF]->(dr)
        WHERE cf.facility_id = $facility
        RETURN ip.id AS id,
               collect(DISTINCT sig.id) AS facility_signals,
               collect(DISTINCT wc.section) AS wiki_mentions,
               collect(DISTINCT cf.path) AS code_files
    """
    results = gc.query(cypher, path_ids=path_ids, facility=facility)
    return {r["id"]: r for r in results}


def _get_version_context(
    gc: GraphClient,
    path_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Get version change context for IMAS paths."""
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (p:IMASPath {id: pid})
        OPTIONAL MATCH (change:IMASPathChange)-[:FOR_IMAS_PATH]->(p)
        WHERE change.semantic_change_type IN
              ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
        RETURN p.id AS id,
               count(change) AS change_count,
               collect({version: change.version,
                        type: change.semantic_change_type,
                        summary: change.summary})[..5] AS notable_changes
    """
    results = gc.query(cypher, path_ids=path_ids)
    return {r["id"]: r for r in results}
