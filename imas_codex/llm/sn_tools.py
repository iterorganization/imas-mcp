"""MCP tools for standard name search, fetch, and listing.

Functions are prefixed with ``_`` — they are registered as MCP tools
in ``server.py`` via ``@self.mcp.tool()``.
"""

from __future__ import annotations

import logging

from neo4j.exceptions import ServiceUnavailable

from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Check service with: systemctl --user status imas-codex-neo4j"
)


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    msg = str(e)
    if "Connection refused" in msg or "ServiceUnavailable" in msg:
        return NEO4J_NOT_RUNNING_MSG
    return msg


# ---------------------------------------------------------------------------
# _search_standard_names
# ---------------------------------------------------------------------------


GRAMMAR_SEGMENTS: tuple[str, ...] = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "transformation",
    "position",
    "process",
    "geometry",
    "object",
    "geometric_base",
    "device",
    "region",
    "secondary_base",
    "binary_operator",
)


def _search_standard_names(
    query: str,
    *,
    kind: str | None = None,
    tags: list[str] | None = None,
    review_status: str | None = None,
    cocos_type: str | None = None,
    grammar_physical_base: str | None = None,
    grammar_subject: str | None = None,
    grammar_component: str | None = None,
    grammar_coordinate: str | None = None,
    grammar_transformation: str | None = None,
    grammar_position: str | None = None,
    grammar_process: str | None = None,
    grammar_object: str | None = None,
    grammar_geometry: str | None = None,
    grammar_geometric_base: str | None = None,
    grammar_device: str | None = None,
    k: int = 20,
    gc: GraphClient | None = None,
) -> str:
    """Search standard names by physics concept.

    Hybrid search (vector + keyword) over StandardName descriptions.
    Falls back to keyword-only if no embeddings present.

    ``grammar_*`` parameters are post-filters applied to the parsed
    grammar segment properties persisted on each StandardName. They
    compare by exact string match (case-insensitive).
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    # Try to get embedding for vector search
    has_embedding = False
    embedding: list[float] = []
    try:
        from imas_codex.embeddings.config import EncoderConfig

        encoder = Encoder(EncoderConfig())
        result = encoder.embed_texts([query])[0]
        embedding = result.tolist() if hasattr(result, "tolist") else list(result)
        has_embedding = True
    except (EmbeddingBackendError, Exception):
        pass

    try:
        if has_embedding:
            rows = _vector_search_sn(gc, embedding, k)
        else:
            rows = _keyword_search_sn(gc, query, k)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Search failed: {_neo4j_error_message(e)}"

    # Post-filter
    if kind:
        rows = [r for r in rows if (r.get("kind") or "").lower() == kind.lower()]
    if tags:
        rows = [r for r in rows if any(t in (r.get("tags") or []) for t in tags)]
    if review_status:
        rows = [
            r
            for r in rows
            if (r.get("review_status") or "").lower() == review_status.lower()
        ]
    if cocos_type:
        rows = [
            r for r in rows if (r.get("cocos_transformation_type") or "") == cocos_type
        ]

    grammar_filters = {
        "grammar_physical_base": grammar_physical_base,
        "grammar_subject": grammar_subject,
        "grammar_component": grammar_component,
        "grammar_coordinate": grammar_coordinate,
        "grammar_transformation": grammar_transformation,
        "grammar_position": grammar_position,
        "grammar_process": grammar_process,
        "grammar_object": grammar_object,
        "grammar_geometry": grammar_geometry,
        "grammar_geometric_base": grammar_geometric_base,
        "grammar_device": grammar_device,
    }
    for key, expected in grammar_filters.items():
        if expected is None:
            continue
        expected_lc = expected.lower()
        rows = [r for r in rows if (r.get(key) or "").lower() == expected_lc]

    return _format_search_report(query, rows)


def _vector_search_sn(gc: GraphClient, embedding: list[float], k: int) -> list[dict]:
    """Run vector search on StandardName nodes."""
    cypher = """
CALL db.index.vector.queryNodes('standard_name_desc_embedding', $k, $embedding)
YIELD node AS sn, score
WHERE sn.id IS NOT NULL
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name, sn.description AS description,
       sn.kind AS kind, coalesce(u.id, sn.unit) AS unit,
       sn.tags AS tags, sn.review_status AS review_status,
       sn.documentation AS documentation,
       sn.grammar_physical_base AS grammar_physical_base,
       sn.grammar_subject AS grammar_subject,
       sn.grammar_component AS grammar_component,
       sn.grammar_coordinate AS grammar_coordinate,
       sn.grammar_transformation AS grammar_transformation,
       sn.grammar_position AS grammar_position,
       sn.grammar_process AS grammar_process,
       sn.grammar_object AS grammar_object,
       sn.grammar_geometry AS grammar_geometry,
       sn.grammar_geometric_base AS grammar_geometric_base,
       sn.grammar_device AS grammar_device,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       score
ORDER BY score DESC
"""
    return gc.query(cypher, embedding=embedding, k=k)


def _keyword_search_sn(gc: GraphClient, query: str, k: int) -> list[dict]:
    """Run keyword search on StandardName nodes."""
    cypher = """
MATCH (sn:StandardName)
WHERE toLower(sn.id) CONTAINS toLower($keyword)
   OR toLower(sn.description) CONTAINS toLower($keyword)
   OR toLower(coalesce(sn.documentation, '')) CONTAINS toLower($keyword)
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name, sn.description AS description,
       sn.kind AS kind, coalesce(u.id, sn.unit) AS unit,
       sn.tags AS tags, sn.review_status AS review_status,
       sn.documentation AS documentation,
       sn.grammar_physical_base AS grammar_physical_base,
       sn.grammar_subject AS grammar_subject,
       sn.grammar_component AS grammar_component,
       sn.grammar_coordinate AS grammar_coordinate,
       sn.grammar_transformation AS grammar_transformation,
       sn.grammar_position AS grammar_position,
       sn.grammar_process AS grammar_process,
       sn.grammar_object AS grammar_object,
       sn.grammar_geometry AS grammar_geometry,
       sn.grammar_geometric_base AS grammar_geometric_base,
       sn.grammar_device AS grammar_device,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       1.0 AS score
LIMIT $k
"""
    return gc.query(cypher, keyword=query, k=k)


def _format_search_report(query: str, rows: list[dict]) -> str:
    """Format search results as a text report."""
    if not rows:
        return (
            f"## Standard Name Search Results\n\nNo standard names found matching "
            f'"{query}".'
        )

    lines = [
        f'## Standard Name Search Results\n\nFound {len(rows)} standard names matching "{query}"\n'
    ]
    for i, row in enumerate(rows, 1):
        name = row.get("name") or "unknown"
        score = row.get("score", 0.0)
        kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        tags = row.get("tags") or []
        review_status = row.get("review_status") or ""
        description = row.get("description") or ""
        documentation = row.get("documentation") or ""
        physical_base = row.get("grammar_physical_base") or ""
        subject = row.get("grammar_subject") or ""
        component = row.get("grammar_component") or ""
        coordinate = row.get("grammar_coordinate") or ""
        transformation = row.get("grammar_transformation") or ""
        position = row.get("grammar_position") or ""
        process = row.get("grammar_process") or ""
        cocos_transformation_type = row.get("cocos_transformation_type") or ""
        cocos = row.get("cocos")

        lines.append(f"### {i}. {name} (score: {score:.2f})")
        if kind:
            lines.append(f"- **Kind:** {kind}")
        if unit:
            lines.append(f"- **Unit:** {unit}")
        if tags:
            tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            lines.append(f"- **Tags:** {tag_str}")
        if review_status:
            lines.append(f"- **Status:** {review_status}")
        if cocos_transformation_type:
            lines.append(f"- **COCOS Transformation:** {cocos_transformation_type}")
        if cocos is not None:
            lines.append(f"- **COCOS:** {cocos}")
        if description:
            lines.append(f"- **Description:** {description}")
        if documentation:
            lines.append(
                f"- **Documentation:** {documentation[:200]}{'...' if len(documentation) > 200 else ''}"
            )
        grammar_parts = []
        for label, val in (
            ("physical_base", physical_base),
            ("subject", subject),
            ("component", component),
            ("coordinate", coordinate),
            ("transformation", transformation),
            ("position", position),
            ("process", process),
        ):
            if val:
                grammar_parts.append(f"{label}={val}")
        if grammar_parts:
            lines.append(f"- **Grammar:** {', '.join(grammar_parts)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _fetch_standard_names
# ---------------------------------------------------------------------------


def _fetch_standard_names(
    names: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """Fetch full entries for known standard names.

    Args:
        names: Space- or comma-separated standard name IDs.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    # Parse names (split on space or comma)
    import re

    name_list = [n.strip() for n in re.split(r"[,\s]+", names) if n.strip()]
    if not name_list:
        return "No names provided."

    cypher = """
UNWIND $names AS name_id
MATCH (sn:StandardName {id: name_id})
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
OPTIONAL MATCH (src)-[:IN_IDS]->(ids:IDS)
RETURN sn.id AS name, sn.description AS description,
       sn.documentation AS documentation,
       sn.kind AS kind, coalesce(u.id, sn.unit) AS unit,
       sn.tags AS tags, sn.links AS links,
       sn.source_paths AS source_paths, sn.constraints AS constraints,
       sn.validity_domain AS validity_domain,
       sn.grammar_physical_base AS physical_base, sn.grammar_subject AS subject,
       sn.grammar_component AS component, sn.grammar_coordinate AS coordinate,
       sn.grammar_position AS position, sn.grammar_process AS process,
       sn.review_status AS review_status,
       sn.confidence AS confidence, sn.model AS model,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       sn.dd_version AS dd_version,
       collect(DISTINCT src.id) AS source_ids,
       collect(DISTINCT ids.id) AS source_ids_names
"""

    try:
        rows = gc.query(cypher, names=name_list)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Fetch failed: {_neo4j_error_message(e)}"

    if not rows:
        not_found = ", ".join(name_list)
        return f"No standard names found for: {not_found}"

    return _format_fetch_report(rows, name_list)


def _format_fetch_report(rows: list[dict], requested: list[str]) -> str:
    """Format fetch results as a detailed report."""
    found_names = {r.get("name") for r in rows}
    not_found = [n for n in requested if n not in found_names]

    lines = ["## Standard Name Details\n"]

    for row in rows:
        name = row.get("name") or "unknown"
        lines.append(f"### {name}")
        lines.append("")

        description = row.get("description") or ""
        documentation = row.get("documentation") or ""
        kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        tags = row.get("tags") or []
        links = row.get("links") or []
        dd_paths = row.get("source_paths") or []
        constraints = row.get("constraints") or []
        validity_domain = row.get("validity_domain") or ""
        physical_base = row.get("physical_base") or ""
        subject = row.get("subject") or ""
        component = row.get("component") or ""
        coordinate = row.get("coordinate") or ""
        position = row.get("position") or ""
        process = row.get("process") or ""
        review_status = row.get("review_status") or ""
        confidence = row.get("confidence")
        model = row.get("model") or ""
        cocos_transformation_type = row.get("cocos_transformation_type") or ""
        cocos = row.get("cocos")
        dd_version = row.get("dd_version") or ""
        source_ids = row.get("source_ids") or []
        source_ids_names = row.get("source_ids_names") or []

        if description:
            lines.append(f"**Description:** {description}")
        if documentation:
            lines.append(f"\n**Documentation:**\n{documentation}")
        lines.append("")

        if kind:
            lines.append(f"- **Kind:** {kind}")
        if unit:
            lines.append(f"- **Unit:** {unit}")
        if review_status:
            lines.append(f"- **Review Status:** {review_status}")
        if confidence is not None:
            lines.append(f"- **Confidence:** {confidence:.2f}")
        if model:
            lines.append(f"- **Model:** {model}")
        if cocos_transformation_type:
            lines.append(f"- **COCOS Transformation:** {cocos_transformation_type}")
        if cocos is not None:
            lines.append(f"- **COCOS:** {cocos}")
        if dd_version:
            lines.append(f"- **DD Version:** {dd_version}")

        # Grammar
        grammar_parts = []
        for field_name, val in [
            ("physical_base", physical_base),
            ("subject", subject),
            ("component", component),
            ("coordinate", coordinate),
            ("position", position),
            ("process", process),
        ]:
            if val:
                grammar_parts.append(f"{field_name}={val}")
        if grammar_parts:
            lines.append(f"- **Grammar:** {', '.join(grammar_parts)}")

        if tags:
            tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            lines.append(f"- **Tags:** {tag_str}")
        if links:
            link_str = ", ".join(links) if isinstance(links, list) else str(links)
            lines.append(f"- **Links:** {link_str}")
        if dd_paths:
            path_str = (
                "\n  - " + "\n  - ".join(dd_paths)
                if isinstance(dd_paths, list)
                else str(dd_paths)
            )
            lines.append(f"- **Sources:**{path_str}")
        if constraints:
            c_str = (
                ", ".join(constraints)
                if isinstance(constraints, list)
                else str(constraints)
            )
            lines.append(f"- **Constraints:** {c_str}")
        if validity_domain:
            lines.append(f"- **Validity Domain:** {validity_domain}")
        if source_ids:
            src_str = ", ".join(s for s in source_ids if s)
            if src_str:
                lines.append(f"- **Source Nodes:** {src_str}")
        if source_ids_names:
            ids_str = ", ".join(s for s in source_ids_names if s)
            if ids_str:
                lines.append(f"- **Source IDS:** {ids_str}")

        lines.append("")

    if not_found:
        lines.append(f"**Not found:** {', '.join(not_found)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _list_standard_names
# ---------------------------------------------------------------------------


def _list_standard_names(
    *,
    tag: str | None = None,
    kind: str | None = None,
    review_status: str | None = None,
    cocos_type: str | None = None,
    gc: GraphClient | None = None,
) -> str:
    """List standard names with optional filters."""
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    # Build WHERE clause
    conditions = []
    params: dict = {}

    if tag:
        conditions.append("$tag IN sn.tags")
        params["tag"] = tag
    if kind:
        conditions.append("toLower(sn.kind) = toLower($kind)")
        params["kind"] = kind
    if review_status:
        conditions.append("toLower(sn.review_status) = toLower($review_status)")
        params["review_status"] = review_status
    if cocos_type:
        conditions.append("sn.cocos_transformation_type = $cocos_type")
        params["cocos_type"] = cocos_type

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    cypher = f"""
MATCH (sn:StandardName)
{where_clause}
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name, sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit,
       sn.review_status AS review_status,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       sn.description AS description
ORDER BY sn.id
"""

    try:
        rows = gc.query(cypher, **params)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"List failed: {_neo4j_error_message(e)}"

    return _format_list_report(
        rows, tag=tag, kind=kind, review_status=review_status, cocos_type=cocos_type
    )


def _format_list_report(
    rows: list[dict],
    *,
    tag: str | None = None,
    kind: str | None = None,
    review_status: str | None = None,
    cocos_type: str | None = None,
) -> str:
    """Format list results as a markdown table."""
    filter_parts = []
    if tag:
        filter_parts.append(f"tag={tag}")
    if kind:
        filter_parts.append(f"kind={kind}")
    if review_status:
        filter_parts.append(f"status={review_status}")
    if cocos_type:
        filter_parts.append(f"cocos_type={cocos_type}")
    filter_str = f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""

    if not rows:
        return f"## Standard Names\n\nNo standard names found{filter_str}."

    lines = [
        f"## Standard Names ({len(rows)} total{filter_str})\n",
        "| Name | Kind | Unit | Status | COCOS Type | COCOS | Description |",
        "|------|------|------|--------|------------|-------|-------------|",
    ]

    for row in rows:
        name = row.get("name") or ""
        row_kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        status = row.get("review_status") or ""
        cocos_type_val = row.get("cocos_transformation_type") or ""
        cocos_val = row.get("cocos") if row.get("cocos") is not None else ""
        desc = row.get("description") or ""
        # Truncate long descriptions
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(
            f"| {name} | {row_kind} | {unit} | {status} | {cocos_type_val} | {cocos_val} | {desc} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _list_grammar_vocabulary
# ---------------------------------------------------------------------------


def _list_grammar_vocabulary(
    segment: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """List the distinct vocabulary observed in a grammar segment.

    Aggregates the values of ``sn.grammar_<segment>`` across all
    persisted StandardName nodes, grouped by token with usage counts.
    Useful for discovering the de-facto vocabulary of a segment (closed
    or open) and for picking exact tokens to pass to the grammar_*
    filters on ``search_standard_names``.

    Args:
        segment: Segment name without the ``grammar_`` prefix (e.g.
            ``"physical_base"``, ``"component"``, ``"subject"``).

    Returns:
        Markdown table ordered by descending usage count.
    """
    seg_lc = (segment or "").strip().lower()
    if seg_lc not in GRAMMAR_SEGMENTS:
        valid = ", ".join(sorted(GRAMMAR_SEGMENTS))
        return f"Unknown grammar segment '{segment}'. Valid segments: {valid}."

    prop = f"grammar_{seg_lc}"
    # Property name is validated against GRAMMAR_SEGMENTS allowlist above,
    # so formatted interpolation is safe (no parameter injection risk).
    cypher = (
        f"MATCH (sn:StandardName) WHERE sn.{prop} IS NOT NULL "
        f"RETURN sn.{prop} AS token, count(*) AS n ORDER BY n DESC, token ASC"
    )

    try:
        if gc is None:
            gc = GraphClient()
        rows = gc.query(cypher)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error: {_neo4j_error_message(e)}"

    if not rows:
        return (
            f"## Grammar Vocabulary: `{seg_lc}`\n\n"
            f"No StandardName nodes have the `{prop}` property set."
        )

    total = sum(int(r.get("n") or 0) for r in rows)
    lines = [
        f"## Grammar Vocabulary: `{seg_lc}`",
        "",
        f"{len(rows)} distinct tokens across {total} StandardName nodes.",
        "",
        "| Token | Count |",
        "|-------|-------|",
    ]
    for r in rows:
        tok = r.get("token") or ""
        n = r.get("n") or 0
        lines.append(f"| {tok} | {n} |")
    return "\n".join(lines)
