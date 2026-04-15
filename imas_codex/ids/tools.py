"""Tool functions for the IMAS mapping pipeline.

Plain functions (not MCP tools) that the pipeline orchestrator calls
to gather context for the LLM at each step. All DD context queries
delegate to shared Graph*Tool classes in ``imas_codex.tools.graph_search``.

Functions that remain local (no shared tool equivalent):
- query_signal_sources — facility-specific SignalSource traversal
- fetch_source_code_refs — SignalSource→FacilitySignal→CodeChunk
- search_existing_mappings — IMASMapping lookup for current facility
- fetch_cross_facility_mappings — cross-facility IMASMapping precedent
- get_sign_flip_paths — COCOS-specific
- analyze_units — pint-based unit compatibility
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from imas_codex.core.node_categories import SEARCHABLE_CATEGORIES
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# IDS structure exploration
# ---------------------------------------------------------------------------


def fetch_imas_subtree(
    ids_name: str,
    path: str | None = None,
    *,
    gc: GraphClient | None = None,
    leaf_only: bool = False,
    max_paths: int | None = None,
    dd_version: str | int | None = None,
) -> list[dict[str, Any]]:
    """Return IDS tree structure with full metadata.

    Each row contains ``id``, ``name``, ``data_type``, ``node_type``,
    ``documentation``, ``units``.

    Delegates to ``GraphListTool.list_dd_paths(response_profile="standard")``.
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.tools.graph_search import GraphListTool

    query = f"{ids_name}/{path}" if path else ids_name
    tool = GraphListTool(gc)
    result = _run_async(
        tool.list_dd_paths(
            paths=query,
            leaf_only=leaf_only,
            max_paths=max_paths,
            dd_version=dd_version,
            response_profile="standard",
        )
    )
    return result.as_dicts()


def fetch_imas_fields(
    ids_name: str,
    paths: list[str],
    *,
    gc: GraphClient | None = None,
    dd_version: str | int | None = None,
) -> list[dict[str, Any]]:
    """Return detailed field info for specific IMAS paths.

    Delegates to ``GraphPathTool.fetch_dd_paths()``.
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.models.error_models import ToolError
    from imas_codex.tools.graph_search import GraphPathTool

    # Qualify paths with ids prefix if missing
    qualified = [
        p if p.startswith(f"{ids_name}/") else f"{ids_name}/{p}" for p in paths
    ]

    tool = GraphPathTool(gc)
    result = _run_async(tool.fetch_dd_paths(paths=qualified, dd_version=dd_version))
    if isinstance(result, ToolError):
        return []
    return result.as_dicts()


def search_imas_semantic(
    query: str,
    ids_name: str | None = None,
    *,
    gc: GraphClient | None = None,
    k: int = 20,
    dd_version: str | int | None = None,
) -> list[dict[str, Any]]:
    """Semantic search for IMAS paths using vector index.

    Delegates to ``GraphSearchTool.search_dd_paths()``.
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.models.error_models import ToolError
    from imas_codex.tools.graph_search import GraphSearchTool

    tool = GraphSearchTool(gc)
    result = _run_async(
        tool.search_dd_paths(
            query=query,
            ids_filter=ids_name,
            max_results=k,
            dd_version=dd_version,
        )
    )
    if isinstance(result, ToolError):
        return []
    return [
        {
            "id": h.path,
            "documentation": h.documentation,
            "data_type": h.data_type,
            "node_type": h.node_type,
            "units": h.units,
            "score": h.score,
        }
        for h in result.hits
    ]


# ---------------------------------------------------------------------------
# COCOS + Units
# ---------------------------------------------------------------------------


def get_sign_flip_paths(ids_name: str) -> list[str]:
    """Return IMAS paths requiring COCOS sign flip for *ids_name*."""
    from imas_codex.cocos.transforms import get_sign_flip_paths as _get

    return _get(ids_name)


def analyze_units(
    signal_unit: str | None,
    imas_unit: str | None,
) -> dict[str, Any]:
    """Analyse unit compatibility between a signal and an IMAS field.

    Returns a dict with keys:
        compatible (bool), conversion_factor (float|None),
        signal_unit, imas_unit, error (str|None).
    """
    result: dict[str, Any] = {
        "signal_unit": signal_unit,
        "imas_unit": imas_unit,
        "compatible": False,
        "conversion_factor": None,
        "error": None,
    }
    if not signal_unit or not imas_unit:
        result["compatible"] = signal_unit == imas_unit  # both None → compatible
        return result

    try:
        from imas_codex.units import unit_registry

        q_sig = unit_registry.Quantity(1.0, signal_unit)
        q_imas = unit_registry.Quantity(1.0, imas_unit)
        if q_sig.dimensionality == q_imas.dimensionality:
            result["compatible"] = True
            result["conversion_factor"] = q_sig.to(imas_unit).magnitude
        else:
            result["error"] = (
                f"Incompatible dimensions: {q_sig.dimensionality} vs "
                f"{q_imas.dimensionality}"
            )
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def check_dd_paths(
    paths: list[str],
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Validate that IMAS paths exist in the graph.

    Returns a list of dicts with ``path``, ``exists``, ``data_type``,
    ``units``, and optionally ``suggestion`` if renamed.
    """
    if gc is None:
        gc = GraphClient()

    results: list[dict[str, Any]] = []
    for path in paths:
        row = gc.query(
            """
            MATCH (p:IMASNode {id: $path})
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS id, p.data_type AS data_type, u.symbol AS units
            """,
            path=path,
        )
        if row:
            results.append(
                {
                    "path": row[0]["id"],
                    "exists": True,
                    "data_type": row[0]["data_type"],
                    "units": row[0]["units"],
                }
            )
        else:
            # Check for rename
            renamed = gc.query(
                """
                MATCH (old:IMASNode {id: $path})-[:RENAMED_TO]->(new:IMASNode)
                RETURN new.id AS new_path
                """,
                path=path,
            )
            entry: dict[str, Any] = {"path": path, "exists": False}
            if renamed:
                entry["suggestion"] = renamed[0]["new_path"]
            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Signal sources + existing mappings
# ---------------------------------------------------------------------------


def query_signal_sources(
    facility: str,
    ids_name: str | None = None,
    *,
    gc: GraphClient | None = None,
    physics_domains: list[str] | None = None,
    status_filter: str = "enriched",
) -> list[dict[str, Any]]:
    """Return enriched signal sources for a facility, optionally filtered by IDS.

    Each row contains group metadata plus member signals and any
    existing MAPS_TO_IMAS connections.

    Args:
        facility: Facility identifier.
        ids_name: Optional IDS name to filter by existing mappings.
        gc: GraphClient instance.
        physics_domains: Optional list of physics domains to filter by.
        status_filter: Status to filter by (default: 'enriched').
    """
    if gc is None:
        gc = GraphClient()

    filters: list[str] = []
    params: dict[str, Any] = {"facility": facility}

    if ids_name:
        filters.append(
            "AND EXISTS { "
            "  MATCH (sg)-[:MAPS_TO_IMAS]->(ip:IMASNode) "
            "  WHERE ip.ids = $ids_name "
            "}"
        )
        params["ids_name"] = ids_name

    if physics_domains:
        filters.append("AND sg.physics_domain IN $domains")
        params["domains"] = physics_domains

    if status_filter:
        filters.append("AND sg.status = $status_filter")
        params["status_filter"] = status_filter

    filter_clause = "\n        ".join(filters)

    cypher = f"""
        MATCH (sg:SignalSource)
        WHERE sg.facility_id = $facility
        {filter_clause}
        OPTIONAL MATCH (m)-[:MEMBER_OF]->(sg)
        WITH sg, count(m) AS member_count,
             collect(DISTINCT m.id)[..5] AS sample_members,
             collect(DISTINCT m.accessor)[..10] AS sample_accessors
        OPTIONAL MATCH (rep:FacilitySignal {{id: sg.representative_id}})
        OPTIONAL MATCH (sg)-[r:MAPS_TO_IMAS]->(ip:IMASNode)
        RETURN sg.id AS id, sg.group_key AS group_key,
               sg.description AS description,
               sg.keywords AS keywords,
               sg.physics_domain AS physics_domain,
               sg.status AS status,
               member_count,
               sample_members,
               sample_accessors,
               rep.description AS rep_description,
               rep.unit AS rep_unit,
               rep.sign_convention AS rep_sign_convention,
               rep.cocos AS rep_cocos,
               collect(DISTINCT {{
                   target_id: ip.id,
                   transform: r.transform_expression,
                   source_units: r.source_units,
                   target_units: r.target_units
               }}) AS imas_mappings
        ORDER BY sg.group_key
    """
    return gc.query(cypher, **params)


def query_ids_physics_domains(
    ids_name: str,
    *,
    gc: GraphClient | None = None,
    dd_version: str | int | None = None,
) -> list[str]:
    """Return distinct physics domains for an IDS from IMASNode paths.

    Uses physics_domain field on IMASNode to find which domains
    the target IDS covers. This enables filtering signal sources
    to only those matching the target IDS physics.
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.tools.graph_search import _dd_version_clause

    dd_params: dict[str, Any] = {}
    dd_clause = _dd_version_clause("p", dd_version, dd_params)

    cypher = f"""
        MATCH (p:IMASNode)
        WHERE p.ids = $ids_name
          AND p.physics_domain IS NOT NULL
          AND p.physics_domain <> ''
        {dd_clause}
        RETURN DISTINCT p.physics_domain AS domain
    """
    rows = gc.query(cypher, ids_name=ids_name, **dd_params)
    return [r["domain"] for r in rows if r.get("domain")]


def discover_mappable_ids(
    facility: str,
    *,
    gc: GraphClient | None = None,
    domains: list[str] | None = None,
    ids_filter: list[str] | None = None,
    dd_version: str | int | None = None,
) -> dict[str, Any]:
    """Discover IDS targets achievable from available signal sources.

    Filtering uses union semantics:
      --domain selects IDS whose IMASNodes touch those physics domains
      --ids selects those IDS names directly
      Both flags produce the union of both result sets
      Neither flag discovers all IDS with matching signal source domains

    Args:
        facility: Facility identifier.
        gc: GraphClient instance.
        domains: Select IDS in these physics domains (union with ids_filter).
        ids_filter: Select these IDS names directly (union with domains).
        dd_version: DD major version filter.

    Returns:
        Dict with keys:
            available_domains: list[str] — physics domains with enriched sources
            ids_targets: list[dict] — [{ids_name, domains, source_count}]
            total_sources: int — total enriched signal sources in scope
    """
    if gc is None:
        gc = GraphClient()

    # Step 1: Get physics domains from enriched signal sources
    source_rows = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility})
        WHERE sg.status = 'enriched'
          AND sg.physics_domain IS NOT NULL
          AND sg.physics_domain <> ''
        RETURN sg.physics_domain AS domain, count(sg) AS cnt
        ORDER BY cnt DESC
        """,
        facility=facility,
    )
    available_domains = [r["domain"] for r in source_rows]
    source_counts = {r["domain"]: r["cnt"] for r in source_rows}

    if not available_domains:
        return {
            "available_domains": [],
            "ids_targets": [],
            "total_sources": 0,
        }

    from imas_codex.tools.graph_search import _dd_version_clause

    dd_params: dict[str, Any] = {}
    dd_clause = _dd_version_clause("p", dd_version, dd_params)

    # Step 2: Build IDS target set via union of --domain and --ids filters
    ids_by_name: dict[str, dict] = {}

    # Branch A: --domain selects IDS whose IMASNodes touch those domains
    if domains:
        # Validate requested domains exist in signal sources
        valid_domains = [d for d in domains if d in available_domains]
        if valid_domains:
            domain_rows = gc.query(
                f"""
                MATCH (p:IMASNode)
                WHERE p.physics_domain IN $filter_domains
                  AND p.ids IS NOT NULL
                  AND p.ids <> ''
                  {dd_clause}
                WITH DISTINCT p.ids AS ids_name,
                     collect(DISTINCT p.physics_domain) AS domains
                RETURN ids_name, domains
                ORDER BY ids_name
                """,
                filter_domains=valid_domains,
                **dd_params,
            )
            for r in domain_rows:
                ids_by_name[r["ids_name"]] = {
                    "ids_name": r["ids_name"],
                    "domains": set(r["domains"]),
                }

    # Branch B: --ids selects IDS directly, resolving their domains
    if ids_filter:
        ids_rows = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids IN $ids_filter
              AND p.physics_domain IS NOT NULL
              AND p.physics_domain <> ''
              {dd_clause}
            WITH DISTINCT p.ids AS ids_name,
                 collect(DISTINCT p.physics_domain) AS domains
            RETURN ids_name, domains
            ORDER BY ids_name
            """,
            ids_filter=ids_filter,
            **dd_params,
        )
        for r in ids_rows:
            name = r["ids_name"]
            if name in ids_by_name:
                ids_by_name[name]["domains"] |= set(r["domains"])
            else:
                ids_by_name[name] = {
                    "ids_name": name,
                    "domains": set(r["domains"]),
                }

    # Branch C: no filters — discover all IDS from available domains
    if not domains and not ids_filter:
        all_rows = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.physics_domain IN $available_domains
              AND p.ids IS NOT NULL
              AND p.ids <> ''
              {dd_clause}
            WITH DISTINCT p.ids AS ids_name,
                 collect(DISTINCT p.physics_domain) AS domains
            RETURN ids_name, domains
            ORDER BY ids_name
            """,
            available_domains=available_domains,
            **dd_params,
        )
        for r in all_rows:
            ids_by_name[r["ids_name"]] = {
                "ids_name": r["ids_name"],
                "domains": set(r["domains"]),
            }

    # Step 3: Compute source counts and build targets list
    ids_targets = []
    all_domains: set[str] = set()
    for entry in sorted(ids_by_name.values(), key=lambda e: e["ids_name"]):
        ds = sorted(entry["domains"])
        source_count = sum(source_counts.get(d, 0) for d in ds)
        ids_targets.append(
            {
                "ids_name": entry["ids_name"],
                "domains": ds,
                "source_count": source_count,
            }
        )
        all_domains.update(ds)

    total_sources = sum(
        source_counts.get(d, 0) for d in all_domains & set(available_domains)
    )

    return {
        "available_domains": available_domains,
        "ids_targets": ids_targets,
        "total_sources": total_sources,
    }


def fetch_source_code_refs(
    source_id: str,
    *,
    gc: GraphClient | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return code chunks showing how a signal source is read.

    Follows: SignalSource → representative FacilitySignal → SignalNode
    → CodeChunk to find actual code snippets that read the signal data.
    """
    if gc is None:
        gc = GraphClient()

    cypher = """
        MATCH (sg:SignalSource {id: $source_id})
        OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
               -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
               -[:EXTRACTED_FROM]->(cc:CodeChunk)
        RETURN cc.text AS code, cc.language AS language,
               cc.source_file AS file
        LIMIT $limit
    """
    return gc.query(cypher, source_id=source_id, limit=limit)


def search_existing_mappings(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Return existing mapping state for a facility+IDS pair.

    Returns a dict with ``mapping`` (IMASMapping node info or None),
    ``sections`` (POPULATES connections), and ``bindings``
    (MAPS_TO_IMAS relationships).
    """
    if gc is None:
        gc = GraphClient()

    # Query by facility_id + ids_name (supports versioned IDs)
    mapping_rows = gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids_name})
        RETURN m.id AS id, m.facility_id AS facility_id,
               m.ids_name AS ids_name, m.dd_version AS dd_version,
               m.status AS status, m.provider AS provider
        ORDER BY m.dd_version DESC
        LIMIT 1
        """,
        facility=facility,
        ids_name=ids_name,
    )

    mapping = mapping_rows[0] if mapping_rows else None
    mapping_id = mapping["id"] if mapping else None

    # Sections via POPULATES
    sections: list[dict[str, Any]] = []
    if mapping:
        sections = gc.query(
            """
            MATCH (m:IMASMapping {id: $id})-[r:POPULATES]->(ip:IMASNode)
            RETURN ip.id AS imas_path, ip.data_type AS data_type,
                   r.config AS config
            """,
            id=mapping_id,
        )

    # Field-level mappings via signal sources
    bindings: list[dict[str, Any]] = []
    if mapping:
        bindings = gc.query(
            """
            MATCH (m:IMASMapping {id: $id})-[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
            MATCH (sg)-[r:MAPS_TO_IMAS]->(ip:IMASNode)
            RETURN sg.id AS source_id,
                   ip.id AS target_id, r.transform_expression AS transform_expression,
                   r.source_units AS source_units, r.target_units AS target_units,
                   r.source_property AS source_property
            """,
            id=mapping_id,
        )

    return {
        "mapping": mapping,
        "sections": sections,
        "bindings": bindings,
    }


# ---------------------------------------------------------------------------
# Wiki and code context for mapping enrichment
# ---------------------------------------------------------------------------


def fetch_wiki_context(
    facility: str,
    physics_domains: list[str],
    *,
    min_imas_relevance: float = 0.5,
    k: int = 10,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch wiki chunks relevant to the mapping task.

    Uses physics_domain + score_imas_relevance filtering on WikiPage to find
    high-value documentation, then returns chunk text + page title + scores.
    """
    if gc is None:
        gc = GraphClient()

    if not physics_domains:
        return []

    results = gc.query(
        """
        MATCH (wp:WikiPage)-[:HAS_CHUNK]->(wc:WikiChunk)
        WHERE wp.facility_id = $facility
          AND wp.physics_domain IN $domains
          AND wp.score_imas_relevance >= $min_score
          AND wp.status = 'ingested'
        RETURN wc.text AS text,
               wp.title AS page_title,
               wp.physics_domain AS physics_domain,
               wp.score_imas_relevance AS score_imas_relevance,
               wp.score_data_access AS score_data_access,
               wp.score_composite AS score_composite
        ORDER BY wp.score_imas_relevance DESC, wp.score_composite DESC
        LIMIT $k
        """,
        facility=facility,
        domains=physics_domains,
        min_score=min_imas_relevance,
        k=k,
    )
    return results


def fetch_code_context(
    facility: str,
    physics_domains: list[str],
    *,
    score_dimension: str = "score_data_access",
    min_score: float = 0.5,
    k: int = 10,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch code chunks demonstrating data access patterns.

    Filters FacilityPaths by physics_domain and the given score dimension,
    then retrieves CodeChunks via CodeFile → CodeExample → CodeChunk.
    This surfaces code that shows HOW signals are read, complementing
    the narrow fetch_source_code_refs() which only finds code for a
    single signal.
    """
    if gc is None:
        gc = GraphClient()

    if not physics_domains:
        return []

    # Sanitize dimension name to prevent injection
    dim = "".join(c for c in score_dimension if c.isalnum() or c == "_")

    results = gc.query(
        f"""
        MATCH (fp:FacilityPath)<-[:IN_DIRECTORY]-(cf:CodeFile)
        WHERE fp.facility_id = $facility
          AND fp.physics_domain IN $domains
          AND cf.{dim} >= $min_score
          AND cf.status = 'ingested'
        WITH cf
        MATCH (cf)-[:HAS_EXAMPLE]->(ce)-[:HAS_CHUNK]->(cc:CodeChunk)
        RETURN cc.text AS text,
               cc.function_name AS function_name,
               cc.source_file AS source_file,
               cc.language AS language,
               cf.{dim} AS score_data_access,
               cf.score_composite AS score_composite
        ORDER BY cf.{dim} DESC, cf.score_composite DESC
        LIMIT $k
        """,
        facility=facility,
        domains=physics_domains,
        min_score=min_score,
        k=k,
    )
    return results


def fetch_cross_facility_mappings(
    ids_name: str,
    exclude_facility: str,
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Return active mappings from other facilities to this IDS.

    Useful as precedent context: if TCV already mapped its plasma current
    to ``equilibrium/time_slice/global_quantities/ip``, that's strong
    evidence for where JET's plasma current should go.
    """
    if gc is None:
        gc = GraphClient()

    return gc.query(
        """
        MATCH (m:IMASMapping)-[:POPULATES]->(ip:IMASNode)
        WHERE m.ids_name = $ids_name
          AND m.facility_id <> $exclude
          AND m.status IN ['active', 'validated']
        RETURN m.facility_id AS facility,
               ip.id AS target_path,
               m.status AS status
        ORDER BY m.facility_id, ip.id
        """,
        ids_name=ids_name,
        exclude=exclude_facility,
    )


# ---------------------------------------------------------------------------
# Semantic match matrix
# ---------------------------------------------------------------------------


def compute_semantic_matches(
    source_descriptions: list[tuple[str, str]],
    target_ids_name: str,
    *,
    gc: GraphClient | None = None,
    k_per_source: int = 5,
    include_wiki: bool = True,
    include_code: bool = True,
    dd_version: str | int | None = None,
    on_progress: Callable[[str], None] | None = None,
    max_workers: int = 8,
    precomputed_embeddings: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Compute semantic match vectors between sources and targets.

    For each source, embeds its description and searches against:
    1. imas_node_embedding (primary: target IMAS fields)
    2. wiki_chunk_embedding (bridging: domain documentation)
    3. code_chunk_embedding (bridging: data access patterns)

    Uses batch embedding for efficiency — all source descriptions are
    embedded in a single encoder call.  Per-source vector queries run
    in a thread pool (``max_workers`` threads) for parallelism over
    the Neo4j tunnel.

    Parameters
    ----------
    precomputed_embeddings : array-like, optional
        If provided, skip the batch embed call and use these embeddings
        directly.  Must have len == len(source_descriptions).

    Returns a dict mapping source_id -> ranked match list, where each
    match has {target_id, score, content_type, excerpt}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if gc is None:
        gc = GraphClient()

    if not source_descriptions:
        return {}

    source_ids = [sid for sid, _ in source_descriptions]

    if precomputed_embeddings is not None:
        embeddings = precomputed_embeddings
    else:
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder()
        texts = [desc for _, desc in source_descriptions]

        if on_progress:
            on_progress(f"embedding {len(texts)} sources")

        embeddings = encoder.embed_texts(texts)

    # Pre-compute shared Cypher fragments
    imas_search_where = ["n.node_category IN $searchable_categories"]
    imas_extra_params: dict[str, Any] = {
        "searchable_categories": list(SEARCHABLE_CATEGORIES),
    }
    if target_ids_name:
        imas_search_where.append("n.id STARTS WITH $ids_prefix")
        imas_extra_params["ids_prefix"] = f"{target_ids_name}/"

    dd_version_join = ""
    if dd_version is not None:
        dd_version_join = (
            "MATCH (n)-[:INTRODUCED_IN]->(iv:DDVersion) "
            "WHERE toInteger(split(iv.id, '.')[0]) <= $dd_version "
        )
        imas_extra_params["dd_version"] = dd_version

    _imas_search_where_str = " AND ".join(imas_search_where)
    imas_cypher = (
        "CYPHER 25\n"
        "MATCH (n:IMASNode)\n"
        "SEARCH n IN (\n"
        "  VECTOR INDEX imas_node_embedding\n"
        "  FOR $embedding\n"
        "  LIMIT $k\n"
        ") SCORE AS score\n"
        f"WHERE {_imas_search_where_str}\n"
        f"WITH n, score {dd_version_join}"
        "RETURN n.id AS id, n.documentation AS doc, score "
        "ORDER BY score DESC LIMIT $limit"
    )

    def _search_one(idx: int) -> tuple[str, list[dict[str, Any]]]:
        """Run all vector searches for a single source (thread-safe)."""
        source_id = source_ids[idx]
        embedding = (
            embeddings[idx].tolist()
            if hasattr(embeddings[idx], "tolist")
            else list(embeddings[idx])
        )
        matches: list[dict[str, Any]] = []

        # Each thread gets its own GraphClient connection
        with GraphClient() as tgc:
            # 1. IMAS node embeddings (primary)
            try:
                imas_params = {
                    "k": k_per_source * 2,
                    "embedding": embedding,
                    "limit": k_per_source,
                    **imas_extra_params,
                }
                imas_hits = tgc.query(imas_cypher, **imas_params)
                for h in imas_hits:
                    doc = h.get("doc") or ""
                    matches.append(
                        {
                            "target_id": h["id"],
                            "score": round(h["score"], 3),
                            "content_type": "imas",
                            "excerpt": doc[:200] + "..." if len(doc) > 200 else doc,
                        }
                    )
            except Exception:
                logger.debug("IMAS vector search failed for %s", source_id)

            # 2. Wiki chunk embeddings (bridging)
            if include_wiki:
                try:
                    wiki_hits = tgc.query(
                        "CYPHER 25\n"
                        "MATCH (c:WikiChunk)\n"
                        "SEARCH c IN (\n"
                        "  VECTOR INDEX wiki_chunk_embedding\n"
                        "  FOR $embedding\n"
                        "  LIMIT $k\n"
                        ") SCORE AS score\n"
                        "MATCH (p:WikiPage)-[:HAS_CHUNK]->(c) "
                        "RETURN p.title AS title, c.text AS text, score "
                        "ORDER BY score DESC LIMIT $limit",
                        k=k_per_source * 2,
                        embedding=embedding,
                        limit=k_per_source,
                    )
                    for h in wiki_hits:
                        text = h.get("text") or ""
                        matches.append(
                            {
                                "target_id": h.get("title", ""),
                                "score": round(h["score"], 3),
                                "content_type": "wiki",
                                "excerpt": text[:200] + "..."
                                if len(text) > 200
                                else text,
                            }
                        )
                except Exception:
                    logger.debug("Wiki vector search failed for %s", source_id)

            # 3. Code chunk embeddings (bridging)
            if include_code:
                try:
                    code_hits = tgc.query(
                        "CYPHER 25\n"
                        "MATCH (cc:CodeChunk)\n"
                        "SEARCH cc IN (\n"
                        "  VECTOR INDEX code_chunk_embedding\n"
                        "  FOR $embedding\n"
                        "  LIMIT $k\n"
                        ") SCORE AS score\n"
                        "RETURN cc.source_file AS source_file, cc.function_name AS func, "
                        "cc.text AS text, score "
                        "ORDER BY score DESC LIMIT $limit",
                        k=k_per_source * 2,
                        embedding=embedding,
                        limit=k_per_source,
                    )
                    for h in code_hits:
                        text = h.get("text") or ""
                        label = h.get("source_file") or ""
                        if h.get("func"):
                            label += f"::{h['func']}"
                        matches.append(
                            {
                                "target_id": label,
                                "score": round(h["score"], 3),
                                "content_type": "code",
                                "excerpt": text[:200] + "..."
                                if len(text) > 200
                                else text,
                            }
                        )
                except Exception:
                    logger.debug("Code vector search failed for %s", source_id)

        matches.sort(key=lambda m: m["score"], reverse=True)
        return source_id, matches

    # Run per-source vector queries in a thread pool
    results: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    total = len(source_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_search_one, i): i for i in range(total)}
        for future in as_completed(futures):
            source_id, matches = future.result()
            results[source_id] = matches
            completed += 1
            if on_progress and completed % 50 == 0:
                on_progress(f"vector search {completed}/{total}")

    return results
