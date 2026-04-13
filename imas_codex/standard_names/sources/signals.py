"""Signals source: extract standard name candidates from FacilitySignal nodes."""

from __future__ import annotations

import logging

from imas_codex.standard_names.sources.base import ExtractionBatch

logger = logging.getLogger(__name__)


def extract_signal_candidates(
    *,
    facility: str,
    domain_filter: str | None = None,
    limit: int = 500,
    existing_names: set[str] | None = None,
) -> list[ExtractionBatch]:
    """Extract candidate quantities from facility signals.

    Queries FacilitySignal nodes from the graph, groups by physics domain,
    and returns batches ready for LLM composition.

    Args:
        facility: Facility identifier (e.g., "tcv", "jet")
        domain_filter: Restrict to physics domain
        limit: Max signals to extract
        existing_names: Known standard names for dedup awareness

    Returns:
        List of ExtractionBatch objects grouped by physics domain
    """
    from imas_codex.graph.client import GraphClient

    if existing_names is None:
        existing_names = set()

    with GraphClient() as gc:
        params: dict = {"facility": facility, "limit": limit}
        where_parts = [
            "s.facility_id = $facility",
            "s.description IS NOT NULL",
            "s.description <> ''",
        ]
        if domain_filter:
            where_parts.append("s.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where_clause = " AND ".join(where_parts)

        results = list(
            gc.query(
                f"""
            MATCH (s:FacilitySignal)
            WHERE {where_clause}
            OPTIONAL MATCH (s)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN s.id AS signal_id, s.description AS description,
                   s.physics_domain AS physics_domain,
                   s.canonical_units AS units,
                   sn.id AS existing_standard_name
            ORDER BY s.physics_domain, s.id
            LIMIT $limit
        """,
                **params,
            )
        )

    if not results:
        logger.info("No signals found for %s matching filters", facility)
        return []

    # Filter out signals that already have standard names
    unmapped = [r for r in results if not r.get("existing_standard_name")]

    # Group by physics domain
    groups: dict[str, list[dict]] = {}
    for row in unmapped:
        domain = row.get("physics_domain") or "unknown"
        groups.setdefault(domain, []).append(dict(row))

    batches = []
    for domain, items in groups.items():
        context = f"Facility: {facility}, Domain: {domain}"
        context += f"\n{len(items)} signals without standard names"

        batches.append(
            ExtractionBatch(
                source="signals",
                group_key=f"{facility}/{domain}",
                items=items,
                context=context,
                existing_names=existing_names,
            )
        )

    logger.info("Extracted %d batches from %d signals", len(batches), len(unmapped))
    return batches
