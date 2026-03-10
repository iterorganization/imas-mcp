"""Common signal grouping layer.

Provides unified functions for creating, claiming, and propagating
SignalGroup nodes. Detection algorithms stay in their respective
discovery modules (regex patterns for FacilitySignal, tree structure
for MDSplus SignalNode).

This module handles the shared graph operations:
- create_signal_group: Create a SignalGroup with MEMBER_OF relationships
- claim_signal_groups: Atomically claim unenriched groups for LLM processing
- propagate_group_enrichment: Copy enrichment from representative to all members
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


def create_signal_group(
    facility: str,
    group_id: str,
    group_key: str,
    member_ids: list[str],
    representative_id: str,
    *,
    member_label: str = "FacilitySignal",
    gc: GraphClient | None = None,
    extra_properties: dict[str, Any] | None = None,
) -> str:
    """Create a SignalGroup node and link members via MEMBER_OF.

    Args:
        facility: Facility ID.
        group_id: Unique SignalGroup ID.
        group_key: Pattern or structural key for this group.
        member_ids: IDs of nodes to link as members.
        representative_id: ID of the representative node.
        member_label: Neo4j label for member nodes (FacilitySignal or SignalNode).
        gc: Optional existing GraphClient.
        extra_properties: Additional properties to set on the SignalGroup.

    Returns:
        The SignalGroup ID.
    """

    def _execute(client: GraphClient) -> str:
        props = {
            "facility_id": facility,
            "group_key": group_key,
            "member_count": len(member_ids),
            "representative_id": representative_id,
            "status": "discovered",
        }
        if extra_properties:
            props.update(extra_properties)

        prop_sets = ", ".join(f"sg.{k} = ${k}" for k in props)

        client.query(
            f"""
            MERGE (sg:SignalGroup {{id: $group_id}})
            ON CREATE SET {prop_sets}
            WITH sg
            MATCH (f:Facility {{id: $facility}})
            MERGE (sg)-[:AT_FACILITY]->(f)
            WITH sg
            UNWIND $member_ids AS mid
            MATCH (m:{member_label} {{id: mid}})
            MERGE (m)-[:MEMBER_OF]->(sg)
            """,
            group_id=group_id,
            facility=facility,
            member_ids=member_ids,
            **props,
        )
        return group_id

    if gc is not None:
        return _execute(gc)
    with GraphClient() as client:
        return _execute(client)


@retry_on_deadlock()
def claim_signal_groups(
    facility: str,
    *,
    batch_size: int = 10,
    member_label: str = "FacilitySignal",
    data_source_name: str | None = None,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim unenriched SignalGroups for LLM processing.

    Uses claim_token two-step verify to prevent double-claiming.

    Args:
        facility: Facility ID.
        batch_size: Max groups to claim.
        member_label: Label of member nodes (for representative lookup).
        data_source_name: Optional filter by data source.
        gc: Optional existing GraphClient.

    Returns:
        List of claimed group dicts with representative info.
    """
    token = str(uuid.uuid4())

    def _execute(client: GraphClient) -> list[dict[str, Any]]:
        where_clause = "WHERE sg.facility_id = $facility AND sg.status = 'discovered'"
        if data_source_name:
            where_clause += " AND sg.data_source_name = $data_source_name"
        where_clause += " AND sg.claimed_at IS NULL"

        # Step 1: Claim
        client.query(
            f"""
            MATCH (sg:SignalGroup)
            {where_clause}
            WITH sg ORDER BY rand() LIMIT $limit
            SET sg.claimed_at = datetime(), sg.claim_token = $token
            """,
            facility=facility,
            data_source_name=data_source_name,
            limit=batch_size,
            token=token,
        )

        # Step 2: Verify and fetch representative info
        result = client.query(
            f"""
            MATCH (sg:SignalGroup {{claim_token: $token}})
            OPTIONAL MATCH (rep:{member_label} {{id: sg.representative_id}})
            RETURN sg.id AS id,
                   sg.group_key AS group_key,
                   sg.member_count AS member_count,
                   sg.representative_id AS representative_id,
                   rep.description AS rep_description,
                   rep.unit AS rep_unit
            """,
            token=token,
        )
        return [dict(r) for r in result]

    if gc is not None:
        return _execute(gc)
    with GraphClient() as client:
        return _execute(client)


def propagate_group_enrichment(
    group_id: str,
    enrichment: dict[str, Any],
    *,
    member_label: str = "FacilitySignal",
    llm_cost: float = 0.0,
    llm_model: str | None = None,
    gc: GraphClient | None = None,
) -> int:
    """Propagate enrichment from group to all member nodes.

    Sets enrichment fields on both the SignalGroup and all its
    MEMBER_OF members. Distributes LLM cost evenly.

    Args:
        group_id: SignalGroup node ID.
        enrichment: Dict with enrichment fields to propagate.
        member_label: Label of member nodes.
        llm_cost: Total LLM cost to distribute across members.
        llm_model: Model used for enrichment.
        gc: Optional existing GraphClient.

    Returns:
        Number of member nodes updated.
    """
    propagated_fields = {
        k: v
        for k, v in enrichment.items()
        if v is not None
        and k
        in {
            "description",
            "keywords",
            "physics_domain",
            "category",
            "sign_convention",
            "name",
        }
    }

    if not propagated_fields:
        return 0

    def _execute(client: GraphClient) -> int:
        group_sets = ", ".join(f"sg.{k} = ${k}" for k in propagated_fields)
        member_sets = ", ".join(f"m.{k} = ${k}" for k in propagated_fields)

        result = client.query(
            f"""
            MATCH (sg:SignalGroup {{id: $group_id}})
            SET {group_sets},
                sg.status = 'enriched',
                sg.claimed_at = null
            WITH sg
            MATCH (m:{member_label})-[:MEMBER_OF]->(sg)
            SET {member_sets},
                m.enrichment_status = 'enriched',
                m.claimed_at = null
            RETURN count(m) AS propagated
            """,
            group_id=group_id,
            **propagated_fields,
        )
        total = result[0]["propagated"] if result else 0

        if llm_cost > 0 and total > 0:
            per_node_cost = llm_cost / total
            client.query(
                f"""
                MATCH (m:{member_label})-[:MEMBER_OF]->(:SignalGroup {{id: $group_id}})
                WHERE m.enrichment_status = 'enriched'
                SET m.llm_cost = $per_node_cost,
                    m.llm_model = $llm_model,
                    m.llm_at = datetime()
                """,
                group_id=group_id,
                per_node_cost=per_node_cost,
                llm_model=llm_model,
            )

        return total

    if gc is not None:
        return _execute(gc)
    with GraphClient() as client:
        return _execute(client)


def release_group_claims(
    group_ids: list[str],
    *,
    gc: GraphClient | None = None,
) -> int:
    """Release claims on SignalGroups (on error)."""
    if not group_ids:
        return 0

    def _execute(client: GraphClient) -> int:
        result = client.query(
            """
            UNWIND $ids AS gid
            MATCH (sg:SignalGroup {id: gid})
            SET sg.claimed_at = null, sg.claim_token = null
            RETURN count(sg) AS released
            """,
            ids=group_ids,
        )
        return result[0]["released"] if result else 0

    if gc is not None:
        return _execute(gc)
    with GraphClient() as client:
        return _execute(client)


def has_pending_groups(
    facility: str,
    *,
    data_source_name: str | None = None,
    gc: GraphClient | None = None,
) -> bool:
    """Check if any SignalGroups need enrichment."""

    def _execute(client: GraphClient) -> bool:
        where = "WHERE sg.facility_id = $facility AND sg.status = 'discovered'"
        if data_source_name:
            where += " AND sg.data_source_name = $data_source_name"

        result = client.query(
            f"""
            MATCH (sg:SignalGroup)
            {where}
            RETURN count(sg) > 0 AS has_work
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        return result[0]["has_work"] if result else False

    if gc is not None:
        return _execute(gc)
    with GraphClient() as client:
        return _execute(client)
