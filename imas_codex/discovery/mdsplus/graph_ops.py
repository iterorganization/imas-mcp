"""Graph operations for MDSplus tree discovery claim coordination.

Provides seed, claim, mark, and query functions for the tree discovery
pipeline. SignalEpoch nodes use status + claimed_at for extraction
coordination. SignalNode enrichment uses enrichment_status.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from imas_codex.discovery.base.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    reset_stale_claims,
    retry_on_deadlock,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = DEFAULT_CLAIM_TIMEOUT_SECONDS  # 300s


# ---------------------------------------------------------------------------
# Seed — create pending SignalEpoch nodes from config
# ---------------------------------------------------------------------------


def seed_versions(
    facility: str,
    data_source_name: str,
    ver_list: list[int],
    version_config: list[dict] | None = None,
) -> int:
    """Create SignalEpoch nodes with status=discovered for each version.

    Idempotent: only creates nodes that don't already exist. Existing nodes
    are not modified (preserves status/claimed_at from prior runs).

    Args:
        facility: Facility identifier
        data_source_name: Static tree name
        ver_list: Version numbers to seed
        version_config: Optional version configs with first_shot info

    Returns:
        Number of new versions seeded
    """
    if not ver_list:
        return 0

    # Build first_shot + description lookups
    first_shot_map: dict[int, int] = {}
    description_map: dict[int, str] = {}
    if version_config:
        for vc in version_config:
            v = vc["version"]
            if "first_shot" in vc:
                first_shot_map[v] = vc["first_shot"]
            if "description" in vc:
                description_map[v] = vc["description"]

    records = []
    for ver in ver_list:
        epoch_id = f"{facility}:{data_source_name}:v{ver}"
        records.append(
            {
                "id": epoch_id,
                "facility_id": facility,
                "data_source_name": data_source_name,
                "version": ver,
                "first_shot": first_shot_map.get(ver, ver),
                "description": description_map.get(ver, ""),
                "status": "discovered",
            }
        )

    with GraphClient() as gc:
        gc.ensure_facility(facility)
        result = gc.query(
            """
            UNWIND $records AS rec
            MERGE (v:SignalEpoch {id: rec.id})
            ON CREATE SET
                v.facility_id = rec.facility_id,
                v.data_source_name = rec.data_source_name,
                v.version = rec.version,
                v.first_shot = rec.first_shot,
                v.description = rec.description,
                v.status = rec.status
            WITH v, rec
            MATCH (f:Facility {id: rec.facility_id})
            MERGE (v)-[:AT_FACILITY]->(f)
            WITH v, rec, f
            MERGE (t:DataSource {id: rec.facility_id + ':' + rec.data_source_name})
            ON CREATE SET t.name = rec.data_source_name, t.facility_id = rec.facility_id
            MERGE (v)-[:IN_DATA_SOURCE]->(t)
            MERGE (t)-[:AT_FACILITY]->(f)
            RETURN count(CASE WHEN v.status = 'discovered' THEN 1 END) AS seeded
            """,
            records=records,
        )
        return result[0]["seeded"] if result else 0


def backfill_tree_relationships(facility: str) -> int:
    """Create missing DataSource nodes and IN_DATA_SOURCE relationships.

    One-time migration for DataNodes ingested before IN_DATA_SOURCE relationships
    were added. Checks cheaply first; skips entirely when nothing is missing.

    Returns:
        Number of DataNodes that gained an IN_DATA_SOURCE relationship.
    """
    with GraphClient() as gc:
        # Cheap existence check — avoids scanning all DataNodes on every run
        check = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility})
            WHERE n.data_source_name IS NOT NULL
              AND NOT (n)-[:IN_DATA_SOURCE]->(:DataSource)
            RETURN count(n) > 0 AS needs_backfill
            """,
            facility=facility,
        )
        if not check or not check[0]["needs_backfill"]:
            return 0

        result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility})
            WHERE n.data_source_name IS NOT NULL
              AND NOT (n)-[:IN_DATA_SOURCE]->(:DataSource)
            WITH n
            MATCH (f:Facility {id: $facility})
            MERGE (t:DataSource {id: $facility + ':' + n.data_source_name})
            ON CREATE SET t.name = n.data_source_name, t.facility_id = $facility
            MERGE (t)-[:AT_FACILITY]->(f)
            MERGE (n)-[:IN_DATA_SOURCE]->(t)
            RETURN count(n) AS backfilled
            """,
            facility=facility,
        )
        count = result[0]["backfilled"] if result else 0
        if count > 0:
            logger.info(
                "Backfilled %d SignalNode→DataSource IN_DATA_SOURCE relationships for %s",
                count,
                facility,
            )
        return count


# ---------------------------------------------------------------------------
# Claim — atomic claim of pending SignalEpoch for extraction
# ---------------------------------------------------------------------------


@retry_on_deadlock()
def claim_version_for_extraction(
    facility: str,
    data_source_name: str,
) -> dict[str, Any] | None:
    """Atomically claim a pending SignalEpoch for extraction.

    Claims an unclaimed version with status=discovered.
    Uses claim_token + ORDER BY rand() to prevent deadlocks.

    Returns:
        Dict with id, version, first_shot or None if no work available
    """
    claim_token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility
              AND v.data_source_name = $data_source_name
              AND v.status = 'discovered'
              AND v.claimed_at IS NULL
            WITH v ORDER BY rand() LIMIT 1
            SET v.claimed_at = datetime(), v.claim_token = $token
            """,
            facility=facility,
            data_source_name=data_source_name,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (v:SignalEpoch {claim_token: $token})
            RETURN v.id AS id, v.version AS version,
                   v.first_shot AS first_shot
            """,
            token=claim_token,
        )
        if result:
            return dict(result[0])
        return None


def mark_version_extracted(
    version_id: str,
    node_count: int,
) -> None:
    """Mark a SignalEpoch as extracted (status=ingested, clear claim)."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch {id: $id})
            SET v.status = 'ingested',
                v.claimed_at = null,
                v.node_count = $node_count,
                v.discovery_date = datetime()
            """,
            id=version_id,
            node_count=node_count,
        )


def release_version_claim(version_id: str) -> None:
    """Release claim on a SignalEpoch (on error)."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch {id: $id})
            SET v.claimed_at = null
            """,
            id=version_id,
        )


def mark_version_failed(version_id: str, error: str) -> None:
    """Mark a SignalEpoch as permanently failed.

    Used when extraction returns a non-transient error (e.g. tree file
    not found on disk). Prevents infinite retry loops by moving the
    version out of 'discovered' status.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch {id: $id})
            SET v.status = 'failed',
                v.claimed_at = null,
                v.error = $error
            """,
            id=version_id,
            error=error[:500],
        )


# ---------------------------------------------------------------------------
# Units — track unit extraction per version
# ---------------------------------------------------------------------------


def has_pending_units_work(
    facility: str,
    data_source_name: str,
) -> bool:
    """Check if any ingested versions need unit extraction."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility AND v.data_source_name = $data_source_name
              AND v.status = 'ingested'
              AND (v.units_extracted IS NULL OR v.units_extracted = false)
            RETURN count(v) AS cnt
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        return result[0]["cnt"] > 0 if result else False


def mark_version_units_extracted(
    version_id: str,
    units_count: int,
) -> None:
    """Mark a SignalEpoch as having completed unit extraction."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch {id: $id})
            SET v.units_extracted = true,
                v.units_count = $units_count
            """,
            id=version_id,
            units_count=units_count,
        )


def mark_all_versions_units_extracted(
    facility: str,
    data_source_name: str,
    units_count: int,
) -> None:
    """Mark all ingested SignalEpochs as having completed unit extraction.

    Units are per-tree (shared nodes), not per-version. Marking all versions
    ensures idempotency so re-runs skip units extraction entirely.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility AND v.data_source_name = $data_source_name
              AND v.status = 'ingested'
            SET v.units_extracted = true,
                v.units_count = $units_count
            """,
            facility=facility,
            data_source_name=data_source_name,
            units_count=units_count,
        )


def merge_units_to_graph(
    facility: str,
    data_source_name: str,
    units: dict[str, str],
) -> int:
    """MERGE Unit nodes and create HAS_UNIT relationships from DataNodes.

    Creates or reuses existing Unit nodes (keyed by normalized symbol),
    then creates HAS_UNIT relationships from matching SignalNode nodes.
    Also sets the ``unit`` string property on each SignalNode for
    efficient property-level filtering.

    Unit symbols are normalized via pint to prevent duplicate Unit nodes
    with equivalent formats (e.g., ``Ohm`` vs ``ohm``, ``m.s^-1`` vs ``m/s``).

    Args:
        facility: Facility identifier
        data_source_name: MDSplus tree name
        units: Mapping of normalized node IDs to unit symbol strings

    Returns:
        Number of HAS_UNIT relationships created.
    """
    from imas_codex.mdsplus.ingestion import normalize_mdsplus_path
    from imas_codex.units import normalize_unit_symbol

    if not units:
        return 0

    updates = []
    for path, unit_str in units.items():
        normalized_path = normalize_mdsplus_path(path)
        node_id = f"{facility}:{data_source_name}:{normalized_path}"
        symbol = normalize_unit_symbol(unit_str) or unit_str
        updates.append({"id": node_id, "symbol": symbol})

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS u
            MATCH (n:SignalNode {id: u.id})
            MERGE (unit:Unit {id: u.symbol})
            SET unit.symbol = u.symbol
            MERGE (n)-[:HAS_UNIT]->(unit)
            SET n.unit = u.symbol
            RETURN count(*) AS created
            """,
            updates=updates,
        )
    return result[0]["created"] if result else 0


# ---------------------------------------------------------------------------
# Enrichment context — tree hierarchy for LLM prompts
# ---------------------------------------------------------------------------


def fetch_enrichment_context(
    facility: str,
    data_source_name: str,
    node_paths: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch tree hierarchy context for nodes being enriched.

    For each node, queries the graph for its parent STRUCTURE node
    and sibling value nodes. This context is injected into the LLM
    prompt so descriptions are informed by the surrounding tree.

    Args:
        facility: Facility identifier
        data_source_name: Static tree name
        node_paths: Paths of nodes being enriched

    Returns:
        Dict mapping node path to context dict with:
        - parent_path: Parent STRUCTURE node path
        - parent_tags: Tags on the parent node
        - siblings: List of sibling dicts (path, node_type, tags, units)
    """
    if not node_paths:
        return {}

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $paths AS data_source_path
            MATCH (n:SignalNode {path: data_source_path, facility_id: $facility})
            WHERE n.data_source_name = $data_source_name
            OPTIONAL MATCH (parent:SignalNode {
                path: n.parent_path, facility_id: $facility
            })
            WHERE parent.data_source_name = $data_source_name
            OPTIONAL MATCH (parent)-[:HAS_NODE]->(sibling:SignalNode)
            WHERE sibling.data_source_name = $data_source_name
              AND sibling.path <> n.path
              AND sibling.node_type IN $node_types
            WITH n, parent, collect(DISTINCT {
                path: sibling.path,
                node_type: sibling.node_type,
                tags: sibling.tags,
                units: sibling.unit
            })[0..20] AS siblings
            RETURN n.path AS path,
                   parent.path AS parent_path,
                   parent.tags AS parent_tags,
                   parent.node_type AS parent_type,
                   siblings
            """,
            paths=node_paths,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )
        context: dict[str, dict[str, Any]] = {}
        for r in result:
            # Filter out null-path siblings from OPTIONAL MATCH
            siblings = [s for s in (r["siblings"] or []) if s.get("path") is not None]
            context[r["path"]] = {
                "parent_path": r["parent_path"],
                "parent_tags": r["parent_tags"],
                "parent_type": r["parent_type"],
                "siblings": siblings,
            }
        return context


# ---------------------------------------------------------------------------
# Pattern detection — indexed parameter groups
# ---------------------------------------------------------------------------

# Minimum indexed instances to form a pattern (avoid trivial 2-element groups)
MIN_PATTERN_INSTANCES = 3


def detect_and_create_signal_sources(
    facility: str,
    data_source_name: str,
    min_instances: int = MIN_PATTERN_INSTANCES,
) -> int:
    """Detect indexed parameter groups and create SignalSource nodes.

    Scans the graph for grandparent STRUCTURE nodes whose children (also
    STRUCTURE) each contain data-bearing leaves with the same name.
    For each (grandparent, leaf_name) combination with enough instances,
    creates a SignalSource and MEMBER_OF relationships.

    Example: TOP.W has children W001-W830, each with leaf R.
    Pattern: grandparent=TOP.W, leaf=R, index_count=830.

    Args:
        facility: Facility identifier
        data_source_name: Static tree name
        min_instances: Minimum indexed parents to qualify as a pattern

    Returns:
        Number of patterns created
    """
    with GraphClient() as gc:
        # Find (grandparent, leaf_name) groups with enough indexed parents
        groups = gc.query(
            """
            MATCH (gp:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE gp.node_type = 'STRUCTURE'
            MATCH (gp)-[:HAS_NODE]->(parent:SignalNode)
            WHERE parent.node_type = 'STRUCTURE'
            MATCH (parent)-[:HAS_NODE]->(leaf:SignalNode)
            WHERE leaf.node_type IN $node_types
            WITH gp.path AS gp_path,
                 split(leaf.path, '.')[-1] AS leaf_name,
                 head(collect(leaf.node_type)) AS leaf_type,
                 count(DISTINCT parent) AS parent_count,
                 head(collect(leaf.path)) AS representative
            WHERE parent_count >= $min_instances
            RETURN gp_path, leaf_name, leaf_type, parent_count, representative
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
            min_instances=min_instances,
        )

        if not groups:
            logger.info(
                "No indexed patterns found for %s:%s", facility, data_source_name
            )
            return 0

        # Create SignalSource nodes and MEMBER_OF relationships
        patterns = []
        for g in groups:
            pattern_id = (
                f"{facility}:{data_source_name}:{g['gp_path']}:{g['leaf_name']}"
            )
            patterns.append(
                {
                    "id": pattern_id,
                    "facility_id": facility,
                    "data_source_name": data_source_name,
                    "grandparent_path": g["gp_path"],
                    "leaf_name": g["leaf_name"],
                    "index_count": g["parent_count"],
                    "node_type": g["leaf_type"],
                    "representative_path": g["representative"],
                }
            )

        # Merge patterns and link followers
        gc.ensure_facility(facility)
        result = gc.query(
            """
            UNWIND $patterns AS p
            MERGE (pat:SignalSource {id: p.id})
            ON CREATE SET
                pat.facility_id = p.facility_id,
                pat.data_source_name = p.data_source_name,
                pat.grandparent_path = p.grandparent_path,
                pat.leaf_name = p.leaf_name,
                pat.index_count = p.index_count,
                pat.node_type = p.node_type,
                pat.representative_path = p.representative_path
            WITH pat, p
            MATCH (f:Facility {id: p.facility_id})
            MERGE (pat)-[:AT_FACILITY]->(f)
            WITH pat, p
            // Link all matching leaf nodes to this group
            MATCH (gp:SignalNode {path: p.grandparent_path, facility_id: p.facility_id})
            WHERE gp.data_source_name = p.data_source_name
            MATCH (gp)-[:HAS_NODE]->(parent:SignalNode)-[:HAS_NODE]->(leaf:SignalNode)
            WHERE leaf.node_type IN $node_types
              AND split(leaf.path, '.')[-1] = p.leaf_name
            MERGE (leaf)-[:MEMBER_OF]->(pat)
            RETURN pat.id AS id, count(leaf) AS linked
            """,
            patterns=patterns,
            node_types=ENRICHABLE_NODE_TYPES,
        )

        created = len(result) if result else 0
        total_linked = sum(r["linked"] for r in result) if result else 0
        logger.info(
            "Created %d patterns for %s:%s (%d nodes linked)",
            created,
            facility,
            data_source_name,
            total_linked,
        )
        return created


def detect_and_create_member_signal_sources(
    facility: str,
    data_source_name: str,
    member_parent_types: list[str] | None = None,
    min_instances: int = MIN_PATTERN_INSTANCES,
) -> int:
    """Detect member-suffix patterns and create SignalSource nodes.

    MDSplus nodes often have member sub-nodes (colon-separated, e.g.
    `:PRE`, `:VAL`, `:STORE`) that share identical semantic meaning
    across thousands of parent nodes. This detects groups of leaf nodes
    whose name (after the last `:` separator) is identical and whose
    parent has one of the configured ``member_parent_types``.

    Which parent node types to look for is facility-specific and
    configured via ``member_parent_types`` in the static tree config
    YAML (e.g. ``["SIGNAL"]`` for TCV). If *None* or empty, this
    function is a no-op.

    Unlike `detect_and_create_signal_sources` (which groups by grandparent →
    indexed-parent → leaf), this groups globally by (leaf_name, parent
    node_type) so a single group covers all instances.

    Args:
        facility: Facility identifier
        data_source_name: Static tree name
        member_parent_types: Parent node types to scan for member children.
            Loaded from facility YAML ``trees[].member_parent_types``.
        min_instances: Minimum instances to qualify

    Returns:
        Number of patterns created
    """
    if not member_parent_types:
        return 0

    with GraphClient() as gc:
        # Find member leaf names that repeat across many parents of
        # the configured types
        groups = gc.query(
            """
            MATCH (parent:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE parent.node_type IN $parent_types
            MATCH (parent)-[:HAS_NODE]->(leaf:SignalNode)
            WHERE leaf.node_type IN $node_types
              AND NOT EXISTS { (leaf)-[:MEMBER_OF]->(:SignalSource) }
            WITH split(leaf.path, ':')[-1] AS leaf_name,
                 parent.node_type AS parent_type,
                 count(leaf) AS instance_count,
                 head(collect(leaf.path)) AS representative,
                 head(collect(leaf.node_type)) AS leaf_type
            WHERE instance_count >= $min_instances
            RETURN leaf_name, parent_type, instance_count, representative, leaf_type
            """,
            facility=facility,
            data_source_name=data_source_name,
            parent_types=member_parent_types,
            node_types=ENRICHABLE_NODE_TYPES,
            min_instances=min_instances,
        )

        if not groups:
            logger.info(
                "No member-suffix patterns found for %s:%s", facility, data_source_name
            )
            return 0

        # Create groups — keyed by facility:tree:member:parent_type:leaf_name
        patterns = []
        for g in groups:
            pattern_id = f"{facility}:{data_source_name}:member:{g['parent_type']}:{g['leaf_name']}"
            patterns.append(
                {
                    "id": pattern_id,
                    "facility_id": facility,
                    "data_source_name": data_source_name,
                    "grandparent_path": g["parent_type"],  # parent type, not a path
                    "leaf_name": g["leaf_name"],
                    "index_count": g["instance_count"],
                    "node_type": g["leaf_type"],
                    "representative_path": g["representative"],
                }
            )

        gc.ensure_facility(facility)
        result = gc.query(
            """
            UNWIND $patterns AS p
            MERGE (pat:SignalSource {id: p.id})
            ON CREATE SET
                pat.facility_id = p.facility_id,
                pat.data_source_name = p.data_source_name,
                pat.grandparent_path = p.grandparent_path,
                pat.leaf_name = p.leaf_name,
                pat.index_count = p.index_count,
                pat.node_type = p.node_type,
                pat.representative_path = p.representative_path
            WITH pat, p
            MATCH (f:Facility {id: p.facility_id})
            MERGE (pat)-[:AT_FACILITY]->(f)
            WITH pat, p
            // Link all matching leaf nodes under parents of the configured type
            MATCH (parent:SignalNode {facility_id: p.facility_id})
            WHERE parent.data_source_name = p.data_source_name
              AND parent.node_type = p.grandparent_path
            MATCH (parent)-[:HAS_NODE]->(leaf:SignalNode)
            WHERE leaf.node_type IN $node_types
              AND split(leaf.path, ':')[-1] = p.leaf_name
            MERGE (leaf)-[:MEMBER_OF]->(pat)
            RETURN pat.id AS id, count(leaf) AS linked
            """,
            patterns=patterns,
            node_types=ENRICHABLE_NODE_TYPES,
        )

        created = len(result) if result else 0
        total_linked = sum(r["linked"] for r in result) if result else 0
        logger.info(
            "Created %d member-suffix signal sources for %s:%s (%d nodes linked)",
            created,
            facility,
            data_source_name,
            total_linked,
        )
        return created


# ---------------------------------------------------------------------------
# Enrichment claiming — SignalSource (group-first enrichment)
# ---------------------------------------------------------------------------


@retry_on_deadlock()
def claim_signal_sources(
    facility: str,
    data_source_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Claim unenriched SignalSources for LLM enrichment.

    Returns group info plus one representative node's details for the
    LLM prompt. After enriching, call mark_signal_sources_enriched() to
    propagate descriptions to all members.

    Uses claim_token + ORDER BY rand() to prevent deadlocks.

    Returns:
        List of dicts with group id, grandparent_path, leaf_name,
        index_count, and representative node details.
    """
    claim_token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:SignalSource)
            WHERE p.facility_id = $facility
              AND p.data_source_name = $data_source_name
              AND (p.description IS NULL OR p.description = '')
              AND p.claimed_at IS NULL
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            data_source_name=data_source_name,
            limit=limit,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (p:SignalSource {claim_token: $token})
            OPTIONAL MATCH (rep:SignalNode {path: p.representative_path,
                                          facility_id: $facility})
            RETURN p.id AS id,
                   p.grandparent_path AS grandparent_path,
                   p.leaf_name AS leaf_name,
                   p.index_count AS index_count,
                   p.node_type AS node_type,
                   p.representative_path AS representative_path,
                   rep.tags AS tags,
                   rep.unit AS unit,
                   rep.parent_path AS parent_path
            """,
            facility=facility,
            token=claim_token,
        )
        return [dict(r) for r in result]


def mark_signal_sources_enriched(
    pattern_ids: list[str],
    descriptions: dict[str, str],
    metadata: dict[str, dict] | None = None,
    *,
    llm_cost: float = 0.0,
    llm_model: str | None = None,
) -> int:
    """Mark signal sources as enriched and propagate to all members.

    Sets description/keywords/category on the SignalSource and
    copies them to every SignalNode linked via MEMBER_OF.
    Distributes llm_cost evenly across all member nodes.

    Returns:
        Number of SignalNodes updated (members).
    """
    updates = []
    for pid in pattern_ids:
        update: dict[str, Any] = {
            "id": pid,
            "description": descriptions.get(pid, ""),
        }
        if metadata and pid in metadata:
            m = metadata[pid]
            if m.get("keywords"):
                update["keywords"] = m["keywords"]
            if m.get("category"):
                update["category"] = m["category"]
        updates.append(update)

    if not updates:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS u
            MATCH (p:SignalSource {id: u.id})
            SET p.description = u.description,
                p.keywords = u.keywords,
                p.category = u.category,
                p.enrichment_status = 'enriched',
                p.claimed_at = null
            WITH p, u
            // Propagate to all members
            MATCH (n:SignalNode)-[:MEMBER_OF]->(p)
            SET n.description = u.description,
                n.keywords = u.keywords,
                n.category = u.category,
                n.enrichment_status = 'enriched',
                n.claimed_at = null
            RETURN p.id AS pattern_id, count(n) AS propagated
            """,
            updates=updates,
        )
        total_propagated = sum(r["propagated"] for r in result) if result else 0

        # Write per-node cost to all enriched nodes
        if llm_cost > 0 and total_propagated > 0:
            per_node_cost = llm_cost / total_propagated
            enriched_pattern_ids = [u["id"] for u in updates if u["description"]]
            if enriched_pattern_ids:
                gc.query(
                    """
                    UNWIND $ids AS pid
                    MATCH (n:SignalNode)-[:MEMBER_OF]->(:SignalSource {id: pid})
                    WHERE n.enrichment_status = 'enriched'
                    SET n.llm_cost = $per_node_cost,
                        n.llm_model = $llm_model,
                        n.llm_at = datetime()
                    """,
                    ids=enriched_pattern_ids,
                    per_node_cost=per_node_cost,
                    llm_model=llm_model,
                )

        return total_propagated


def release_signal_source_claims(pattern_ids: list[str]) -> int:
    """Release claims on SignalSources (on error)."""
    if not pattern_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:SignalSource {id: pid})
            SET p.claimed_at = null
            RETURN count(p) AS released
            """,
            ids=pattern_ids,
        )
        return result[0]["released"] if result else 0


def has_pending_signal_source_work(facility: str, data_source_name: str) -> bool:
    """Check if any SignalSources need enrichment."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:SignalSource)
            WHERE p.facility_id = $facility
              AND p.data_source_name = $data_source_name
              AND (p.description IS NULL OR p.description = '')
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        return result[0]["has_work"] if result else False


# ---------------------------------------------------------------------------
# Enrichment claiming — SignalNode nodes (non-pattern nodes only)
# ---------------------------------------------------------------------------


# Enrichable node types — only data-bearing nodes, not structural.
# AXIS nodes define array indices, TEXT nodes hold labels/comments;
# neither contains physics quantities worth describing.
ENRICHABLE_NODE_TYPES = ["NUMERIC", "SIGNAL"]


@retry_on_deadlock()
def claim_parent_for_enrichment(
    facility: str,
    data_source_name: str,
) -> dict[str, Any] | None:
    """Claim a parent node whose children need enrichment.

    Finds any SignalNode with un-enriched NUMERIC/SIGNAL children
    that don't follow a SignalSource. Claims the parent by setting
    claimed_at. Uses claim_token + ORDER BY rand() to prevent deadlocks.

    Returns:
        Dict with parent_id, parent_path, parent_tags, and children list,
        or None if no work available.
    """
    claim_token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (parent:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE parent.claimed_at IS NULL
            WITH parent
            MATCH (parent)-[:HAS_NODE]->(child:SignalNode)
            WHERE child.node_type IN $node_types
              AND (child.description IS NULL OR child.description = '')
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
            WITH parent, count(child) AS unenriched
            WHERE unenriched > 0
            WITH parent ORDER BY rand() LIMIT 1
            SET parent.claimed_at = datetime(), parent.claim_token = $token
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (parent:SignalNode {claim_token: $token})
            MATCH (parent)-[:HAS_NODE]->(child:SignalNode)
            WHERE child.node_type IN $node_types
              AND (child.description IS NULL OR child.description = '')
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
            RETURN parent.id AS parent_id, parent.path AS parent_path,
                   parent.tags AS parent_tags,
                   collect({
                       id: child.id, path: child.path,
                       node_type: child.node_type,
                       tags: child.tags, unit: child.unit
                   }) AS children
            """,
            node_types=ENRICHABLE_NODE_TYPES,
            token=claim_token,
        )
        if result:
            return dict(result[0])
        return None


@retry_on_deadlock()
def claim_orphan_nodes_for_enrichment(
    facility: str,
    data_source_name: str,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Claim enrichable nodes that have no parent HAS_NODE relationship.

    These are top-level tree nodes not nested under any parent.
    Uses claim_token + ORDER BY rand() to prevent deadlocks.

    Returns:
        List of node dicts with id, path, node_type, tags, units.
    """
    claim_token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (child:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE child.node_type IN $node_types
              AND (child.description IS NULL OR child.description = '')
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
              AND NOT EXISTS { (:SignalNode)-[:HAS_NODE]->(child) }
              AND child.claimed_at IS NULL
            WITH child ORDER BY rand() LIMIT $limit
            SET child.claimed_at = datetime(), child.claim_token = $token
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
            limit=limit,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (child:SignalNode {claim_token: $token})
            RETURN child.id AS id, child.path AS path,
                   child.node_type AS node_type,
                   child.tags AS tags, child.unit AS unit
            """,
            token=claim_token,
        )
        return [dict(r) for r in result] if result else []


def mark_parent_children_enriched(
    parent_id: str,
    descriptions: dict[str, str],
    metadata: dict[str, dict] | None = None,
    *,
    llm_cost: float = 0.0,
    llm_model: str | None = None,
) -> int:
    """Mark all children as enriched and release parent claim.

    Distributes llm_cost evenly across enriched children.

    Args:
        parent_id: ID of the parent STRUCTURE node (to release claim)
        descriptions: Map of child node ID to description
        metadata: Optional map of child node ID to keywords/category
        llm_cost: Total LLM cost for this batch
        llm_model: Model identifier used

    Returns:
        Number of children enriched.
    """
    updates = []
    for nid, desc in descriptions.items():
        update: dict[str, Any] = {
            "id": nid,
            "description": desc,
        }
        if metadata and nid in metadata:
            m = metadata[nid]
            if m.get("keywords"):
                update["keywords"] = m["keywords"]
            if m.get("category"):
                update["category"] = m["category"]
        updates.append(update)

    if not updates:
        # Release parent claim even if no enrichments
        with GraphClient() as gc:
            gc.query(
                "MATCH (n:SignalNode {id: $id}) SET n.claimed_at = null",
                id=parent_id,
            )
        return 0

    per_node_cost = llm_cost / len(updates) if llm_cost > 0 else 0.0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS u
            MATCH (n:SignalNode {id: u.id})
            SET n.description = u.description,
                n.keywords = u.keywords,
                n.category = u.category,
                n.enrichment_status = 'enriched',
                n.claimed_at = null,
                n.llm_cost = $per_node_cost,
                n.llm_model = $llm_model,
                n.llm_at = datetime()
            RETURN count(n) AS updated
            """,
            updates=updates,
            per_node_cost=per_node_cost,
            llm_model=llm_model,
        )
        # Release parent claim
        gc.query(
            "MATCH (n:SignalNode {id: $id}) SET n.claimed_at = null",
            id=parent_id,
        )
        return result[0]["updated"] if result else 0


def release_parent_claim(parent_id: str) -> None:
    """Release claim on a parent node (on error)."""
    with GraphClient() as gc:
        gc.query(
            "MATCH (n:SignalNode {id: $id}) SET n.claimed_at = null",
            id=parent_id,
        )


def release_orphan_claims(node_ids: list[str]) -> None:
    """Release claims on orphan nodes (on error)."""
    if not node_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS nid
            MATCH (n:SignalNode {id: nid})
            SET n.claimed_at = null
            """,
            ids=node_ids,
        )


def mark_orphan_nodes_enriched(
    node_ids: list[str],
    descriptions: dict[str, str],
    metadata: dict[str, dict] | None = None,
    *,
    llm_cost: float = 0.0,
    llm_model: str | None = None,
) -> int:
    """Mark orphan nodes as enriched and release claims.

    Distributes llm_cost evenly across enriched nodes.

    Args:
        node_ids: IDs of all claimed nodes (to release claims)
        descriptions: Map of node ID to description
        metadata: Optional map of node ID to keywords/category
        llm_cost: Total LLM cost for this batch
        llm_model: Model identifier used

    Returns:
        Number of nodes enriched.
    """
    updates = []
    for nid, desc in descriptions.items():
        update: dict[str, Any] = {
            "id": nid,
            "description": desc,
        }
        if metadata and nid in metadata:
            m = metadata[nid]
            if m.get("keywords"):
                update["keywords"] = m["keywords"]
            if m.get("category"):
                update["category"] = m["category"]
        updates.append(update)

    per_node_cost = llm_cost / len(updates) if llm_cost > 0 and updates else 0.0

    with GraphClient() as gc:
        enriched = 0
        if updates:
            result = gc.query(
                """
                UNWIND $updates AS u
                MATCH (n:SignalNode {id: u.id})
                SET n.description = u.description,
                    n.keywords = u.keywords,
                    n.category = u.category,
                    n.enrichment_status = 'enriched',
                    n.claimed_at = null,
                    n.llm_cost = $per_node_cost,
                    n.llm_model = $llm_model,
                    n.llm_at = datetime()
                RETURN count(n) AS updated
                """,
                updates=updates,
                per_node_cost=per_node_cost,
                llm_model=llm_model,
            )
            enriched = result[0]["updated"] if result else 0

        # Release claims on any nodes that weren't enriched
        enriched_ids = {u["id"] for u in updates}
        remaining = [nid for nid in node_ids if nid not in enriched_ids]
        if remaining:
            gc.query(
                """
                UNWIND $ids AS nid
                MATCH (n:SignalNode {id: nid})
                SET n.claimed_at = null
                """,
                ids=remaining,
            )
        return enriched


# ---------------------------------------------------------------------------
# Has-work queries (for PipelinePhase completion detection)
# ---------------------------------------------------------------------------


def has_pending_extract_work(facility: str, data_source_name: str) -> bool:
    """Check if any SignalEpochs need extraction (discovered or claimed)."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility
              AND v.data_source_name = $data_source_name
              AND v.status = 'discovered'
            RETURN count(v) > 0 AS has_work
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        return result[0]["has_work"] if result else False


def has_pending_enrich_work(facility: str, data_source_name: str) -> bool:
    """Check if any signal sources, parent groups, or orphan nodes need enrichment."""
    if has_pending_signal_source_work(facility, data_source_name):
        return True
    with GraphClient() as gc:
        # Check for parents with unenriched children (any parent type)
        result = gc.query(
            """
            MATCH (parent:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WITH parent
            MATCH (parent)-[:HAS_NODE]->(child:SignalNode)
            WHERE child.node_type IN $node_types
              AND (child.description IS NULL OR child.description = '')
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
            RETURN count(DISTINCT parent) > 0 AS has_work
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )
        if result and result[0]["has_work"]:
            return True

        # Check for orphan nodes (no parent HAS_NODE relationship)
        orphan_result = gc.query(
            """
            MATCH (child:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE child.node_type IN $node_types
              AND (child.description IS NULL OR child.description = '')
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
              AND NOT EXISTS { (:SignalNode)-[:HAS_NODE]->(child) }
            RETURN count(child) > 0 AS has_work
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )
        return orphan_result[0]["has_work"] if orphan_result else False


def has_pending_ingest_work(facility: str, data_source_name: str) -> bool:
    """Check if there are extracted versions not yet fully ingested.

    This is always False since we ingest inline after extraction.
    Kept for PipelinePhase interface consistency.
    """
    return False


# ---------------------------------------------------------------------------
# Startup — orphan recovery
# ---------------------------------------------------------------------------


def reset_orphaned_static_claims(
    facility: str,
    data_source_name: str,
    *,
    silent: bool = False,
) -> dict[str, int]:
    """Release stale claims for static tree discovery."""
    version_reset = reset_stale_claims(
        "SignalEpoch",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )
    node_reset = reset_stale_claims(
        "SignalNode",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )
    pattern_reset = reset_stale_claims(
        "SignalSource",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )
    total = version_reset + node_reset + pattern_reset
    if total and not silent:
        logger.info(
            "Released %d orphaned static claims (%d versions, %d nodes, %d patterns)",
            total,
            version_reset,
            node_reset,
            pattern_reset,
        )
    return {
        "version_reset": version_reset,
        "node_reset": node_reset,
        "pattern_reset": pattern_reset,
    }


# ---------------------------------------------------------------------------
# Stats query (for progress display refresh)
# ---------------------------------------------------------------------------


def get_static_discovery_stats(
    facility: str,
    data_source_name: str,
) -> dict[str, int | float]:
    """Get static discovery statistics from graph for progress display."""
    with GraphClient() as gc:
        # Version status counts
        ver_result = gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility AND v.data_source_name = $data_source_name
            RETURN v.status AS status, count(v) AS cnt,
                   sum(coalesce(v.node_count, 0)) AS nodes
            """,
            facility=facility,
            data_source_name=data_source_name,
        )

        stats: dict[str, int | float] = {
            "versions_total": 0,
            "versions_discovered": 0,
            "versions_ingested": 0,
            "versions_claimed": 0,
            "nodes_total": 0,
        }

        for r in ver_result:
            cnt = r["cnt"]
            status = r["status"]
            stats["versions_total"] += cnt
            if status == "discovered":
                stats["versions_discovered"] += cnt
            elif status == "ingested":
                stats["versions_ingested"] += cnt
                stats["nodes_total"] += r["nodes"]

        # Claimed versions (in-progress extraction)
        claimed = gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility AND v.data_source_name = $data_source_name
              AND v.claimed_at IS NOT NULL
            RETURN count(v) AS cnt
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        stats["versions_claimed"] = claimed[0]["cnt"] if claimed else 0

        # Units extraction stats from SignalEpoch flags
        units_result = gc.query(
            """
            MATCH (v:SignalEpoch)
            WHERE v.facility_id = $facility AND v.data_source_name = $data_source_name
              AND v.status = 'ingested'
            RETURN
                count(v) AS total,
                sum(CASE WHEN v.units_extracted = true
                    THEN 1 ELSE 0 END) AS extracted,
                sum(coalesce(v.units_count, 0)) AS units_count
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        if units_result:
            ur = units_result[0]
            stats["units_versions_total"] = ur["total"]
            stats["units_versions_extracted"] = ur["extracted"]
            stats["units_count"] = ur["units_count"]
        else:
            stats["units_versions_total"] = 0
            stats["units_versions_extracted"] = 0
            stats["units_count"] = 0

        # Also count actual HAS_UNIT relationships for display
        hu_result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN
                count(DISTINCT n) AS nodes_total,
                count(DISTINCT u) AS unique_units,
                count(u) AS nodes_with_units
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        if hu_result:
            stats["nodes_with_units"] = hu_result[0]["nodes_with_units"]
            stats["unique_units"] = hu_result[0]["unique_units"]
        else:
            stats["nodes_with_units"] = 0
            stats["unique_units"] = 0

        # Node enrichment stats — count parent groups, not individual nodes
        node_result = gc.query(
            """
            MATCH (n:SignalNode)
            WHERE n.facility_id = $facility AND n.data_source_name = $data_source_name
            RETURN
                count(n) AS total,
                sum(CASE WHEN n.description IS NOT NULL AND n.description <> ''
                    THEN 1 ELSE 0 END) AS enriched,
                sum(CASE WHEN n.node_type IN $node_types
                    THEN 1 ELSE 0 END) AS enrichable
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )

        if node_result:
            r = node_result[0]
            stats["nodes_graph"] = r["total"]
            stats["nodes_enriched"] = r["enriched"]
            stats["nodes_enrichable"] = r["enrichable"]
        else:
            stats["nodes_graph"] = 0
            stats["nodes_enriched"] = 0
            stats["nodes_enrichable"] = 0

        # Parent groups pending enrichment (non-pattern work units, any parent type)
        parent_result = gc.query(
            """
            MATCH (parent:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WITH parent
            MATCH (parent)-[:HAS_NODE]->(child:SignalNode)
            WHERE child.node_type IN $node_types
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
            WITH parent,
                 count(child) AS total_children,
                 sum(CASE WHEN child.description IS NOT NULL
                         AND child.description <> ''
                    THEN 1 ELSE 0 END) AS enriched_children
            WHERE total_children > 0
            RETURN count(parent) AS total_parents,
                   sum(CASE WHEN enriched_children < total_children
                       THEN 1 ELSE 0 END) AS pending_parents,
                   sum(CASE WHEN enriched_children >= total_children
                       THEN 1 ELSE 0 END) AS enriched_parents
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )
        if parent_result:
            pr = parent_result[0]
            stats["parent_groups_total"] = pr["total_parents"]
            stats["parent_groups_pending"] = pr["pending_parents"]
            stats["parent_groups_enriched"] = pr["enriched_parents"]
        else:
            stats["parent_groups_total"] = 0
            stats["parent_groups_pending"] = 0
            stats["parent_groups_enriched"] = 0

        # Orphan nodes (no parent HAS_NODE relationship)
        orphan_result = gc.query(
            """
            MATCH (child:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE child.node_type IN $node_types
              AND NOT EXISTS { (child)-[:MEMBER_OF]->(:SignalSource) }
              AND NOT EXISTS { (:SignalNode)-[:HAS_NODE]->(child) }
            RETURN count(child) AS total_orphans,
                   sum(CASE WHEN child.description IS NOT NULL
                            AND child.description <> ''
                       THEN 1 ELSE 0 END) AS enriched_orphans
            """,
            facility=facility,
            data_source_name=data_source_name,
            node_types=ENRICHABLE_NODE_TYPES,
        )
        if orphan_result:
            orph = orphan_result[0]
            stats["orphan_nodes_total"] = orph["total_orphans"]
            stats["orphan_nodes_enriched"] = orph["enriched_orphans"]
        else:
            stats["orphan_nodes_total"] = 0
            stats["orphan_nodes_enriched"] = 0

        # Pattern stats
        pattern_result = gc.query(
            """
            MATCH (p:SignalSource)
            WHERE p.facility_id = $facility AND p.data_source_name = $data_source_name
            RETURN
                count(p) AS total,
                sum(CASE WHEN p.description IS NOT NULL AND p.description <> ''
                    THEN 1 ELSE 0 END) AS enriched,
                sum(CASE WHEN p.description IS NULL OR p.description = ''
                    THEN 1 ELSE 0 END) AS pending
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        if pattern_result:
            pr = pattern_result[0]
            stats["patterns_total"] = pr["total"]
            stats["patterns_enriched"] = pr["enriched"]
            stats["pending_patterns"] = pr["pending"]
        else:
            stats["patterns_total"] = 0
            stats["patterns_enriched"] = 0
            stats["pending_patterns"] = 0

        # Accumulated LLM cost from per-node llm_cost fields
        cost_result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE n.llm_cost IS NOT NULL
            RETURN sum(n.llm_cost) AS total_cost
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        stats["accumulated_cost"] = (
            cost_result[0]["total_cost"] if cost_result else 0.0
        ) or 0.0

        return stats


# ---------------------------------------------------------------------------
# Reset enrichment — clear enrichment data so nodes can be re-enriched
# ---------------------------------------------------------------------------


def reset_enrichment(
    facility: str,
    data_source_name: str,
) -> dict[str, int]:
    """Reset all enrichment data for static tree nodes.

    Clears descriptions, enrichment status, LLM cost/model/timestamp,
    keywords, and category from all SignalNode and SignalSource nodes.
    After reset, the enrichment pipeline will re-process all nodes.

    Returns:
        Dict with counts: nodes_reset, patterns_reset
    """
    with GraphClient() as gc:
        # Reset SignalNode enrichment fields
        node_result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility, data_source_name: $data_source_name})
            WHERE (n.description IS NOT NULL OR n.enrichment_status IS NOT NULL
                   OR n.llm_cost IS NOT NULL)
            SET n.description = null,
                n.enrichment_status = null,
                n.enrichment_source = null,
                n.enrichment_confidence = null,
                n.enrichment_notes = null,
                n.keywords = null,
                n.category = null,
                n.llm_cost = null,
                n.llm_model = null,
                n.llm_at = null,
                n.llm_tokens_in = null,
                n.llm_tokens_out = null,
                n.enrichment_model = null,
                n.enrichment_at = null,
                n.claimed_at = null
            RETURN count(n) AS reset
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        nodes_reset = node_result[0]["reset"] if node_result else 0

        # Reset SignalSource enrichment fields
        pattern_result = gc.query(
            """
            MATCH (p:SignalSource {facility_id: $facility, data_source_name: $data_source_name})
            WHERE p.description IS NOT NULL OR p.enrichment_status IS NOT NULL
            SET p.description = null,
                p.enrichment_status = null,
                p.keywords = null,
                p.category = null,
                p.claimed_at = null
            RETURN count(p) AS reset
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        patterns_reset = pattern_result[0]["reset"] if pattern_result else 0

        return {
            "nodes_reset": nodes_reset,
            "patterns_reset": patterns_reset,
        }


# ---------------------------------------------------------------------------
# Clear — delete all static discovery data for a facility
# ---------------------------------------------------------------------------


def clear_facility_static(
    facility: str,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Clear all static tree discovery data for a facility.

    Deletes SignalNode nodes, SignalSource nodes,
    and their SignalEpoch nodes in batches.

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch

    Returns:
        Dict with counts: nodes_deleted, versions_deleted, patterns_deleted
    """
    results = {
        "nodes_deleted": 0,
        "versions_deleted": 0,
        "patterns_deleted": 0,
    }

    with GraphClient() as gc:
        # Delete SignalSource nodes first (they reference DataNodes)
        result = gc.query(
            """
            MATCH (p:SignalSource {facility_id: $facility})
            DETACH DELETE p
            RETURN count(p) AS deleted
            """,
            facility=facility,
        )
        results["patterns_deleted"] = result[0]["deleted"] if result else 0

        # Delete static SignalNode nodes in batches
        while True:
            result = gc.query(
                """
                MATCH (n:SignalNode {facility_id: $facility})
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["nodes_deleted"] += deleted
            if deleted < batch_size:
                break

        # Delete static SignalEpoch nodes
        while True:
            result = gc.query(
                """
                MATCH (v:SignalEpoch {facility_id: $facility})
                WHERE v.status IS NOT NULL
                WITH v LIMIT $batch_size
                DETACH DELETE v
                RETURN count(v) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["versions_deleted"] += deleted
            if deleted < batch_size:
                break

    logger.info(
        "Cleared static data for %s: %d nodes, %d patterns, %d versions",
        facility,
        results["nodes_deleted"],
        results["patterns_deleted"],
        results["versions_deleted"],
    )
    return results


# ---------------------------------------------------------------------------
# Summary stats — facility-level stats across all static trees
# ---------------------------------------------------------------------------


def get_static_summary_stats(facility: str) -> dict[str, int]:
    """Get static discovery statistics across all trees for a facility.

    Unlike get_static_discovery_stats which requires a data_source_name, this
    returns aggregate stats suitable for status/clear commands.
    """
    with GraphClient() as gc:
        ver_result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            WHERE v.status IS NOT NULL
            RETURN v.status AS status, count(v) AS cnt,
                   sum(coalesce(v.node_count, 0)) AS nodes
            """,
            facility=facility,
        )

        stats: dict[str, int] = {
            "versions_total": 0,
            "versions_discovered": 0,
            "versions_ingested": 0,
            "nodes_total": 0,
        }

        for r in ver_result:
            cnt = r["cnt"]
            status = r["status"]
            stats["versions_total"] += cnt
            if status == "discovered":
                stats["versions_discovered"] += cnt
            elif status == "ingested":
                stats["versions_ingested"] += cnt
                stats["nodes_total"] += r["nodes"]

        # Node enrichment stats
        node_result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility})
            RETURN
                count(n) AS total,
                sum(CASE WHEN n.description IS NOT NULL AND n.description <> ''
                    THEN 1 ELSE 0 END) AS enriched
            """,
            facility=facility,
        )

        if node_result:
            stats["nodes_graph"] = node_result[0]["total"]
            stats["nodes_enriched"] = node_result[0]["enriched"]
        else:
            stats["nodes_graph"] = 0
            stats["nodes_enriched"] = 0

        return stats


def promote_leaf_nodes_to_signals(
    facility: str,
    data_source_name: str,
    batch_size: int = 500,
) -> int:
    """Create FacilitySignal nodes from leaf DataNodes (NUMERIC/SIGNAL).

    Promotes leaf DataNodes to FacilitySignals with status=discovered.
    Signals get no description at this stage — that comes from the
    signals enrichment pipeline later.

    Args:
        facility: Facility identifier
        data_source_name: MDSplus tree name
        batch_size: Number of nodes to process per transaction

    Returns:
        Number of FacilitySignal nodes created or updated.
    """
    da_id = f"{facility}:mdsplus:tree_tdi"
    conn_tpl = (
        f"import MDSplus\n"
        f"tree = MDSplus.Tree('{data_source_name}', {{shot}}, 'readonly')"
    )
    data_tpl = "data = tree.getNode('{data_source_path}').data()"

    with GraphClient() as gc:
        # Ensure DataAccess node exists (only if Facility exists)
        gc.query(
            """
            MATCH (f:Facility {id: $facility})
            MERGE (da:DataAccess {id: $id})
            ON CREATE SET
                da.facility_id = $facility,
                da.method_type = 'mdsplus',
                da.library = 'MDSplus',
                da.access_type = 'local',
                da.data_source = 'mdsplus',
                da.name = 'MDSplus tree TDI',
                da.connection_template = $conn_tpl,
                da.data_template = $data_tpl
            MERGE (da)-[:AT_FACILITY]->(f)
            """,
            id=da_id,
            facility=facility,
            conn_tpl=conn_tpl,
            data_tpl=data_tpl,
        )

        # Count promotable leaf nodes
        count_result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility, data_source_name: $tree})
            WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
              AND NOT EXISTS {
                  MATCH (n)<-[:HAS_DATA_SOURCE_NODE]-(:FacilitySignal)
              }
            RETURN count(n) AS total
            """,
            facility=facility,
            tree=data_source_name,
        )
        total = count_result[0]["total"] if count_result else 0
        if total == 0:
            logger.info(
                "No new leaf nodes to promote for %s:%s", facility, data_source_name
            )
            return 0

        logger.info(
            "Promoting %d leaf DataNodes to FacilitySignals for %s:%s",
            total,
            facility,
            data_source_name,
        )

        promoted = 0
        for offset in range(0, total, batch_size):
            batch_limit = min(batch_size, total - offset)
            result = gc.query(
                """
                MATCH (n:SignalNode {facility_id: $facility,
                                   data_source_name: $tree})
                WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
                  AND NOT EXISTS {
                      MATCH (n)<-[:HAS_DATA_SOURCE_NODE]-(:FacilitySignal)
                  }
                WITH n ORDER BY n.id LIMIT $batch_limit
                WITH n,
                     split(n.path, '::')[-1] AS node_path_raw
                WITH n, node_path_raw,
                     $facility + ':' +
                     $tree + '/' +
                     toLower(replace(node_path_raw, ':', '/'))
                     AS sig_id
                MERGE (s:FacilitySignal {id: sig_id})
                ON CREATE SET
                    s.facility_id = $facility,
                    s.status = 'discovered',
                    s.accessor = node_path_raw,
                    s.data_access = $da_id,
                    s.data_source_name = $tree,
                    s.data_source_path = n.path,
                    s.unit = n.unit,
                    s.discovery_source = 'tree_traversal',
                    s.data_source_node = n.id,
                    s.discovered_at = datetime()
                WITH s, n
                MERGE (s)-[:HAS_DATA_SOURCE_NODE]->(n)
                WITH s
                MATCH (f:Facility {id: $facility})
                MERGE (s)-[:AT_FACILITY]->(f)
                WITH s
                MATCH (da:DataAccess {id: $da_id})
                MERGE (s)-[:DATA_ACCESS]->(da)
                RETURN count(s) AS promoted
                """,
                facility=facility,
                tree=data_source_name,
                da_id=da_id,
                batch_limit=batch_limit,
            )
            batch_count = result[0]["promoted"] if result else 0
            promoted += batch_count
            logger.debug(
                "Promote batch %d: %d signals (%d total)",
                offset // batch_size + 1,
                batch_count,
                promoted,
            )

        logger.info(
            "Promoted %d FacilitySignals for %s:%s",
            promoted,
            facility,
            data_source_name,
        )
        return promoted


# ---------------------------------------------------------------------------
# Facility-wide queries — used by top-level pipeline workers
# ---------------------------------------------------------------------------


def get_version_counts(facility: str) -> dict[str, int]:
    """Get SignalEpoch counts by status for a facility.

    Returns:
        Dict with keys: total, discovered, ingested, failed
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            RETURN count(v) AS total,
                   sum(CASE WHEN v.status = 'discovered' THEN 1 ELSE 0 END) AS discovered,
                   sum(CASE WHEN v.status = 'ingested' THEN 1 ELSE 0 END) AS ingested,
                   sum(CASE WHEN v.status = 'failed' THEN 1 ELSE 0 END) AS failed
            """,
            facility=facility,
        )
        if result:
            return dict(result[0])
        return {"total": 0, "discovered": 0, "ingested": 0, "failed": 0}


def get_signal_counts(facility: str) -> dict[str, int]:
    """Get FacilitySignal counts by status for a facility.

    Returns:
        Dict with keys: total, discovered, enriched, checked
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            RETURN count(s) AS total,
                   sum(CASE WHEN s.status = 'discovered' THEN 1 ELSE 0 END) AS discovered,
                   sum(CASE WHEN s.status = 'enriched' THEN 1 ELSE 0 END) AS enriched,
                   sum(CASE WHEN s.status = 'checked' THEN 1 ELSE 0 END) AS checked
            """,
            facility=facility,
        )
        if result:
            return dict(result[0])
        return {"total": 0, "discovered": 0, "enriched": 0, "checked": 0}


def has_pending_extract_work_facility(facility: str) -> bool:
    """Check if any SignalEpochs across all trees need extraction."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            WHERE v.status = 'discovered'
            RETURN count(v) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


@retry_on_deadlock()
def claim_version_for_extraction_facility(facility: str) -> dict | None:
    """Claim a pending SignalEpoch for extraction across all trees.

    Uses claim_token + ORDER BY rand() to prevent deadlocks.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            WHERE v.status = 'discovered'
              AND (v.claimed_at IS NULL
                   OR v.claimed_at < datetime() - duration($cutoff))
            WITH v ORDER BY rand() LIMIT 1
            SET v.claimed_at = datetime(), v.claim_token = $token
            """,
            facility=facility,
            cutoff=cutoff,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (v:SignalEpoch {claim_token: $token})
            RETURN v.id AS id, v.version AS version,
                   v.first_shot AS first_shot, v.data_source_name AS data_source_name
            """,
            token=claim_token,
        )
        if result:
            return dict(result[0])
        return None


def has_pending_units_work_facility(facility: str) -> bool:
    """Check if any ingested tree versions need unit extraction (facility-wide)."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            WHERE v.status = 'ingested'
              AND (v.units_extracted IS NULL OR v.units_extracted = false)
            RETURN count(v) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


@retry_on_deadlock()
def claim_tree_for_units(facility: str) -> dict | None:
    """Claim a tree that needs unit extraction (facility-wide).

    Returns one data_source_name with at least one ingested version that
    hasn't had units extracted yet.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility})
            WHERE v.status = 'ingested'
              AND (v.units_extracted IS NULL OR v.units_extracted = false)
            WITH v.data_source_name AS data_source_name, v
            ORDER BY data_source_name, coalesce(v.first_shot, v.version) DESC, v.version DESC
            WITH data_source_name, collect(v)[0] AS latest
            RETURN data_source_name,
                   latest.version AS latest_version,
                   coalesce(latest.first_shot, latest.version) AS latest_shot
            LIMIT 1
            """,
            facility=facility,
        )
        if result:
            return dict(result[0])
        return None


def get_latest_ingested_epoch_target(
    facility: str,
    data_source_name: str,
) -> dict[str, int] | None:
    """Get the latest ingested epoch version and extraction shot for a tree."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:SignalEpoch {facility_id: $facility, data_source_name: $data_source_name})
            WHERE v.status = 'ingested'
            WITH v
            ORDER BY coalesce(v.first_shot, v.version) DESC, v.version DESC
            RETURN v.version AS latest_version,
                   coalesce(v.first_shot, v.version) AS latest_shot
            LIMIT 1
            """,
            facility=facility,
            data_source_name=data_source_name,
        )
        if result:
            return dict(result[0])
        return None


def has_pending_promote_work_facility(facility: str) -> bool:
    """Check if any trees have leaf DataNodes not yet promoted to FacilitySignal."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility})
            WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
              AND NOT EXISTS {
                  MATCH (n)<-[:HAS_DATA_SOURCE_NODE]-(:FacilitySignal)
              }
            RETURN count(n) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


@retry_on_deadlock()
def claim_tree_for_promote(facility: str) -> str | None:
    """Claim a tree that has un-promoted leaf DataNodes.

    Returns the data_source_name for a tree with NUMERIC/SIGNAL nodes
    that don't yet have a FacilitySignal backed by HAS_DATA_SOURCE_NODE.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (n:SignalNode {facility_id: $facility})
            WHERE n.node_type IN ['NUMERIC', 'SIGNAL']
              AND NOT EXISTS {
                  MATCH (n)<-[:HAS_DATA_SOURCE_NODE]-(:FacilitySignal)
              }
            RETURN DISTINCT n.data_source_name AS data_source_name
            LIMIT 1
            """,
            facility=facility,
        )
        if result:
            return result[0]["data_source_name"]
        return None
