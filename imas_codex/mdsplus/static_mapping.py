"""Static tree to IMAS mapping — deterministic + LLM-augmented.

Maps MDSplus static tree nodes (coils, vessel, flux loops, probes, tiles)
to their IMAS Data Dictionary equivalents. The core mapping table is
deterministic (known physics correspondences), while the LLM is used
for confidence scoring, description matching, and edge-case resolution.

Mapping table:
    Static System → IMAS IDS
    ─────────────────────────────────
    C (coils)        → pf_active/coil
    A (as-connected) → pf_active/coil (grouped)
    W (turns)        → pf_active/coil/element
    V (vessel)       → pf_passive/loop (or wall/description_2d/vessel)
    F (flux loops)   → magnetics/flux_loop
    M (mag probes)   → magnetics/bpol_probe
    T (tile contour) → wall/description_2d/limiter/unit
    X, XC (mesh)     → No direct IMAS mapping (computational grids)
    E (eigenmodes)   → No direct IMAS mapping (derived quantities)

Generic: the mapping table is per-facility (different machines name their
systems differently), but the IMAS targets are universal.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticMapping:
    """A deterministic mapping from a static tree parameter to IMAS."""

    system_symbol: str
    parameter: str
    imas_ids: str
    imas_path: str
    units: str = ""
    description: str = ""
    is_array: bool = True
    confidence: float = 0.95


# Deterministic mapping table for TCV static tree → IMAS
# This table captures the known physics correspondences between
# MDSplus static tree tags and IMAS data dictionary paths.
# Format: (system_symbol, parameter) → (imas_ids, imas_path, description)
TCV_STATIC_MAPPINGS: list[StaticMapping] = [
    # === PF Active Coils (C system → pf_active/coil) ===
    StaticMapping(
        "C",
        "R",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/r",
        units="m",
        description="Coil centre major radius",
    ),
    StaticMapping(
        "C",
        "Z",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/z",
        units="m",
        description="Coil centre vertical position",
    ),
    StaticMapping(
        "C",
        "W",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/width",
        units="m",
        description="Coil horizontal full width",
    ),
    StaticMapping(
        "C",
        "H",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/height",
        units="m",
        description="Coil vertical full height",
    ),
    StaticMapping(
        "C",
        "INOM",
        "pf_active",
        "pf_active/coil/current_limit_max",
        units="A",
        description="Nominal current limit",
    ),
    StaticMapping(
        "C",
        "UNOM",
        "pf_active",
        "pf_active/coil/voltage/data",
        units="V",
        description="Nominal voltage",
        confidence=0.7,
    ),
    # === Individual Coil Turns (W system → pf_active/coil/element) ===
    StaticMapping(
        "W",
        "R",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/r",
        units="m",
        description="Turn centre major radius",
    ),
    StaticMapping(
        "W",
        "Z",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/z",
        units="m",
        description="Turn centre vertical position",
    ),
    StaticMapping(
        "W",
        "W",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/width",
        units="m",
        description="Turn horizontal width",
    ),
    StaticMapping(
        "W",
        "H",
        "pf_active",
        "pf_active/coil/element/geometry/rectangle/height",
        units="m",
        description="Turn vertical height",
    ),
    StaticMapping(
        "W",
        "N",
        "pf_active",
        "pf_active/coil/element/turns_with_sign",
        description="Number of turns with polarity sign",
        confidence=0.85,
    ),
    # === Vessel Filaments (V system → wall or pf_passive) ===
    StaticMapping(
        "V",
        "R",
        "wall",
        "wall/description_2d/vessel/unit/annular/outline_inner/r",
        units="m",
        description="Vessel filament major radius",
        confidence=0.8,
    ),
    StaticMapping(
        "V",
        "Z",
        "wall",
        "wall/description_2d/vessel/unit/annular/outline_inner/z",
        units="m",
        description="Vessel filament vertical position",
        confidence=0.8,
    ),
    StaticMapping(
        "V",
        "RHO",
        "pf_passive",
        "pf_passive/loop/resistivity",
        units="ohm.m",
        description="Vessel filament resistivity",
        confidence=0.8,
    ),
    StaticMapping(
        "V",
        "PERIM",
        "wall",
        "wall/description_2d/vessel/unit/annular/outline_inner/r",
        units="m",
        description="Vessel filament perimeter",
        confidence=0.5,
    ),
    # === Flux Loops (F system → magnetics/flux_loop) ===
    StaticMapping(
        "F",
        "R",
        "magnetics",
        "magnetics/flux_loop/position/r",
        units="m",
        description="Flux loop major radius position",
    ),
    StaticMapping(
        "F",
        "Z",
        "magnetics",
        "magnetics/flux_loop/position/z",
        units="m",
        description="Flux loop vertical position",
    ),
    StaticMapping(
        "F",
        "ANG",
        "magnetics",
        "magnetics/flux_loop/position/phi",
        units="rad",
        description="Flux loop toroidal angle",
        confidence=0.7,
    ),
    # === Magnetic Probes (M system → magnetics/bpol_probe) ===
    StaticMapping(
        "M",
        "R",
        "magnetics",
        "magnetics/bpol_probe/position/r",
        units="m",
        description="Magnetic probe major radius",
    ),
    StaticMapping(
        "M",
        "Z",
        "magnetics",
        "magnetics/bpol_probe/position/z",
        units="m",
        description="Magnetic probe vertical position",
    ),
    StaticMapping(
        "M",
        "ANG",
        "magnetics",
        "magnetics/bpol_probe/poloidal_angle",
        units="rad",
        description="Magnetic probe poloidal angle",
    ),
    # === Tile Contour (T system → wall/description_2d/limiter) ===
    StaticMapping(
        "T",
        "R",
        "wall",
        "wall/description_2d/limiter/unit/outline/r",
        units="m",
        description="First wall/tile contour major radius",
    ),
    StaticMapping(
        "T",
        "Z",
        "wall",
        "wall/description_2d/limiter/unit/outline/z",
        units="m",
        description="First wall/tile contour vertical position",
    ),
]


@dataclass
class MappingResult:
    """Result of mapping static tree nodes to IMAS paths."""

    proposals: list[dict[str, Any]] = field(default_factory=list)
    unmapped_nodes: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def build_mapping_proposals(
    facility: str,
    tree_data: dict[str, Any],
    mapping_table: list[StaticMapping] | None = None,
) -> MappingResult:
    """Build IMAS mapping proposals from static tree extraction data.

    Uses a deterministic mapping table to propose mappings between
    static tree nodes and IMAS paths. Each proposal includes the
    source TreeNode path, target IMASPath, confidence score, and
    evidence from the tree metadata.

    Args:
        facility: Facility identifier (e.g., "tcv")
        tree_data: Output from discover_static_tree()
        mapping_table: Override mapping table (default: TCV_STATIC_MAPPINGS)

    Returns:
        MappingResult with proposals, unmapped nodes, and statistics
    """
    if mapping_table is None:
        mapping_table = TCV_STATIC_MAPPINGS

    tree_name = tree_data["tree_name"]
    result = MappingResult()

    # Build lookup: (system_symbol, parameter) -> StaticMapping
    mapping_lookup: dict[tuple[str, str], StaticMapping] = {}
    for m in mapping_table:
        mapping_lookup[(m.system_symbol, m.parameter)] = m

    # Use the most recent version's nodes for mapping
    versions = tree_data.get("versions", {})
    if not versions:
        return result

    latest_ver = max(versions.keys(), key=int)
    latest_data = versions[latest_ver]
    if "error" in latest_data:
        logger.warning(
            "Latest version %s has error: %s", latest_ver, latest_data["error"]
        )
        return result

    mapped_count = 0
    unmapped_count = 0

    for node in latest_data.get("nodes", []):
        path = node["path"]
        tags = node.get("tags", [])

        # Try to match via tag pattern: \<PARAM>_<SYSTEM>
        matched = False
        for tag in tags:
            tag_clean = tag.lstrip("\\").upper()
            parts = tag_clean.split("_", 1)
            if len(parts) == 2:
                param, system = parts
                key = (system, param)
                if key in mapping_lookup:
                    mapping = mapping_lookup[key]
                    proposal = {
                        "id": f"{facility}:{tree_name}:{tag_clean}→{mapping.imas_path}",
                        "facility_id": facility,
                        "source_path": f"{facility}:{tree_name}:{path}",
                        "source_tag": tag_clean,
                        "target_path": mapping.imas_path,
                        "target_ids": mapping.imas_ids,
                        "confidence": mapping.confidence,
                        "evidence_type": "deterministic_mapping",
                        "description": mapping.description,
                        "units_source": node.get("units", ""),
                        "units_target": mapping.units,
                        "is_static": True,
                    }
                    # Include extracted value summary if available
                    if node.get("shape"):
                        proposal["source_shape"] = node["shape"]
                    if node.get("value") is not None and isinstance(
                        node["value"], int | float
                    ):
                        proposal["source_value"] = node["value"]
                    result.proposals.append(proposal)
                    mapped_count += 1
                    matched = True
                    break

        if not matched and node.get("node_type") in ("NUMERIC", "SIGNAL"):
            result.unmapped_nodes.append(path)
            unmapped_count += 1

    result.stats = {
        "mapped": mapped_count,
        "unmapped": unmapped_count,
        "total_proposals": len(result.proposals),
        "unique_imas_targets": len({p["target_path"] for p in result.proposals}),
    }

    logger.info(
        "Built %d mapping proposals (%d mapped, %d unmapped)",
        len(result.proposals),
        mapped_count,
        unmapped_count,
    )

    return result


def persist_mapping_proposals(
    client: "GraphClient",
    result: MappingResult,
    dry_run: bool = False,
) -> int:
    """Persist mapping proposals to the Neo4j graph.

    Creates MappingProposal nodes with PROPOSES_SOURCE → TreeNode
    and PROPOSES_TARGET → IMASPath relationships.

    Args:
        client: Neo4j GraphClient
        result: MappingResult from build_mapping_proposals()
        dry_run: If True, log but don't write

    Returns:
        Number of proposals persisted
    """
    if not result.proposals:
        logger.info("No proposals to persist")
        return 0

    if dry_run:
        logger.info(
            "[DRY RUN] Would create %d MappingProposal nodes", len(result.proposals)
        )
        for p in result.proposals[:10]:
            logger.info(
                "  %s → %s (confidence=%.2f)",
                p["source_tag"],
                p["target_path"],
                p["confidence"],
            )
        if len(result.proposals) > 10:
            logger.info("  ... and %d more", len(result.proposals) - 10)
        return len(result.proposals)

    # Batch create proposals
    records = []
    for p in result.proposals:
        records.append(
            {
                "id": p["id"],
                "facility_id": p["facility_id"],
                "source_path": p["source_path"],
                "source_tag": p["source_tag"],
                "target_path": p["target_path"],
                "target_ids": p["target_ids"],
                "confidence": p["confidence"],
                "evidence_type": p["evidence_type"],
                "description": p["description"],
                "status": "proposed",
                "is_static": True,
            }
        )

    client.query(
        """
        UNWIND $proposals AS prop
        MERGE (mp:MappingProposal {id: prop.id})
        SET mp += prop,
            mp.proposed_at = datetime()
        """,
        proposals=records,
    )

    # Link to IMASPath targets
    client.query(
        """
        UNWIND $proposals AS prop
        MATCH (mp:MappingProposal {id: prop.id})
        MATCH (t:IMASPath {id: prop.target_path})
        MERGE (mp)-[:PROPOSES_TARGET]->(t)
        """,
        proposals=records,
    )

    # Link to TreeNode sources (if they exist in graph)
    client.query(
        """
        UNWIND $proposals AS prop
        MATCH (mp:MappingProposal {id: prop.id})
        MATCH (n:TreeNode {id: prop.source_path})
        MERGE (mp)-[:PROPOSES_SOURCE]->(n)
        """,
        proposals=records,
    )

    # Link to facility
    facility_id = records[0]["facility_id"] if records else None
    if facility_id:
        client.query(
            """
            MATCH (mp:MappingProposal)
            WHERE mp.facility_id = $facility AND mp.is_static = true
            WITH mp
            MATCH (f:Facility {id: $facility})
            MERGE (mp)-[:AT_FACILITY]->(f)
            """,
            facility=facility_id,
        )

    logger.info("Persisted %d mapping proposals", len(records))
    return len(records)
