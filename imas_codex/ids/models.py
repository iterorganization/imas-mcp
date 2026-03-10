"""Response models for the IMAS mapping pipeline.

Pydantic models used as structured output targets for each LLM step:
  Step 1 — SectionAssignmentBatch: assign signal groups to IMAS sections
  Step 2 — FieldMappingBatch: field-level mappings with transforms
  Step 3 — ValidatedMappingResult: reviewed and corrected mappings

Also contains the adapter that converts ValidatedMappingResult into
graph operations (MAPS_TO_IMAS relationships, POPULATES, IMASMapping).
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Section assignment
# ---------------------------------------------------------------------------


class SectionAssignment(BaseModel):
    """Map a signal group to an IMAS structural array section."""

    signal_group_id: str = Field(description="SignalGroup node id")
    imas_section_path: str = Field(
        description="IMAS struct-array path (e.g. pf_active/coil)"
    )
    confidence: float = Field(ge=0, le=1, description="Assignment confidence 0-1")
    reasoning: str = Field(description="Brief justification")


class SectionAssignmentBatch(BaseModel):
    """Batch of section assignments from Step 1."""

    ids_name: str
    assignments: list[SectionAssignment]
    unassigned_groups: list[str] = Field(
        default_factory=list,
        description="Signal group IDs that could not be assigned",
    )


# ---------------------------------------------------------------------------
# Step 2: Field-level mapping
# ---------------------------------------------------------------------------


class EscalationSeverity(StrEnum):
    """Severity of an escalation flag."""

    WARNING = "warning"
    ERROR = "error"


class EscalationFlag(BaseModel):
    """Flag for a field the LLM cannot confidently map."""

    signal_group_id: str
    imas_path: str
    severity: EscalationSeverity = EscalationSeverity.WARNING
    reason: str = Field(description="Why this mapping is uncertain")


class FieldMappingEntry(BaseModel):
    """Single field-level mapping with transform details."""

    signal_group_id: str = Field(description="Source SignalGroup id")
    source_property: str = Field(
        default="value",
        description="Property on the source node to map (default: value)",
    )
    target_imas_path: str = Field(description="Full IMAS path for the target field")
    transform_code: str = Field(
        default="value",
        description="Python expression to transform the source value",
    )
    units_in: str | None = Field(default=None, description="Source unit")
    units_out: str | None = Field(default=None, description="Target IMAS unit")
    cocos_label: str | None = Field(
        default=None,
        description="COCOS transformation label (e.g. ip_like, psi_like)",
    )
    confidence: float = Field(ge=0, le=1, description="Mapping confidence 0-1")
    reasoning: str = Field(default="", description="Brief justification")


class FieldMappingBatch(BaseModel):
    """Batch of field mappings from Step 2 (per section)."""

    ids_name: str
    section_path: str = Field(description="IMAS struct-array section")
    mappings: list[FieldMappingEntry]
    escalations: list[EscalationFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 3: Validation + final result
# ---------------------------------------------------------------------------


class ValidatedFieldMapping(BaseModel):
    """A field mapping that has passed validation."""

    signal_group_id: str
    source_property: str = "value"
    target_imas_path: str
    transform_code: str = "value"
    units_in: str | None = None
    units_out: str | None = None
    cocos_label: str | None = None
    confidence: float = Field(ge=0, le=1)


class ValidatedMappingResult(BaseModel):
    """Final validated mapping result from Step 3."""

    facility: str
    ids_name: str
    dd_version: str
    sections: list[SectionAssignment]
    field_mappings: list[ValidatedFieldMapping]
    escalations: list[EscalationFlag] = Field(default_factory=list)
    corrections: list[str] = Field(
        default_factory=list,
        description="Summary of corrections made during validation",
    )


# ---------------------------------------------------------------------------
# Adapter: ValidatedMappingResult → graph operations
# ---------------------------------------------------------------------------


def persist_mapping_result(
    result: ValidatedMappingResult,
    *,
    gc: GraphClient | None = None,
    provider: str = "imas-codex",
) -> str:
    """Write a validated mapping result to the graph.

    Creates/updates:
      - IMASMapping node
      - POPULATES relationships to section roots
      - USES_SIGNAL_GROUP relationships
      - MAPS_TO_IMAS relationships on signal groups
      - MappingEvidence nodes for escalations

    Returns:
        The IMASMapping node id.
    """
    if gc is None:
        gc = GraphClient()

    mapping_id = f"{result.facility}:{result.ids_name}"

    # 1. Create or update IMASMapping node
    gc.query(
        """
        MERGE (m:IMASMapping {id: $id})
        SET m.facility_id = $facility,
            m.ids_name = $ids_name,
            m.dd_version = $dd_version,
            m.provider = $provider,
            m.status = 'validated'
        """,
        id=mapping_id,
        facility=result.facility,
        ids_name=result.ids_name,
        dd_version=result.dd_version,
        provider=provider,
    )

    # 2. Create POPULATES relationships to section roots
    for section in result.sections:
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
            MATCH (ip:IMASNode {id: $imas_path})
            MERGE (m)-[r:POPULATES]->(ip)
            SET r.confidence = $confidence
            """,
            mapping_id=mapping_id,
            imas_path=section.imas_section_path,
            confidence=section.confidence,
        )

    # 3. Collect signal group IDs and create USES_SIGNAL_GROUP
    sg_ids = {fm.signal_group_id for fm in result.field_mappings}
    for sg_id in sg_ids:
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
            MATCH (sg:SignalGroup {id: $sg_id})
            MERGE (m)-[:USES_SIGNAL_GROUP]->(sg)
            """,
            mapping_id=mapping_id,
            sg_id=sg_id,
        )

    # 4. Create MAPS_TO_IMAS relationships
    for fm in result.field_mappings:
        gc.query(
            """
            MATCH (sg:SignalGroup {id: $sg_id})
            MATCH (ip:IMASNode {id: $imas_path})
            MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)
            SET r.source_property = $source_property,
                r.transform_code = $transform_code,
                r.units_in = $units_in,
                r.units_out = $units_out,
                r.cocos_label = $cocos_label,
                r.confidence = $confidence
            """,
            sg_id=fm.signal_group_id,
            imas_path=fm.target_imas_path,
            source_property=fm.source_property,
            transform_code=fm.transform_code,
            units_in=fm.units_in,
            units_out=fm.units_out,
            cocos_label=fm.cocos_label,
            confidence=fm.confidence,
        )

    # 5. Persist escalations as MappingEvidence
    for esc in result.escalations:
        gc.query(
            """
            MATCH (sg:SignalGroup {id: $sg_id})
            MERGE (ev:MappingEvidence {
                signal_group_id: $sg_id,
                imas_path: $imas_path,
                type: 'escalation'
            })
            SET ev.severity = $severity,
                ev.reason = $reason
            MERGE (sg)-[:HAS_EVIDENCE]->(ev)
            """,
            sg_id=esc.signal_group_id,
            imas_path=esc.imas_path,
            severity=esc.severity.value,
            reason=esc.reason,
        )

    logger.info(
        "Persisted mapping %s with %d field mappings",
        mapping_id,
        len(result.field_mappings),
    )
    return mapping_id
