"""Response models for the IMAS signal mapping pipeline.

Pydantic models used as structured output targets for each LLM step:
  assign_targets      — TargetAssignmentBatch: assign signal sources to IDS target paths
  map_signals         — SignalMappingBatch: signal-level mappings with transforms
  discover_assembly   — AssemblyBatch: assembly patterns for struct-array population
  validate_mappings   — ValidatedMappingResult: programmatically validated mappings

Also contains the adapter that converts ValidatedMappingResult into
graph operations (MAPS_TO_IMAS relationships, POPULATES, IMASMapping).
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class MappingDisposition(StrEnum):
    """Why a signal was or was not mapped."""

    MAPPED = "mapped"
    NO_IMAS_EQUIVALENT = "no_imas_equivalent"
    METADATA_ONLY = "metadata_only"
    FACILITY_SPECIFIC = "facility_specific"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class TargetType(StrEnum):
    """Classification of the IDS target path structure."""

    STRUCT_ARRAY = "struct_array"
    TIME_SLICE = "time_slice"
    SCALAR = "scalar"
    PROFILE = "profile"


# ---------------------------------------------------------------------------
# Step 1: Section assignment
# ---------------------------------------------------------------------------


class TargetAssignment(BaseModel):
    """Map a signal source to an IDS target path."""

    source_id: str = Field(description="SignalSource node id")
    imas_target_path: str = Field(
        description=(
            "IMAS subtree path — struct-array (e.g. pf_active/coil), "
            "time-slice container (e.g. equilibrium/time_slice), "
            "or scalar path (e.g. summary/global_quantities/ip)"
        )
    )
    target_type: TargetType = Field(
        default=TargetType.STRUCT_ARRAY,
        description=(
            "Classification of the target path structure: "
            "struct_array, time_slice, scalar, or profile"
        ),
    )
    confidence: float = Field(ge=0, le=1, description="Assignment confidence 0-1")
    reasoning: str = Field(description="Brief justification")


class UnassignedSource(BaseModel):
    """A source that could not be assigned to any IDS section."""

    source_id: str = Field(description="SignalSource node id")
    disposition: MappingDisposition = Field(
        description="Why this source has no section assignment"
    )
    evidence: str = Field(description="Concise evidence why no section fits")


class TargetAssignmentBatch(BaseModel):
    """Batch of target assignments from Step 1."""

    ids_name: str
    assignments: list[TargetAssignment]
    unassigned_groups: list[str] = Field(
        default_factory=list,
        description="Signal source IDs that could not be assigned (deprecated)",
    )
    unassigned: list[UnassignedSource] = Field(
        default_factory=list,
        description="Sources with structured reasoning for non-assignment",
    )


# ---------------------------------------------------------------------------
# Step 2: Signal-level mapping
# ---------------------------------------------------------------------------


class EscalationSeverity(StrEnum):
    """Severity of an escalation flag."""

    WARNING = "warning"
    ERROR = "error"


class EscalationFlag(BaseModel):
    """Flag for a signal the LLM cannot confidently map."""

    source_id: str = Field(description="SignalSource node id")
    target_id: str = Field(description="Target IMAS path")
    severity: EscalationSeverity = EscalationSeverity.WARNING
    reason: str = Field(description="Why this mapping is uncertain")


# Allowed identifiers in transform expressions — value variable,
# math/numpy builtins, and imas_codex utility functions.
_TRANSFORM_ALLOWED_NAMES: frozenset[str] = frozenset(
    {
        "value",
        "math",
        "np",
        "abs",
        "float",
        "int",
        "str",
        "len",
        "convert_units",
        "cocos_sign",
        "radians",
        "degrees",
        "pi",
        "sqrt",
        "log",
        "log10",
        "exp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "True",
        "False",
        "None",
    }
)


class UnmappedSignal(BaseModel):
    """A signal that was evaluated and determined to have no valid IMAS target."""

    source_id: str = Field(description="SignalSource node id")
    disposition: MappingDisposition = Field(
        description="Why this signal has no mapping"
    )
    evidence: str = Field(
        description=(
            "Concise evidence-based explanation. Must reference concrete facts: "
            "searched IMAS paths, checked IDS sections, why no match exists."
        )
    )
    nearest_imas_path: str | None = Field(
        default=None,
        description="Closest IMAS path considered but rejected, if any",
    )
    nearest_similarity: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Cosine similarity to nearest_imas_path, if computed",
    )


class SignalMappingEntry(BaseModel):
    """Single signal mapping entry with transform details."""

    source_id: str = Field(description="SignalSource node id")
    source_property: str = Field(
        default="value",
        description="Property on the source node to map (default: value)",
    )
    target_id: str = Field(description="Full IMAS path for the target field")
    transform_expression: str = Field(
        default="value",
        description="Python expression to transform the source value",
    )
    source_units: str | None = Field(default=None, description="Source unit")
    target_units: str | None = Field(default=None, description="Target IMAS unit")
    cocos_label: str | None = Field(
        default=None,
        description="COCOS transformation label (e.g. ip_like, psi_like)",
    )
    confidence: float = Field(ge=0, le=1, description="Mapping confidence 0-1")
    reasoning: str = Field(default="", description="Brief justification")
    many_to_one_note: str | None = Field(
        default=None,
        description=(
            "When multiple sources map to this target, explain why. "
            "E.g., 'Epoch variant: same coil geometry from 2019 commissioning' "
            "or 'Processing stage: raw measurement before ELM filtering'"
        ),
    )

    @field_validator("transform_expression")
    @classmethod
    def validate_transform_expression(cls, v: str) -> str:
        """Validate transform expression contains only safe constructs."""
        if not v or v == "value":
            return v

        # Extract all identifier-like tokens from the expression
        identifiers = set(re.findall(r"\b([a-zA-Z_]\w*)\b", v))
        unknown = identifiers - _TRANSFORM_ALLOWED_NAMES
        if unknown:
            logger.warning(
                "Transform expression contains unknown identifiers: %s "
                "(expression: %s). Allowed: %s",
                unknown,
                v,
                sorted(_TRANSFORM_ALLOWED_NAMES),
            )

        # Block dangerous patterns (import, exec, eval, __builtins__)
        dangerous = re.findall(
            r"\b(import|exec|eval|compile|__\w+__|open|getattr|setattr"
            r"|delattr|globals|locals|vars|dir|type|super)\b",
            v,
        )
        if dangerous:
            raise ValueError(
                f"Transform expression contains forbidden construct(s): "
                f"{dangerous}. Expression: {v}"
            )

        return v


class SignalMappingBatch(BaseModel):
    """Batch of signal mappings from map_signals (per target)."""

    ids_name: str
    target_path: str = Field(description="IMAS target path")
    mappings: list[SignalMappingEntry]
    unmapped: list[UnmappedSignal] = Field(default_factory=list)
    escalations: list[EscalationFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Assembly discovery
# ---------------------------------------------------------------------------


class AssemblyPattern(StrEnum):
    """How signal sources compose into an IDS target structure."""

    DIRECT = "direct"
    ARRAY_PER_NODE = "array_per_node"
    CONCATENATE = "concatenate"
    CONCATENATE_TRANSPOSE = "concatenate_transpose"
    MATRIX_ASSEMBLY = "matrix_assembly"
    NESTED_ARRAY = "nested_array"


class AssemblyConfig(BaseModel):
    """Assembly configuration for one IDS target path."""

    target_path: str
    pattern: AssemblyPattern = AssemblyPattern.ARRAY_PER_NODE
    init_arrays: dict[str, int] | None = None
    elements_config: str | None = Field(
        default=None, description="JSON string for element structure"
    )
    nested_path: str | None = None
    parent_size: int | None = None
    source_system: str | None = None
    source_data_source: str | None = None
    source_epoch_field: str | None = None
    source_select_via: str | None = None
    ordering_field: str | None = None
    reasoning: str = ""
    confidence: float = Field(default=0.8, ge=0, le=1)
    assembly_code: str | None = Field(
        default=None,
        description="Generated imas-python assembly code snippet",
    )
    assembly_function_name: str | None = Field(
        default=None,
        description="Name of the assembly function in assembly_code",
    )


class AssemblyBatch(BaseModel):
    """Assembly configurations for all sections in an IDS."""

    ids_name: str
    configs: list[AssemblyConfig]


# ---------------------------------------------------------------------------
# Step 3: Validation + final result
# ---------------------------------------------------------------------------


class ValidatedSignalMapping(BaseModel):
    """A signal mapping that has passed validation."""

    source_id: str
    source_property: str = "value"
    target_id: str
    transform_expression: str = "value"
    source_units: str | None = None
    target_units: str | None = None
    cocos_label: str | None = None
    confidence: float = Field(ge=0, le=1)
    disposition: MappingDisposition = MappingDisposition.MAPPED
    evidence: str = ""


class ValidatedMappingResult(BaseModel):
    """Final validated mapping result from Step 3."""

    facility: str
    ids_name: str
    dd_version: str
    sections: list[TargetAssignment]
    bindings: list[ValidatedSignalMapping]
    unmapped: list[UnmappedSignal] = Field(default_factory=list)
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
    assembly: AssemblyBatch | None = None,
    gc: GraphClient | None = None,
    provider: str = "imas-codex",
    status: str = "generated",
    extraction_script: str | None = None,
    assembly_script: str | None = None,
    validated_shot: int | None = None,
    validated_at: str | None = None,
    validation_strategy: str | None = None,
) -> str:
    """Write a validated mapping result to the graph.

    Creates/updates:
      - IMASMapping node
      - POPULATES relationships to section roots (with assembly config)
      - USES_SIGNAL_SOURCE relationships
      - MAPS_TO_IMAS relationships on signal sources
      - MappingEvidence nodes for escalations

    Args:
        result: The validated mapping result to persist.
        assembly: Assembly configurations for struct-array sections.
        gc: GraphClient instance (created if None).
        provider: Provider identifier.
        status: Initial status for the mapping node.
        extraction_script: Generated extraction script code.
        assembly_script: Generated assembly script code.
        validated_shot: Shot used for E2E validation.
        validated_at: ISO timestamp of E2E validation.
        validation_strategy: Strategy used ("client" or "remote").

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
            m.status = $status,
            m.extraction_script = $extraction_script,
            m.assembly_script = $assembly_script,
            m.validated_shot = $validated_shot,
            m.validated_at = $validated_at,
            m.validation_strategy = $validation_strategy
        WITH m
        MATCH (f:Facility {id: $facility})
        MERGE (m)-[:AT_FACILITY]->(f)
        """,
        id=mapping_id,
        facility=result.facility,
        ids_name=result.ids_name,
        dd_version=result.dd_version,
        provider=provider,
        status=status,
        extraction_script=extraction_script,
        assembly_script=assembly_script,
        validated_shot=validated_shot,
        validated_at=validated_at,
        validation_strategy=validation_strategy,
    )

    # 2. Create POPULATES relationships to target roots with assembly config
    assembly_by_target: dict[str, AssemblyConfig] = {}
    if assembly:
        assembly_by_target = {c.target_path: c for c in assembly.configs}

    for section in result.sections:
        asm = assembly_by_target.get(section.imas_target_path)
        asm_params: dict[str, Any] = {}
        if asm:
            asm_params = {
                "structure": asm.pattern.value,
                "init_arrays": json.dumps(asm.init_arrays) if asm.init_arrays else None,
                "elements_config": asm.elements_config,
                "nested_path": asm.nested_path,
                "parent_size": asm.parent_size,
                "source_system": asm.source_system,
                "source_data_source": asm.source_data_source,
                "source_epoch_field": asm.source_epoch_field,
                "source_select_via": asm.source_select_via,
                "assembly_code": asm.assembly_code,
                "assembly_function_name": asm.assembly_function_name,
            }

        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
            MATCH (ip:IMASNode {id: $imas_path})
            MERGE (m)-[r:POPULATES]->(ip)
            SET r.confidence = $confidence,
                r.structure = $structure,
                r.init_arrays = $init_arrays,
                r.elements_config = $elements_config,
                r.nested_path = $nested_path,
                r.parent_size = $parent_size,
                r.source_system = $source_system,
                r.source_data_source = $source_data_source,
                r.source_epoch_field = $source_epoch_field,
                r.source_select_via = $source_select_via,
                r.assembly_code = $assembly_code,
                r.assembly_function_name = $assembly_function_name
            """,
            mapping_id=mapping_id,
            imas_path=section.imas_target_path,
            confidence=section.confidence,
            structure=asm_params.get("structure"),
            init_arrays=asm_params.get("init_arrays"),
            elements_config=asm_params.get("elements_config"),
            nested_path=asm_params.get("nested_path"),
            parent_size=asm_params.get("parent_size"),
            source_system=asm_params.get("source_system"),
            source_data_source=asm_params.get("source_data_source"),
            source_epoch_field=asm_params.get("source_epoch_field"),
            source_select_via=asm_params.get("source_select_via"),
            assembly_code=asm_params.get("assembly_code"),
            assembly_function_name=asm_params.get("assembly_function_name"),
        )

    # 3. Collect signal source IDs and create USES_SIGNAL_SOURCE
    sg_ids = {fm.source_id for fm in result.bindings}
    for sg_id in sg_ids:
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
            MATCH (sg:SignalSource {id: $sg_id})
            MERGE (m)-[:USES_SIGNAL_SOURCE]->(sg)
            """,
            mapping_id=mapping_id,
            sg_id=sg_id,
        )

    # 4. Create MAPS_TO_IMAS relationships
    for fm in result.bindings:
        gc.query(
            """
            MATCH (sg:SignalSource {id: $sg_id})
            MATCH (ip:IMASNode {id: $target_id})
            MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)
            SET r.source_property = $source_property,
                r.transform_expression = $transform_expression,
                r.source_units = $source_units,
                r.target_units = $target_units,
                r.cocos_label = $cocos_label,
                r.confidence = $confidence,
                r.evidence = $evidence
            """,
            sg_id=fm.source_id,
            target_id=fm.target_id,
            source_property=fm.source_property,
            transform_expression=fm.transform_expression,
            source_units=fm.source_units,
            target_units=fm.target_units,
            cocos_label=fm.cocos_label,
            confidence=fm.confidence,
            evidence=fm.evidence,
        )

    # 5. Persist escalations as MappingEvidence
    for esc in result.escalations:
        ev_id = f"{esc.source_id}:{esc.target_id}:escalation"
        gc.query(
            """
            MATCH (sg:SignalSource {id: $sg_id})
            MERGE (ev:MappingEvidence {id: $ev_id})
            SET ev.evidence_type = 'escalation',
                ev.text = $reason,
                ev.supports_mapping = false,
                ev.confidence = 0.0,
                ev.source_node_id = $sg_id,
                ev.url = $target_id
            MERGE (sg)-[:HAS_EVIDENCE]->(ev)
            """,
            ev_id=ev_id,
            sg_id=esc.source_id,
            target_id=esc.target_id,
            reason=esc.reason,
        )

    # 6. Persist unmapped signal dispositions
    for um in result.unmapped:
        gc.query(
            """
            MATCH (sg:SignalSource {id: $sg_id})
            SET sg.mapping_disposition = $disposition,
                sg.mapping_evidence = $evidence
            """,
            sg_id=um.source_id,
            disposition=um.disposition.value,
            evidence=um.evidence,
        )

    logger.info(
        "Persisted mapping %s with %d bindings, %d unmapped",
        mapping_id,
        len(result.bindings),
        len(result.unmapped),
    )
    return mapping_id
