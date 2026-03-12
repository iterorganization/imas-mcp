"""IMAS signal mapping pipeline orchestrator.

Multi-step LLM pipeline that generates signal-level IMAS mappings
from facility signal sources:

  gather_context:     Fetch signal sources + DD context (programmatic)
  assign_sections:    LLM assigns sources to IMAS sections
  map_signals:        For each section, LLM generates signal mappings
  discover_assembly:  For each section, LLM discovers assembly patterns
  validate_mappings:  Programmatic validation (source/target existence, transforms, units)
  persist:            Write to graph

Usage:
    from imas_codex.ids.mapping import generate_mapping

    result = generate_mapping("jet", "pf_active")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.ids.models import (
    AssemblyBatch,
    AssemblyConfig,
    EscalationFlag,
    SignalMappingBatch,
    SectionAssignmentBatch,
    ValidatedSignalMapping,
    ValidatedMappingResult,
    persist_mapping_result,
)
from imas_codex.ids.tools import (
    analyze_units,
    fetch_imas_fields,
    fetch_imas_subtree,
    fetch_source_code_refs,
    get_sign_flip_paths,
    query_signal_sources,
    search_existing_mappings,
    search_imas_semantic,
)
from imas_codex.settings import get_model

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agentic" / "prompts" / "mapping"


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


@dataclass
class PipelineCost:
    """Accumulated cost across pipeline steps."""

    steps: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0

    @property
    def total_usd(self) -> float:
        return sum(self.steps.values())

    def add(self, step: str, cost: float, tokens: int) -> None:
        self.steps[step] = self.steps.get(step, 0) + cost
        self.total_tokens += tokens


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    """Load a mapping prompt template by name."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text()
    # Strip YAML frontmatter
    if text.startswith("---"):
        _, _, text = text.split("---", 2)
    return text.strip()


def _render_prompt(name: str, **context: Any) -> str:
    """Render a prompt template with Jinja2 substitutions."""
    from jinja2 import BaseLoader, Environment

    template_str = _load_prompt(name)
    env = Environment(loader=BaseLoader(), autoescape=False)
    template = env.from_string(template_str)
    return template.render(**context)


def _format_subtree(rows: list[dict[str, Any]]) -> str:
    """Format IMAS subtree rows into a readable tree summary."""
    lines: list[str] = []
    for r in rows:
        path = r.get("id", "")
        dtype = r.get("data_type", "")
        units = r.get("units", "")
        doc = r.get("documentation", "")
        parts = [path]
        if dtype:
            parts.append(f"({dtype})")
        if units:
            parts.append(f"[{units}]")
        line = " ".join(parts)
        if doc:
            line += f" — {doc[:100]}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no paths)"


def _format_sources(groups: list[dict[str, Any]]) -> str:
    """Format signal sources into a readable summary."""
    lines: list[str] = []
    for g in groups:
        gid = g.get("id", "")
        key = g.get("group_key", "")
        desc = g.get("description", "")
        members = g.get("member_count", 0)
        domain = g.get("physics_domain", "")
        existing = g.get("imas_mappings", [])
        mapped = [m for m in existing if m.get("target_id")]
        line = f"- {gid}"
        if domain:
            line += f" (domain={domain}, members={members})"
        else:
            line += f" (key={key}, members={members})"
        if desc:
            line += f": {desc}"
        # Enriched metadata from representative signal
        rep_desc = g.get("rep_description")
        if rep_desc and rep_desc != desc:
            line += f"\n  Representative: {rep_desc[:200]}"
        rep_unit = g.get("rep_unit")
        if rep_unit:
            line += f"\n  Unit: {rep_unit}"
        rep_cocos = g.get("rep_cocos")
        if rep_cocos:
            line += f"\n  COCOS: {rep_cocos}"
        rep_sign = g.get("rep_sign_convention")
        if rep_sign:
            line += f"\n  Sign convention: {rep_sign}"
        accessors = g.get("sample_accessors")
        if accessors:
            line += f"\n  Accessors: {', '.join(str(a) for a in accessors[:5])}"
        if mapped:
            targets = ", ".join(m["target_id"] for m in mapped)
            line += f"\n  [already mapped → {targets}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(no sources)"


def _format_source_detail(source: dict[str, Any]) -> str:
    """Format a single signal source as structured markdown for the prompt."""
    parts: list[str] = []
    sid = source.get("id", "unknown")
    parts.append(f"**Source ID**: {sid}")
    desc = source.get("description", "")
    if desc:
        parts.append(f"**Description**: {desc}")
    rep_desc = source.get("rep_description")
    if rep_desc and rep_desc != desc:
        parts.append(f"**Representative Signal**: {rep_desc[:200]}")
    domain = source.get("physics_domain")
    if domain:
        parts.append(f"**Physics Domain**: {domain}")
    parts.append(f"**Units**: {source.get('rep_unit') or 'unknown'}")
    sign = source.get("rep_sign_convention")
    parts.append(f"**Sign Convention**: {sign or 'unknown'}")
    cocos = source.get("rep_cocos")
    parts.append(f"**COCOS**: {cocos or 'not set'}")
    members = source.get("member_count", 0)
    parts.append(f"**Members**: {members} signals")
    key = source.get("group_key", "")
    if key:
        parts.append(f"**Accessor Pattern**: {key}")
    accessors = source.get("sample_accessors")
    if accessors:
        parts.append(f"**Sample Accessors**: {', '.join(str(a) for a in accessors[:5])}")
    existing = source.get("imas_mappings", [])
    mapped = [m for m in existing if m.get("target_id")]
    if mapped:
        targets = ", ".join(m["target_id"] for m in mapped)
        parts.append(f"**Existing Mappings**: {targets}")
    return "\n".join(parts)


def _format_fields(fields: list[dict[str, Any]]) -> str:
    """Format IMAS field details for the field mapping prompt."""
    lines: list[str] = []
    for f in fields:
        path = f.get("id", "")
        dtype = f.get("data_type", "")
        units = f.get("units", "")
        doc = f.get("documentation", "")
        ndim = f.get("ndim")
        parts = [f"- {path} ({dtype})"]
        if units:
            parts.append(f"[{units}]")
        if ndim is not None:
            parts.append(f"ndim={ndim}")
        line = " ".join(parts)
        if doc:
            line += f"\n  {doc[:200]}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no fields)"


def _format_unit_analysis(
    groups: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> str:
    """Run unit analysis between signal sources and IMAS fields.

    Uses the rep_unit field (dot-exp format from normalize_unit_symbol)
    instead of extracting units from keywords.
    """
    from imas_codex.units import normalize_unit_symbol

    lines: list[str] = []
    for g in groups:
        signal_unit = g.get("rep_unit")
        if not signal_unit:
            continue
        # Normalize to dot-exp
        signal_unit = normalize_unit_symbol(signal_unit) or signal_unit
        for f in fields:
            imas_unit = f.get("units")
            if imas_unit:
                imas_unit = normalize_unit_symbol(imas_unit) or imas_unit
                result = analyze_units(signal_unit, imas_unit)
                if result.get("compatible"):
                    factor = result.get("conversion_factor", 1.0)
                    lines.append(
                        f"  {signal_unit} → {imas_unit}: compatible"
                        + (f" (×{factor})" if factor != 1.0 else "")
                    )
                elif result.get("error"):
                    lines.append(f"  {signal_unit} → {imas_unit}: {result['error']}")
    return "\n".join(lines) if lines else "(no unit analysis needed)"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _call_llm(
    messages: list[dict[str, str]],
    response_model: type,
    *,
    model: str | None = None,
    step_name: str = "",
    cost: PipelineCost | None = None,
) -> Any:
    """Call LLM with structured output parsing."""
    from imas_codex.discovery.base.llm import call_llm_structured

    llm_model = model or get_model("language")
    logger.info("Step %s: calling %s", step_name, llm_model)

    result, usd, tokens = call_llm_structured(
        llm_model,
        messages,
        response_model,
    )

    if cost:
        cost.add(step_name, usd, tokens)

    logger.info("Step %s: %d tokens, $%.4f", step_name, tokens, usd)
    return result


def gather_context(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient,
) -> dict[str, Any]:
    """Gather all context needed for the signal mapping pipeline."""
    logger.info("Gathering context for %s/%s", facility, ids_name)

    # Don't filter by ids_name — Step 1 assigns groups to IDS sections.
    # Filtering here would require pre-existing MAPS_TO_IMAS relationships,
    # creating a chicken-and-egg problem for new mappings.
    groups = query_signal_sources(facility, gc=gc)
    subtree = fetch_imas_subtree(ids_name, gc=gc)

    # Semantic search is supplementary — degrade gracefully if embedding
    # server is unavailable.
    try:
        semantic = search_imas_semantic(f"{facility} {ids_name}", ids_name, gc=gc, k=10)
    except Exception:
        logger.warning("Semantic search unavailable — continuing without it")
        semantic = []

    existing = search_existing_mappings(facility, ids_name, gc=gc)
    cocos_paths = get_sign_flip_paths(ids_name)

    return {
        "groups": groups,
        "subtree": subtree,
        "semantic": semantic,
        "existing": existing,
        "cocos_paths": cocos_paths,
    }


def assign_sections(
    facility: str,
    ids_name: str,
    context: dict[str, Any],
    *,
    model: str | None = None,
    cost: PipelineCost,
) -> SectionAssignmentBatch:
    """Assign signal sources to IMAS struct-array sections."""
    logger.info("Assigning signal sources to IMAS sections")

    prompt = _render_prompt(
        "section_assignment",
        facility=facility,
        ids_name=ids_name,
        signal_sources=_format_sources(context["groups"]),
        imas_subtree=_format_subtree(context["subtree"]),
        semantic_results=_format_subtree(context["semantic"]),
    )

    messages = [
        {"role": "system", "content": "You are an IMAS mapping expert."},
        {"role": "user", "content": prompt},
    ]

    return _call_llm(
        messages,
        SectionAssignmentBatch,
        model=model,
        step_name="assign_sections",
        cost=cost,
    )


def map_signals(
    facility: str,
    ids_name: str,
    sections: SectionAssignmentBatch,
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> list[SignalMappingBatch]:
    """Generate signal mappings per section."""
    logger.info(
        "Generating signal mappings for %d sections", len(sections.assignments)
    )

    batches: list[SignalMappingBatch] = []

    for assignment in sections.assignments:
        section_path = assignment.imas_section_path

        # Get detailed fields for this section
        fields = fetch_imas_fields(ids_name, [section_path], gc=gc)
        # Also get leaf fields under this section
        subtree_fields = fetch_imas_subtree(
            ids_name, section_path.removeprefix(f"{ids_name}/"), gc=gc, leaf_only=True
        )

        # Find the signal source details
        sg_detail = next(
            (g for g in context["groups"] if g["id"] == assignment.source_id),
            {},
        )

        # Fetch code references for this source
        code_refs = fetch_source_code_refs(assignment.source_id, gc=gc)
        code_context = ""
        if code_refs:
            snippets = []
            for ref in code_refs:
                if ref.get("code"):
                    lang = ref.get("language", "")
                    snippets.append(f"```{lang}\n{ref['code']}\n```")
            if snippets:
                code_context = "\n".join(snippets)

        # Build per-source COCOS context
        source_cocos = sg_detail.get("rep_cocos")
        cocos_context = ""
        if source_cocos:
            flip_paths = [
                p for p in context["cocos_paths"]
                if p.startswith(section_path)
            ]
            cocos_context = f"Signal COCOS convention: {source_cocos}"
            if flip_paths:
                cocos_context += (
                    "\nTarget IMAS paths requiring sign flip:\n"
                    + "\n".join(f"- {p}" for p in flip_paths)
                )

        prompt = _render_prompt(
            "signal_mapping",
            facility=facility,
            ids_name=ids_name,
            section_path=section_path,
            signal_source_detail=_format_source_detail(sg_detail),
            imas_fields=_format_fields(subtree_fields or fields),
            unit_analysis=_format_unit_analysis(
                [sg_detail] if sg_detail else [], subtree_fields or fields
            ),
            cocos_paths="\n".join(f"- {p}" for p in context["cocos_paths"]) or "(none)",
            existing_mappings=json.dumps(context["existing"], indent=2, default=str),
            code_references=code_context or "(no code references available)",
            source_cocos=cocos_context or "(no COCOS context)",
        )

        messages = [
            {"role": "system", "content": "You are an IMAS mapping expert."},
            {"role": "user", "content": prompt},
        ]

        batch = _call_llm(
            messages,
            SignalMappingBatch,
            model=model,
            step_name=f"map_signals_{section_path}",
            cost=cost,
        )
        batches.append(batch)

    return batches


def _format_signal_mappings(batch: SignalMappingBatch) -> str:
    """Format signal mappings for the assembly prompt."""
    lines: list[str] = []
    for m in batch.mappings:
        line = f"- {m.source_id}.{m.source_property} → {m.target_id}"
        if m.transform_expression != "value":
            line += f" (transform: {m.transform_expression})"
        if m.source_units or m.target_units:
            line += f" [{m.source_units or '?'} → {m.target_units or '?'}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(no mappings)"


def _format_source_metadata(
    assignment: Any,
    context: dict[str, Any],
) -> str:
    """Format source metadata for the assembly prompt."""
    sg = next(
        (g for g in context["groups"] if g["id"] == assignment.source_id),
        {},
    )
    if not sg:
        return "(no source metadata)"
    parts = [
        f"Source: {sg.get('id', '')}",
        f"Key: {sg.get('group_key', '')}",
        f"Members: {sg.get('member_count', 0)}",
        f"Domain: {sg.get('physics_domain', '')}",
    ]
    desc = sg.get("rep_description") or sg.get("description")
    if desc:
        parts.append(f"Description: {desc}")
    return "\n".join(parts)


def discover_assembly(
    facility: str,
    ids_name: str,
    sections: SectionAssignmentBatch,
    signal_batches: list[SignalMappingBatch],
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> AssemblyBatch:
    """Discover assembly patterns for each section."""
    logger.info(
        "Discovering assembly patterns for %d sections",
        len(sections.assignments),
    )

    configs: list[AssemblyConfig] = []

    for assignment, batch in zip(sections.assignments, signal_batches):
        section_path = assignment.imas_section_path

        section_structure = fetch_imas_subtree(
            ids_name,
            section_path.removeprefix(f"{ids_name}/"),
            gc=gc,
        )

        prompt = _render_prompt(
            "assembly",
            facility=facility,
            ids_name=ids_name,
            section_path=section_path,
            signal_mappings=_format_signal_mappings(batch),
            imas_section_structure=_format_subtree(section_structure),
            source_metadata=_format_source_metadata(assignment, context),
        )

        messages = [
            {"role": "system", "content": "You are an IMAS assembly expert."},
            {"role": "user", "content": prompt},
        ]

        config = _call_llm(
            messages,
            AssemblyConfig,
            model=model,
            step_name=f"discover_assembly_{section_path}",
            cost=cost,
        )
        configs.append(config)

    return AssemblyBatch(ids_name=ids_name, configs=configs)


def validate_mappings(
    facility: str,
    ids_name: str,
    dd_version: str,
    sections: SectionAssignmentBatch,
    field_batches: list[SignalMappingBatch],
    *,
    gc: GraphClient,
) -> ValidatedMappingResult:
    """Assemble and programmatically validate all signal mappings.

    Runs concrete checks via ``validate_mapping()``: source/target existence,
    transform execution, unit compatibility, and duplicate target detection.
    """
    from imas_codex.ids.validation import validate_mapping

    logger.info("Running programmatic validation")

    # Assemble bindings + escalations from Step 2 batches
    all_bindings: list[ValidatedSignalMapping] = []
    all_escalations: list[EscalationFlag] = []
    for batch in field_batches:
        for m in batch.mappings:
            all_bindings.append(
                ValidatedSignalMapping(
                    source_id=m.source_id,
                    source_property=m.source_property,
                    target_id=m.target_id,
                    transform_expression=m.transform_expression,
                    source_units=m.source_units,
                    target_units=m.target_units,
                    cocos_label=m.cocos_label,
                    confidence=m.confidence,
                )
            )
        all_escalations.extend(batch.escalations)

    # Run programmatic validation
    from imas_codex.ids.tools import get_sign_flip_paths

    flip_paths = get_sign_flip_paths(ids_name)
    report = validate_mapping(all_bindings, gc=gc, sign_flip_paths=flip_paths)
    all_escalations.extend(report.escalations)

    corrections: list[str] = []
    if report.duplicate_targets:
        corrections.append(
            f"Duplicate targets detected: {', '.join(report.duplicate_targets)}"
        )
    if not report.all_passed:
        failed = sum(
            1
            for c in report.binding_checks
            if not (
                c.source_exists
                and c.target_exists
                and c.transform_executes
                and c.units_compatible
            )
        )
        corrections.append(f"{failed} binding(s) failed validation checks")

    return ValidatedMappingResult(
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        sections=sections.assignments,
        bindings=all_bindings,
        escalations=all_escalations,
        corrections=corrections,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class MappingResult:
    """Result of the mapping pipeline."""

    mapping_id: str
    validated: ValidatedMappingResult
    assembly: AssemblyBatch | None
    cost: PipelineCost
    persisted: bool = False
    unassigned_groups: list[str] = field(default_factory=list)


def generate_mapping(
    facility: str,
    ids_name: str,
    *,
    model: str | None = None,
    dd_version: str | None = None,
    persist: bool = True,
    activate: bool = True,
    gc: GraphClient | None = None,
) -> MappingResult:
    """Generate IMAS signal mapping via multi-step LLM pipeline.

    Steps:
        1. Gather signal sources + DD context (programmatic)
        2. LLM assigns sources to IMAS sections
        3. For each section, LLM generates signal mappings
        4. Programmatic validation (source/target existence, transforms, units)
        5. (Optional) Persist to graph

    Args:
        facility: Facility name (e.g., "jet").
        ids_name: IDS name (e.g., "pf_active").
        model: LLM model override (default: language tier).
        dd_version: DD version override (default: from settings).
        persist: Whether to persist results to graph.
        activate: Whether to promote status to 'active' after persisting.
        gc: GraphClient instance (created if None).

    Returns:
        MappingResult with validated mappings and cost breakdown.
    """
    if gc is None:
        gc = GraphClient()

    if dd_version is None:
        from imas_codex import dd_version as default_dd

        dd_version = default_dd

    cost = PipelineCost()

    # Step 0: Gather context
    context = gather_context(facility, ids_name, gc=gc)

    if not context["groups"]:
        raise ValueError(
            f"No signal sources found for {facility}/{ids_name}. "
            "Run signal discovery first."
        )

    # Step 1: Assign sections
    sections = assign_sections(
        facility, ids_name, context, model=model, cost=cost
    )

    if not sections.assignments:
        raise ValueError(
            f"LLM could not assign any signal sources to IMAS sections "
            f"for {facility}/{ids_name}."
        )

    # Step 2: Signal mappings
    field_batches = map_signals(
        facility,
        ids_name,
        sections,
        context,
        gc=gc,
        model=model,
        cost=cost,
    )

    # Step 3: Assembly discovery (LLM)
    assembly = discover_assembly(
        facility,
        ids_name,
        sections,
        field_batches,
        context,
        gc=gc,
        model=model,
        cost=cost,
    )

    # Step 4: Programmatic validation (no LLM call)
    validated = validate_mappings(
        facility,
        ids_name,
        dd_version,
        sections,
        field_batches,
        gc=gc,
    )

    # Persist
    mapping_id = f"{facility}:{ids_name}"
    persisted = False
    if persist:
        status = "active" if activate else "generated"
        mapping_id = persist_mapping_result(
            validated, assembly=assembly, gc=gc, status=status
        )
        persisted = True
        logger.info("Persisted mapping %s with status '%s'", mapping_id, status)

    logger.info(
        "Pipeline complete: %d bindings, %d escalations, $%.4f total",
        len(validated.bindings),
        len(validated.escalations),
        cost.total_usd,
    )

    return MappingResult(
        mapping_id=mapping_id,
        validated=validated,
        assembly=assembly,
        cost=cost,
        persisted=persisted,
        unassigned_groups=sections.unassigned_groups,
    )
