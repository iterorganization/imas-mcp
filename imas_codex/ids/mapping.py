"""IMAS mapping pipeline orchestrator.

Multi-step LLM pipeline that generates field-level IMAS mappings
from facility signal groups:

  Step 0: Gather signal groups + DD context
  Step 1: LLM assigns groups to IMAS sections
  Step 2: For each section, LLM generates field mappings
  Step 3: LLM validates and finalises
  Persist: Write to graph

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
    FieldMappingBatch,
    SectionAssignmentBatch,
    ValidatedMappingResult,
    persist_mapping_result,
)
from imas_codex.ids.tools import (
    analyze_units,
    check_imas_paths,
    fetch_imas_fields,
    fetch_imas_subtree,
    get_sign_flip_paths,
    query_signal_groups,
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


def _format_groups(groups: list[dict[str, Any]]) -> str:
    """Format signal groups into a readable summary."""
    lines: list[str] = []
    for g in groups:
        gid = g.get("id", "")
        key = g.get("group_key", "")
        desc = g.get("description", "")
        members = g.get("member_count", 0)
        existing = g.get("imas_mappings", [])
        mapped = [m for m in existing if m.get("target_id")]
        line = f"- {gid} (key={key}, members={members})"
        if desc:
            line += f": {desc}"
        if mapped:
            targets = ", ".join(m["target_id"] for m in mapped)
            line += f" [already mapped → {targets}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(no groups)"


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
    """Run unit analysis between groups and IMAS fields."""
    lines: list[str] = []
    # Collect signal units from groups
    for g in groups:
        signal_unit = None
        for kw in g.get("keywords") or []:
            if kw and kw.startswith("unit:"):
                signal_unit = kw[5:]
        if not signal_unit:
            continue
        for f in fields:
            imas_unit = f.get("units")
            if imas_unit:
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


def _step0_gather_context(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient,
) -> dict[str, Any]:
    """Step 0: Gather all context needed for the pipeline."""
    logger.info("Step 0: gathering context for %s/%s", facility, ids_name)

    # Don't filter by ids_name — Step 1 assigns groups to IDS sections.
    # Filtering here would require pre-existing MAPS_TO_IMAS relationships,
    # creating a chicken-and-egg problem for new mappings.
    groups = query_signal_groups(facility, gc=gc)
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


def _step1_assign_sections(
    facility: str,
    ids_name: str,
    context: dict[str, Any],
    *,
    model: str | None = None,
    cost: PipelineCost,
) -> SectionAssignmentBatch:
    """Step 1: Assign signal groups to IMAS sections."""
    logger.info("Step 1: assigning signal groups to IMAS sections")

    prompt = _render_prompt(
        "exploration",
        facility=facility,
        ids_name=ids_name,
        signal_groups=_format_groups(context["groups"]),
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
        step_name="step1_sections",
        cost=cost,
    )


def _step2_field_mappings(
    facility: str,
    ids_name: str,
    sections: SectionAssignmentBatch,
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> list[FieldMappingBatch]:
    """Step 2: Generate field mappings per section."""
    logger.info(
        "Step 2: generating field mappings for %d sections", len(sections.assignments)
    )

    batches: list[FieldMappingBatch] = []

    for assignment in sections.assignments:
        section_path = assignment.imas_section_path

        # Get detailed fields for this section
        fields = fetch_imas_fields(ids_name, [section_path], gc=gc)
        # Also get leaf fields under this section
        subtree_fields = fetch_imas_subtree(
            ids_name, section_path.removeprefix(f"{ids_name}/"), gc=gc, leaf_only=True
        )

        # Find the signal group details
        sg_detail = next(
            (g for g in context["groups"] if g["id"] == assignment.signal_group_id),
            {},
        )

        prompt = _render_prompt(
            "field_mapping",
            facility=facility,
            ids_name=ids_name,
            section_path=section_path,
            signal_group_detail=json.dumps(sg_detail, indent=2, default=str),
            imas_fields=_format_fields(subtree_fields or fields),
            unit_analysis=_format_unit_analysis(
                [sg_detail] if sg_detail else [], subtree_fields or fields
            ),
            cocos_paths="\n".join(f"- {p}" for p in context["cocos_paths"]) or "(none)",
            existing_mappings=json.dumps(context["existing"], indent=2, default=str),
        )

        messages = [
            {"role": "system", "content": "You are an IMAS mapping expert."},
            {"role": "user", "content": prompt},
        ]

        batch = _call_llm(
            messages,
            FieldMappingBatch,
            model=model,
            step_name=f"step2_fields_{section_path}",
            cost=cost,
        )
        batches.append(batch)

    return batches


def _step3_validate(
    facility: str,
    ids_name: str,
    dd_version: str,
    sections: SectionAssignmentBatch,
    field_batches: list[FieldMappingBatch],
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> ValidatedMappingResult:
    """Step 3: Validate and finalize all mappings."""
    logger.info("Step 3: validating mappings")

    # Collect all proposed mappings
    all_mappings = []
    all_escalations = []
    for batch in field_batches:
        all_mappings.extend([m.model_dump() for m in batch.mappings])
        all_escalations.extend([e.model_dump() for e in batch.escalations])

    # Validate paths exist
    target_paths = list({m["target_id"] for m in all_mappings})
    validation_results = check_imas_paths(target_paths, gc=gc)

    prompt = _render_prompt(
        "validation",
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        proposed_mappings=json.dumps(all_mappings, indent=2, default=str),
        validation_results=json.dumps(validation_results, indent=2, default=str),
        existing_mappings=json.dumps(context["existing"], indent=2, default=str),
        escalations=json.dumps(all_escalations, indent=2, default=str),
    )

    messages = [
        {"role": "system", "content": "You are an IMAS mapping validator."},
        {"role": "user", "content": prompt},
    ]

    return _call_llm(
        messages,
        ValidatedMappingResult,
        model=model,
        step_name="step3_validation",
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class MappingResult:
    """Result of the mapping pipeline."""

    mapping_id: str
    validated: ValidatedMappingResult
    cost: PipelineCost
    persisted: bool = False


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
    """Generate IMAS mapping via multi-step LLM pipeline.

    Steps:
        0. Gather signal groups + DD context
        1. LLM assigns groups to IMAS sections
        2. For each section, LLM generates field mappings
        3. LLM validates and finalizes
        4. (Optional) Persist to graph

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
    context = _step0_gather_context(facility, ids_name, gc=gc)

    if not context["groups"]:
        raise ValueError(
            f"No signal groups found for {facility}/{ids_name}. "
            "Run signal discovery first."
        )

    # Step 1: Assign sections
    sections = _step1_assign_sections(
        facility, ids_name, context, model=model, cost=cost
    )

    if not sections.assignments:
        raise ValueError(
            f"LLM could not assign any signal groups to IMAS sections "
            f"for {facility}/{ids_name}."
        )

    # Step 2: Field mappings
    field_batches = _step2_field_mappings(
        facility,
        ids_name,
        sections,
        context,
        gc=gc,
        model=model,
        cost=cost,
    )

    # Step 3: Validate
    validated = _step3_validate(
        facility,
        ids_name,
        dd_version,
        sections,
        field_batches,
        context,
        gc=gc,
        model=model,
        cost=cost,
    )

    # Persist
    mapping_id = f"{facility}:{ids_name}"
    persisted = False
    if persist:
        status = "active" if activate else "generated"
        mapping_id = persist_mapping_result(validated, gc=gc, status=status)
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
        cost=cost,
        persisted=persisted,
    )
