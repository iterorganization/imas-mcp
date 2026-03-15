"""Rich progress display for the signal mapping pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    build_pipeline_section,
    build_resource_section,
)
from imas_codex.ids.mapping import PipelineCost


@dataclass
class MappingProgressState:
    """Live state for the mapping pipeline progress display."""

    ids_name: str
    model: str | None = None
    # Pipeline progress
    sources_found: int = 0
    sections_assigned: int = 0
    sections_total: int = 0
    sections_mapped: int = 0
    sections_assembled: int = 0
    bindings_total: int = 0
    bindings_passed: int = 0
    escalations: int = 0
    # Step activity
    current_step: str = ""
    current_detail: str = ""
    # Cost tracking
    cost: PipelineCost = field(default_factory=PipelineCost)
    # Time tracking
    start_time: float = field(default_factory=time.time)
    deadline: float | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def deadline_expired(self) -> bool:
        return time.time() >= self.deadline if self.deadline else False


class MappingProgressDisplay(BaseProgressDisplay):
    """Rich progress display for the signal mapping pipeline.

    Shows five stages: CONTEXT, SECTIONS, MAPPING, ASSEMBLY, VALIDATE.
    """

    def __init__(
        self,
        facility: str,
        ids_name: str,
        cost_limit: float,
        model: str | None = None,
        console: Any | None = None,
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            title_suffix=f"Signal Mapping — {ids_name}",
        )
        self.state = MappingProgressState(
            ids_name=ids_name,
            model=model,
        )

    def _header_mode_label(self) -> str | None:
        return None

    def _build_pipeline_section(self) -> Text:
        s = self.state
        rows: list[PipelineRowConfig] = []

        # CONTEXT stage
        context_done = s.sources_found > 0
        rows.append(
            PipelineRowConfig(
                name="CONTEXT",
                style="bold blue",
                completed=1 if context_done else 0,
                total=1,
                is_complete=context_done,
                description=(
                    f"{s.sources_found} sources, "
                    f"{s.sections_total or '?'} sections"
                    if context_done
                    else ""
                ),
            )
        )

        # SECTIONS stage
        sections_done = s.sections_assigned > 0
        rows.append(
            PipelineRowConfig(
                name="SECTIONS",
                style="bold green",
                completed=s.sections_assigned,
                total=max(s.sections_total, s.sections_assigned, 1),
                is_complete=sections_done,
                cost=s.cost.steps.get("assign_targets"),
                description=(
                    f"{s.sections_assigned} assigned"
                    if sections_done
                    else ""
                ),
            )
        )

        # MAPPING stage
        mapping_total = max(s.sections_assigned, 1)
        mapping_done = s.sections_mapped >= mapping_total and s.sections_mapped > 0
        rows.append(
            PipelineRowConfig(
                name="MAPPING",
                style="bold yellow",
                completed=s.sections_mapped,
                total=mapping_total,
                is_complete=mapping_done,
                cost=sum(
                    v for k, v in s.cost.steps.items()
                    if k.startswith("map_signals_")
                ),
                description=(
                    f"{s.bindings_total} bindings"
                    if mapping_done
                    else (s.current_detail if s.current_step == "mapping" else "")
                ),
            )
        )

        # ASSEMBLY stage
        assembly_total = max(s.sections_assigned, 1)
        assembly_done = (
            s.sections_assembled >= assembly_total and s.sections_assembled > 0
        )
        rows.append(
            PipelineRowConfig(
                name="ASSEMBLY",
                style="bold magenta",
                completed=s.sections_assembled,
                total=assembly_total,
                is_complete=assembly_done,
                cost=sum(
                    v for k, v in s.cost.steps.items()
                    if k.startswith("discover_assembly_")
                ),
                description=(
                    s.current_detail if s.current_step == "assembly" else ""
                ),
            )
        )

        # VALIDATE stage
        validate_done = s.bindings_passed > 0 or (
            s.bindings_total > 0 and s.current_step == "done"
        )
        rows.append(
            PipelineRowConfig(
                name="VALIDATE",
                style="bold cyan",
                completed=1 if validate_done else 0,
                total=1,
                is_complete=validate_done,
                description=(
                    f"{s.bindings_passed} passed"
                    + (f", {s.escalations} escalations" if s.escalations else "")
                    if validate_done
                    else ""
                ),
            )
        )

        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        config = ResourceConfig(
            elapsed=self.elapsed,
            run_cost=self.state.cost.total_usd or None,
            cost_limit=self.cost_limit,
        )
        return build_resource_section(config, self.gauge_width)
