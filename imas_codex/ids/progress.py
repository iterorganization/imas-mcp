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
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
)
from imas_codex.ids.mapping import PipelineCost

# ---------------------------------------------------------------------------
# Typed stream items for display rows 2-3
# ---------------------------------------------------------------------------


@dataclass
class ContextItem:
    """Streamed from context worker: source/domain discovery."""

    detail: str = ""
    sources: int = 0
    domains: int = 0


@dataclass
class AssignItem:
    """Streamed from assign worker: source → target path assignment."""

    source_id: str = ""
    target_path: str = ""
    physics_domain: str = ""


@dataclass
class MapItem:
    """Streamed from map worker: source → IMAS binding."""

    source_id: str = ""
    source_property: str = ""
    target_path: str = ""
    physics_domain: str = ""
    bindings: int = 0


@dataclass
class AssemblyItem:
    """Streamed from assembly worker: pattern discovery."""

    target_path: str = ""
    pattern: str = ""
    physics_domain: str = ""


@dataclass
class ValidateItem:
    """Streamed from validate worker: binding validation."""

    target_path: str = ""
    passed: int = 0
    escalations: int = 0


# ---------------------------------------------------------------------------
# Progress state
# ---------------------------------------------------------------------------


@dataclass
class MappingProgressState:
    """Live state for the mapping pipeline progress display.

    Tracks both per-IDS pipeline progress and multi-IDS iteration state.
    """

    # Multi-IDS tracking
    ids_targets: list[str] = field(default_factory=list)
    current_ids_idx: int = 0
    completed_ids: list[dict] = field(default_factory=list)

    model: str | None = None

    # Per-IDS pipeline progress (reset between IDS)
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

    # Cost tracking (cumulative across all IDS)
    cost: PipelineCost = field(default_factory=PipelineCost)

    # Time tracking
    start_time: float = field(default_factory=time.time)
    deadline: float | None = None

    # Streaming queues — rate-limited for smooth display
    context_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(rate=1.0, max_rate=3.0)
    )
    assign_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(rate=0.8, max_rate=2.0)
    )
    map_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5,
            max_rate=2.0,
            min_display_time=0.5,
            max_display_time=3.0,
        )
    )
    assembly_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(rate=0.8, max_rate=2.0)
    )
    validate_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(rate=0.8, max_rate=2.0)
    )

    # Current display items (popped from queues by tick)
    current_context: ContextItem | None = None
    current_assign: AssignItem | None = None
    current_map: MapItem | None = None
    current_assembly: AssemblyItem | None = None
    current_validate: ValidateItem | None = None

    @property
    def current_ids(self) -> str:
        if self.ids_targets and self.current_ids_idx < len(self.ids_targets):
            return self.ids_targets[self.current_ids_idx]
        return ""

    @property
    def total_ids(self) -> int:
        return len(self.ids_targets)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def deadline_expired(self) -> bool:
        return time.time() >= self.deadline if self.deadline else False

    def reset_for_ids(self, idx: int) -> None:
        """Reset per-IDS counters when starting a new IDS."""
        self.current_ids_idx = idx
        self.sources_found = 0
        self.sections_assigned = 0
        self.sections_total = 0
        self.sections_mapped = 0
        self.sections_assembled = 0
        self.bindings_total = 0
        self.bindings_passed = 0
        self.escalations = 0
        self.current_step = ""
        self.current_detail = ""
        # Clear stream items
        self.current_context = None
        self.current_assign = None
        self.current_map = None
        self.current_assembly = None
        self.current_validate = None
        # Clear queues
        self.context_queue.clear()
        self.assign_queue.clear()
        self.map_queue.clear()
        self.assembly_queue.clear()
        self.validate_queue.clear()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


class MappingProgressDisplay(BaseProgressDisplay):
    """Rich progress display for the signal mapping pipeline.

    Shows five pipeline stages (CONTEXT, SECTIONS, MAPPING, ASSEMBLY,
    VALIDATE) for the current IDS, with a header showing multi-IDS
    progress. Each stage streams structured items on rows 2-3.
    """

    def __init__(
        self,
        facility: str,
        ids_targets: list[str],
        cost_limit: float,
        model: str | None = None,
        console: Any | None = None,
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            title_suffix="Signal Mapping",
        )
        self.state = MappingProgressState(
            ids_targets=ids_targets,
            model=model,
        )

    def _header_mode_label(self) -> str | None:
        return None

    def _build_pipeline_section(self) -> Text:
        s = self.state
        rows: list[PipelineRowConfig] = []

        # --- CONTEXT stage ---
        context_done = s.sources_found > 0
        ctx = s.current_context
        rows.append(
            PipelineRowConfig(
                name="CONTEXT",
                style="bold blue",
                completed=1 if context_done else 0,
                total=1,
                is_complete=context_done,
                description=(
                    f"{s.sources_found} sources, {s.sections_total or '?'} sections"
                    if context_done
                    else ""
                ),
                is_processing=s.current_step == "context" and not context_done,
                processing_label=ctx.detail if ctx else "gathering context...",
            )
        )

        # --- SECTIONS stage ---
        sections_done = s.sections_assigned > 0
        asg = s.current_assign
        rows.append(
            PipelineRowConfig(
                name="SECTIONS",
                style="bold green",
                completed=s.sections_assigned,
                total=max(s.sections_total, s.sections_assigned, 1),
                is_complete=sections_done,
                cost=s.cost.steps.get("assign_targets"),
                primary_text=asg.source_id if asg else "",
                physics_domain=asg.physics_domain if asg else "",
                description=(
                    f"→ {asg.target_path}"
                    if asg
                    else (f"{s.sections_assigned} assigned" if sections_done else "")
                ),
                is_processing=s.current_step == "assign" and not sections_done,
                processing_label="assigning sources...",
            )
        )

        # --- MAPPING stage (rich content: source → target) ---
        mapping_total = max(s.sections_assigned, 1)
        mapping_done = s.sections_mapped >= mapping_total and s.sections_mapped > 0
        mp = s.current_map
        map_primary = ""
        map_desc = ""
        map_domain = ""
        if mp:
            map_primary = f"{mp.source_id}"
            map_desc = f"→ {mp.target_path}"
            if mp.bindings:
                map_desc += f" ({mp.bindings} bindings)"
            map_domain = mp.physics_domain
        elif mapping_done:
            map_desc = f"{s.bindings_total} bindings"
        rows.append(
            PipelineRowConfig(
                name="MAPPING",
                style="bold yellow",
                completed=s.sections_mapped,
                total=mapping_total,
                is_complete=mapping_done,
                cost=sum(
                    v for k, v in s.cost.steps.items() if k.startswith("map_signals_")
                ),
                primary_text=map_primary,
                physics_domain=map_domain,
                description=map_desc,
                is_processing=s.current_step == "mapping" and not mapping_done,
                processing_label="mapping signals...",
            )
        )

        # --- ASSEMBLY stage ---
        assembly_total = max(s.sections_assigned, 1)
        assembly_done = (
            s.sections_assembled >= assembly_total and s.sections_assembled > 0
        )
        asm = s.current_assembly
        rows.append(
            PipelineRowConfig(
                name="ASSEMBLY",
                style="bold magenta",
                completed=s.sections_assembled,
                total=assembly_total,
                is_complete=assembly_done,
                cost=sum(
                    v
                    for k, v in s.cost.steps.items()
                    if k.startswith("discover_assembly_")
                ),
                primary_text=asm.target_path if asm else "",
                physics_domain=asm.physics_domain if asm else "",
                description=(asm.pattern if asm else ""),
                is_processing=s.current_step == "assembly" and not assembly_done,
                processing_label="discovering patterns...",
            )
        )

        # --- VALIDATE stage ---
        validate_done = s.bindings_passed > 0 or (
            s.bindings_total > 0 and s.current_step == "done"
        )
        val = s.current_validate
        rows.append(
            PipelineRowConfig(
                name="VALIDATE",
                style="bold cyan",
                completed=1 if validate_done else 0,
                total=1,
                is_complete=validate_done,
                primary_text=val.target_path if val else "",
                description=(
                    f"{s.bindings_passed} passed"
                    + (f", {s.escalations} escalations" if s.escalations else "")
                    if validate_done
                    else ""
                ),
                is_processing=s.current_step == "validate" and not validate_done,
                processing_label="validating...",
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

    def tick(self) -> None:
        """Drain streaming queues for smooth display."""
        s = self.state

        # Context queue
        if item := s.context_queue.pop():
            s.current_context = ContextItem(
                detail=item.get("detail", ""),
                sources=item.get("sources", 0),
                domains=item.get("domains", 0),
            )
        elif s.context_queue.is_stale():
            s.current_context = None

        # Assign queue
        if item := s.assign_queue.pop():
            s.current_assign = AssignItem(
                source_id=item.get("source_id", ""),
                target_path=item.get("target_path", ""),
                physics_domain=item.get("physics_domain", ""),
            )
        elif s.assign_queue.is_stale():
            s.current_assign = None

        # Map queue
        if item := s.map_queue.pop():
            s.current_map = MapItem(
                source_id=item.get("source_id", ""),
                source_property=item.get("source_property", ""),
                target_path=item.get("target_path", ""),
                physics_domain=item.get("physics_domain", ""),
                bindings=item.get("bindings", 0),
            )
        elif s.map_queue.is_stale():
            s.current_map = None

        # Assembly queue
        if item := s.assembly_queue.pop():
            s.current_assembly = AssemblyItem(
                target_path=item.get("target_path", ""),
                pattern=item.get("pattern", ""),
                physics_domain=item.get("physics_domain", ""),
            )
        elif s.assembly_queue.is_stale():
            s.current_assembly = None

        # Validate queue
        if item := s.validate_queue.pop():
            s.current_validate = ValidateItem(
                target_path=item.get("target_path", ""),
                passed=item.get("passed", 0),
                escalations=item.get("escalations", 0),
            )
        elif s.validate_queue.is_stale():
            s.current_validate = None

        self._refresh()
