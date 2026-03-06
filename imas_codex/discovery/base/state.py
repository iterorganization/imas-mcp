"""Base discovery state dataclass.

Provides the common structural skeleton shared by all domain-specific
discovery state classes. Domains extend this with their own fields,
phases, and stopping logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class DiscoveryStateBase:
    """Common fields and utilities for all discovery state objects.

    Subclasses add domain-specific fields (phases, stats, limits) and
    override ``should_stop()`` with domain-appropriate logic.

    Convention: subclass ``__post_init__`` should call
    ``super().__post_init__()`` if it exists to ensure base setup runs.
    """

    facility: str
    cost_limit: float = 10.0
    deadline: float | None = None
    stop_requested: bool = False
    service_monitor: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Common properties
    # ------------------------------------------------------------------

    @property
    def deadline_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() >= self.deadline

    @property
    def total_cost(self) -> float:
        """Override in subclass to sum domain-specific cost stats."""
        return 0.0

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------

    def should_stop(self) -> bool:
        """Base stopping logic — stop_requested or deadline expired.

        Subclasses should call ``super().should_stop()`` first, then add
        domain-specific checks (budget, phase completion, etc.).
        """
        if self.stop_requested:
            return True
        if self.deadline_expired:
            return True
        return False

    # ------------------------------------------------------------------
    # Service health
    # ------------------------------------------------------------------

    async def await_services(self) -> bool:
        """Block until all critical services are healthy.

        Returns True when services are ready, False if monitor was stopped.
        No-op (returns True immediately) when no service_monitor is set.
        """
        if self.service_monitor is None:
            return True
        return await self.service_monitor.await_services_ready()

    # ------------------------------------------------------------------
    # Phase / stats introspection
    # ------------------------------------------------------------------

    @property
    def all_phases(self) -> dict[str, PipelinePhase]:
        """Collect all PipelinePhase fields for iteration/refresh."""
        return {
            name: val
            for name, val in vars(self).items()
            if isinstance(val, PipelinePhase)
        }

    @property
    def all_stats(self) -> dict[str, WorkerStats]:
        """Collect all WorkerStats fields for aggregation."""
        return {
            name: val
            for name, val in vars(self).items()
            if isinstance(val, WorkerStats)
        }
