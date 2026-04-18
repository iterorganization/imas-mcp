"""SN enrich pipeline orchestrator.

Wires the EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST
workers into the generic discovery engine and runs them with supervision
and progress tracking.

The enrich pipeline runs *after* the generate pipeline completes for a
corpus.  It takes ``review_status='named'`` StandardName nodes and
enriches them with documentation, descriptions, and cross-reference
links produced by LLM calls.

Usage::

    from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

    state = StandardNameEnrichState(
        facility="dd",
        domain="equilibrium",
        cost_limit=2.0,
        dry_run=True,
    )
    await run_sn_enrich_engine(state)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.standard_names.enrich_state import StandardNameEnrichState
from imas_codex.standard_names.enrich_workers import (
    enrich_contextualise_worker,
    enrich_document_worker,
    enrich_extract_worker,
    enrich_persist_worker,
    enrich_validate_worker,
)

logger = logging.getLogger(__name__)


async def run_sn_enrich_engine(
    state: StandardNameEnrichState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN enrich pipeline.

    Pipeline::

        EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST

    Extract queries the graph for ``review_status='named'`` StandardName
    nodes, builds enrichment batches.
    Contextualise gathers DD docs, nearby SNs, domain siblings.
    Document uses an LLM to generate descriptions and documentation.
    Validate checks spelling, link integrity, description quality.
    Persist writes enriched data and REFERENCES relationships to graph.

    Args:
        state: Populated ``StandardNameEnrichState`` with filter config.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """

    # Downstream workers should run to completion even when cost limit is hit.
    # They only stop on CLI shutdown (stop_requested), not on budget.
    def _downstream_should_stop():
        return state.stop_requested

    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            enrich_extract_worker,
        ),
        WorkerSpec(
            "contextualise",
            "contextualise_phase",
            enrich_contextualise_worker,
            depends_on=["extract_phase"],
        ),
        WorkerSpec(
            "document",
            "document_phase",
            enrich_document_worker,
            depends_on=["contextualise_phase"],
            should_stop_fn=_downstream_should_stop,
        ),
        WorkerSpec(
            "validate",
            "validate_phase",
            enrich_validate_worker,
            depends_on=["document_phase"],
            group="finalize",
            should_stop_fn=_downstream_should_stop,
        ),
        WorkerSpec(
            "persist",
            "persist_phase",
            enrich_persist_worker,
            depends_on=["validate_phase"],
            group="finalize",
            should_stop_fn=_downstream_should_stop,
        ),
    ]

    # The supervised loop's stop function must NOT check budget_exhausted.
    # Budget only stops document; downstream workers must run to completion.
    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
        stop_fn=lambda: state.stop_requested,
    )
