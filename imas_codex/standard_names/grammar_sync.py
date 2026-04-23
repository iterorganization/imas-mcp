"""Sync ISN grammar spec (segments, tokens, templates) to Neo4j.

The grammar spec is the canonical vocabulary for Standard Names and is
owned exclusively by the SN subsystem. This module provides the
library-level sync helper used by:

* ``sn sync-grammar`` CLI — manual / initial sync
* ``sn clear`` CLI — auto re-seed after a full subsystem wipe
* Release CLI — called during tag creation to stamp grammar

The spec is loaded from the installed ``imas_standard_names`` package.
Writes are idempotent — re-running is a no-op at the database level.

Two pieces of state are owned by imas-codex rather than ISN and live
here:

1. Composite ``id`` properties — the LinkML schema declares ``id`` as
   the identifier slot for every grammar node using a deterministic
   composite format (e.g. ``{version}:{segment}:{value}``). ISN's
   sync_grammar keys nodes on the natural composite ``(version, name)``
   but does not project into a single ``id`` slot — we project here.
2. ``active`` flag rotation — "which grammar version is the running
   composition pipeline using" is imas-codex pipeline state (ADR-8).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


_FINALISE_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "set ISNGrammarVersion.id",
        "MATCH (v:ISNGrammarVersion) "
        "WHERE v.id IS NULL AND v.version IS NOT NULL "
        "SET v.id = v.version",
    ),
    (
        "set GrammarSegment.id",
        "MATCH (s:GrammarSegment) "
        "WHERE s.id IS NULL AND s.version IS NOT NULL AND s.name IS NOT NULL "
        "SET s.id = s.version + ':' + s.name",
    ),
    (
        "set GrammarToken.id",
        "MATCH (t:GrammarToken) "
        "WHERE t.id IS NULL AND t.version IS NOT NULL "
        "  AND t.segment IS NOT NULL AND t.value IS NOT NULL "
        "SET t.id = t.version + ':' + t.segment + ':' + t.value",
    ),
    (
        "set GrammarTemplate.id",
        "MATCH (tpl:GrammarTemplate) "
        "WHERE tpl.id IS NULL AND tpl.version IS NOT NULL AND tpl.name IS NOT NULL "
        "SET tpl.id = tpl.version + ':template:' + tpl.name",
    ),
    (
        "rotate ISNGrammarVersion.active flag",
        "MATCH (v:ISNGrammarVersion) SET v.active = (v.version = $version)",
    ),
)


@dataclass
class GrammarSyncReport:
    """Result of a grammar sync run."""

    isn_version: str
    spec_version: str
    segments: int
    templates: int
    dry_run: bool
    applied: bool
    raw_report: dict[str, Any] = field(default_factory=dict)
    finalise_report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _finalise_active_version(
    gc: GraphClient, version: str, dry_run: bool
) -> dict[str, Any]:
    """Set composite ``id`` props + rotate ``active`` flag to ``version``."""
    report: dict[str, Any] = {"target_version": version, "applied": not dry_run}

    if dry_run:
        report["planned_statements"] = list(_FINALISE_STATEMENTS)
        return report

    for label, cypher in _FINALISE_STATEMENTS:
        gc.query(cypher, version=version)
        report[label] = "ok"
    return report


def sync_isn_grammar_to_graph(
    *,
    dry_run: bool = False,
    gc: GraphClient | None = None,
) -> GrammarSyncReport:
    """Sync the installed ISN grammar spec into Neo4j.

    Writes ``ISNGrammarVersion``, ``GrammarSegment``, ``GrammarToken``,
    ``GrammarTemplate`` nodes plus ``DEFINES`` / ``HAS_TOKEN`` / ``NEXT``
    / ``USES_TEMPLATE`` edges. Idempotent — safe to re-run.

    Parameters
    ----------
    dry_run:
        When True, return planned statements without touching the graph.
    gc:
        Optional open :class:`GraphClient`. When None, the function opens
        and closes its own client.

    Returns
    -------
    :class:`GrammarSyncReport` with ISN version, counts, and
    per-statement report.

    Raises
    ------
    RuntimeError
        If the ISN package is not installed or the sync fails.
    """
    try:
        from imas_standard_names import __version__ as isn_version
        from imas_standard_names.graph.spec import get_grammar_graph_spec
        from imas_standard_names.graph.sync import sync_grammar
    except ImportError as exc:
        raise RuntimeError(
            "imas_standard_names package not available — cannot sync grammar."
        ) from exc

    spec = get_grammar_graph_spec()
    spec_version = spec.get("version", "unknown")
    segments = len(spec["segments"])
    templates = len(spec["templates"])

    logger.info(
        "Sync ISN grammar: isn=%s spec=%s segments=%d templates=%d dry_run=%s",
        isn_version,
        spec_version,
        segments,
        templates,
        dry_run,
    )

    owns_client = gc is None
    client_cm: GraphClient | None = None
    try:
        if owns_client:
            client_cm = GraphClient()
            client_cm.__enter__()
            gc_local: GraphClient = client_cm
        else:
            assert gc is not None
            gc_local = gc

        report = sync_grammar(gc_local, active_version=isn_version, dry_run=dry_run)
        finalise_report = _finalise_active_version(
            gc_local, version=isn_version, dry_run=dry_run
        )
    except Exception as exc:  # noqa: BLE001 — surface as RuntimeError
        raise RuntimeError(f"Failed to sync grammar to Neo4j: {exc}") from exc
    finally:
        if owns_client and client_cm is not None:
            client_cm.__exit__(None, None, None)

    if dataclasses.is_dataclass(report):
        raw = dataclasses.asdict(report)
    elif hasattr(report, "__dict__"):
        raw = dict(report.__dict__)
    else:
        raw = dict(report)

    return GrammarSyncReport(
        isn_version=isn_version,
        spec_version=str(spec_version),
        segments=segments,
        templates=templates,
        dry_run=dry_run,
        applied=not dry_run,
        raw_report=raw,
        finalise_report=finalise_report,
    )
