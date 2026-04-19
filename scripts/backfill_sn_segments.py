#!/usr/bin/env python3
"""Back-fill HAS_SEGMENT edges for existing named StandardName nodes.

The persist_worker pipeline (``imas_codex.standard_names.graph_ops._write_segment_edges``)
emits ``(sn:StandardName)-[:HAS_SEGMENT {position, segment}]->(t:GrammarToken)``
edges for every newly persisted SN. This script is a one-shot catch-up for
corpora that predate that wiring.

Dry-run (default):
    uv run python scripts/backfill_sn_segments.py

Apply edges:
    uv run python scripts/backfill_sn_segments.py --apply

Idempotent: per-SN the existing ``_write_segment_edges`` helper deletes any
stale HAS_SEGMENT edges before re-inserting, so re-running produces zero
net change. Parse failures (e.g. quarantined names) are logged and skipped —
the SN node is left intact.

Depends on:
    - ``imas_standard_names.grammar.parse_standard_name`` for decomposition
    - ``imas_standard_names.graph.spec.segment_edge_specs`` for edge layout
    - ``imas_codex.standard_names.graph_ops._write_segment_edges`` for writes
    - Active ``ISNGrammarVersion`` with populated GrammarToken vocabulary
      (run ``imas-codex graph sync-isn-grammar`` first).

Plan reference: 29-architectural-pivot.md Phase E (E.7 back-fill).
"""

from __future__ import annotations

import argparse
import logging
import sys

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import _write_segment_edges

logger = logging.getLogger("backfill_sn_segments")


REVIEW_STATUSES = ("named", "enriched", "reviewable")


def _fetch_named_sn_ids(gc: GraphClient) -> list[str]:
    """Return ids of StandardName nodes in review_status ∈ REVIEW_STATUSES."""
    rows = (
        gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.review_status IN $statuses "
            "RETURN sn.id AS id ORDER BY sn.id",
            statuses=list(REVIEW_STATUSES),
        )
        or []
    )
    return [row["id"] for row in rows]


def _preflight(gc: GraphClient) -> tuple[int, int]:
    """Return (parseable_count, edges_planned) for the named corpus.

    Does not touch the graph; uses ISN's parser to estimate the delta.
    """
    try:
        from imas_standard_names.grammar import parse_standard_name
        from imas_standard_names.graph.spec import segment_edge_specs
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"imas_standard_names not importable: {exc}") from exc

    ids = _fetch_named_sn_ids(gc)
    parseable = 0
    edges = 0
    for sn_id in ids:
        try:
            parsed = parse_standard_name(sn_id)
            specs = segment_edge_specs(parsed)
        except Exception:
            continue
        parseable += 1
        edges += len(specs)
    return parseable, edges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write edges. Without this flag the script reports expected delta only.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    with GraphClient() as gc:
        # Preflight: sanity-check active grammar + parseability
        active = list(
            gc.query(
                "MATCH (v:ISNGrammarVersion {active: true}) RETURN v.version AS version"
            )
            or []
        )
        if not active:
            print(
                "ERROR: no ISNGrammarVersion{active:true} — "
                "run `imas-codex graph sync-isn-grammar` first.",
                file=sys.stderr,
            )
            return 2
        active_version = active[0]["version"]

        ids = _fetch_named_sn_ids(gc)
        parseable, planned_edges = _preflight(gc)

        existing = list(
            gc.query(
                "MATCH (:StandardName)-[r:HAS_SEGMENT]->(:GrammarToken) "
                "RETURN count(r) AS c"
            )
            or []
        )[0]["c"]

        print(f"Active ISN grammar version: {active_version}")
        print(f"Named StandardNames (status ∈ {REVIEW_STATUSES}): {len(ids)}")
        print(
            f"Parseable via ISN grammar: {parseable}  "
            f"({100 * parseable / max(1, len(ids)):.1f}%)"
        )
        print(f"Existing HAS_SEGMENT edges: {existing}")
        print(f"Planned HAS_SEGMENT edges after back-fill: {planned_edges}")

        if not args.apply:
            print("\nDry-run only. Re-run with --apply to write edges.")
            return 0

        # _write_segment_edges is idempotent per-name (deletes stale edges
        # before re-inserting). Call once with the full name list; the
        # function logs token-miss warnings internally.
        _write_segment_edges(gc, ids)

        after = list(
            gc.query(
                "MATCH (:StandardName)-[r:HAS_SEGMENT]->(:GrammarToken) "
                "RETURN count(r) AS c"
            )
            or []
        )[0]["c"]
        coverage = list(
            gc.query(
                "MATCH (sn:StandardName) WHERE sn.review_status IN $statuses "
                "OPTIONAL MATCH (sn)-[r:HAS_SEGMENT]->(:GrammarToken) "
                "WITH sn, count(r) AS n "
                "RETURN sum(CASE WHEN n > 0 THEN 1 ELSE 0 END) AS covered, "
                "       count(sn) AS total",
                statuses=list(REVIEW_STATUSES),
            )
            or []
        )[0]

        print(f"\nHAS_SEGMENT edges written: {after} (was {existing})")
        print(
            f"SN coverage: {coverage['covered']}/{coverage['total']} "
            f"({100 * coverage['covered'] / max(1, coverage['total']):.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
