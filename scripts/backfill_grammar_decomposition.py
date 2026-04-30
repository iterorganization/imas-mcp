"""Phase-1 self-backfill: re-enter ``_write_grammar_decomposition`` for every
existing StandardName so the new bare-name segment columns and the
``grammar_parse_fallback`` flag are populated graph-wide.

Plan 40 §4.3 — idempotent. Re-running on already-populated nodes overwrites
columns and refreshes typed edges. Names whose grammar narrowed since last
write will have stale columns cleared.

Usage::

    uv run python scripts/backfill_grammar_decomposition.py
    uv run python scripts/backfill_grammar_decomposition.py --batch-size 50
    uv run python scripts/backfill_grammar_decomposition.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import _write_grammar_decomposition

logger = logging.getLogger(__name__)


def _list_standard_name_ids(gc: GraphClient, limit: int | None = None) -> list[str]:
    cypher = "MATCH (sn:StandardName) RETURN sn.id AS id ORDER BY sn.id"
    if limit is not None:
        cypher += f" LIMIT {int(limit)}"
    rows = gc.query(cypher) or []
    return [r["id"] for r in rows if r.get("id")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Names per batch (default 50)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total names processed (debug helper).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with GraphClient() as gc:
        all_ids = _list_standard_name_ids(gc, limit=args.limit)
        logger.info("Backfilling grammar decomposition for %d names", len(all_ids))

        total_gaps = 0
        for i in range(0, len(all_ids), args.batch_size):
            batch = all_ids[i : i + args.batch_size]
            gaps = _write_grammar_decomposition(gc, batch)
            total_gaps += len(gaps)
            logger.info(
                "Batch %d-%d/%d done (%d token-miss gaps so far)",
                i,
                i + len(batch),
                len(all_ids),
                total_gaps,
            )

    logger.info("Backfill complete. Total token-miss gaps: %d", total_gaps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
