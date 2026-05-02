#!/usr/bin/env python3
"""Backfill StandardName HAS_UNIT for SNs that were silent-coerced to "1".

Plan reference: plan-mcp-and-units.md Track B step B6.

Identifies StandardName nodes where:
  - ``sn.unit = '1'`` AND there is a HAS_UNIT edge to ``Unit{id:'1'}``
  - At least one StandardNameSource produced the SN via PRODUCED_NAME
  - The DD-side IMASNode (via FROM_DD_PATH) has a HAS_UNIT relationship
  - All such DD source units agree on a single value
  - That single value is **not** ``'1'`` (so it's a real coercion, not a
    legitimate dimensionless name)

For each row that meets the WHERE-selected subset:
  - Delete the ``(sn)-[:HAS_UNIT]->(:Unit{id:'1'})`` edge
  - MERGE the correct ``Unit`` and ``HAS_UNIT`` edge
  - Set ``sn.unit`` to the correct string

Acceptance per plan: 100% of WHERE-selected SNs corrected, 0 silent
failures. Non-corrected rows are logged with a reason.

Dry-run (default):
    uv run python scripts/backfill_unit_invariant.py

Apply fixes:
    uv run python scripts/backfill_unit_invariant.py --apply

Print per-SN log only:
    uv run python scripts/backfill_unit_invariant.py --apply --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imas_codex.graph.client import GraphClient

ARTEFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "plans"
    / "research"
    / "standard-names"
    / "backfill-unit-invariant.json"
)


CANDIDATES_QUERY = """
MATCH (sn:StandardName)-[:HAS_UNIT]->(:Unit{id:'1'})
WHERE sn.unit = '1'
MATCH (sn)<-[:PRODUCED_NAME]-(:StandardNameSource)-[:FROM_DD_PATH]->(n:IMASNode)
MATCH (n)-[:HAS_UNIT]->(u:Unit)
WITH sn, collect(DISTINCT u.id) AS dd_units
WHERE size(dd_units) = 1
  AND dd_units[0] <> '1'
RETURN sn.id AS sn_id, dd_units[0] AS correct_unit
ORDER BY sn_id
"""


def find_candidates(gc: GraphClient) -> list[dict]:
    rows = gc.query(CANDIDATES_QUERY)
    return [{"sn_id": r["sn_id"], "correct_unit": r["correct_unit"]} for r in rows]


def count_unit_one_baseline(gc: GraphClient) -> int:
    """Count SNs that currently have unit='1' and HAS_UNIT->Unit{id:'1'}."""
    rows = gc.query(
        """
        MATCH (sn:StandardName)-[:HAS_UNIT]->(:Unit{id:'1'})
        WHERE sn.unit = '1'
        RETURN count(sn) AS c
        """
    )
    return rows[0]["c"] if rows else 0


def apply_one(gc: GraphClient, sn_id: str, correct_unit: str) -> bool:
    """Apply the fix for one SN; return True iff the row was updated."""
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit)
        DELETE r
        WITH sn
        MERGE (u:Unit {id: $unit})
        MERGE (sn)-[:HAS_UNIT]->(u)
        SET sn.unit = $unit
        RETURN sn.id AS sn_id
        """,
        sn_id=sn_id,
        unit=correct_unit,
    )
    return bool(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gc = GraphClient()
    try:
        before = count_unit_one_baseline(gc)
        print(f"BEFORE: SNs with unit='1' and HAS_UNIT->Unit{{id:'1'}}: {before}")

        candidates = find_candidates(gc)
        print(f"WHERE-selected (fixable) candidates: {len(candidates)}")

        if args.verbose or not args.apply:
            for c in candidates[:50]:
                print(f"  {c['sn_id']}: 1 -> {c['correct_unit']}")
            if len(candidates) > 50:
                print(f"  ... and {len(candidates) - 50} more")

        ARTEFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        artefact = {
            "before": before,
            "candidates": candidates,
            "applied": [],
            "failed": [],
        }

        if args.apply:
            print()
            for c in candidates:
                ok = apply_one(gc, c["sn_id"], c["correct_unit"])
                if ok:
                    artefact["applied"].append(c)
                    if args.verbose:
                        print(f"  APPLIED: {c['sn_id']} -> {c['correct_unit']}")
                else:
                    artefact["failed"].append(c)
                    print(
                        f"  FAILED:  {c['sn_id']} -> {c['correct_unit']} "
                        "(node not updated; check existence)"
                    )

            after = count_unit_one_baseline(gc)
            artefact["after"] = after
            print()
            print(
                f"APPLIED: {len(artefact['applied'])} / {len(candidates)} "
                f"(failed: {len(artefact['failed'])})"
            )
            print(f"AFTER:  SNs with unit='1' and HAS_UNIT->Unit{{id:'1'}}: {after}")
        else:
            print()
            print("Dry-run — pass --apply to write changes.")

        ARTEFACT_PATH.write_text(json.dumps(artefact, indent=2) + "\n")
        print(f"Wrote artefact: {ARTEFACT_PATH}")
        # Acceptance: 100% of WHERE-selected — non-zero failed exits non-zero.
        if args.apply and artefact["failed"]:
            return 2
    finally:
        gc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
