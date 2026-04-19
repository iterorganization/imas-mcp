#!/usr/bin/env python3
"""Backfill StandardName HAS_UNIT edges to match canonical DD-path units.

Dry-run (default):
    uv run python scripts/backfill_sn_unit_mismatches.py

Apply fixes:
    uv run python scripts/backfill_sn_unit_mismatches.py --apply

Produces a JSON artefact at plans/research/standard-names/sn-unit-mismatches.json
listing all (StandardName, current_unit, canonical_unit, source_path) triples
where they disagree and the SN has only one source path (unambiguous case).
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
    / "sn-unit-mismatches.json"
)

# DD-side bugs: the SN unit is correct, the DD unit is wrong.
# These should NOT be backfilled — they need a DD rebuild.
DD_SIDE_BUGS = frozenset(
    {
        "electron_temperature_peaking_factor",
        "ion_average_charge_of_ion_state",
        "ion_average_square_charge_of_ion_state",
    }
)


def find_mismatches(gc: GraphClient) -> list[dict]:
    """Find all SNs whose unit disagrees with their DD source-path unit.

    Returns only unambiguous cases: SNs where all DD source paths
    agree on a single unit that differs from the SN unit.
    """
    rows = gc.query("""
        MATCH (sn:StandardName {validation_status: 'valid'})
        WHERE sn.unit IS NOT NULL AND sn.source_paths IS NOT NULL
        UNWIND sn.source_paths AS sp_raw
        WITH sn, replace(sp_raw, 'dd:', '') AS sp
        OPTIONAL MATCH (dd:IMASNode {id: sp})-[:HAS_UNIT]->(du:Unit)
        WITH sn.id AS name, sn.unit AS sn_unit,
             collect(DISTINCT du.id) AS raw_dd_units,
             collect(DISTINCT sp) AS source_paths
        WITH name, sn_unit,
             [x IN raw_dd_units WHERE x IS NOT NULL] AS dd_units,
             source_paths
        WHERE size(dd_units) = 1 AND NOT sn_unit IN dd_units
        RETURN name, sn_unit, dd_units[0] AS canonical_unit, source_paths
        ORDER BY name
    """)

    return [
        {
            "name": r["name"],
            "current_unit": r["sn_unit"],
            "canonical_unit": r["canonical_unit"],
            "source_paths": r["source_paths"],
            "dd_side_bug": r["name"] in DD_SIDE_BUGS,
        }
        for r in rows
    ]


def apply_fixes(gc: GraphClient, mismatches: list[dict]) -> int:
    """Rewrite HAS_UNIT edges for SN-side mismatches. Idempotent."""
    fixable = [m for m in mismatches if not m["dd_side_bug"]]
    if not fixable:
        print("No SN-side mismatches to fix.")
        return 0

    fixed = 0
    for m in fixable:
        name = m["name"]
        canonical = m["canonical_unit"]
        gc.query(
            """
            MATCH (sn:StandardName {id: $name})
            OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit)
            DELETE r
            WITH sn
            MERGE (u:Unit {id: $unit})
            MERGE (sn)-[:HAS_UNIT]->(u)
            SET sn.unit = $unit
            """,
            name=name,
            unit=canonical,
        )
        print(f"  Fixed: {name} — {m['current_unit']} → {canonical}")
        fixed += 1

    return fixed


def write_artefact(mismatches: list[dict]) -> None:
    """Write the mismatch report as a JSON artefact."""
    ARTEFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Flatten source_paths to first entry for readability
    artefact = [
        {
            "name": m["name"],
            "current_unit": m["current_unit"],
            "canonical_unit": m["canonical_unit"],
            "source_path": m["source_paths"][0] if m["source_paths"] else None,
            "dd_side_bug": m["dd_side_bug"],
        }
        for m in mismatches
    ]
    ARTEFACT_PATH.write_text(json.dumps(artefact, indent=2) + "\n")
    print(f"Wrote {len(artefact)} mismatches to {ARTEFACT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply fixes (rewrite HAS_UNIT edges). Default is dry-run.",
    )
    args = parser.parse_args()

    gc = GraphClient()
    try:
        mismatches = find_mismatches(gc)
        print(f"Found {len(mismatches)} unit mismatches:")
        for m in mismatches:
            tag = " [DD-side bug]" if m["dd_side_bug"] else " [SN-side bug]"
            print(
                f"  {m['name']}: current={m['current_unit']}, "
                f"canonical={m['canonical_unit']}{tag}"
            )

        write_artefact(mismatches)

        if args.apply:
            fixed = apply_fixes(gc, mismatches)
            print(f"\nApplied {fixed} fix(es).")
        else:
            fixable = sum(1 for m in mismatches if not m["dd_side_bug"])
            print(f"\nDry run — {fixable} fixable mismatch(es). Use --apply to fix.")
    finally:
        gc.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
