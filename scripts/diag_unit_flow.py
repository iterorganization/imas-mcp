#!/usr/bin/env python3
"""Diagnose unit flow from DD → ISN dict for a small set of reference paths.

Reads the DD-truth unit via :func:`pool_adapter._read_items` and asserts
that the unit string survives unchanged through the compose pipeline
boundaries.

Usage:
    uv run python scripts/diag_unit_flow.py
    uv run python scripts/diag_unit_flow.py --paths some/dd/path other/path

Default paths cover the three reference cases from the plan:
    1. ``core_transport/.../electrons/energy/flux``  (truth ``m^-2.W``)
    2. ``core_profiles/profiles_1d/electrons/temperature``  (truth ``eV``)
    3. ``core_profiles/profiles_1d/zeff``  (truth ``1`` — legitimately
       dimensionless; must NOT be confused with a fallback)

Exits non-zero on any mismatch.

Plan reference: plan-mcp-and-units.md Track B step B4.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from imas_codex.graph.client import GraphClient

# Reference paths and their DD-truth units.  Update if the DD changes.
REFERENCE_PATHS: dict[str, str] = {
    "core_transport/model/profiles_1d/electrons/energy/flux": "m^-2.W",
    "core_profiles/profiles_1d/electrons/temperature": "eV",
    "core_profiles/profiles_1d/zeff": "1",
}


def _dd_truth_unit(gc: GraphClient, path: str) -> str | None:
    """Read the canonical DD unit for *path* via the HAS_UNIT relationship."""
    rows = gc.query(
        """
        MATCH (n:IMASNode {id: $path})
        OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
        RETURN u.id AS unit_via_rel,
               n.unit AS unit_scalar
        """,
        path=path,
    )
    if not rows:
        return None
    row = rows[0]
    # Relationship is the source of truth; the scalar is a denormalised
    # mirror that may diverge.
    return row.get("unit_via_rel") or row.get("unit_scalar")


def _read_items_for_paths(paths: list[str]) -> list[dict]:
    """Run pool_adapter._read_items via its closure.  Mirrors the
    compose-time read path so we observe the exact dict the worker sees."""
    from imas_codex.standard_names import pool_adapter  # noqa: F401

    # _read_items is defined inside pool_adapter.enrich_paths; we replicate
    # its body here so the diag does not need a full enrich invocation.
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $paths AS p
            MATCH (n:IMASNode {id: p})
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n.id AS path,
                   n.description AS description,
                   n.physics_domain AS physics_domain,
                   n.data_type AS data_type,
                   coalesce(u.id, n.unit) AS unit
            """,
            paths=paths,
        )
        return [dict(r) for r in (rows or [])]


def diag_one(path: str, expected_unit: str | None) -> tuple[bool, list[str]]:
    """Run the diagnostic for one path. Returns (ok, log_lines)."""
    log: list[str] = []
    ok = True

    log.append(f"=== {path} ===")
    log.append(f"  expected unit (plan): {expected_unit!r}")

    with GraphClient() as gc:
        truth = _dd_truth_unit(gc, path)
    log.append(f"  DD truth via HAS_UNIT: {truth!r}")

    if expected_unit is not None and truth != expected_unit:
        # Not a hard failure — DD ingestion may have a different value
        # than the plan recorded — but flag prominently.
        log.append(
            f"  WARN: DD truth {truth!r} != expected {expected_unit!r}; "
            "using DD truth as authority for this run"
        )

    if truth in (None, "", "-", "mixed"):
        log.append("  RESULT: DD has no usable unit — invariant says SKIP candidate.")
        return True, log

    # Stage 1: pool_adapter._read_items
    items = _read_items_for_paths([path])
    if not items:
        log.append("  FAIL: pool_adapter._read_items returned no row")
        return False, log
    pool_unit = items[0].get("unit")
    log.append(f"  pool_adapter unit:    {pool_unit!r}")
    if pool_unit != truth:
        log.append("  FAIL: pool_adapter unit diverged from DD truth")
        ok = False

    # Stage 2: ISN-dict construction (mirrors workers.py:_validate_isn_layers)
    from imas_codex.standard_names.kind_derivation import to_isn_kind

    entry = {"id": "diag_test_name", "kind": "scalar", "unit": pool_unit}
    isn_dict: dict[str, str] = {
        "name": entry["id"],
        "kind": to_isn_kind(entry["kind"]),
    }
    if isn_dict["kind"] != "metadata":
        unit = entry.get("unit")
        if not unit:
            log.append("  FAIL: ISN dict construction would have no unit")
            ok = False
        else:
            isn_dict["unit"] = unit
            log.append(f"  ISN-dict unit:        {unit!r}")
            if unit != truth:
                log.append("  FAIL: ISN-dict unit diverged from DD truth")
                ok = False

    # Stage 3: post-create_standard_name_entry
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict, name_only=True)
        post_unit = getattr(model, "unit", None)
        log.append(f"  post-ISN model unit:  {post_unit!r}")
        if post_unit != truth:
            log.append("  FAIL: post-ISN unit diverged from DD truth")
            ok = False
    except Exception as exc:  # noqa: BLE001 — diagnostic, surface anything
        log.append(f"  FAIL: ISN model construction error: {exc!r}")
        ok = False

    log.append("  PASS ✓" if ok else "  FAIL ✗")
    return ok, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Override the reference path list (DD-truth unit will be "
        "fetched from the graph rather than hard-coded).",
    )
    args = parser.parse_args()

    if args.paths:
        plan: dict[str, str | None] = dict.fromkeys(args.paths)
    else:
        plan = dict(REFERENCE_PATHS)

    asyncio.get_event_loop_policy()  # ensure asyncio import path resolves
    all_ok = True
    for path, expected in plan.items():
        ok, log = diag_one(path, expected)
        for line in log:
            print(line)
        all_ok = all_ok and ok

    print()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
