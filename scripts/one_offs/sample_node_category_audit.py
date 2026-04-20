"""Plan 32 Phase 1 — sample paths for node_category false-negative audit.

Samples DD paths stratified by (ids_name × node_category) and emits a
JSON file that can be fed to the human reviewer (or an LLM judge) to
identify paths that *should* be Standard Name candidates but are
currently filtered out by ``SN_SOURCE_CATEGORIES = {quantity, geometry}``.

Outputs ``plans/research/standard-names/node-category-audit-samples.json``
with this schema::

    {
      "generated_at": "2025-...",
      "categories_tested": ["quantity", "geometry", "coordinate",
                            "identifier", "meta"],
      "samples_per_stratum": 10,
      "samples": [
        {
          "path": "equilibrium/time_slice/profiles_1d/psi",
          "ids": "equilibrium",
          "node_category": "quantity",
          "node_type": "dynamic",
          "description": "...",
          "unit": "Wb",
          "physics_domain": "equilibrium",
          "would_be_sn": true,
          "reviewer_verdict": ""
        },
        ...
      ]
    }

The reviewer fills in ``reviewer_verdict`` (``keep``, ``drop``,
``borderline``) for each row. If > 2% of non-SN_SOURCE_CATEGORIES paths
are flagged ``keep``, Phase 1 exit criteria says to widen
``SN_SOURCE_CATEGORIES``.

Run::

    uv run python scripts/one_offs/sample_node_category_audit.py \\
        --samples-per-stratum 10 \\
        --output plans/research/standard-names/node-category-audit-samples.json

This script is **read-only** (SELECT-style queries only); safe to run
against the live graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# Categories we want to probe. The first two are the current SN-eligible
# set; the remaining three are what Phase 1 investigates for FN paths.
CATEGORIES = [
    "quantity",
    "geometry",
    "coordinate",
    "identifier",
    "metadata",
    "representation",
    "fit_artifact",
]

# Top IDSs by path count — stratify samples so the reviewer sees a
# representative slice rather than e.g. 200 random edge_profiles paths.
STRATA_IDS = [
    "equilibrium",
    "core_profiles",
    "edge_profiles",
    "core_sources",
    "mhd",
    "magnetics",
    "interferometer",
    "thomson_scattering",
    "wall",
    "pf_active",
]


def _sample_paths(gc: GraphClient, ids_name: str, category: str, n: int) -> list[dict]:
    """Fetch up to ``n`` random paths for a given (ids, category) stratum."""
    rows = list(
        gc.query(
            """
            MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS {id: $ids_name})
            WHERE n.node_category = $category
              AND n.node_type IN ['dynamic', 'constant']
              AND trim(coalesce(n.description, '')) <> ''
            RETURN n.id AS path,
                   ids.id AS ids,
                   n.node_category AS node_category,
                   n.node_type AS node_type,
                   n.description AS description,
                   n.units AS unit,
                   n.physics_domain AS physics_domain
            """,
            ids_name=ids_name,
            category=category,
        )
    )
    if len(rows) <= n:
        return rows
    return random.sample(rows, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-per-stratum",
        type=int,
        default=10,
        help="Paths to sample per (ids × category) cell (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plans/research/standard-names/node-category-audit-samples.json"),
        help="Output JSON file (default: plans/research/standard-names/…).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(args.seed)

    samples: list[dict] = []
    with GraphClient() as gc:
        for ids_name in STRATA_IDS:
            for category in CATEGORIES:
                batch = _sample_paths(gc, ids_name, category, args.samples_per_stratum)
                for row in batch:
                    row["would_be_sn"] = category in ("quantity", "geometry")
                    row["reviewer_verdict"] = ""  # to be filled by reviewer
                samples.extend(batch)
                logger.info(
                    "ids=%s category=%s → %d samples",
                    ids_name,
                    category,
                    len(batch),
                )

    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "categories_tested": CATEGORIES,
        "samples_per_stratum": args.samples_per_stratum,
        "strata_ids": STRATA_IDS,
        "seed": args.seed,
        "samples": samples,
        "reviewer_schema": {
            "keep": (
                "path SHOULD become a Standard Name "
                "(currently filtered out → false negative)"
            ),
            "drop": "path correctly excluded (not an SN candidate)",
            "borderline": "ambiguous; depends on convention",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=False))

    # Summary
    by_cat = dict.fromkeys(CATEGORIES, 0)
    for s in samples:
        by_cat[s["node_category"]] += 1
    logger.info("Wrote %d samples to %s", len(samples), args.output)
    for c, n in by_cat.items():
        in_sn = "(SN)" if c in ("quantity", "geometry") else "(non-SN)"
        logger.info("  %-12s %-8s %d", c, in_sn, n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
