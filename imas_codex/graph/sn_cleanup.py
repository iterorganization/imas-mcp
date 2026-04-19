"""StandardName graph cleanup helpers for plan 31 (rc13 → rc14).

Reusable Cypher helpers for WS-F graph-remediation operations described in
``plans/features/standard-names/31-quality-bootstrap-v2.md`` §8 (F.1–F.6):

* F.1 — purge quarantined StandardNames matching WS-A exclusion rules
  (representation artefacts, pulse_schedule/reference sentinels,
  /diamagnetic axis leaves).
* F.3 — consolidate the ``wave_absorbed_power`` family under a parent SN
  via ``PART_OF`` edges.
* F.4 — consolidate 12 metric-tensor components under
  ``metric_tensor_component`` via ``PART_OF`` edges.
* F.5 — normalize gas-injection segment ordering and add ``REFERENCES``
  aliases.
* F.6 — populate ``NEAR_DUPLICATE_OF`` edges for known silent duplicates
  (research §2.12).

Per ``AGENTS.md`` graph migrations are invoked as inline Cypher via the
REPL / CLI — these functions are idempotent helpers, not a CLI or a
one-off script. Call sites pass a :class:`GraphClient` instance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient


PULSE_REFERENCE_RE = re.compile(r"pulse_schedule/.+/reference(_waveform)?(/.+)?$")
DIAMAGNETIC_AXIS_RE = re.compile(r"/diamagnetic(/[^/]+)?$")


@dataclass(frozen=True)
class PurgeCandidate:
    """A quarantined StandardName that matches a WS-A exclusion rule."""

    sn_id: str
    reasons: tuple[str, ...]
    source_paths: tuple[str, ...]


def collect_purge_candidates(client: GraphClient) -> list[PurgeCandidate]:
    """Return quarantined StandardName ids that F.1 should delete.

    Matches any SN with ``validation_status='quarantined'`` whose:

    * ``validation_issues`` contain ``representation_artifact_check``
      or ``diamagnetic_component_check``, **or**
    * ``source_paths`` include a ``pulse_schedule/.../reference[...]``
      entry, **or** a path whose final segment is ``/diamagnetic`` or a
      single leaf under ``/diamagnetic/<leaf>``.
    """
    rows = client.query(
        """
        MATCH (sn:StandardName {validation_status: 'quarantined'})
        RETURN sn.id AS id,
               coalesce(sn.validation_issues, []) AS issues,
               coalesce(sn.source_paths, []) AS paths
        """
    )
    candidates: list[PurgeCandidate] = []
    for row in rows:
        reasons: list[str] = []
        issues_joined = "\n".join(row["issues"])
        if "representation_artifact_check" in issues_joined:
            reasons.append("representation_artifact_audit")
        if "diamagnetic_component_check" in issues_joined:
            reasons.append("diamagnetic_component_audit")
        for raw in row["paths"]:
            path = raw[3:] if raw.startswith("dd:") else raw
            if PULSE_REFERENCE_RE.match(path):
                reasons.append("pulse_schedule_reference_path")
                break
        for raw in row["paths"]:
            path = raw[3:] if raw.startswith("dd:") else raw
            if DIAMAGNETIC_AXIS_RE.search(path):
                reasons.append("diamagnetic_axis_path")
                break
        if reasons:
            candidates.append(
                PurgeCandidate(
                    sn_id=row["id"],
                    reasons=tuple(reasons),
                    source_paths=tuple(row["paths"]),
                )
            )
    return candidates


def purge_standard_names(client: GraphClient, ids: list[str]) -> int:
    """Detach-delete StandardName nodes by id. Returns count deleted."""
    if not ids:
        return 0
    res = client.query(
        """
        UNWIND $ids AS nid
        MATCH (sn:StandardName {id: nid})
        DETACH DELETE sn
        RETURN count(*) AS deleted
        """,
        ids=ids,
    )
    return int(res[0]["deleted"]) if res else 0


def add_part_of_family(
    client: GraphClient,
    parent_id: str,
    parent_props: dict,
    child_ids: list[str],
) -> dict:
    """Create/merge a parent StandardName and attach PART_OF edges from
    each existing child to the parent. Idempotent.

    Children that do not exist in the graph are skipped and returned under
    ``missing`` so callers can surface them without silently dropping.
    """
    # Merge parent.
    client.query(
        """
        MERGE (p:StandardName {id: $pid})
        ON CREATE SET p += $props, p.created_at = datetime()
        ON MATCH  SET p += $props
        """,
        pid=parent_id,
        props=parent_props,
    )
    res = client.query(
        """
        UNWIND $cids AS cid
        OPTIONAL MATCH (c:StandardName {id: cid})
        WITH cid, c
        WHERE c IS NOT NULL
        MATCH (p:StandardName {id: $pid})
        MERGE (c)-[:PART_OF]->(p)
        RETURN collect(cid) AS linked
        """,
        pid=parent_id,
        cids=child_ids,
    )
    linked = set(res[0]["linked"]) if res else set()
    return {
        "parent": parent_id,
        "linked": sorted(linked),
        "missing": sorted(set(child_ids) - linked),
    }


def add_near_duplicate_edges(
    client: GraphClient, pairs: list[tuple[str, str, float]]
) -> int:
    """Add undirected ``NEAR_DUPLICATE_OF`` edges (stored as a single
    directed edge id_a → id_b with weight). Idempotent via MERGE."""
    created = 0
    for a, b, weight in pairs:
        res = client.query(
            """
            MATCH (x:StandardName {id: $a}), (y:StandardName {id: $b})
            MERGE (x)-[r:NEAR_DUPLICATE_OF]-(y)
            ON CREATE SET r.weight = $w, r.created_at = datetime()
            ON MATCH  SET r.weight = $w
            RETURN count(r) AS n
            """,
            a=a,
            b=b,
            w=float(weight),
        )
        if res:
            created += int(res[0]["n"])
    return created
