"""End-to-end tests for the signals discovery pipeline.

Tests run sequentially against the live TCV facility using CLI subprocess
calls (to avoid async event-loop issues) and sync graph queries for
validation. A module-scoped fixture runs MDSplus scan once for all tests.

Requirements:
    - Live Neo4j instance (auto-skipped if unavailable)
    - SSH access to TCV (auto-skipped if unavailable)

Run explicitly:
    uv run pytest tests/integration/test_signals_e2e.py -m slow -v
"""

from __future__ import annotations

import subprocess

import pytest

from imas_codex.graph.client import GraphClient

# ── Markers ──────────────────────────────────────────────────────────────
pytestmark = [pytest.mark.graph, pytest.mark.slow, pytest.mark.integration]

FACILITY = "tcv"


# ── Connectivity checks ─────────────────────────────────────────────────

_neo4j_ok: bool | None = None
_ssh_ok: bool | None = None


def _check_neo4j() -> bool:
    global _neo4j_ok
    if _neo4j_ok is not None:
        return _neo4j_ok
    try:
        with GraphClient() as gc:
            gc.get_stats()
        _neo4j_ok = True
    except Exception:
        _neo4j_ok = False
    return _neo4j_ok


def _check_ssh() -> bool:
    global _ssh_ok
    if _ssh_ok is not None:
        return _ssh_ok
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                FACILITY,
                "echo ok",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _ssh_ok = result.returncode == 0
    except Exception:
        _ssh_ok = False
    return _ssh_ok


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip E2E tests when Neo4j or SSH is unavailable."""
    if not _check_neo4j():
        skip = pytest.mark.skip(reason="Neo4j not available")
        for item in items:
            item.add_marker(skip)
        return
    if not _check_ssh():
        skip = pytest.mark.skip(reason=f"SSH to {FACILITY} not available")
        for item in items:
            item.add_marker(skip)


# ── Helpers ──────────────────────────────────────────────────────────────


def run_cli_signals(
    scanners: str,
    *,
    signal_limit: int = 30,
    scan_only: bool = True,
    timeout: int = 1200,
) -> subprocess.CompletedProcess:
    """Run signal discovery via CLI subprocess."""
    cmd = [
        "uv",
        "run",
        "imas-codex",
        "discover",
        "signals",
        FACILITY,
        "-s",
        scanners,
        "-n",
        str(signal_limit),
    ]
    if scan_only:
        cmd.append("--scan-only")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def clear_signals() -> dict[str, int]:
    """Clear all signal discovery data via Python API."""
    from imas_codex.discovery.signals.parallel import clear_facility_signals

    return clear_facility_signals(FACILITY)


def query_graph(cypher: str, **params) -> list[dict]:
    """Execute a Cypher query against the live graph."""
    with GraphClient() as gc:
        return gc.query(cypher, **params)


def get_signal_count() -> int:
    result = query_graph(
        "MATCH (s:FacilitySignal {facility_id: $f}) RETURN count(s) AS cnt",
        f=FACILITY,
    )
    return result[0]["cnt"] if result else 0


def get_data_node_count() -> int:
    result = query_graph(
        "MATCH (n:DataNode {facility_id: $f}) RETURN count(n) AS cnt",
        f=FACILITY,
    )
    return result[0]["cnt"] if result else 0


# ── Module fixture: ensure MDSplus data exists ───────────────────────────


@pytest.fixture(scope="module")
def mdsplus_data():
    """Ensure MDSplus signal data exists in the graph.

    If FacilitySignals already exist (from previous runs), skip the scan.
    Otherwise, run the CLI to populate. This fixture runs once per module.
    """
    if get_signal_count() == 0:
        result = run_cli_signals("mdsplus", signal_limit=50, timeout=1200)
        assert result.returncode == 0, (
            f"MDSplus scan CLI failed:\nstdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
    # Verify data was created
    count = get_signal_count()
    assert count > 0, "No FacilitySignals after MDSplus scan"
    return {"signal_count": count}


# ── E2E-1: MDSplus scan creates nodes ────────────────────────────────────


@pytest.mark.timeout(60)
def test_01_data_nodes_exist(mdsplus_data):
    """E2E-1: MDSplus scan creates DataNode nodes with required properties."""
    data_nodes = query_graph(
        """
        MATCH (n:DataNode {facility_id: $f})
        RETURN n.path AS path, n.data_source_name AS data_source_name,
               n.facility_id AS facility_id
        LIMIT 20
        """,
        f=FACILITY,
    )
    assert len(data_nodes) > 0, "No DataNode nodes"
    for n in data_nodes:
        assert n["path"] is not None, f"Missing path: {n}"
        assert n["data_source_name"] is not None, f"Missing data_source_name: {n}"
        assert n["facility_id"] == FACILITY, f"Bad facility_id: {n}"


@pytest.mark.timeout(60)
def test_02_facility_signals_exist(mdsplus_data):
    """E2E-1: MDSplus scan creates FacilitySignal nodes with required properties."""
    signals = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})
        RETURN s.id AS id, s.status AS status, s.accessor AS accessor,
               s.facility_id AS facility_id
        LIMIT 20
        """,
        f=FACILITY,
    )
    assert len(signals) > 0, "No FacilitySignal nodes"
    for s in signals:
        assert s["facility_id"] == FACILITY
        assert s["status"] is not None
        assert s["accessor"] is not None
        assert s["id"] is not None


# ── E2E-2: MDSplus relationships and versions ───────────────────────────


@pytest.mark.timeout(60)
def test_03_source_node_edges(mdsplus_data):
    """E2E-2: FacilitySignals have HAS_DATA_SOURCE_NODE edges to DataNodes."""
    result = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})-[:HAS_DATA_SOURCE_NODE]->(n:DataNode)
        RETURN count(s) AS cnt
        """,
        f=FACILITY,
    )
    assert result[0]["cnt"] > 0, "No HAS_DATA_SOURCE_NODE edges"


@pytest.mark.timeout(60)
def test_04_data_access_edges(mdsplus_data):
    """E2E-2: FacilitySignals have DATA_ACCESS edges."""
    result = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})-[:DATA_ACCESS]->(da:DataAccess)
        RETURN count(s) AS cnt
        """,
        f=FACILITY,
    )
    assert result[0]["cnt"] > 0, "No DATA_ACCESS edges"


@pytest.mark.timeout(60)
def test_05_at_facility_edges(mdsplus_data):
    """E2E-2: FacilitySignals have AT_FACILITY edges to correct Facility."""
    result = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})
              -[:AT_FACILITY]->(fac:Facility {id: $f})
        RETURN count(s) AS cnt
        """,
        f=FACILITY,
    )
    assert result[0]["cnt"] > 0, "No AT_FACILITY edges"

    # No edges to wrong facility
    wrong = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})-[:AT_FACILITY]->(fac:Facility)
        WHERE fac.id <> $f
        RETURN count(s) AS cnt
        """,
        f=FACILITY,
    )
    assert wrong[0]["cnt"] == 0, "AT_FACILITY edges point to wrong facility"


@pytest.mark.timeout(60)
def test_06_tree_model_versions(mdsplus_data):
    """E2E-2: MDSplus scan creates StructuralEpoch for versioned trees."""
    versions = query_graph(
        """
        MATCH (v:StructuralEpoch {facility_id: $f})
        RETURN v.data_source_name AS data_source_name, v.version AS version,
               v.facility_id AS facility_id
        """,
        f=FACILITY,
    )
    assert len(versions) > 0, "No StructuralEpoch nodes"
    for v in versions:
        assert v["data_source_name"] is not None
        assert v["facility_id"] == FACILITY


@pytest.mark.timeout(60)
def test_07_signal_id_format(mdsplus_data):
    """E2E-2: Signal IDs follow the {facility}: prefix format."""
    signal_ids = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})
        RETURN s.id AS id
        LIMIT 20
        """,
        f=FACILITY,
    )
    for s in signal_ids:
        assert s["id"].startswith(f"{FACILITY}:"), (
            f"Signal ID doesn't start with '{FACILITY}:': {s['id']}"
        )


# ── Graph structural invariants ──────────────────────────────────────────


@pytest.mark.timeout(60)
def test_08_no_orphaned_data_access(mdsplus_data):
    """No DataAccess nodes without connected FacilitySignal."""
    orphans = query_graph(
        """
        MATCH (da:DataAccess {facility_id: $f})
        WHERE NOT EXISTS { MATCH (da)<-[:DATA_ACCESS]-(:FacilitySignal) }
        RETURN count(da) AS cnt
        """,
        f=FACILITY,
    )
    assert orphans[0]["cnt"] == 0, "Orphaned DataAccess nodes found"


@pytest.mark.timeout(60)
def test_09_unique_signal_ids(mdsplus_data):
    """All FacilitySignal IDs are unique."""
    dupes = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})
        WITH s.id AS id, count(*) AS cnt
        WHERE cnt > 1
        RETURN id, cnt
        """,
        f=FACILITY,
    )
    assert len(dupes) == 0, f"Duplicate signal IDs: {dupes}"


@pytest.mark.timeout(60)
def test_10_all_signals_have_facility_edge(mdsplus_data):
    """Every FacilitySignal has an AT_FACILITY edge."""
    missing = query_graph(
        """
        MATCH (s:FacilitySignal {facility_id: $f})
        WHERE NOT EXISTS { MATCH (s)-[:AT_FACILITY]->(:Facility) }
        RETURN count(s) AS cnt
        """,
        f=FACILITY,
    )
    assert missing[0]["cnt"] == 0, "Signals missing AT_FACILITY edge"


# ── E2E-7: Clear signals ────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_11_clear_signals():
    """Clear removes all FacilitySignal and related nodes.

    Runs last — validates the clear_facility_signals cleanup function.
    """
    # Ensure data exists to clear
    if get_signal_count() == 0:
        pytest.skip("No signals to clear (previous tests may have failed)")

    result = clear_signals()
    assert result["signals_deleted"] > 0

    assert get_signal_count() == 0, "Signals not fully cleared"

    tmv = query_graph(
        """
        MATCH (v:StructuralEpoch {facility_id: $f})
        RETURN count(v) AS cnt
        """,
        f=FACILITY,
    )
    assert tmv[0]["cnt"] == 0, "StructuralEpoch not cleaned up"
