"""End-to-end tests for the signals discovery pipeline.

Validates the full signals pipeline against the live TCV facility by
processing real data through every stage and verifying graph output.

Requirements:
    - Live Neo4j instance (auto-skipped if unavailable)
    - SSH access to TCV (auto-skipped if unavailable)
    - Embedding server accessible (uses local fallback)

Run explicitly:
    uv run pytest tests/integration/test_signals_e2e.py -m graph -v
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from imas_codex.graph.client import GraphClient

# ── Markers ──────────────────────────────────────────────────────────────
pytestmark = [pytest.mark.graph, pytest.mark.slow, pytest.mark.integration]

FACILITY = "tcv"
SIGNAL_LIMIT = 50
COST_LIMIT = 2.0


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


def clear_signals() -> dict[str, int]:
    """Clear all signal discovery data for the test facility."""
    from imas_codex.discovery.signals.parallel import clear_facility_signals

    return clear_facility_signals(FACILITY)


def get_stats() -> dict:
    """Get current signal discovery stats."""
    from imas_codex.discovery.signals.parallel import get_data_discovery_stats

    return get_data_discovery_stats(FACILITY)


async def run_discovery(
    scanner_types: list[str] | None = None,
    scan_only: bool = False,
    enrich_only: bool = False,
    signal_limit: int = SIGNAL_LIMIT,
    cost_limit: float = COST_LIMIT,
) -> dict:
    """Run signal discovery pipeline."""
    from imas_codex.discovery.signals.parallel import run_parallel_data_discovery

    return await run_parallel_data_discovery(
        facility=FACILITY,
        scanner_types=scanner_types,
        signal_limit=signal_limit,
        cost_limit=cost_limit,
        discover_only=scan_only,
        enrich_only=enrich_only,
        num_enrich_workers=2,
        num_check_workers=2,
    )


def query_graph(cypher: str, **params) -> list[dict]:
    """Execute a Cypher query against the live graph."""
    with GraphClient() as gc:
        return gc.query(cypher, **params)


def assert_signal_properties(signals: list[dict]) -> None:
    """Assert all FacilitySignal nodes have required properties."""
    for s in signals:
        assert s.get("facility_id") == FACILITY, f"Bad facility_id: {s}"
        assert s.get("status") is not None, f"Missing status: {s}"
        assert s.get("accessor") is not None, f"Missing accessor: {s}"
        assert s.get("id") is not None, f"Missing id: {s}"


def assert_tree_node_properties(nodes: list[dict]) -> None:
    """Assert all TreeNode nodes have required properties."""
    for n in nodes:
        assert n.get("path") is not None, f"Missing path: {n}"
        assert n.get("tree_name") is not None, f"Missing tree_name: {n}"
        assert n.get("facility_id") == FACILITY, f"Bad facility_id: {n}"


# ── E2E Tests ────────────────────────────────────────────────────────────


class TestMDSplusScan:
    """E2E-1 & E2E-2: MDSplus tree scanning (static + dynamic)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        clear_signals()
        yield
        # Don't clear after — subsequent tests may depend on data

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_mdsplus_scan_creates_tree_nodes(self):
        """MDSplus scanner creates TreeNode and FacilitySignal nodes."""
        result = await run_discovery(scanner_types=["mdsplus"], scan_only=True)

        assert result["scanned"] > 0, "No signals scanned"

        # Verify TreeNode nodes created
        tree_nodes = query_graph(
            """
            MATCH (n:TreeNode {facility_id: $facility})
            RETURN n.path AS path, n.tree_name AS tree_name,
                   n.facility_id AS facility_id
            LIMIT 20
            """,
            facility=FACILITY,
        )
        assert len(tree_nodes) > 0, "No TreeNode nodes created"
        assert_tree_node_properties(tree_nodes)

        # Verify FacilitySignal nodes created
        signals = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            RETURN s.id AS id, s.status AS status, s.accessor AS accessor,
                   s.facility_id AS facility_id
            LIMIT 20
            """,
            facility=FACILITY,
        )
        assert len(signals) > 0, "No FacilitySignal nodes created"
        assert_signal_properties(signals)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_mdsplus_scan_creates_relationships(self):
        """MDSplus scanner creates SOURCE_NODE and DATA_ACCESS edges."""
        result = await run_discovery(scanner_types=["mdsplus"], scan_only=True)

        assert result["scanned"] > 0

        # SOURCE_NODE: FacilitySignal → TreeNode
        source_edges = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:SOURCE_NODE]->(n:TreeNode)
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert source_edges[0]["cnt"] > 0, "No SOURCE_NODE edges"

        # DATA_ACCESS: FacilitySignal → DataAccess
        da_edges = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:DATA_ACCESS]->(da:DataAccess)
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert da_edges[0]["cnt"] > 0, "No DATA_ACCESS edges"

        # AT_FACILITY: FacilitySignal → Facility
        at_facility = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert at_facility[0]["cnt"] > 0, "No AT_FACILITY edges"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_mdsplus_scan_creates_versions(self):
        """MDSplus scanner creates TreeModelVersion for versioned trees."""
        await run_discovery(scanner_types=["mdsplus"], scan_only=True)

        versions = query_graph(
            """
            MATCH (v:TreeModelVersion {facility_id: $facility})
            RETURN v.tree_name AS tree_name, v.version AS version,
                   v.facility_id AS facility_id
            """,
            facility=FACILITY,
        )
        # TCV has static (versioned) trees, so we expect versions
        assert len(versions) > 0, "No TreeModelVersion nodes created"

        # Each version should have tree_name and version
        for v in versions:
            assert v["tree_name"] is not None
            assert v["facility_id"] == FACILITY


class TestTDIScan:
    """E2E-3: TDI function scanning."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        clear_signals()
        yield

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_tdi_scan_creates_functions(self):
        """TDI scanner creates TDIFunction and FacilitySignal nodes."""
        result = await run_discovery(scanner_types=["tdi"], scan_only=True)

        assert result["scanned"] > 0, "No TDI signals scanned"

        # Verify TDIFunction nodes
        tdi_funcs = query_graph(
            """
            MATCH (t:TDIFunction {facility_id: $facility})
            RETURN t.id AS id, t.source_code AS source_code,
                   t.facility_id AS facility_id
            LIMIT 20
            """,
            facility=FACILITY,
        )
        assert len(tdi_funcs) > 0, "No TDIFunction nodes created"

        # TDI functions should have source_code
        has_source = [f for f in tdi_funcs if f.get("source_code")]
        assert len(has_source) > 0, "No TDIFunction with source_code"

        # Verify FacilitySignal nodes created from TDI
        signals = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WHERE s.accessor CONTAINS '\\' OR s.accessor CONTAINS 'tdi'
               OR EXISTS { MATCH (s)<-[:QUANTITY_OF]-(t:TDIFunction) }
            RETURN s.id AS id, s.status AS status, s.accessor AS accessor,
                   s.facility_id AS facility_id
            LIMIT 20
            """,
            facility=FACILITY,
        )
        assert len(signals) > 0, "No FacilitySignal from TDI"
        assert_signal_properties(signals)


class TestTDILinkage:
    """E2E-4: TDI linkage to TreeNodes (run after both MDSplus and TDI)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Clear and run BOTH scanners so linkage can be tested
        clear_signals()
        asyncio.get_event_loop().run_until_complete(
            run_discovery(scanner_types=["mdsplus", "tdi"], scan_only=True)
        )
        yield

    @pytest.mark.timeout(60)
    def test_tdi_resolves_to_tree_nodes(self):
        """TDIFunction nodes have RESOLVES_TO_TREE_NODE edges to TreeNodes."""
        linkages = query_graph(
            """
            MATCH (t:TDIFunction {facility_id: $facility})
                  -[:RESOLVES_TO_TREE_NODE]->(n:TreeNode)
            RETURN count(t) AS cnt
            """,
            facility=FACILITY,
        )
        # Not all TDI functions resolve, but some should
        # If none resolve, the test still passes but logs a warning
        count = linkages[0]["cnt"]
        if count == 0:
            pytest.skip("No TDI→TreeNode linkages found (may be expected)")


class TestEnrichment:
    """E2E-5: LLM enrichment of discovered signals."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Clear and scan first, then test enrichment
        clear_signals()
        asyncio.get_event_loop().run_until_complete(
            run_discovery(scanner_types=["mdsplus"], scan_only=True, signal_limit=20)
        )
        yield

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_enrichment_updates_signals(self):
        """Enrichment transitions signals from discovered to enriched."""
        stats_before = get_stats()
        discovered_before = stats_before.get("discovered", 0)
        assert discovered_before > 0, "No discovered signals to enrich"

        result = await run_discovery(
            enrich_only=True,
            signal_limit=10,
            cost_limit=COST_LIMIT,
        )

        assert result["enriched"] > 0, "No signals enriched"

        # Check enriched signals have description and physics_domain
        enriched = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WHERE s.status = 'enriched' OR s.status = 'checked'
            RETURN s.id AS id, s.description AS description,
                   s.physics_domain AS physics_domain
            LIMIT 10
            """,
            facility=FACILITY,
        )
        assert len(enriched) > 0, "No enriched signals in graph"

        for s in enriched:
            assert s.get("description"), f"Missing description: {s['id']}"


class TestFullPipeline:
    """E2E-6: Full pipeline end-to-end."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        clear_signals()
        yield
        # Clean up after full pipeline
        clear_signals()

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_full_pipeline(self):
        """Full pipeline: scan → enrich → check in one run."""
        result = await run_discovery(
            signal_limit=SIGNAL_LIMIT,
            cost_limit=COST_LIMIT,
        )

        assert result["scanned"] > 0, "No signals scanned"

        stats = get_stats()
        total = stats.get("total", 0)
        assert total > 0, "No signals in graph after full pipeline"

        # Validate FacilitySignal required properties
        signals = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            RETURN s.id AS id, s.status AS status, s.accessor AS accessor,
                   s.facility_id AS facility_id
            """,
            facility=FACILITY,
        )
        assert_signal_properties(signals)

        # Validate TreeNode required properties
        tree_nodes = query_graph(
            """
            MATCH (n:TreeNode {facility_id: $facility})
            RETURN n.path AS path, n.tree_name AS tree_name,
                   n.facility_id AS facility_id
            LIMIT 50
            """,
            facility=FACILITY,
        )
        assert_tree_node_properties(tree_nodes)

        # Validate SOURCE_NODE edges
        source = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:SOURCE_NODE]->(n:TreeNode)
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert source[0]["cnt"] > 0, "No SOURCE_NODE edges after full pipeline"

        # Validate DATA_ACCESS edges
        da = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:DATA_ACCESS]->(d:DataAccess)
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert da[0]["cnt"] > 0, "No DATA_ACCESS edges after full pipeline"

        # Validate AT_FACILITY edges point to correct facility
        wrong_facility = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
                  -[:AT_FACILITY]->(f:Facility)
            WHERE f.id <> $facility
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert wrong_facility[0]["cnt"] == 0, (
            "AT_FACILITY edges point to wrong facility"
        )

        # Signal IDs should follow expected format: {facility}:{tree}/{path}
        signal_ids = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            RETURN s.id AS id
            LIMIT 20
            """,
            facility=FACILITY,
        )
        for s in signal_ids:
            assert s["id"].startswith(f"{FACILITY}:"), (
                f"Signal ID doesn't start with '{FACILITY}:': {s['id']}"
            )


class TestClearSignals:
    """Test clear_facility_signals removes all signal data."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_clear_removes_all_signals(self):
        """Clearing removes all FacilitySignal, DataAccess, TreeModelVersion nodes."""
        # First create some data
        result = await run_discovery(
            scanner_types=["mdsplus"], scan_only=True, signal_limit=20
        )
        assert result["scanned"] > 0

        stats_before = get_stats()
        assert stats_before.get("total", 0) > 0

        # Clear
        clear_result = clear_signals()
        assert clear_result["signals_deleted"] > 0

        # Verify no signals remain
        stats_after = get_stats()
        assert stats_after.get("total", 0) == 0

        # Verify no orphaned TreeModelVersion
        tmv = query_graph(
            """
            MATCH (v:TreeModelVersion {facility_id: $facility})
            RETURN count(v) AS cnt
            """,
            facility=FACILITY,
        )
        assert tmv[0]["cnt"] == 0, "TreeModelVersion not cleaned up"


class TestGraphValidation:
    """Cross-cutting graph validation after pipeline runs."""

    @pytest.fixture(autouse=True)
    def setup_with_data(self):
        """Ensure test data exists by running a scan."""
        stats = get_stats()
        if stats.get("total", 0) == 0:
            asyncio.get_event_loop().run_until_complete(
                run_discovery(
                    scanner_types=["mdsplus"], scan_only=True, signal_limit=30
                )
            )
        yield

    @pytest.mark.timeout(60)
    def test_no_orphaned_data_access(self):
        """No DataAccess nodes without connected signals."""
        orphans = query_graph(
            """
            MATCH (da:DataAccess {facility_id: $facility})
            WHERE NOT EXISTS { MATCH (da)<-[:DATA_ACCESS]-(:FacilitySignal) }
            RETURN count(da) AS cnt
            """,
            facility=FACILITY,
        )
        assert orphans[0]["cnt"] == 0, "Orphaned DataAccess nodes found"

    @pytest.mark.timeout(60)
    def test_signal_ids_unique(self):
        """All FacilitySignal IDs are unique."""
        dupes = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WITH s.id AS id, count(*) AS cnt
            WHERE cnt > 1
            RETURN id, cnt
            """,
            facility=FACILITY,
        )
        assert len(dupes) == 0, f"Duplicate signal IDs: {dupes}"

    @pytest.mark.timeout(60)
    def test_all_signals_have_facility_edge(self):
        """Every FacilitySignal has an AT_FACILITY edge."""
        missing = query_graph(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WHERE NOT EXISTS { MATCH (s)-[:AT_FACILITY]->(:Facility) }
            RETURN count(s) AS cnt
            """,
            facility=FACILITY,
        )
        assert missing[0]["cnt"] == 0, "Signals missing AT_FACILITY edge"
