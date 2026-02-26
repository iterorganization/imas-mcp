"""Fixtures for graph-native MCP tests.

Provides a small hand-crafted DD graph loaded into a test Neo4j instance.
Tests auto-skip when Neo4j is not reachable. The fixture graph contains
a minimal but complete DD structure: DDVersion nodes, IDS, IMASPath hierarchy,
units, clusters, identifier schemas, and path changes.
"""

import os

import pytest
from dotenv import load_dotenv

# Load .env file for local Neo4j credentials
load_dotenv(override=False)

# Mark all tests in this directory as requiring graph
pytestmark = pytest.mark.graph_mcp

# ── Neo4j availability check ──────────────────────────────────────────────

_neo4j_available: bool | None = None


def _get_neo4j_params() -> dict[str, str]:
    """Get Neo4j connection parameters from environment or defaults."""
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "imas-codex"),
    }


def _make_client():
    """Create a GraphClient with explicit connection parameters."""
    from imas_codex.graph.client import GraphClient

    params = _get_neo4j_params()
    return GraphClient(
        uri=params["uri"],
        username=params["username"],
        password=params["password"],
        graph_name="test",
    )


def _check_neo4j_available() -> bool:
    """Quick probe to see if Neo4j is reachable (cached)."""
    global _neo4j_available
    if _neo4j_available is not None:
        return _neo4j_available
    try:
        client = _make_client()
        client.get_stats()
        client.close()
        _neo4j_available = True
    except Exception:
        _neo4j_available = False
    return _neo4j_available


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip all graph_mcp-marked tests when Neo4j is unavailable."""
    if _check_neo4j_available():
        return
    skip_marker = pytest.mark.skip(reason="Neo4j not available")
    for item in items:
        if item.get_closest_marker("graph_mcp"):
            item.add_marker(skip_marker)


# ── Fixture graph data ────────────────────────────────────────────────────


DD_VERSIONS = [
    {
        "id": "3.42.0",
        "major": 3,
        "minor": 42,
        "patch": 0,
        "is_current": False,
    },
    {
        "id": "4.0.0",
        "major": 4,
        "minor": 0,
        "patch": 0,
        "is_current": False,
    },
    {
        "id": "4.1.0",
        "major": 4,
        "minor": 1,
        "patch": 0,
        "is_current": True,
    },
]

UNITS = [
    {"id": "eV", "name": "electronvolt"},
    {"id": "m", "name": "metre"},
    {"id": "m^-3", "name": "per cubic metre"},
    {"id": "T", "name": "tesla"},
    {"id": "Pa", "name": "pascal"},
    {"id": "-", "name": "dimensionless"},
]

IDS_NODES = [
    {
        "id": "equilibrium",
        "name": "equilibrium",
        "documentation": "Plasma equilibrium quantities",
        "lifecycle_status": "active",
        "ids_type": "constant_or_dynamic",
        "path_count": 5,
        "physics_domain": "magnetics",
    },
    {
        "id": "core_profiles",
        "name": "core_profiles",
        "documentation": "Core plasma profiles (1D radial)",
        "lifecycle_status": "active",
        "ids_type": "constant_or_dynamic",
        "path_count": 4,
        "physics_domain": "transport",
    },
]

IMAS_PATHS = [
    # equilibrium paths
    {
        "id": "equilibrium/time_slice/profiles_1d/psi",
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "name": "psi",
        "ids_name": "equilibrium",
        "documentation": "Poloidal magnetic flux profile",
        "data_type": "FLT_1D",
        "units": "T.m^2",
        "node_type": "leaf",
        "physics_domain": "magnetics",
        "introduced_in": "3.42.0",
    },
    {
        "id": "equilibrium/time_slice/profiles_1d/pressure",
        "path": "equilibrium/time_slice/profiles_1d/pressure",
        "name": "pressure",
        "ids_name": "equilibrium",
        "documentation": "Plasma pressure profile",
        "data_type": "FLT_1D",
        "units": "Pa",
        "node_type": "leaf",
        "physics_domain": "magnetics",
        "introduced_in": "3.42.0",
    },
    {
        "id": "equilibrium/time_slice/boundary/psi",
        "path": "equilibrium/time_slice/boundary/psi",
        "name": "psi",
        "ids_name": "equilibrium",
        "documentation": "Poloidal flux at the boundary",
        "data_type": "FLT_0D",
        "units": "T.m^2",
        "node_type": "leaf",
        "physics_domain": "magnetics",
        "introduced_in": "3.42.0",
    },
    {
        "id": "equilibrium/time_slice/boundary/psi_norm",
        "path": "equilibrium/time_slice/boundary/psi_norm",
        "name": "psi_norm",
        "ids_name": "equilibrium",
        "documentation": "Normalized poloidal flux at the boundary",
        "data_type": "FLT_0D",
        "units": "-",
        "node_type": "leaf",
        "physics_domain": "magnetics",
        "introduced_in": "3.42.0",
    },
    {
        "id": "equilibrium/time_slice/boundary/type",
        "path": "equilibrium/time_slice/boundary/type",
        "name": "type",
        "ids_name": "equilibrium",
        "documentation": "Type of boundary (integer identifier)",
        "data_type": "INT_0D",
        "units": "-",
        "node_type": "leaf",
        "physics_domain": "magnetics",
        "introduced_in": "3.42.0",
    },
    # core_profiles paths
    {
        "id": "core_profiles/profiles_1d/electrons/temperature",
        "path": "core_profiles/profiles_1d/electrons/temperature",
        "name": "temperature",
        "ids_name": "core_profiles",
        "documentation": "Electron temperature profile",
        "data_type": "FLT_1D",
        "units": "eV",
        "node_type": "leaf",
        "physics_domain": "transport",
        "introduced_in": "3.42.0",
    },
    {
        "id": "core_profiles/profiles_1d/electrons/density",
        "path": "core_profiles/profiles_1d/electrons/density",
        "name": "density",
        "ids_name": "core_profiles",
        "documentation": "Electron density profile",
        "data_type": "FLT_1D",
        "units": "m^-3",
        "node_type": "leaf",
        "physics_domain": "transport",
        "introduced_in": "3.42.0",
    },
    {
        "id": "core_profiles/profiles_1d/electrons/pressure",
        "path": "core_profiles/profiles_1d/electrons/pressure",
        "name": "pressure",
        "ids_name": "core_profiles",
        "documentation": "Electron pressure profile (derived from temperature * density)",
        "data_type": "FLT_1D",
        "units": "Pa",
        "node_type": "leaf",
        "physics_domain": "transport",
        "introduced_in": "4.0.0",
    },
    {
        "id": "core_profiles/profiles_1d/ion/temperature",
        "path": "core_profiles/profiles_1d/ion/temperature",
        "name": "temperature",
        "ids_name": "core_profiles",
        "documentation": "Ion temperature profile",
        "data_type": "FLT_1D",
        "units": "eV",
        "node_type": "leaf",
        "physics_domain": "transport",
        "introduced_in": "3.42.0",
    },
]

CLUSTERS = [
    {
        "id": "cluster_temperature",
        "label": "Temperature Profiles",
        "description": "Temperature measurements for plasma species",
        "scope": "global",
        "path_count": 3,
    },
    {
        "id": "cluster_equilibrium_boundary",
        "label": "Equilibrium Boundary",
        "description": "Plasma boundary and separatrix quantities",
        "scope": "ids",
        "path_count": 3,
    },
]

PATH_CHANGES = [
    {
        "id": "change_pressure_added",
        "path_id": "core_profiles/profiles_1d/electrons/pressure",
        "from_version": "3.42.0",
        "to_version": "4.0.0",
        "change_type": "added",
        "description": "New electron pressure path added in 4.0.0",
    },
]

IDENTIFIER_SCHEMAS = [
    {
        "id": "equilibrium/time_slice/boundary/type",
        "name": "boundary_type",
        "description": "Type of boundary shape",
        "options": '[{"value": 0, "label": "last_closed_flux_surface"}, {"value": 1, "label": "limiter"}]',
    },
]


def _load_fixture_graph(client) -> None:
    """Load the fixture DD graph data into Neo4j.

    Only runs on a clean or small graph (< 100 nodes) to avoid
    accidentally wiping production data. In CI, the Neo4j service
    starts empty, so this always runs. Locally, tests that need
    fixture data will skip if the graph is too large.
    """
    stats = client.get_stats()
    if stats["nodes"] > 100:
        pytest.skip(
            f"Neo4j has {stats['nodes']} nodes — refusing to clear production data. "
            "Graph-native MCP tests require a clean Neo4j instance (e.g., CI)."
        )

    # Clear existing data
    client.query("MATCH (n) DETACH DELETE n")

    # Create DDVersion nodes and chain them
    for v in DD_VERSIONS:
        client.query(
            "CREATE (d:DDVersion {id: $id, major: $major, minor: $minor, "
            "patch: $patch, is_current: $is_current})",
            **v,
        )

    # Chain versions with PREDECESSOR
    client.query(
        "MATCH (a:DDVersion {id: '4.0.0'}), (b:DDVersion {id: '3.42.0'}) "
        "CREATE (a)-[:PREDECESSOR]->(b)"
    )
    client.query(
        "MATCH (a:DDVersion {id: '4.1.0'}), (b:DDVersion {id: '4.0.0'}) "
        "CREATE (a)-[:PREDECESSOR]->(b)"
    )

    # Create Unit nodes
    for u in UNITS:
        client.query("CREATE (u:Unit {id: $id, name: $name})", **u)

    # Create IDS nodes
    for ids in IDS_NODES:
        client.query(
            "CREATE (i:IDS {id: $id, name: $name, documentation: $documentation, "
            "lifecycle_status: $lifecycle_status, ids_type: $ids_type, "
            "path_count: $path_count, physics_domain: $physics_domain})",
            **ids,
        )

    # Create IMASPath nodes with relationships
    for p in IMAS_PATHS:
        client.query(
            "CREATE (p:IMASPath {id: $id, path: $path, ids: $ids_name, "
            "name: $name, "
            "documentation: $documentation, data_type: $data_type, units: $units, "
            "node_type: $node_type, physics_domain: $physics_domain})",
            **p,
        )
        # Link to IDS
        client.query(
            "MATCH (p:IMASPath {id: $path_id}), (i:IDS {id: $ids_name}) "
            "CREATE (p)-[:IN_IDS]->(i)",
            path_id=p["id"],
            ids_name=p["ids_name"],
        )
        # Link to introduced version
        if p.get("introduced_in"):
            client.query(
                "MATCH (p:IMASPath {id: $path_id}), (v:DDVersion {id: $version}) "
                "CREATE (p)-[:INTRODUCED_IN]->(v)",
                path_id=p["id"],
                version=p["introduced_in"],
            )
        # Link to Unit
        unit_id = p.get("units")
        if unit_id and unit_id != "-":
            client.query(
                "MATCH (p:IMASPath {id: $path_id}), (u:Unit {id: $unit_id}) "
                "CREATE (p)-[:HAS_UNIT]->(u)",
                path_id=p["id"],
                unit_id=unit_id,
            )

    # Create IMASSemanticCluster nodes
    for c in CLUSTERS:
        client.query(
            "CREATE (c:IMASSemanticCluster {id: $id, label: $label, "
            "description: $description, scope: $scope, path_count: $path_count})",
            **c,
        )

    # Link paths to clusters
    cluster_memberships = [
        ("core_profiles/profiles_1d/electrons/temperature", "cluster_temperature"),
        ("core_profiles/profiles_1d/ion/temperature", "cluster_temperature"),
        ("core_profiles/profiles_1d/electrons/pressure", "cluster_temperature"),
        ("equilibrium/time_slice/boundary/psi", "cluster_equilibrium_boundary"),
        ("equilibrium/time_slice/boundary/psi_norm", "cluster_equilibrium_boundary"),
        ("equilibrium/time_slice/boundary/type", "cluster_equilibrium_boundary"),
    ]
    for path_id, cluster_id in cluster_memberships:
        client.query(
            "MATCH (p:IMASPath {id: $path_id}), (c:IMASSemanticCluster {id: $cluster_id}) "
            "CREATE (p)-[:IN_CLUSTER]->(c)",
            path_id=path_id,
            cluster_id=cluster_id,
        )

    # Create IMASPathChange nodes
    for ch in PATH_CHANGES:
        client.query(
            "CREATE (c:IMASPathChange {id: $id, path_id: $path_id, "
            "from_version: $from_version, to_version: $to_version, "
            "change_type: $change_type, description: $description})",
            **ch,
        )
        # Link to path and versions
        client.query(
            "MATCH (c:IMASPathChange {id: $id}), (p:IMASPath {id: $path_id}) "
            "CREATE (c)-[:FOR_IMAS_PATH]->(p)",
            id=ch["id"],
            path_id=ch["path_id"],
        )
        client.query(
            "MATCH (c:IMASPathChange {id: $id}), (v:DDVersion {id: $to}) "
            "CREATE (c)-[:IN_VERSION]->(v)",
            id=ch["id"],
            to=ch["to_version"],
        )

    # Create IdentifierSchema nodes
    for ident in IDENTIFIER_SCHEMAS:
        client.query(
            "CREATE (s:IdentifierSchema {id: $id, name: $name, "
            "description: $description, options: $options})",
            **ident,
        )
        # Link to IMASPath
        client.query(
            "MATCH (s:IdentifierSchema {id: $id}), (p:IMASPath {id: $path_id}) "
            "CREATE (p)-[:HAS_IDENTIFIER_SCHEMA]->(s)",
            id=ident["id"],
            path_id=ident["id"],
        )


# ── Session-scoped fixtures ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def graph_client():
    """Session-scoped GraphClient for graph-native MCP tests.

    Loads fixture DD data once per session. Skips if Neo4j is unavailable.
    """
    try:
        client = _make_client()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
        return  # unreachable but satisfies type checker

    _load_fixture_graph(client)
    yield client
    client.close()


@pytest.fixture(scope="session")
def fixture_paths():
    """All fixture IMASPath data."""
    return IMAS_PATHS


@pytest.fixture(scope="session")
def fixture_ids():
    """All fixture IDS data."""
    return IDS_NODES


@pytest.fixture(scope="session")
def fixture_versions():
    """All fixture DDVersion data."""
    return DD_VERSIONS


@pytest.fixture(scope="session")
def fixture_clusters():
    """All fixture cluster data."""
    return CLUSTERS
