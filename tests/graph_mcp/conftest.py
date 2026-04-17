"""Fixtures for graph-native MCP tests.

Provides a small hand-crafted DD graph loaded into a test Neo4j instance.
Tests auto-skip when Neo4j is not reachable. The fixture graph contains
a minimal but complete DD structure: DDVersion nodes, IDS, IMASNode hierarchy,
units, clusters, identifier schemas, and path changes.

Connection uses the project's profile-aware resolution (handles SLURM
compute nodes, tunnels, env overrides) via ``get_graph_uri()`` etc.
"""

import pytest

# Mark all tests in this directory as requiring graph
pytestmark = pytest.mark.graph_mcp

# ── Neo4j availability check ──────────────────────────────────────────────

_neo4j_available: bool | None = None


def _make_client():
    """Create a GraphClient using profile-aware connection resolution.

    Uses ``get_graph_uri()`` / ``get_graph_username()`` / ``get_graph_password()``
    from ``imas_codex.settings`` which handle SLURM node discovery, SSH tunnels,
    and env var overrides automatically.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.settings import (
        get_graph_password,
        get_graph_uri,
        get_graph_username,
    )

    return GraphClient(
        uri=get_graph_uri(),
        username=get_graph_username(),
        password=get_graph_password(),
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


@pytest.fixture(autouse=True)
def _skip_fixture_only_on_production(request):
    """Auto-skip tests marked ``fixture_only`` when running against production data."""
    if request.node.get_closest_marker("fixture_only") and is_production_graph():
        pytest.skip("Test requires fixture data (CI mode only)")


# ── Fixture graph data ────────────────────────────────────────────────────


DD_VERSIONS = [
    {
        "id": "3.42.0",
        "major": 3,
        "minor": 42,
        "patch": 0,
        "is_current": False,
        "is_major_boundary": False,
    },
    {
        "id": "4.0.0",
        "major": 4,
        "minor": 0,
        "patch": 0,
        "is_current": False,
        "is_major_boundary": True,
    },
    {
        "id": "4.1.0",
        "major": 4,
        "minor": 1,
        "patch": 0,
        "is_current": True,
        "is_major_boundary": False,
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
        "physics_domain": "equilibrium",
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
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "cocos_transformation_type": "psi_like",
        "lifecycle_status": "active",
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
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "cocos_transformation_type": "psi_like",
        "lifecycle_status": "active",
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
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "cocos_transformation_type": "psi_like",
        "lifecycle_status": "active",
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
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "lifecycle_status": "active",
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
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "lifecycle_status": "active",
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
        "lifecycle_status": "active",
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
        "lifecycle_status": "active",
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
        "lifecycle_status": "alpha",
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
        "lifecycle_status": "active",
    },
    {
        "id": "equilibrium/time_slice/profiles_1d",
        "path": "equilibrium/time_slice/profiles_1d",
        "name": "profiles_1d",
        "ids_name": "equilibrium",
        "documentation": "Equilibrium 1D profiles structure",
        "data_type": "STRUCTURE",
        "units": "-",
        "node_type": "static",
        "physics_domain": "equilibrium",
        "introduced_in": "3.42.0",
        "lifecycle_status": "active",
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
        "version": "4.0.0",
        "change_type": "added",
        "semantic_change_type": "units",
        "description": "New electron pressure path added in 4.0.0",
        "summary": "New electron pressure path added in 4.0.0",
        "breaking_level": None,
    },
    {
        "id": "change_psi_cocos",
        "path_id": "equilibrium/time_slice/profiles_1d/psi",
        "from_version": "3.42.0",
        "to_version": "4.0.0",
        "version": "4.0.0",
        "change_type": "cocos_transformation_type",
        "semantic_change_type": "sign_convention",
        "description": "COCOS convention changed from 11 to 17",
        "summary": "Sign convention change",
        "breaking_level": "required",
    },
]

IDENTIFIER_SCHEMAS = [
    {
        "id": "equilibrium/time_slice/boundary/type",
        "name": "boundary_type",
        "documentation": "Type of boundary shape",
        "description": "Classifies the plasma boundary topology. The distinction between last closed flux surface and limiter boundary determines how edge transport and scrape-off layer physics are modelled in equilibrium reconstruction.",
        "keywords": ["boundary", "topology", "plasma edge", "separatrix"],
        "options": '[{"value": 0, "label": "last_closed_flux_surface"}, {"value": 1, "label": "limiter"}]',
    },
]


_is_production_graph: bool = False


def is_production_graph() -> bool:
    """Return True if the test session is running against a production graph."""
    return _is_production_graph


def _load_fixture_graph(client) -> None:
    """Load the fixture DD graph data into Neo4j.

    Two modes:
    - **CI mode** (< 100 nodes): wipes graph and loads fixtures. Tests use
      exact fixture assertions.
    - **Production mode** (>= 100 nodes): skips fixture loading, sets the
      ``_is_production_graph`` flag. Tests that need exact fixture data
      should use ``@pytest.mark.fixture_only`` and will be auto-skipped.
      Tests for new capabilities should use relaxed assertions (``>=``).
    """
    global _is_production_graph
    stats = client.get_stats()
    if stats["nodes"] >= 100:
        _is_production_graph = True
        return

    _is_production_graph = False

    # Clear existing data (CI mode only)
    client.query("MATCH (n) DETACH DELETE n")

    # Create DDVersion nodes and chain them
    for v in DD_VERSIONS:
        client.query(
            "CREATE (d:DDVersion {id: $id, major: $major, minor: $minor, "
            "patch: $patch, is_current: $is_current})",
            **v,
        )

    # Chain versions with HAS_PREDECESSOR
    client.query(
        "MATCH (a:DDVersion {id: '4.0.0'}), (b:DDVersion {id: '3.42.0'}) "
        "CREATE (a)-[:HAS_PREDECESSOR]->(b)"
    )
    client.query(
        "MATCH (a:DDVersion {id: '4.1.0'}), (b:DDVersion {id: '4.0.0'}) "
        "CREATE (a)-[:HAS_PREDECESSOR]->(b)"
    )
    # Chain versions with HAS_SUCCESSOR (symmetric with HAS_PREDECESSOR)
    client.query(
        "MATCH (a:DDVersion {id: '3.42.0'}), (b:DDVersion {id: '4.0.0'}) "
        "CREATE (a)-[:HAS_SUCCESSOR]->(b)"
    )
    client.query(
        "MATCH (a:DDVersion {id: '4.0.0'}), (b:DDVersion {id: '4.1.0'}) "
        "CREATE (a)-[:HAS_SUCCESSOR]->(b)"
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

    # Create IMASNode nodes with relationships
    for p in IMAS_PATHS:
        client.query(
            "CREATE (p:IMASNode {id: $id, path: $path, ids: $ids_name, "
            "name: $name, "
            "documentation: $documentation, data_type: $data_type, units: $units, "
            "node_type: $node_type, physics_domain: $physics_domain, "
            "node_category: 'quantity', "
            "lifecycle_status: $lifecycle_status, "
            "cocos_transformation_type: $cocos_transformation_type})",
            id=p["id"],
            path=p["path"],
            ids_name=p["ids_name"],
            name=p["name"],
            documentation=p["documentation"],
            data_type=p["data_type"],
            units=p["units"],
            node_type=p["node_type"],
            physics_domain=p["physics_domain"],
            lifecycle_status=p.get("lifecycle_status"),
            cocos_transformation_type=p.get("cocos_transformation_type"),
        )
        # Link to IDS
        client.query(
            "MATCH (p:IMASNode {id: $path_id}), (i:IDS {id: $ids_name}) "
            "CREATE (p)-[:IN_IDS]->(i)",
            path_id=p["id"],
            ids_name=p["ids_name"],
        )
        # Link to introduced version
        if p.get("introduced_in"):
            client.query(
                "MATCH (p:IMASNode {id: $path_id}), (v:DDVersion {id: $version}) "
                "CREATE (p)-[:INTRODUCED_IN]->(v)",
                path_id=p["id"],
                version=p["introduced_in"],
            )
        # Link to Unit
        unit_id = p.get("units")
        if unit_id and unit_id != "-":
            client.query(
                "MATCH (p:IMASNode {id: $path_id}), (u:Unit {id: $unit_id}) "
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
            "MATCH (p:IMASNode {id: $path_id}), (c:IMASSemanticCluster {id: $cluster_id}) "
            "CREATE (p)-[:IN_CLUSTER]->(c)",
            path_id=path_id,
            cluster_id=cluster_id,
        )

    # Create IMASNodeChange nodes
    for ch in PATH_CHANGES:
        client.query(
            "CREATE (c:IMASNodeChange {id: $id, path_id: $path_id, "
            "from_version: $from_version, to_version: $to_version, "
            "version: $version, "
            "change_type: $change_type, semantic_change_type: $semantic_change_type, "
            "description: $description, summary: $summary, "
            "breaking_level: $breaking_level})",
            **ch,
        )
        # Link to path and versions
        client.query(
            "MATCH (c:IMASNodeChange {id: $id}), (p:IMASNode {id: $path_id}) "
            "CREATE (c)-[:FOR_IMAS_PATH]->(p)",
            id=ch["id"],
            path_id=ch["path_id"],
        )
        client.query(
            "MATCH (c:IMASNodeChange {id: $id}), (v:DDVersion {id: $to}) "
            "CREATE (c)-[:IN_VERSION]->(v)",
            id=ch["id"],
            to=ch["to_version"],
        )

    # Create IdentifierSchema nodes
    for ident in IDENTIFIER_SCHEMAS:
        client.query(
            "CREATE (s:IdentifierSchema {id: $id, name: $name, "
            "documentation: $documentation, options: $options, "
            "description: $description, "
            "keywords: $keywords})",
            **ident,
        )
        # Link to IMASNode
        client.query(
            "MATCH (s:IdentifierSchema {id: $id}), (p:IMASNode {id: $path_id}) "
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
    """All fixture IMASNode data."""
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
