"""Tests for Neo4j profile resolution.

These tests verify port conventions, data-dir conventions, and profile
resolution logic.  They do NOT require a live Neo4j connection.
All calls to ``is_local_host`` are mocked so no SSH probes occur.
"""

import pytest

from imas_codex.graph.profiles import (
    BOLT_BASE_PORT,
    HTTP_BASE_PORT,
    Neo4jProfile,
    _convention_bolt_port,
    _convention_data_dir,
    _convention_http_port,
    _get_all_hosts,
    _get_all_offsets,
    _resolved_uri_cache,
    check_graph_conflict,
    get_active_graph_name,
    is_port_bound_by_tunnel,
    list_profiles,
    resolve_neo4j,
)
from imas_codex.settings import _load_pyproject_settings


@pytest.fixture(autouse=True)
def _clear_uri_cache():
    """Clear the URI resolution cache before each test to avoid cross-test pollution."""
    _resolved_uri_cache.clear()
    yield
    _resolved_uri_cache.clear()


@pytest.fixture(autouse=True)
def _mock_is_local_host(monkeypatch):
    """Prevent any SSH probes — treat all hosts as local."""
    monkeypatch.setattr(
        "imas_codex.remote.executor.is_local_host",
        lambda h: True,
    )


# ── Port convention ─────────────────────────────────────────────────────────


class TestPortConvention:
    """Tests for convention-based port mapping."""

    def test_iter_default_ports(self):
        """iter (offset 0) uses Neo4j default ports."""
        assert _convention_bolt_port("iter") == 7687
        assert _convention_http_port("iter") == 7474

    def test_tcv_ports(self):
        """tcv (offset 1) offsets both bolt and http."""
        assert _convention_bolt_port("tcv") == 7688
        assert _convention_http_port("tcv") == 7475

    def test_jt60sa_ports(self):
        """jt60sa (offset 2)."""
        assert _convention_bolt_port("jt60sa") == 7689
        assert _convention_http_port("jt60sa") == 7476

    def test_all_facilities_have_unique_ports(self):
        """Every known facility has a unique bolt and http port."""
        offsets = _get_all_offsets()
        bolt_ports = {_convention_bolt_port(f) for f in offsets}
        http_ports = {_convention_http_port(f) for f in offsets}
        assert len(bolt_ports) == len(offsets)
        assert len(http_ports) == len(offsets)

    def test_bolt_and_http_dont_clash(self):
        """Bolt and HTTP port ranges don't overlap."""
        offsets = _get_all_offsets()
        bolt_ports = {_convention_bolt_port(f) for f in offsets}
        http_ports = {_convention_http_port(f) for f in offsets}
        assert bolt_ports.isdisjoint(http_ports)

    def test_unknown_facility_gets_base_ports(self):
        """Unknown facility name falls back to base ports (offset 0)."""
        assert _convention_bolt_port("unknown") == BOLT_BASE_PORT
        assert _convention_http_port("unknown") == HTTP_BASE_PORT


# ── Data directory convention ───────────────────────────────────────────────


class TestDataDirConvention:
    """Tests for convention-based data directory mapping."""

    def test_codex_uses_neo4j_codex(self):
        """codex graph uses neo4j-codex/ like all other names."""
        data_dir = _convention_data_dir("codex")
        assert data_dir.name == "neo4j-codex"

    def test_tcv_uses_neo4j_tcv(self):
        data_dir = _convention_data_dir("tcv")
        assert data_dir.name == "neo4j-tcv"

    def test_dev_uses_neo4j_dev(self):
        data_dir = _convention_data_dir("dev")
        assert data_dir.name == "neo4j-dev"

    def test_iter_uses_neo4j_iter(self):
        data_dir = _convention_data_dir("iter")
        assert data_dir.name == "neo4j-iter"


# ── Profile resolution ──────────────────────────────────────────────────────


class TestResolveNeo4j:
    """Tests for resolve_neo4j() profile resolution."""

    def test_resolve_default(self, monkeypatch):
        """Default resolution uses 'codex' name and iter location."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        monkeypatch.delenv("IMAS_CODEX_GRAPH_LOCATION", raising=False)

        profile = resolve_neo4j()
        assert isinstance(profile, Neo4jProfile)
        assert profile.name == "codex"
        assert profile.location == "iter"
        assert profile.bolt_port == 7687
        assert profile.http_port == 7474
        assert profile.uri == "bolt://localhost:7687"
        assert profile.data_dir.name == "neo4j-codex"

    def test_resolve_explicit_name(self, monkeypatch):
        """Explicit name sets data dir but uses default location ports."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("IMAS_CODEX_GRAPH_LOCATION", raising=False)

        profile = resolve_neo4j("dev")
        assert profile.name == "dev"
        assert profile.data_dir.name == "neo4j-dev"
        # Ports come from default location (iter)
        assert profile.bolt_port == 7687

    def test_env_graph_name_override(self, monkeypatch):
        """IMAS_CODEX_GRAPH env var sets the graph name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("IMAS_CODEX_GRAPH", "staging")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_neo4j()
        assert profile.name == "staging"
        assert profile.data_dir.name == "neo4j-staging"

    def test_env_uri_overrides_convention(self, monkeypatch):
        """NEO4J_URI env var overrides convention URI."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_URI", "bolt://remote:9999")
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_neo4j("codex")
        assert profile.uri == "bolt://remote:9999"
        # Convention ports still available for service management
        assert profile.bolt_port == 7687

    def test_env_password_overrides_convention(self, monkeypatch):
        """NEO4J_PASSWORD env var overrides convention password."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_PASSWORD", "super-secret")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)

        profile = resolve_neo4j()
        assert profile.password == "super-secret"

    def test_shared_credentials(self, monkeypatch):
        """All profiles share credentials from [graph] section."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        p1 = resolve_neo4j("codex")
        p2 = resolve_neo4j("dev")
        assert p1.username == p2.username
        assert p1.password == p2.password


# ── Host field ──────────────────────────────────────────────────────────────


class TestHostField:
    """Tests for the host field on Neo4jProfile."""

    def test_default_location_has_host(self, monkeypatch):
        """Default location (iter) sets host."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("IMAS_CODEX_GRAPH_LOCATION", raising=False)

        profile = resolve_neo4j()
        assert profile.host == "iter"

    def test_host_defaults_match_known_locations(self):
        """All hosts have a corresponding location offset."""
        for name in _get_all_hosts():
            assert name in _get_all_offsets()


# ── Tunnel conflict detection ───────────────────────────────────────────────


class TestTunnelConflict:
    """Tests for tunnel conflict detection functions."""

    def test_is_port_bound_by_tunnel_returns_bool(self):
        """is_port_bound_by_tunnel returns a boolean."""
        result = is_port_bound_by_tunnel(99999)
        assert isinstance(result, bool)

    def test_check_graph_conflict_no_tunnel(self):
        """check_graph_conflict returns None when no tunnel on obscure port."""
        result = check_graph_conflict(99999)
        assert result is None


# ── Active graph name ───────────────────────────────────────────────────────


class TestGetActiveGraphName:
    """Tests for get_active_graph_name()."""

    def test_default_is_codex(self, monkeypatch):
        """Default active graph is codex (hardcoded default)."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        name = get_active_graph_name()
        assert name == "codex"

    def test_env_override(self, monkeypatch):
        """IMAS_CODEX_GRAPH env var overrides default."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("IMAS_CODEX_GRAPH", "tcv")
        name = get_active_graph_name()
        assert name == "tcv"

    def test_resolve_none_uses_active(self, monkeypatch):
        """resolve_neo4j(None) resolves via get_active_graph_name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("IMAS_CODEX_GRAPH", "jet")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_neo4j()
        assert profile.name == "jet"
        # Uses default location ports (iter)
        assert profile.bolt_port == 7687


# ── List profiles ───────────────────────────────────────────────────────────


class TestListProfiles:
    """Tests for list_profiles()."""

    def test_returns_profiles_per_location(self, monkeypatch):
        """list_profiles returns one profile per location."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        locations = {p.location for p in profiles}
        assert locations >= set(_get_all_offsets().keys())

    def test_profiles_sorted_by_location(self, monkeypatch):
        """list_profiles returns sorted by location."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        locations = [p.location for p in profiles]
        assert locations == sorted(locations)


# ── Backward compat ─────────────────────────────────────────────────────────


class TestBackwardCompat:
    """Tests for backward-compatible aliases."""

    def test_graph_profile_alias(self):
        """GraphProfile is an alias for Neo4jProfile."""
        from imas_codex.graph.profiles import GraphProfile

        assert GraphProfile is Neo4jProfile

    def test_resolve_graph_alias(self):
        """resolve_graph is an alias for resolve_neo4j."""
        from imas_codex.graph.profiles import resolve_graph

        assert resolve_graph is resolve_neo4j
