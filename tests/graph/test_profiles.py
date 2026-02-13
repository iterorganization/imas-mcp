"""Tests for graph profile resolution."""

import pytest

from imas_codex.graph.profiles import (
    BOLT_BASE_PORT,
    HTTP_BASE_PORT,
    GraphProfile,
    _convention_bolt_port,
    _convention_data_dir,
    _convention_http_port,
    _get_all_hosts,
    _get_all_offsets,
    check_graph_conflict,
    get_active_graph_name,
    is_port_bound_by_tunnel,
    list_profiles,
    resolve_graph,
)
from imas_codex.settings import _load_pyproject_settings

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

    def test_codex_uses_plain_neo4j(self):
        """Default graph (codex) uses neo4j/ without suffix."""
        data_dir = _convention_data_dir("codex")
        assert data_dir.name == "neo4j"

    def test_tcv_uses_suffixed_dir(self):
        """Non-default names use neo4j-{name}/ suffix."""
        data_dir = _convention_data_dir("tcv")
        assert data_dir.name == "neo4j-tcv"

    def test_iter_uses_suffixed_dir(self):
        """iter (a location name, not the default) gets neo4j-iter/."""
        data_dir = _convention_data_dir("iter")
        assert data_dir.name == "neo4j-iter"

    def test_jt60sa_uses_suffixed_dir(self):
        data_dir = _convention_data_dir("jt60sa")
        assert data_dir.name == "neo4j-jt60sa"


# ── Profile resolution ──────────────────────────────────────────────────────


class TestResolveGraph:
    """Tests for resolve_graph() profile resolution."""

    def test_resolve_known_facility(self, monkeypatch):
        """Known facility resolves by convention."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        # Treat tcv as local so auto-tunnel doesn't fire
        monkeypatch.setattr(
            "imas_codex.remote.executor.is_local_host",
            lambda h: True,
        )

        profile = resolve_graph("tcv")
        assert isinstance(profile, GraphProfile)
        assert profile.name == "tcv"
        assert profile.bolt_port == 7688
        assert profile.http_port == 7475
        assert profile.uri == "bolt://localhost:7688"
        assert profile.data_dir.name == "neo4j-tcv"

    def test_resolve_iter_as_location(self, monkeypatch):
        """iter is a known location — resolves directly to iter's port slot."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        # Treat iter as local so auto-tunnel doesn't fire
        monkeypatch.setattr(
            "imas_codex.remote.executor.is_local_host",
            lambda h: True,
        )

        profile = resolve_graph("iter")
        assert profile.name == "iter"
        assert profile.bolt_port == 7687
        assert profile.http_port == 7474
        assert profile.uri == "bolt://localhost:7687"

    def test_resolve_codex_uses_location(self, monkeypatch):
        """codex (not a location) resolves via get_graph_location → iter."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("IMAS_CODEX_GRAPH_LOCATION", raising=False)
        monkeypatch.setattr(
            "imas_codex.remote.executor.is_local_host",
            lambda h: True,
        )

        profile = resolve_graph("codex")
        assert profile.name == "codex"
        # codex is not a location, so ports come from default location (iter)
        assert profile.bolt_port == 7687
        assert profile.http_port == 7474
        assert profile.data_dir.name == "neo4j"  # codex gets the plain dir

    def test_env_uri_overrides_convention(self, monkeypatch):
        """NEO4J_URI env var overrides convention URI."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_URI", "bolt://remote:9999")
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_graph("tcv")
        assert profile.uri == "bolt://remote:9999"
        # Convention ports still available for service management
        assert profile.bolt_port == 7688

    def test_env_password_overrides_convention(self, monkeypatch):
        """NEO4J_PASSWORD env var overrides convention password."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_PASSWORD", "super-secret")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)

        profile = resolve_graph("iter")
        assert profile.password == "super-secret"

    def test_shared_credentials(self, monkeypatch):
        """All profiles share credentials from [graph] section."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        iter_p = resolve_graph("iter")
        tcv_p = resolve_graph("tcv")
        assert iter_p.username == tcv_p.username
        assert iter_p.password == tcv_p.password


# ── Host field ──────────────────────────────────────────────────────────────


class TestHostField:
    """Tests for the host field on GraphProfile."""

    def test_known_facility_has_host(self, monkeypatch):
        """Known facilities in graph.hosts config get host set."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_graph("iter")
        assert profile.host == "iter"

    def test_tcv_has_host(self, monkeypatch):
        """tcv facility has host set."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_graph("tcv")
        assert profile.host == "tcv"

    def test_location_in_list_has_implicit_host(self, monkeypatch):
        """Every location in the list gets an implicit host = name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.setattr(
            "imas_codex.remote.executor.is_local_host",
            lambda h: True,
        )

        profile = resolve_graph("kstar")
        assert profile.host == "kstar"

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
        """Default active graph is codex (from pyproject.toml)."""
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
        """resolve_graph(None) resolves via get_active_graph_name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.setenv("IMAS_CODEX_GRAPH", "jet")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_graph()
        assert profile.name == "jet"
        assert profile.bolt_port == 7690


# ── List profiles ───────────────────────────────────────────────────────────


class TestListProfiles:
    """Tests for list_profiles()."""

    def test_returns_all_known_locations(self, monkeypatch):
        """list_profiles returns at least all known locations."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        names = {p.name for p in profiles}
        assert names >= set(_get_all_offsets().keys())

    def test_profiles_sorted_by_name(self, monkeypatch):
        """list_profiles returns sorted by name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert names == sorted(names)
