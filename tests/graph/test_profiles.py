"""Tests for graph profile resolution."""

import pytest

from imas_codex.graph.profiles import (
    BOLT_BASE_PORT,
    FACILITY_PORT_OFFSETS,
    HTTP_BASE_PORT,
    GraphProfile,
    _convention_bolt_port,
    _convention_data_dir,
    _convention_http_port,
    get_active_graph_name,
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
        bolt_ports = {_convention_bolt_port(f) for f in FACILITY_PORT_OFFSETS}
        http_ports = {_convention_http_port(f) for f in FACILITY_PORT_OFFSETS}
        assert len(bolt_ports) == len(FACILITY_PORT_OFFSETS)
        assert len(http_ports) == len(FACILITY_PORT_OFFSETS)

    def test_bolt_and_http_dont_clash(self):
        """Bolt and HTTP port ranges don't overlap."""
        bolt_ports = {_convention_bolt_port(f) for f in FACILITY_PORT_OFFSETS}
        http_ports = {_convention_http_port(f) for f in FACILITY_PORT_OFFSETS}
        assert bolt_ports.isdisjoint(http_ports)

    def test_unknown_facility_gets_base_ports(self):
        """Unknown facility name falls back to base ports (offset 0)."""
        assert _convention_bolt_port("unknown") == BOLT_BASE_PORT
        assert _convention_http_port("unknown") == HTTP_BASE_PORT


# ── Data directory convention ───────────────────────────────────────────────


class TestDataDirConvention:
    """Tests for convention-based data directory mapping."""

    def test_iter_uses_plain_neo4j(self):
        """Default profile (iter) uses neo4j/ without suffix."""
        data_dir = _convention_data_dir("iter")
        assert data_dir.name == "neo4j"

    def test_tcv_uses_suffixed_dir(self):
        """Non-default profiles use neo4j-{name}/ suffix."""
        data_dir = _convention_data_dir("tcv")
        assert data_dir.name == "neo4j-tcv"

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

        profile = resolve_graph("tcv")
        assert isinstance(profile, GraphProfile)
        assert profile.name == "tcv"
        assert profile.bolt_port == 7688
        assert profile.http_port == 7475
        assert profile.uri == "bolt://localhost:7688"
        assert profile.data_dir.name == "neo4j-tcv"

    def test_resolve_iter_default(self, monkeypatch):
        """iter profile uses Neo4j default ports."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profile = resolve_graph("iter")
        assert profile.bolt_port == 7687
        assert profile.http_port == 7474
        assert profile.uri == "bolt://localhost:7687"

    def test_resolve_unknown_raises(self, monkeypatch):
        """Unknown profile name raises ValueError."""
        _load_pyproject_settings.cache_clear()
        with pytest.raises(ValueError, match="Unknown graph profile"):
            resolve_graph("nonexistent-facility")

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


# ── Active graph name ───────────────────────────────────────────────────────


class TestGetActiveGraphName:
    """Tests for get_active_graph_name()."""

    def test_default_is_iter(self, monkeypatch):
        """Default active graph is iter (from pyproject.toml)."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        name = get_active_graph_name()
        assert name == "iter"

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

    def test_returns_all_known_facilities(self, monkeypatch):
        """list_profiles returns at least all known facilities."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        names = {p.name for p in profiles}
        assert names >= set(FACILITY_PORT_OFFSETS.keys())

    def test_profiles_sorted_by_name(self, monkeypatch):
        """list_profiles returns sorted by name."""
        _load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert names == sorted(names)
