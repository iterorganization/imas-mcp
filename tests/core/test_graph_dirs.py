"""Tests for graph directory store (dirs.py).

All tests use tmp_path fixtures to avoid touching the real filesystem.
"""

import pytest

from imas_codex.graph.dirs import (
    GraphDirInfo,
    create_graph_dir,
    delete_graph_dir,
    find_graph,
    get_active_graph,
    is_legacy_data_dir,
    list_local_graphs,
    switch_active_graph,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point GRAPH_STORE, ACTIVE_LINK and EXPORTS_DIR to tmp_path."""
    monkeypatch.setattr("imas_codex.graph.dirs.GRAPH_STORE", tmp_path / ".neo4j")
    monkeypatch.setattr("imas_codex.graph.dirs.ACTIVE_LINK", tmp_path / "neo4j")
    monkeypatch.setattr("imas_codex.graph.dirs.EXPORTS_DIR", tmp_path / "exports")


# ── Create ──────────────────────────────────────────────────────────────────


class TestCreateGraphDir:
    def test_creates_directory(self):
        info = create_graph_dir("codex")
        assert info.path.exists()
        assert (info.path / "data").is_dir()
        assert (info.path / "logs").is_dir()
        assert (info.path / "conf").is_dir()
        assert (info.path / "import").is_dir()

    def test_generates_neo4j_conf(self):
        info = create_graph_dir("codex")
        conf = info.path / "conf" / "neo4j.conf"
        assert conf.exists()
        content = conf.read_text()
        assert "server.bolt.listen_address=:7687" in content
        assert "server.http.listen_address=:7474" in content
        assert "server.default_listen_address=127.0.0.1" in content

    def test_custom_ports_in_neo4j_conf(self):
        info = create_graph_dir("tcv", bolt_port=7688, http_port=7475)
        conf = info.path / "conf" / "neo4j.conf"
        content = conf.read_text()
        assert "server.bolt.listen_address=:7688" in content
        assert "server.http.listen_address=:7475" in content

    def test_name_is_directory_name(self):
        info = create_graph_dir("codex")
        assert info.name == "codex"
        assert info.path.name == "codex"

    def test_duplicate_raises(self):
        create_graph_dir("codex")
        with pytest.raises(FileExistsError):
            create_graph_dir("codex")

    def test_force_allows_existing(self):
        create_graph_dir("codex")
        info = create_graph_dir("codex", force=True)
        assert info.name == "codex"
        assert info.path.exists()

    def test_returns_graphdirinfo(self):
        info = create_graph_dir("dev")
        assert isinstance(info, GraphDirInfo)
        assert info.name == "dev"
        assert not info.active


# ── Exports ─────────────────────────────────────────────────────────────────


class TestEnsureExportsDir:
    def test_creates_exports_dir(self, tmp_path):
        from imas_codex.graph.dirs import ensure_exports_dir

        exports = ensure_exports_dir()
        assert exports == tmp_path / "exports"
        assert exports.is_dir()
        assert oct(exports.stat().st_mode & 0o777) == "0o700"

    def test_idempotent(self, tmp_path):
        from imas_codex.graph.dirs import ensure_exports_dir

        ensure_exports_dir()
        ensure_exports_dir()
        assert (tmp_path / "exports").is_dir()


# ── List ────────────────────────────────────────────────────────────────────


class TestListLocalGraphs:
    def test_empty_store(self):
        assert list_local_graphs() == []

    def test_lists_created_graphs(self):
        create_graph_dir("alpha")
        create_graph_dir("beta")
        graphs = list_local_graphs()
        names = [g.name for g in graphs]
        assert "alpha" in names
        assert "beta" in names

    def test_sorted_by_name(self):
        create_graph_dir("beta")
        create_graph_dir("alpha")
        graphs = list_local_graphs()
        assert graphs[0].name == "alpha"
        assert graphs[1].name == "beta"

    def test_warns_on_missing_data_dir(self, tmp_path):
        """Directory without data/ subdirectory produces a warning."""
        store = tmp_path / ".neo4j"
        store.mkdir()
        (store / "orphan").mkdir()
        graphs = list_local_graphs()
        assert len(graphs) == 1
        assert graphs[0].name == "orphan"
        assert any("Missing data/" in w for w in graphs[0].warnings)

    def test_marks_active(self):
        create_graph_dir("codex")
        switch_active_graph("codex")
        graphs = list_local_graphs()
        active = [g for g in graphs if g.active]
        assert len(active) == 1
        assert active[0].name == "codex"


# ── Active graph ────────────────────────────────────────────────────────────


class TestGetActiveGraph:
    def test_no_link_returns_none(self):
        assert get_active_graph() is None

    def test_real_dir_returns_none(self, tmp_path):
        """Legacy real directory (not symlink) returns None."""
        link = tmp_path / "neo4j"
        link.mkdir()
        assert get_active_graph() is None

    def test_returns_info_for_symlink(self):
        create_graph_dir("codex")
        switch_active_graph("codex")
        active = get_active_graph()
        assert active is not None
        assert active.name == "codex"
        assert active.active is True


# ── Switch ──────────────────────────────────────────────────────────────────


class TestSwitchActiveGraph:
    def test_switch_creates_symlink(self, tmp_path):
        create_graph_dir("codex")
        result = switch_active_graph("codex")
        link = tmp_path / "neo4j"
        assert link.is_symlink()
        assert result.name == "codex"
        assert result.active is True

    def test_switch_replaces_symlink(self, tmp_path):
        create_graph_dir("g1")
        create_graph_dir("g2")
        switch_active_graph("g1")
        switch_active_graph("g2")
        link = tmp_path / "neo4j"
        assert link.resolve().name == "g2"

    def test_switch_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            switch_active_graph("nonexistent")

    def test_switch_real_dir_raises(self, tmp_path):
        """Can't switch if neo4j/ is a real directory."""
        create_graph_dir("codex")
        (tmp_path / "neo4j").mkdir()
        with pytest.raises(FileExistsError):
            switch_active_graph("codex")


# ── Find ────────────────────────────────────────────────────────────────────


class TestFindGraph:
    def test_find_by_name(self):
        create_graph_dir("codex")
        info = find_graph("codex")
        assert info.name == "codex"

    def test_not_found_raises(self):
        with pytest.raises(LookupError, match="No graph found"):
            find_graph("nope")

    def test_marks_active_correctly(self):
        create_graph_dir("codex")
        create_graph_dir("dev")
        switch_active_graph("codex")

        info_active = find_graph("codex")
        assert info_active.active is True

        info_inactive = find_graph("dev")
        assert info_inactive.active is False


# ── Legacy detection ────────────────────────────────────────────────────────


class TestIsLegacyDataDir:
    def test_no_dir(self):
        assert is_legacy_data_dir() is False

    def test_real_dir(self, tmp_path):
        (tmp_path / "neo4j").mkdir()
        assert is_legacy_data_dir() is True

    def test_symlink(self):
        create_graph_dir("codex")
        switch_active_graph("codex")
        assert is_legacy_data_dir() is False


# ── Delete ──────────────────────────────────────────────────────────────────


class TestDeleteGraphDir:
    def test_delete_existing(self, tmp_path):
        create_graph_dir("to-delete")
        delete_graph_dir("to-delete")
        assert not (tmp_path / ".neo4j" / "to-delete").exists()

    def test_delete_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            delete_graph_dir("nope")

    def test_delete_active_raises(self):
        create_graph_dir("codex")
        switch_active_graph("codex")
        with pytest.raises(ValueError, match="Cannot delete active"):
            delete_graph_dir("codex")
