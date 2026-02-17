"""Tests for graph directory store (dirs.py).

All tests use tmp_path fixtures to avoid touching the real filesystem.
"""

import json

import pytest

from imas_codex.graph.dirs import (
    GraphDirInfo,
    compute_graph_hash,
    create_graph_dir,
    find_graph,
    get_active_graph,
    list_local_graphs,
    migrate_legacy_dir,
    read_dir_meta,
    rename_graph_dir,
    switch_active_graph,
    validate_graph_dir,
    write_dir_meta,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point GRAPH_STORE and ACTIVE_LINK to tmp_path."""
    monkeypatch.setattr("imas_codex.graph.dirs.GRAPH_STORE", tmp_path / ".neo4j")
    monkeypatch.setattr("imas_codex.graph.dirs.ACTIVE_LINK", tmp_path / "neo4j")


# ── Hash computation ────────────────────────────────────────────────────────


class TestComputeGraphHash:
    def test_deterministic(self):
        h1 = compute_graph_hash("codex", ["iter", "tcv"])
        h2 = compute_graph_hash("codex", ["iter", "tcv"])
        assert h1 == h2

    def test_twelve_hex_chars(self):
        h = compute_graph_hash("codex", ["iter"])
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_insensitive(self):
        h1 = compute_graph_hash("codex", ["tcv", "iter"])
        h2 = compute_graph_hash("codex", ["iter", "tcv"])
        assert h1 == h2

    def test_different_name_different_hash(self):
        h1 = compute_graph_hash("codex", ["iter"])
        h2 = compute_graph_hash("dev", ["iter"])
        assert h1 != h2

    def test_different_facilities_different_hash(self):
        h1 = compute_graph_hash("codex", ["iter"])
        h2 = compute_graph_hash("codex", ["tcv"])
        assert h1 != h2

    def test_empty_facilities(self):
        h = compute_graph_hash("codex", [])
        assert len(h) == 12


# ── Metadata read/write ────────────────────────────────────────────────────


class TestMetadata:
    def test_write_and_read(self, tmp_path):
        d = tmp_path / "graph-dir"
        d.mkdir()
        write_dir_meta(d, "codex", ["iter", "tcv"], "abc123def456")
        info = read_dir_meta(d)
        assert info is not None
        assert info.name == "codex"
        assert info.facilities == ["iter", "tcv"]
        assert info.hash == "abc123def456"
        assert info.created_at != ""

    def test_read_missing(self, tmp_path):
        d = tmp_path / "no-meta"
        d.mkdir()
        assert read_dir_meta(d) is None

    def test_read_corrupt(self, tmp_path):
        d = tmp_path / "corrupt"
        d.mkdir()
        (d / ".meta.json").write_text("not json{")
        assert read_dir_meta(d) is None


# ── Create ──────────────────────────────────────────────────────────────────


class TestCreateGraphDir:
    def test_creates_directory(self, tmp_path):
        info = create_graph_dir("codex", ["iter"])
        assert info.path.exists()
        assert (info.path / "data").is_dir()
        assert (info.path / "logs").is_dir()
        assert (info.path / ".meta.json").exists()

    def test_hash_matches(self):
        info = create_graph_dir("codex", ["iter"])
        expected = compute_graph_hash("codex", ["iter"])
        assert info.hash == expected
        assert info.path.name == expected

    def test_duplicate_raises(self):
        create_graph_dir("codex", ["iter"])
        with pytest.raises(FileExistsError):
            create_graph_dir("codex", ["iter"])


# ── List ────────────────────────────────────────────────────────────────────


class TestListLocalGraphs:
    def test_empty_store(self):
        assert list_local_graphs() == []

    def test_lists_created_graphs(self):
        create_graph_dir("alpha", ["iter"])
        create_graph_dir("beta", ["tcv"])
        graphs = list_local_graphs()
        names = [g.name for g in graphs]
        assert "alpha" in names
        assert "beta" in names

    def test_warns_on_hash_drift(self, tmp_path):
        """If dir name doesn't match computed hash, produces a warning."""
        store = tmp_path / ".neo4j"
        store.mkdir()
        wrong_dir = store / "wronghash1234"
        wrong_dir.mkdir()
        meta = {"name": "codex", "facilities": ["iter"], "hash": "wronghash1234"}
        (wrong_dir / ".meta.json").write_text(json.dumps(meta))

        graphs = list_local_graphs()
        assert len(graphs) == 1
        assert any("Hash drift" in w for w in graphs[0].warnings)

    def test_warns_on_no_meta(self, tmp_path):
        """Directory without .meta.json gets <unknown> name."""
        store = tmp_path / ".neo4j"
        store.mkdir()
        (store / "orphandir12ab").mkdir()
        graphs = list_local_graphs()
        assert len(graphs) == 1
        assert graphs[0].name == "<unknown>"


# ── Active graph ────────────────────────────────────────────────────────────


class TestGetActiveGraph:
    def test_no_link_returns_none(self):
        assert get_active_graph() is None

    def test_real_dir_returns_none(self, tmp_path):
        """Legacy real directory (not symlink) returns None."""
        link = tmp_path / "neo4j"
        link.mkdir()
        assert get_active_graph() is None

    def test_returns_info_for_symlink(self, tmp_path):
        info = create_graph_dir("codex", ["iter"])
        link = tmp_path / "neo4j"
        link.symlink_to(info.path)
        active = get_active_graph()
        assert active is not None
        assert active.name == "codex"


# ── Switch ──────────────────────────────────────────────────────────────────


class TestSwitchActiveGraph:
    def test_switch_creates_symlink(self, tmp_path):
        info = create_graph_dir("codex", ["iter"])
        result = switch_active_graph(info.hash)
        link = tmp_path / "neo4j"
        assert link.is_symlink()
        assert link.resolve() == info.path.resolve()
        assert result.name == "codex"

    def test_switch_replaces_symlink(self, tmp_path):
        g1 = create_graph_dir("g1", ["iter"])
        g2 = create_graph_dir("g2", ["tcv"])
        switch_active_graph(g1.hash)
        switch_active_graph(g2.hash)
        link = tmp_path / "neo4j"
        assert link.resolve() == g2.path.resolve()

    def test_switch_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            switch_active_graph("nonexistent12")

    def test_switch_real_dir_raises(self, tmp_path):
        """Can't switch if neo4j/ is a real directory."""
        g = create_graph_dir("codex", ["iter"])
        (tmp_path / "neo4j").mkdir()
        with pytest.raises(FileExistsError):
            switch_active_graph(g.hash)


# ── Find ────────────────────────────────────────────────────────────────────


class TestFindGraph:
    def test_find_by_name(self):
        create_graph_dir("codex", ["iter"])
        info = find_graph("codex")
        assert info.name == "codex"

    def test_find_by_hash_prefix(self):
        g = create_graph_dir("codex", ["iter"])
        info = find_graph(g.hash[:4])
        assert info.hash == g.hash

    def test_not_found_raises(self):
        with pytest.raises(LookupError, match="No graph found"):
            find_graph("nope")


# ── Rename ──────────────────────────────────────────────────────────────────


class TestRenameGraphDir:
    def test_rename_updates_dir(self, tmp_path):
        g = create_graph_dir("codex", ["iter"])
        old_hash = g.hash
        new = rename_graph_dir(old_hash, "codex", ["iter", "tcv"])
        assert new.hash == compute_graph_hash("codex", ["iter", "tcv"])
        assert not (tmp_path / ".neo4j" / old_hash).exists()
        assert new.path.exists()

    def test_rename_updates_symlink(self, tmp_path):
        g = create_graph_dir("codex", ["iter"])
        switch_active_graph(g.hash)
        new = rename_graph_dir(g.hash, "codex", ["iter", "tcv"])
        link = tmp_path / "neo4j"
        assert link.resolve() == new.path.resolve()

    def test_same_hash_noop(self):
        g = create_graph_dir("codex", ["iter"])
        new = rename_graph_dir(g.hash, "codex", ["iter"])
        assert new.hash == g.hash


# ── Validate ────────────────────────────────────────────────────────────────


class TestValidateGraphDir:
    def test_valid_dir(self):
        g = create_graph_dir("codex", ["iter"])
        assert validate_graph_dir(g.path) == []

    def test_missing_meta(self, tmp_path):
        d = tmp_path / "nometa"
        d.mkdir()
        warnings = validate_graph_dir(d)
        assert len(warnings) == 1

    def test_hash_drift(self, tmp_path):
        store = tmp_path / ".neo4j"
        store.mkdir()
        wrong = store / "wronghash1234"
        wrong.mkdir()
        write_dir_meta(wrong, "codex", ["iter"], "wronghash1234")
        warnings = validate_graph_dir(wrong)
        assert any("Hash drift" in w for w in warnings)


# ── Migrate legacy ─────────────────────────────────────────────────────────


class TestMigrateLegacyDir:
    def test_migrate(self, tmp_path):
        # Create a real neo4j/ directory (legacy)
        legacy = tmp_path / "neo4j"
        legacy.mkdir()
        (legacy / "data").mkdir()
        (legacy / "some-file.txt").write_text("test")

        info = migrate_legacy_dir("codex", ["iter"])
        link = tmp_path / "neo4j"
        assert link.is_symlink()
        assert info.name == "codex"
        assert (info.path / "some-file.txt").read_text() == "test"

    def test_migrate_already_symlink_raises(self, tmp_path):
        g = create_graph_dir("codex", ["iter"])
        link = tmp_path / "neo4j"
        link.symlink_to(g.path)
        with pytest.raises(ValueError, match="already a symlink"):
            migrate_legacy_dir("codex", ["iter"])

    def test_migrate_no_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            migrate_legacy_dir("codex", ["iter"])
