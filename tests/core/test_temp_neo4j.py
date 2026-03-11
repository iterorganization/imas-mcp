"""Unit tests for imas_codex.graph.temp_neo4j module.

Tests for temporary Neo4j instance management extracted from graph_cli.py.
"""

from __future__ import annotations

from pathlib import Path

from imas_codex.graph.temp_neo4j import IMAS_DD_LABELS, write_temp_neo4j_conf


# ============================================================================
# IMAS_DD_LABELS
# ============================================================================


class TestImasDdLabels:
    """Tests for the IMAS_DD_LABELS constant."""

    def test_not_empty(self):
        assert len(IMAS_DD_LABELS) > 0

    def test_contains_core_labels(self):
        assert "DDVersion" in IMAS_DD_LABELS
        assert "IDS" in IMAS_DD_LABELS
        assert "IMASNode" in IMAS_DD_LABELS


# ============================================================================
# write_temp_neo4j_conf
# ============================================================================


class TestWriteTempNeo4jConf:
    """Tests for write_temp_neo4j_conf()."""

    def test_creates_conf_file(self, tmp_path):
        conf = write_temp_neo4j_conf(tmp_path, 7688, 7475)
        assert conf.exists()
        content = conf.read_text()
        assert "7688" in content
        assert "7475" in content

    def test_disables_auth(self, tmp_path):
        conf = write_temp_neo4j_conf(tmp_path, 7688, 7475)
        content = conf.read_text()
        assert "auth_enabled=false" in content

    def test_sets_memory_limits(self, tmp_path):
        conf = write_temp_neo4j_conf(tmp_path, 7688, 7475)
        content = conf.read_text()
        assert "heap" in content
        assert "pagecache" in content
