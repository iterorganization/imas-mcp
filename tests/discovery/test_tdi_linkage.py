"""Tests for TDI-to-DataNode linkage (discovery/mdsplus/tdi_linkage.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.mdsplus.tdi_linkage import (
    _extract_build_paths,
    link_tdi_to_data_nodes,
    update_signal_accessors,
)


class TestExtractBuildPaths:
    """Test _extract_build_paths helper."""

    def test_simple_build_path(self):
        source = r'build_path("\\RESULTS::TOP.I_P")'
        paths = _extract_build_paths(source)
        assert paths == ["TOP.I_P"]

    def test_single_backslash(self):
        source = r'build_path("\RESULTS::TOP.R_AXIS")'
        paths = _extract_build_paths(source)
        assert paths == ["TOP.R_AXIS"]

    def test_multiple_paths(self):
        source = (
            'build_path("\\\\RESULTS::TOP.I_P")\nbuild_path("\\\\RESULTS::TOP.Q_95")\n'
        )
        paths = _extract_build_paths(source)
        assert sorted(paths) == ["TOP.I_P", "TOP.Q_95"]

    def test_deduplication(self):
        source = (
            'build_path("\\\\RESULTS::TOP.I_P")\nbuild_path("\\\\RESULTS::TOP.I_P")\n'
        )
        paths = _extract_build_paths(source)
        assert paths == ["TOP.I_P"]

    def test_different_trees(self):
        source = (
            'build_path("\\\\RESULTS::TOP.FOO")\nbuild_path("\\\\MAGNETICS::TOP.BAR")\n'
        )
        paths = _extract_build_paths(source)
        assert sorted(paths) == ["TOP.BAR", "TOP.FOO"]

    def test_empty_source(self):
        assert _extract_build_paths("") == []

    def test_no_build_path(self):
        assert _extract_build_paths("fun foo() { return 42; }") == []

    def test_single_quotes(self):
        source = "build_path('\\\\RESULTS::TOP.KAPPA')"
        paths = _extract_build_paths(source)
        assert paths == ["TOP.KAPPA"]


class TestLinkTdiToDataNodes:
    """Test link_tdi_to_data_nodes graph operation."""

    @patch("imas_codex.discovery.mdsplus.tdi_linkage.GraphClient")
    def test_creates_resolves_to_edges(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Query 1: Get TDI functions
        mock_gc.query.side_effect = [
            [
                {
                    "id": "tcv:tdi:tcv_eq",
                    "name": "tcv_eq",
                    "source_code": 'build_path("\\\\RESULTS::TOP.I_P")',
                    "trees": ["results"],
                }
            ],
            # Query 2: Create edges
            [{"linked": 3}],
        ]

        result = link_tdi_to_data_nodes("tcv")

        assert result == 3
        assert mock_gc.query.call_count == 2

    @patch("imas_codex.discovery.mdsplus.tdi_linkage.GraphClient")
    def test_no_tdi_functions(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_gc.query.return_value = []

        result = link_tdi_to_data_nodes("tcv")

        assert result == 0

    @patch("imas_codex.discovery.mdsplus.tdi_linkage.GraphClient")
    def test_no_matching_paths(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_gc.query.side_effect = [
            [
                {
                    "id": "tcv:tdi:tcv_eq",
                    "name": "tcv_eq",
                    "source_code": "fun tcv_eq() { return 1; }",  # no build_path
                    "trees": ["results"],
                }
            ],
        ]

        result = link_tdi_to_data_nodes("tcv")
        assert result == 0


class TestUpdateSignalAccessors:
    """Test update_signal_accessors graph operation."""

    @patch("imas_codex.discovery.mdsplus.tdi_linkage.GraphClient")
    def test_updates_signals(self, mock_gc_cls):
        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_gc.query.return_value = [{"updated": 15}]

        result = update_signal_accessors("tcv")

        assert result == 15
        assert mock_gc.query.call_count == 1
