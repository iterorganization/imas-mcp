"""Tests for standard name MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSearchStandardNames:
    """Test _search_standard_names tool."""

    def test_keyword_fallback(self):
        """Search falls back to keyword when no embeddings."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": ["core_profiles"],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": "temperature",
                    "subject": "electron",
                    "score": 1.0,
                }
            ]
        )

        # Patch Encoder to fail (trigger keyword fallback)
        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("electron temperature", gc=mock_gc)

        assert "electron_temperature" in result
        mock_gc.query.assert_called()

    def test_empty_results(self):
        """Empty results produce informative message."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("nonexistent quantity", gc=mock_gc)

        assert "No" in result or "0" in result

    def test_kind_filter(self):
        """Kind filter is applied to results."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": [],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": "temperature",
                    "subject": None,
                    "score": 1.0,
                },
                {
                    "name": "velocity_field",
                    "description": "v",
                    "kind": "vector",
                    "unit": "m/s",
                    "tags": [],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": None,
                    "subject": None,
                    "score": 0.8,
                },
            ]
        )

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("temperature", kind="scalar", gc=mock_gc)

        assert "electron_temperature" in result
        assert "velocity_field" not in result

    def test_pipeline_status_filter(self):
        """pipeline_status filter is applied."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "drafted_name",
                    "description": "d",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": [],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": None,
                    "subject": None,
                    "score": 1.0,
                },
                {
                    "name": "published_name",
                    "description": "p",
                    "kind": "scalar",
                    "unit": "A",
                    "tags": [],
                    "pipeline_status": "published",
                    "documentation": None,
                    "physical_base": None,
                    "subject": None,
                    "score": 0.9,
                },
            ]
        )

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names(
                "test", pipeline_status="drafted", gc=mock_gc
            )

        assert "drafted_name" in result
        assert "published_name" not in result

    def test_tags_filter(self):
        """tags filter is applied."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": ["core_profiles", "kinetics"],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": None,
                    "subject": None,
                    "score": 1.0,
                },
                {
                    "name": "equilibrium_shape",
                    "description": "shape",
                    "kind": "scalar",
                    "unit": "m",
                    "tags": ["equilibrium"],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "physical_base": None,
                    "subject": None,
                    "score": 0.8,
                },
            ]
        )

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names(
                "temperature", tags=["core_profiles"], gc=mock_gc
            )

        assert "electron_temperature" in result
        assert "equilibrium_shape" not in result

    def test_result_format_no_grammar_fields(self):
        """Result format no longer includes grammar_* fields (vNext)."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": [],
                    "pipeline_status": "drafted",
                    "documentation": None,
                    "score": 0.92,
                }
            ]
        )

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("electron temperature", gc=mock_gc)

        assert "electron_temperature" in result
        assert "0.92" in result
        # Grammar fields are no longer stored on nodes or displayed
        assert "physical_base=" not in result
        assert "subject=" not in result


class TestFetchStandardNames:
    """Test _fetch_standard_names tool."""

    def test_fetch_single(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te profile",
                    "documentation": "The $T_e$ profile",
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": ["core_profiles"],
                    "links": ["ion_temperature"],
                    "dd_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                    "constraints": ["T_e > 0"],
                    "validity_domain": "core plasma",
                    "physical_base": "temperature",
                    "subject": "electron",
                    "component": None,
                    "coordinate": None,
                    "position": None,
                    "process": None,
                    "pipeline_status": "drafted",
                    "confidence": 0.95,
                    "model": "test",
                    "source_ids": ["core_profiles/profiles_1d/electrons/temperature"],
                    "source_ids_names": ["core_profiles"],
                }
            ]
        )

        result = _fetch_standard_names("electron_temperature", gc=mock_gc)
        assert "electron_temperature" in result
        assert "eV" in result
        assert "$T_e$" in result

    def test_fetch_multiple_comma_separated(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": [],
                    "links": [],
                    "dd_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "physical_base": "temperature",
                    "subject": "electron",
                    "component": None,
                    "coordinate": None,
                    "position": None,
                    "process": None,
                    "pipeline_status": "drafted",
                    "confidence": None,
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                },
                {
                    "name": "plasma_current",
                    "description": "Ip",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "A",
                    "tags": [],
                    "links": [],
                    "dd_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "physical_base": None,
                    "subject": None,
                    "component": None,
                    "coordinate": None,
                    "position": None,
                    "process": None,
                    "pipeline_status": "drafted",
                    "confidence": None,
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                },
            ]
        )

        result = _fetch_standard_names(
            "electron_temperature,plasma_current", gc=mock_gc
        )
        assert "electron_temperature" in result
        assert "plasma_current" in result

    def test_fetch_not_found(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _fetch_standard_names("nonexistent_name", gc=mock_gc)
        assert "not found" in result.lower() or "No" in result

    def test_fetch_partial_not_found(self):
        """Shows not found message for missing names."""
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": [],
                    "links": [],
                    "dd_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "physical_base": None,
                    "subject": None,
                    "component": None,
                    "coordinate": None,
                    "position": None,
                    "process": None,
                    "pipeline_status": "drafted",
                    "confidence": None,
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                }
            ]
        )

        result = _fetch_standard_names("electron_temperature missing_name", gc=mock_gc)
        assert "electron_temperature" in result
        assert "missing_name" in result
        assert "Not found" in result


class TestListStandardNames:
    """Test _list_standard_names tool."""

    def test_list_all(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
                {
                    "name": "plasma_current",
                    "kind": "scalar",
                    "unit": "A",
                    "pipeline_status": "drafted",
                    "description": "Ip",
                },
            ]
        )

        result = _list_standard_names(gc=mock_gc)
        assert "electron_temperature" in result
        assert "plasma_current" in result

    def test_list_with_tag_filter(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(tag="core_profiles", gc=mock_gc)
        assert "electron_temperature" in result
        mock_gc.query.assert_called_once()

    def test_list_empty_results(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _list_standard_names(tag="nonexistent_tag", gc=mock_gc)
        assert "No standard names" in result

    def test_list_filter_info_in_header(self):
        """Filter params appear in header."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(kind="scalar", gc=mock_gc)
        assert "kind=scalar" in result

    def test_list_table_format(self):
        """Output is a markdown table."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(gc=mock_gc)
        assert "| Name |" in result
        assert "| electron_temperature |" in result


class TestMCPToolRegistration:
    """Test that SN tools are importable and callable."""

    def test_tools_importable(self):
        """SN tools should be importable from sn_tools."""
        from imas_codex.llm.sn_tools import (
            _fetch_standard_names,
            _list_standard_names,
            _search_standard_names,
        )

        assert callable(_search_standard_names)
        assert callable(_fetch_standard_names)
        assert callable(_list_standard_names)

    def test_search_signature(self):
        """search_standard_names accepts expected kwargs."""
        import inspect

        from imas_codex.llm.sn_tools import _search_standard_names

        sig = inspect.signature(_search_standard_names)
        params = set(sig.parameters.keys())
        assert "query" in params
        assert "kind" in params
        assert "tags" in params
        assert "pipeline_status" in params
        assert "k" in params
        assert "gc" in params

    def test_fetch_signature(self):
        """fetch_standard_names accepts expected kwargs."""
        import inspect

        from imas_codex.llm.sn_tools import _fetch_standard_names

        sig = inspect.signature(_fetch_standard_names)
        params = set(sig.parameters.keys())
        assert "names" in params
        assert "gc" in params

    def test_list_signature(self):
        """list_standard_names accepts expected kwargs."""
        import inspect

        from imas_codex.llm.sn_tools import _list_standard_names

        sig = inspect.signature(_list_standard_names)
        params = set(sig.parameters.keys())
        assert "tag" in params
        assert "kind" in params
        assert "pipeline_status" in params
        assert "gc" in params
