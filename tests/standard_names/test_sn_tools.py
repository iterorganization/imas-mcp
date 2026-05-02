"""Tests for standard name MCP tools.

Plan-MCP-and-units Track A: signatures dropped ``tag``/``tags`` MCP kwargs
(every prior call raised ``TypeError`` because the backing functions never
accepted them). New canonical filter is ``physics_domain``, which is
pushed into Cypher in all three search branches (segment-filter, vector,
keyword) and into the WHERE clause of ``_list_standard_names``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _row(**overrides):
    """Return a minimal row dict matching the columns returned by the
    three ``_*_search_standard_names`` helpers, with optional overrides."""
    base = {
        "name": "electron_temperature",
        "description": "Te",
        "kind": "scalar",
        "unit": "eV",
        "pipeline_status": "drafted",
        "documentation": None,
        "physics_domain": "transport",
        "cocos_transformation_type": None,
        "cocos": None,
        "score": 1.0,
    }
    base.update(overrides)
    return base


class TestSearchStandardNames:
    """Test _search_standard_names tool."""

    def test_keyword_fallback(self):
        """Search falls back to keyword when no embeddings."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row()])

        # Patch Encoder to fail (trigger keyword fallback)
        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("electron temperature", gc=mock_gc)

        assert "electron_temperature" in result
        mock_gc.query.assert_called()
        # Default physics_domain=None must be forwarded as $pd=None
        call_kwargs = mock_gc.query.call_args.kwargs
        assert call_kwargs.get("pd") is None

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

    def test_no_args_no_typeerror(self):
        """Plan-MCP regression: search with only the query must not raise.

        Before this plan landed, the MCP wrapper forwarded ``tags=None``
        unconditionally, which the backing function never accepted →
        ``TypeError`` on every call.
        """
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            # Must not raise
            _search_standard_names("temperature", gc=mock_gc)

    def test_kind_filter(self):
        """Kind filter is applied to results."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                _row(kind="scalar"),
                _row(name="velocity_field", kind="vector", unit="m/s"),
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
                _row(name="drafted_name", pipeline_status="drafted"),
                _row(name="published_name", pipeline_status="published", unit="A"),
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

    def test_physics_domain_pushed_into_cypher_keyword(self):
        """physics_domain is pushed into the keyword Cypher as $pd."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row(physics_domain="transport")])

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            _search_standard_names(
                "temperature", physics_domain="transport", gc=mock_gc
            )

        call = mock_gc.query.call_args
        cypher = call.args[0] if call.args else ""
        assert "$pd" in cypher, "physics_domain not pushed into Cypher"
        assert call.kwargs.get("pd") == "transport"

    def test_physics_domain_pushed_into_cypher_vector(self):
        """physics_domain is forwarded as $pd to the vector branch."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row(physics_domain="equilibrium")])

        # Make Encoder succeed so we hit the vector branch
        mock_enc = MagicMock()
        mock_enc.embed_texts = MagicMock(return_value=[[0.1] * 8])
        with patch("imas_codex.llm.sn_tools.Encoder", return_value=mock_enc):
            _search_standard_names(
                "temperature", physics_domain="equilibrium", gc=mock_gc
            )

        call = mock_gc.query.call_args
        cypher = call.args[0] if call.args else ""
        assert "vector.queryNodes" in cypher
        assert "$pd" in cypher
        assert call.kwargs.get("pd") == "equilibrium"

    def test_physics_domain_pushed_into_cypher_segment(self):
        """physics_domain is pushed into the segment-filter branch as $pd."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row()])

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            _search_standard_names(
                "temperature",
                physics_domain="transport",
                physical_base="temperature",
                gc=mock_gc,
            )

        call = mock_gc.query.call_args
        cypher = call.args[0] if call.args else ""
        assert "$pd" in cypher
        assert call.kwargs.get("pd") == "transport"

    def test_physics_domain_default_null(self):
        """When physics_domain is None, $pd=null short-circuits the filter."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row()])

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            _search_standard_names("temperature", gc=mock_gc)

        assert mock_gc.query.call_args.kwargs.get("pd") is None

    def test_result_format_no_grammar_fields(self):
        """Result format no longer includes grammar_* fields (vNext)."""
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[_row(score=0.92)])

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
                    "links": ["ion_temperature"],
                    "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                    "constraints": ["T_e > 0"],
                    "validity_domain": "core plasma",
                    "pipeline_status": "drafted",
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
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
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
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
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
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
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

    def test_list_no_args_no_typeerror(self):
        """Plan-MCP regression: list with no args must not raise.

        Before this plan landed, the MCP wrapper forwarded ``tag=None``
        unconditionally, which the backing function never accepted.
        """
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        # Must not raise
        _list_standard_names(gc=mock_gc)

    def test_list_with_physics_domain_filter(self):
        """physics_domain filter is pushed into the Cypher WHERE clause."""
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

        result = _list_standard_names(physics_domain="transport", gc=mock_gc)
        assert "electron_temperature" in result
        call = mock_gc.query.call_args
        cypher = call.args[0] if call.args else ""
        assert "physics_domain" in cypher
        assert call.kwargs.get("physics_domain") == "transport"

    def test_list_empty_results(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _list_standard_names(physics_domain="nonexistent_domain", gc=mock_gc)
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

    def test_list_physics_domain_in_header(self):
        """physics_domain filter shows in header."""
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

        result = _list_standard_names(physics_domain="transport", gc=mock_gc)
        assert "physics_domain=transport" in result

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
    """Test that SN tools are importable and have the post-plan signatures."""

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
        """search_standard_names accepts physics_domain (not tags)."""
        import inspect

        from imas_codex.llm.sn_tools import _search_standard_names

        sig = inspect.signature(_search_standard_names)
        params = set(sig.parameters.keys())
        assert "query" in params
        assert "kind" in params
        assert "physics_domain" in params
        assert "pipeline_status" in params
        assert "k" in params
        assert "gc" in params
        # The legacy ``tags`` filter has been dropped (Plan MCP+units Track A)
        assert "tags" not in params
        assert "tag" not in params

    def test_fetch_signature(self):
        """fetch_standard_names accepts expected kwargs."""
        import inspect

        from imas_codex.llm.sn_tools import _fetch_standard_names

        sig = inspect.signature(_fetch_standard_names)
        params = set(sig.parameters.keys())
        assert "names" in params
        assert "gc" in params

    def test_list_signature(self):
        """list_standard_names accepts physics_domain (not tag)."""
        import inspect

        from imas_codex.llm.sn_tools import _list_standard_names

        sig = inspect.signature(_list_standard_names)
        params = set(sig.parameters.keys())
        assert "physics_domain" in params
        assert "kind" in params
        assert "pipeline_status" in params
        assert "gc" in params
        # The legacy ``tag`` filter has been dropped
        assert "tag" not in params
        assert "tags" not in params

    def test_mcp_wrapper_no_tags_tag(self):
        """MCP wrappers must not declare tag/tags parameters."""
        # Server module must import cleanly and contain no `tag`/`tags`
        # default parameters in the SN wrapper signatures.
        import inspect

        from imas_codex.llm import server  # noqa: F401

        src = inspect.getsource(server)
        # Anchor checks to the SN tool definitions, not surrounding prose.
        assert "tags: list[str] | None = None" not in src
        # The list_standard_names wrapper must not declare ``tag``.
        assert "def list_standard_names(\n                tag:" not in src


class TestSupersededExclusion:
    """Superseded SNs must not appear in search or list results."""

    def _superseded_row(self, name: str = "electron_heating_power") -> dict:
        return {
            "name": name,
            "description": "Superseded fossil",
            "kind": "scalar",
            "unit": "1",
            "pipeline_status": "superseded",
            "documentation": None,
            "physics_domain": "transport",
            "cocos_transformation_type": None,
            "cocos": None,
            "score": 0.99,
        }

    def _active_row(self, name: str = "electron_temperature") -> dict:
        return {
            "name": name,
            "description": "Electron temperature",
            "kind": "scalar",
            "unit": "eV",
            "pipeline_status": "drafted",
            "documentation": None,
            "physics_domain": "transport",
            "cocos_transformation_type": None,
            "cocos": None,
            "score": 0.85,
        }

    def test_keyword_cypher_excludes_superseded(self):
        """Keyword-branch Cypher must include name_stage <> 'superseded'."""
        from imas_codex.llm.sn_tools import _keyword_search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _keyword_search_standard_names(mock_gc, "electron temperature", k=5)
        cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in cypher and "superseded" in cypher

    def test_vector_cypher_excludes_superseded(self):
        """Vector-branch Cypher must include name_stage <> 'superseded'."""
        from imas_codex.llm.sn_tools import _vector_search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _vector_search_standard_names(mock_gc, [0.1] * 8, k=5)
        cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in cypher and "superseded" in cypher

    def test_segment_filter_cypher_excludes_superseded(self):
        """Segment-filter Cypher must include name_stage <> 'superseded'."""
        from imas_codex.llm.sn_tools import _segment_filter_search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _segment_filter_search_standard_names(
            mock_gc, "temperature", 5, {"physical_base": "temperature"}
        )
        cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in cypher and "superseded" in cypher

    def test_search_superseded_row_filtered_by_default(self):
        """Cypher guard must contain name_stage <> 'superseded' in all branches.

        Validates the Cypher-level fix is in place (DB will exclude superseded
        nodes before returning rows).  The mock returns an empty list since the
        Cypher guard is what prevents fossils from appearing in production.
        """
        from imas_codex.llm.sn_tools import _search_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch(
            "imas_codex.llm.sn_tools.Encoder", side_effect=Exception("no embeddings")
        ):
            result = _search_standard_names("electron temperature", gc=mock_gc)

        # The Cypher itself must include the name_stage guard.
        call_cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in call_cypher, (
            "name_stage guard missing from keyword Cypher"
        )
        assert "superseded" in call_cypher, (
            "'superseded' literal missing from keyword Cypher"
        )

        # No superseded fossil should appear in the formatted output.
        assert "electron_heating_power" not in result

    def test_list_excludes_superseded_by_default(self):
        """_list_standard_names must exclude superseded by default."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _list_standard_names(gc=mock_gc)
        cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in cypher and "superseded" in cypher

    def test_list_include_superseded_flag(self):
        """With include_superseded=True, no name_stage guard in Cypher."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _list_standard_names(include_superseded=True, gc=mock_gc)
        cypher = mock_gc.query.call_args.args[0]
        # The guard should be absent when caller opts-in to superseded
        assert "name_stage" not in cypher or "superseded" not in cypher
