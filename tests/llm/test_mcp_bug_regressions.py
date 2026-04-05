"""Regression tests for MCP tool bugs found during A/B testing.

Each test targets a specific bug that was discovered when comparing
DD-only vs graph-backed tool behavior.  Tests are designed to fail on
the buggy code and pass on the fixed code, without requiring a live
Neo4j connection.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bug 1 & 2: export_imas_ids / export_imas_domain must NOT expose
# ``include_errors`` — the underlying GraphStructureTool methods don't
# accept that parameter, so passing it would raise TypeError.
# ---------------------------------------------------------------------------


class TestExportHandlersNoIncludeErrors:
    """Bug 1 & 2: Server handlers must not pass include_errors to graph tools."""

    def test_export_imas_ids_tool_has_no_include_errors_param(self):
        """GraphStructureTool.export_imas_ids must not accept include_errors."""
        from imas_codex.tools.graph_search import GraphStructureTool

        sig = inspect.signature(GraphStructureTool.export_imas_ids)
        assert "include_errors" not in sig.parameters, (
            "GraphStructureTool.export_imas_ids gained an unexpected "
            "'include_errors' parameter — the server handler must not "
            "pass this kwarg"
        )

    def test_export_imas_domain_tool_has_no_include_errors_param(self):
        """GraphStructureTool.export_imas_domain must not accept include_errors."""
        from imas_codex.tools.graph_search import GraphStructureTool

        sig = inspect.signature(GraphStructureTool.export_imas_domain)
        assert "include_errors" not in sig.parameters, (
            "GraphStructureTool.export_imas_domain gained an unexpected "
            "'include_errors' parameter — the server handler must not "
            "pass this kwarg"
        )

    def test_export_imas_ids_server_handler_no_include_errors(self):
        """The DD-only server handler for export_imas_ids must omit include_errors."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        # Walk registered tool components to find the handler
        for key, component in server.mcp._local_provider._components.items():
            if key == "tool:export_imas_ids":
                fn = component.fn
                sig = inspect.signature(fn)
                assert "include_errors" not in sig.parameters, (
                    "DD-only export_imas_ids handler still has include_errors"
                )
                break

    def test_export_imas_domain_server_handler_no_include_errors(self):
        """The DD-only server handler for export_imas_domain must omit include_errors."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        for key, component in server.mcp._local_provider._components.items():
            if key == "tool:export_imas_domain":
                fn = component.fn
                sig = inspect.signature(fn)
                assert "include_errors" not in sig.parameters, (
                    "DD-only export_imas_domain handler still has include_errors"
                )
                break


# ---------------------------------------------------------------------------
# Bug 4: Short physics terms like "ip", "q", "b0" must not be filtered out
# by the ``len(w) > 2`` word-length check in query_words construction.
# ---------------------------------------------------------------------------


class TestShortPhysicsTermsPreserved:
    """Bug 4: Physics abbreviations <=2 chars must survive word filtering."""

    def test_physics_short_terms_set_exists(self):
        """A frozenset of short physics terms must be defined for exemption."""
        from imas_codex.tools.graph_search import _PHYSICS_SHORT_TERMS

        assert isinstance(_PHYSICS_SHORT_TERMS, frozenset)
        # Must contain the canonical short physics abbreviations
        for term in ("ip", "q", "b0", "te", "ne", "ti", "ni", "li"):
            assert term in _PHYSICS_SHORT_TERMS, (
                f"'{term}' missing from _PHYSICS_SHORT_TERMS"
            )

    def test_search_dd_paths_path_segment_boost_preserves_short_terms(self):
        """query_words in search_dd_paths must exempt _PHYSICS_SHORT_TERMS."""
        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_dd_paths)

        # The bug was: [w.lower() for w in ... if len(w) > 2]
        # The fix adds: or w.lower() in _PHYSICS_SHORT_TERMS
        # Verify the exemption is present wherever query_words is built
        assert "_PHYSICS_SHORT_TERMS" in source, (
            "search_dd_paths must reference _PHYSICS_SHORT_TERMS to "
            "exempt short physics abbreviations from length filtering"
        )

    def test_text_search_query_words_preserves_short_terms(self):
        """_text_search_dd_paths query_words must exempt short physics terms."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        source = inspect.getsource(_text_search_dd_paths)

        # Same check: must reference the exemption set
        assert "_PHYSICS_SHORT_TERMS" in source, (
            "_text_search_dd_paths must reference _PHYSICS_SHORT_TERMS to "
            "exempt short physics abbreviations from length filtering"
        )

    def test_physics_abbreviations_include_short_terms(self):
        """PHYSICS_ABBREVIATIONS must contain common short-form terms."""
        from imas_codex.tools.query_analysis import PHYSICS_ABBREVIATIONS

        for term in ("ip", "q", "te", "ne", "ti", "ni", "li"):
            assert term in PHYSICS_ABBREVIATIONS, (
                f"'{term}' missing from PHYSICS_ABBREVIATIONS"
            )

    @pytest.mark.skip(reason="Planned feature: abbreviation boost not yet implemented")
    def test_abbreviation_exact_match_boost_exists(self):
        """Abbreviation queries must get a strong terminal-match boost.

        When the original query is a physics abbreviation like "ip", the
        search must apply a large extra boost (≥0.30) to paths whose
        terminal segment exactly matches the original (pre-expansion)
        query.  This ensures .../ip ranks above .../bootstrap_current.
        """
        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_dd_paths)

        # Must check is_abbreviation from the query intent
        assert "is_abbreviation" in source, (
            "search_dd_paths must check intent.is_abbreviation to apply "
            "an extra boost for abbreviation queries"
        )
        # Must reference original_query for the pre-expansion terms
        assert "original_query" in source, (
            "search_dd_paths must use intent.original_query for "
            "abbreviation terminal-match boosting"
        )


# ---------------------------------------------------------------------------
# Bug 5: Coordinate channel in find_related_dd_paths (get_dd_path_context)
# must traverse through IMASCoordinateSpec for coordinate partner discovery.
# The HAS_COORDINATE relationship now correctly points to IMASCoordinateSpec
# nodes, which hold coordinate specifications used across IDSs.
# ---------------------------------------------------------------------------


class TestCoordinateChannelTargetsIMASCoordinateSpec:
    """Bug 5 (updated): Coordinate query must match through (coord:IMASCoordinateSpec)."""

    def test_coordinate_query_uses_coordinate_spec_label(self):
        """The HAS_COORDINATE Cypher must traverse (coord:IMASCoordinateSpec)."""
        from imas_codex.tools.graph_search import GraphPathContextTool

        source = inspect.getsource(GraphPathContextTool.get_dd_path_context)

        # Find the coordinate partners query
        coord_section = source[source.index("Coordinate partners") :]
        coord_section = coord_section[: coord_section.index("Unit companions")]

        # Must use (coord:IMASCoordinateSpec)
        assert "(coord:IMASCoordinateSpec)" in coord_section, (
            "Coordinate partner query should use (coord:IMASCoordinateSpec) to match "
            "coordinate specs for cross-IDS discovery"
        )

    @pytest.mark.asyncio
    async def test_coordinate_query_dispatched_correctly(self):
        """Mock-verify the coordinate Cypher sent to the graph client."""
        from imas_codex.tools.graph_search import GraphPathContextTool

        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphPathContextTool(gc)

        await tool.get_dd_path_context(
            path="equilibrium/time_slice/profiles_1d/psi",
            relationship_types="coordinate",
        )

        # At least one query should have been dispatched
        assert gc.query.called
        coord_calls = [
            call.args[0]
            for call in gc.query.call_args_list
            if "HAS_COORDINATE" in call.args[0]
        ]
        assert coord_calls, "No HAS_COORDINATE query was dispatched"
        for cypher in coord_calls:
            assert "IMASCoordinateSpec" in cypher, (
                f"Coordinate query must target IMASCoordinateSpec, got: {cypher[:200]}"
            )


# ---------------------------------------------------------------------------
# Bug 6: Cluster search must increase vector k when ids_filter is applied.
# Post-retrieval filtering reduces result count, so the initial retrieval
# must fetch more candidates to compensate.
# ---------------------------------------------------------------------------


class TestClusterSearchIncreasesKWithIdsFilter:
    """Bug 6: Vector search k should increase when ids_filter narrows results."""

    @staticmethod
    def _extract_k_from_calls(
        gc: MagicMock, pattern: str = "vector.queryNodes"
    ) -> list[int]:
        """Extract the k values sent in vector search queries."""
        k_values = []
        for call in gc.query.call_args_list:
            cypher = call.args[0] if call.args else ""
            if pattern in cypher:
                kwargs = call.kwargs
                k = kwargs.get("k")
                if k is not None:
                    k_values.append(k)
        return k_values

    @patch("imas_codex.tools.graph_search._get_encoder")
    def test_k_increases_with_ids_filter(self, mock_encoder):
        """k should be larger when ids_filter is set than without it."""
        from imas_codex.tools.graph_search import GraphClustersTool

        mock_enc = MagicMock()
        mock_enc.encode.return_value = [0.1] * 384
        mock_encoder.return_value = mock_enc

        # Without filter
        gc_no_filter = MagicMock()
        gc_no_filter.query.return_value = []
        tool_no_filter = GraphClustersTool(gc_no_filter)
        tool_no_filter._search_by_text("safety factor", scope=None, ids_filter=None)
        k_no_filter = self._extract_k_from_calls(gc_no_filter)

        # With filter
        gc_with_filter = MagicMock()
        gc_with_filter.query.return_value = []
        tool_with_filter = GraphClustersTool(gc_with_filter)
        tool_with_filter._search_by_text(
            "safety factor", scope=None, ids_filter="equilibrium"
        )
        k_with_filter = self._extract_k_from_calls(gc_with_filter)

        assert k_no_filter, "No vector query dispatched without filter"
        assert k_with_filter, "No vector query dispatched with filter"
        assert k_with_filter[0] > k_no_filter[0], (
            f"k with ids_filter ({k_with_filter[0]}) should be larger than "
            f"k without filter ({k_no_filter[0]}) to compensate for "
            f"post-retrieval filtering"
        )


# ---------------------------------------------------------------------------
# Bug 7: In _get_removals, the ids_clause (WHERE) must appear BEFORE
# OPTIONAL MATCH so it filters the required MATCH, not the optional one.
# Placing WHERE after OPTIONAL MATCH turns the IDS filter into a filter
# on the replacement node instead of the removed path.
# ---------------------------------------------------------------------------


class TestMigrationRemovalsWhereClause:
    """Bug 7: ids_clause must precede OPTIONAL MATCH in _get_removals."""

    def test_where_before_optional_match_in_source(self):
        """In _get_removals Cypher, WHERE/ids_clause must come before OPTIONAL MATCH."""
        from imas_codex.tools.migration_guide import _get_removals

        source = inspect.getsource(_get_removals)

        # Extract the Cypher template from the source
        # Find the f-string or string containing the query
        assert "OPTIONAL MATCH" in source, "_get_removals must use OPTIONAL MATCH"

        # The ids_clause must appear BEFORE the OPTIONAL MATCH line.
        # In Cypher, a WHERE after OPTIONAL MATCH applies to the OPTIONAL
        # pattern, not the preceding MATCH — that's the bug.
        #
        # Correct order:
        #   MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        #   WHERE true {ids_clause}
        #   OPTIONAL MATCH (p)-[:RENAMED_TO]->(replacement)
        #
        # Buggy order:
        #   MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        #   OPTIONAL MATCH (p)-[:RENAMED_TO]->(replacement)
        #   WHERE true {ids_clause}

        # Find positions in source
        optional_pos = source.index("OPTIONAL MATCH")
        ids_clause_pos = source.index("ids_clause")

        # The ids_clause interpolation must appear before OPTIONAL MATCH
        assert ids_clause_pos < optional_pos, (
            "ids_clause is placed AFTER OPTIONAL MATCH — the WHERE filter "
            "will apply to the optional pattern instead of the required MATCH"
        )

    def test_get_removals_ids_filter_restricts_path_not_replacement(self):
        """With ids_filter, only paths from that IDS are returned."""
        from imas_codex.tools.migration_guide import _get_removals

        gc = MagicMock()
        gc.query.return_value = []

        _get_removals(gc, ["4.0.0"], ids_filter="equilibrium")

        assert gc.query.called
        cypher = gc.query.call_args.args[0]

        # The WHERE clause with ids_filter must be between the second MATCH
        # and the OPTIONAL MATCH
        match2_pos = cypher.index("FOR_IMAS_PATH")
        optional_pos = cypher.index("OPTIONAL MATCH")

        # Find 'ids_filter' reference in the cypher
        ids_ref_pos = cypher.index("$ids_filter")

        assert match2_pos < ids_ref_pos < optional_pos, (
            "ids_filter reference must appear between the MATCH for p "
            "and the OPTIONAL MATCH for replacement. "
            f"Positions: FOR_IMAS_PATH={match2_pos}, "
            f"ids_filter={ids_ref_pos}, OPTIONAL={optional_pos}"
        )
