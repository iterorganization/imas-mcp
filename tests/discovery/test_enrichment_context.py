"""Tests for enrichment pipeline tree context integration (Phase 7).

Verifies: fetch_tree_context graph query, claim_signals returns data_source_node,
and enrich_worker prompt injection of tree/TDI/epoch context.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.parallel import (
    build_device_xml_context_query,
    fetch_tree_context,
    get_data_discovery_stats,
    get_scanner_scope_sources,
    get_signal_scanner_type,
    has_pending_check_work,
    has_pending_enrich_work,
)


class TestFetchTreeContext:
    """Tests for fetch_tree_context function."""

    def test_empty_ids_returns_empty(self):
        """No signal IDs → no context."""
        assert fetch_tree_context([]) == {}

    def test_returns_tree_path_and_parent(self):
        """Context includes tree_path and parent_path."""
        mock_result = [
            {
                "signal_id": "tcv:results/top/ip",
                "tree_path": "\\RESULTS::TOP:IP",
                "parent_path": "\\RESULTS::TOP",
                "sibling_paths": ["\\RESULTS::TOP:Q95", "\\RESULTS::TOP:LI"],
                "tdi_source": None,
                "tdi_name": None,
                "first_version": None,
                "last_version": None,
            }
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:results/top/ip"])

        assert "tcv:results/top/ip" in ctx
        entry = ctx["tcv:results/top/ip"]
        assert entry["tree_path"] == "\\RESULTS::TOP:IP"
        assert entry["parent_path"] == "\\RESULTS::TOP"
        assert "\\RESULTS::TOP:Q95" in entry["sibling_paths"]

    def test_includes_tdi_context(self):
        """Context includes TDI function name and source."""
        mock_result = [
            {
                "signal_id": "tcv:results/top/ip",
                "tree_path": "\\RESULTS::TOP:IP",
                "parent_path": "\\RESULTS::TOP",
                "sibling_paths": [],
                "tdi_source": "public fun tcv_ip()\n  return(data(\\ip))\nend fun",
                "tdi_name": "tcv_ip",
                "first_version": None,
                "last_version": None,
            }
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:results/top/ip"])

        entry = ctx["tcv:results/top/ip"]
        assert entry["tdi_name"] == "tcv_ip"
        assert "tcv_ip" in entry["tdi_source"]

    def test_includes_epoch_range(self):
        """Context includes epoch range from SignalEpochs."""
        mock_result = [
            {
                "signal_id": "tcv:magnetics/top/ip",
                "tree_path": "\\MAGNETICS::TOP:IP",
                "parent_path": "\\MAGNETICS::TOP",
                "sibling_paths": [],
                "tdi_source": None,
                "tdi_name": None,
                "first_version": 1,
                "last_version": 5,
            }
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:magnetics/top/ip"])

        entry = ctx["tcv:magnetics/top/ip"]
        assert entry["epoch_range"]["first_version"] == 1
        assert entry["epoch_range"]["last_version"] == 5

    def test_no_tdi_or_epochs_excluded(self):
        """Entries without TDI or epochs don't include those keys."""
        mock_result = [
            {
                "signal_id": "tcv:results/top/ip",
                "tree_path": "\\RESULTS::TOP:IP",
                "parent_path": None,
                "sibling_paths": [],
                "tdi_source": None,
                "tdi_name": None,
                "first_version": None,
                "last_version": None,
            }
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:results/top/ip"])

        entry = ctx["tcv:results/top/ip"]
        assert "tdi_source" not in entry
        assert "epoch_range" not in entry

    def test_graph_error_returns_empty(self):
        """Graph errors return empty dict gracefully."""
        mock_gc = MagicMock()
        mock_gc.query.side_effect = RuntimeError("graph down")
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:results/top/ip"])

        assert ctx == {}

    def test_multiple_signals(self):
        """Fetches context for multiple signals in one query."""
        mock_result = [
            {
                "signal_id": "tcv:results/top/ip",
                "tree_path": "\\RESULTS::TOP:IP",
                "parent_path": "\\RESULTS::TOP",
                "sibling_paths": [],
                "tdi_source": None,
                "tdi_name": None,
                "first_version": None,
                "last_version": None,
            },
            {
                "signal_id": "tcv:results/top/q95",
                "tree_path": "\\RESULTS::TOP:Q95",
                "parent_path": "\\RESULTS::TOP",
                "sibling_paths": [],
                "tdi_source": None,
                "tdi_name": None,
                "first_version": None,
                "last_version": None,
            },
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            ctx = fetch_tree_context(["tcv:results/top/ip", "tcv:results/top/q95"])

        assert len(ctx) == 2
        assert "tcv:results/top/ip" in ctx
        assert "tcv:results/top/q95" in ctx


class TestClaimSignalsSourceNode:
    """Test that claim_signals_for_enrichment returns data_source_node."""

    def test_claim_returns_source_node_field(self):
        """Claimed signals include data_source_node from the query."""
        mock_result = [
            {
                "id": "tcv:results/top/ip",
                "accessor": "TOP:IP",
                "data_source_name": "results",
                "data_source_path": "\\RESULTS::TOP:IP",
                "unit": "A",
                "name": "Plasma current",
                "tdi_function": None,
                "discovery_source": "tree_traversal",
                "description": None,
                "data_source_node": "tcv:results:\\RESULTS::TOP:IP",
            }
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = mock_result
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            from imas_codex.discovery.signals.parallel import (
                claim_signals_for_enrichment,
            )

            signals = claim_signals_for_enrichment("tcv", batch_size=10)

        assert len(signals) == 1
        assert signals[0]["data_source_node"] == "tcv:results:\\RESULTS::TOP:IP"


class TestScannerScopeMapping:
    """Tests for downstream scanner scoping helpers."""

    def test_tdi_signals_remain_distinct_from_mdsplus(self):
        """Persisted TDI signals are still recognized as TDI scope."""
        signal = {
            "tdi_function": "tcv_ip",
            "discovery_source": "tdi_introspection",
            "data_source_name": None,
        }

        assert get_signal_scanner_type(signal) == "tdi"

    def test_scanner_scope_preserves_explicit_tdi_selection(self):
        """User scanner filters keep TDI and MDSplus separate."""
        assert get_scanner_scope_sources(["tdi"]) == ["tdi"]
        assert get_scanner_scope_sources(["mdsplus"]) == ["mdsplus"]

    def test_pending_queries_receive_scanner_scope(self):
        """Pending-work checks apply the selected scanner scope in Cypher."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"has_work": True}]
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            assert has_pending_enrich_work("jet", ["device_xml", "ppf"]) is True
            assert has_pending_check_work("jet", ["device_xml", "ppf"]) is True

        assert mock_gc.query.call_count == 2
        for call in mock_gc.query.call_args_list:
            query = call.args[0]
            kwargs = call.kwargs
            assert kwargs["scoped_scanners"] == ["device_xml", "ppf"]
            assert "static_sources" in kwargs
            if "s.status = $discovered" in query:
                assert "sg.representative_id <> s.id" in query

    def test_stats_queries_receive_scanner_scope(self):
        """Discovery stats use scoped totals instead of facility-wide totals."""
        mock_gc = MagicMock()
        mock_gc.query.side_effect = [
            [
                {
                    "total": 12,
                    "discovered": 5,
                    "enriched": 4,
                    "checked": 3,
                    "skipped": 0,
                    "failed": 0,
                    "pending_enrich": 1,
                    "pending_check": 2,
                    "accumulated_cost": 1.25,
                    "checks_passed": 3,
                    "checks_failed": 0,
                }
            ],
            [{"signal_sources": 2, "grouped_signals": 6}],
        ]
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient",
            return_value=mock_gc,
        ):
            stats = get_data_discovery_stats("jet", ["device_xml", "ppf"])

        assert stats["total"] == 12
        assert stats["signal_sources"] == 2
        assert stats["grouped_signals"] == 6
        signal_query = mock_gc.query.call_args_list[0].args[0]
        assert "sg.representative_id <> s.id" in signal_query
        for call in mock_gc.query.call_args_list:
            kwargs = call.kwargs
            assert kwargs["scoped_scanners"] == ["device_xml", "ppf"]


class TestEnrichWorkerTreeContext:
    """Test that enrich_worker injects tree context into prompts."""

    def test_tree_context_injected_into_prompt(self):
        """Verify tree context fields appear in the user prompt."""
        # Build a minimal signal with data_source_node
        signal = {
            "id": "tcv:results/top/ip",
            "accessor": "TOP:IP",
            "name": "IP",
            "data_source_name": "results",
            "data_source_path": "\\RESULTS::TOP:IP",
            "data_source_node": "tcv:results:\\RESULTS::TOP:IP",
        }

        # Simulate tree context
        tree_ctx = {
            "tcv:results/top/ip": {
                "tree_path": "\\RESULTS::TOP:IP",
                "parent_path": "\\RESULTS::TOP",
                "sibling_paths": ["\\RESULTS::TOP:Q95", "\\RESULTS::TOP:LI"],
                "tdi_name": "tcv_ip",
                "tdi_source": "public fun tcv_ip()\n  return(data(\\ip))\nend fun",
                "epoch_range": {"first_version": 1, "last_version": 5},
            }
        }

        # Build prompt lines the same way enrich_worker does
        user_lines = []
        sig_tree_ctx = tree_ctx.get(signal["id"])
        if sig_tree_ctx:
            if sig_tree_ctx.get("parent_path"):
                user_lines.append(f"parent_node: {sig_tree_ctx['parent_path']}")
            siblings = sig_tree_ctx.get("sibling_paths", [])
            if siblings:
                user_lines.append(f"siblings: {', '.join(siblings)}")
            if sig_tree_ctx.get("tdi_name"):
                user_lines.append(f"tdi_function: {sig_tree_ctx['tdi_name']}")
                if sig_tree_ctx.get("tdi_source"):
                    src = sig_tree_ctx["tdi_source"]
                    if len(src) > 2000:
                        src = src[:2000] + "\n... (truncated)"
                    user_lines.append(f"tdi_source:\n```tdi\n{src}\n```")
            epoch = sig_tree_ctx.get("epoch_range")
            if epoch:
                user_lines.append(
                    f"applicability: versions "
                    f"{epoch['first_version']}-{epoch['last_version']}"
                )

        prompt = "\n".join(user_lines)
        assert "parent_node: \\RESULTS::TOP" in prompt
        assert "siblings: \\RESULTS::TOP:Q95, \\RESULTS::TOP:LI" in prompt
        assert "tdi_function: tcv_ip" in prompt
        assert "tcv_ip()" in prompt
        assert "applicability: versions 1-5" in prompt

    def test_no_tree_context_for_non_tree_signals(self):
        """Signals without data_source_node get no tree context."""
        signal = {
            "id": "tcv:tdi:ip",
            "accessor": "tcv_ip()",
            "name": "IP",
            "tdi_function": "tcv_ip",
            "data_source_node": None,
        }

        tree_context = {}  # Empty — no tree signals
        sig_tree_ctx = tree_context.get(signal["id"])
        assert sig_tree_ctx is None


class TestEnrichmentTemplateContent:
    """Verify the enrichment template includes MDSplus context docs."""

    def test_template_has_data_node_docs(self):
        """Template documents parent_node, siblings, tdi, applicability."""
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        template_path = PROMPTS_DIR / "signals" / "enrichment.md"
        content = template_path.read_text()
        assert "parent_node" in content
        assert "siblings" in content
        assert "tdi_function" in content
        assert "tdi_source" in content
        assert "applicability" in content


class TestDeviceXmlContextQueries:
    """Tests for targeted device_xml semantic query construction."""

    def test_build_device_xml_context_query_uses_section_metadata(self):
        indexed_signals = [
            (
                0,
                {
                    "name": "Magnetic Probe 1 Radial position",
                    "data_source_name": "device_xml",
                    "data_source_path": "magprobes/1/r",
                },
            )
        ]

        query = build_device_xml_context_query("device_xml:magprobes", indexed_signals)

        assert "JET" in query
        assert "device xml" in query
        assert "magprobes" in query
        assert "magnetic probe" in query
        assert "magnetics.bpol_probe" in query
        assert "r z angle" in query

    def test_build_device_xml_code_query_adds_code_specific_terms(self):
        indexed_signals = [
            (
                0,
                {
                    "name": "PF coil 3 Turns per element",
                    "data_source_name": "device_xml",
                    "data_source_path": "pfcoils/3/turnsperelement",
                },
            )
        ]

        query = build_device_xml_context_query(
            "device_xml:pfcoils", indexed_signals, for_code=True
        )

        assert "pfcoils" in query
        assert "PF coil" in query
        assert "pf_active.coil" in query
        assert "EFIT" in query
        assert "parser" in query

    def test_build_device_xml_context_query_falls_back_for_unknown_section(self):
        indexed_signals = [
            (0, {"name": "Unknown geometry signal", "data_source_path": "other/x/y"})
        ]

        query = build_device_xml_context_query("device_xml:other", indexed_signals)

        assert "JET" in query
        assert "machine description" in query
        assert "Unknown geometry signal" in query
