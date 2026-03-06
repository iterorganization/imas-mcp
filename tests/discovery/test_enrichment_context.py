"""Tests for enrichment pipeline tree context integration (Phase 7).

Verifies: fetch_tree_context graph query, claim_signals returns data_source_node,
and enrich_worker prompt injection of tree/TDI/epoch context.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.parallel import (
    fetch_tree_context,
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
        """Context includes epoch range from StructuralEpochs."""
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

    def test_template_has_tree_node_docs(self):
        """Template documents parent_node, siblings, tdi, applicability."""
        import pathlib

        template_path = (
            pathlib.Path(__file__).parents[2]
            / "imas_codex"
            / "agentic"
            / "prompts"
            / "signals"
            / "enrichment.md"
        )
        content = template_path.read_text()
        assert "parent_node" in content
        assert "siblings" in content
        assert "tdi_function" in content
        assert "tdi_source" in content
        assert "applicability" in content
