"""Tests for the redesigned `sn clear`, `sn prune`, and `sn sync-grammar`.

The redesign (Plan 39 follow-up) makes `sn clear` an unconditional full
subsystem wipe with auto grammar re-seed, moves scoped deletes to
`sn prune`, and exposes the ISN grammar sync as `sn sync-grammar` (was
`graph sync-isn-grammar`).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnClearHelp:
    """`sn clear --help` should reflect the full-wipe redesign."""

    def _help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["clear", "--help"])
        assert result.exit_code == 0
        return result.output

    def test_clear_help_mentions_full_wipe(self):
        txt = self._help()
        assert "subsystem" in txt.lower() or "wipe" in txt.lower()

    def test_clear_help_has_dry_run(self):
        assert "--dry-run" in self._help()

    def test_clear_help_has_no_reseed(self):
        assert "--no-reseed" in self._help()

    def test_clear_help_has_no_status_flag(self):
        # Scoped flags must have moved to `sn prune`.
        assert "--status" not in self._help()

    def test_clear_help_has_no_source_flag(self):
        assert "--source" not in self._help()

    def test_clear_help_has_no_ids_flag(self):
        assert "--ids" not in self._help()


class TestSnPruneHelp:
    """`sn prune` is the new scoped-delete tool."""

    def _help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["prune", "--help"])
        assert result.exit_code == 0
        return result.output

    def test_prune_registered(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert "prune" in result.output

    def test_prune_has_status(self):
        assert "--status" in self._help()

    def test_prune_has_all(self):
        assert "--all" in self._help()

    def test_prune_has_source(self):
        assert "--source" in self._help()

    def test_prune_has_ids(self):
        assert "--ids" in self._help()

    def test_prune_has_include_accepted(self):
        assert "--include-accepted" in self._help()

    def test_prune_has_include_sources(self):
        assert "--include-sources" in self._help()


class TestSnSyncGrammarHelp:
    """`sn sync-grammar` is the new canonical entry point."""

    def test_sync_grammar_registered(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert "sync-grammar" in result.output

    def test_sync_grammar_help_has_dry_run(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["sync-grammar", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output


class TestClearSnSubsystemLabels:
    """`clear_sn_subsystem` must touch every SN-owned label."""

    def test_counts_all_nine_labels_in_dry_run(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_sn_subsystem(dry_run=True)

        expected_labels = {
            "StandardName",
            "Review",
            "StandardNameSource",
            "VocabGap",
            "SNRun",
            "GrammarToken",
            "GrammarSegment",
            "GrammarTemplate",
            "ISNGrammarVersion",
        }
        assert set(result.keys()) == expected_labels

    def test_dry_run_does_not_reseed(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with (
            patch.object(graph_ops, "GraphClient", return_value=fake_gc),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            graph_ops.clear_sn_subsystem(dry_run=True, reseed_grammar=True)

        mock_sync.assert_not_called()

    def test_wipe_deletes_all_labels_and_reseeds(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 5}])

        with (
            patch.object(graph_ops, "GraphClient", return_value=fake_gc),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            graph_ops.clear_sn_subsystem(dry_run=False, reseed_grammar=True)

        # Should have issued 9 count queries + 9 DETACH DELETE statements
        # (plus any queries the mocked reseed would make — which is
        # patched out so only clear_sn_subsystem's own queries are here).
        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        detach_deletes = [q for q in queries if "DETACH DELETE" in q]
        assert len(detach_deletes) == 9
        # Cover every label exactly once
        for label in (
            "StandardName",
            "Review",
            "StandardNameSource",
            "VocabGap",
            "SNRun",
            "GrammarToken",
            "GrammarTemplate",
            "GrammarSegment",
            "ISNGrammarVersion",
        ):
            assert any(label in q for q in detach_deletes), (
                f"Missing DETACH DELETE for {label}"
            )
        mock_sync.assert_called_once()

    def test_no_reseed_skips_sync(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with (
            patch.object(graph_ops, "GraphClient", return_value=fake_gc),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            graph_ops.clear_sn_subsystem(dry_run=False, reseed_grammar=False)

        mock_sync.assert_not_called()

    def test_review_deleted_before_standardname(self):
        """Pre-p39 bug: deleting StandardName first left orphan Review nodes.

        The order in `clear_sn_subsystem` must be Review → StandardName so
        even in the absence of HAS_STANDARD_NAME edges (pathological data)
        no orphan Reviews are left behind.
        """
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with (
            patch.object(graph_ops, "GraphClient", return_value=fake_gc),
            patch("imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"),
        ):
            graph_ops.clear_sn_subsystem(dry_run=False, reseed_grammar=False)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        review_idx = next(
            i for i, q in enumerate(queries) if "Review" in q and "DELETE" in q
        )
        sn_idx = next(
            i
            for i, q in enumerate(queries)
            if "StandardName" in q and "DELETE" in q and "Source" not in q
        )
        assert review_idx < sn_idx
