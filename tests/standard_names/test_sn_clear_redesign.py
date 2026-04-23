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

    def test_clear_help_has_no_reseed_flag_removed(self):
        # Grammar is no longer touched by clear, so --no-reseed is gone.
        assert "--no-reseed" not in self._help()

    def test_clear_help_mentions_sync_grammar(self):
        # The help text should point users at `sn sync-grammar` for
        # grammar refreshes (since clear no longer touches grammar).
        assert "sync-grammar" in self._help()

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
    """`clear_sn_subsystem` must touch every SN-pipeline-output label.

    Grammar labels (GrammarToken, GrammarSegment, GrammarTemplate,
    ISNGrammarVersion) are ISN-authoritative reference data and are
    NEVER touched by `sn clear` — they are refreshed via
    `sn sync-grammar` when ISN is upgraded.
    """

    _EXPECTED_LABELS = {
        "StandardName",
        "Review",
        "StandardNameSource",
        "VocabGap",
        "SNRun",
    }

    _GRAMMAR_LABELS = {
        "GrammarToken",
        "GrammarSegment",
        "GrammarTemplate",
        "ISNGrammarVersion",
    }

    def test_counts_only_pipeline_output_labels(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_sn_subsystem(dry_run=True)

        assert set(result.keys()) == self._EXPECTED_LABELS
        # Must not touch grammar labels.
        assert not (set(result.keys()) & self._GRAMMAR_LABELS)

    def test_dry_run_does_not_touch_graph(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=True)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries)

    def test_wipe_deletes_only_pipeline_labels(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 5}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        detach_deletes = [q for q in queries if "DETACH DELETE" in q]
        # One DETACH DELETE per pipeline label — NOT 9.
        assert len(detach_deletes) == len(self._EXPECTED_LABELS)
        for label in self._EXPECTED_LABELS:
            assert any(label in q for q in detach_deletes), (
                f"Missing DETACH DELETE for {label}"
            )
        # Must NOT issue DETACH DELETE on any grammar label.
        for label in self._GRAMMAR_LABELS:
            assert not any(label in q for q in detach_deletes), (
                f"clear_sn_subsystem should not touch grammar label {label}"
            )

    def test_wipe_never_calls_grammar_sync(self):
        """Post-redesign: clear does not auto-reseed grammar.

        Grammar is reference data that stays in the graph. Refreshing
        is a separate concern exposed via `sn sync-grammar`.
        """
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
            graph_ops.clear_sn_subsystem(dry_run=False)

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

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

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
