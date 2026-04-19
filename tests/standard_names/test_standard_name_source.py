"""Tests for StandardNameSource schema and graph operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestStandardNameSourceSchema:
    """Verify schema generates correct models."""

    def test_source_importable(self):
        from imas_codex.graph.models import StandardNameSource

        assert StandardNameSource is not None

    def test_status_enum_importable(self):
        from imas_codex.graph.models import StandardNameSourceStatus

        members = set(StandardNameSourceStatus.__members__.keys())
        assert members == {
            "extracted",
            "composed",
            "attached",
            "vocab_gap",
            "failed",
            "stale",
            "skipped",
        }

    def test_source_type_enum_distinct(self):
        """StandardNameSourceType (enum) != StandardNameSource (class)."""
        from imas_codex.graph.models import StandardNameSource, StandardNameSourceType

        assert StandardNameSource is not StandardNameSourceType

    def test_source_has_expected_fields(self):
        from imas_codex.graph.models import StandardNameSource

        fields = set(StandardNameSource.model_fields.keys())
        expected = {
            "id",
            "source_type",
            "source_id",
            "batch_key",
            "status",
            "claimed_at",
            "claim_token",
            "attempt_count",
            "last_error",
            "failed_at",
            "composed_at",
            "dd_path",
            "signal",
            "standard_name",
            "embedding",
            "embedded_at",
            "description",
        }
        assert expected.issubset(fields), f"Missing fields: {expected - fields}"

    def test_status_enum_values(self):
        from imas_codex.graph.models import StandardNameSourceStatus

        assert StandardNameSourceStatus.extracted.value == "extracted"
        assert StandardNameSourceStatus.composed.value == "composed"
        assert StandardNameSourceStatus.attached.value == "attached"
        assert StandardNameSourceStatus.vocab_gap.value == "vocab_gap"
        assert StandardNameSourceStatus.failed.value == "failed"
        assert StandardNameSourceStatus.stale.value == "stale"

    def test_source_type_enum_has_dd_and_signals(self):
        from imas_codex.graph.models import StandardNameSourceType

        members = set(StandardNameSourceType.__members__.keys())
        assert "dd" in members
        assert "signals" in members


class TestMergeStandardNameSources:
    """Test merge_standard_name_sources validation logic."""

    def _patch_gc(self, return_value=None):
        """Return a (patch, mock_ctx) pair for GraphClient."""
        if return_value is None:
            return_value = [{"affected": 1}]
        mock_ctx = MagicMock()
        mock_ctx.query.return_value = return_value
        patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
        mock_gc_cls = patcher.start()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        return patcher, mock_ctx

    def test_rejects_invalid_source_type(self):
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        with pytest.raises(ValueError, match="Invalid source_type"):
            merge_standard_name_sources(
                [{"id": "manual:test", "source_type": "manual", "source_id": "test"}]
            )

    def test_rejects_reference_source_type(self):
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        with pytest.raises(ValueError, match="Invalid source_type"):
            merge_standard_name_sources(
                [
                    {
                        "id": "reference:test",
                        "source_type": "reference",
                        "source_id": "test",
                    }
                ]
            )

    def test_empty_sources_returns_zero(self):
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        assert merge_standard_name_sources([]) == 0

    def test_accepts_dd_source_type(self):
        """dd is a valid pipeline source type (doesn't raise)."""
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        patcher, mock_ctx = self._patch_gc()
        try:
            result = merge_standard_name_sources(
                [
                    {
                        "id": "dd:eq/ts/p1d/psi",
                        "source_type": "dd",
                        "source_id": "eq/ts/p1d/psi",
                        "batch_key": "test",
                        "status": "extracted",
                    }
                ]
            )
            assert result == 1
            mock_ctx.query.assert_called_once()
        finally:
            patcher.stop()

    def test_accepts_signals_source_type(self):
        """signals is a valid pipeline source type."""
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        patcher, mock_ctx = self._patch_gc()
        try:
            result = merge_standard_name_sources(
                [
                    {
                        "id": "signals:tcv:ip/measured",
                        "source_type": "signals",
                        "source_id": "tcv:ip/measured",
                        "batch_key": "test",
                        "status": "extracted",
                    }
                ]
            )
            assert result == 1
            mock_ctx.query.assert_called_once()
        finally:
            patcher.stop()

    def test_mixed_invalid_type_raises(self):
        """A batch with even one invalid source_type should raise."""
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        with pytest.raises(ValueError, match="Invalid source_type"):
            merge_standard_name_sources(
                [
                    {"id": "dd:a/b", "source_type": "dd", "source_id": "a/b"},
                    {"id": "bad:x", "source_type": "bad_type", "source_id": "x"},
                ]
            )

    def test_returns_affected_count(self):
        """Return value matches the 'affected' field from the Cypher result."""
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        patcher, _ = self._patch_gc(return_value=[{"affected": 7}])
        try:
            result = merge_standard_name_sources(
                [
                    {
                        "id": f"dd:path/{i}",
                        "source_type": "dd",
                        "source_id": f"path/{i}",
                        "batch_key": "test",
                        "status": "extracted",
                    }
                    for i in range(7)
                ]
            )
            assert result == 7
        finally:
            patcher.stop()


class TestMarkSourcesFailed:
    """Test durable retry logic."""

    def _patch_gc(self, return_value=None):
        if return_value is None:
            return_value = [{"affected": 1}]
        mock_ctx = MagicMock()
        mock_ctx.query.return_value = return_value
        patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
        mock_gc_cls = patcher.start()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        return patcher, mock_ctx

    def test_returns_affected_count(self):
        from imas_codex.standard_names.graph_ops import mark_sources_failed

        patcher, mock_ctx = self._patch_gc()
        try:
            result = mark_sources_failed(
                "token123",
                ["dd:test/path"],
                "test error",
                max_attempts=3,
            )
            assert result == 1
            mock_ctx.query.assert_called_once()
        finally:
            patcher.stop()

    def test_query_uses_token_and_error(self):
        """The Cypher query receives token and error as parameters."""
        from imas_codex.standard_names.graph_ops import mark_sources_failed

        patcher, mock_ctx = self._patch_gc()
        try:
            mark_sources_failed(
                "mytoken", ["dd:a/b"], "something broke", max_attempts=3
            )
            call_kwargs = mock_ctx.query.call_args
            assert call_kwargs is not None
            # token and error should be passed as kwargs
            assert call_kwargs.kwargs.get("token") == "mytoken"
            assert call_kwargs.kwargs.get("error") == "something broke"
        finally:
            patcher.stop()

    def test_max_attempts_forwarded(self):
        """max_attempts parameter is forwarded to the query."""
        from imas_codex.standard_names.graph_ops import mark_sources_failed

        patcher, mock_ctx = self._patch_gc()
        try:
            mark_sources_failed("tok", ["dd:x/y"], "err", max_attempts=5)
            call_kwargs = mock_ctx.query.call_args.kwargs
            assert call_kwargs.get("max_attempts") == 5
        finally:
            patcher.stop()

    def test_empty_result_returns_zero(self):
        from imas_codex.standard_names.graph_ops import mark_sources_failed

        patcher, _ = self._patch_gc(return_value=[])
        try:
            result = mark_sources_failed("tok", ["dd:x/y"], "err", max_attempts=3)
            assert result == 0
        finally:
            patcher.stop()


class TestMarkSourcesStale:
    """Test stale marking."""

    def test_empty_list_returns_zero(self):
        from imas_codex.standard_names.graph_ops import mark_sources_stale

        assert mark_sources_stale([]) == 0

    def test_calls_graph_for_non_empty(self):
        from imas_codex.standard_names.graph_ops import mark_sources_stale

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = [{"affected": 2}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = mark_sources_stale(["dd:a/b", "dd:c/d"])
            assert result == 2
            mock_ctx.query.assert_called_once()

    def test_returns_affected_count(self):
        from imas_codex.standard_names.graph_ops import mark_sources_stale

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = [{"affected": 5}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = mark_sources_stale(["dd:x/y"] * 5)
            assert result == 5


class TestGetStandardNameSourceStats:
    """Test stats function."""

    def test_returns_dict(self):
        from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = [
            {"status": "extracted", "count": 100},
            {"status": "composed", "count": 50},
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = get_standard_name_source_stats()
            assert result == {"extracted": 100, "composed": 50}

    def test_with_source_type_filter(self):
        from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = [{"status": "extracted", "count": 30}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = get_standard_name_source_stats(source_type="dd")
            assert result == {"extracted": 30}

    def test_empty_graph_returns_empty_dict(self):
        from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = []
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = get_standard_name_source_stats()
            assert result == {}

    def test_source_type_filter_forwarded(self):
        """source_type kwarg should be forwarded to the query."""
        from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

        mock_ctx = MagicMock()
        mock_ctx.query.return_value = []
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            get_standard_name_source_stats(source_type="signals")
            call_kwargs = mock_ctx.query.call_args.kwargs
            assert call_kwargs.get("source_type") == "signals"
