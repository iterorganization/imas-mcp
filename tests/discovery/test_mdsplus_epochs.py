"""Tests for epoch detection integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.mdsplus.epochs import (
    detect_epochs_for_subtrees,
    detect_epochs_for_tree,
    epochs_to_versions,
)


class TestDetectEpochsForTree:
    """Tests for detect_epochs_for_tree wrapper."""

    @patch("imas_codex.discovery.mdsplus.epochs.discover_epochs_optimized")
    def test_returns_epochs(self, mock_discover):
        mock_discover.return_value = (
            [
                {
                    "id": "tcv:results:v1",
                    "data_source_name": "results",
                    "facility_id": "tcv",
                    "version": 1,
                    "first_shot": 3000,
                    "last_shot": 42000,
                    "representative_shot": 3000,
                    "node_count": 1523,
                    "is_new": True,
                },
                {
                    "id": "tcv:results:v2",
                    "data_source_name": "results",
                    "facility_id": "tcv",
                    "version": 2,
                    "first_shot": 42001,
                    "last_shot": 65000,
                    "representative_shot": 42001,
                    "node_count": 1481,
                    "is_new": True,
                },
            ],
            {},
        )

        config = {"source_name": "results", "detect_epochs": True}
        result = detect_epochs_for_tree("tcv", "results", config)

        assert len(result) == 2
        assert result[0]["version"] == 1
        assert result[1]["version"] == 2
        mock_discover.assert_called_once()

    @patch("imas_codex.discovery.mdsplus.epochs.discover_epochs_optimized")
    def test_empty_when_no_epochs(self, mock_discover):
        mock_discover.return_value = ([], {})

        config = {"source_name": "magnetics", "detect_epochs": True}
        result = detect_epochs_for_tree("tcv", "magnetics", config)

        assert result == []

    @patch("imas_codex.discovery.mdsplus.epochs.discover_epochs_optimized")
    def test_uses_epoch_config_params(self, mock_discover):
        mock_discover.return_value = ([], {})

        config = {
            "data_source_name": "results",
            "detect_epochs": True,
            "epoch_config": {
                "start_shot": 5000,
                "step_size": 500,
            },
        }
        detect_epochs_for_tree("tcv", "results", config)

        call_kwargs = mock_discover.call_args
        assert call_kwargs.kwargs["start_shot"] == 5000
        assert call_kwargs.kwargs["coarse_step"] == 500

    @patch("imas_codex.discovery.mdsplus.epochs.discover_epochs_optimized")
    def test_passes_client_for_incremental(self, mock_discover):
        mock_discover.return_value = ([], {})
        mock_client = MagicMock()

        config = {"source_name": "results", "detect_epochs": True}
        detect_epochs_for_tree("tcv", "results", config, client=mock_client)

        call_kwargs = mock_discover.call_args
        assert call_kwargs.kwargs["client"] is mock_client

    @patch("imas_codex.discovery.mdsplus.epochs.discover_epochs_optimized")
    def test_default_step_size(self, mock_discover):
        mock_discover.return_value = ([], {})

        config = {"source_name": "results", "detect_epochs": True}
        detect_epochs_for_tree("tcv", "results", config)

        call_kwargs = mock_discover.call_args
        assert call_kwargs.kwargs["coarse_step"] == 1000


class TestDetectEpochsForSubtrees:
    """Tests for detect_epochs_for_subtrees."""

    @patch("imas_codex.discovery.mdsplus.epochs.detect_epochs_for_tree")
    def test_processes_only_detect_epochs_trees(self, mock_detect):
        mock_detect.return_value = [{"version": 1, "first_shot": 3000}]

        subtrees = [
            {"source_name": "results", "detect_epochs": True},
            {"source_name": "magnetics"},  # No detect_epochs
            {"source_name": "diagz", "detect_epochs": False},
        ]

        result = detect_epochs_for_subtrees("tcv", "tcv_shot", subtrees)

        assert "results" in result
        assert "magnetics" not in result
        assert "diagz" not in result
        mock_detect.assert_called_once()

    @patch("imas_codex.discovery.mdsplus.epochs.detect_epochs_for_tree")
    def test_returns_empty_for_no_epoched_subtrees(self, mock_detect):
        subtrees = [
            {"source_name": "results"},
            {"source_name": "magnetics"},
        ]

        result = detect_epochs_for_subtrees("tcv", "tcv_shot", subtrees)
        assert result == {}
        mock_detect.assert_not_called()


class TestEpochsToVersions:
    """Tests for epochs_to_versions helper."""

    def test_extracts_sorted_versions(self):
        epochs = [
            {"version": 3, "first_shot": 65001},
            {"version": 1, "first_shot": 3000},
            {"version": 2, "first_shot": 42001},
        ]
        assert epochs_to_versions(epochs) == [1, 2, 3]

    def test_empty_epochs(self):
        assert epochs_to_versions([]) == []

    def test_skips_entries_without_version(self):
        epochs = [
            {"version": 1, "first_shot": 3000},
            {"first_shot": 42001},  # No version key
        ]
        assert epochs_to_versions(epochs) == [1]
