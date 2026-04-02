"""Tests for MDSplus extract_worker error handling.

Validates the two-tier error handling pattern:
- Permanent errors (extraction-returned) → mark_version_failed()
- Transient errors (SSH/network) → release_version_claim() for retry
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

FACILITY = "test_facility"


def _make_state():
    """Create a minimal TreeDiscoveryState mock."""
    state = MagicMock()
    state.facility = FACILITY
    state.ssh_host = "testhost"
    state.data_source_name = "magnetics"
    state.facility_config = {
        "data_systems": {
            "mdsplus": {
                "trees": {
                    "magnetics": {
                        "reference_shot": 1000,
                    }
                }
            }
        }
    }
    state.extract_stats = MagicMock(processed=0, errors=0, skipped=0)
    state.extract_phase = MagicMock()
    state.extract_phase.is_done.return_value = False
    state.extract_phase.done = False

    call_count = 0

    def should_stop():
        nonlocal call_count
        call_count += 1
        return call_count > 2  # Stop after second iteration

    state.should_stop = should_stop
    return state


class TestExtractWorkerErrorHandling:
    """Verify permanent vs transient error routing in extract_worker."""

    @pytest.mark.anyio
    async def test_extraction_returned_error_calls_mark_failed(self):
        """When extraction returns an error in ver_data, mark_version_failed is called."""
        from imas_codex.discovery.mdsplus.workers import extract_worker

        state = _make_state()

        # Claim returns one version, then None (which marks phase done)
        claim_calls = 0

        def mock_claim(facility, data_source_name=None):
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls == 1:
                return {
                    "id": f"{facility}:magnetics:v1",
                    "version": 1,
                    "data_source_name": "magnetics",
                    "first_shot": 1000,
                }
            state.extract_phase.mark_done()
            return None

        # Extraction returns an error for the version
        async def mock_extract(
            *, facility, data_source_name, shot, timeout=600, node_usages=None
        ):
            return {
                "versions": {
                    str(shot): {
                        "error": "tree file not found on disk: /trees/magnetics_1000.tree"
                    }
                }
            }

        mock_mark_failed = MagicMock()
        mock_release = MagicMock()

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction",
                side_effect=mock_claim,
            ),
            patch(
                "imas_codex.mdsplus.extraction.async_extract_tree_version",
                side_effect=mock_extract,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_failed",
                mock_mark_failed,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.release_version_claim",
                mock_release,
            ),
        ):
            await extract_worker(state)

        # Permanent error: mark_version_failed MUST be called
        mock_mark_failed.assert_called_once_with(
            f"{FACILITY}:magnetics:v1",
            "tree file not found on disk: /trees/magnetics_1000.tree",
        )
        # release_version_claim must NOT be called for permanent errors
        mock_release.assert_not_called()
        assert state.extract_stats.errors == 1

    @pytest.mark.anyio
    async def test_ssh_exception_calls_release_claim(self):
        """When SSH extraction raises an exception, release_version_claim is called."""
        from imas_codex.discovery.mdsplus.workers import extract_worker

        state = _make_state()

        claim_calls = 0

        def mock_claim(facility, data_source_name=None):
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls <= 5:
                return {
                    "id": f"{facility}:magnetics:v1",
                    "version": 1,
                    "data_source_name": "magnetics",
                    "first_shot": 1000,
                }
            state.extract_phase.mark_done()
            return None

        # SSH extraction raises a transient error
        async def mock_extract(
            *, facility, data_source_name, shot, timeout=600, node_usages=None
        ):
            raise ConnectionError("SSH connection timed out")

        mock_mark_failed = MagicMock()
        mock_release = MagicMock()

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction",
                side_effect=mock_claim,
            ),
            patch(
                "imas_codex.mdsplus.extraction.async_extract_tree_version",
                side_effect=mock_extract,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.release_version_claim",
                mock_release,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_failed",
                mock_mark_failed,
            ),
        ):
            await extract_worker(state)

        # Transient error: release_version_claim MUST be called (for retry)
        assert mock_release.call_count >= 1
        # mark_version_failed must NOT be called for transient errors
        mock_mark_failed.assert_not_called()

    @pytest.mark.anyio
    async def test_successful_extraction_marks_extracted(self):
        """Successful extraction calls mark_version_extracted, not failed or release."""
        from imas_codex.discovery.mdsplus.workers import extract_worker

        state = _make_state()

        claim_calls = 0

        def mock_claim(facility, data_source_name=None):
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls == 1:
                return {
                    "id": f"{facility}:magnetics:v1",
                    "version": 1,
                    "data_source_name": "magnetics",
                    "first_shot": 1000,
                }
            state.extract_phase.mark_done()
            return None

        async def mock_extract(
            *, facility, data_source_name, shot, timeout=600, node_usages=None
        ):
            return {
                "versions": {
                    str(shot): {
                        "node_count": 42,
                        "tags": {"ip": {}, "bt": {}},
                    }
                }
            }

        mock_mark_failed = MagicMock()
        mock_release = MagicMock()
        mock_mark_extracted = MagicMock()

        with (
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.claim_version_for_extraction",
                side_effect=mock_claim,
            ),
            patch(
                "imas_codex.mdsplus.extraction.async_extract_tree_version",
                side_effect=mock_extract,
            ),
            patch(
                "imas_codex.mdsplus.extraction.merge_version_results",
                return_value={},
            ),
            patch(
                "imas_codex.mdsplus.extraction.ingest_static_tree",
                return_value=MagicMock(nodes_merged=42, relationships_created=10),
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_extracted",
                mock_mark_extracted,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.mark_version_failed",
                mock_mark_failed,
            ),
            patch(
                "imas_codex.discovery.mdsplus.graph_ops.release_version_claim",
                mock_release,
            ),
        ):
            await extract_worker(state)

        # Success: mark_version_extracted called
        mock_mark_extracted.assert_called_once()
        # Neither failed nor released
        mock_mark_failed.assert_not_called()
        mock_release.assert_not_called()
        assert state.extract_stats.processed == 1
