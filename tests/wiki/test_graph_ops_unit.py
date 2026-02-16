"""Tests for wiki graph operations.

Covers retry_on_deadlock decorator, claim/mark functions, bulk create
functions, and type classification constants — all with mocked GraphClient.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import TransientError

from imas_codex.discovery.wiki.graph_ops import (
    CLAIM_TIMEOUT_SECONDS,
    IMAGE_ARTIFACT_TYPES,
    INGESTABLE_ARTIFACT_TYPES,
    SCORABLE_ARTIFACT_TYPES,
    _bulk_create_wiki_artifacts,
    _bulk_create_wiki_pages,
    retry_on_deadlock,
)


# =============================================================================
# Type classification constants
# =============================================================================


class TestTypeClassification:
    """Tests for artifact type classification constants."""

    def test_ingestable_types(self):
        """Ingestable types should include text-extractable formats."""
        expected = {"pdf", "document", "presentation", "spreadsheet", "notebook", "json"}
        assert INGESTABLE_ARTIFACT_TYPES == expected

    def test_image_types(self):
        """Image types should only include image."""
        assert IMAGE_ARTIFACT_TYPES == {"image"}

    def test_scorable_types(self):
        """Scorable types should be ingestable + metadata-only types."""
        assert INGESTABLE_ARTIFACT_TYPES.issubset(SCORABLE_ARTIFACT_TYPES)
        assert "data" in SCORABLE_ARTIFACT_TYPES
        assert "archive" in SCORABLE_ARTIFACT_TYPES
        assert "other" in SCORABLE_ARTIFACT_TYPES

    def test_images_not_scorable(self):
        """Image types should NOT be scorable (they use VLM pipeline)."""
        assert IMAGE_ARTIFACT_TYPES.isdisjoint(SCORABLE_ARTIFACT_TYPES)


# =============================================================================
# retry_on_deadlock decorator
# =============================================================================


class TestRetryOnDeadlock:
    """Tests for the retry_on_deadlock decorator."""

    def test_success_on_first_try(self):
        """Function succeeding on first call should work normally."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3)
        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Should retry on TransientError."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3, base_delay=0.001, max_delay=0.01)
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Deadlock detected")
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 3

    def test_exhaust_retries(self):
        """Should raise after exhausting all attempts."""

        @retry_on_deadlock(max_attempts=2, base_delay=0.001, max_delay=0.01)
        def my_func():
            raise TransientError("Persistent deadlock")

        with pytest.raises(TransientError, match="Persistent deadlock"):
            my_func()

    def test_non_transient_error_not_retried(self):
        """Non-TransientError should propagate immediately."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3)
        def my_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not transient")

        with pytest.raises(ValueError, match="Not transient"):
            my_func()
        assert call_count == 1

    def test_preserves_function_metadata(self):
        """Decorated function should preserve __name__."""

        @retry_on_deadlock()
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"


# =============================================================================
# _bulk_create_wiki_pages
# =============================================================================


class TestBulkCreateWikiPages:
    """Tests for _bulk_create_wiki_pages with mocked GraphClient."""

    def test_creates_pages(self):
        """Should create pages via graph query."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 3}]

        pages = [
            {"id": "tcv:Page1", "title": "Page 1", "url": "https://wiki/Page1"},
            {"id": "tcv:Page2", "title": "Page 2", "url": "https://wiki/Page2"},
            {"id": "tcv:Page3", "title": "Page 3", "url": "https://wiki/Page3"},
        ]

        result = _bulk_create_wiki_pages(gc, "tcv", pages)
        assert result == 3
        gc.query.assert_called_once()

    def test_batch_processing(self):
        """Should process in batches."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 2}]

        pages = [
            {"id": f"tcv:Page{i}", "title": f"Page {i}", "url": f"url{i}"}
            for i in range(5)
        ]

        result = _bulk_create_wiki_pages(gc, "tcv", pages, batch_size=2)
        # 5 pages / 2 per batch = 3 batches
        assert gc.query.call_count == 3
        assert result == 6  # 2 * 3 batches

    def test_empty_batch(self):
        """Empty batch should return 0."""
        gc = MagicMock()
        result = _bulk_create_wiki_pages(gc, "tcv", [])
        assert result == 0
        gc.query.assert_not_called()

    def test_progress_callback(self):
        """Should invoke progress callback."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        progress_calls = []

        def on_progress(msg, stats):
            progress_calls.append(msg)

        pages = [{"id": "tcv:P1", "title": "P1", "url": "u1"}]
        _bulk_create_wiki_pages(gc, "tcv", pages, on_progress=on_progress)
        assert len(progress_calls) > 0
        assert "creating pages" in progress_calls[0]


# =============================================================================
# _bulk_create_wiki_artifacts
# =============================================================================


class TestBulkCreateWikiArtifacts:
    """Tests for _bulk_create_wiki_artifacts with mocked GraphClient."""

    def test_creates_artifacts(self):
        """Should create artifact nodes."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 2}]

        artifacts = [
            {
                "id": "tcv:report.pdf",
                "filename": "report.pdf",
                "url": "https://wiki/report.pdf",
                "artifact_type": "pdf",
            },
            {
                "id": "tcv:img.png",
                "filename": "img.png",
                "url": "https://wiki/img.png",
                "artifact_type": "image",
            },
        ]

        result = _bulk_create_wiki_artifacts(gc, "tcv", artifacts)
        assert result == 2

    def test_score_exempt_flag_set(self):
        """Image artifacts should get score_exempt=True."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        artifacts = [
            {
                "id": "tcv:photo.png",
                "filename": "photo.png",
                "url": "https://wiki/photo.png",
                "artifact_type": "image",
            },
        ]

        _bulk_create_wiki_artifacts(gc, "tcv", artifacts)
        # Verify score_exempt was set to True for image artifact
        assert artifacts[0]["score_exempt"] is True

    def test_non_image_not_exempt(self):
        """Non-image artifacts should not be score_exempt."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        artifacts = [
            {
                "id": "tcv:report.pdf",
                "filename": "report.pdf",
                "url": "x",
                "artifact_type": "pdf",
            },
        ]

        _bulk_create_wiki_artifacts(gc, "tcv", artifacts)
        assert artifacts[0]["score_exempt"] is False

    def test_linked_pages_relationships(self):
        """Should create HAS_ARTIFACT relationships for linked_pages."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        artifacts = [
            {
                "id": "tcv:file.pdf",
                "filename": "file.pdf",
                "url": "x",
                "artifact_type": "pdf",
                "linked_pages": ["MainPage", "Reports"],
            },
        ]

        _bulk_create_wiki_artifacts(gc, "tcv", artifacts)
        # Should have 2 query calls: 1 for artifacts + 1 for page links
        assert gc.query.call_count == 2

    def test_no_linked_pages_skips_link_query(self):
        """No linked_pages should skip the relationship query."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        artifacts = [
            {
                "id": "tcv:file.pdf",
                "filename": "file.pdf",
                "url": "x",
                "artifact_type": "pdf",
            },
        ]

        _bulk_create_wiki_artifacts(gc, "tcv", artifacts)
        # Only 1 query (artifact creation, no page links)
        assert gc.query.call_count == 1


# =============================================================================
# Pending work checks (with mocked GraphClient)
# =============================================================================


class TestPendingWorkChecks:
    """Tests for has_pending_* functions with mocked GraphClient."""

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_work_true(self, mock_gc_class):
        """Should return True when pending pages exist."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 42}]

        result = has_pending_work("tcv")
        assert result is True

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_work_false(self, mock_gc_class):
        """Should return False when no pending pages."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 0}]

        result = has_pending_work("tcv")
        assert result is False

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_artifact_work(self, mock_gc_class):
        """Should check artifact pending state."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_artifact_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 5}]

        result = has_pending_artifact_work("tcv")
        assert result is True

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_scan_work(self, mock_gc_class):
        """Should check for scanned pages awaiting scoring."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_scan_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 0}]

        result = has_pending_scan_work("tcv")
        assert result is False


# =============================================================================
# Claim timeout constant
# =============================================================================


class TestClaimConfiguration:
    """Tests for claim configuration constants."""

    def test_timeout_value(self):
        """Claim timeout should be 5 minutes."""
        assert CLAIM_TIMEOUT_SECONDS == 300
