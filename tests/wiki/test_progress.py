"""Tests for wiki progress monitor."""

import pytest

from imas_codex.wiki.progress import WikiIngestionStats, WikiProgressMonitor


class TestWikiIngestionStats:
    """Tests for WikiIngestionStats dataclass."""

    def test_default_values(self):
        """Stats should have sensible defaults."""
        stats = WikiIngestionStats()
        assert stats.pages_total == 0
        assert stats.pages_scraped == 0
        assert stats.pages_failed == 0
        assert stats.chunks_created == 0
        assert stats.tree_nodes_linked == 0
        assert stats.imas_paths_linked == 0
        assert stats.conventions_found == 0
        assert stats.units_found == 0

    def test_to_dict(self):
        """Stats should convert to dictionary."""
        stats = WikiIngestionStats(
            pages_total=10,
            pages_scraped=8,
            chunks_created=50,
            pages_failed=2,
        )
        d = stats.to_dict()
        assert d["pages_total"] == 10
        assert d["pages_scraped"] == 8
        assert d["chunks_created"] == 50
        assert d["pages_failed"] == 2
        assert isinstance(d, dict)

    def test_elapsed_seconds(self):
        """Elapsed time should be tracked."""
        import time

        stats = WikiIngestionStats()
        time.sleep(0.1)
        assert stats.elapsed_seconds >= 0.1

    def test_pages_per_second(self):
        """Pages per second rate should be calculated."""
        stats = WikiIngestionStats(pages_scraped=10)
        # Will be very high because started_at is recent
        assert stats.pages_per_second >= 0

    def test_eta_seconds(self):
        """ETA should be calculated based on rate."""
        stats = WikiIngestionStats(pages_total=100, pages_scraped=50)
        # Should have some ETA estimate
        assert stats.eta_seconds >= 0

    def test_completion_pct(self):
        """Completion percentage should be in to_dict."""
        stats = WikiIngestionStats(pages_total=10, pages_scraped=5)
        d = stats.to_dict()
        assert d["completion_pct"] == 50.0

    def test_completion_pct_zero_total(self):
        """Completion percentage should handle zero total."""
        stats = WikiIngestionStats(pages_total=0)
        d = stats.to_dict()
        assert d["completion_pct"] == 0.0


class TestWikiProgressMonitor:
    """Tests for WikiProgressMonitor."""

    def test_init_console(self):
        """Monitor should initialize with console output."""
        monitor = WikiProgressMonitor(use_rich=True)
        assert monitor is not None
        assert monitor.stats is not None

    def test_init_no_console(self):
        """Monitor should work without Rich console."""
        monitor = WikiProgressMonitor(use_rich=False)
        assert monitor is not None
        assert monitor.stats is not None

    def test_update_stats(self):
        """Monitor should update stats correctly."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.stats.pages_total = 5
        monitor.stats.pages_scraped = 3
        assert monitor.stats.pages_total == 5
        assert monitor.stats.pages_scraped == 3

    def test_start(self):
        """Monitor should set total pages on start."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=10)
        assert monitor.stats.pages_total == 10

    def test_update_scrape(self):
        """Monitor should track page scrape events."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=5)
        monitor.update_scrape("Test Page", chunks=5, tree_nodes=3, imas_paths=2)
        assert monitor.stats.pages_scraped == 1
        assert monitor.stats.chunks_created == 5
        assert monitor.stats.tree_nodes_linked == 3
        assert monitor.stats.imas_paths_linked == 2

    def test_update_scrape_failed(self):
        """Monitor should track failed pages."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=5)
        monitor.update_scrape("Test Page", failed=True)
        assert monitor.stats.pages_failed == 1
        assert monitor.stats.pages_scraped == 0

    def test_update_scrape_conventions_units(self):
        """Monitor should track conventions and units."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=5)
        monitor.update_scrape("Test Page", conventions=2, units=5)
        assert monitor.stats.conventions_found == 2
        assert monitor.stats.units_found == 5

    def test_finish_returns_stats(self):
        """Monitor finish should return stats."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=5)
        monitor.update_scrape("Page 1", chunks=3)
        result = monitor.finish()
        assert isinstance(result, WikiIngestionStats)
        assert result.pages_scraped == 1
        assert result.chunks_created == 3

    def test_get_status(self):
        """Monitor should provide status dict."""
        monitor = WikiProgressMonitor(use_rich=False)
        monitor.start(total_pages=10)
        monitor.update_scrape("Page 1", chunks=5)

        status = monitor.get_status()
        assert isinstance(status, dict)
        assert status["pages_total"] == 10
        assert status["pages_scraped"] == 1
        assert status["chunks_created"] == 5
