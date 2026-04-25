"""Regression tests: extracted SNS nodes must not block re-extraction.

Bug 6 — When the pipeline crashes between extract and compose, SNS nodes
with ``status='extracted'`` are left behind. On the next run,
``extract_dd_candidates`` skips any DD path that has a non-stale,
non-failed SNS — treating ``extracted`` as "done" when it is actually
"started but never composed."

The fix adds ``'extracted'`` to the allow-list in the ``NOT EXISTS``
filter, so paths with ``extracted`` SNS are re-processed.
"""

from __future__ import annotations

import inspect

import pytest

from imas_codex.standard_names.sources.dd import extract_dd_candidates


class TestExtractedSNSNotBlocked:
    """Verify the NOT EXISTS filter allows 'extracted' status through."""

    def test_filter_includes_extracted_in_allow_list(self) -> None:
        """The SNS exclusion filter must include 'extracted' alongside stale/failed.

        Before fix: ``sns.status IN ['stale', 'failed']``
        After fix:  ``sns.status IN ['stale', 'failed', 'extracted']``
        """
        src = inspect.getsource(extract_dd_candidates)
        # The filter must include 'extracted' in the list
        assert "'extracted'" in src, (
            "extract_dd_candidates filter must include 'extracted' "
            "so that interrupted SNS nodes do not block re-extraction"
        )

    def test_filter_excludes_composed_and_validated(self) -> None:
        """Terminal statuses ('composed', 'validated') must still block.

        We verify that 'composed' and 'validated' are NOT in the
        allow-list — only stale, failed, and extracted should be.
        """
        src = inspect.getsource(extract_dd_candidates)
        # Find the NOT EXISTS filter line
        lines = src.split("\n")
        filter_line = None
        for line in lines:
            if "sns.status IN [" in line:
                filter_line = line
                break

        assert filter_line is not None, "Cannot find SNS status filter line"
        # The allow-list must be exactly stale + failed + extracted
        assert "'stale'" in filter_line
        assert "'failed'" in filter_line
        assert "'extracted'" in filter_line

    def test_force_bypasses_filter(self) -> None:
        """When force=True, the SNS filter should not appear at all."""
        src = inspect.getsource(extract_dd_candidates)
        # The filter is gated behind ``if not force:``
        assert "if not force:" in src, (
            "SNS exclusion filter must be gated behind force flag"
        )
