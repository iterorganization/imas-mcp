"""Regression test: scheduler must treat 'extracted' SNS as eligible.

Sibling of ``test_extracted_sns_not_blocked.py`` (which covers the
worker-side filter in ``extract_dd_candidates``). This test covers the
**loop scheduler** side — ``_count_eligible_domains`` in ``loop.py``.

Before fix: ``WHERE NOT (sns.status IN ['stale', 'failed'])``
After fix:  ``WHERE NOT (sns.status IN ['stale', 'failed', 'extracted'])``

Without ``'extracted'`` in the scheduler allow-list, domains whose SNS
nodes are all in ``extracted`` state appear ineligible to the scheduler —
the loop says "No eligible domains" even though the worker would happily
process them (W18A/W18B hit this).
"""

from __future__ import annotations

import inspect

import pytest


class TestSchedulerEligibilityExtracted:
    """Verify _count_eligible_domains allows 'extracted' status through."""

    def test_scheduler_filter_includes_extracted(self) -> None:
        """The scheduler eligibility Cypher must include 'extracted'
        alongside 'stale' and 'failed' in its NOT EXISTS filter."""
        from imas_codex.standard_names.loop import _count_eligible_domains

        src = inspect.getsource(_count_eligible_domains)
        assert "'extracted'" in src, (
            "_count_eligible_domains must include 'extracted' in the "
            "NOT EXISTS status allow-list so that domains with only "
            "extracted SNS nodes remain eligible for scheduling"
        )

    def test_scheduler_filter_mirrors_worker_filter(self) -> None:
        """Scheduler and worker eligibility filters must use the same
        allow-list to avoid scheduling mismatches."""
        from imas_codex.standard_names.loop import _count_eligible_domains
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        sched_src = inspect.getsource(_count_eligible_domains)
        worker_src = inspect.getsource(extract_dd_candidates)

        # Both must contain the same three-element allow-list
        allow_list = ["'stale'", "'failed'", "'extracted'"]
        for token in allow_list:
            assert token in sched_src, f"Scheduler filter missing {token}"
            assert token in worker_src, f"Worker filter missing {token}"

    def test_terminal_statuses_still_block(self) -> None:
        """Terminal statuses ('composed', 'validated') must block in scheduler."""
        from imas_codex.standard_names.loop import _count_eligible_domains

        src = inspect.getsource(_count_eligible_domains)
        # The filter is a NOT EXISTS with NOT IN — only the listed statuses
        # pass through. Composed/validated are NOT listed, so they block.
        assert "'composed'" not in src or "NOT" in src, (
            "Terminal status 'composed' must not appear in the allow-list"
        )
