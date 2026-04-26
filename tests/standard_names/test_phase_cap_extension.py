"""Regression tests for phase-cap enforcement on reservation extension.

W36 root cause: ``_extend_reservation`` previously bypassed phase caps,
allowing a compose batch to drain the global pool via in-flight overshoot
and starve downstream review/regen phases.  The fix enforces phase caps
on extensions identically to initial reservations.
"""

from __future__ import annotations

from imas_codex.standard_names.budget import BudgetManager


class TestExtensionRespectsPhaseCap:
    """``_extend_reservation`` must not exceed ``phase_caps[phase] × 1.5``."""

    def test_extension_blocked_when_phase_cap_exhausted(self):
        # $5 pool, compose capped at 30% = $1.50; hard cap = $2.25.
        mgr = BudgetManager(5.0, phase_caps={"compose": 1.50})
        lease = mgr.reserve(1.0, phase="compose")
        assert lease is not None

        # Extend up to the hard cap: another $1.25 should succeed
        # (committed = 1.0 → 2.25; cap*1.5 = 2.25).
        extended = mgr._extend_reservation(lease._lease_id, 1.25)
        assert extended == 1.25

        # A further extension must be refused — phase cap exhausted.
        extended_again = mgr._extend_reservation(lease._lease_id, 0.50)
        assert extended_again == 0.0

    def test_extension_clamped_to_remaining_phase_cap(self):
        mgr = BudgetManager(10.0, phase_caps={"compose": 2.0})
        lease = mgr.reserve(1.0, phase="compose")
        assert lease is not None

        # Hard cap = 3.0; committed = 1.0; room = 2.0.
        # Asking for $5 should be clamped to $2.
        extended = mgr._extend_reservation(lease._lease_id, 5.0)
        assert extended == 2.0

    def test_extension_unrestricted_when_no_phase_cap(self):
        # No phase_caps configured → extension still pool-bounded only.
        mgr = BudgetManager(5.0)
        lease = mgr.reserve(1.0, phase="compose")
        assert lease is not None
        extended = mgr._extend_reservation(lease._lease_id, 4.0)
        assert extended == 4.0

    def test_extension_preserves_phase_committed_invariant(self):
        # phase_committed must equal sum of original reserve + extensions.
        mgr = BudgetManager(10.0, phase_caps={"compose": 5.0})
        lease = mgr.reserve(1.0, phase="compose")
        assert lease is not None
        mgr._extend_reservation(lease._lease_id, 2.0)
        assert mgr._phase_committed["compose"] == 3.0

    def test_review_phase_isolated_from_compose_overshoot(self):
        """Critical: compose extensions must NOT consume review's allocation.

        Pre-fix behaviour: a single compose batch could drain the pool past
        compose's 30% cap, leaving nothing for review_names.  Post-fix:
        compose hits its hard cap and stops; review's reservation succeeds.
        """
        mgr = BudgetManager(
            5.0,
            phase_caps={"compose": 1.50, "review_names": 0.75},
        )
        # Compose maxes out at hard cap (2.25) by repeated extension.
        compose_lease = mgr.reserve(0.10, phase="compose")
        assert compose_lease is not None
        # Try to drain $5 — should be clamped at (2.25 - 0.10) = 2.15.
        extended = mgr._extend_reservation(compose_lease._lease_id, 5.0)
        assert extended == 2.15

        # Review must still have its $0.75 cap available.
        review_lease = mgr.reserve(0.50, phase="review_names")
        assert review_lease is not None, (
            "Review phase reservation failed — compose drained pool past cap"
        )
