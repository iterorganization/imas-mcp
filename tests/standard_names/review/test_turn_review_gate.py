"""Failing tests for the status_filter review gate bug (rc22 B1).

Root cause: ``StandardNameReviewState.status_filter`` defaults to ``"drafted"``.
``extract_review_worker`` (review/pipeline.py) drops every name whose
``pipeline_status != status_filter``.  The generate pipeline writes
``pipeline_status='named'`` or ``'enriched'``, never ``'drafted'``.
``_run_review_phase`` (turn.py) constructs ``StandardNameReviewState`` without
passing ``status_filter``, so the default ``"drafted"`` wins → every review phase
extracts 0 names.

Commit 1 (this file): two failing tests documenting the bug.
Commit 2 (fix in turn.py): pass ``status_filter=None`` + add invariant.
"""

from __future__ import annotations

from imas_codex.standard_names.turn import TurnConfig, _run_review_phase

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_FAKE_ENRICHED_NAME: dict = {
    "id": "electron_temperature",
    "description": "Electron temperature profile along the minor radius",
    "documentation": "Te measured by Thomson scattering.",
    "unit": "eV",
    "kind": "scalar",
    "physics_domain": "equilibrium",
    "pipeline_status": "enriched",  # NOT 'drafted' — this is what the bug blocks
    "validation_status": "valid",
    "reviewer_score": None,
    "review_input_hash": None,
    "embedding": None,
    "links": [],
    "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "physical_base": "temperature",
    "subject": "electron",
    "component": None,
    "coordinate": None,
    "position": None,
    "process": None,
    "source_types": ["dd"],
    "source_id": "core_profiles/profiles_1d/electrons/temperature",
    "generated_at": "2024-01-01T00:00:00Z",
    "reviewed_at": None,
    "link_status": "unresolved",
}


def _make_extract_mock(fake_names: list[dict]):
    """Return an async ``run_sn_review_engine`` stub that mirrors the filter logic.

    The stub replicates *only* the ``status_filter`` branch of
    ``extract_review_worker`` so both the failing assertion (current code) and
    the passing assertion (fixed code) are driven by the same logic path.
    """

    async def _fake_engine(state, *, stop_event=None, **_kwargs):
        state.all_names = list(fake_names)

        # Mirror extract_review_worker filter (pipeline.py lines 235-239)
        if state.status_filter:
            targets = [
                n
                for n in state.all_names
                if (n.get("pipeline_status") or "drafted") == state.status_filter
            ]
        else:
            targets = list(state.all_names)

        # Apply domain filter (same as extract worker)
        if state.domain_filter:
            targets = [
                n
                for n in targets
                if (n.get("physics_domain") or "").lower()
                == state.domain_filter.lower()
            ]

        state.target_names = targets
        state.stats["persist_count"] = len(targets)

    return _fake_engine


# ===========================================================================
# Test 1 — enriched/valid names reach the reviewer
# ===========================================================================


async def test_run_review_phase_extracts_enriched_valid_names(monkeypatch):
    """``_run_review_phase`` must pass ``pipeline_status='enriched'`` names to the reviewer.

    FAILS with current code:
        ``StandardNameReviewState`` defaults to ``status_filter='drafted'``.
        The extract filter in ``extract_review_worker`` drops every name whose
        ``pipeline_status != 'drafted'``.  Because the generate pipeline writes
        ``pipeline_status='enriched'`` (never ``'drafted'``), ``state.target_names``
        ends up empty → 0 names reach the LLM reviewer.

    PASSES after fix:
        ``_run_review_phase`` passes ``status_filter=None`` to
        ``StandardNameReviewState``.  The filter then accepts any
        ``pipeline_status`` and our enriched/valid name passes through.
    """
    captured_states: list = []

    async def fake_engine(state, *, stop_event=None, **kwargs):
        await _make_extract_mock([_FAKE_ENRICHED_NAME])(state, stop_event=stop_event)
        captured_states.append(state)

    monkeypatch.setattr(
        "imas_codex.standard_names.review.pipeline.run_sn_review_engine",
        fake_engine,
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.review.consolidation.run_consolidation",
        lambda _state: None,
    )

    cfg = TurnConfig(domain="equilibrium", dry_run=False, cost_limit=1.0)
    await _run_review_phase(cfg)

    assert len(captured_states) == 1, "run_sn_review_engine was not called"
    state = captured_states[0]

    # KEY ASSERTION: fails with status_filter='drafted' (current bug), passes when None
    assert len(state.target_names) >= 1, (
        f"Expected ≥1 target names for enriched/valid name, "
        f"got {len(state.target_names)}. "
        f"state.status_filter={state.status_filter!r} likely blocked "
        "pipeline_status='enriched'. "
        "Fix: pass status_filter=None to StandardNameReviewState in _run_review_phase."
    )


# ===========================================================================
# Test 2 — invariant: eligible names but zero persisted must fail
# ===========================================================================


async def test_run_review_phase_invariant_eligible_without_progress_fails(monkeypatch):
    """``_run_review_phase`` must return ``exit_code != 0`` when names were eligible
    but persist_count==0 and cost is below the budget floor.

    FAILS with current code:
        No invariant exists.  The function returns ``PhaseResult(exit_code=0,
        count=0)`` silently even when ``state.target_names`` is non-empty and
        nothing was persisted — hiding LLM / pipeline failures.

    PASSES after fix:
        An invariant in ``_run_review_phase`` detects the mismatch and returns
        ``exit_code=1`` with an explanatory error message.
    """

    async def fake_engine_zero_persist(state, *, stop_event=None, **_kwargs):
        # Simulate: extraction found targets but review produced nothing
        # (e.g., budget exhausted before any LLM call, or all batches errored)
        state.target_names = [_FAKE_ENRICHED_NAME]
        state.stats["persist_count"] = 0  # zero persisted despite having targets

    monkeypatch.setattr(
        "imas_codex.standard_names.review.pipeline.run_sn_review_engine",
        fake_engine_zero_persist,
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.review.consolidation.run_consolidation",
        lambda _state: None,
    )

    cfg = TurnConfig(domain="equilibrium", dry_run=False, cost_limit=1.0)
    result = await _run_review_phase(cfg)

    # FAILS with current code (exit_code defaults to 0).
    # PASSES after fix (invariant fires → exit_code=1).
    assert result.exit_code != 0, (
        f"Expected exit_code != 0 (invariant: eligible names but zero persisted), "
        f"got exit_code={result.exit_code}. "
        "Fix: add invariant in _run_review_phase after run_sn_review_engine."
    )
    assert result.count == 0, (
        f"Expected count=0 for failed invariant, got {result.count}"
    )
    assert result.error is not None, "Expected error message when invariant fires"
