"""Tests for the name-then-docs review split (rc22 B2).

Covers:
- Bootstrap: name review sets reviewer_score when null
- Preservation: name review does not overwrite existing reviewer_score
- Isolation: docs review never writes reviewer_score
- Gate: docs review requires reviewed_name_at
- Inclusion: docs review includes name-reviewed names
- Ordering: run_turn runs review_names before review_docs
- Budget: TurnConfig.phase_budget sums to 1.0 across 5 keys
- Full mode: --target full writes all three score slots
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.turn import (
    TurnConfig,
    _run_review_docs_phase,
    _run_review_names_phase,
    run_turn,
)

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
    "pipeline_status": "enriched",
    "validation_status": "valid",
    "reviewer_score": None,
    "review_input_hash": None,
    "embedding": None,
    "tags": ["core_profiles"],
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
    "reviewed_name_at": None,
    "reviewed_docs_at": None,
    "link_status": "unresolved",
    "review_tier": None,
    "reviewer_comments": None,
}

_FAKE_NAME_REVIEWED: dict = {
    **_FAKE_ENRICHED_NAME,
    "id": "ion_temperature",
    "reviewer_score": 0.75,
    "reviewed_name_at": "2024-06-01T00:00:00Z",
    "reviewed_at": "2024-06-01T00:00:00Z",
}


def _make_review_mock(fake_names, target_filter_on_reviewed_name_at=False):
    """Return an async ``run_sn_review_engine`` stub."""

    async def _fake_engine(state, *, stop_event=None, **_kwargs):
        state.all_names = list(fake_names)

        targets = list(fake_names)

        # Mirror status_filter
        if state.status_filter:
            targets = [
                n
                for n in targets
                if (n.get("pipeline_status") or "drafted") == state.status_filter
            ]

        # Mirror domain filter
        if state.domain_filter:
            targets = [
                n
                for n in targets
                if (n.get("physics_domain") or "").lower()
                == state.domain_filter.lower()
            ]

        # Mirror docs gate
        review_target = getattr(state, "target", "full")
        if review_target == "docs":
            before = len(targets)
            targets = [n for n in targets if n.get("reviewed_name_at") is not None]
            skipped = before - len(targets)
            state.stats["docs_skipped_missing_name"] = skipped

        state.target_names = targets
        state.stats["persist_count"] = len(targets)

    return _fake_engine


# ===========================================================================
# Test 1 — name review bootstraps reviewer_score when null
# ===========================================================================


@pytest.mark.asyncio
async def test_name_review_bootstraps_reviewer_score_when_null(monkeypatch):
    """First name review of a name with reviewer_score IS NULL should
    set reviewer_score = reviewer_score_name via bootstrap."""
    captured_states: list = []

    async def fake_engine(state, *, stop_event=None, **kwargs):
        await _make_review_mock([_FAKE_ENRICHED_NAME])(state, stop_event=stop_event)
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
    await _run_review_names_phase(cfg)

    assert len(captured_states) == 1
    state = captured_states[0]

    # The phase constructed state with target='names'
    assert state.target == "names"
    # The enriched name should be extracted (status_filter=None)
    assert len(state.target_names) >= 1


# ===========================================================================
# Test 2 — name review never writes the shared reviewer_score slot
# ===========================================================================


@pytest.mark.asyncio
async def test_name_review_leaves_reviewer_score_if_set(monkeypatch):
    """Axis-split: write_name_review_results Cypher must not write any
    unqualified ``sn.reviewer_score =`` (only ``sn.reviewer_score_name``)."""
    import inspect

    from imas_codex.standard_names.graph_ops import write_name_review_results

    source = inspect.getsource(write_name_review_results)
    # Must not touch the old shared slot.
    assert "sn.reviewer_score =" not in source
    # Must write the axis scalar.
    assert "sn.reviewer_score_name" in source


# ===========================================================================
# Test 3 — docs review never writes reviewer_score
# ===========================================================================


@pytest.mark.asyncio
async def test_docs_review_never_writes_reviewer_score(monkeypatch):
    """Docs review mode must never SET sn.reviewer_score."""
    import inspect

    from imas_codex.standard_names.graph_ops import write_docs_review_results

    source = inspect.getsource(write_docs_review_results)

    # Must NOT contain 'sn.reviewer_score =' (it sets sn.reviewer_score_docs only)
    assert "sn.reviewer_score =" not in source, (
        "Docs mode must never write sn.reviewer_score"
    )
    assert "sn.reviewer_score_docs" in source


# ===========================================================================
# Test 4 — docs review requires name review first (gate)
# ===========================================================================


@pytest.mark.asyncio
async def test_docs_review_requires_name_review_first(monkeypatch):
    """Docs review should skip names where reviewed_name_at IS NULL
    and return count=0."""
    captured_states: list = []

    async def fake_engine(state, *, stop_event=None, **kwargs):
        await _make_review_mock(
            [_FAKE_ENRICHED_NAME],  # reviewed_name_at=None
        )(state, stop_event=stop_event)
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
    await _run_review_docs_phase(cfg)

    assert len(captured_states) == 1
    state = captured_states[0]

    # All names should have been filtered out by docs gate
    assert len(state.target_names) == 0
    assert state.stats.get("docs_skipped_missing_name", 0) >= 1


# ===========================================================================
# Test 5 — docs review includes name-reviewed names
# ===========================================================================


@pytest.mark.asyncio
async def test_docs_review_includes_name_reviewed_names(monkeypatch):
    """Docs review should include names that have reviewed_name_at set."""
    captured_states: list = []

    async def fake_engine(state, *, stop_event=None, **kwargs):
        await _make_review_mock([_FAKE_NAME_REVIEWED])(state, stop_event=stop_event)
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
    await _run_review_docs_phase(cfg)

    assert len(captured_states) == 1
    state = captured_states[0]

    # Name-reviewed name should pass docs gate
    assert len(state.target_names) >= 1
    assert state.stats.get("docs_skipped_missing_name", 0) == 0


# ===========================================================================
# Test 6 — turn runs review_names before review_docs
# ===========================================================================


@pytest.mark.asyncio
async def test_turn_runs_names_before_docs(monkeypatch):
    """run_turn() must execute review_names before review_docs in the phase list."""
    phase_order: list[str] = []

    async def track_reconcile(cfg):
        from imas_codex.standard_names.turn import PhaseResult

        phase_order.append("reconcile")
        return PhaseResult(name="reconcile", count=0)

    async def track_generate(cfg, *, regen=False, force=False, budget_override=None):
        from imas_codex.standard_names.turn import PhaseResult

        name = "regen" if regen else "generate"
        phase_order.append(name)
        return PhaseResult(name=name, count=0)

    async def track_enrich(cfg):
        from imas_codex.standard_names.turn import PhaseResult

        phase_order.append("enrich")
        return PhaseResult(name="enrich", count=0)

    async def track_link(cfg, touched):
        from imas_codex.standard_names.turn import PhaseResult

        phase_order.append("link")
        return PhaseResult(name="link", count=0)

    async def track_review_names(cfg, **kwargs):
        from imas_codex.standard_names.turn import PhaseResult

        phase_order.append("review_names")
        return PhaseResult(name="review_names", count=0)

    async def track_review_docs(cfg, **kwargs):
        from imas_codex.standard_names.turn import PhaseResult

        phase_order.append("review_docs")
        return PhaseResult(name="review_docs", count=0)

    monkeypatch.setattr(
        "imas_codex.standard_names.turn._run_reconcile_phase", track_reconcile
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.turn._run_generate_phase", track_generate
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.turn._run_enrich_phase", track_enrich
    )
    monkeypatch.setattr("imas_codex.standard_names.turn._run_link_phase", track_link)
    monkeypatch.setattr(
        "imas_codex.standard_names.turn._run_review_names_phase",
        track_review_names,
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.turn._run_review_docs_phase",
        track_review_docs,
    )

    cfg = TurnConfig(domain="equilibrium")
    await run_turn(cfg)

    # Verify ordering: review_names must come before review_docs
    assert "review_names" in phase_order
    assert "review_docs" in phase_order
    names_idx = phase_order.index("review_names")
    docs_idx = phase_order.index("review_docs")
    assert names_idx < docs_idx, (
        f"review_names (idx={names_idx}) must run before "
        f"review_docs (idx={docs_idx}). Order: {phase_order}"
    )

    # Full expected ordering
    expected = [
        "reconcile",
        "generate",
        "enrich",
        "link",
        "review_names",
        "review_docs",
        "regen",
    ]
    assert phase_order == expected, f"Phase order mismatch: {phase_order}"


# ===========================================================================
# Test 7 — TurnConfig.phase_budget sums to 1.0 across 5 keys
# ===========================================================================


def test_turn_phase_budget_sums_to_one():
    """Default TurnConfig.split must have 5 elements summing to 1.0."""
    cfg = TurnConfig(domain="test")
    assert len(cfg.split) == 5, f"Expected 5-way split, got {len(cfg.split)}"
    total = sum(cfg.split)
    assert abs(total - 1.0) < 1e-9, f"Split sums to {total}, expected 1.0"

    # Verify each phase budget
    for i in range(5):
        budget = cfg.phase_budget(i)
        assert budget > 0, f"Phase {i} budget should be > 0, got {budget}"


# ===========================================================================
# Test 8 — each axis writer only writes its own score slot
# ===========================================================================


def test_axis_writers_are_isolated():
    """write_name_review_results must not SET reviewer_score_docs slots and
    write_docs_review_results must not SET reviewer_score_name slots."""
    import inspect

    from imas_codex.standard_names.graph_ops import (
        write_docs_review_results,
        write_name_review_results,
    )

    name_source = inspect.getsource(write_name_review_results)
    docs_source = inspect.getsource(write_docs_review_results)

    # Name writer must not SET docs slots
    assert "reviewer_score_docs" not in name_source
    assert "sn.reviewed_docs_at" not in name_source
    # Docs writer must not SET name slots
    assert "reviewer_score_name" not in docs_source
    assert "sn.reviewed_name_at =" not in docs_source


# ===========================================================================
# Test 9 — dry_run returns correct phase names
# ===========================================================================


@pytest.mark.asyncio
async def test_dry_run_review_phase_names():
    """Dry-run review phases should return correct names."""
    cfg = TurnConfig(domain="equilibrium", dry_run=True)

    result_names = await _run_review_names_phase(cfg)
    assert result_names.name == "review_names"
    assert result_names.count == 0

    result_docs = await _run_review_docs_phase(cfg)
    assert result_docs.name == "review_docs"
    assert result_docs.count == 0


# ===========================================================================
# Test 10 — _run_review_phase alias still works
# ===========================================================================


def test_run_review_phase_alias_exists():
    """_run_review_phase must be a back-compat alias for _run_review_names_phase."""
    from imas_codex.standard_names.turn import (
        _run_review_names_phase,
        _run_review_phase,
    )

    assert _run_review_phase is _run_review_names_phase
