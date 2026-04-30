"""Plan 40 Phase 2 — query tokenisation + tier policy tests."""

from __future__ import annotations

from imas_codex.standard_names.grammar_query import (
    ALL_TIER_SEGMENTS,
    STOPWORDS,
    TIER1_SEGMENTS,
    TIER2_SEGMENTS,
    TIER3_SEGMENTS,
    TIER_WEIGHT,
    filter_by_tier_policy,
    tier_of,
    tokenise_query,
)


def test_tokenise_query_snake_case() -> None:
    """T1 — snake-cased queries split on underscores; stopwords dropped."""
    assert tokenise_query("x_component_of_magnetic_field") == [
        "x",
        "component",
        "magnetic",
        "field",
    ]


def test_tokenise_query_drops_stopwords() -> None:
    """T1 — ISN connector words ('of', 'at', 'from', …) are dropped."""
    out = tokenise_query("electron_temperature_at_outboard_midplane")
    assert "at" not in out
    assert out == ["electron", "temperature", "outboard", "midplane"]


def test_tokenise_query_empty() -> None:
    assert tokenise_query("") == []
    assert tokenise_query("   ") == []


def test_tokenise_query_lowercases() -> None:
    assert tokenise_query("Electron_Temperature") == ["electron", "temperature"]


def test_stopwords_set_is_frozen() -> None:
    """Defensive — STOPWORDS is a frozenset so it cannot be mutated at runtime."""
    assert isinstance(STOPWORDS, frozenset)
    assert "of" in STOPWORDS and "at" in STOPWORDS


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def test_tier_of_classifies_segments() -> None:
    assert tier_of("physical_base") == 1
    assert tier_of("subject") == 1
    assert tier_of("geometric_base") == 1
    assert tier_of("transformation") == 2
    assert tier_of("component") == 2
    assert tier_of("position") == 2
    assert tier_of("process") == 2
    assert tier_of("coordinate") == 3
    assert tier_of("region") == 3
    assert tier_of("device") == 3


def test_tier_of_unknown_segment_zero() -> None:
    assert tier_of("nonexistent") == 0


def test_tier_partitions_disjoint() -> None:
    """T2 — tier sets do not overlap and cover all 12 segments."""
    assert TIER1_SEGMENTS.isdisjoint(TIER2_SEGMENTS)
    assert TIER1_SEGMENTS.isdisjoint(TIER3_SEGMENTS)
    assert TIER2_SEGMENTS.isdisjoint(TIER3_SEGMENTS)
    assert ALL_TIER_SEGMENTS == TIER1_SEGMENTS | TIER2_SEGMENTS | TIER3_SEGMENTS


def test_tier_weights_strictly_decreasing() -> None:
    """T2 — weights enforce tier hierarchy (1 > 2 > 3)."""
    assert TIER_WEIGHT[1] > TIER_WEIGHT[2] > TIER_WEIGHT[3] > 0


# ---------------------------------------------------------------------------
# AND-gate (plan §5.4 v3.2 strict)
# ---------------------------------------------------------------------------


def _hits(segment: str, rank: int = 0) -> tuple[str, int, float]:
    return (segment, rank, TIER_WEIGHT[tier_of(segment)])


def test_t1_alone_without_vk_is_dropped() -> None:
    """T2 — Tier-1 hit without vector/keyword corroboration is NOT admitted.

    This is the strict v3.2 AND-gate (§9.4 wins over §5.4 sketch).
    """
    by_id = {"sn_a": {1: [_hits("physical_base")]}}
    out = filter_by_tier_policy(by_id, vector_hits=set(), keyword_hits=set())
    assert out == set()


def test_t1_with_vk_is_admitted() -> None:
    """T2 — Tier-1 hit + vector co-occurrence → admitted."""
    by_id = {"sn_a": {1: [_hits("physical_base")]}}
    out = filter_by_tier_policy(by_id, vector_hits={"sn_a"}, keyword_hits=set())
    assert out == {"sn_a"}


def test_t2_without_t1_anchor_is_dropped() -> None:
    """T2 — Tier-2-only candidates (no anchor) are dropped regardless of v/k."""
    by_id = {"sn_a": {2: [_hits("transformation")]}}
    out = filter_by_tier_policy(by_id, vector_hits={"sn_a"}, keyword_hits={"sn_a"})
    assert out == set()


def test_t2_requires_t1_anchor_and_vk_cooccurrence() -> None:
    """T2 / §9.4 — Tier-2 needs BOTH Tier-1 anchor AND vector/keyword hit."""
    base = {"sn_anchored_no_vk": {1: [_hits("subject")], 2: [_hits("component")]}}
    # No vk → dropped (strict AND-gate; this is the §9.4 assertion)
    assert filter_by_tier_policy(base, set(), set()) == set()

    # T1 anchor + vector hit → admitted
    assert filter_by_tier_policy(
        base, vector_hits={"sn_anchored_no_vk"}, keyword_hits=set()
    ) == {"sn_anchored_no_vk"}


def test_t3_requires_t1_anchor_only() -> None:
    """T2 — Tier-3 needs only a Tier-1 anchor; v/k irrelevant."""
    by_id = {"sn_geo": {1: [_hits("physical_base")], 3: [_hits("device")]}}
    # Even without vk, Tier-3 + Tier-1 admits — but the §9.4 strict gate
    # does not apply to Tier-3 paths because Tier-1 anchor is sufficient
    # corroboration. Implementation choice.
    out = filter_by_tier_policy(by_id, set(), set())
    assert out == {"sn_geo"}


def test_t3_alone_dropped() -> None:
    by_id = {"sn_geo_only": {3: [_hits("device")]}}
    assert (
        filter_by_tier_policy(by_id, vector_hits={"sn_geo_only"}, keyword_hits=set())
        == set()
    )


def test_x_component_query_does_not_flood() -> None:
    """§9.4 worked example — ``x_component`` query with 50 decoy SNs.

    50 SNs match only on ``component=x`` (Tier-2). One SN matches on
    ``physical_base=temperature`` (Tier-1) AND has vector co-occurrence.
    Strict AND-gate must surface ONLY the anchored candidate.
    """
    by_id: dict[str, dict[int, list[tuple[str, int, float]]]] = {
        f"decoy_{i}": {2: [_hits("component", rank=i)]} for i in range(50)
    }
    by_id["true_match"] = {1: [_hits("physical_base", rank=0)]}

    admitted = filter_by_tier_policy(
        by_id, vector_hits={"true_match"}, keyword_hits=set()
    )
    assert admitted == {"true_match"}, f"Tier-2 flood must be filtered: got {admitted}"
